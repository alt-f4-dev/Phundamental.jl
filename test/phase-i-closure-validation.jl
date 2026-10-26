using Test
using LinearAlgebra
using Phundamental

const PIC = Phundamental

@testset "Phase I: generated fermion observable interoperability" begin
    lattice = PIC.BravaisLattice(Matrix(Diagonal([1.0, 7.0, 8.0])))
    crystal = PIC.LatticeCrystalStructure(lattice, [PIC.BasisSite(:X, :X, [0.0, 0.0, 0.0])])
    material = PIC.Material("two-site Hubbard operator validation", crystal; species=Dict(:X => PIC.AtomicSpecies(:X; mass=1.0)))
    reference = PIC.Supercell(crystal, (2, 1, 1); periodic=true)
    hamiltonian = PIC.build_model(material, reference, PIC.HubbardModel(0.7, 1.2; shell=1)).hamiltonian
    algebra = PIC.FermionAlgebra(4)
    operation = PIC.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    dataset = PIC.CrystallographicDataset(spacegroup_number=1, operations=[operation], crystallographic_orbits=[1, 2])
    generated = PIC.generate_hamiltonian_family(material, reference, algebra; dataset=dataset,
        symmetry_action=PIC.SpinfulFermionAction(), candidates=PIC.AbstractOperatorExpr[hamiltonian], basis_labels=[:hub],
        max_body_order=4)
    representation = PIC.Representation(:spinful, PIC.FermionFockSpace(4), algebra, PIC.FermionOccupationBasis(collect(1:4)))
    context = PIC.default_observable_context(generated, representation; metadata_parameters=Dict(:hbar => 1.7))
    reference_operators = PIC.build_model(material, reference, PIC.HubbardModel(0.7, 1.2; shell=1)).observables
    for site in 1:2
        for axis in (:x, :y, :z)
            alias = Symbol("spin_", axis, "_", site)
            original = Symbol("fermion_spin_", axis, "_", site)
            @test haskey(context.observables, alias)
            @test haskey(context.observables, original)
            @test PIC.matrix(context.observables[alias], representation; sparse=false) ≈ PIC.matrix(context.observables[original], representation; sparse=false)
            @test isapprox(PIC.matrix(context.observables[alias], representation; sparse=false),
                1.7 .* PIC.matrix(reference_operators[alias], representation; sparse=false); atol=1e-12)
        end
        @test haskey(context.observables, Symbol("density_", site))
        @test haskey(context.observables, Symbol("double_occupancy_", site))
    end
    generated_model = PIC.instantiate_model(generated, representation, [1.0])
    conventional_model = PIC.build_model(material, reference, PIC.HubbardModel(0.7, 1.2; shell=1))
    request = PIC.SpinTensorStructureFactorRequest([0.2, 0.0, 0.0], collect(range(-2.0, 2.0; length=31));
        broadening=PIC.LorentzianBroadening(0.1))
    generated_response = PIC.forward(PIC.ForwardProblem(generated_model; solver=PIC.ExactDiagonalization(), observable=request))
    conventional_response = PIC.forward(PIC.ForwardProblem(conventional_model; solver=PIC.ExactDiagonalization(), observable=request))
    @test generated_response.observable_result.intensity ≈ conventional_response.observable_result.intensity atol=1e-10
end

@testset "Phase I: typed thermal and magnetic NLCE" begin
    lattice = PIC.BravaisLattice(reshape([1.0], 1, 1))
    crystal = PIC.LatticeCrystalStructure(lattice, [PIC.BasisSite(:S, :S, [0.0])])
    material = PIC.Material("uncoupled spin chain", crystal; species=Dict(:S => PIC.AtomicSpecies(:S; mass=1.0)),
        basis_properties=[PIC.SiteProperties(spin=1//2)])
    reference = PIC.Supercell(crystal, (2,); periodic=false)
    model = PIC.build_model(material, reference, PIC.XXZModel(0.0, 1.0; neighbor_cutoff=1.01))
    thermal_request = PIC.ThermodynamicCurveRequest([0.4, 1.0]; kB=1.0)
    thermal_context = PIC.NLCEForwardContext(2; inner_solver=PIC.ThermodynamicForwardSolver(PIC.ExactGibbs()))
    @test PIC.solver_compatibility(PIC.NLCESolver(), model, thermal_request, thermal_context).compatible
    thermal = PIC.forward(PIC.ForwardProblem(model; solver=PIC.NLCESolver(), method_context=thermal_context, observable=thermal_request))
    expected = PIC.thermal_curve(model, PIC.ExactGibbs(), thermal_request.temperatures)
    @test thermal.observable_result[:, 1] ≈ expected.free_energy ./ 2 atol=1e-10
    @test thermal.observable_result[:, 2] ≈ expected.internal_energy ./ 2 atol=1e-10
    @test thermal.observable_result[:, 3] ≈ expected.entropy ./ 2 atol=1e-10
    @test thermal.observable_result[:, 4] ≈ expected.heat_capacity ./ 2 atol=1e-10
    @test thermal.solution.metadata[:observable_components] == (:free_energy, :internal_energy, :entropy, :heat_capacity)
    magnetic_request = PIC.MagneticSusceptibilityRequest(:total_spin_z, [0.4, 1.0]; direction=:z,
        field_unit="energy/moment", moment_unit="moment")
    magnetic_context = PIC.NLCEForwardContext(2; inner_solver=PIC.ExactDiagonalization())
    @test PIC.solver_compatibility(PIC.NLCESolver(), model, magnetic_request, magnetic_context).compatible
    magnetic = PIC.forward(PIC.ForwardProblem(model; solver=PIC.NLCESolver(), method_context=magnetic_context, observable=magnetic_request))
    direct = PIC.forward(PIC.ForwardProblem(model; solver=PIC.ExactDiagonalization(), observable=magnetic_request))
    @test magnetic.observable_result ≈ direct.observable_result.values ./ 2 atol=1e-10
    magnetization_request = PIC.MagnetizationCurveRequest(:total_spin_z, [-0.1, 0.0, 0.2], [0.5, 1.0];
        direction=:z, field_unit="energy/moment", moment_unit="moment")
    magnetization = PIC.forward(PIC.ForwardProblem(model; solver=PIC.NLCESolver(), method_context=magnetic_context,
        observable=magnetization_request))
    direct_magnetization = PIC.forward(PIC.ForwardProblem(model; solver=PIC.ExactDiagonalization(), observable=magnetization_request))
    @test magnetization.observable_result ≈ direct_magnetization.observable_result.values ./ 2 atol=1e-10
end

@testset "Phase I: multidimensional Q-E resolution and lazy covariance" begin
    coords = [Float64[i / 4, j / 4] for i in 0:3 for j in 0:3]
    q = [copy(v) for v in coords]
    positions = reduce(vcat, [permutedims(v) for v in coords])
    energy = [-0.5, 0.0, 0.5, 1.0]
    values = [0.2 + abs(sin(i + j)) for i in 1:length(q), j in eachindex(energy)]
    result = PIC.Observables.ScatteringResult(q, energy, values, Dict{Symbol,Any}())
    weights = fill(0.25, length(q))
    C = [0.07 0.008 0.002; 0.008 0.06 -0.003; 0.002 -0.003 0.1]
    tensor = repeat(reshape(C, 3, 3, 1, 1), 1, 1, length(q), length(energy))
    kwargs = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=4.0)
    dense = PIC.SpatialGaussianQEResolution(tensor, positions, energy, weights; kwargs...)
    compact = PIC.SpatialGaussianQEResolution(tensor, positions, energy, weights; kwargs..., storage=:lazy)
    callable = PIC.SpatialGaussianQEResolution((_, _) -> C, positions, energy, weights; kwargs...)
    @test isnothing(compact.inverse_covariance)
    @test isnothing(callable.covariance)
    @test isnothing(callable.inverse_covariance)
    @test PIC.convolve_resolution(dense, result).intensity ≈ PIC.convolve_resolution(compact, result).intensity atol=1e-12
    @test PIC.convolve_resolution(dense, result).intensity ≈ PIC.convolve_resolution(callable, result).intensity atol=1e-12
    brute = zeros(size(values))
    ew = PIC.Observables._energy_quadrature_weights(energy)
    for jq in eachindex(q), jx in eachindex(energy), iq in eachindex(q), ix in eachindex(energy)
        z = [q[iq] .- q[jq]; energy[ix] - energy[jx]]
        distance = dot(z, C \ z)
        if distance <= 16.0
            brute[iq, ix] += values[jq, jx] * weights[jq] * ew[jx] * exp(-0.5 * distance) / sqrt((2π)^3 * det(C))
        end
    end
    @test PIC.convolve_resolution(callable, result).intensity ≈ brute atol=1e-11 rtol=1e-11

    # The point index is cached on each resolution, not reconstructed for every convolution.
    @test !isnothing(callable.spatial_index)
    @test !isnothing(dense.spatial_index)
    serial = PIC.convolve_resolution(callable, result).intensity
    @test PIC.convolve_resolution(callable, result).intensity == serial
    @test PIC.convolve_resolution(callable, result; threaded=true).intensity ≈ serial atol=1e-11 rtol=1e-11
    @test PIC.convolve_resolution(dense, result; threaded=true).intensity ≈ serial atol=1e-11 rtol=1e-11

    stationary = PIC.GaussianQEResolution(C, positions, weights; kwargs...)
    @test !isnothing(stationary.spatial_index)
    @test PIC.convolve_resolution(stationary, result).intensity ≈ serial atol=1e-11 rtol=1e-11
    @test PIC.convolve_resolution(stationary, result; threaded=true).intensity ≈ serial atol=1e-11 rtol=1e-11

    # Correlation terms and nonuniform true-energy quadrature must survive the exact energy-root pruning.
    energies_irregular = [-0.5, -0.19, 0.17, 1.0]
    shifted = PIC.Observables.ScatteringResult(q, energies_irregular, values, Dict{Symbol,Any}())
    function varying_covariance(e)
        D = copy(C)
        D[1, 3] += 0.001 * e
        D[3, 1] += 0.001 * e
        return D
    end
    covariance_irregular = PIC.SpatialGaussianQEResolution((_, e) -> varying_covariance(e),
                                                           positions, energies_irregular, weights; kwargs...)
    direct_irregular = zeros(size(values))
    irregular_weights = PIC.Observables._energy_quadrature_weights(energies_irregular)
    for jq in eachindex(q), jx in eachindex(energies_irregular), iq in eachindex(q), ix in eachindex(energies_irregular)
        D = varying_covariance(energies_irregular[jx])
        z = [q[iq] .- q[jq]; energies_irregular[ix] - energies_irregular[jx]]
        distance = dot(z, D \ z)
        if distance <= 16.0
            direct_irregular[iq, ix] += values[jq, jx] * weights[jq] * irregular_weights[jx] *
                                        exp(-0.5 * distance) / sqrt((2π)^3 * det(D))
        end
    end
    @test PIC.convolve_resolution(covariance_irregular, shifted).intensity ≈ direct_irregular atol=1e-11 rtol=1e-11
    @test PIC.convolve_resolution(covariance_irregular, shifted; threaded=true).intensity ≈ direct_irregular atol=1e-11 rtol=1e-11

    # Dense source coverage detects corruption masked by sparse single-source diagnostics.
    # The parallel path must agree across repeated calls, not merely on its first execution.
    for _ in 1:20
        @test PIC.convolve_resolution(callable, result; threaded=true).intensity ≈ serial atol=1e-11 rtol=1e-11
        @test PIC.convolve_resolution(covariance_irregular, shifted; threaded=true).intensity ≈ direct_irregular atol=1e-11 rtol=1e-11
        @test PIC.convolve_resolution(compact, result; threaded=true).intensity ≈ serial atol=1e-11 rtol=1e-11
    end
end

@testset "Phase I: source-dependent 3D Q-E covariance factorization" begin
    dimension = 3
    qmatrix = reduce(vcat, ([0.05 + 0.14 * mod(i, 5)  0.10 + 0.12 * mod(i, 4)  0.07 + 0.13 * mod(i, 3)] for i in 1:12))
    q = [collect(view(qmatrix, i, :)) for i in axes(qmatrix, 1)]
    energy = collect(range(-5.0, 25.0; length=8))
    values = [0.1 + 0.05 * (i + j) for i in axes(qmatrix, 1), j in eachindex(energy)]
    spectrum = PIC.Observables.ScatteringResult(q, energy, values, Dict{Symbol,Any}())
    weights = fill(1.0 / length(q), length(q))
    correlation = [1.0 0.10 0.0 0.28; 0.10 1.0 0.0 -0.20; 0.0 0.0 1.0 0.12; 0.28 -0.20 0.12 1.0]
    @test isposdef(Symmetric(correlation))
    function source_covariance(source_q, source_energy)
        phase = 2π * source_q[1]
        widths = [0.019 + 0.009 * (0.5 + 0.5 * sin(phase + component / 2)) for component in 1:dimension]
        push!(widths, 0.48 + 0.20 * (0.5 + 0.5 * cos(0.3 * source_energy + phase)))
        return Diagonal(widths) * correlation * Diagonal(widths)
    end
    for i in eachindex(q), j in eachindex(energy)
        @test isposdef(Hermitian(source_covariance(q[i], energy[j])))
    end
    options = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=3.0)
    lazy = PIC.SpatialGaussianQEResolution(source_covariance, qmatrix, energy, weights; options..., storage=:lazy)
    dense = PIC.SpatialGaussianQEResolution(source_covariance, qmatrix, energy, weights; options..., storage=:dense)
    expected = PIC.convolve_resolution(dense, spectrum; threaded=false).intensity
    for _ in 1:3
        @test PIC.convolve_resolution(lazy, spectrum; threaded=false).intensity ≈ expected atol=1e-11 rtol=1e-11
        @test PIC.convolve_resolution(lazy, spectrum; threaded=true).intensity ≈ expected atol=1e-11 rtol=1e-11
    end
end

@testset "Phase I: task-owned threaded Q-E covariance workspaces" begin
    # Match the covariance-provider algebra used by the experimental resolution benchmark.
    nQ = max(64, 2 * Threads.nthreads())
    nE = 12
    qmatrix = reduce(vcat, ([mod(0.17 * i, 1.0) mod(0.31 * i, 1.0)] for i in 1:nQ))
    q = [collect(view(qmatrix, i, :)) for i in 1:nQ]
    energies = collect(range(-5.0, 25.0; length=nE))
    values = [0.1 + abs(sin(0.23 * i + 0.13 * j)) for i in 1:nQ, j in 1:nE]
    spectrum = PIC.Observables.ScatteringResult(q, energies, values, Dict{Symbol,Any}())
    weights = fill(1.0 / nQ, nQ)
    correlation = [1.0 0.10 0.28; 0.10 1.0 -0.20; 0.28 -0.20 1.0]
    @test isposdef(Symmetric(correlation))
    function varying_covariance(source_q, energy)
        qphase = 2π * source_q[1]
        widths = [0.019 + 0.009 * (0.5 + 0.5 * sin(qphase + component / 2)) for component in 1:2]
        push!(widths, 0.48 + 0.20 * (0.5 + 0.5 * cos(0.3 * energy + qphase)))
        covariance = Diagonal(widths) * correlation * Diagonal(widths)
        yield()  # Encourage task interleaving during threaded source evaluation.
        return covariance
    end
    options = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=3.0)
    resolution = PIC.SpatialGaussianQEResolution(varying_covariance, qmatrix, energies, weights; options..., storage=:lazy)
    dense = PIC.SpatialGaussianQEResolution(varying_covariance, qmatrix, energies, weights; options..., storage=:dense)
    expected = PIC.convolve_resolution(dense, spectrum; threaded=false).intensity
    @test PIC.convolve_resolution(resolution, spectrum; threaded=false).intensity ≈ expected atol=1e-11 rtol=1e-11
    for _ in 1:6
        @test PIC.convolve_resolution(resolution, spectrum; threaded=true).intensity ≈ expected atol=1e-10 rtol=1e-10
    end
end


@testset "Phase I: threaded callable covariance with reused provider buffer" begin
    nQ = 64
    nE = 12
    coordinates = reduce(vcat, ([mod(0.17 * i, 1.0) mod(0.31 * i, 1.0)] for i in 1:nQ))
    q = [collect(view(coordinates, i, :)) for i in 1:nQ]
    energy = collect(range(-5.0, 25.0; length=nE))
    values = [mod(i + 17 * j, 5) == 0 ? 0.5 + abs(sin(0.23 * i + 0.13 * j)) : 0.0 for i in 1:nQ, j in 1:nE]
    spectrum = PIC.Observables.ScatteringResult(q, energy, values, Dict{Symbol,Any}())
    weights = fill(1.0 / nQ, nQ)
    correlation = [1.0 0.10 0.28; 0.10 1.0 -0.20; 0.28 -0.20 1.0]
    options = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=3.0)

    function widths(source_q, source_energy)
        phase = 2π * source_q[1]
        w = [0.019 + 0.009 * (0.5 + 0.5 * sin(phase + component / 2)) for component in 1:2]
        push!(w, 0.48 + 0.20 * (0.5 + 0.5 * cos(0.3 * source_energy + phase)))
        return w
    end
    function independent_covariance(source_q, source_energy)
        w = widths(source_q, source_energy)
        return [w[i] * correlation[i, j] * w[j] for i in 1:3, j in 1:3]
    end

    shared_buffer = zeros(3, 3)
    function reused_covariance(source_q, source_energy)
        w = widths(source_q, source_energy)
        for j in 1:3, i in 1:3
            shared_buffer[i, j] = w[i] * correlation[i, j] * w[j]
            yield()
        end
        return shared_buffer
    end

    reference_resolution = PIC.SpatialGaussianQEResolution(independent_covariance, coordinates, energy, weights; options..., storage=:dense)
    reference = PIC.convolve_resolution(reference_resolution, spectrum; threaded=false).intensity
    reused_resolution = PIC.SpatialGaussianQEResolution(reused_covariance, coordinates, energy, weights; options..., storage=:lazy)
    for _ in 1:10
        serial = PIC.convolve_resolution(reused_resolution, spectrum; threaded=false).intensity
        threaded = PIC.convolve_resolution(reused_resolution, spectrum; threaded=true).intensity
        @test serial ≈ reference atol=1e-10 rtol=1e-10
        @test threaded ≈ reference atol=1e-10 rtol=1e-10
    end
end

@testset "Phase I: immutable threaded source batches and boundary crossing" begin
    nQ = 64
    nE = 258
    qmatrix = reduce(vcat, ([mod(0.17 * i, 1.0) mod(0.31 * i, 1.0)] for i in 1:nQ))
    q = [collect(view(qmatrix, i, :)) for i in 1:nQ]
    energy = collect(range(-5.0, 25.0; length=nE))
    values = [mod(i + 17 * j, 31) == 0 ? 0.3 + abs(sin(i + j)) : 0.0 for i in 1:nQ, j in 1:nE]
    spectrum = PIC.Observables.ScatteringResult(q, energy, values, Dict{Symbol,Any}())
    weights = fill(1.0 / nQ, nQ)
    C = [0.0009 0.00012 0.002; 0.00012 0.0008 -0.002; 0.002 -0.002 0.20]
    kwargs = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=3.0)
    lazy = PIC.SpatialGaussianQEResolution((_, _) -> C, qmatrix, energy, weights; kwargs...)
    serial = PIC.convolve_resolution(lazy, spectrum; threaded=false).intensity
    @test length(values) > PIC.Observables._QE_THREAD_SOURCE_BATCH
    for _ in 1:8
        threaded = PIC.convolve_resolution(lazy, spectrum; threaded=true).intensity
        @test threaded ≈ serial atol=1e-11 rtol=1e-11
        @test all(isfinite, threaded)
    end
end
