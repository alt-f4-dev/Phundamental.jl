using Test
using LinearAlgebra
using Phundamental

function _generalized_spinwave_fixture(; periodic=false, exchange=-Matrix{Float64}(I, 3, 3), field=(0.0, 0.0, 1.0))
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    properties = SiteProperties(spin=0.5, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0]; properties)])
    cluster = Supercell(crystal, (2,); periodic)
    material = Material("generalized spin-wave fixture", crystal;
        species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    model = build_model(material, cluster, TensorSpinModel(exchange; field, shell=1))
    return model, material, cluster
end

@testset "Generalized noncollinear SU(2) spin waves" begin
    z = [0.0, 0.0, 1.0]
    ferromagnet, material, cluster = _generalized_spinwave_fixture()
    reference = MagneticReferenceState([z, z])
    solver = SpinWaveSolver(reference=reference)
    @test solver_name(solver) == :spin_wave
    @test solver_compatibility(solver, ferromagnet).compatible
    modes = solve(solver, ferromagnet)
    @test modes isa SpinWaveResult
    @test minimum(modes.frequencies) > 0
    @test maximum(modes.metadata[:stationarity_residuals]) < 1e-8
    @test isapprox(modes.frequencies, solve(LinearSpinWaveSolver(), build_model(material, cluster, HeisenbergModel(-1.0; field=(0, 0, 1), shell=1))).frequencies; atol=1e-9)

    # Gapped collinear antiferromagnet reduces exactly to the legacy paired BdG blocks.
    antiferromagnet = build_model(material, cluster, TensorSpinModel(Matrix{Float64}(I, 3, 3);
        field=[[0.0, 0.0, 0.6], [0.0, 0.0, -0.6]], shell=1))
    antiferro_reference = MagneticReferenceState([z, -z])
    antiferro_result = solve(SpinWaveSolver(reference=antiferro_reference), antiferromagnet)
    legacy_antiferro = solve(LinearSpinWaveSolver(reference_signs=[1, -1]),
        build_model(material, cluster, XXZModel(1.0, 1.0; field_z=[0.6, -0.6], shell=1)))
    @test isapprox(antiferro_result.frequencies, legacy_antiferro.frequencies; atol=1e-9)
    @test all(isapprox(ω, sqrt(0.96); atol=1e-8) for ω in antiferro_result.frequencies)

    # Two inequivalent reference directions held stationary by site-dependent fields.
    n1, n2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    J = Matrix{Float64}(I, 3, 3)
    J[1, 2] += 0.2
    J[2, 1] -= 0.2
    fields = [0.5 * J * n2 + 2n1, 0.5 * transpose(J) * n1 + 2n2]
    noncollinear = build_model(material, cluster, TensorSpinModel(J; field=fields, shell=1))
    tilted = MagneticReferenceState([n1, n2])
    solver2 = SpinWaveSolver(reference=tilted)
    @test solver_compatibility(solver2, noncollinear).compatible
    result = solve(solver2, noncollinear)
    @test all(>(0), result.frequencies)
    @test result.metadata[:interaction_scope] == :bilinear_distinct_spin_sites
    @test isapprox(result.A, result.A'; atol=1e-12)
    @test isapprox(result.B, transpose(result.B); atol=1e-12)

    # Changing local transverse axes is a gauge transformation, not a new magnet.
    alternate = MagneticReferenceState([n1, n2]; transverse_axes=[0.0 0.0; 0.0 0.0; 1.0 1.0])
    alternate_result = solve(SpinWaveSolver(reference=alternate), noncollinear)
    @test isapprox(result.frequencies, alternate_result.frequencies; atol=1e-8)

    # Typed high-level request retains full laboratory-frame polarization channels.
    q = [[0.0], [π / 2]]
    axis = collect(range(-4.0, 4.0; length=301))
    request = SpinTensorStructureFactorRequest(q, axis)
    @test solver_compatibility(solver2, noncollinear, request).compatible
    calculated = forward(ForwardProblem(noncollinear; solver=solver2, observable=request))
    @test calculated.solution isa SpinWaveResult
    @test calculated.observable_result isa Phundamental.Observables.TensorSpectrumResult
    @test size(calculated.observable_result.intensity) == (2, 301, 3, 3)
    tensor_alternate = forward(ForwardProblem(noncollinear;
        solver=SpinWaveSolver(reference=alternate), observable=request)).observable_result
    @test isapprox(calculated.observable_result.intensity, tensor_alternate.intensity; atol=1e-8)
    @test_throws ArgumentError spin_tensor_structure_factor(noncollinear, result,
        Phundamental.Observables._spin_tensor_request_field(request, noncollinear), q, axis; channels=(:two_magnon,))

    # Reference-state validation rejects torques rather than deleting linear HP terms.
    invalid = MagneticReferenceState([z, z])
    @test !solver_compatibility(SpinWaveSolver(reference=invalid), noncollinear).compatible
    @test_throws ArgumentError solve(SpinWaveSolver(reference=invalid), noncollinear)

    # Bloch dispersion uses bond displacements and a magnetic periodic cell.
    periodic_model, _, _ = _generalized_spinwave_fixture(periodic=true)
    spectrum = spinwave_dispersion(solver, periodic_model, [[0.0], [π / 3]])
    @test spectrum isa SpinWaveDispersionResult
    @test size(spectrum.frequencies) == (2, 2)
    @test minimum(spectrum.frequencies) > 0
end
