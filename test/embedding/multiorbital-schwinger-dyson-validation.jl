#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiOrbitalSchwingerDysonValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _fixture(; interacting=false)
    labels = interacting ? P.kanamori_interaction(1; U=1.2).labels : [:a, :b]
    U = interacting ? P.kanamori_interaction(1; U=1.2) : P.SpinOrbitalInteraction(zeros(ComplexF64, 2, 2, 2, 2); labels=labels)
    onsite = ComplexF64[0.1 0.03im; -0.03im -0.12]
    hopping = interacting ? zeros(ComplexF64, 2, 2) : ComplexF64[0.24 0; 0 -0.20]
    lattice = P.MatrixMomentumLattice(cat(onsite + hopping, onsite - hopping; dims=3);
        labels=labels, kpoints=[:k0, :kpi], qmaps=[[1, 2], [2, 1]], qpoints=[:q0, :qpi])
    axis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    baxis = P.BosonicMatsubaraAxis(4.0, [0, 1])
    problem = P.MultiOrbitalDMFTProblem(lattice, U; temperature=0.25, axis=axis)
    ed = P.MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized, max_full_dimension=256,
        max_sector_dimension=256, max_eigensystem_bytes=20_000_000)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=ed, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=5)
    dmft = P.solve(solver, problem)
    @test dmft.converged
    green = P.expanded_dmft_momentum_green(dmft, baxis)
    return (; U, lattice, axis, baxis, dmft, green, labels)
end

@testset "Orbital Schwinger-Dyson: antisymmetrized static Hartree-Fock" begin
    U = P.kanamori_interaction(1; U=3.0)
    density = ComplexF64[0.3 0; 0 0.7]
    hf = P.orbital_hartree_fock(U, density)
    @test hf ≈ ComplexF64[2.1 0; 0 0.9] atol=1e-13
    @test_throws DimensionMismatch P.orbital_hartree_fock(U, ones(3, 3))
    @test_throws ArgumentError P.orbital_hartree_fock(U, ComplexF64[0 1; 0 0])
    @test all(iszero, P.orbital_hartree_fock(P.SpinOrbitalInteraction(zeros(ComplexF64, 2, 2, 2, 2)), density))
end

@testset "Orbital Schwinger-Dyson: independently checked particle-hole full vertex" begin
    faxis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    baxis = P.BosonicMatsubaraAxis(4.0, [0])
    pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    values = zeros(ComplexF64, 4, 4, 2, 2, 1)
    for j in 1:4, i in 1:4, n in 1:2
        values[i, j, n, n, 1] = i == j ? -1.0 - 0.03i : 0.004im * (i-j)
    end
    bubble = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:a, :b], values)
    gamma = zeros(ComplexF64, size(values))
    for i in 1:4, n in 1:2
        gamma[i, i, n, n, 1] = 0.025
    end
    vertex = P.OrbitalPairIrreducibleVertex(faxis, baxis, pairs, [:a, :b], gamma, Dict{Symbol,Any}())
    response = P.bethe_salpeter(bubble, vertex)
    recovered = P.orbital_full_vertex(bubble, response)
    B = P.Embedding._orbital_bse_matrix!(zeros(ComplexF64, 8, 8), bubble.values, 1)
    X = P.Embedding._orbital_bse_matrix!(zeros(ComplexF64, 8, 8), response.values, 1)
    expected = B \ (X - B) / B
    actual = P.Embedding._orbital_bse_matrix!(zeros(ComplexF64, 8, 8), recovered.values, 1)
    @test actual ≈ expected atol=1e-11 rtol=1e-11
    @test recovered.metadata[:crossing_complete] === false
    @test_throws ArgumentError P.orbital_full_vertex(bubble, response; max_workspace_bytes=1)
end

@testset "Orbital Schwinger-Dyson: exact frequency guards, local cancellation and Dyson matrix" begin
    f = _fixture()
    @test f.green.metadata[:expanded_finite_bath_lehmann] === true
    @test length(f.green.axis) > length(f.axis)
    @test f.green.labels == f.labels
    zero_kernel = P.OrbitalSchwingerDysonKernel((destination, iq, iν, iΩ) -> nothing, f.labels;
        provenance=:screened_particle_hole)
    zero_sigma = P.orbital_schwinger_dyson(f.dmft, f.lattice, f.green, f.lattice.qmaps, f.baxis, zero_kernel)
    for ik in 1:2
        @test zero_sigma.values[:, :, ik, :] ≈ f.dmft.self_energy.values atol=1e-13
    end
    custom = P.OrbitalSchwingerDysonKernel((destination, iq, iν, iΩ) -> (destination[1, 1, 1, 1] = iq == 1 ? 0.2 : -0.13),
        f.labels; provenance=:screened_particle_hole)
    sigma = P.orbital_schwinger_dyson(f.dmft, f.lattice, f.green, f.lattice.qmaps, f.baxis, custom)
    report = P.validate_orbital_schwinger_dyson(f.dmft, sigma)
    @test report.valid
    @test report.local_residual <= 1e-12
    @test !sigma.metadata[:crossing_complete_certified_by_package]
    @test sigma.metadata[:local_double_counting_scheme] == :weighted_local_projection
    assert_nonlocal = maximum(abs, sigma.values[:, :, 1, :] - sigma.values[:, :, 2, :])
    @test assert_nonlocal > 1e-8
    # Independent direct contraction; no orbital-pair flattening or BLAS workspace.
    raw = zeros(ComplexF64, 2, 2, 2, length(f.axis))
    for iq in 1:2, (iΩ, m) in enumerate(f.baxis.indices), (iν, n) in enumerate(f.axis.indices), ik in 1:2
        shifted = findfirst(==(n + m), f.green.axis.indices)
        map = f.lattice.qmaps[iq][ik]
        weight = iq == 1 ? 0.2 : -0.13
        raw[1, 1, ik, iν] -= 0.5 / f.axis.beta * weight * f.green.values[1, 1, map, shifted]
    end
    for iν in 1:length(f.axis), ik in 1:2
        projected = sum(f.lattice.weights[k] * raw[1, 1, k, iν] for k in 1:2)
        @test sigma.values[1, 1, ik, iν] - f.dmft.self_energy.values[1, 1, iν] ≈
            raw[1, 1, ik, iν] - projected atol=1e-12
    end
    corrected = P.dyson_momentum_green(f.lattice, sigma)
    for ik in 1:2, iν in 1:length(f.axis)
        expected = inv(f.axis[iν] * Matrix{ComplexF64}(I, 2, 2) - f.lattice.hamiltonians[:, :, ik] - sigma.values[:, :, ik, iν])
        @test corrected.values[:, :, ik, iν] ≈ expected atol=1e-12
    end
    @test_throws ArgumentError P.OrbitalSchwingerDysonKernel((a, b, c, d) -> nothing, f.labels; provenance=:unknown)
    @test_throws ArgumentError P.orbital_schwinger_dyson(f.dmft, f.lattice, f.green, f.lattice.qmaps, f.baxis, custom; max_workspace_bytes=1)
    @test_throws DimensionMismatch P.orbital_schwinger_dyson(f.dmft, f.lattice, f.green, f.lattice.qmaps, f.baxis, custom; qweights=[1.0])
    @test_throws ArgumentError P.orbital_schwinger_dyson(f.dmft, f.lattice, f.green, f.lattice.qmaps, f.baxis, custom; max_output_bytes=1)
end

@testset "Orbital screened ladder: q-local cancellation, interacting tensor and zero-coupling limit" begin
    f = _fixture()
    chosen = [(a, b) for a in 1:2 for b in 1:2]
    values = zeros(ComplexF64, 4, 4, 2, 2, 2)
    for iΩ in 1:2, i in 1:4, n in 1:2
        values[i, i, n, n, iΩ] = 0.2
    end
    U = P.kanamori_interaction(1; U=2.0)
    names = U.labels
    aa = P.OrbitalPairSusceptibility(f.axis, f.baxis, chosen, names, values)
    bb = P.OrbitalPairSusceptibility(f.axis, f.baxis, chosen, names, values .+ 0.01)
    symmetric = P.OrbitalLadderResult(Any[:q0, :qpi], [aa, aa], nothing, [NamedTuple[], NamedTuple[]], Dict{Symbol,Any}())
    contrast = P.OrbitalLadderResult(Any[:q0, :qpi], [aa, bb], nothing, [NamedTuple[], NamedTuple[]], Dict{Symbol,Any}())
    tmp = zeros(ComplexF64, 2, 2, 2, 2)
    kernel0 = P.screened_orbital_ladder_kernel(symmetric, U)
    kernel0.fill!(tmp, 1, 1, 1)
    @test all(iszero, tmp)
    kernel = P.screened_orbital_ladder_kernel(contrast, U)
    kernel.fill!(tmp, 1, 1, 1)
    @test maximum(abs, tmp) > 1e-8
    @test_throws ArgumentError P.screened_orbital_ladder_kernel(contrast, U; max_kernel_bytes=1)
    @test_throws ArgumentError P.screened_orbital_ladder_kernel(contrast, U; qweights=[1])
    @test_throws ArgumentError P.screened_orbital_ladder_kernel(contrast, f.U)
end

@testset "Orbital screened ladder: low-level and explicit typed forward route" begin
    f = _fixture()
    pairs = [(a, b) for a in 1:2 for b in 1:2]
    vertex = P.OrbitalPairIrreducibleVertex(f.axis, f.baxis, pairs, f.labels,
        zeros(ComplexF64, 4, 4, 2, 2, 2), Dict{Symbol,Any}())
    problem = P.OrbitalScreenedLadderProblem(f.dmft, f.green, vertex, f.baxis)
    solver = P.OrbitalScreenedLadderSolver(max_workspace_bytes=20_000_000)
    outcome = P.solve(solver, problem)
    @test outcome.metadata[:full_multiorbital_dga] === false
    @test outcome.metadata[:frequency_box_complete]
    @test isnothing(outcome.ladder.susceptibilities)
    @test outcome.ladder.metadata[:full_susceptibilities_retained] === false
    @test outcome.nonlocal_self_energy.values ≈ repeat(reshape(f.dmft.self_energy.values, 2, 2, 1, 2), 1, 1, 2, 1) atol=1e-10
    @test P.validate_orbital_schwinger_dyson(f.dmft, outcome.nonlocal_self_energy).valid
    @test outcome.corrected_green isa P.MatrixMomentumResolvedGreenFunction
    direct_problem = P.OrbitalScreenedLadderProblem(f.dmft, vertex, f.baxis)
    @test direct_problem.green.values ≈ f.green.values atol=1e-11
    rep = P.Representation(:orbital_sde, P.FermionFockSpace(2), P.FermionAlgebra(2),
        P.FermionOccupationBasis([1, 2]); ordering=[1, 2])
    model = P.ManyBodyModel(rep, 0.1 * P.nf(1) - 0.12 * P.nf(2))
    context = P.OrbitalScreenedLadderForwardContext(model, problem)
    request = P.MomentumMatsubaraSelfEnergyRequest()
    @test P.solver_compatibility(solver, model, request, context).compatible
    @test !P.solver_compatibility(solver, model, request).compatible
    forwarded = P.forward(P.ForwardProblem(model; solver=solver, method_context=context, observable=request))
    @test forwarded.result.values ≈ outcome.nonlocal_self_energy.values atol=1e-12
    @test forwarded.solution.metadata[:symbolic_lowering_certified] === false
    @test_throws ArgumentError P.solve(P.OrbitalScreenedLadderSolver(max_output_bytes=1), problem)
end

end # module MultiOrbitalSchwingerDysonValidation
