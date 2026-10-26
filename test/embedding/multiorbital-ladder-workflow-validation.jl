#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalLadderWorkflowValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _fixture(; interaction_strength=0.0, atomic=true)
    interaction = P.kanamori_interaction(1; U=interaction_strength, J=0.0)
    onsite = Matrix(Diagonal(ComplexF64[0.1, 0.1]))
    hop = Matrix(Diagonal(ComplexF64[0.12, 0.12]))
    blocks = atomic ? reshape(onsite, 2, 2, 1) : cat(onsite + hop, onsite - hop; dims=3)
    labels = atomic ? [:k0] : [:k0, :kpi]
    qmaps = atomic ? [[1]] : [[1, 2], [2, 1]]
    lattice = P.MatrixMomentumLattice(blocks; labels=interaction.labels, kpoints=labels, qmaps=qmaps)
    axis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis, chemical_potential=0.2)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=P.MultiOrbitalEDSolver(bath_sites=0),
        mixing=P.LinearMixing(1.0), tol=1e-9, maxiter=8)
    dmft = P.solve(solver, problem)
    @test dmft.converged
    return dmft
end

function _axes(beta=4.0)
    return P.FermionicMatsubaraAxis(beta, [-2, -1, 0, 1]), P.BosonicMatsubaraAxis(beta, [-1, 0, 1])
end

@testset "Orbital ladder workflow: input and frequency-support contracts" begin
    dmft = _fixture()
    faxis, baxis = _axes()
    p = P.OrbitalLadderWorkflowProblem(dmft, faxis, baxis; core_bosonic_indices=[0])
    @test p.dmft === dmft
    @test p.core_bosonic_indices == [0]
    @test_throws ArgumentError P.OrbitalLadderWorkflowProblem(dmft, faxis, baxis; core_bosonic_indices=[2])
    @test_throws ArgumentError P.OrbitalLadderWorkflowProblem(dmft, P.FermionicMatsubaraAxis(4.0, [-1, 0]),
        P.BosonicMatsubaraAxis(4.0, [0]); core_bosonic_indices=[0])
    @test_throws ArgumentError P.OrbitalLadderWorkflowProblem(dmft, faxis, P.BosonicMatsubaraAxis(5.0, [-1, 0, 1]))
    @test_throws ArgumentError P.OrbitalLadderWorkflowSolver(max_retained_bytes=0)
    @test_throws ArgumentError P.OrbitalLadderWorkflowSolver(min_relative_pivot=1)
    green = P.orbital_ladder_expanded_green(dmft, faxis, baxis)
    @test green.labels == dmft.problem.labels
    @test all(n -> n in green.axis.indices, [k + m for k in faxis.indices for m in baxis.indices])
    @test green.metadata[:expanded_finite_bath_lehmann]
    @test_throws ArgumentError P.orbital_ladder_expanded_green(dmft, faxis, baxis; max_output_bytes=1)
end

@testset "Orbital ladder workflow: finite-box atomic limit and typed forward integration" begin
    dmft = _fixture(atomic=true)
    faxis, baxis = _axes()
    p = P.OrbitalLadderWorkflowProblem(dmft, faxis, baxis; core_bosonic_indices=[0])
    solver = P.OrbitalLadderWorkflowSolver(four_point_backend=P.TauGridFourPoint(ntime=8, max_dimension=64),
        pseudoinverse=true, max_contractions=50_000_000)
    result = P.solve(solver, p)
    @test result isa P.OrbitalLadderWorkflowResult
    @test result.metadata[:crossing_support_complete]
    @test result.metadata[:high_frequency_completion] === :not_certified
    @test result.metadata[:full_multiorbital_dga] === false
    @test result.metadata[:end_to_end_forward_integrated]
    @test result.nonlocal_self_energy.values[:, :, 1, :] ≈ dmft.self_energy.values atol=1e-8
    @test result.corrected_green isa P.MatrixMomentumResolvedGreenFunction
    @test result.crossing_core.diagnostics[:crossing_support_complete]
    @test all(isfinite, real.(result.nonlocal_self_energy.values))
    larger_f = P.FermionicMatsubaraAxis(4.0, [-3, -2, -1, 0, 1, 2])
    larger_b = P.BosonicMatsubaraAxis(4.0, [-2, -1, 0, 1, 2])
    larger_problem = P.OrbitalLadderWorkflowProblem(dmft, larger_f, larger_b; core_bosonic_indices=[0])
    larger_metadata = copy(result.metadata)
    larger_metadata[:support_fermionic_indices] = copy(larger_f.indices)
    larger_metadata[:support_bosonic_indices] = copy(larger_b.indices)
    synthetic_larger = P.OrbitalLadderWorkflowResult(larger_problem, result.two_particle,
        result.spin_channels, result.irreducible_vertices, result.ladders, result.crossing_core,
        result.nonlocal_self_energy, result.corrected_green, larger_metadata)
    comparison = P.orbital_ladder_support_convergence(result, synthetic_larger)
    @test comparison.converged
    @test comparison.maximum_absolute_difference == 0
    @test comparison.high_frequency_tail_certified === false
    @test_throws ArgumentError P.orbital_ladder_support_convergence(result, result)
    rep = P.Representation(:orbital_ladder_test, P.FermionFockSpace(2), P.FermionAlgebra(2),
        P.FermionOccupationBasis([1, 2]); ordering=[1, 2])
    model = P.ManyBodyModel(rep, 0.1 * P.nf(1) + 0.1 * P.nf(2))
    context = P.OrbitalLadderWorkflowForwardContext(model, p)
    request = P.MomentumMatsubaraSelfEnergyRequest()
    @test P.solver_compatibility(solver, model, request, context).compatible
    @test !P.solver_compatibility(solver, model, request).compatible
    @test request(model, result) === result.nonlocal_self_energy
end

@testset "Orbital ladder workflow: support comparisons never certify analytic tails" begin
    dmft = _fixture()
    faxis, baxis = _axes()
    p = P.OrbitalLadderWorkflowProblem(dmft, faxis, baxis; core_bosonic_indices=[0])
    @test_throws ArgumentError P.OrbitalLadderWorkflowProblem(dmft, P.FermionicMatsubaraAxis(4.0, [-1, 0]),
        baxis; core_bosonic_indices=[1])
    @test_throws ArgumentError P.OrbitalLadderWorkflowProblem(dmft, faxis, P.BosonicMatsubaraAxis(4.0, [0]);
        core_bosonic_indices=[0])
    @test P.OrbitalLadderWorkflowSolver().min_relative_pivot > 0
    @test P.solver_name(P.OrbitalLadderWorkflowSolver()) === :orbital_su2_ladder_workflow
end

end # module MultiorbitalLadderWorkflowValidation
