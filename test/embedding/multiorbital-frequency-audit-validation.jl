#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalFrequencyAuditValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _dmft_fixture(; U=0.0)
    interaction = P.kanamori_interaction(1; U=U, J=0.0)
    onsite = Matrix(Diagonal(ComplexF64[0.15, 0.15]))
    lattice = P.MatrixMomentumLattice(reshape(onsite, 2, 2, 1);
        labels=interaction.labels, kpoints=[:atomic], qmaps=[[1]])
    axis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis, chemical_potential=0.2)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=P.MultiOrbitalEDSolver(bath_sites=0),
        mixing=P.LinearMixing(1.0), tol=1e-9, maxiter=8)
    result = P.solve(solver, problem)
    @test result.converged
    return result
end

@testset "Finite-bath Lehmann high-frequency moments: rigorous matrix remainder" begin
    dmft = _dmft_fixture(U=1.2)
    state = dmft.impurity_result.thermal_state
    moments = P.orbital_impurity_lehmann_moments(state; order=4)
    @test size(moments.coefficients) == (2, 2, 4)
    @test moments.spectral_radius > 0
    @test moments.metadata[:one_particle_only]
    @test !moments.metadata[:two_particle_vertex_tail_certified]
    @test moments.metadata[:zeroth_moment_residual] <= 1e-10
    @test moments.coefficients[:, :, 1] ≈ Matrix{ComplexF64}(I, 2, 2) atol=1e-10
    axis = P.FermionicMatsubaraAxis(4.0, [30, 45, 60])
    approximate = P.orbital_impurity_lehmann_tail(moments, axis)
    exact = P.multiorbital_impurity_green(state, axis)
    @test approximate.certified_for === :one_particle_finite_bath_only
    @test !approximate.green.metadata[:two_particle_vertex_tail_certified]
    for i in eachindex(axis.values)
        discrepancy = opnorm(approximate.green.values[:, :, i] - exact.values[:, :, i], 2)
        @test discrepancy <= approximate.error_bounds[i] + 1e-13
        @test isfinite(approximate.error_bounds[i])
    end
    @test approximate.error_bounds[end] < approximate.error_bounds[1]
    @test_throws ArgumentError P.orbital_impurity_lehmann_moments(state; order=0)
    @test_throws ArgumentError P.orbital_impurity_lehmann_moments(state; max_workspace_bytes=1)
    @test_throws ArgumentError P.orbital_impurity_lehmann_tail(moments, P.FermionicMatsubaraAxis(5.0, [50]))
    @test_throws ArgumentError P.orbital_impurity_lehmann_tail(moments, P.FermionicMatsubaraAxis(4.0, [0]))
end

@testset "Finite-bath Lehmann moments: complex orbital coherence and dense ED" begin
    interaction = P.kanamori_interaction(2; U=2.0, J=0.25)
    h = ComplexF64[0.1 0 0.0+0.08im 0;
                   0 0.1 0 0.0-0.08im;
                   0.0-0.08im 0 -0.2 0;
                   0 0.0+0.08im 0 -0.2]
    axis = P.FermionicMatsubaraAxis(5.0, [-1, 0])
    bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, 4, 0); labels=interaction.labels)
    impurity = P.MultiOrbitalImpurityProblem(h, interaction, axis; bath=bath, chemical_potential=0.3)
    sectorized = P.solve(P.MultiOrbitalEDSolver(bath_sites=0), impurity)
    dense = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:dense), impurity)
    moments = P.orbital_impurity_lehmann_moments(sectorized.thermal_state; order=3)
    dense_moments = P.orbital_impurity_lehmann_moments(dense.thermal_state; order=3)
    @test moments.coefficients ≈ dense_moments.coefficients atol=1e-10
    @test moments.coefficients[:, :, 1] ≈ Matrix{ComplexF64}(I, 4, 4) atol=1e-10
    @test moments.coefficients[:, :, 2] ≈ moments.coefficients[:, :, 2]' atol=1e-10
    high_axis = P.FermionicMatsubaraAxis(5.0, [50, 70])
    result = P.orbital_impurity_lehmann_tail(moments, high_axis)
    reference = P.multiorbital_impurity_green(sectorized.thermal_state, high_axis)
    for i in eachindex(high_axis.values)
        @test opnorm(result.green.values[:, :, i] - reference.values[:, :, i], 2) <=
            result.error_bounds[i] + 2e-13
    end
end

@testset "Ladder convergence: independent frequency support and quadrature dimensions" begin
    dmft = _dmft_fixture()
    small_f = P.FermionicMatsubaraAxis(4.0, [-2, -1, 0, 1])
    small_b = P.BosonicMatsubaraAxis(4.0, [-1, 0, 1])
    large_f = P.FermionicMatsubaraAxis(4.0, [-3, -2, -1, 0, 1, 2])
    large_b = P.BosonicMatsubaraAxis(4.0, [-2, -1, 0, 1, 2])
    small = P.OrbitalLadderWorkflowProblem(dmft, small_f, small_b; core_bosonic_indices=[0])
    large = P.OrbitalLadderWorkflowProblem(dmft, large_f, large_b; core_bosonic_indices=[0])
    solver = P.OrbitalLadderWorkflowSolver(four_point_backend=P.TauGridFourPoint(ntime=12, max_dimension=64),
        pseudoinverse=true, max_contractions=50_000_000)
    @test_throws ArgumentError P.orbital_ladder_convergence_study(solver, small, large;
        ntime_coarse=6, ntime_fine=16)
    @test_throws ArgumentError P.orbital_ladder_convergence_study(solver, large, small;
        ntime_coarse=12, ntime_fine=16)
    @test_throws ArgumentError P.orbital_ladder_convergence_study(solver, small, large;
        ntime_coarse=12, ntime_fine=16, max_total_contractions=1)
    report = P.orbital_ladder_convergence_study(solver, small, large;
        ntime_coarse=12, ntime_fine=16, max_total_contractions=200_000_000)
    @test report.final_result isa P.OrbitalLadderWorkflowResult
    @test length(report.stages) == 4
    @test [stage.name for stage in report.stages] == [:small_coarse, :small_fine, :large_coarse, :large_fine]
    @test [stage.ntime for stage in report.stages] == [12, 16, 12, 16]
    @test all(stage -> stage.time_seconds >= 0 && stage.allocated_bytes >= 0, report.stages)
    @test all(isfinite, (report.frequency_comparison.maximum_absolute_difference,
        report.quadrature_comparison.maximum_absolute_difference, report.coupled_difference.maximum_absolute_difference))
    @test report.metadata[:midpoint_aliasing_preflight_passed]
    @test report.metadata[:two_particle_vertex_tail_certified] === false
    @test report.metadata[:full_multiorbital_dga] === false
    @test report.final_result.metadata[:high_frequency_completion] === :not_certified
    @test size(report.final_result.nonlocal_self_energy.values) == (2, 2, 1, 2)
end

end # module MultiorbitalFrequencyAuditValidation
