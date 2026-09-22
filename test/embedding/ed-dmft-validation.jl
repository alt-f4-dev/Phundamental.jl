#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module EDDMFTValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

import Main.Phundamental.Solvers: solve

mutable struct ClosureRetryImpuritySolver <: P.AbstractImpuritySolver
    calls::Int
end

function solve(solver::ClosureRetryImpuritySolver, problem::P.ImpurityProblem)
    solver.calls += 1
    amplitude = solver.calls == 1 ? 0.05 : solver.calls == 2 ? 0.20 : 0.05
    sigma = P.SelfEnergy(problem.axis, fill(ComplexF64(amplitude), length(problem.axis)); labels=[:impurity])
    green = P.GreenFunction(problem.axis, fill(-1.0im, length(problem.axis)); labels=[:impurity])
    noninteracting = P.NonInteractingGreenFunction(problem.axis, fill(-1.0im, length(problem.axis)); labels=[:impurity])
    bath = P.DiscreteBath(Float64[], ComplexF64[])
    metadata = Dict{Symbol,Any}(:bath_fit_converged => true, :bath_fit_objective => 0.0)
    return P.ImpuritySolverResult(nothing, green, noninteracting, sigma, 1.0, 0.25, bath, nothing, metadata)
end

@testset "Finite ED bath and noninteracting impurity" begin
    axis = P.FermionicMatsubaraAxis(6.0, 48)
    bath = P.DiscreteBath([-0.8, 0.9], [0.7, 0.5])
    target = P.hybridization_function(bath, axis; chemical_potential=0.15)
    options = P.BathFitOptions(maxiter=300, tol=1e-9, energy_window=(-2.0, 2.0), multistart=5)
    fit = P.fit_bath(target, 2; chemical_potential=0.15, options=options)
    fitted = P.hybridization_function(fit.bath, axis; chemical_potential=0.15)
    relative_fit_error = norm(fitted.values - target.values) / norm(target.values)
    @test relative_fit_error < 5e-3

    problem = P.ImpurityProblem(0.0, axis; chemical_potential=0.15, bath=bath)
    result = P.solve(P.EDImpuritySolver(bath_sites=2), problem)
    @test P.noninteracting_self_energy_residual(result) <= 2e-11
    @test P.causality_residual(result.green_function) <= 2e-12
    @test 0.0 <= result.density <= 2.0
    @test 0.0 <= result.double_occupancy <= 1.0
end

@testset "Half-filled atomic ED-DMFT" begin
    U = 2.0
    temperature = 0.5
    problem = P.DMFTProblem(P.AtomicLattice(), U; temperature=temperature, chemical_potential=U / 2, nfrequencies=32)
    impurity_solver = P.EDImpuritySolver(bath_sites=0, bath_fit=P.BathFitOptions(tol=1e-12))
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=P.LinearMixing(1.0), tol=1e-10, maxiter=5)
    result = P.solve(solver, problem)
    expected = P.atomic_hubbard_self_energy(problem.axis, U)
    diagnostics = P.validate_dmft_result(result)

    @test result.converged
    @test P.self_energy_residual(expected, result.self_energy) <= 2e-10
    @test abs(result.impurity_result.density - 1.0) <= 2e-12
    @test P.impurity_local_residual(result) <= 2e-10
    @test diagnostics[:atomic_green_residual] <= 2e-10
    @test diagnostics[:atomic_self_energy_residual] <= 2e-10
    @test diagnostics[:self_energy_causality_residual] <= 2e-10
end

@testset "Noninteracting Bethe ED-DMFT closure" begin
    fit_options = P.BathFitOptions(maxiter=120, tol=1e-7, energy_window=(-3.0, 3.0), multistart=3)
    impurity_solver = P.EDImpuritySolver(bath_sites=2, bath_fit=fit_options)
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=P.LinearMixing(1.0), tol=1e-9, maxiter=3)
    problem = P.DMFTProblem(P.BetheLattice(2.0), 0.0; temperature=0.5, chemical_potential=0.0, nfrequencies=24)
    result = P.solve(solver, problem)
    diagnostics = P.validate_dmft_result(result)

    @test result.converged
    @test maximum(abs.(result.self_energy.values)) <= 2e-10
    @test diagnostics[:bethe_self_consistency_residual] <= 2e-10
    @test diagnostics[:self_energy_causality_residual] <= 2e-10

    continued = P.solve(solver, problem; initial_self_energy=result.self_energy, initial_bath=result.impurity_result.bath)
    @test continued.converged
    @test continued.metadata[:continuation_self_energy]
    @test continued.metadata[:continuation_bath]
    @test P.self_energy_residual(result.self_energy, continued.self_energy) <= 2e-10
end


@testset "Particle-hole-constrained half-filled Bethe ED-DMFT" begin
    interaction = 1.5
    chemical_potential = interaction / 2
    fit_options = P.BathFitOptions(maxiter=300, tol=1e-9, weight_power=1.0,
                                   energy_window=(chemical_potential - 3.0, chemical_potential + 3.0), multistart=5, symmetry=:particle_hole)
    impurity_solver = P.EDImpuritySolver(bath_sites=2, bath_fit=fit_options, backend=:sectorized)
    mixing = P.AdaptiveLinearMixing(initial_alpha=0.5, min_alpha=0.1, max_alpha=0.8, recovery_window=3)
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=mixing, tol=1e-6, maxiter=60, verify_closure=true)
    problem = P.DMFTProblem(P.BetheLattice(sqrt(2.0)), interaction; temperature=0.25, chemical_potential=chemical_potential, nfrequencies=32)
    result = P.solve(solver, problem)
    diagnostics = P.validate_dmft_result(result)

    @test result.converged
    @test result.metadata[:closure_residual] <= solver.tol
    @test diagnostics[:bath_particle_hole_residual] <= 2e-14
    @test maximum(abs(real(result.self_energy[1, 1, i]) - interaction / 2) for i in 1:length(result.self_energy)) <= 2e-6
    @test result.impurity_result.metadata[:bath_fit_metadata][:symmetry] === :particle_hole
    @test result.impurity_result.metadata[:bath_fit_metadata][:parameter_count] == 2
end

@testset "Adaptive DMFT convergence diagnostics" begin
    U = 2.0
    problem = P.DMFTProblem(P.AtomicLattice(), U; temperature=0.5, chemical_potential=U / 2, nfrequencies=24)
    impurity_solver = P.EDImpuritySolver(bath_sites=0, bath_fit=P.BathFitOptions(tol=1e-12))
    mixing = P.AdaptiveLinearMixing(initial_alpha=0.6, min_alpha=0.1, max_alpha=0.8)
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=mixing, tol=1e-10, maxiter=20, verify_closure=true)
    result = P.solve(solver, problem)
    diagnostics = P.validate_dmft_result(result)

    @test result.converged
    @test result.metadata[:termination] === :converged
    @test diagnostics[:termination] === :converged
    @test result.metadata[:best_iteration_residual] <= maximum(entry.residual for entry in result.history)
    @test result.metadata[:best_residual] == result.metadata[:best_iteration_residual]
    @test result.metadata[:final_iteration_residual] <= solver.tol
    @test result.metadata[:closure_residual] <= solver.tol
    @test diagnostics[:best_iteration_residual] == result.metadata[:best_iteration_residual]
    @test diagnostics[:closure_residual] == result.metadata[:closure_residual]
    @test 1 <= result.metadata[:iteration_of_best_residual] <= result.iterations
    @test all(entry -> mixing.min_alpha <= entry.mixing_alpha <= mixing.max_alpha, result.history)
    @test all(entry -> isfinite(entry.bath_fit_residual), result.history)
    @test P.Embedding._next_mixing_alpha(mixing, 0.5, 0.1, 0.2) == 0.25
    @test isapprox(P.Embedding._next_mixing_alpha(mixing, 0.5, 0.1, 0.05), 0.55; atol=0.0, rtol=8eps(Float64))

    recovery_mixing = P.AdaptiveLinearMixing(initial_alpha=0.1, min_alpha=0.05, max_alpha=0.8, recovery_window=3)
    recovery_alpha = 0.1
    recovery_streak = 0
    recovery_alpha, recovery_streak = P.Embedding._next_mixing_state(recovery_mixing, recovery_alpha, 1.0, 0.95, recovery_streak)
    @test recovery_alpha == 0.1
    @test recovery_streak == 1
    recovery_alpha, recovery_streak = P.Embedding._next_mixing_state(recovery_mixing, recovery_alpha, 0.95, 0.90, recovery_streak)
    @test recovery_alpha == 0.1
    @test recovery_streak == 2
    recovery_alpha, recovery_streak = P.Embedding._next_mixing_state(recovery_mixing, recovery_alpha, 0.90, 0.85, recovery_streak)
    @test isapprox(recovery_alpha, 0.11; atol=0.0, rtol=8eps(Float64))
    @test recovery_streak == 0

    legacy_mixing = P.AdaptiveLinearMixing(0.5, 0.05, 0.8, 0.5, 1.1, 0.8, 1.05)
    @test legacy_mixing.recovery_window == 3
end

@testset "DMFT closure retry continues within the iteration budget" begin
    impurity_solver = ClosureRetryImpuritySolver(0)
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=P.LinearMixing(1.0), tol=0.1, maxiter=3, verify_closure=true)
    problem = P.DMFTProblem(P.AtomicLattice(), 0.0; temperature=0.5, chemical_potential=0.0, nfrequencies=8)
    result = P.solve(solver, problem)

    @test result.converged
    @test result.metadata[:termination] === :converged
    @test result.iterations == 2
    @test impurity_solver.calls == 4
    @test result.metadata[:closure_attempts] == 2
    @test result.metadata[:closure_retry_started]
    @test result.metadata[:closure_residual] <= solver.tol
    @test result.history[1].residual <= solver.tol
end

@testset "DMFT termination classification" begin
    policy = P.DMFTConvergencePolicy(stagnation_window=4, oscillation_window=6, oscillation_fraction=0.75)
    make_entry(iteration, residual) = P.DMFTIteration(iteration, residual, 1.0, 0.1, nothing, nothing, 0.0, 0.0, 0.0, 0.5)
    stagnated = [make_entry(i, 0.1 + 1e-6 * i) for i in 1:6]
    oscillatory = [make_entry(i, residual) for (i, residual) in enumerate((0.2, 0.1, 0.2, 0.1, 0.2, 0.1))]
    decreasing = [make_entry(i, 0.2 / i) for i in 1:6]

    @test P.Embedding._classify_dmft_termination(stagnated, 1e-8, policy) === :stagnated
    @test P.Embedding._classify_dmft_termination(oscillatory, 1e-8, policy) === :oscillatory
    @test P.Embedding._classify_dmft_termination(decreasing, 1e-8, policy) === :maxiter
end

end
