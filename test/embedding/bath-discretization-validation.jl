#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module BathDiscretizationValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

# References:
# M. Caffarel and W. Krauth, Phys. Rev. Lett. 72, 1545-1548 (1994), DOI: 10.1103/PhysRevLett.72.1545.
# A. Liebsch and H. Ishida, J. Phys.: Condens. Matter 24, 053201 (2012), DOI: 10.1088/0953-8984/24/5/053201.

@testset "Finite-bath hybridization invariants" begin
    axis = P.FermionicMatsubaraAxis(8.0, 64)
    bath = P.DiscreteBath([-1.0, 0.0, 1.0], [0.4, 0.5, 0.4])
    permuted = P.DiscreteBath([1.0, -1.0, 0.0], [0.4, 0.4, 0.5])
    target = P.hybridization_function(bath, axis)
    permuted_target = P.hybridization_function(permuted, axis)

    @test target.values ≈ permuted_target.values atol=2e-14 rtol=2e-14
    @test P.bath_discretization_residual(target, bath) <= 2e-14
    @test P.bath_particle_hole_residual(bath) <= 2e-14
    @test P.causality_residual(target) <= 2e-14
end

@testset "Bath-size convergence and multistart diagnostics" begin
    axis = P.FermionicMatsubaraAxis(8.0, 64)
    generating_bath = P.DiscreteBath([-1.0, 0.0, 1.0], [0.4, 0.5, 0.4])
    target = P.hybridization_function(generating_bath, axis)
    options = P.BathFitOptions(maxiter=500, tol=1e-10, energy_window=(-2.0, 2.0), multistart=9)
    fits = [P.fit_bath(target, sites; options=options) for sites in 1:3]
    residuals = [P.bath_discretization_residual(target, fit.bath) for fit in fits]

    @test residuals[3] < 2e-8
    @test residuals[3] < residuals[2] < residuals[1]
    @test all(isfinite(fit.objective) for fit in fits)
    @test all(all(isfinite, fit.metadata[:start_objectives]) for fit in fits)
    @test all(length(fit.metadata[:start_termination]) == options.multistart for fit in fits)
    @test length(fits[3].metadata[:start_objectives]) == options.multistart
    @test fits[3].objective == minimum(fits[3].metadata[:start_objectives])
    @test fits[3].metadata[:objective_spread] >= 0.0

    impurity_problem = P.ImpurityProblem(0.0, axis; target_hybridization=target)
    impurity_result = P.solve(P.EDImpuritySolver(bath_sites=3, bath_fit=options), impurity_problem)
    @test impurity_result.metadata[:bath_fit_metadata][:selected_start] == fits[3].metadata[:selected_start]
    @test impurity_result.metadata[:bath_fit_metadata][:start_objectives] ≈ fits[3].metadata[:start_objectives] atol=0.0 rtol=0.0
end


@testset "Particle-hole-symmetric bath fitting" begin
    axis = P.FermionicMatsubaraAxis(10.0, 64)
    chemical_potential = 1.25
    generating_bath = P.DiscreteBath([chemical_potential - 1.1, chemical_potential - 0.35,
                                      chemical_potential + 0.35, chemical_potential + 1.1], [0.32, 0.47, 0.47, 0.32])
    target = P.hybridization_function(generating_bath, axis; chemical_potential=chemical_potential)
    options = P.BathFitOptions(maxiter=500, tol=1e-10, weight_power=1.0,
                               energy_window=(chemical_potential - 2.0, chemical_potential + 2.0), multistart=7, symmetry=:particle_hole)
    fit = P.fit_bath(target, 4; chemical_potential=chemical_potential, options=options)
    fitted = P.hybridization_function(fit.bath, axis; chemical_potential=chemical_potential)

    @test fit.metadata[:symmetry] === :particle_hole
    @test fit.metadata[:parameter_count] == 4
    @test P.bath_particle_hole_residual(fit.bath; chemical_potential=chemical_potential) <= 2e-14
    @test maximum(abs(real(fitted[1, 1, i])) for i in 1:length(fitted)) <= 2e-14
    @test P.bath_discretization_residual(target, fit.bath; chemical_potential=chemical_potential) < 2e-6
    @test_throws ArgumentError P.fit_bath(target, 3; chemical_potential=chemical_potential, options=options)

    asymmetric_seed = P.DiscreteBath([chemical_potential - 1.0, chemical_potential - 0.2,
                                      chemical_potential + 0.4, chemical_potential + 1.2], [0.30, 0.42, 0.49, 0.34])
    warm_fit = P.fit_bath(target, 4; chemical_potential=chemical_potential, options=options, initial_bath=asymmetric_seed)
    @test warm_fit.metadata[:warm_start_used]
    @test P.bath_particle_hole_residual(warm_fit.bath; chemical_potential=chemical_potential) <= 2e-14

    legacy_options = P.BathFitOptions(200, 1e-10, 1e-4, 1.0, nothing, Inf, 3)
    @test legacy_options.symmetry === :none
    @test P.BathFitOptions().symmetry === :none
end

@testset "Interacting atomic impurity thermodynamics" begin
    U = 2.0
    beta = 6.0
    axis = P.FermionicMatsubaraAxis(beta, 48)
    empty_bath = P.DiscreteBath(Float64[], ComplexF64[])
    problem = P.ImpurityProblem(U, axis; chemical_potential=U / 2, bath=empty_bath)
    result = P.solve(P.EDImpuritySolver(bath_sites=0), problem)
    exact_double_occupancy = inv(2 * (1 + exp(beta * U / 2)))

    @test abs(result.density - 1.0) <= 2e-12
    @test abs(result.double_occupancy - exact_double_occupancy) <= 2e-12
    @test result.green_function.values ≈ P.atomic_hubbard_green(axis, U).values atol=2e-12 rtol=2e-12
    @test result.self_energy.values ≈ P.atomic_hubbard_self_energy(axis, U).values atol=2e-11 rtol=2e-11
end

end
