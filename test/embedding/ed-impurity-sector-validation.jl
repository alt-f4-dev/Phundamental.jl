#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module EDImpuritySectorValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

@testset "Spin-resolved Anderson sector basis" begin
    bath = P.DiscreteBath([-0.7, 0.9], [0.55, 0.42])
    model = P.build_model(P.AndersonImpurityModel(2.0; chemical_potential=1.0, bath_energies=bath.energies, hybridizations=bath.hybridizations))
    sector_data = P.optimized_matrix_free_operator(model; sector=P.FermionSector(nup=1, ndown=2), backend=:fermion)
    @test sector_data.dimension == 9
    @test length(model.parameters[:fermion_mode_map]) == 6
end

@testset "Sectorized ED matches dense impurity reference" begin
    axis = P.FermionicMatsubaraAxis(5.0, 32)
    bath = P.DiscreteBath([-0.8, 0.75], [0.62, 0.47])
    problem = P.ImpurityProblem(2.4, axis; chemical_potential=1.2, impurity_energy=0.0, bath=bath)
    dense = P.solve(P.EDImpuritySolver(bath_sites=2, backend=:dense), problem)
    sectorized = P.solve(P.EDImpuritySolver(bath_sites=2, backend=:sectorized), problem)

    green_scale = max(norm(P.scalar_frequency_values(dense.green_function)), 1.0)
    sigma_scale = max(norm(P.scalar_frequency_values(dense.self_energy)), 1.0)
    @test norm(P.scalar_frequency_values(sectorized.green_function) - P.scalar_frequency_values(dense.green_function)) / green_scale <= 2e-11
    @test norm(P.scalar_frequency_values(sectorized.self_energy) - P.scalar_frequency_values(dense.self_energy)) / sigma_scale <= 5e-11
    @test abs(sectorized.density - dense.density) <= 2e-12
    @test abs(sectorized.double_occupancy - dense.double_occupancy) <= 2e-12
    @test sectorized.metadata[:backend] == :sectorized
    @test dense.metadata[:backend] == :dense
    @test sectorized.metadata[:max_sector_dimension] < sectorized.metadata[:full_hilbert_dimension]
    @test abs(P.expectation(sectorized.thermal_state, :impurity_density).value - sectorized.density) <= 2e-12
end

@testset "Threaded sectorized ED matches serial sectorized reference" begin
    axis = P.FermionicMatsubaraAxis(6.0, 48)
    bath = P.DiscreteBath([-0.9, 0.7], [0.58, 0.46])
    problem = P.ImpurityProblem(2.2, axis; chemical_potential=1.1, impurity_energy=0.0, bath=bath)
    serial = P.solve(P.EDImpuritySolver(bath_sites=2, backend=:sectorized, threading=:serial), problem)
    threaded = P.solve(P.EDImpuritySolver(bath_sites=2, backend=:sectorized, threading=:threads), problem)

    green_scale = max(norm(P.scalar_frequency_values(serial.green_function)), 1.0)
    sigma_scale = max(norm(P.scalar_frequency_values(serial.self_energy)), 1.0)
    @test norm(P.scalar_frequency_values(threaded.green_function) - P.scalar_frequency_values(serial.green_function)) / green_scale <= 2e-12
    @test norm(P.scalar_frequency_values(threaded.self_energy) - P.scalar_frequency_values(serial.self_energy)) / sigma_scale <= 5e-12
    @test abs(threaded.density - serial.density) <= 2e-13
    @test abs(threaded.double_occupancy - serial.double_occupancy) <= 2e-13
    @test abs(threaded.thermal_state.logZ - serial.thermal_state.logZ) <= 2e-13
    @test length(threaded.thermal_state.sectors) == length(serial.thermal_state.sectors)
    @test all(norm(threaded.thermal_state.sectors[i].energies - serial.thermal_state.sectors[i].energies) <= 2e-12 for i in eachindex(serial.thermal_state.sectors))
    @test serial.metadata[:threading_effective] == :serial
    @test threaded.metadata[:threading_requested] == :threads
    @test threaded.metadata[:threading_effective] == (Threads.nthreads() > 1 ? :threads : :serial)
    @test threaded.metadata[:julia_threads] == Threads.nthreads()
    @test_throws ArgumentError P.EDImpuritySolver(bath_sites=2, backend=:sectorized, threading=:invalid)
end

@testset "Warm-started bath fitting preserves the optimum" begin
    axis = P.FermionicMatsubaraAxis(12.0, 64)
    reference_bath = P.DiscreteBath([-1.1, 0.8], [0.63, 0.51])
    target = P.hybridization_function(reference_bath, axis; chemical_potential=0.2)
    options = P.BathFitOptions(maxiter=300, tol=1e-9, energy_window=(-2.5, 2.5), multistart=5)
    cold = P.fit_bath(target, 2; chemical_potential=0.2, options=options)
    warm = P.fit_bath(target, 2; chemical_potential=0.2, options=options, initial_bath=cold.bath)

    @test warm.metadata[:warm_start_used]
    @test warm.metadata[:starts_executed] >= 2
    @test warm.objective <= cold.objective + max(1e-12, 1e-7 * cold.objective)
    @test P.bath_discretization_residual(target, warm.bath; chemical_potential=0.2) <=
          P.bath_discretization_residual(target, cold.bath; chemical_potential=0.2) + 1e-9
end

@testset "Optional DMFT closure verification" begin
    fit_options = P.BathFitOptions(maxiter=80, tol=1e-7, energy_window=(-3.0, 3.0), multistart=2)
    impurity_solver = P.EDImpuritySolver(bath_sites=1, bath_fit=fit_options, backend=:sectorized)
    solver = P.DMFTSolver(impurity_solver=impurity_solver, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=3, verify_closure=false)
    problem = P.DMFTProblem(P.BetheLattice(2.0), 0.0; temperature=0.5, chemical_potential=0.0, nfrequencies=16)
    result = P.solve(solver, problem)

    @test result.converged
    @test result.metadata[:closure_verified] == false
    @test result.impurity_result.metadata[:backend] == :sectorized
end

end
