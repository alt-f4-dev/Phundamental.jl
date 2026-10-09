#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

@testset "end-to-end noninteracting ladder DGA" begin
    shape = (2, 2)
    lattice = periodic_momentum_lattice(shape, k -> -2 * (cos(k[1]) + cos(k[2])))
    beta = 2.0
    temperature = inv(beta)
    dmft_problem = DMFTProblem(lattice, 0.0; temperature=temperature, chemical_potential=0.0, nfrequencies=3,
                               axis=FermionicMatsubaraAxis(beta, [-2, -1, 0, 1, 2]))
    dmft_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=0, backend=:dense), mixing=LinearMixing(1.0), tol=1e-10, maxiter=3)
    dmft = solve(dmft_solver, dmft_problem)
    @test dmft.converged

    bosonic = BosonicMatsubaraAxis(beta, [0])
    backend = TauGridFourPoint(ntime=16, max_dimension=16, algorithm=:factored, parallel=:serial)
    impurity = impurity_two_particle(dmft.impurity_result.thermal_state, dmft.problem.axis, bosonic; backend=backend)
    problem = LadderDGAProblem(dmft, lattice, bosonic; impurity_backend=backend, impurity_two_particle=impurity)
    method = LadderDGA(pseudoinverse=true, reconstruct_self_energy=true)
    solver = DiagrammaticDMFTSolver(method=method, execution=DiagrammaticExecutionPolicy(parallel=:serial))
    result = solve(solver, problem)
    diagnostics = validate_diagrammatic_dmft_result(result; require_self_energy=true)

    @test diagnostics.valid
    @test result.metadata[:nonlocal_self_energy_reconstructed]
    @test result.metadata[:shared_lattice_bubble_storage]
    @test result.lattice_bubbles[:charge][1].values === result.lattice_bubbles[:magnetic][1].values
    @test maximum(abs.(result.self_energy.values)) < 1e-8
    @test size(susceptibility(result, :charge).values, 1) == prod(shape)
    @test size(susceptibility(result, :magnetic).values) == size(susceptibility(result, :charge).values)
    @test response_kind(susceptibility(result, :charge)) == :susceptibility
    @test frequency_domain(susceptibility(result, :charge)) == :matsubara
    @test response_statistics(susceptibility(result, :charge)) == :bosonic
    @test requires_analytic_continuation(susceptibility(result, :charge))

    if Base.Threads.nthreads() > 1
        threaded = solve(DiagrammaticDMFTSolver(method=method, execution=DiagrammaticExecutionPolicy(parallel=:threads)), problem)
        @test isapprox(susceptibility(threaded, :charge).values, susceptibility(result, :charge).values; rtol=2e-12, atol=2e-12)
        @test isapprox(susceptibility(threaded, :magnetic).values, susceptibility(result, :magnetic).values; rtol=2e-12, atol=2e-12)
        @test isapprox(threaded.self_energy.values, result.self_energy.values; rtol=2e-12, atol=2e-12)
    end
end
