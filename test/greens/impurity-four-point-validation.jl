#!/usr/bin/env julia

using Test
using LinearAlgebra
using Random
using Phundamental


@testset "factored ordered-trace algebra" begin
    rng = MersenneTwister(0x5eed)
    dimension = 4
    beta = 2.3
    ntime = 7
    energies = sort!(randn(rng, dimension))
    random_operator() = ComplexF64.(randn(rng, dimension, dimension) .+ im .* randn(rng, dimension, dimension))
    creation1 = random_operator()
    annihilation1 = random_operator()
    creation2 = random_operator()
    annihilation2 = random_operator()

    for equal_time in (:ordered, :symmetric)
        direct_backend = TauGridFourPoint(ntime=ntime, max_dimension=dimension, subtract_disconnected=true, algorithm=:direct, parallel=:serial,
                                          equal_time=equal_time)
        serial_backend = TauGridFourPoint(ntime=ntime, max_dimension=dimension, subtract_disconnected=true, algorithm=:factored, parallel=:serial,
                                          equal_time=equal_time)
        direct_cube = Phundamental.Greens._connected_tau_cube_direct(energies, creation1, annihilation1, creation2, annihilation2, beta, direct_backend)
        serial_cube = Phundamental.Greens._connected_tau_cube_factored(energies, creation1, annihilation1, creation2, annihilation2, beta, serial_backend)
        @test all(isfinite, serial_cube)
        @test isapprox(serial_cube, direct_cube; rtol=5e-12, atol=5e-12)

        if Base.Threads.nthreads() > 1
            threaded_backend = TauGridFourPoint(ntime=ntime, max_dimension=dimension, subtract_disconnected=true, algorithm=:factored, parallel=:threads,
                                                equal_time=equal_time)
            threaded_cube = Phundamental.Greens._connected_tau_cube_factored(energies, creation1, annihilation1, creation2, annihilation2, beta, threaded_backend)
            @test all(isfinite, threaded_cube)
            @test isapprox(threaded_cube, serial_cube; rtol=5e-12, atol=5e-12)
        end
    end

    ordered_backend = TauGridFourPoint(ntime=ntime, max_dimension=dimension, subtract_disconnected=true, algorithm=:factored, parallel=:serial,
                                       equal_time=:ordered)
    symmetric_backend = TauGridFourPoint(ntime=ntime, max_dimension=dimension, subtract_disconnected=true, algorithm=:factored, parallel=:serial,
                                         equal_time=:symmetric)
    ordered_cube = Phundamental.Greens._connected_tau_cube_factored(energies, creation1, annihilation1, creation2, annihilation2, beta, ordered_backend)
    symmetric_cube = Phundamental.Greens._connected_tau_cube_factored(energies, creation1, annihilation1, creation2, annihilation2, beta, symmetric_backend)
    @inbounds for i in 1:ntime, j in 1:ntime, k in 1:ntime
        if i != j && i != k && j != k
            @test isapprox(symmetric_cube[i, j, k], ordered_cube[i, j, k]; rtol=5e-12, atol=5e-12)
        end
    end
end

@testset "noninteracting impurity four-point reference backend" begin
    beta = 4.0
    axis = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1])
    bosonic = BosonicMatsubaraAxis(beta, [0])
    problem = ImpurityProblem(0.0, axis; chemical_potential=0.0, bath=DiscreteBath(Float64[], ComplexF64[]))
    impurity = solve(EDImpuritySolver(bath_sites=0, backend=:dense), problem)

    direct = impurity_two_particle(impurity.thermal_state, axis, bosonic;
                                   backend=TauGridFourPoint(ntime=6, max_dimension=16, algorithm=:direct, parallel=:serial, equal_time=:symmetric))
    factored = impurity_two_particle(impurity.thermal_state, axis, bosonic;
                                     backend=TauGridFourPoint(ntime=6, max_dimension=16, algorithm=:factored, parallel=:serial, equal_time=:symmetric))
    @test isapprox(factored.spin_resolved[:upup].values, direct.spin_resolved[:upup].values; rtol=2e-12, atol=2e-12)
    @test isapprox(factored.spin_resolved[:updown].values, direct.spin_resolved[:updown].values; rtol=2e-12, atol=2e-12)

    if Base.Threads.nthreads() > 1
        threaded = impurity_two_particle(impurity.thermal_state, axis, bosonic;
                                         backend=TauGridFourPoint(ntime=6, max_dimension=16, algorithm=:factored, parallel=:threads, equal_time=:symmetric))
        @test isapprox(threaded.spin_resolved[:upup].values, factored.spin_resolved[:upup].values; rtol=2e-12, atol=2e-12)
        @test isapprox(threaded.spin_resolved[:updown].values, factored.spin_resolved[:updown].values; rtol=2e-12, atol=2e-12)
    end

    coarse = impurity_two_particle(impurity.thermal_state, axis, bosonic; backend=TauGridFourPoint(ntime=12, max_dimension=16))
    fine = impurity_two_particle(impurity.thermal_state, axis, bosonic; backend=TauGridFourPoint(ntime=24, max_dimension=16))
    reference = local_bubble(impurity.green_function, bosonic; channel=:upup, missing_frequency=:zero)
    active = reference.metadata[:active_indices_by_bosonic][1]
    coarse_error = norm(coarse.spin_resolved[:upup].values[active, active, 1] - reference.values[active, active, 1]) / norm(reference.values[active, active, 1])
    fine_error = norm(fine.spin_resolved[:upup].values[active, active, 1] - reference.values[active, active, 1]) / norm(reference.values[active, active, 1])

    @test fine_error < coarse_error
    @test fine_error < 0.20
    @test norm(fine.spin_resolved[:updown].values) < 0.20 * max(norm(fine.spin_resolved[:upup].values), eps(Float64))
    @test impurity_two_particle_residual(coarse, fine) > 0
    @test get(reference.metadata, :diagonal_in_fermionic_indices, false)
end
