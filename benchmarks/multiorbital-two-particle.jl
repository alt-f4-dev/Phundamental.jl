#!/usr/bin/env julia

# Optional profiling harness. Not part of the package correctness tests.
using Phundamental
using Printf

function benchmark_two_particle(; ntimes=(6, 8, 12), samples=3)
    beta = 4.0
    axis = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1])
    bosonic = BosonicMatsubaraAxis(beta, [0, 1])
    interaction = kanamori_interaction(1; U=1.4)
    bath = MatrixDiscreteBath(Float64[], zeros(ComplexF64, 2, 0); labels=interaction.labels)
    problem = MultiOrbitalImpurityProblem(ComplexF64[0.2 0.08im; -0.08im -0.1], interaction, axis;
        chemical_potential=0.1, bath=bath)
    state = solve(MultiOrbitalEDSolver(; bath_sites=0), problem).thermal_state
    println("ntime  pairs    best_seconds    allocated_MiB    output_MiB    workspace_MiB")
    for ntime in ntimes, pairs in ([(1, 1)], [(1, 1), (2, 2)], [(a, b) for a in 1:2 for b in 1:2])
        method = TauGridFourPoint(; ntime=ntime, max_dimension=16, algorithm=:factored)
        task() = multiorbital_impurity_two_particle(state, axis, bosonic; backend=method, pairs=pairs)
        task()  # Compile and populate any method caches outside the measurement.
        best_time = Inf
        allocated = typemax(Int)
        for _ in 1:samples
            GC.gc()
            seconds = @elapsed result = task()
            bytes = @allocated task()
            best_time = min(best_time, seconds)
            allocated = min(allocated, bytes)
        end
        result = task()
        @printf("%5d  %5d  %14.6f  %15.3f  %12.4f  %15.4f\n", ntime, length(pairs),
            best_time, allocated / 2.0^20, result.diagnostics[:output_bytes] / 2.0^20,
            result.diagnostics[:estimated_workspace_bytes] / 2.0^20)
    end
end

benchmark_two_particle()
