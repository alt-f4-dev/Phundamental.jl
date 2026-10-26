#!/usr/bin/env julia

# Optional performance and allocation harness for finite-box multiorbital ladder calculations.
using Phundamental
using Printf

function benchmark_orbital_ladder(; counts=(2, 4), frequency_counts=(2, 4, 8), samples=3)
    println("orbital_modes  frequency_count  qpoints  best_seconds  allocated_MiB  output_MiB  workspace_MiB")
    for modes in counts, frequencies in frequency_counts
        beta = 4.0
        selected = FermionicMatsubaraAxis(beta, collect(-frequencies÷2:frequencies÷2-1))
        bosonic = BosonicMatsubaraAxis(beta, [0, 1])
        extended_indices = sort!(unique(vcat(selected.indices, selected.indices .+ 1)))
        extended = FermionicMatsubaraAxis(beta, extended_indices)
        nk = 4
        cube = zeros(ComplexF64, modes, modes, nk, length(extended))
        for ik in 1:nk, n in 1:length(extended), a in 1:modes, b in 1:modes
            cube[a, b, ik, n] = (a == b ? 1.0 : 0.02im) / (extended[n] + 0.5 + 0.1a + 0.02ik)
        end
        names = [Symbol("orbital_", a) for a in 1:modes]
        green = MatrixMomentumResolvedGreenFunction(extended, collect(1:nk), cube; labels=names)
        maps = periodic_momentum_shift_maps((nk,)).maps
        pairs = [(a, b) for a in 1:modes for b in 1:modes]
        vertex_values = zeros(ComplexF64, length(pairs), length(pairs), length(selected), length(selected), length(bosonic))
        vertex = OrbitalPairIrreducibleVertex(selected, bosonic, pairs, names, vertex_values, Dict{Symbol,Any}())
        weights = ones(ComplexF64, length(pairs))
        calculate() = multiorbital_ladder_susceptibility(green, maps, vertex; projection_weights=weights, return_full=false)
        calculate()
        seconds = Inf
        allocations = typemax(Int)
        for _ in 1:samples
            GC.gc()
            seconds = min(seconds, @elapsed calculate())
            allocations = min(allocations, @allocated calculate())
        end
        result = calculate()
        @printf("%13d  %15d  %7d  %12.6f  %13.3f  %10.4f  %13.4f\n", modes, frequencies, nk,
            seconds, allocations / 2.0^20, result.metadata[:estimated_output_bytes] / 2.0^20,
            result.metadata[:estimated_workspace_bytes] / 2.0^20)
    end
end

benchmark_orbital_ladder()
