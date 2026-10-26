#!/usr/bin/env julia

using Printf
using Phundamental

function benchmark_crossing(norb::Int, nk::Int; repeats::Int=9)
    faxis = FermionicMatsubaraAxis(6.0, [0])
    baxis = BosonicMatsubaraAxis(6.0, [0])
    pairs = [(a, b) for a in 1:norb for b in 1:norb]
    labels = [Symbol("orb", i) for i in 1:norb]
    cube = zeros(ComplexF64, norb^2, norb^2, 1, 1, 1)
    for i in 1:norb^2
        cube[i, i, 1, 1, 1] = 0.1
    end
    local_d = OrbitalPairSusceptibility(faxis, baxis, pairs, labels, cube)
    local_m = OrbitalPairSusceptibility(faxis, baxis, pairs, labels, 0.5 .* cube)
    mappings = [[mod1(ik + offset, nk) for ik in 1:nk] for offset in 0:nk-1]
    density = [OrbitalPairSusceptibility(faxis, baxis, pairs, labels, cube .+ iq * 0.0001) for iq in 1:nk]
    magnetic = [OrbitalPairSusceptibility(faxis, baxis, pairs, labels, 0.5 .* cube .+ iq * 0.0002) for iq in 1:nk]
    channels = OrbitalSpinLadderChannels(local_d, local_m, density, magnetic, mappings; spin_symmetry_certified=true)
    function evaluate()
        output = 0.0
        for ik in 1:nk, ikprime in 1:nk, iq in 1:nk, a in 1:norb, b in 1:norb
            response = orbital_crossed_vertex(channels, iq, ik, ikprime, 1, 1, 1, a, b, a, b)
            output += real(response.density + response.magnetic)
        end
        return output
    end
    evaluate()
    times = Float64[]
    allocations = Int[]
    for _ in 1:repeats
        push!(times, @elapsed evaluate())
        push!(allocations, @allocated evaluate())
    end
    return (minimum(times), minimum(allocations) / 2.0^20, channels.metadata[:retained_vertex_bytes] / 2.0^20)
end

println("orbitals  kpoints  best_seconds  allocated_MiB  retained_MiB")
for norb in (1, 2, 4), nk in (2, 4)
    runtime, allocations, retained = benchmark_crossing(norb, nk)
    @printf("%8d %8d %13.6f %14.4f %13.4f\n", norb, nk, runtime, allocations, retained)
end
