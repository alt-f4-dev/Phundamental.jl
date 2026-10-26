#!/usr/bin/env julia

using Printf
using Phundamental

function projection_benchmark(norb::Int, nf::Int; repeats::Int=9)
    axis = FermionicMatsubaraAxis(6.0, collect(0:nf-1))
    bosons = BosonicMatsubaraAxis(6.0, [0])
    nspin = 2norb
    pairs = [(a, b) for a in 1:nspin for b in 1:nspin]
    names = [Symbol("mode", i) for i in 1:nspin]
    values = zeros(ComplexF64, nspin^2, nspin^2, nf, nf, 1)
    for d in 1:norb, c in 1:norb, b in 1:norb, a in 1:norb
        uu = (2a - 2) * nspin + 2b - 1
        dd = (2a - 1) * nspin + 2b
        vv = (2c - 2) * nspin + 2d - 1
        ww = (2c - 1) * nspin + 2d
        values[uu, vv, :, :, :] .= 0.1
        values[dd, ww, :, :, :] .= 0.1
        values[uu, ww, :, :, :] .= 0.02
        values[dd, vv, :, :, :] .= 0.02
    end
    source = OrbitalPairSusceptibility(axis, bosons, pairs, names, values)
    evaluate() = orbital_impurity_spin_channels(source)
    evaluate()
    runtimes = Float64[]
    allocations = Int[]
    for _ in 1:repeats
        push!(runtimes, @elapsed evaluate())
        push!(allocations, @allocated evaluate())
    end
    outputs = 2 * 16 * norb^4 * nf^2 / 2.0^20
    return minimum(runtimes), minimum(allocations) / 2.0^20, outputs
end

println("spatial_orbitals  frequencies  best_seconds  allocated_MiB  output_MiB")
for norb in (1, 2, 3), nf in (2, 4, 8)
    duration, allocations, output = projection_benchmark(norb, nf)
    @printf("%16d %12d %13.6f %14.4f %11.4f\n", norb, nf, duration, allocations, output)
end
