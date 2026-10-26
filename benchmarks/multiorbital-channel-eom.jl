#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Phundamental

const P = Phundamental

function channel_fixture(norb::Int, nfreq::Int; nk::Int=2)
    U = P.kanamori_interaction(norb; U=1.0, J=0.0)
    nspin = length(U.labels)
    h0 = zeros(ComplexF64, nspin, nspin)
    hopping = 0.08 .* Matrix{ComplexF64}(I, nspin, nspin)
    matrices = nk == 1 ? reshape(h0, nspin, nspin, 1) : cat(h0+hopping, h0-hopping; dims=3)
    maps = nk == 1 ? [[1]] : [[1, 2], [2, 1]]
    lattice = P.MatrixMomentumLattice(matrices; labels=U.labels, qmaps=maps)
    indices = collect(-div(nfreq, 2):div(nfreq-1, 2))
    axis = P.FermionicMatsubaraAxis(4.0, indices)
    problem = P.MultiOrbitalDMFTProblem(lattice, U; temperature=0.25, axis=axis)
    ed = P.MultiOrbitalEDSolver(bath_sites=0, max_full_dimension=65536, max_sector_dimension=65536)
    dmft = P.solve(P.MultiOrbitalDMFTSolver(impurity_solver=ed, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=4), problem)
    dmft.converged || throw(ErrorException("channel benchmark requires converged DMFT"))
    core_baxis = P.BosonicMatsubaraAxis(4.0, [0])
    green = P.expanded_dmft_momentum_green(dmft, core_baxis)
    support_indices = collect(-nfreq:nfreq)
    support_transfers = collect(-nfreq:nfreq)
    fermion_support = P.FermionicMatsubaraAxis(4.0, support_indices)
    boson_support = P.BosonicMatsubaraAxis(4.0, support_transfers)
    pairs = [(a, b) for a in 1:norb for b in 1:norb]
    data = zeros(ComplexF64, norb^2, norb^2, length(fermion_support), length(fermion_support), length(boson_support))
    for iΩ in axes(data, 5), j in axes(data, 2), i in axes(data, 1), nu in axes(data, 4), nup in axes(data, 3)
        data[i, j, nup, nu, iΩ] = (i == j ? 0.012 : 0.001) / (1 + abs(nup-nu) + abs(iΩ - nfreq - 1))
    end
    local_d = P.OrbitalPairSusceptibility(fermion_support, boson_support, pairs, [Symbol("orb", i) for i in 1:norb], data)
    local_m = P.OrbitalPairSusceptibility(fermion_support, boson_support, pairs, local_d.labels, -0.8 .* data)
    qd = [P.OrbitalPairSusceptibility(fermion_support, boson_support, pairs, local_d.labels, (1+0.1iq).*data) for iq in eachindex(maps)]
    qm = [P.OrbitalPairSusceptibility(fermion_support, boson_support, pairs, local_d.labels, (-0.8+0.05iq).*data) for iq in eachindex(maps)]
    channels = P.OrbitalSpinLadderChannels(local_d, local_m, qd, qm, maps; spin_symmetry_certified=true)
    core = P.OrbitalCrossingCore(channels, axis.indices, [0])
    return (; dmft, green, core)
end

println("spatial_orbitals  frequencies  kpoints  best_seconds  allocated_MiB  output_MiB  workspace_MiB")
for norb in (1, 2), nfreq in (2, 4)
    fixture = channel_fixture(norb, nfreq)
    P.orbital_channel_ladder_self_energy(fixture.dmft, fixture.green, fixture.core)
    best = minimum(@elapsed(P.orbital_channel_ladder_self_energy(fixture.dmft, fixture.green, fixture.core)) for _ in 1:5)
    allocated = @allocated P.orbital_channel_ladder_self_energy(fixture.dmft, fixture.green, fixture.core)
    result = P.orbital_channel_ladder_self_energy(fixture.dmft, fixture.green, fixture.core)
    @printf("%16d  %11d  %7d  %12.6f  %13.4f  %10.4f  %13.4f\n", norb, nfreq,
        length(fixture.dmft.problem.lattice.kpoints), best, allocated/2.0^20,
        result.metadata[:estimated_output_bytes]/2.0^20, result.metadata[:estimated_workspace_bytes]/2.0^20)
end
