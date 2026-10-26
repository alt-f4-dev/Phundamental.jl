#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Phundamental

const P = Phundamental

function fixture(norbitals::Int, nfreq::Int; nk::Int=2)
    interaction = P.kanamori_interaction(norbitals; U=1.25, J=0.0)
    n = length(interaction.labels)
    onsite = zeros(ComplexF64, n, n)
    hopping = Matrix{ComplexF64}(I, n, n) .* 0.08
    blocks = cat([onsite .+ (ik == 1 ? hopping : -hopping) for ik in 1:nk]...; dims=3)
    maps = nk == 2 ? [[1, 2], [2, 1]] : [collect(1:nk)]
    lattice = P.MatrixMomentumLattice(blocks; labels=interaction.labels, qmaps=maps)
    axis = P.FermionicMatsubaraAxis(4.0, collect(0:nfreq-1))
    baxis = P.BosonicMatsubaraAxis(4.0, [0])
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis)
    ed = P.MultiOrbitalEDSolver(bath_sites=0, max_full_dimension=65536, max_sector_dimension=65536)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=ed, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=3)
    dmft = P.solve(solver, problem)
    dmft.converged || throw(ErrorException("benchmark requires a converged DMFT reference"))
    green = P.expanded_dmft_momentum_green(dmft, baxis)
    pairs = [(a, b) for a in 1:n for b in 1:n]
    data = zeros(ComplexF64, n^2, n^2, nfreq, nfreq, 1)
    for i in 1:n^2, j in 1:n^2, nu in 1:nfreq, nup in 1:nfreq
        data[i, j, nup, nu, 1] = (i == j ? 0.02 : 0.002) / (1+abs(nup-nu))
    end
    local_vertex = P.OrbitalPairSusceptibility(axis, baxis, pairs, interaction.labels, data)
    vertices = [P.OrbitalPairSusceptibility(axis, baxis, pairs, interaction.labels, (1 + 0.1*iq) .* data) for iq in eachindex(maps)]
    return P.OrbitalDiagrammaticEOMProblem(dmft, green, vertices, local_vertex, baxis)
end

println("spatial_orbitals  frequencies  kpoints  best_seconds  allocated_MiB  output_MiB  workspace_MiB")
for norbitals in (1, 2), nfreq in (2, 4, 8)
    problem = fixture(norbitals, nfreq)
    P.orbital_diagrammatic_self_energy(problem)
    times = Float64[]
    for trial in 1:5
        push!(times, @elapsed P.orbital_diagrammatic_self_energy(problem))
    end
    allocated = @allocated P.orbital_diagrammatic_self_energy(problem)
    result = P.orbital_diagrammatic_self_energy(problem)
    @printf("%16d  %11d  %7d  %12.6f  %13.4f  %10.4f  %13.4f\n", norbitals, nfreq,
        length(problem.qmaps), minimum(times), allocated/2.0^20,
        result.metadata[:estimated_output_bytes]/2.0^20, result.metadata[:estimated_workspace_bytes]/2.0^20)
end
