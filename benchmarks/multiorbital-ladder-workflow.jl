#!/usr/bin/env julia

using Printf
using LinearAlgebra
using Phundamental

function benchmark_case(nk::Int, ntime::Int)
    interaction = kanamori_interaction(1; U=0.0, J=0.0)
    onsite = Matrix(Diagonal(ComplexF64[0.1, 0.1]))
    blocks = Array{ComplexF64}(undef, 2, 2, nk)
    for k in 1:nk
        blocks[:, :, k] .= onsite + 0.15cos(2pi * (k-1)/nk) * I
    end
    qmaps = [[mod1(k+q-1, nk) for k in 1:nk] for q in 1:nk]
    lattice = MatrixMomentumLattice(blocks; labels=interaction.labels, kpoints=collect(1:nk), qmaps=qmaps)
    axis = FermionicMatsubaraAxis(4.0, [-1, 0])
    problem = MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis)
    dmft = solve(MultiOrbitalDMFTSolver(impurity_solver=MultiOrbitalEDSolver(bath_sites=0),
        mixing=LinearMixing(1.0), maxiter=8), problem)
    dmft.converged || error("DMFT failed to converge")
    faxis = FermionicMatsubaraAxis(4.0, [-2, -1, 0, 1])
    baxis = BosonicMatsubaraAxis(4.0, [-1, 0, 1])
    input = OrbitalLadderWorkflowProblem(dmft, faxis, baxis; core_bosonic_indices=[0])
    solver = OrbitalLadderWorkflowSolver(four_point_backend=TauGridFourPoint(ntime=ntime, max_dimension=64),
        pseudoinverse=true, max_contractions=100_000_000)
    solve(solver, input)
    samples = Float64[]
    allocations = Int[]
    for _ in 1:3
        started = time_ns()
        result = solve(solver, input)
        push!(samples, (time_ns() - started) * 1e-9)
        push!(allocations, @allocated solve(solver, input))
        result.metadata[:crossing_support_complete] || error("crossing grid became incomplete")
    end
    return minimum(samples), minimum(allocations) / 2.0^20
end

println("kpoints  ntime  best_seconds  allocated_MiB")
for nk in (1, 2), ntime in (6, 8)
    seconds, mib = benchmark_case(nk, ntime)
    @printf("%7d %6d %13.6f %14.3f\n", nk, ntime, seconds, mib)
end
