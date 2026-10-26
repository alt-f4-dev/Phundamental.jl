#!/usr/bin/env julia

using Phundamental
using LinearAlgebra
using Printf

interaction = kanamori_interaction(1; U=0.0, J=0.0)
one_body = Matrix(Diagonal(ComplexF64[0.15, 0.15]))
lattice = MatrixMomentumLattice(reshape(one_body, 2, 2, 1);
    labels=interaction.labels, kpoints=[:atomic], qmaps=[[1]])
axis = FermionicMatsubaraAxis(4.0, [-1, 0])
problem = MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis)
dmft = solve(MultiOrbitalDMFTSolver(impurity_solver=MultiOrbitalEDSolver(bath_sites=0),
    mixing=LinearMixing(1.0), maxiter=8), problem)
dmft.converged || error("the benchmark requires converged DMFT")
small = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(4.0, [-2, -1, 0, 1]),
    BosonicMatsubaraAxis(4.0, [-1, 0, 1]); core_bosonic_indices=[0])
large = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(4.0, [-3, -2, -1, 0, 1, 2]),
    BosonicMatsubaraAxis(4.0, [-2, -1, 0, 1, 2]); core_bosonic_indices=[0])
solver = OrbitalLadderWorkflowSolver(four_point_backend=TauGridFourPoint(ntime=12, max_dimension=64),
    pseudoinverse=true, max_contractions=50_000_000)
# Compile the full calculation before collecting per-stage timing and allocation data.
orbital_ladder_convergence_study(solver, small, large; ntime_coarse=12, ntime_fine=16)
GC.gc()
report = orbital_ladder_convergence_study(solver, small, large; ntime_coarse=12, ntime_fine=16)
println("fixture: noninteracting atomic reference (U=0, one momentum point)")
println("measurement: one warmed execution per stage; compilation excluded")
println("stage  ntime  fermionic  bosonic  seconds  allocated_MiB")
for stage in report.stages
    @printf("%-13s %5d %10d %8d %9.6f %14.4f\n", String(stage.name), stage.ntime,
        stage.fermionic_count, stage.bosonic_count, stage.time_seconds, stage.allocated_bytes / 2.0^20)
end
@printf("frequency_residual %.4e\n", report.frequency_comparison.relative_residual)
@printf("quadrature_residual %.4e\n", report.quadrature_comparison.relative_residual)
println("two_particle_tail_certified ", report.metadata[:two_particle_vertex_tail_certified])
println("interacting_convergence_test false")
