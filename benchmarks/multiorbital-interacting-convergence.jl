#!/usr/bin/env julia

using Phundamental
using LinearAlgebra
using Printf

# A small interacting, dispersive stress test; not a converged continuous-bath DMFT reference.
# The zero-bath impurity ED is exact for its finite Hamiltonian, but cannot fit a
# nonzero target hybridization. Record the bath error rather than hiding it.
beta = 4.0
U = 0.7
interaction = kanamori_interaction(1; U=U, J=0.0)
hamiltonians = zeros(ComplexF64, 2, 2, 2)
hamiltonians[:, :, 1] .= Diagonal(ComplexF64[-0.20, -0.20])
hamiltonians[:, :, 2] .= Diagonal(ComplexF64[0.20, 0.20])
lattice = MatrixMomentumLattice(hamiltonians; labels=interaction.labels,
    kpoints=[:k0, :kpi], qmaps=[[1, 2], [2, 1]])
axis = FermionicMatsubaraAxis(beta, [-1, 0])
problem = MultiOrbitalDMFTProblem(lattice, interaction; temperature=inv(beta), axis=axis, chemical_potential=0.1)
dmft_solver = MultiOrbitalDMFTSolver(impurity_solver=MultiOrbitalEDSolver(bath_sites=0),
    mixing=LinearMixing(1.0), maxiter=8)
solve(dmft_solver, problem)
measured_dmft = @timed solve(dmft_solver, problem)
dmft = measured_dmft.value
dmft.converged || error("finite-bath reference failed to reach a self-energy fixed point")
validate_dmft_result(dmft).valid || error("finite-bath reference fails matrix-DMFT identities")
small = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(beta, -2:1),
    BosonicMatsubaraAxis(beta, -1:1); core_bosonic_indices=[0])
large = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(beta, -3:2),
    BosonicMatsubaraAxis(beta, -2:2); core_bosonic_indices=[0])
solver = OrbitalLadderWorkflowSolver(four_point_backend=TauGridFourPoint(ntime=12, max_dimension=64),
    pseudoinverse=true, max_contractions=100_000_000)
# Warm the entire compilation path once, then measure one complete four-stage study.
orbital_ladder_convergence_study(solver, small, large; ntime_coarse=12, ntime_fine=16)
report = orbital_ladder_convergence_study(solver, small, large; ntime_coarse=12, ntime_fine=16)
println("fixture: interacting two-momentum Hubbard model, U=$(U), bath_sites=0")
@printf("dmft_seconds %.6f\n", measured_dmft.time)
@printf("dmft_allocated_MiB %.4f\n", measured_dmft.bytes / 2.0^20)
@printf("finite_bath_fit_absolute_residual %.5e\n", dmft.metadata[:bath_fit_absolute_residual])
println("stage  ntime  fermionic  bosonic  seconds  allocated_MiB")
for stage in report.stages
    @printf("%-13s %5d %10d %8d %9.6f %14.4f\n", String(stage.name), stage.ntime,
        stage.fermionic_count, stage.bosonic_count, stage.time_seconds, stage.allocated_bytes / 2.0^20)
end
@printf("frequency_residual %.6e\n", report.frequency_comparison.relative_residual)
@printf("quadrature_residual %.6e\n", report.quadrature_comparison.relative_residual)
@printf("coupled_residual %.6e\n", report.coupled_difference.relative_residual)
@printf("maximum_nonlocal_self_energy %.6e\n", maximum(abs, report.final_result.nonlocal_self_energy.values))
@printf("end_to_end_seconds %.6f\n", measured_dmft.time + report.metadata[:total_time_seconds])
@printf("end_to_end_allocated_MiB %.4f\n", (measured_dmft.bytes + report.metadata[:total_allocated_bytes]) / 2.0^20)
println("interacting_convergence_test_completed true")
println("finite_box_tolerance_met ", report.metadata[:finite_box_numerically_converged])
println("two_particle_tail_certified ", report.metadata[:two_particle_vertex_tail_certified])
println("full_multiorbital_dga ", report.metadata[:full_multiorbital_dga])
