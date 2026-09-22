#!/usr/bin/env julia

using Printf
using Phundamental
#include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
#using .Phundamental

U = 2.0
problem = DMFTProblem(AtomicLattice(), U; temperature=0.25, chemical_potential=U / 2, nfrequencies=128)
impurity_solver = EDImpuritySolver(bath_sites=0, bath_fit=BathFitOptions(tol=1e-12))
solver = DMFTSolver(impurity_solver=impurity_solver, mixing=LinearMixing(1.0), tol=1e-10, maxiter=5)

result = nothing
elapsed = @elapsed global result = solve(solver, problem)
diagnostics = validate_dmft_result(result)

@printf("Atomic ED-DMFT benchmark\n")
@printf("  elapsed: %.6f s\n", elapsed)
@printf("  converged: %s\n", string(result.converged))
@printf("  iterations: %d\n", result.iterations)
@printf("  fixed-point residual: %.6e\n", result.residual)
@printf("  atomic self-energy residual: %.6e\n", diagnostics[:atomic_self_energy_residual])
