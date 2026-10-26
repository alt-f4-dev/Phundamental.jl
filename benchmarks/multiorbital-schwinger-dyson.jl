#!/usr/bin/env julia

# Focused kernel-contraction timing and allocations, excluding ED and matrix DMFT solves.
using Phundamental
using Printf
using LinearAlgebra

function benchmark_orbital_contraction(; dimensions=(2, 4), frequencies=(2, 4, 8), samples=5)
    println("orbital_modes  frequency_count  qpoints  best_seconds  allocated_MiB  estimated_workspace_MiB")
    for n in dimensions, nf in frequencies
        labels = [Symbol("mode_", i) for i in 1:n]
        axis = FermionicMatsubaraAxis(4.0, collect(0:nf-1))
        baxis = BosonicMatsubaraAxis(4.0, [0])
        nk = 4
        blocks = cat([Matrix(Diagonal(fill(0.1 * ik, n))) for ik in 1:nk]...; dims=3)
        lattice = MatrixMomentumLattice(blocks; labels=labels, qmaps=periodic_momentum_shift_maps((nk,)).maps)
        interaction = SpinOrbitalInteraction(zeros(ComplexF64, n, n, n, n); labels=labels)
        ed = MultiOrbitalEDSolver(bath_sites=0, max_full_dimension=1 << n,
            max_sector_dimension=1 << n, max_eigensystem_bytes=100_000_000)
        driver = MultiOrbitalDMFTSolver(impurity_solver=ed, mixing=LinearMixing(1.0), maxiter=2, tol=1e-8)
        problem = MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis)
        dmft = solve(driver, problem)
        dmft.converged || error("noninteracting benchmark reference did not converge")
        green = expanded_dmft_momentum_green(dmft, baxis)
        kernel = OrbitalSchwingerDysonKernel((dest, iq, iν, iΩ) -> (dest[1, 1, 1, 1] = 0.03 * iq), labels;
            provenance=:screened_particle_hole)
        calculate() = orbital_schwinger_dyson(dmft, lattice, green, lattice.qmaps, baxis, kernel)
        reference = calculate()
        best_seconds = Inf
        allocated = typemax(Int)
        for _ in 1:samples
            GC.gc()
            best_seconds = min(best_seconds, @elapsed calculate())
            allocated = min(allocated, @allocated calculate())
        end
        @printf("%13d  %15d  %7d  %12.6f  %13.3f  %23.4f\n", n, nf, nk, best_seconds,
            allocated / 2.0^20, reference.metadata[:estimated_workspace_bytes] / 2.0^20)
    end
end

benchmark_orbital_contraction()
