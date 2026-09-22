```@meta
CurrentModule = Phundamental.Embedding
```

# Embedding

`Phundamental.Embedding` implements the first self-consistent many-body embedding layer in Phundamental.jl: normal-state, single-site, single-orbital dynamical mean-field theory (DMFT) with a finite exact-diagonalization impurity bath. The implementation is built directly on the typed Green-function objects in [Greens](greens.md), the Hamiltonian models in [Models](models.md), finite-temperature states in [Thermodynamics](thermodynamics.md), and the solver protocol in [Solvers](solvers.md).

The v0.0.7 validated loop is

$$\Sigma(i\omega_n)\rightarrow G_{\mathrm{loc}}(i\omega_n)\rightarrow\mathcal G_0(i\omega_n)\rightarrow\Delta(i\omega_n)\rightarrow\text{finite ED bath}\rightarrow G_{\mathrm{imp}}(i\omega_n)\rightarrow\Sigma_{\mathrm{new}}(i\omega_n).$$

The embedding layer keeps the lattice problem, Weiss field, bath discretization, impurity solve, self-energy mixing, and independent closure verification as separate steps. This makes convergence failures diagnosable rather than reducing the calculation to a single opaque iteration function.

## Scope of v0.0.7

The stable v0.0.7 implementation assumes a scalar local self-energy, one correlated impurity orbital, spin-degenerate paramagnetic normal-state ED, and single-site self-consistency. The Green-function layer is matrix-valued and supports Nambu indexing, but superconducting/Nambu DMFT is not part of the validated embedding scope.

The package does not perform analytic continuation as part of the DMFT fixed-point loop. Matsubara convergence and real-frequency continuation are different numerical problems and remain separated.

## Discrete finite bath

`DiscreteBath` stores physical one-particle bath energies $\epsilon_p$ and impurity hybridizations $V_p$. For chemical potential $\mu$, its hybridization function is

$$\Delta_{\mathrm{ED}}(z)=\sum_p\frac{|V_p|^2}{z+\mu-\epsilon_p}.$$

`hybridization_function` evaluates this expression on a `FermionicMatsubaraAxis` and returns a typed `HybridizationFunction`.

The bath is spin degenerate in the v0.0.7 impurity model. Each stored bath orbital is realized for both spin channels by `AndersonImpurityModel` when the impurity Hamiltonian is built.

## Bath fitting

`BathFitOptions` controls deterministic nonlinear least-squares fitting of a target Matsubara hybridization to a finite bath. The defaults are `maxiter=200`, `tol=1e-10`, `damping=1e-4`, `weight_power=1`, no explicit energy window, no finite hybridization ceiling, `multistart=3`, and `symmetry=:none`.

The objective uses weights proportional to $|\omega_n|^{-p}$, where `p=weight_power`, so low Matsubara frequencies can be emphasized without modifying the target function itself.

`fit_bath(target, nbath; chemical_potential, options, initial_bath)` returns a `BathFitResult` containing the optimized `DiscreteBath`, objective, numerical convergence flag, iteration count, and diagnostics.

The fitter uses bounded damping, rejects nonfinite trial states before entering linear algebra routines, retains the best finite solution, and supports deterministic multistart initialization. In successive DMFT iterations an earlier bath may be supplied as `initial_bath` to warm-start the fit.

### Particle--hole-constrained fitting

At exact half filling, an unconstrained finite bath can drift along a weakly constrained particle--hole-asymmetric direction even when the physical solution is symmetric. v0.0.7 therefore adds `symmetry=:particle_hole`.

For an even bath, the constrained parameterization imposes pairs

$$\epsilon_j^-=\mu-\xi_j,\qquad\epsilon_j^+=\mu+\xi_j,$$

with equal hybridization magnitudes inside each pair. The optimizer works only with the positive offsets $\xi_j$ and one coupling magnitude per pair, then expands them to the full bath.

This both enforces the physical invariant and reduces the nonlinear parameter count. Odd bath sizes are rejected in particle--hole-symmetric mode.

The unconstrained fitter remains the generic default. Symmetry constraints should be enabled because the physical model warrants them, not merely because a difficult fit is encountered.

## Typed impurity problems

`ImpurityProblem` packages the interaction $U$, temperature axis, chemical potential, impurity energy, and either a target hybridization or an already specified discrete bath.

`AbstractImpuritySolver` is the impurity-specific solver abstraction. `ImpuritySolverResult` stores the constructed impurity model, interacting Green function, noninteracting/Weiss Green function, self-energy, density, double occupancy, fitted bath, thermal state, and metadata.

## ED impurity solver

`EDImpuritySolver` fits a finite bath when necessary, builds an `AndersonImpurityModel`, solves it at finite temperature, evaluates the exact Lehmann Green function, constructs the impurity $G_0$, and obtains the self-energy through Dyson's equation.

The constructor options include the number of bath sites, `BathFitOptions`, the dense ED configuration, $k_B$, numerical backend, and threading mode.

### Dense and sectorized backends

`backend=:dense` realizes the complete finite impurity Hilbert space and is retained as the independent small-system correctness reference.

`backend=:sectorized` exploits exact conservation of $(N_\uparrow,N_\downarrow)$ for the normal Anderson impurity. Instead of diagonalizing the full dimension $4^{N_b+1}$, it diagonalizes independent particle-number blocks. For four bath sites the full Hilbert space has dimension 1024 while the largest symmetry block has dimension 100.

`backend=:auto` selects the sectorized path for the normal Hermitian impurity model and retains the dense path when the sectorized assumptions are not satisfied.

### Julia-thread parallelism

`threading=:serial` preserves single-threaded deterministic execution. `:threads` and `:auto` can parallelize independent sector diagonalizations and independent Lehmann transition blocks when Julia has multiple threads.

The implementation stores independent task results in canonical slots and performs final reductions in a deterministic order. It does not modify global BLAS threading. For controlled benchmarks, run Julia threads with BLAS restricted separately, for example

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/ed-dmft-scaling-benchmark.jl
```

The v0.0.7 scaling campaign verified serial/threaded equivalence and observed substantial speedup for the larger finite baths used by DMFT.

## Lattice embeddings

`AbstractLatticeEmbedding` is the common lattice abstraction used by single-site self-consistency.

`AtomicLattice` contains one local orbital and no hopping. It provides the exact atomic limit used for validation.

`DiscreteLattice` stores a finite set of one-particle energies and normalized quadrature weights and evaluates

$$G_{\mathrm{loc}}(z)=\sum_k\frac{w_k}{z+\mu-\epsilon_k-\Sigma(z)}.$$

`BetheLattice` represents the infinite-coordination semicircular density of states through its half-bandwidth $D$. The local Green function uses the analytically stable rationalized form of the Hilbert transform, avoiding catastrophic cancellation at large Matsubara frequency.

`lattice_green` dispatches on the lattice representation and returns a scalar `GreenFunction` compatible with the current local self-energy.

## Weiss field and hybridization

The DMFT Weiss propagator is reconstructed from

$$\mathcal G_0^{-1}(i\omega_n)=G_{\mathrm{loc}}^{-1}(i\omega_n)+\Sigma(i\omega_n).$$

`weiss_green` evaluates this relation using matrix solves and returns a `NonInteractingGreenFunction`.

For the scalar impurity used in v0.0.7,

$$\Delta(z)=z+\mu-\epsilon_d-\mathcal G_0(z)^{-1}.$$

`hybridization_from_weiss` evaluates the corresponding target `HybridizationFunction` that is fitted by the finite bath.

## Self-energy mixing

`LinearMixing(alpha)` implements

$$\Sigma\leftarrow(1-\alpha)\Sigma_{\mathrm{old}}+\alpha\Sigma_{\mathrm{new}}.$$

`AdaptiveLinearMixing` retains this convex update but changes $\alpha$ according to the observed residual trajectory. The defaults are `initial_alpha=0.5`, `min_alpha=0.05`, `max_alpha=0.8`, `decrease_factor=0.5`, `increase_factor=1.1`, `improvement_ratio=0.8`, `worsening_ratio=1.05`, and `recovery_window=3`.

A worsening step reduces the mixing fraction. A sufficiently strong improvement increases it. Sustained monotonic improvement over `recovery_window` also increases the fraction, preventing the solver from becoming permanently trapped at a very small damping value after an early difficult step.

`self_energy_residual` reports the normalized maximum difference between two compatible self-energies.

## DMFT problem

`DMFTProblem` defines a normal-state single-site, single-orbital calculation. It stores the lattice, interaction, temperature, chemical potential, impurity energy, $k_B$, and a compatible fermionic Matsubara axis.

If an axis is not supplied, the constructor builds one from the requested temperature and `nfrequencies`. If an explicit axis is supplied, its $\beta$ must agree with the requested temperature and $k_B$.

A half-filled Hubbard benchmark commonly uses $\mu=U/2$ with zero impurity energy.

## DMFT solver

`DMFTSolver` combines an impurity solver, mixing policy, convergence tolerance, iteration budget, optional requirement that every bath fit report convergence, independent closure verification, and a `DMFTConvergencePolicy`.

The fixed-point iteration performs the following steps:

1. compute the local lattice Green function from the current self-energy;
2. construct the Weiss field;
3. recover the target hybridization;
4. fit or warm-start the finite ED bath;
5. solve the interacting impurity;
6. compute the candidate self-energy;
7. record residuals and update the mixing state;
8. mix and continue until the convergence criterion is eligible for independent verification.

`DMFTIteration` stores the self-energy residual, density, double occupancy, bath-fit objective and convergence flag, Weiss residual, hybridization residual, bath-discretization residual, and active mixing fraction for each iteration.

`DMFTResult` stores the final self-energy, local Green function, Weiss field, hybridization, impurity result, Boolean convergence state, iteration count, residual, full iteration history, and metadata.

## Convergence and independent closure

A small iterative update is necessary but not sufficient for a validated result. With `verify_closure=true`, the solver performs an independent impurity solve at the same accepted self-energy and checks that the recomputed map closes within tolerance.

A key v0.0.7 rule is that the self-energy that satisfied the iterative criterion is the point tested by the independent closure calculation. The solver does not first replace it by the next candidate and then test a different point.

If an iterative residual falls below tolerance but the independent closure residual remains above tolerance and iteration budget remains, the run resumes from the accepted state and retries closure when a later iterate again becomes eligible. `:closure_failed` is reserved for a run that exhausts its budget after entering this closure-retry sequence.

Metadata distinguishes `:final_iteration_residual`, `:best_iteration_residual`, `:closure_residual`, the iteration of the best residual, closure-attempt count, and whether closure retry was started. These quantities should not be collapsed into one ambiguous residual column in scientific reporting.

## Termination classification

`DMFTConvergencePolicy` classifies a finite run that does not converge before exhausting its budget. The stable termination symbols include:

- `:converged` for independently verified closure within tolerance;
- `:maxiter` for an otherwise finite trajectory that simply exhausts the budget;
- `:stagnated` for a residual history consistent with a numerical plateau under the configured criterion;
- `:oscillatory` for a history dominated by alternating/worsening behavior under the configured criterion;
- `:diverged` for nonfinite or strongly divergent behavior;
- `:closure_failed` when an iterative criterion was reached but repeated independent closure checks did not pass before the budget was exhausted.

The policy is diagnostic. It does not silently relax the requested numerical tolerance.

## Continuation and warm starts

`solve(DMFTSolver, problem; initial_self_energy=..., initial_bath=...)` can reuse a previous converged solution as a starting point. Successive impurity solves within one run also warm-start their finite-bath optimization from the previous bath.

Continuation is especially useful for parameter sweeps. It improves initial conditions without changing the final closure requirement, so a continued calculation is still independently verified at its own target parameters.

## Validation diagnostics

`noninteracting_self_energy_residual` checks that a nominally noninteracting impurity has negligible self-energy. `impurity_local_residual` compares converged impurity and lattice Green functions.

The exact half-filled atomic Hubbard references are

$$G_{\mathrm{atom}}(i\omega)=\frac{i\omega}{(i\omega)^2-(U/2)^2},$$

$$\Sigma_{\mathrm{atom}}(i\omega)=\frac{U}{2}+\frac{U^2}{4i\omega}.$$

`atomic_hubbard_green` and `atomic_hubbard_self_energy` provide these references.

`bath_discretization_residual` measures the relative $L^2$ discrepancy between a target hybridization and the fitted finite bath. `bath_particle_hole_residual` measures energy/coupling pairing about the chemical potential.

`low_frequency_self_energy_residual` compares the lowest positive Matsubara frequencies of two self-energies, useful for finite-bath convergence studies.

For a Bethe lattice with semicircular half-bandwidth $D$ and $t=D/2$, `bethe_self_consistency_residual` checks

$$\mathcal G_0^{-1}(i\omega)=i\omega+\mu-t^2G(i\omega).$$

`quasiparticle_weight_estimate` fits the low-frequency Fermi-liquid form

$$\operatorname{Im}\Sigma(i\omega)\approx(1-1/Z)\omega.$$

!!! warning
    The reported low-frequency $Z$ estimate is meaningful as a quasiparticle weight only in a Fermi-liquid regime. Near an insulating or non-Fermi-liquid solution it is merely a low-frequency slope diagnostic.

`validate_dmft_result` collects convergence history, closure, causality, bath quality, particle--hole symmetry, Bethe self-consistency, noninteracting limits, and atomic-limit comparisons when those checks are applicable.

## v0.0.7 physical-regime benchmark

The final v0.0.7 half-filled Bethe benchmark uses $\beta=32$, $D=\sqrt2$, 128 positive Matsubara frequencies, adaptive mixing, particle--hole-constrained bath fitting, and a benchmark-only iteration ceiling of 320. The general `DMFTSolver` default iteration budget remains 100.

All tested points converged under an independent $10^{-7}$ closure criterion:

| $U$ | $N_b$ | Closure residual | Bath residual | $Z$ diagnostic | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2 | $1.067\times10^{-14}$ | $2.694\times10^{-1}$ | 1.00000 | 1 |
| 0.0 | 4 | $1.425\times10^{-14}$ | $1.146\times10^{-2}$ | 1.00000 | 1 |
| 2.0 | 2 | $3.931\times10^{-8}$ | $3.078\times10^{-1}$ | 0.64843 | 16 |
| 2.0 | 4 | $3.698\times10^{-8}$ | $3.142\times10^{-2}$ | 0.66306 | 16 |
| 3.0 | 2 | $8.602\times10^{-8}$ | $4.236\times10^{-1}$ | 0.35118 | 30 |
| 3.0 | 4 | $9.188\times10^{-8}$ | $3.027\times10^{-2}$ | 0.36096 | 31 |
| 4.8 | 2 | $7.377\times10^{-8}$ | $3.743\times10^{-2}$ | 0.04521 | 76 |
| 4.8 | 4 | $2.870\times10^{-8}$ | $1.179\times10^{-3}$ | 0.04601 | 194 |

The final $U=4.8$, $N_b=4$ point is important because it was the slowest strong-coupling case. Exact particle--hole bath pairing removed an unphysical asymmetric drift, after which the trajectory converged normally under the unchanged $10^{-7}$ closure criterion.

## Caffarel--Krauth benchmark

The literature benchmark follows M. Caffarel and W. Krauth, *Phys. Rev. Lett.* **72**, 1545 (1994), DOI: 10.1103/PhysRevLett.72.1545, with the formal DMFT context of A. Georges *et al.*, *Rev. Mod. Phys.* **68**, 13 (1996), DOI: 10.1103/RevModPhys.68.13, and finite-temperature ED context from A. Liebsch and H. Ishida, *J. Phys.: Condens. Matter* **24**, 053201 (2012), DOI: 10.1088/0953-8984/24/5/053201.

For $\beta=32$, $U=3$, $\mu=1.5$, $D=\sqrt2$, and 128 Matsubara frequencies, both finite baths converge under the particle--hole constraint:

| Impurity sites $n_s$ | Bath sites $N_b$ | $\chi^2$ | Bath residual | Closure residual | $Z$ diagnostic | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | $3.7796\times10^{-5}$ | $8.0706\times10^{-2}$ | $9.8578\times10^{-8}$ | 0.17690 | 32 |
| 5 | 4 | $3.5946\times10^{-6}$ | $1.9529\times10^{-2}$ | $9.5613\times10^{-8}$ | 0.36054 | 45 |

The larger bath improves both the inverse-Weiss fitting distance and the direct bath residual. The benchmark source intentionally does not treat visually digitized figure points as exact targets.

A CT-QMC comparison, discussed in the finite-temperature ED literature, is not implemented in v0.0.7 and remains an external validation target rather than a claimed package result.

## Related modules

- [Greens](greens.md) defines every frequency-dependent object used by the self-consistency loop.
- [Models](models.md) supplies `AndersonImpurityModel`.
- [Thermodynamics](thermodynamics.md) supplies exact finite-temperature impurity states.
- [Solvers](solvers.md) supplies dense reference ED and the common solver protocol.
- [ED-DMFT Workflow](../workflows/ed-dmft.md) gives an end-to-end construction and interpretation guide.
- [Validation](../validation.md) places the DMFT benchmarks in the package-wide validation hierarchy.

## API reference

### Bath discretization and fitting

```@docs
DiscreteBath
hybridization_function
BathFitOptions
BathFitResult
fit_bath
```

### Impurity problems and solvers

```@docs
ImpurityProblem
ImpuritySolverResult
EDImpuritySolver
```

### Lattices and self-consistency

```@docs
AtomicLattice
DiscreteLattice
BetheLattice
weiss_green
hybridization_from_weiss
```

### Mixing and DMFT

```@docs
LinearMixing
AdaptiveLinearMixing
DMFTProblem
DMFTConvergencePolicy
DMFTIteration
DMFTSolver
DMFTResult
```

### Validation

```@docs
noninteracting_self_energy_residual
impurity_local_residual
atomic_hubbard_green
atomic_hubbard_self_energy
bath_discretization_residual
bath_particle_hole_residual
low_frequency_self_energy_residual
bethe_self_consistency_residual
quasiparticle_weight_estimate
validate_dmft_result
```

## Additional exported embedding API

The remaining exports are the abstract embedding/lattice/impurity-solver types, `lattice_green`, `mix_self_energy`, and `self_energy_residual`.
