# ED-DMFT

The v0.0.7 embedding workflow implements normal-state, single-site, single-orbital dynamical mean-field theory with a finite exact-diagonalization bath. This page gives the recommended construction, convergence, validation, and parameter-sweep procedure.

## Mathematical loop

For a scalar local self-energy,

$$G_{\mathrm{loc}}(i\omega_n)=\int d\epsilon\,\frac{\rho_0(\epsilon)}{i\omega_n+\mu-\epsilon-\Sigma(i\omega_n)}.$$

The Weiss field is

$$\mathcal G_0^{-1}(i\omega_n)=G_{\mathrm{loc}}^{-1}(i\omega_n)+\Sigma(i\omega_n),$$

and the target impurity hybridization is

$$\Delta(i\omega_n)=i\omega_n+\mu-\epsilon_d-\mathcal G_0^{-1}(i\omega_n).$$

The finite ED bath approximates this target as

$$\Delta_{\mathrm{ED}}(i\omega_n)=\sum_p\frac{|V_p|^2}{i\omega_n+\mu-\epsilon_p}.$$

After the impurity solve,

$$\Sigma_{\mathrm{new}}(i\omega_n)=\mathcal G_0^{-1}(i\omega_n)-G_{\mathrm{imp}}^{-1}(i\omega_n).$$

The fixed point is accepted only after the iterative residual and the independently recomputed closure satisfy the requested tolerance.

## Construct the Matsubara axis

Choose the inverse temperature and number of positive frequencies explicitly:

```julia
beta = 32.0
axis = FermionicMatsubaraAxis(beta, 128)
```

The physical temperature is $T=1/(k_B\beta)$. If `kB=1`, use

```julia
temperature = inv(beta)
```

The number of Matsubara points controls both the self-consistency representation and the bath-fitting objective. Bath quality should therefore be interpreted at the frequency resolution actually used in the run.

## Choose the lattice

For the semicircular Bethe benchmark,

```julia
lattice = BetheLattice(sqrt(2))
```

`BetheLattice(D)` uses `D` as the semicircular half-bandwidth. `DiscreteLattice` can represent a numerical density of states through energies and normalized quadrature weights, and `AtomicLattice` provides the no-hopping validation limit.

## Define the DMFT problem

For a half-filled Hubbard problem with $\epsilon_d=0$ and $\mu=U/2$,

```julia
U = 3.0
problem = DMFTProblem(lattice, U; temperature=temperature, chemical_potential=U/2, impurity_energy=0.0, axis=axis)
```

The problem constructor checks that the explicit axis and requested temperature correspond to the same $\beta$.

## Configure the finite bath

At generic filling, use the unconstrained bath fitter:

```julia
fit_options = BathFitOptions(maxiter=200, tol=1e-10, weight_power=1.0, multistart=3, symmetry=:none)
```

At exact particle--hole symmetry with an even bath, use

```julia
fit_options = BathFitOptions(maxiter=200, tol=1e-10, weight_power=1.0, multistart=3, symmetry=:particle_hole)
```

The constrained parameterization enforces bath energies in $\mu\pm\xi_j$ pairs and equal hybridization magnitudes. The v0.0.7 half-filled validation benchmarks use this mode because it removes an otherwise weak particle--hole-asymmetric direction in the finite-bath fit.

Do not enable the constraint away from a physically particle--hole-symmetric point.

## Configure the ED impurity solver

```julia
impurity_solver = EDImpuritySolver(bath_sites=4, bath_fit=fit_options, backend=:auto, threading=:auto)
```

`backend=:auto` uses the sectorized $(N_\uparrow,N_\downarrow)$ ED implementation for the normal Hermitian Anderson impurity. `backend=:dense` is useful for small-system reference checks.

`threading=:auto` uses Julia threads for independent sector and Lehmann work when multiple threads are available. For controlled parallel runs, keep BLAS threading independent:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. your-script.jl
```

## Choose a mixing policy

Simple problems may use fixed linear mixing:

```julia
mixing = LinearMixing(0.5)
```

For interaction sweeps and strong-coupling points, the validated v0.0.7 benchmarks use

```julia
mixing = AdaptiveLinearMixing()
```

The adaptive method remains a convex linear mixture. It reduces the mixing fraction after worsening steps and recovers it after strong or sustained improvement.

## Configure convergence

```julia
solver = DMFTSolver(impurity_solver=impurity_solver, mixing=mixing, tol=1e-7, maxiter=320, verify_closure=true, convergence_policy=DMFTConvergencePolicy())
```

The `maxiter=320` value above is the conservative benchmark budget used to close the strongest v0.0.7 physical-regime point. It is not the package-wide default; `DMFTSolver()` defaults to 100 iterations.

The independent closure check should normally remain enabled for validation and scientific benchmark runs.

## Solve

```julia
result = solve(solver, problem)
```

Inspect the high-level status:

```julia
result.converged
result.iterations
result.residual
result.metadata[:termination]
result.metadata[:closure_residual]
```

A scientifically accepted run should normally satisfy both `result.converged == true` and `result.metadata[:termination] == :converged`.

## Inspect the history

Every iteration records convergence diagnostics:

```julia
for entry in result.history
    println((
        iteration=entry.iteration,
        sigma_residual=entry.residual,
        weiss_residual=entry.weiss_residual,
        hybridization_residual=entry.hybridization_residual,
        bath_residual=entry.bath_fit_residual,
        alpha=entry.mixing_alpha
    ))
end
```

A `:maxiter` result should not be treated generically as “failed.” The history distinguishes smooth critical slowing, stagnation, oscillation, and bath-fit problems. Likewise, `:closure_failed` specifically means that an iterative criterion was reached but independent closure did not pass before the available budget was exhausted.

## Run the validation bundle

```julia
diagnostics = validate_dmft_result(result)
```

Important fields include:

```text
:termination
:final_iteration_residual
:best_iteration_residual
:closure_residual
:impurity_local_residual
:local_causality_residual
:impurity_causality_residual
:self_energy_causality_residual
:bath_discretization_residual
:bath_particle_hole_residual
:bethe_self_consistency_residual
```

Additional atomic, noninteracting, and particle--hole diagnostics appear when the corresponding physical conditions hold.

## Interpret bath residuals correctly

The bath-discretization residual measures how closely a finite rational function represents the target hybridization over the selected Matsubara grid. It is not the same as the DMFT fixed-point residual.

A run can be self-consistent with a visibly finite bath residual because ED-DMFT is solving the self-consistency problem within a restricted finite-bath manifold. Bath-size convergence therefore needs to be studied separately.

The v0.0.7 physical-regime benchmark compares $N_b=2$ and $N_b=4$ and reports the low-frequency self-energy change as `Sigma_dNb`.

## Parameter continuation

When sweeping interaction or another parameter, seed a nearby problem with the previous converged self-energy:

```julia
result_next = solve(solver, problem_next; initial_self_energy=result.self_energy)
```

If the bath size is unchanged, a compatible previous bath may also be supplied:

```julia
result_next = solve(solver, problem_next; initial_self_energy=result.self_energy, initial_bath=result.impurity_result.bath)
```

Continuation improves the initial condition but does not bypass the final closure verification.

When the bath size changes, reuse the self-energy but let the new bath dimension be fitted appropriately rather than passing an incompatible bath object.

## Bath-size convergence

A minimal finite-bath study should compare at least two bath sizes at identical $\beta$, frequency grid, lattice, interaction, and convergence criterion.

Useful quantities are the direct bath-fit residual, low-frequency self-energy difference, density, double occupancy, causality, particle--hole residual when applicable, and the final independent closure residual.

For half-filled v0.0.7 Bethe tests, the $N_b=4$ bath gives a substantially better representation of the Weiss field than $N_b=2$ at the tested intermediate and strong couplings.

## Quasiparticle-weight diagnostic

```julia
zdiag = quasiparticle_weight_estimate(result.self_energy)
```

The function fits the low-frequency slope of $\operatorname{Im}\Sigma(i\omega_n)$. It is useful for metallic Fermi-liquid comparisons but should not be called a quasiparticle weight once the self-energy is insulating or otherwise outside the Fermi-liquid regime.

## Atomic validation

For `AtomicLattice` at half filling, compare the converged result with

```julia
G_atomic = atomic_hubbard_green(axis, U)
Sigma_atomic = atomic_hubbard_self_energy(axis, U)
```

This is a strong exact test because it checks the thermal Green function and self-energy against closed-form expressions rather than against another numerical DMFT implementation.

## Bethe validation

For a converged Bethe result,

```julia
residual = bethe_self_consistency_residual(problem.lattice, result.local_green_function, result.weiss_green_function; chemical_potential=problem.chemical_potential)
```

This checks the semicircular self-consistency identity independently of the generic Weiss construction.

## Published-reference benchmark

The repository benchmark

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/caffarel-krauth-1994-benchmark.jl
```

implements the finite-temperature ED-DMFT comparison to Caffarel and Krauth (1994) using the package's $D=\sqrt2$ convention. It reports the finite-bath inverse-Weiss distance, direct bath residual, independent closure, low-frequency $Z$ diagnostic, density, and iteration count.

The benchmark intentionally does not treat digitized figure values as exact targets. It validates equations, trends, finite-bath improvement, and self-consistent closure. A CT-QMC comparison remains outside the v0.0.7 implemented validation suite.

## v0.0.7 validated strong-coupling point

The most difficult final physical-regime case was $U=4.8$, $N_b=4$, $\beta=32$, and $D=\sqrt2$. With adaptive mixing, a particle--hole-constrained bath, and the unchanged $10^{-7}$ convergence criterion, it converged in 194 iterations with independent closure residual $2.870\times10^{-8}$ and bath residual $1.179\times10^{-3}$.

This point motivated retaining an iteration budget larger than the observed convergence time in the benchmark while leaving the package-wide default unchanged.

## Scope limits

v0.0.7 does not claim superconducting/Nambu ED-DMFT, multi-orbital DMFT, cluster DMFT, continuous-time QMC impurity solving, analytic continuation, or real-frequency self-consistency. The abstractions were designed so additional impurity solvers and matrix-valued embeddings can be added later without redefining the stable normal-state objects.

## See also

- [Embedding](../modules/embedding.md) for complete type and diagnostic documentation.
- [Greens](../modules/greens.md) for Matsubara, Dyson, Nambu, and causality conventions.
- [Thermodynamics](../modules/thermodynamics.md) for exact finite-temperature impurity states.
- [Validation](../validation.md) for the final v0.0.7 numerical evidence and literature references.
