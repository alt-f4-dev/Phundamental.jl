# Validation

Phundamental v0.0.7 treats validation as a hierarchy rather than a single test-suite result: symbolic invariants are checked first, numerical realizations are compared against independent reference paths, truncations and finite-size approximations are tested for convergence, and selected physical calculations are compared with analytic limits or published benchmarks.

## Validation hierarchy

The validation strategy proceeds from the most implementation-local checks to the most physics-specific checks so that a failed scientific benchmark can be traced to a narrower layer rather than treated as an undifferentiated discrepancy.

1. **Algebraic and representation invariants.** Operator relations, state-space constraints, representation metadata, transformation certificates, and crystallographic symmetry actions are checked before matrix realization.
2. **Independent numerical realizations.** Generic dense or symbolic paths are retained as small-system references for optimized packed, sectorized, matrix-free, or threaded backends.
3. **Truncation and finite-size convergence.** Bosonic cutoffs, cluster sizes, finite baths, and stochastic estimates are varied explicitly rather than being hidden in solver defaults.
4. **Thermodynamic and spectral identities.** Sum rules, detailed balance, causality, high-frequency moments, exact thermal limits, and reconstruction identities are checked where the corresponding assumptions apply.
5. **Published physical benchmarks.** Selected model, solver, observable, and embedding workflows are compared with analytic formulas or published calculations without treating values digitized from plots as exact targets.
6. **Performance validation.** An optimized path is accepted only after agreement with an independent numerical reference has been established; speedup is not used as evidence of correctness.

## v0.0.7 regression gate

The frozen v0.0.7 development state reported **864 passing regression tests with no failures or errors**. The regression suite spans the generalized API, crystallographic invariant generation, transformation numerics, finite-temperature Green functions, ED-DMFT, optimized fermionic impurity sectors, particle-hole-constrained bath fitting, thermodynamics, observables, and bosonic truncation convergence.

Run the complete regression gate from the repository root with:

```bash
julia --project=. test/runtests.jl
```

The test suite is intentionally distinct from the larger scientific benchmark scripts. A unit or regression test answers whether a documented invariant still holds; a scientific benchmark records how a controlled approximation or physical calculation behaves over a parameter range.

## Representation and transformation numerics

The transformation-validation suite compares source Hamiltonians, transformed Hamiltonians, target-space realizations, and spectra for the exact transformation families implemented by `Transformations`. Exact fermionic mappings are checked to machine precision on finite systems, while bosonic mappings additionally report target-basis leakage when a finite numerical cutoff is present.

The validation explicitly distinguishes physical equivalence from finite-basis artifacts. A nonzero leakage diagnostic for a bosonic target representation is therefore not automatically a transformation failure: the transformed low-energy spectrum and observables must be examined as the target cutoff increases.

The empirical benchmark suite also contains a Pfeuty transverse-field Ising-chain transformation map, exercising the sequence spin $\rightarrow$ Jordan--Wigner $\rightarrow$ Fourier $\rightarrow$ Bogoliubov against the known finite-chain solution. The external reference is P. Pfeuty, *Annals of Physics* **57**, 79--90 (1970), DOI: [10.1016/0003-4916(70)90270-8](https://doi.org/10.1016/0003-4916(70)90270-8).

## Bosonic truncation convergence

The regression suite includes an explicit truncation-convergence campaign for Fourier bosons, bosonic Bogoliubov transformations, displacement transformations, Lang--Firsov transformations, spin-polaron transformations, and the coordinate--ladder control problem. The source-space error, transformed low-energy error, observable error, pipeline Hamiltonian/spectral error, and target-basis leakage are tracked independently as the cutoff increases.

The important interpretation is that a numerical cutoff is a property of the realization, not a physical constraint. Validation therefore asks whether the low-energy quantities of interest converge with cutoff even when the norm of the state outside a fixed target basis is not itself small.

## Optimized solver equivalence

Optimized exact backends are validated against the generic reference machinery before they are used for scaling studies. The solver tests cover packed spin and fermion bases, rank lookup, spin-sector decomposition, optimized-versus-generic matrix realization, and Krylov interfaces.

The v0.0.7 Anderson impurity implementation retains both a dense ED reference and a spin-resolved sectorized backend. The sectorized backend decomposes the normal single-orbital impurity problem by conserved $(N_\uparrow,N_\downarrow)$ sectors and compares spectra, finite-temperature observables, $G(i\omega_n)$, and $\Sigma(i\omega_n)$ against the dense reference on systems where both are practical.

Threaded impurity ED is separately compared with the serial sectorized path. With BLAS restricted to one thread, the frozen scaling data show that explicit Julia threading becomes useful once the impurity sectors are nontrivial; for example, the $N_b=4$ sectorized impurity solve decreased from approximately $0.2065$ s to $0.0357$ s on the recorded 32-thread run, a speedup of approximately $5.79\times$. The corresponding $N_b=4$ complete DMFT-driver timing decreased from approximately $0.509$ s to $0.110$ s, or approximately $4.62\times$.

Use the scaling benchmark to characterize a particular machine rather than assuming these timing ratios are portable:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/ed-dmft-scaling-benchmark.jl
```

## Thermodynamics

`Thermodynamics` is validated by comparing exact Gibbs calculations with direct spectral formulas on small systems, reconstructing thermodynamics from symmetry sectors, and checking linked-cluster closure where that expansion is used. Stochastic thermal methods are compared against exact finite-system results before they are used in regimes where exact dense thermodynamics is no longer practical.

The empirical benchmark suite contains the canonical-TPQ workflow for the spin-$1/2$ kagome Heisenberg antiferromagnet and compares small-system cTPQ thermodynamics with `ExactGibbs`, while also exercising sector-decomposed TPQ architecture for larger systems. The external reference is S. Sugiura and A. Shimizu, *Physical Review Letters* **111**, 010401 (2013), DOI: [10.1103/PhysRevLett.111.010401](https://doi.org/10.1103/PhysRevLett.111.010401).

## Observables and scattering

Observable validation separates intrinsic correlation functions from probe contraction and instrumental resolution. This permits spectral sum rules, detailed balance, tensor symmetries, and Fourier conventions to be checked before a probe-specific cross section is applied.

The benchmark suite includes a Bonner--Fisher finite-chain magnetic calculation, an Rb$_2$MnF$_4$ linear-spin-wave/neutron-tensor benchmark, a Bogoliubov BEC response benchmark, a graphene harmonic-phonon benchmark, a controlled coherent one-phonon neutron formalism benchmark, a Dy$_2$Ti$_2$O$_7$ spin-ice reproduction script, and an integrated Ice Ih harmonic/multiphonon workflow. These scripts exercise combinations of `Models`, `Solvers`, `Thermodynamics`, and `Observables`; paper-specific constructions that are not generic package abstractions remain in the benchmark scripts rather than being promoted into public API.

Representative external references used by those scripts include T. Huberman *et al.*, *Physical Review B* **72**, 014413 (2005), DOI: [10.1103/PhysRevB.72.014413](https://doi.org/10.1103/PhysRevB.72.014413); J. Steinhauer *et al.*, *Physical Review Letters* **88**, 120407 (2002), DOI: [10.1103/PhysRevLett.88.120407](https://doi.org/10.1103/PhysRevLett.88.120407); L. A. Falkovsky, *JETP* **105**, 397--403 (2007), DOI: [10.1134/S1063776107080122](https://doi.org/10.1134/S1063776107080122); R. Fair *et al.*, *Journal of Applied Crystallography* **55**, 1689--1703 (2022), DOI: [10.1107/S1600576722009256](https://doi.org/10.1107/S1600576722009256); and D. J. P. Morris *et al.*, *Science* **326**, 411--414 (2009), DOI: [10.1126/science.1178868](https://doi.org/10.1126/science.1178868).

## Finite-temperature Green functions

The `Greens` validation suite checks the noninteracting limit, the interacting atomic Hubbard limit, high-frequency moment sum rules, Matsubara conjugation, particle-hole symmetry, causality, and the Nambu block identities implemented by the current normal/anomalous propagator abstraction.

For a scalar normal-state fermionic Green function, causality on positive Matsubara frequencies requires the sign of the imaginary part to be consistent with the spectral representation. The validation helpers report a residual rather than silently projecting a noncausal object back into the admissible set.

The atomic Hubbard test is particularly useful because both $G(i\omega_n)$ and $\Sigma(i\omega_n)$ are known independently of the lattice embedding. It therefore validates the finite-temperature Lehmann implementation and the Dyson relation before DMFT self-consistency is introduced.

## ED-DMFT validation

The v0.0.7 embedding milestone is normal-state, single-site, single-orbital ED-DMFT. Its validation campaign combines exact limiting cases, finite-bath diagnostics, independent impurity backends, deterministic convergence histories, an independent closure solve, particle-hole-constrained bath fitting at half filling, and published finite-temperature ED-DMFT references.

The fixed-point driver distinguishes the iteration residual from the independent closure residual. Crossing the iteration tolerance triggers closure verification at the same accepted self-energy; if closure does not yet pass and iteration budget remains, the trajectory continues and closure is retried. Exhausted trajectories are classified separately from converged trajectories, and the complete iteration history retains self-energy, Weiss, hybridization, bath-fit, and mixing diagnostics.

### Half-filled Bethe physical-regime sweep

The frozen physical-regime benchmark uses $\beta=32$, half bandwidth $D=\sqrt{2}$, 128 positive Matsubara frequencies, adaptive linear mixing, particle-hole-symmetric bath fitting, and a closure tolerance of $10^{-7}$. Every reported v0.0.7 point converges.

| $U$ | $N_b$ | Closure residual | Bath residual | $\Delta\Sigma_{N_b}$ | $Z_{\mathrm{est}}$ | Density | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2 | $1.07\times10^{-14}$ | $2.69\times10^{-1}$ | -- | 1.00000 | 1.000000 | 1 |
| 0.0 | 4 | $1.42\times10^{-14}$ | $1.15\times10^{-2}$ | $0$ | 1.00000 | 1.000000 | 1 |
| 2.0 | 2 | $3.93\times10^{-8}$ | $3.08\times10^{-1}$ | -- | 0.64843 | 1.000000 | 16 |
| 2.0 | 4 | $3.70\times10^{-8}$ | $3.14\times10^{-2}$ | $3.20\times10^{-2}$ | 0.66306 | 1.000000 | 16 |
| 3.0 | 2 | $8.60\times10^{-8}$ | $4.24\times10^{-1}$ | -- | 0.35118 | 1.000000 | 30 |
| 3.0 | 4 | $9.19\times10^{-8}$ | $3.03\times10^{-2}$ | $5.50\times10^{-2}$ | 0.36096 | 1.000000 | 31 |
| 4.8 | 2 | $7.38\times10^{-8}$ | $3.74\times10^{-2}$ | -- | 0.04521 | 1.000000 | 76 |
| 4.8 | 4 | $2.87\times10^{-8}$ | $1.18\times10^{-3}$ | $2.38\times10^{-2}$ | 0.04601 | 1.000000 | 194 |

`$Z_{\mathrm{est}}$` is the low-frequency Fermi-liquid diagnostic obtained from the Matsubara self-energy. It must not be interpreted as a quasiparticle weight when the converged self-energy is insulating or otherwise non-Fermi-liquid.

The particle-hole-constrained fit removes the weak asymmetric bath direction that appeared in the unconstrained strong-coupling calculation. In the constrained benchmark the bath particle-hole residual is at floating-point precision, density remains at half filling, and the difficult $U=4.8$, $N_b=4$ trajectory closes at $2.87\times10^{-8}$ rather than being accepted on an iteration residual alone.

Run the sweep with:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/ed-dmft-physical-regimes-benchmark.jl
```

The script writes summary, per-iteration, and Matsubara-resolved CSV files under `benchmarks/dmft/results/`.

### Caffarel--Krauth finite-temperature benchmark

The published ED-DMFT benchmark follows M. Caffarel and W. Krauth, *Physical Review Letters* **72**, 1545--1548 (1994), DOI: [10.1103/PhysRevLett.72.1545](https://doi.org/10.1103/PhysRevLett.72.1545), with the general DMFT framework cross-checked against A. Georges *et al.*, *Reviews of Modern Physics* **68**, 13--125 (1996), DOI: [10.1103/RevModPhys.68.13](https://doi.org/10.1103/RevModPhys.68.13), and finite-temperature bath-size practice compared with A. Liebsch and H. Ishida, *Journal of Physics: Condensed Matter* **24**, 053201 (2012), DOI: [10.1088/0953-8984/24/5/053201](https://doi.org/10.1088/0953-8984/24/5/053201).

For $\beta=32$, $U=3$, $\mu=1.5$, $D=\sqrt{2}$, and 128 positive Matsubara frequencies, both finite baths converge under the v0.0.7 particle-hole constraint:

| $n_s$ | $N_b$ | $\chi^2$ | Bath residual | Closure residual | $Z_{\mathrm{est}}$ | Density | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | $3.78\times10^{-5}$ | $8.07\times10^{-2}$ | $9.86\times10^{-8}$ | 0.17690 | 1.000000 | 32 |
| 5 | 4 | $3.59\times10^{-6}$ | $1.95\times10^{-2}$ | $9.56\times10^{-8}$ | 0.36054 | 1.000000 | 45 |

The benchmark implements the Caffarel--Krauth inverse-Weiss-field distance and verifies the Bethe normalization in the package convention. It deliberately does not assign exact numerical targets to curves that were only published graphically.

Run it with:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/caffarel-krauth-1994-benchmark.jl
```

Liebsch and Ishida also use comparison with continuous-time quantum Monte Carlo as an independent-solver accuracy criterion. Phundamental v0.0.7 does not contain a CT-QMC impurity solver, so that comparison remains an external validation target rather than being replaced by a weaker internal check.

## Additional empirical benchmark inventory

The repository contains larger empirical and literature-oriented scripts in `benchmarks/` in addition to the DMFT suite. They include `empirical-benchmark-suite.jl`, `huberman-native-lswt-validation.jl`, `senechal-hubbard-cpt-benchmark.jl`, `ice-ih-integrated-benchmark.jl`, and `morris_2009_dy2ti2o7.jl`.

The Hubbard cluster-perturbation benchmark follows D. Sénéchal, D. Perez, and D. Plouffe, *Physical Review Letters* **84**, 522--525 (2000), DOI: [10.1103/PhysRevLett.84.522](https://doi.org/10.1103/PhysRevLett.84.522), and specifically exercises sector-changing fermionic Green functions and CPT reconstruction. These larger scripts are useful release-level physics checks but should not be confused with the fast deterministic unit-test gate.

## Scope of the v0.0.7 claims

The validation documented here supports the APIs and physical regimes actually exercised by the frozen source tree. It does not imply that every mathematically representable Hamiltonian, every truncation, or every thermodynamic regime has been benchmarked against independent experimental data.

In particular, the current ED-DMFT validation does **not** establish superconducting/Nambu impurity self-consistency, multi-orbital DMFT, cluster DMFT, a continuum-bath limit, analytic continuation, or CT-QMC agreement. Nambu Green-function structures exist in `Greens`, but the validated v0.0.7 DMFT closure is normal-state and scalar in the correlated orbital.

Likewise, an optimized backend being available does not imply that every source representation can be realized without a numerical cutoff. Use `numerical_capabilities`, transformation certificates, truncation diagnostics, and the module-specific validation helpers to determine which claims apply to a particular calculation.

## Reproducibility

For deterministic numerical comparisons, record the Julia version, thread count, BLAS thread count, solver backend, truncation or bath size, convergence tolerances, random seed for stochastic methods, and any environment variables used by a benchmark. The DMFT performance scripts explicitly recommend `OPENBLAS_NUM_THREADS=1` when measuring Julia-thread scaling so that nested BLAS threading does not obscure the source of speedup.

The documentation records the v0.0.7 reference behavior, while CSV outputs under `benchmarks/` remain the machine-readable source for detailed benchmark trajectories and frequency-resolved data.
