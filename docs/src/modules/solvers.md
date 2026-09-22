```@meta
CurrentModule = Phundamental.Solvers
```

# Solvers

`Phundamental.Solvers` converts representation-aware symbolic models into numerical calculations. It provides a generic finite-basis reference path, symmetry sectors, dense and Krylov eigensolvers, time evolution, exact packed backends, quadratic fermion/boson solvers, harmonic phonons, linear spin-wave theory, classical Monte Carlo, Green-function resolvents, and cluster perturbation theory.

The solver layer is intentionally downstream of representations and transformations. A solver may change how a state is stored or how a Hamiltonian action is computed, but an exact backend must not silently change the represented physics.

$$\boxed{\text{physical representation}\neq\text{numerical backend}}$$

Likewise, `AutoSolver` performs exact specialization only; it does not choose a physical approximation such as linear spin-wave theory on the user's behalf.

## Solver protocol and results

All numerical solvers derive from `AbstractSolver` and implement `solve(solver, model, ...)`. `solver_name` returns a short symbolic identifier suitable for result metadata.

`SolverResult` is the common eigenproblem result. It stores the solver, model, eigenvalues, right states, optional left states, basis, and metadata. `EnergySpectrumResult` and `TimeEvolutionResult` provide specialized result containers for spectrum and dynamics workflows.

Convenience accessors include `energy_spectrum`, `nlevels`, `states`, `eigenstate`, `groundenergy`, `groundstate`, and `isbiorthogonal`. These functions let downstream observables avoid depending on a particular solver implementation.

## Numerical capability analysis

`NumericalCapabilities` describes whether a model can enter the generic finite-basis pipeline without changing its physical sector. The diagnostic records symbolic-operator support, whether a finite basis exists, whether that basis can be constructed, whether numerical realization is possible, whether a truncation is required, whether exact constraints remain unresolved, whether truncation metadata was merely transported from another basis, whether coordinate discretization is required, whether biorthogonal treatment is needed, Hermiticity, compatible solvers, and explanatory reasons.

Call

```julia
caps = numerical_capabilities(model)
```

before debugging a failed matrix realization. The `reasons` field is designed to explain why a representation cannot yet be solved by the generic pipeline.

`TransformationCapabilities` combines a transformation certificate with numerical closure information for the actual transformed target model. This is useful because an exact transformation can still produce a target representation that requires an explicit truncation or projector before numerical realization.

### Important capability rules

Bare coordinate/momentum representations are not silently discretized. They must first be transformed with `CoordinateLadder` or handled by a dedicated coordinate-aware backend.

Bosonic factors without an exact finite occupation bound require an explicit numerical truncation. A truncation transported through an exact representation change remains provenance until the user chooses a cutoff appropriate to the target basis.

The current generic packed fermionic basis implementation supports at most 64 fermionic modes per factor; the equivalent Majorana implementation is limited to 64 complex fermionic modes.

Unresolved physical-sector metadata is an error condition. The numerical layer will not discard a transported constraint merely to make a finite matrix constructible.

## Computational bases

`ComputationalBasis` is the concrete finite enumeration used by the generic matrix pipeline. Its state types are `SpinBasisState`, `FermionBasisState`, `BosonBasisState`, and `CompositeBasisState`.

`computational_basis(model)` inspects the model representation, exact physical restrictions, and numerical truncations to construct a finite basis. `basis_dimension` and `basis_state` provide basic access.

These objects are reference representations of the numerical basis. Optimized packed backends may avoid materializing equivalent state objects while preserving the same basis ordering and physical sector.

## Matrix realization

`realize` maps a symbolic operator or Hamiltonian into the selected computational basis. `MatrixRealizationResult` stores the realized matrix together with basis and metadata; `matrix` extracts the numerical matrix and `hamiltonian_matrix` is the Hamiltonian-oriented convenience entry point.

`SymbolicLinearOperator` provides a matrix-free action for the generic symbolic pipeline. Matrix-free realization is especially useful for Lanczos or Krylov evolution when a dense matrix would be wasteful.

Matrix realization is exact relative to the supplied finite basis and truncation. It does not claim convergence with respect to a bosonic cutoff; convergence is a separate validation problem.

## Exact diagonalization

`ExactDiagonalization` is the small-system reference eigensolver. Hermitian problems use the Hermitian pathway; explicitly non-Hermitian or biorthogonal representations can request the corresponding general eigensystem path.

Dense exact diagonalization is intentionally retained even when optimized backends exist. It provides an independent correctness oracle for small-system tests, representation-transform validation, impurity-solver validation, and new backend development.

For a model `model`, the basic workflow is

```julia
solution = solve(ExactDiagonalization(), model)
E0 = groundenergy(solution)
psi0 = groundstate(solution)
```

## Lanczos and Krylov methods

`LanczosSolver` computes low-energy eigenpairs without requiring a full dense eigensystem and may use matrix-free Hamiltonian actions. It is appropriate when the finite basis is too large for dense diagonalization but a modest number of extremal states is sufficient.

`ContinuedFractionResult`, `continued_fraction`, `continued_fraction_value`, and `continued_fraction_spectrum` implement Lanczos continued-fraction evaluation for resolvents and spectral functions.

`exp_action` applies a matrix exponential to a vector using Krylov methods and is used by time-evolution and thermal-sampling workflows.

`TimeEvolutionSolver` and `evolve` expose real- or imaginary-time evolution through the shared model representation rather than requiring callers to construct matrices manually.

## Symmetry sectors

`AbstractSector` is the base abstraction for a numerically restricted symmetry sector. `SpinSector` represents fixed total spin-$z$ occupation for spin-$1/2$ packed bases. `FermionSector` represents fixed fermion-number sectors. `QuantumNumber`, `QuantumNumberSector`, and `CompositeSector` provide a more generic framework for conserved labels.

`quantum_numbers`, `sector_value`, and `has_quantum_number` expose sector metadata. `spin_sector_profile` and `spin_sector_plan` analyze spin Hamiltonians and plan sector decompositions without changing the physical model.

A sector restriction is valid only when the Hamiltonian is known to conserve the requested quantity. The optimized backends reject uncertified sector choices rather than assuming conservation from the user's intent.

## Optimized packed backends

`AbstractOptimizedBackend` is the base for representation-specific exact numerical encodings. `PackedSpinBackend`, `PackedFermionBackend`, and `PackedBosonBackend` use compact state encodings and specialized operator compilation.

`optimized_backend(model)` reports the exact packed backend naturally associated with the current representation, and `supports_optimized_backend` checks availability. This dispatch does not apply a physical representation transformation.

`optimized_matrix_free_operator` constructs an exact representation-specific Hamiltonian action. For packed spin-$1/2$ sectors it can keep the basis implicit, use optional rank-lookup tables, and use a race-free threaded gather kernel for conserved fixed-$S^z$ sectors.

`OptimizedLanczos` combines these backends with the Lanczos algorithm. If an optimized backend is unavailable and `fallback=true`, it may fall back to the generic symbolic Lanczos implementation without changing the represented model.

## `AutoSolver`

`AutoSolver` is an exact-specialization dispatcher. It selects dedicated exact solvers when the representation and Hamiltonian structure make them applicable: harmonic phonons for a coordinate model carrying force constants and masses, non-Hermitian exact diagonalization for a biorthogonal representation, exact quadratic fermion or boson solvers for certified quadratic Hamiltonians, and otherwise the optimized Lanczos path.

It never automatically invokes an approximation. In particular, a spin Hamiltonian is not replaced by linear spin-wave theory merely because `LinearSpinWaveSolver` exists.

## Quadratic fermions

`QuadraticFermionSolver` extracts and diagonalizes an exactly quadratic fermionic Hamiltonian. The solver is applicable only when the symbolic model can be certified as quadratic within tolerance.

This path is useful for free-fermion and Bogoliubov--de Gennes models because it solves the one-particle/BdG structure rather than enumerating the full many-body Fock space. The result remains an exact specialization of the supplied quadratic model.

## Quadratic bosons

`QuadraticBosonSolver` handles exactly quadratic bosonic Hamiltonians, including the metric structure required by bosonic Bogoliubov problems. `BosonSubspaceProjection` and `orthogonal_boson_subspace` provide explicit control over projected bosonic subspaces when zero modes or constrained directions must be handled.

A quadratic boson solver is not a substitute for establishing how an interacting spin or phonon model became quadratic. If the quadratic Hamiltonian arose from an approximation, that approximation should be represented and recorded before the solver is invoked.

## Harmonic phonons

`HarmonicPhononSolver` solves periodic harmonic lattice dynamics from masses and real-space force constants. `PhononModeResult` and `PhononDispersionResult` store mode frequencies, eigenvectors, momenta, and metadata.

`dynamical_matrix`/`dynamical_matrix!` construct the mass-weighted dynamical matrix. `dynamical_gradient!` and `dynamical_hessian!` provide derivative information used for mode analysis. `acoustic_sum_rule_residual` diagnoses translational-invariance defects in the force constants.

`phonon_dispersion` evaluates a sequence of wave vectors. `group_velocities` and `directional_group_velocities` compute dispersion derivatives. `phonon_zero_point_energy` computes the harmonic zero-point contribution.

The phonon backend is a specialized exact solver for the supplied harmonic model. Any approximation used to obtain the force constants belongs upstream in model construction.

## Linear spin-wave theory

`LinearSpinWaveSolver` implements the quadratic bosonic theory obtained after expanding a spin model around a supplied ordered reference state. `SpinWaveResult` stores the resulting modes and metadata, while `spinwave_zero_point_correction` and `spinwave_ground_energy` provide the standard harmonic corrections.

Linear spin-wave theory is an approximation even though the final quadratic boson problem is solved exactly. Phundamental keeps this distinction explicit in result metadata and does not allow `AutoSolver` to choose LSWT implicitly.

## Fermionic resolvents and cluster perturbation theory

`ProjectedFermionResolvent` and `FermionGreenWorkspace` support zero-temperature projected fermionic Green-function calculations on finite interacting clusters. `fermion_transition_vector` and `fermion_green_workspace` prepare the addition/removal sectors needed for resolvent evaluation.

`cluster_green_matrix` computes the interacting cluster Green matrix. `cpt_intercluster_hopping` constructs the intercluster hopping contribution used by cluster perturbation theory (CPT), and `cluster_perturbation_green_matrix`, `cpt_periodize`, and `cluster_perturbation_green` implement the lattice reconstruction.

This zero-temperature cluster Green-function machinery is distinct from the finite-temperature Matsubara abstractions in [Greens](greens.md). Both are public because they serve different numerical regimes.

## Classical Ising sampling

The solver module also contains a classical Monte Carlo pathway. `IsingProblem` represents

$$E(\mathbf s)=E_0+\frac{1}{2}\mathbf s^T J\mathbf s+\mathbf h^T\mathbf s,\qquad s_i\in\{-1,+1\}.$$

`MetropolisSampler` performs local Metropolis updates and `ConfigurationEnsemble` stores sampled configurations. `sample` runs the chain, while `ising_energy` and `ising_flip_delta` evaluate energies and single-spin update costs.

`autocorrelation_time`, `effective_sample_size`, and `correlated_stderr` provide correlation-aware statistical diagnostics. `validate_metropolis_detailed_balance` and `validate_sampling_against_exact_classical` provide small-system validation against exact classical calculations.

## Validation

`validate_matrix_realization` compares symbolic and realized operators where an independent reference is available. The optimized backends have dedicated validation functions that compare packed and generic pipelines, basis ordering, rank lookup, sector decomposition, mode results, phonon dispersions, and spin-wave results.

The package maintains dense and generic paths as references rather than deleting them after optimization. This policy was important in v0.0.7: the ED-DMFT impurity backend was first validated against dense ED, then sectorized, then threaded while preserving serial/dense equivalence.

## Related modules

- [Core / Foundation](core.md) defines the representation and physical restrictions that determine numerical admissibility.
- [Transformations](transformations.md) changes representations before a solver is selected.
- [Observables](observables.md) consumes solver results without depending on the underlying numerical backend.
- [Thermodynamics](thermodynamics.md) builds finite-temperature states on top of exact, Krylov, or sampled numerical machinery.
- [Greens](greens.md) provides finite-temperature Matsubara propagators from thermal eigensystems.
- [Embedding](embedding.md) uses the solver protocol for ED impurity problems and DMFT.

## API reference

### Solver protocol and results

```@docs
AbstractSolver
solve
solver_name
SolverResult
EnergySpectrumResult
nlevels
states
eigenstate
groundenergy
groundstate
isbiorthogonal
TimeEvolutionResult
energy_spectrum
```

### Numerical capabilities

```@docs
NumericalCapabilities
TransformationCapabilities
numerical_capabilities
transformation_capabilities
```

### Computational basis and matrix realization

```@docs
SpinBasisState
FermionBasisState
BosonBasisState
CompositeBasisState
ComputationalBasis
computational_basis
MatrixRealizationResult
realize
matrix
hamiltonian_matrix
SymbolicLinearOperator
```

### Optimized dispatch

```@docs
OptimizedLanczos
AutoSolver
optimized_backend
optimized_matrix_free_operator
```

### Sectors

```@docs
SpinSector
FermionSector
QuantumNumber
QuantumNumberSector
CompositeSector
spin_sector_profile
spin_sector_plan
```

### Mode solvers

```@docs
QuadraticModeResult
PhononModeResult
PhononDispersionResult
SpinWaveResult
BosonSubspaceProjection
orthogonal_boson_subspace
HarmonicPhononSolver
LinearSpinWaveSolver
mode_energies
mode_vectors
dynamical_matrix
dynamical_matrix!
dynamical_gradient!
dynamical_hessian!
acoustic_sum_rule_residual
group_velocities
directional_group_velocities
phonon_dispersion
phonon_zero_point_energy
spinwave_zero_point_correction
spinwave_ground_energy
```

### Krylov tools

```@docs
ContinuedFractionResult
continued_fraction
continued_fraction_spectrum
exp_action
```

### Fermion Green functions and CPT

```@docs
FermionGreenWorkspace
fermion_transition_vector
fermion_green_workspace
cluster_green_matrix
cpt_intercluster_hopping
cluster_perturbation_green_matrix
cpt_periodize
cluster_perturbation_green
```

### Classical sampling

```@docs
IsingProblem
MetropolisSampler
ConfigurationEnsemble
sample
ising_flip_delta
autocorrelation_time
validate_metropolis_detailed_balance
validate_sampling_against_exact_classical
```

### Validation helpers

```@docs
validate_optimized_against_generic
validate_packed_spin_implicit_basis
validate_packed_spin_rank_lookup
validate_spin_sector_decomposition
```

## Additional exported solver API

The remaining exports include `basis_dimension`, `basis_state`, `ExactDiagonalization`, `LanczosSolver`, `TimeEvolutionSolver`, the abstract sector and backend types, `quantum_numbers`, `sector_value`, `has_quantum_number`, `ProjectedFermionResolvent`, `PackedSpinBackend`, `PackedFermionBackend`, `PackedBosonBackend`, `QuadraticFermionSolver`, `QuadraticBosonSolver`, `supports_optimized_backend`, the in-place/derivative mode helpers, additional continued-fraction evaluation utilities, mode and spin-wave validation functions, abstract classical sampling types, `nsamples`, `configuration`, `ising_energy`, `effective_sample_size`, `correlated_stderr`, `evolve`, and `validate_matrix_realization`.
