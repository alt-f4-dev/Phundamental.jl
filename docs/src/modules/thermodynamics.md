```@meta
CurrentModule = Phundamental.Thermodynamics
```

# Thermodynamics

`Phundamental.Thermodynamics` constructs finite-temperature states and thermodynamic observables on top of representation-aware models and numerical solvers. The module includes exact Gibbs ensembles, stochastic typicality methods, thermal Lanczos, canonical TPQ, symmetry-sector decomposition, density matrices, thermal workspaces, and a finite-cluster linked-cluster framework.

The thermal method is an explicit part of every result. An exact finite-system Gibbs calculation, a stochastic trace estimate, and a linked-cluster bulk estimate are therefore not represented as interchangeable values with hidden provenance.

## Thermal methods and states

`AbstractThermalMethod` is the common method abstraction and `AbstractThermalState` is the common state abstraction. `thermal_state(model, method; temperature, kB)` constructs a state at the requested physical temperature.

Thermal result metadata records the method, temperature, inverse temperature, solver details, sample/sector information, and any approximation controls required by the chosen method.

`ThermodynamicPoint` stores a set of scalar thermodynamic quantities at one temperature. `ThermodynamicCurve` collects a temperature sweep. `thermal_curve` evaluates a model over a requested grid while retaining the method used at each point.

## Exact Gibbs thermodynamics

`ExactGibbs` uses a complete finite eigensystem and therefore provides the reference thermal calculation whenever the numerical basis is small enough.

For eigenvalues $E_n$ at inverse temperature $\beta$,

$$Z=\sum_n e^{-\beta E_n},$$

and the exact probabilities are $p_n=e^{-\beta E_n}/Z$. `ExactGibbsState` stores the model, solver result, probabilities, temperature metadata, and observable information needed for exact expectation values and finite-temperature Green functions.

`partition_function`, `log_partition_function`, `free_energy`, `internal_energy`, `entropy`, and `heat_capacity` provide the standard thermodynamic quantities. `expectation` and `thermal_variance` evaluate observables in a thermal state.

At zero temperature, the exact pathway reduces to the ground-state limit rather than evaluating unstable exponentials at formally infinite $\beta$.

## Density matrices

`density_matrix` constructs the finite-dimensional thermal density operator when the state representation supports it. `density_matrix_trace_error` checks normalization.

Density-matrix construction is intentionally separate from ordinary scalar thermodynamic observables because materializing the full matrix can be much more expensive than evaluating traces from eigenvalues and matrix elements.

## Thermal typicality

`ThermalTypicality` estimates thermal traces using random normalized vectors filtered by $e^{-\beta H/2}$. It can retain filtered vectors when arbitrary observables will be evaluated later.

The method trades deterministic completeness for stochastic sampling. `ThermalEstimate` therefore stores an estimate together with uncertainty information rather than masquerading as an exact scalar.

Typicality becomes useful when the Hilbert space is too large for a complete eigenspectrum but matrix-free exponential actions remain practical.

## Thermal Lanczos

`ThermalLanczos` implements a low-memory stochastic trace estimate based on Lanczos quadrature. It avoids retaining full filtered vectors and is therefore useful when thermodynamic traces are required but arbitrary post-hoc observables are not.

The quadrature depth, number of random vectors, and random seed are numerical controls and should be reported with quantitative results.

## Canonical TPQ

`CanonicalTPQ` implements canonical thermal pure quantum states. The v0.0.7 implementation uses explicit imaginary-time filtering with a Krylov propagator option and exposes controls for sampling, Krylov size/tolerance, parallel execution, basis-size limits, and optional symmetry-sector decomposition.

Canonical TPQ and generic thermal typicality are conceptually related but remain distinct methods because their state construction and estimator semantics differ.

## Symmetry-sector decomposition

`SectorDecomposition` divides an eligible spin-$1/2$ model into exactly conserved total-$S^z$ sectors. All physical sectors are retained; the decomposition is an exact reorganization of the trace rather than a truncation.

`SectorThermalWorkspace` caches sector-specific numerical objects. Equivalent sectors may only be paired when global spin-inversion symmetry is certified. The implementation does not infer such an equivalence merely from the visual form of a Hamiltonian.

`validate_sector_thermodynamic_recombination` checks that summing sector contributions reproduces the corresponding full-space thermodynamics on systems where both calculations are feasible.

## Thermal workspaces

`ThermalWorkspace` and `SectorThermalWorkspace` cache expensive basis, operator, or Krylov data that can be reused across a temperature sweep. `thermal_workspace` constructs an appropriate workspace for the selected method and model.

The workspace is a performance object, not a thermal state. It may be reused to construct multiple states or estimates without changing the represented model or ensemble.

## Thermodynamic observables

The common scalar functions operate on supported thermal states and estimates:

- `partition_function` and `log_partition_function` quantify state normalization;
- `free_energy` evaluates $F=-k_BT\log Z$ when the method provides an absolute partition function;
- `internal_energy` evaluates $\langle H\rangle$;
- `entropy` evaluates the thermodynamic entropy for methods that provide the necessary normalization;
- `heat_capacity` evaluates thermal energy fluctuations or the corresponding method-specific estimator;
- `expectation` evaluates a registered or supplied observable;
- `thermal_variance` evaluates $\langle O^2\rangle-\langle O\rangle^2$.

Stochastic methods may return uncertainty-bearing `ThermalEstimate` objects rather than bare numbers.

## Connected clusters and linked-cluster reconstruction

`ThermalCluster` represents a connected finite subcluster with site and bond identity. `connected_clusters` enumerates the connected clusters available inside a finite reference graph.

`linked_cluster_weights` recursively forms linked weights according to

$$W_P(C)=P(C)-\sum_{C'\subsetneq C}W_P(C'),$$

where the sum runs over proper connected subclusters included by the selected cluster construction.

`cluster_thermodynamics` evaluates the selected thermal property on each cluster and returns `ClusterThermodynamicsResult`. `bulk_estimate` combines the linked weights with embedding/multiplicity information supplied by the finite reference construction.

The v0.0.7 cluster machinery is a finite-reference linked-cluster framework. It does not claim an automatic infinite-lattice thermodynamic limit; result metadata explicitly records `:infinite_lattice_claim => false` where appropriate.

!!! note
    The linked-cluster machinery here is a statistical-mechanics cluster expansion and is unrelated to quantum-chemistry coupled-cluster theory.

## Validation

`validate_thermal_state` checks normalization and method-specific invariants. `validate_sampling_against_exact` compares stochastic methods with exact Gibbs results on tractable systems. `validate_exact_spectrum_thermodynamics` checks thermodynamic quantities against direct spectral formulas.

`validate_sector_thermodynamic_recombination` verifies exact sector recombination. `validate_linked_cluster_closure` checks the recursive cluster-weight identities on finite references.

The validation philosophy is to compare approximate or stochastic methods against an independently computed exact reference before using them at scales where the reference is unavailable.

## Related modules

- [Solvers](solvers.md) provides eigensolvers, Krylov exponential actions, matrix-free backends, and symmetry sectors used by thermal methods.
- [Observables](observables.md) converts thermal states or sampled ensembles into physical response functions.
- [Greens](greens.md) uses `ExactGibbsState` for exact finite-temperature Lehmann Green functions.
- [Embedding](embedding.md) uses finite-temperature impurity thermodynamics internally during ED-DMFT.

## API reference

### Methods, workspaces, and state types

```@docs
ExactGibbs
ThermalTypicality
ThermalLanczos
CanonicalTPQ
SectorDecomposition
ThermalWorkspace
SectorThermalWorkspace
thermal_workspace
ThermalEstimate
ThermodynamicPoint
ThermodynamicCurve
ExactGibbsState
SampledThermalState
thermal_state
thermal_curve
```

### Density matrices and observables

```@docs
density_matrix
density_matrix_trace_error
expectation
thermal_variance
```

### Cluster thermodynamics

```@docs
ThermalCluster
ClusterThermodynamicsResult
connected_clusters
linked_cluster_weights
cluster_thermodynamics
bulk_estimate
```

### Validation

```@docs
validate_thermal_state
validate_sampling_against_exact
validate_exact_spectrum_thermodynamics
validate_sector_thermodynamic_recombination
validate_linked_cluster_closure
```

## Additional exported thermodynamic API

The remaining public functions are `partition_function`, `log_partition_function`, `free_energy`, `internal_energy`, `entropy`, `heat_capacity`, `cluster_order`, `cluster_sites`, and `cluster_bonds`, together with the abstract method/state types used for extension.
