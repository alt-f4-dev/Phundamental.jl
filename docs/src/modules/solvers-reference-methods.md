```@meta
CurrentModule = Phundamental.Solvers
```

# Solver API — Krylov, Fermionic, and Classical Methods

Reference documentation for Krylov resolvents and propagation, cluster Green functions and CPT, classical sampling, and numerical validation helpers.

The [Solvers overview](solvers.md) explains applicability, physical approximation boundaries, and the relationship with the rest of the module.

## Krylov tools

```@docs
ContinuedFractionResult
continued_fraction
continued_fraction_spectrum
exp_action
```

## Fermion Green functions and CPT

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

## Classical sampling

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

## Validation helpers

```@docs
validate_optimized_against_generic
validate_packed_spin_implicit_basis
validate_packed_spin_rank_lookup
validate_spin_sector_decomposition
```
