```@meta
CurrentModule = Phundamental.Solvers
```

# Solver API — Protocol, Capabilities, and Bases

Reference documentation for solver results, numerical capabilities, computational basis construction, matrix realization, optimized dispatch, and symmetry sectors.

The [Solvers overview](solvers.md) explains applicability, physical approximation boundaries, and the relationship with the rest of the module.

## Solver protocol and results

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

## Numerical capabilities

```@docs
NumericalCapabilities
TransformationCapabilities
numerical_capabilities
transformation_capabilities
```

## Computational basis and matrix realization

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

## Optimized dispatch

```@docs
OptimizedLanczos
AutoSolver
optimized_backend
optimized_matrix_free_operator
```

## Sectors

```@docs
SpinSector
FermionSector
QuantumNumber
QuantumNumberSector
CompositeSector
spin_sector_profile
spin_sector_plan
```
