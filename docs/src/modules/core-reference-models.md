```@meta
CurrentModule = Phundamental.PhundamentalCore
```

# Foundation API — Model Structures and Interchange

## Reciprocal-space decomposition

```@docs
ReciprocalDecomposition
decompose_reciprocal_vector
```

## Canonical Hamiltonian structures

```@docs
AbstractModelStructure
StructuralSupport
StructuralTerm
InteractionEdge
InteractionGraph
StructuralSignature
ModelStructure
representation_signature
canonical_model_expression
structure
canonicalize_structure
structural_signature
interaction_graph
structurally_equivalent
structural_span_membership
```

## Generative model spaces

```@docs
AbstractModelSpace
AdmissibilityCheck
AdmissibilityReport
PredicateAdmissibility
admissibility_check
admissibility_report
admissibility_reasons
GenerativeSpecification
HamiltonianBasis
HamiltonianSpace
generated_basis
hamiltonian_from_coefficients
```

## Parameters and models

```@docs
ParameterSpec
AbstractParameterConstraint
FixedParameterConstraint
EqualParameterConstraint
LinearParameterConstraint
ParameterConstraintCheck
ParameterConstraintReport
ParameterBinding
ParameterSpace
ParameterPoint
parameter_constraints
parameter_constraint_check
parameter_constraint_report
parameter_constraint_reasons
valid_parameter_point
parameter_constraint_matrix
parameter_dof
parameter_bindings
parameter_binding
with_parameters
ManyBodyModel
with_provenance
remap_model
with_truncation
instantiate_model
```

## Additional exported utilities

The remaining lightweight exported accessors and constructors are intentionally documented here by role rather than repeated as individual prose sections: `physicaldimension`, `factorconstraint`, `factortruncation`, the primitive operator types and constructor aliases, `atfactor`, `isprimitive`, `children`, the public basis constructors, wave-vector constructors and `wavevector_coordinates`, `admissible`, `model_space_dimension`, `generate_hamiltonian_space`, `parameter_space`, `parameter_point`, `parameter_names`, `parameter_bounds`, `parameter_units`, `parameter_vector`, `free_parameter_indices`, and `rebind_parameters`.

## Typed generation contexts and canonical coordinates

The foundation layer provides `GeometryContext`, `ComponentIndexLayout`, and `PhysicalModelContext` as typed carriers for validated source geometry, site-to-component ordering, and provenance. Generated spinful-fermion and Cartesian-vector families retain physical site positions independently of orbital, spin, or vector-component multiplicity. `component_index` maps physical sites and components into operator indices, while `component_site` and `component_coordinates` invert the map. Existing code that reads `model.parameters[:positions]` continues to work with physical site positions; certified Cartesian phonon models also carry atomic masses per operator coordinate. `structural_span_coordinates` reports real coefficients and residuals in the canonical operator basis and is shared by structural localization.

```@docs
ComponentIndexLayout
GeometryContext
PhysicalModelContext
ObservableContext
model_context
observable_context
geometry_context
structural_span_coordinates
```

## Experimental interchange and statistical comparison (Milestone C)

The package-level `ExperimentalGrid`, `PortablePrediction`, and `ExperimentalDataset` describe explicit measured and calculated numerical quantities without modifying the underlying `ForwardResult` or observable result types. `export_result` and `import_result` exchange a versioned, language-neutral dictionary in Julia column-major array order. `compare_measurement` uses only jointly valid bins and explicit experimental uncertainties or covariance. See [Experimental Data and Instrument Resolution](../workflows/experimental-data-resolution.md).

```@meta
CurrentModule = Phundamental
```

```@docs
ExperimentalGrid
PortablePrediction
ExperimentalDataset
MeasurementComparison
portable_prediction
experimental_dataset
export_result
import_result
compare_measurement
```

```@meta
CurrentModule = Phundamental.PhundamentalCore
```
