```@meta
CurrentModule = Phundamental.PhundamentalCore
```

# Core / Foundation

`Phundamental.PhundamentalCore` defines the mathematical ontology used by every higher layer of Phundamental.jl. The public alias `Phundamental.Foundation` refers to this module; the package intentionally does not define `Phundamental.Core` because `Core` is a built-in Julia module.

The Foundation layer answers four questions before any numerical method is selected: what state space is being represented, what operator algebra acts on it, what physical restrictions define the admissible theory, and how the Hamiltonian belongs to a model and parameter space. Numerical solvers can then inspect this information without changing its physical meaning.

A package-wide invariant is

$$\text{ambient state space}\neq\text{physical subspace}\neq\text{numerical truncation}.$$

A physical subspace is part of the mathematical model. A numerical truncation is a computational approximation to an otherwise larger or infinite representation. Code that treats these concepts as interchangeable will generally produce incorrect dimension counting, transformation certificates, or solver capability decisions.

## State spaces

All state spaces derive from `AbstractStateSpace`. `AbstractHilbertSpace` and `AbstractFockSpace` refine the hierarchy for ordinary Hilbert and occupation-number constructions.

`SpinHilbertSpace` represents a tensor product of finite spin spaces. `FermionFockSpace` represents a finite set of fermionic modes. `BosonFockSpace` and `CoordinateHilbertSpace` are intrinsically infinite unless a numerical truncation is supplied elsewhere. `CompositeStateSpace` forms tensor products of heterogeneous factors, while `BiorthogonalSpace` records the paired state structure required by non-Hermitian or similarity-transformed calculations.

`spacedimension(space)` reports the exact ambient dimension when it is finite and returns `nothing` for an intrinsically infinite space. `physicaldimension(...)` instead refers to the exact dimension of an explicitly defined `PhysicalSubspace` when that dimension is known.

A `PhysicalSubspace` stores exact restrictions on the ambient theory, such as an occupancy or gauge constraint. A `NumericalTruncation` stores a deliberate finite computational restriction. The distinction is particularly important for bosonic mappings such as Holstein--Primakoff and Schwinger-boson constructions.

### Composite restrictions

A composite representation may attach restrictions to individual factors through `FactorConstraint` and `FactorTruncation`. `factorconstraint` and `factortruncation` retrieve factor-specific information without flattening heterogeneous spaces into one undifferentiated global constraint.

!!! warning
    A finite boson cutoff is normally a `NumericalTruncation`, not a physical occupancy law. Exact occupancy bounds introduced by a representation map belong to the physical subspace or constraint metadata instead.

## Operator algebras

`AbstractOperatorAlgebra` identifies the algebraic family in which symbolic primitives are interpreted. The concrete algebras are `SpinAlgebra`, `BosonAlgebra`, `FermionAlgebra`, `MajoranaAlgebra`, `CoordinateMomentumAlgebra`, and `CompositeAlgebra`.

The algebra object is part of a `Representation`; it is not inferred merely from the spelling of an operator. This allows a transformation to construct an explicit target representation and gives the algebra layer a well-defined context for commutators, normal ordering, and canonicalization.

`algebra_kind` returns the symbolic family associated with an algebra object. `CompositeAlgebra` preserves the factor structure required when a model contains, for example, both fermionic and bosonic operators.

## Symbolic operator expressions

Every symbolic operator expression derives from `AbstractOperatorExpr`, while primitive generators derive from `AbstractPrimitiveOperator`. Composite expressions are represented by `OperatorSum`, `OperatorProduct`, and `ScaledOperator`; `IdentityOperator` provides the multiplicative identity.

The primary constructor families are:

| Physical degree of freedom | Constructors |
| --- | --- |
| Spin | `Sx(i)`, `Sy(i)`, `Sz(i)`, `Sp(i)`, `Sm(i)` |
| Boson | `b(i)`, `b(i)'`, `nb(i)` |
| Fermion | `c(i)`, `c(i)'`, `nf(i)` |
| Majorana | `γ(i)` |
| Coordinate/momentum | `xop(i)`, `pop(i)` |

Creation operators use Julia adjoint syntax canonically. The symbolic tree remains noncommutative until the [Algebra](algebra.md) layer applies algebra-specific rewrite and canonicalization rules.

`FactorOperator` and `atfactor` qualify an operator by a factor of a composite state space. This is the preferred mechanism for expressing heterogeneous models because the factor identity remains available to transformations and numerical realization.

Structural accessors such as `isprimitive`, `children`, `operator_degree`, `support_size`, and `body_order` allow algorithms to reason about an expression without evaluating it. Invariant generation, capability analysis, and model-space construction use these structural properties to restrict candidate operators.

## Bases

The Foundation layer distinguishes mathematical bases from numerical storage backends. Public basis types include `SpinProductBasis`, `FermionOccupationBasis`, `BosonOccupationBasis`, `CoordinateBasis`, `CompositeBasis`, and `BiorthogonalBasis`.

These basis objects describe how a representation is interpreted. They do not imply that a solver must literally store a dense vector in that basis. For example, the optimized solver layer may use packed-bit encodings while the physical representation still carries a spin product basis.

This separation is summarized by

$$\text{physical representation}\neq\text{numerical backend}.$$

## Representations

`Representation` is the central mathematical descriptor of a model. It stores the ambient state space, operator algebra, basis, exact physical subspace, additional constraints, gauge structure, reference state, ordering convention, and numerical truncation.

Conceptually,

$$\mathfrak R=(\mathcal S,\mathcal S_{\mathrm{phys}},\mathcal A,\mathcal B,\mathcal C,\mathcal G,|\Omega\rangle,\prec,\mathcal T_{\mathrm{num}}).$$

`isconstrained` reports whether exact constraints are present. `istruncated` reports whether a numerical truncation is present. `ambientdimension`, `representationdimension`, and `physicalspace` expose different aspects of the representation rather than collapsing them into a single dimension field.

`GaugeStructure` records gauge redundancy introduced by enlarged representations. This information is used by transformation certificates to distinguish an exact algebraic embedding from a one-to-one unitary change of basis.

!!! note
    Nambu indexing is not represented by `Representation`. In v0.0.7 it is an index structure on Green functions and is documented in [Greens](greens.md).

## Reciprocal-space primitives

Reciprocal vectors are typed to prevent ambiguous coordinate interpretation. `CartesianWaveVector` represents Cartesian reciprocal-space coordinates, while `ReciprocalWaveVector` represents coordinates in a reciprocal basis. `wavevector_coordinates` extracts the stored coordinates.

`decompose_reciprocal_vector` implements the decomposition

$$\mathbf Q=\mathbf q+\mathbf G,$$

where `q` is chosen as the nearest-origin representative in the first Brillouin-zone Wigner--Seitz cell and `G` is the reciprocal-lattice translation. The result is stored in `ReciprocalDecomposition`.

The API intentionally rejects ambiguous raw vectors in places where Cartesian versus reciprocal coordinates would change the physics. Callers should construct the appropriate wave-vector type explicitly.

## Admissibility and generative model spaces

`AbstractAdmissibilityCondition` defines a predicate over candidate Hamiltonian terms. `PredicateAdmissibility` packages a Julia predicate as a typed admissibility rule. `GenerativeSpecification` combines generation rules with admissibility conditions so that a model class can be defined before specific couplings are chosen.

`HamiltonianBasis` stores a linearly independent operator basis for a model class. `HamiltonianSpace` combines that basis with the associated constraints and metadata. `generated_basis` exposes the basis produced by a generative specification, while `model_space_dimension` reports its dimension.

`generate_hamiltonian_space` applies a specification to construct an admissible model space. `hamiltonian_from_coefficients` converts a coefficient vector into a symbolic Hamiltonian in that basis. This layer is the natural connection point for topology-aware or data-driven model inference because inference can act on a constrained Hamiltonian space rather than on arbitrary operator strings.

## Parameter spaces

`ParameterSpec` describes one named parameter, including bounds, units, and whether it is fixed or free. `ParameterSpace` collects those specifications, and `ParameterPoint` assigns concrete values.

Convenience functions include `parameter_names`, `parameter_bounds`, `parameter_units`, `parameter_vector`, and `free_parameter_indices`. `parameter_point` validates concrete values against a `ParameterSpace`, while `rebind_parameters` and `with_parameters` create updated parameterized objects without mutating the original model.

The parameter system is metadata-aware but does not impose a particular optimizer. Inverse methods can therefore use the same model-space and parameter-space definitions while selecting their own optimization or probabilistic backend.

## `ManyBodyModel`

`ManyBodyModel` is the primary package-level physical model object. It stores a `Representation`, symbolic Hamiltonian, parameter information, observable registry, provenance, optional originating specification, model-space information, and parameterization metadata.

A useful conceptual form is

$$\mathfrak M=(\mathfrak R,\hat H,\boldsymbol\theta,\mathcal O,\text{provenance}).$$

`instantiate_model` builds a `ManyBodyModel` from a `GenerativeSpecification` and a parameter point. `with_parameters` changes parameter values while preserving the model definition. `with_provenance` appends construction history, and `remap_model` is used by representation transformations to create a mathematically related model while retaining traceability.

`with_truncation` attaches a numerical truncation without rewriting the physical model definition. This is the preferred way to make an infinite bosonic representation computationally realizable.

## Invariants and failure modes

Foundation-level errors are usually semantic rather than numerical. Common causes include inconsistent state-space dimensions, malformed basis/representation combinations, duplicate parameter names, parameter values outside declared bounds, ambiguous reciprocal coordinates, or treating an infinite space as finite without a numerical truncation.

Higher layers should inspect the representation rather than infer requirements ad hoc. In particular, [Solvers](solvers.md) uses the Foundation metadata to decide whether a finite computational basis can be constructed and whether unresolved physical constraints remain.

## Related modules

- [Algebra](algebra.md) manipulates `AbstractOperatorExpr` objects using the declared operator algebra.
- [Crystallography](crystallography.md) generates symmetry-admissible Hamiltonian bases.
- [Transformations](transformations.md) maps complete representations and models rather than only substituting operators.
- [Models](models.md) supplies concrete physical model specifications that build `ManyBodyModel` instances.
- [Solvers](solvers.md) realizes a representation numerically without changing its physical identity.

## API reference

### State spaces and exact restrictions

```@docs
AbstractStateSpace
AbstractHilbertSpace
AbstractFockSpace
SpinHilbertSpace
FermionFockSpace
BosonFockSpace
CoordinateHilbertSpace
CompositeStateSpace
BiorthogonalSpace
PhysicalSubspace
NumericalTruncation
spacedimension
FactorConstraint
FactorTruncation
```

### Operator algebras

```@docs
AbstractOperatorAlgebra
SpinAlgebra
BosonAlgebra
FermionAlgebra
MajoranaAlgebra
CoordinateMomentumAlgebra
CompositeAlgebra
algebra_kind
```

### Symbolic expressions

```@docs
IdentityOperator
NumberOperator
OperatorSum
OperatorProduct
ScaledOperator
FactorOperator
operator_degree
support_size
body_order
```

### Representations

```@docs
GaugeStructure
Representation
isconstrained
istruncated
ambientdimension
representationdimension
physicalspace
```

### Reciprocal-space decomposition

```@docs
ReciprocalDecomposition
decompose_reciprocal_vector
```

### Generative model spaces

```@docs
PredicateAdmissibility
GenerativeSpecification
HamiltonianBasis
HamiltonianSpace
generated_basis
hamiltonian_from_coefficients
```

### Parameters and models

```@docs
ParameterSpec
ParameterSpace
ParameterPoint
with_parameters
ManyBodyModel
with_provenance
remap_model
with_truncation
instantiate_model
```

## Additional exported utilities

The remaining lightweight exported accessors and constructors are intentionally documented here by role rather than repeated as individual prose sections: `physicaldimension`, `factorconstraint`, `factortruncation`, the primitive operator types and constructor aliases, `atfactor`, `isprimitive`, `children`, the public basis constructors, wave-vector constructors and `wavevector_coordinates`, `admissible`, `model_space_dimension`, `generate_hamiltonian_space`, `parameter_space`, `parameter_point`, `parameter_names`, `parameter_bounds`, `parameter_units`, `parameter_vector`, `free_parameter_indices`, and `rebind_parameters`.
