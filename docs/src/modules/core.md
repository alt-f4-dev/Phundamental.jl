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
    Nambu indexing is not represented by `Representation`. It is an index structure on Green functions and is documented in [Greens](greens.md).

## Reciprocal-space primitives

Reciprocal vectors are typed to prevent ambiguous coordinate interpretation. `CartesianWaveVector` represents Cartesian reciprocal-space coordinates, while `ReciprocalWaveVector` represents coordinates in a reciprocal basis. `wavevector_coordinates` extracts the stored coordinates.

`decompose_reciprocal_vector` implements the decomposition

$$\mathbf Q=\mathbf q+\mathbf G,$$

where `q` is chosen as the nearest-origin representative in the first Brillouin-zone Wigner--Seitz cell and `G` is the reciprocal-lattice translation. The result is stored in `ReciprocalDecomposition`.

The API intentionally rejects ambiguous raw vectors in places where Cartesian versus reciprocal coordinates would change the physics. Callers should construct the appropriate wave-vector type explicitly.

## Canonical Hamiltonian structures

Batch 8 introduces a coefficient-independent structural intermediate representation between symbolic Hamiltonians and later structural inference. `structure(model)` canonicalizes the Hamiltonian in its declared operator algebra, removes numerical coupling magnitudes from structural identity, records local supports and operator topology, and returns a `ModelStructure`.

The structural signature has the conceptual form

$$\mathfrak G=(\mathfrak R_{\mathrm{phys}},\{\mathcal O_a\},\mathcal E),$$

where the physical representation context excludes numerical truncation and user-facing representation names, the operator set contains canonical nonzero term templates, and $\mathcal E$ is an interaction hypergraph connecting the supports of each term.

`structurally_equivalent(A, B)` means that two Hamiltonians have the same canonical operator structure in the same physical representation context. Numerical coefficients are deliberately excluded, so changing a nonzero coupling does not create a new structure. Setting a coupling exactly to zero can remove an active structural term from an instantiated model. `structure(space::HamiltonianSpace)` instead reports the full family structure admitted by the basis, independent of a particular parameter point.

Exact representation transformations do not collapse structural identity. A spin Hamiltonian and its exact Jordan--Wigner fermion image generally have different `StructuralSignature`s because locality and operator topology have changed. The transformation layer exposes `structurally_related(source, target, transformation)` for this distinct notion of exact cross-representation relation.

Batch 8 also uses the algebraic canonicalizer to certify real linear-span membership in a `HamiltonianSpace`. This closes the Batch 7 case in which an algebraically equivalent expression could remain structurally unresolved even though canonicalization proves its basis coordinates.

## Admissibility and generative model spaces

`AbstractAdmissibilityCondition` is the extension point for physical restrictions on candidate Hamiltonians, while `PredicateAdmissibility` packages an ordinary Julia predicate as a named rule. `GenerativeSpecification` now defines the explicit physical admissibility boundary through its operator algebra, optional algebra-aware `max_body_order`, optional physical `max_range`, and custom admissibility conditions. These criteria are properties of the model space and are independent of any solver that may or may not support the resulting Hamiltonian.

`admissibility_report` returns an `AdmissibilityReport` composed of named `AdmissibilityCheck` objects. A report distinguishes rejection from incomplete evaluation: `report.complete` is false if a declared condition cannot be evaluated, while `report.admissible` is true only when the report is complete and every check passes. `admissibility_reasons` exposes the corresponding rejection or incompleteness messages. Custom `AbstractAdmissibilityCondition` implementations may specialize `admissibility_check`; implementations that only define the existing `admissible(condition, candidate)` protocol remain supported.

Every candidate is checked against the declared operator algebra, including primitive-family and index bounds. `max_body_order` uses the existing algebra-aware `body_order` definition. A declared `max_range` is never inferred from operator or site indices: the specification must provide an explicit `range_evaluator(candidate)` that returns the physical interaction range. If `max_range` is present without such an evaluator, the range check is `missing`, the report is incomplete, and the candidate is not certified admissible.

`HamiltonianBasis` stores the operator basis associated with a model class, and `HamiltonianSpace <: AbstractModelSpace` ties that basis to its `GenerativeSpecification`. Direct `HamiltonianSpace` construction remains declarative so externally generated bases can be inspected even when they are invalid or incompletely certified; `admissibility_report(space)` audits every basis direction. By contrast, `generate_hamiltonian_space` now requires every generated basis direction to have a complete passing admissibility report before it returns the space. This audit does not yet claim Hermiticity, linear independence, symmetry completeness, or universal-generator completeness beyond the evidence recorded by the generator.

`admissibility_report(space, candidate)` also checks membership in the real span represented by the current `HamiltonianBasis`. Batch 8 supplies canonical algebraic linear coordinates, so algebraically equivalent numeric expressions can now be certified against the recorded basis span. Expressions whose coefficients cannot be reduced to numeric linear coordinates remain explicitly unresolved rather than being guessed.

`hamiltonian_from_coefficients` converts a real coefficient vector into a symbolic Hamiltonian in the chosen basis. Because a user-defined admissibility predicate need not be closed under linear combination, auditing every basis direction does not imply that every coefficient combination is valid. `instantiate_model` therefore validates the resulting `ManyBodyModel` against the attached model space, and `with_parameters(...; rebuild=true)` revalidates the rebuilt Hamiltonian. `admissibility_report(model)` uses the model's attached `HamiltonianSpace` or `GenerativeSpecification`; a model without either cannot make a package-level physical-admissibility claim. This layer is therefore the authoritative Julia boundary for the currently declared and evaluable admissibility constraints, not a replacement for the later Hermiticity, symmetry, linear-independence, and completeness certificate, and not a statement that a numerical backend can solve the model.

## Constraint-bearing parameter schemas

Batch 9 promotes the existing `ParameterSpace` into the explicit real-coupling schema associated with a Hamiltonian family. `ParameterSpec` still describes one scalar coefficient, including bounds, units, a reference value, prior metadata, and the legacy `fixed` flag. The key invariant is

$$\mathfrak G+\Theta(\mathfrak G)+\boldsymbol\theta\longrightarrow H_{\mathfrak G}(\boldsymbol\theta),$$

where the Batch 8 structure $\mathfrak G$ defines the Hamiltonian topology, `ParameterSpace` represents the admissible coupling domain $\Theta(\mathfrak G)$, and `ParameterPoint` represents one concrete real coefficient vector $\boldsymbol\theta$. Parameter constraints do not decide whether the Hamiltonian structure is physically admissible and do not select a numerical solver.

`FixedParameterConstraint`, `EqualParameterConstraint`, and `LinearParameterConstraint` express exact affine equality restrictions on named couplings. A fixed `ParameterSpec` is enforced as an implicit fixed-value constraint, and a degenerate bound `lower == upper` has the same effect. Ordinary lower/upper bounds remain inequalities. `parameter_constraint_report` evaluates bounds and equality constraints without constructing a model, while `valid_parameter_point` provides the Boolean form. Invalid `ParameterPoint` construction raises an error instead of creating an out-of-schema coefficient vector.

Every `ParameterSpace` attached to a `HamiltonianSpace` also carries deterministic `ParameterBinding` objects. A binding associates a parameter name and basis index with the Batch 8 `StructuralSignature` of that basis direction. The binding therefore identifies what a coupling multiplies without folding the coupling value into structural identity. A basis direction may itself contain several canonical operator terms, so the binding targets the complete basis-direction structure rather than assuming one scalar parameter corresponds to one graph edge.

The affine equality constraints can be exported as

$$A\boldsymbol\theta=\mathbf b$$

through `parameter_constraint_matrix`. `parameter_dof` reports $N_\theta-\mathrm{rank}(A)$; inequality bounds do not reduce this dimension unless they collapse to an equality. `free_parameter_indices` retains its pre-Batch-9 meaning of parameters not explicitly marked `fixed`, so constraint-aware inverse code should use `parameter_constraint_matrix` or `parameter_dof` when shared or linear relations matter.

Exact representation transformations retain the parameter names, values, and equality constraints while `rebind_parameters` regenerates each `ParameterBinding` against the transformed Hamiltonian basis. Consequently the same physical coupling schema can survive an exact change of representation without incorrectly claiming that the source and target operator structures are identical.

`with_parameters` treats the schema already attached to a `ManyBodyModel` as authoritative. Values from a foreign `ParameterPoint` are rebound and validated against the model's own schema before the Hamiltonian is rebuilt, preventing a caller from bypassing fixed, shared, or linear coupling constraints by constructing a looser parameter space elsewhere. The parameter layer remains optimizer-independent: posterior sampling, optimization, learned priors, and model comparison belong to later inverse machinery.

## `ManyBodyModel`

`ManyBodyModel` is the primary package-level physical model object. It stores a `Representation`, symbolic Hamiltonian, parameter information, observable registry, provenance, optional originating specification, model-space information, and parameterization metadata.

A useful conceptual form is

$$\mathfrak M=(\mathfrak R,\hat H,\boldsymbol\theta,\mathcal O,\text{provenance}).$$

`instantiate_model` builds a `ManyBodyModel` from a `HamiltonianSpace` and a parameter point or coefficient vector. Generated-model construction and rebuilt parameter points are checked against the attached physical specification before they are returned. `with_provenance` appends construction history, and `remap_model` is used by representation transformations to create a mathematically related model while retaining traceability.

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

### Canonical Hamiltonian structures

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

### Generative model spaces

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

### Parameters and models

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
