```@meta
CurrentModule = Phundamental.Transformations
```

# Transformations

`Phundamental.Transformations` represents changes of mathematical description as first-class objects. A transformation acts on a complete `ManyBodyModel`: it constructs a target `Representation`, maps the Hamiltonian and registered observables, canonicalizes the result in the target algebra, records provenance, and returns mathematical metadata describing what relation has been implemented.

The module makes a strict distinction between an exact representation map and an approximation. A numerical backend is a third, separate concept documented in [Solvers](solvers.md).

$$\text{representation transformation}\neq\text{numerical backend}\neq\text{physical approximation}.$$

## Common transformation interface

All representation transformations derive from `AbstractTransformation`. The shared interface includes `transformation_name`, `certificate`, `applicable`, `target_representation`, `transform_generator`, `transform_operator`, `transform_hamiltonian`, `transform_observable`, and `transform`.

Primitive transformation rules are defined through `transform_generator`. `transform_operator` recursively extends those rules by linearity and multiplicativity to sums, products, and scalar multiples. The mapped expression is then canonicalized in the target algebra before it becomes part of the target model.

The principal user operation is

```julia
mapped = transform(T, model)
new_model = mapped.model
```

A transformation is therefore more than an operator-substitution helper. It is responsible for the target state space, physical constraints, algebra, basis, ordering, truncation transport, and validation hooks required to interpret the transformed Hamiltonian correctly.

## Transformation certificates

Every standard transformation provides a `TransformationCertificate`. The certificate records whether the implementation is exact and invertible, its primary mathematical relation, locality, whether constraints are required, whether a gauge redundancy is introduced, whether the operation is approximate, and explanatory notes.

The `RelationType` values are:

| Relation | Meaning in Phundamental |
| --- | --- |
| `UnitaryEquivalence` | Exact unitary change of representation. |
| `CanonicalMap` | Exact map preserving the relevant canonical algebraic structure. |
| `AlgebraIsomorphism` | Exact isomorphism between operator algebras. |
| `ConstrainedEmbedding` | Exact physical theory embedded in a larger ambient space with explicit constraints. |
| `SimilarityEquivalence` | Exact similarity relation that need not be unitary and may require biorthogonal numerics. |
| `GaugeRedundantEmbedding` | Enlarged representation with an explicit gauge redundancy or constraint. |
| `CompositeRelation` | Relation induced by composing multiple transformations. |

`certificate(T).invertible` is a mathematical statement about the transformation class, while `has_inverse(T)` reports whether an executable inverse has actually been implemented in software. These values should not be conflated.

## Composition

`compose(T1, T2, ...)` constructs a `CompositeTransformation`. Its certificate combines the properties of the component transformations: exactness and invertibility require all components to have those properties, while nonlocality, constraints, gauge redundancy, or approximation metadata propagate if any component introduces them.

Composition is useful when a calculation naturally proceeds through several representation layers. It also keeps provenance explicit rather than manually rewriting a Hamiltonian through unrelated helper functions.

## Exact transformation families

### Fourier transformation

`FourierTransformation` maps bosonic or fermionic modes between real-space and momentum-space representations through a unitary matrix. Its certificate identifies a `UnitaryEquivalence` when the supplied transformation matrix defines the required exact square unitary map.

A Fourier transformation changes the mathematical basis but does not itself approximate the model. Momentum truncation, interpolation, or coarse sampling would be separate numerical choices.

### Bogoliubov transformation

`BogoliubovTransformation` implements a canonical linear transformation mixing annihilation and creation operators. The implementation validates the bosonic or fermionic canonical conditions appropriate to the supplied coefficients.

The map is a `CanonicalMap`: it is exact when its canonical constraints are satisfied. Solving a quadratic model after this transformation is not equivalent to applying a quadratic approximation to an interacting model; the latter would need to be represented separately.

### Particle--hole transformation

`ParticleHoleTransformation` implements a local exact fermionic particle--hole map. In the basic convention, annihilation and creation operators are exchanged and occupation transforms as $n\mapsto1-n$. The standard map is self-inverse.

Particle--hole symmetry is used independently by the [Embedding](embedding.md) layer to constrain a half-filled ED bath. The bath constraint is a physical symmetry restriction on fit parameters, not an invocation of `ParticleHoleTransformation` on the many-body Hamiltonian.

### Majorana transformation

`MajoranaTransformation` provides the exact algebra isomorphism between complex fermions and pairs of Majorana generators. In the package convention,

$$c_m=\frac{\gamma_{2m-1}+i\gamma_{2m}}{2}.$$

The transformation changes the operator algebra while preserving the exact finite fermionic theory.

### Jordan--Wigner transformation

`JordanWigner` maps an ordered spin-$1/2$ chain or compatible ordered spin structure to spinless fermions. The fermionic generator acquires the nonlocal parity string required to reproduce spin commutation relations at distinct sites.

Ordering is therefore part of the transformation definition. The certificate records the nonlocal character of the map, and the target representation preserves the ordering convention required to interpret the strings.

### Holstein--Primakoff transformation

`HolsteinPrimakoff` maps a spin-$S$ degree of freedom into a bosonic representation with the exact occupancy restriction $0\leq n\leq2S$. Exact square-root factors are retained symbolically through `SqrtNumberFactor` rather than expanded automatically.

The exact Holstein--Primakoff map is a constrained embedding, not linear spin-wave theory. Linear spin-wave theory additionally truncates the transformed Hamiltonian to a quadratic approximation and is exposed through the solver/approximation machinery rather than being hidden inside this transformation.

### Dyson--Maleev transformation

`DysonMaleev` implements a nonunitary spin-to-boson representation related by similarity rather than a unitary map. The resulting representation may require biorthogonal numerical treatment. This distinction is carried by the certificate and propagated into numerical capability analysis.

### Schwinger-boson transformation

`SchwingerBoson` embeds spin degrees of freedom into a bosonic representation with an exact occupation constraint and gauge redundancy. The larger bosonic Fock space is not itself the physical spin space; the target representation records the physical restriction explicitly.

### Abrikosov-fermion transformation

`AbrikosovFermion` embeds spin degrees of freedom into auxiliary fermions with the appropriate exact occupancy constraint and gauge redundancy. As with Schwinger bosons, the enlarged ambient Fock space and the physical sector remain separate objects.

### Slave-particle transformation

`SlaveParticleTransformation` is a generic transformation wrapper for user-defined enlarged representations and generator maps. It is appropriate when the transformation follows the same representation-aware protocol but is not one of the package's named canonical constructions.

Users are responsible for supplying a target representation and mapping logic whose certificate accurately describes constraints, gauge redundancy, and invertibility.

### Coordinate--ladder transformation

`CoordinateLadder` converts canonical coordinate/momentum variables to bosonic ladder operators using supplied mass, frequency, and $\hbar$ information. This is the standard route for moving a coordinate representation into a finite bosonic numerical treatment after an explicit truncation is chosen.

The [Solvers](solvers.md) capability layer intentionally does not silently discretize bare coordinate/momentum representations. A coordinate model must be mapped or discretized explicitly.

### Bosonic displacement

`BosonicDisplacement` implements an exact unitary displacement. `BosonicDisplacementFactor` represents the atomic operator

$$D(\alpha)=\exp(\alpha b^\dagger-\alpha^*b).$$

The v0.0.7 active convention maps an operator as $O\mapsto D(\alpha)OD(\alpha)^\dagger$. The inverse displacement is obtained by negating the displacement amplitude.

### Lang--Firsov transformation

`LangFirsov` implements an exact occupation-conditioned phonon displacement for coupled fermion--boson models. It is useful for moving electron--phonon coupling into transformed hopping and interaction terms without declaring the transformed Hamiltonian to be approximate.

### Spin-polaron transformation

`SpinPolaron` implements the analogous exact longitudinal spin-conditioned bosonic displacement used for spin--phonon coupling. It preserves the exact transformation semantics while producing displacement-dressed spin operators or couplings in the target representation.

## Exact symbolic transformation atoms

Some exact transformations generate operator factors that should not be prematurely expanded. `SqrtNumberFactor` stores exact square-root number-operator factors, while `BosonicDisplacementFactor` stores exact displacement operators. The Algebra layer treats these atoms as noncommutative objects so their mathematical meaning survives canonicalization.

## Registry

`TRANSFORMATION_REGISTRY` stores named transformation constructors. `register_transformation!` adds a constructor, `transformation_constructor` retrieves it, `make_transformation` constructs a registered transformation, and `available_transformations` reports registered names.

The registry is intended for discoverability and configuration-driven workflows. It does not replace the type-specific certificate and applicability checks performed by the transformation itself.

## Approximations

`AbstractApproximation` is intentionally separate from `AbstractTransformation`. `ApproximationCertificate` records assumptions, validity conditions, approximation order, an optional error model, and notes. `ApproximationResult` retains the approximated model together with discarded information and validation metadata.

`ModelSpaceProjection` implements a generic projection onto a target `HamiltonianSpace`. The caller supplies a projector that returns either the retained Hamiltonian or the pair $(H_\parallel,H_\perp)$. An optional error metric can quantify the discarded component.

`transformation_closure_defect` measures whether a mapped operator basis closes in a target basis when supplied with an independent-rank function. This is useful when assessing whether a proposed transformation preserves a chosen finite model class even if the full operator transformation is exact.

!!! warning
    An exact representation transformation may map a simple Hamiltonian outside a restricted target model class. Projecting it back into that model class is an approximation and should be represented as such.

## Representation graph

`RepresentationGraph` stores named `Representation` nodes and `TransformationEdge` objects. An edge may contain either an exact transformation or an approximation, along with forward and optional inverse parameter maps.

`transformation_path` finds an executable breadth-first path between representations. Reverse traversal is only available when the edge is invertible, the transformation has an implemented inverse, and an inverse parameter map is available.

`representation_class` returns the exact mathematical orbit reachable through invertible exact edges. `transform_along` executes a graph path and transports parameter metadata as each edge is applied.

This graph is useful for treating representation equivalence as a structured object rather than a collection of ad hoc conversion functions.

## Validation and numerical realization

Transformation validation occurs at several levels. Parameter and applicability checks run before mapping. The target representation is constructed explicitly. The Hamiltonian and observables are canonicalized in the target algebra. The test suite then compares exact transformed Hamiltonians and spectra against independently realized source and target calculations where applicable.

Fixed numerical cutoffs can break exact equality even when the mathematical transformation is exact. Such leakage must be interpreted as a numerical-truncation effect rather than silently changing the transformation certificate. The v0.0.7 bosonic validation suite therefore tracks cutoff convergence and leakage separately from exact symbolic transformation identities.

## Related modules

- [Core / Foundation](core.md) defines representations, constraints, model spaces, and `ManyBodyModel`.
- [Algebra](algebra.md) canonicalizes mapped expressions.
- [Solvers](solvers.md) determines whether the target representation is numerically realizable and chooses a backend without changing the representation.
- [Representation Transformations](../workflows/representation-transformations.md) gives end-to-end examples and interpretation guidance.

## API reference

### Transformation protocol

```@docs
AbstractTransformation
TransformationCertificate
has_inverse
transform_operator
transform_hamiltonian
transform_observable
SqrtNumberFactor
BosonicDisplacementFactor
```

### Registry

```@docs
register_transformation!
transformation_constructor
make_transformation
available_transformations
```

### Approximations

```@docs
ApproximationCertificate
ApproximationResult
ModelSpaceProjection
transformation_closure_defect
```

### Representation graph

```@docs
TransformationEdge
transformation_path
representation_class
```

## Additional exported transformation types and utilities

The concrete transformation types are `FourierTransformation`, `BogoliubovTransformation`, `ParticleHoleTransformation`, `MajoranaTransformation`, `JordanWigner`, `HolsteinPrimakoff`, `DysonMaleev`, `SchwingerBoson`, `AbrikosovFermion`, `SlaveParticleTransformation`, `CoordinateLadder`, `BosonicDisplacement`, `LangFirsov`, and `SpinPolaron`. Additional public protocol functions are `transformation_name`, `certificate`, `applicable`, `target_representation`, `transform_generator`, `transform`, `compose`, `CompositeTransformation`, `inverse`, the approximation protocol functions, `TRANSFORMATION_REGISTRY`, and the graph mutation/traversal utilities.
