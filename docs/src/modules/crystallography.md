```@meta
CurrentModule = Phundamental.Crystallography
```

# Crystallography

`Phundamental.Crystallography` connects a crystal structure to symmetry-aware operator generation. It discovers crystallographic symmetry through Spglib, converts backend data into Phundamental-owned types, defines how symbolic operators transform under those symmetries, and constructs linearly independent Hermitian Hamiltonian bases in the crystallographic fixed space.

The layer is deliberately separated from [Models](models.md). `Models.CrystalStructure` describes simulation geometry and neighbor construction, whereas `Crystallography.CrystalStructure` is the fractional-coordinate structure used by symmetry discovery and invariant generation. At the package root, `CrystalStructure` refers to the crystallographic type; the model-geometry type is available as `Models.CrystalStructure` and through the root alias `LatticeCrystalStructure`.

## Crystallographic structures

`Crystallography.CrystalStructure` stores a $3\times3$ lattice matrix whose columns are lattice vectors, fractional site positions, integer species identifiers, and unique symbolic site labels. Fractional positions may be supplied as either a $3\times N$ or $N\times3$ array.

A finite symmetry operation is represented by `SpaceGroupOperation` as

$$\mathbf x\mapsto W\mathbf x+\mathbf w,$$

where $W$ is an integer $3\times3$ rotation in fractional coordinates and $\mathbf w$ is a fractional translation. The operation may additionally carry `time_reversal=true`.

`wrap_fractional` maps fractional coordinates back into the unit interval. `fractional_displacement` returns the minimum-image fractional displacement, which is used when matching transformed sites to the supplied crystal.

`site_permutation` determines the unique site permutation induced by a symmetry operation while preserving species identity. Failure to find exactly one matching target site is treated as an error because an ambiguous permutation invalidates subsequent operator transformations.

## Backend-neutral symmetry datasets

`CrystallographicDataset` is a Phundamental-owned record of a discovered space group. It stores the international and Hall identifiers, point-group symbol, symmetry operations, Wyckoff/site-symmetry data, equivalent-atom classes, crystallographic orbits, and backend metadata.

The package does not expose Spglib-native result types as its public symmetry representation. This boundary makes the rest of the package independent of backend object layouts and lets a discovered dataset be persisted, inspected, or supplied directly to a generative specification.

`CrystallographyOptions` controls tolerances. In v0.0.7 the defaults are `symprec=1e-5`, `site_tolerance=1e-7`, and `linear_tolerance=1e-10`. These parameters have different roles: `symprec` is passed to symmetry discovery, `site_tolerance` controls site matching under operations, and `linear_tolerance` controls rank decisions when an invariant basis is reduced to linear independence.

## Spglib interface

`spglib_cell` converts a Phundamental crystallographic structure to the cell representation required by Spglib. `discover_symmetry` invokes the backend and returns a detached `CrystallographicDataset`.

Because the returned dataset is backend-neutral, downstream functions such as Reynolds projection and invariant generation operate on Phundamental types even when Spglib performed the initial discovery.

A minimal discovery workflow is

```julia
using .Phundamental

structure = Crystallography.CrystalStructure(Matrix{Float64}(I, 3, 3), [0.0 0.0 0.0], [1])
dataset = discover_symmetry(structure)

println(dataset.spacegroup_number)
println(dataset.international_symbol)
```

## Polar and axial rotations

`cartesian_rotation` converts a fractional space-group rotation into the Cartesian transformation appropriate to polar vectors. Spin and magnetic moments are axial vectors, so their spatial rotation is instead

$$R_{\mathrm{axial}}=\det(R)R.$$

`axial_rotation` implements this distinction. If a space-group operation also includes time reversal, spin-like axial quantities acquire the additional sign reversal required by the symmetry action.

This distinction matters for Hamiltonian generation: transforming a spin operator as if it were a polar displacement generally gives the wrong invariant space for improper rotations.

## Symbolic symmetry actions

`transform_primitive` applies a crystallographic operation to a primitive symbolic operator. `transform_operator` extends the action recursively to complete operator expressions. `standard_symmetry_action` supplies the standard package action for the supported operator families.

Spin operators use the axial-vector transformation. Site-labelled scalar occupation operators are carried by the site permutation. Coordinate and momentum primitives require vector-component information that is not encoded by the bare mode index, so applications that need those fields should provide an appropriate custom symmetry action rather than assuming that a scalar site permutation is sufficient.

A `GenerativeSpecification` stores the symmetry action used to generate a Hamiltonian class. This keeps the group action itself explicit and makes nonstandard order parameters or internal degrees of freedom extensible without modifying the core invariant-generation algorithm.

## Candidate operator generation

`enumerate_operator_candidates` constructs a finite candidate set from the primitive generators of an operator algebra. Candidate enumeration is controlled by `max_body_order`, an optional operator-degree cutoff, and whether the identity is included.

Automatic enumeration is intentionally finite. It does not claim to enumerate every possible Hamiltonian term of arbitrary range or degree. The generated model class is complete only relative to the supplied or automatically generated candidate space and the explicit admissibility restrictions.

## Reynolds projection

For a finite symmetry group $G$, `reynolds_project` forms the fixed-space projection

$$\mathcal P_G[O]=\frac{1}{|G|}\sum_{g\in G}g\cdot O.$$

Each transformed term is canonicalized in the model algebra. The projected operator is subsequently decomposed into Hermitian components, and linearly dependent components are removed using the configured numerical tolerance.

This procedure turns a candidate operator list into a symmetry-invariant Hermitian basis without assigning coupling constants. Couplings belong to the parameterization of the resulting Hamiltonian space rather than to the symmetry projector.

## Invariant Hamiltonian bases

`generate_invariant_basis` accepts a `GenerativeSpecification`, crystallographic structure, and candidate operators. It applies body-order and admissibility restrictions, projects accepted candidates through the space group, forms Hermitian components, and extracts a linearly independent basis.

The result is a `HamiltonianBasis` whose metadata records the symmetry backend, candidate count, rejected candidates, invariant dimension, tolerances, and the key completeness statement

```text
:completeness_scope => :supplied_candidate_space
```

This metadata is important: symmetry projection proves invariance of the generated basis but does not prove that the initial candidate set contained every physically relevant interaction.

`CrystallographicGenerator` packages a structure, candidates, options, and optional labels so that it can be supplied directly to the generic `generate_hamiltonian_space` machinery in Foundation.

## High-level crystallographic model-space construction

`crystallographic_specification` discovers the symmetry of a crystal and constructs a `GenerativeSpecification` using the resulting crystallographic orbits and a supplied operator algebra.

`generate_crystallographic_hamiltonian_space` is the higher-level route. If no candidate list is supplied, it performs finite automatic candidate enumeration before fixed-space projection; if candidates are supplied explicitly, those candidates define the completeness scope.

A typical structure is

```julia
spec = crystallographic_specification(structure, local_spaces, SpinAlgebra(nsites), max_body_order=2)

space = generate_crystallographic_hamiltonian_space(spec, structure)
```

Applications that require a specialized finite interaction family should normally supply their own candidate set rather than increasing automatic enumeration without a physical cutoff.

## Invariants and failure modes

Crystallographic generation assumes that the supplied structure and the discovered operations are mutually consistent. Typical errors include non-bijective site mappings, species mismatches, malformed lattice/position arrays, a candidate set that projects entirely to zero, or an invariant basis that becomes empty after admissibility and Hermiticity constraints.

The numerical rank tolerance in invariant extraction is not a physical tolerance. It controls when coefficient vectors are treated as linearly dependent. Users generating nearly redundant bases should test stability against reasonable variations of `linear_tolerance`.

## Related modules

- [Core / Foundation](core.md) provides `GenerativeSpecification`, `HamiltonianBasis`, `HamiltonianSpace`, and symbolic operators.
- [Algebra](algebra.md) canonicalizes transformed and projected expressions.
- [Models](models.md) provides physical geometry and Hamiltonian specifications after an invariant operator family has been identified.
- [Transformations](transformations.md) changes mathematical representations after the symmetry-admissible model has been constructed.

## API reference

### Structures and discovery

```@docs
CrystalStructure
SpaceGroupOperation
CrystallographicDataset
CrystallographyOptions
spglib_cell
discover_symmetry
```

### Symmetry action

```@docs
transform_primitive
```

### Invariant generation

```@docs
CrystallographicGenerator
```

## Additional exported utilities

The remaining public crystallography utilities are `wrap_fractional`, `fractional_displacement`, `cartesian_rotation`, `axial_rotation`, `site_permutation`, `transform_operator`, `standard_symmetry_action`, `enumerate_operator_candidates`, `reynolds_project`, `generate_invariant_basis`, `crystallographic_specification`, and `generate_crystallographic_hamiltonian_space`.
