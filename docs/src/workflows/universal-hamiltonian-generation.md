```@meta
CurrentModule = Phundamental
```

# Universal Hamiltonian Generation

Material-aware Hamiltonian generation connects material geometry and crystallographic symmetry to the existing physical model-space, structural, and parameter layers. The generator does not replace those layers; it coordinates them in a single material-aware construction.

The resulting pipeline is

$$\text{material/crystal}\rightarrow\text{crystallographic dataset}\rightarrow\mathcal G\rightarrow\mathcal H(\mathcal G)\rightarrow\mathfrak G\rightarrow\Theta(\mathfrak G).$$

Here `GenerativeSpecification` represents $\mathcal G$, `HamiltonianSpace` represents $\mathcal H(\mathcal G)$, `ModelStructure` represents $\mathfrak G$, and `ParameterSpace` represents the admissible continuous coupling domain $\Theta(\mathfrak G)$.

## Material geometry to crystallography

`material_crystallographic_structure(material)` converts a three-dimensional `Models.CrystalStructure` attached to a `Material` into the backend-neutral `Crystallography.CrystalStructure` used by Spglib and the invariant generator. Species symbols are mapped to deterministic integer classes only for symmetry matching; the original material remains attached to the generation result.

The overload `material_crystallographic_structure(material, cluster)` converts a fully periodic three-dimensional `Supercell`. This route is required when translated sites must be represented as distinct operator indices in the finite generated model space. Open clusters are rejected because treating an open boundary as a periodic Spglib cell would change the physical geometry silently.

The material-only route acts on primitive-cell-indexed degrees of freedom. With the current operator ontology, an intercell interaction whose endpoints must remain distinct requires an explicit periodic supercell large enough to contain those endpoints as separate site indices.

## Material-derived generative specifications

`material_generative_specification` builds a `GenerativeSpecification` from the material or material-plus-supercell domain. By default it discovers crystallographic symmetry with `discover_symmetry`, records the resulting `CrystallographicDataset`, derives one local state space per crystallographic site when that mapping is unambiguous, and uses `standard_symmetry_action`.

The standard symmetry action has an explicit indexing contract: one symbolic spin, boson, fermion, or Majorana index must correspond to one crystallographic site. Multi-orbital, spinful-fermion, vector-component, or other richer internal indexing conventions must provide a custom `symmetry_action` and explicit `local_spaces`; the generator does not guess how crystallographic operations act on hidden internal indices. A Majorana site action exists, but automatic site-local Hilbert-space construction is intentionally not inferred from one Majorana generator per site, so Majorana material specifications must also supply `local_spaces` explicitly.

If `max_range` is declared while the standard site indexing is valid and no custom `range_evaluator` is supplied, `generate_hamiltonian_family` constructs a geometry-aware evaluator from the crystallographic lattice. The range of one canonical operator term is the maximum minimum-image Cartesian separation between its distinct site supports. This supplies the explicit physical geometry required by the declared physical-range admissibility constraint rather than interpreting symbolic site numbers as distances.

## Generating a complete Hamiltonian family object

`generate_hamiltonian_family` invokes the existing crystallographic candidate projection and returns a `HamiltonianGenerationResult` containing the material source, optional periodic supercell, crystallographic structure, crystallographic dataset, physical generative specification, generated Hamiltonian space, canonical model structure, constraint-bearing parameter space, and generation metadata.

A minimal material-aware spin example is

```julia
using Phundamental
using LinearAlgebra

lattice = BravaisLattice(Matrix{Float64}(I, 3, 3))
crystal = LatticeCrystalStructure(lattice, [BasisSite(:A, :A, [0.0, 0.0, 0.0]), BasisSite(:B, :B, [0.4, 0.0, 0.0])])
material = Material("two-site magnet", crystal;
                    species=Dict(:A => AtomicSpecies(:A; mass=1.0), :B => AtomicSpecies(:B; mass=1.0)),
                    basis_properties=[SiteProperties(spin=1//2), SiteProperties(spin=1//2)])

family = generate_hamiltonian_family(material, SpinAlgebra(2); max_body_order=2, max_range=0.6)
```

For controlled candidate spaces, supply `candidates` explicitly. `basis_labels` may name the independent invariant basis directions and therefore the corresponding `ParameterSpace` coefficients; its length must match the final invariant-space dimension. Otherwise the existing `enumerate_operator_candidates` machinery determines the finite candidate set from the algebra and body-order/operator-degree controls. Symmetry projection, Hermitian decomposition, linear-independence filtering, and physical admissibility remain owned by the crystallography and model-space layers. Automatic generation is complete only relative to this finite candidate space and never claims enumeration of arbitrary interaction order or range.

## Parameter schemas and model instantiation

The generator does not introduce a second coupling schema. The generated basis is passed directly to the existing `ParameterSpace` constructor. Bounds, units, fixed flags, priors, and explicit parameter equality constraints may be supplied through the `parameter_*` keywords of `generate_hamiltonian_family`.

A generated family can be instantiated with an explicit representation:

```julia
representation = Representation(:spin, SpinHilbertSpace([1//2, 1//2]), SpinAlgebra(2), SpinProductBasis([1, 2]))
model = instantiate_model(family, representation)
```

The ordinary Foundation validation still applies. The representation must agree with the generated specification algebra, the parameter point must satisfy the `ParameterSpace` schema, and the resulting Hamiltonian must remain inside the generated physical model space. The convenience method adds generation provenance but does not bypass these checks.

## Architectural boundary

Crystallography is upstream of structural identity. `ModelStructure` does not reconstruct a material or space group from a Hamiltonian expression. Instead, material/crystallographic information restricts the invariant Hamiltonian basis before the canonical `ModelStructure` is extracted.

Hamiltonian-family generation performs no solver selection, forward execution, structural search, parameter optimization, posterior inference, or machine learning. Structural search operates on crystallography-constrained neighborhoods, while structural forward closure provides the material-to-forward validation boundary.

## API reference

```@docs; canonical=false
HamiltonianGenerationResult
material_crystallographic_structure
material_generative_specification
generate_hamiltonian_family
```

## Symmetry-complete structural search

Structural search operates downstream of the crystallographic invariant generator. A `StructuralFamily` partitions the **complete invariant basis directions**, rather than the monomials inside their symbolic expansions, into nonoverlapping `StructuralUnit`s. Each unit may contain one direction (the default) or an explicitly grouped set of whole directions. Structural deletion never removes only one symmetry-related monomial from an invariant direction.

```julia
family = structural_family(result)
selection = structural_selection(family, [family.units[1].id])
neighbors = structural_neighbors(selection; max_neighbors=128)
parameter_schema = parameter_subspace(selection, zeros(length(selection.basis_indices)))
parent_coefficients = embed_parameters(selection, zeros(length(selection.basis_indices)))
```

The structural search stays finite and explicit. `enumerate_structures` applies required/forbidden unit and size constraints, checks the exact combinatorial selection count, and raises an error if `max_structures` would be exceeded. `model_space_hierarchy` constructs the finite within-family Hasse graph whose edges add exactly one complete structural unit. The graph changes no crystallographic, range, or body-order assumption in the parent `GenerativeSpecification`.

Structural identity uses coefficient-ratio-aware `InvariantDirectionSignature` coordinates. Two directions related by an overall nonzero **real** multiplier have the same direction signature, while different relative coefficients do not. `StructuralFamilySignature` identifies the physical parent and structural grouping; `StructuralSelectionSignature` is independent of unit-selector order. Parent coefficient-vector order remains authoritative for parameter bindings. These structural signatures are Julia-internal identity values; stable cross-language serialized identifiers are provided by the separate structural-interchange interface.

Structural absence is explicit, rather than inferred from a small numerical coefficient. An omitted direction has **exactly zero** parent coefficient. `embed_parameters` validates the complete parent schema after zero insertion; strict `project_parameters` rejects discarded nonzero coefficients. `parameter_subspace` inherits the original parameter bounds, names, units, fixed flags, priors, and supported equality constraints. A parent equality between one retained and one omitted parameter becomes a fixed-zero child constraint. Unknown custom constraint types are never silently dropped.

A selected structural Hamiltonian subspace remains distinct from an executable numerical representation. Solver compatibility and selected-model forward instantiation are handled separately through structural forward closure.

### Structural-search API

```@docs; canonical=false
InvariantDirectionSignature
StructuralFamilySignature
StructuralUnit
StructuralFamily
StructuralSelectionSignature
StructuralSelection
StructuralMove
StructuralNeighborhood
StructuralCoordinateMap
ModelSpaceHierarchy
structural_family
structural_selection
structural_transition
structural_move
apply_structural_move
structural_neighbors
enumerate_structures
model_space_hierarchy
parameter_subspace
```

## F4 optional group and finite-reference rotational certification

For material-based generation, `certify_group=true` checks the complete list of crystallographic operations modulo lattice translations, including identity, closure, site permutations and metric preservation. This is particularly useful with externally supplied symmetry datasets; it does not change the default handling of intentionally restricted operation lists. The physical-symmetry workflow documents a cubic and screw-axis benchmark against discovered Spglib operations.

For quadratic Cartesian coordinate models intended as finite **open** force-constant references, `rotational_sum_rule=true`, `rotational_scope=:finite_site_hessian` and explicit Cartesian `rotational_positions` augment `acoustic_sum_rule=true` with independent rigid-body rotation constraints. This assertion must not be made for a periodically aliased Hessian: correct periodic Born–Huang constraints require image-resolved force constants, not a finite-index Hessian alone.
