```@meta
CurrentModule = Phundamental.Models
```

# Models

`Phundamental.Models` defines concrete physical geometry, materials, interactions, and Hamiltonian specifications. It converts recognizable many-body model definitions into the representation-aware `ManyBodyModel` objects used by transformations, solvers, observables, thermodynamics, Green functions, and embedding calculations.

The module deliberately separates a model specification from its instantiated symbolic model. An `AbstractHamiltonianSpec` such as `XXZModel`, `HubbardModel`, or `AndersonImpurityModel` stores physical parameters and construction rules; the applicable `build_model` method combines that specification with the required geometry or impurity data to construct a `ManyBodyModel` with representation, Hamiltonian, observables, and provenance.

$$\text{geometry/material/specification}\xrightarrow{\texttt{build\_model}}\texttt{ManyBodyModel}.$$

## Geometry

### Bravais lattices

`BravaisLattice` stores the direct-lattice basis used by periodic model geometry. `lattice_dimension`, `direct_matrix`, `reciprocal_matrix`, and `cell_measure` expose its dimensional and metric information.

`fractional_to_cartesian` and `cartesian_to_fractional` convert coordinates using the lattice basis. Reciprocal vectors may be constructed from the lattice with `reciprocal_vector` and converted back to reciprocal coordinates with `reciprocal_coordinates`.

### Basis sites and crystal structures

`BasisSite` defines a site inside one primitive cell. `Models.CrystalStructure` combines a Bravais lattice with basis sites for geometry, neighbor generation, and material attachment.

This type is distinct from `Crystallography.CrystalStructure`. The crystallography type stores a backend-neutral fractional structure for symmetry discovery; the Models type is a simulation geometry. At the root namespace the crystallographic type owns the name `CrystalStructure`, so use `Models.CrystalStructure` or the root alias `LatticeCrystalStructure` when constructing model geometry.

`CrystalSite` represents a specific translated basis site and `Supercell` constructs finite periodic or cluster geometries. `nsites`, `site_position`, `site_positions`, and `site_index` provide indexed access.

### Bonds and neighbor shells

`Bond` stores a pair of sites together with the displacement information required by lattice Hamiltonians. `neighbor_bonds` generates bonds according to geometric cutoffs, while `bonds_in_shell` and `coordination_numbers` organize the resulting graph by neighbor shell and site coordination.

Hamiltonian specifications that accept a `neighbor_cutoff` use this geometry rather than hard-coding a particular lattice topology. The cutoff is therefore part of model construction and should be reported with any result that depends on neighbor enumeration.

## Local frames

`LocalFrame` defines a site-dependent orthonormal frame. `local_x`, `local_y`, and `local_z` expose its axes; `global_vector`/`local_vector` transform vector components; `global_tensor`/`local_tensor` transform rank-two tensors.

Pair couplings require independent frames at the two sites. `global_pair_tensor` and `local_pair_tensor` perform that two-frame transformation explicitly.

`LocalFrameField` associates frames with lattice sites. `frame_moment_vectors` builds site moment directions from the field, and `validate_local_frames` checks normalization and orthogonality.

Local frames are particularly important for anisotropic magnets and local-axis Ising systems because a scalar coupling in one local frame may correspond to a nontrivial tensor interaction in global Cartesian coordinates.

## Long-range interactions

`AbstractLongRangeInteraction` is the common abstraction for interactions whose numerical evaluation requires a summation strategy. v0.0.8 includes `CoulombInteraction` and `DipolarInteraction`.

`AbstractInteractionSummation` separates the physical interaction from the summation method. `RealSpaceCutoff` truncates in real space; `EwaldSummation` performs a periodic Ewald decomposition. This separation lets the same physical interaction be evaluated with multiple numerical controls.

`interaction_matrix` returns scalar pair interactions and `interaction_tensor` returns tensor-valued interactions where appropriate. `interaction_energy` evaluates an interaction for a supplied configuration.

For local-axis magnetic systems, `project_to_local_frames` projects a global interaction tensor into site frames. `ising_coupling_matrix` and `ising_zeeman_field` construct the scalar quantities needed by an effective Ising description, and `dipolar_hamiltonian` constructs the corresponding symbolic Hamiltonian.

`validate_long_range_result` checks consistency of a computed interaction result. `validate_ewald_alpha_independence` checks that a converged Ewald calculation is stable against the arbitrary Ewald splitting parameter within the requested numerical tolerance.

## Harmonic interactions and force constants

Harmonic lattice dynamics are represented in real space before they are handed to the phonon solver. `AbstractHarmonicInteraction` is the common interface for interaction-level contributions to force constants.

`HarmonicBondInteraction` describes bond-stretching contributions. `HarmonicVectorCoordinate`, `HarmonicBendingInteraction`, and `HarmonicAngleInteraction` describe vector or angular internal coordinates and their harmonic penalties.

`harmonic_coordinate_vector` evaluates the internal coordinate associated with a harmonic term. `bending_force_constant_matrix` constructs the Hessian contribution for bending interactions, and `force_constants` assembles force-constant contributions from the chosen interaction set.

`ForceConstantBlock` stores one periodic real-space force-constant block and its cell displacement. `RealSpaceForceConstants` collects those blocks. `project_force_constants` projects a set into a constrained subspace, while `enforce_acoustic_sum_rule` imposes the translational acoustic sum rule when appropriate.

!!! note
    Enforcing an acoustic sum rule is a model-processing operation. It should be recorded in provenance rather than treated as an invisible numerical cleanup.

## Materials and scattering properties

`AtomicSpecies` is the package representation of an atomic species entry. `lookup_species` accesses the packaged species information used by model/material workflows.

Magnetic form factors derive from `AbstractMagneticFormFactor`. `ConstantFormFactor` and `GaussianFormFactor` provide simple concrete models, while `form_factor` evaluates the selected form-factor object.

`DebyeWallerTensor` stores anisotropic mean-square displacement information. `debye_waller_factor` evaluates the attenuation associated with a momentum transfer.

`SiteProperties` stores per-site material information such as mass, spin, $g$ tensor, magnetic form factor, and coherent scattering length. `Material` associates species or basis-site properties with a structure. Accessors include `site_mass`, `site_spin`, `site_g_tensor`, `site_form_factor`, and `site_scattering_length`.

`mean_square_displacement` and `phonon_debye_waller_tensors` connect a harmonic phonon solution to thermal displacement and Debye--Waller information used by neutron-scattering observables.

## Hamiltonian specifications

All standard model constructors derive from `AbstractHamiltonianSpec`. Their job is to define a physical Hamiltonian family and the representation required to express it; the result is built with `build_model`.

### Spin models

`HeisenbergModel` builds an isotropic exchange model. `XXZModel` provides anisotropic exchange with separate transverse and longitudinal couplings and may construct neighbor bonds from geometry.

The resulting model remains in the spin representation unless the user explicitly applies a transformation such as `JordanWigner`, `HolsteinPrimakoff`, or another supported map.

### Fermion models

`HubbardModel` constructs the standard lattice fermion Hamiltonian with hopping and on-site interaction. The model stores the fermionic representation and parameter metadata required by exact diagonalization, optimized fermion backends, cluster perturbation theory, or other fermionic workflows.

### Anderson impurity model

`AndersonImpurityModel` constructs the finite single-orbital impurity Hamiltonian used by the v0.0.8 ED-DMFT implementation. The bath is spin degenerate: each finite bath orbital is represented for both spin components, and `impurity_bath_sites` exposes the associated impurity/bath indexing.

The [Embedding](embedding.md) layer normally constructs this model internally after fitting a `DiscreteBath`, but the specification is public so the same finite impurity problem can be built and solved directly.

### Harmonic phonons

`HarmonicPhononModel` stores periodic harmonic force constants together with masses and unit conventions. `PhononUnitConvention` makes the frequency/energy convention explicit rather than leaving dimensional factors implicit in the dynamical matrix.

`harmonic_model` is the higher-level constructor that combines geometry, material masses, and force constants into a `ManyBodyModel` suitable for `HarmonicPhononSolver`.

`LocalPhononModel` represents independent or locally coupled bosonic modes when a periodic force-constant description is unnecessary.

### Hybrid models

`HolsteinModel` combines fermions with local phonons through density--displacement coupling. `LongitudinalSpinBosonModel` provides the analogous longitudinal spin--boson coupling used by the spin-polaron transformation and hybrid spin/phonon studies.

Hybrid models use composite state spaces and algebras so that factor identity is preserved rather than flattening all primitive operators into one global mode index.

## Material-aware universal Hamiltonian generation

The material-aware API provides the orchestration layer for the generalized Hamiltonian-space machinery. `material_crystallographic_structure` converts a three-dimensional material geometry, or an explicitly periodic three-dimensional supercell, into the backend-neutral crystallographic structure used by `Crystallography`. `material_generative_specification` then combines that geometry with an operator algebra, discovered or supplied space-group data, local state spaces, the symmetry action, and physical admissibility restrictions.

`generate_hamiltonian_family` coordinates the existing crystallographic invariant generator with the physical model-space, structural, and parameter layers and returns a `HamiltonianGenerationResult`. The result retains the material source, crystallographic dataset, `GenerativeSpecification`, generated `HamiltonianSpace`, canonical `ModelStructure`, and constraint-bearing `ParameterSpace` separately rather than flattening them into a second model hierarchy. A curated candidate set may provide `basis_labels` for the final independent invariant directions, giving the resulting parameters stable physical names instead of default `Phi_i` labels.

The standard crystallographic action requires one symbolic site or mode index per crystallographic site. Multi-orbital, spinful-fermion, coordinate-component, or other richer internal-index conventions must supply the corresponding `symmetry_action` and `local_spaces` explicitly. When that site mapping is explicit, a declared `max_range` can be evaluated from minimum-image Cartesian distances in the material geometry instead of assigning geometric meaning to symbolic indices.

The material-only route describes primitive-cell-indexed degrees of freedom. Use the periodic-supercell overload when translated sites must appear as distinct finite operator supports. See [Universal Hamiltonian Generation](../workflows/universal-hamiltonian-generation.md) for the complete workflow and boundary conditions.

## Model construction and validation

`build_model` is the common constructor from an `AbstractHamiltonianSpec`. The model-specific implementation chooses the representation, creates the symbolic Hamiltonian, attaches standard observables when defined, and records the originating specification.

`model_positions`, `model_cluster`, and `mode_index` provide shared geometry/index helpers for constructed models.

Validation is layered. `validate_lattice` checks lattice geometry, `validate_bonds` checks bond data, and `validate_material_model` checks material/geometry consistency. Force-constant validation is represented by `ForceConstantValidationResult` and `validate_force_constants`.

A model passing structural validation does not automatically imply that a chosen numerical solver is compatible. Computational compatibility is checked separately through [Solvers](solvers.md) and its numerical-capability system; physical admissibility remains a model-space property.

## Related modules

- [Core / Foundation](core.md) provides `ManyBodyModel`, representations, symbolic operators, and parameter spaces.
- [Crystallography](crystallography.md) generates symmetry-admissible Hamiltonian bases from crystal symmetry.
- [Transformations](transformations.md) changes the representation of a constructed model.
- [Solvers](solvers.md) supplies generic and optimized numerical realizations.
- [Observables](observables.md) uses material properties and model geometry to construct experimental response functions.

## API reference

### Geometry and local frames

```@docs
neighbor_bonds
LocalFrame
global_tensor
global_pair_tensor
LocalFrameField
frame_moment_vectors
validate_local_frames
```

### Long-range interactions

```@docs
CoulombInteraction
DipolarInteraction
RealSpaceCutoff
EwaldSummation
project_to_local_frames
ising_coupling_matrix
ising_zeeman_field
dipolar_hamiltonian
validate_long_range_result
validate_ewald_alpha_independence
```

### Harmonic interactions and force constants

```@docs
HarmonicBondInteraction
HarmonicVectorCoordinate
HarmonicBendingInteraction
HarmonicAngleInteraction
harmonic_coordinate_vector
bending_force_constant_matrix
force_constants
ForceConstantBlock
RealSpaceForceConstants
project_force_constants
```

### Materials

```@docs
lookup_species
GaussianFormFactor
debye_waller_factor
mean_square_displacement
phonon_debye_waller_tensors
```

### Hamiltonian specifications

```@docs
build_model
AndersonImpurityModel
impurity_bath_sites
PhononUnitConvention
HarmonicPhononModel
harmonic_model
```

### Universal Hamiltonian generation

```@docs
HamiltonianGenerationResult
material_crystallographic_structure
material_generative_specification
generate_hamiltonian_family
```

### Validation

```@docs
validate_force_constants
```

## Additional exported model API

The remaining exports cover the geometry primitives and accessors (`BravaisLattice`, lattice conversion functions, `BasisSite`, `Models.CrystalStructure`, `CrystalSite`, `Supercell`, site and reciprocal helpers, `Bond`, shell/coordination utilities), local-frame accessors, abstract interaction/result types and evaluation functions, atomic/material types and accessors, `AbstractHamiltonianSpec`, `HeisenbergModel`, `XXZModel`, `HubbardModel`, `LocalPhononModel`, `HolsteinModel`, `LongitudinalSpinBosonModel`, acoustic-sum-rule processing, and the lattice/bond/material validation helpers.

## Structural search

`StructuralFamily` partitions an existing finite Hermitian invariant basis into `StructuralUnit`s. `StructuralSelection` retains complete symmetry-allowed basis directions and provides a child `HamiltonianSpace`, `ModelStructure`, and canonical identity. `structural_transition`, `structural_neighbors`, `enumerate_structures`, and `model_space_hierarchy` define the legal within-family search graph without assigning model probabilities or changing the generative physical assumptions.

`structural_coordinate_map` embeds or restricts nested selected-basis coefficient vectors. `embed_parameters`, `project_parameters`, and `parameter_subspace` preserve omitted-coefficient-zero semantics and parent equality constraints. Structural selection never assumes a solver or numerical truncation; the generated-model-to-forward closure is handled by structural forward closure. See [universal Hamiltonian generation and structural search](../workflows/universal-hamiltonian-generation.md).

## Structural selection and forward closure

`instantiate_model(selection, representation, values)` now accepts a `StructuralSelection` directly, preserves complete invariant directions, validates its child parameter schema against all parent equality constraints, and retains structural provenance. `validate_structural_representation` verifies the unambiguous spin, fermion, and boson local-space relationships without attempting to invent component-aware or composite symmetry actions.

`structural_id` and `describe` expose deterministic finite-domain identifiers and neutral structural descriptions. `reconstruct_selection` and `reconstruct_model` validate a proposed family, selection, and coefficient payload entirely in Julia. The forward execution helpers `prepare_forward`, `forward_parameters`, and `forward_batch` reuse the existing `ForwardProblem` pipeline without introducing shared mutable numerical caches. See [Structural Forward Closure](../workflows/structural-forward-closure.md).

## High-Level Physical Interaction Specifications

The high-level physical-interaction layer permits finite, named user-defined operator sums over arbitrary supported operator algebras, while retaining the authoritative crystallographic generator and structural-search/parameter contracts. `OperatorOrbit` accepts a user-supplied operator builder, and `InteractionSpecification` declares channel groups; specialized spin, fermionic and bosonic constructors are thin conveniences rather than independent model-generation implementations.

```@docs
OperatorOrbit
InteractionGroup
InteractionSpecification
PhysicalCoefficientMap
InteractionGenerationResult
PhysicalCouplings
physical_operator_sum
generate_interaction_family
physical_parameters
```

The spin-scattering convenience interface lives at the root of `Phundamental` and composes the existing observables, probes and forward pipeline without introducing a new numerical solver. See [Physical Interaction Generation](../workflows/physical-interaction-generation.md).

```@meta
CurrentModule = Phundamental
```

```@docs
SpinScatteringProblem
spin_observables
spin_scattering_problem
validate_forward_equivalence
```

```@meta
CurrentModule = Phundamental.Models
```

## Model contexts and structural localization

`generation_context` builds typed material and geometry provenance from a `HamiltonianGenerationResult`; `default_observable_context` combines it with supported standard observable registries. Generated selections automatically retain these contexts, with explicit input validated against the original site order. `locate_in_family` maps a symbolic external Hamiltonian or a `ManyBodyModel` into its parent invariant basis, returning structured span residuals, constraint checks, and a nonempty selection only when admissible. See [Model Context and Localization](../workflows/model-context-and-localization.md).

```@docs
generation_context
default_observable_context
StructuralLocation
locate_in_family
```

## Cartesian phonon acoustic constraints (Milestone B)

The acoustic sum rule is a linear constraint on the coefficients of the **complete harmonic potential**, rather than a projection of individual invariant operators. See [Physical Symmetry and Transformed Specifications](../workflows/physical-symmetry-and-transport.md).

```@docs
acoustic_sum_rule_constraints
```
