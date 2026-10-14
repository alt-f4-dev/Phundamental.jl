```@meta
CurrentModule = Phundamental
```

# Batch 12: Structural Forward Closure and Reconstruction

Batch 12 connects the Batch 10 crystallographic generator and Batch 11 structural search to the existing representation-aware forward pipeline. Julia remains authoritative for physical admissibility, structural selection, parameter constraints, Hamiltonian assembly, numerical transformations, solver execution, and observable/probe/resolution calculations. Machine-readable records identify objects retained in Julia; they are not serialized Julia operator trees, executable callbacks, or independently executable physical model definitions.

The operational chain is

```math
\mathrm{Material}\rightarrow\mathfrak G\rightarrow\mathcal H(\mathfrak G)\rightarrow\mathcal H'\rightarrow\Theta(\mathcal H')\rightarrow H(\boldsymbol\theta)\rightarrow\mathrm{ForwardResult}.
```

## Instantiating a selected structural hypothesis

The generator remains unchanged. The following example uses an explicit finite candidate domain and an identity-only crystallographic dataset so its physical assumptions can be inspected without relying on automated symmetry discovery. The two basis directions are distinct because the two site species are different.

```julia
using Phundamental
using LinearAlgebra

lattice = BravaisLattice(Matrix{Float64}(I, 3, 3))
crystal = LatticeCrystalStructure(lattice, [BasisSite(:A1, :A, [0.0, 0.0, 0.0]), BasisSite(:B1, :B, [0.4, 0.0, 0.0])])
material = Material("two-site magnet", crystal;
    species=Dict(:A => AtomicSpecies(:A; mass=1.0), :B => AtomicSpecies(:B; mass=1.0)),
    basis_properties=[SiteProperties(spin=1//2), SiteProperties(spin=1//2)])
operation = SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
dataset = CrystallographicDataset(spacegroup_number=1, international_symbol="P1", operations=[operation], crystallographic_orbits=[1, 2])
result = generate_hamiltonian_family(material, SpinAlgebra(2); dataset=dataset, max_body_order=1,
    candidates=AbstractOperatorExpr[Sz(1), Sz(2)], basis_labels=[:h1, :h2])
family = structural_family(result)
selection = structural_selection(family, [:h1])
representation = Representation(:finite_spin, SpinHilbertSpace([1//2, 1//2]), SpinAlgebra(2), SpinProductBasis([1, 2]))
model = instantiate_model(selection, representation, Dict(:h1 => 0.7))
```

`instantiate_model(selection, representation, values)` constructs the selected `ParameterSpace` through `parameter_subspace`, checks all parent bounds and equality constraints after omitted coefficients are set to zero, validates the representation's algebra and unambiguous local state spaces, constructs the Hamiltonian from complete selected invariant directions, and stores a structural provenance record. A caller-provided `ParameterPoint` must refer to the exact selected `HamiltonianSpace` and the correct ordering, and its values are nevertheless revalidated against the parent-derived schema. Unsupported coordinate-component, Majorana, or composite internal mappings are not inferred by this layer.

## Deterministic structural identifiers and neutral descriptions

`structural_id` computes versioned SHA-256 digests from the canonical Batch 11 structural signatures, rather than Julia's session-dependent `hash` values. `describe` supplies nested dictionaries containing only neutral scalar values, strings, arrays and dictionaries, with explicit schema versions. The stable identity is relative to the declared finite generation domain, numerical signature quantization, Julia package semantics, and captured callback descriptors. Because arbitrary Julia callbacks cannot be reconstructed or certified from a text record, their implementation semantics require a pinned Julia execution environment; matching identifiers do not independently prove two callback implementations are behaviorally equivalent.

```julia
family_record = describe(family)
selection_record = describe(selection)
parameter_record = describe(parameter_subspace(selection, [0.7]))
model_record = describe(model)
```

The live `StructuralFamily` in Julia is authoritative. Reconstruction never executes an input operator tree or trusts a submitted declaration of admissibility.

```julia
request = Dict("schema_version" => "1.0", "family_id" => structural_id(family),
    "selection_id" => structural_id(selection), "active_units" => ["h1"],
    "parameters" => Dict("h1" => 0.7))
recovered = reconstruct_selection(family, request)
recovered_model = reconstruct_model(family, representation, request)
```

The unit-set form reconstructs directly and checks both the family and selection identities. If the unit set is omitted, `reconstruct_selection` may search the family with a hard `max_structures` guard; an unknown, inconsistent, or forged identifier is rejected. `reconstruct_model` accepts parameters only after successful reconstruction and performs the same Julia-side physical checks as ordinary instantiation. This is a bounded lookup interface, not a global registry of arbitrary structural models.

## Forward evaluation and safe repeated parameter calls

The forward layer preserves the complete intermediate `ForwardResult` with source model, transformed working model, transformation record, solver output, intrinsic observable, probe-contracted scattering, and final instrument-convolved result. A prepared structural forward request stores only immutable structural bindings and explicit solver/observable configuration. It does not reuse a mutable model, computational basis, or solver cache across numerical points.

```julia
prepared = prepare_forward(selection, representation; solver=ExactDiagonalization())
prediction = forward_parameters(prepared, Dict(:h1 => 0.7))
other_predictions = forward_batch(prepared, [[0.7], [0.9]])
prediction_record = describe(prediction)
```

With a typed `AbstractObservable`, a caller can optionally supply `policy=ForwardExecutionPolicy()` to use the existing request-aware `plan_forward` machinery. Without a policy, the configured solver is executed through the ordinary `ForwardProblem` implementation; no solver compatibility is confused with physical admissibility. Probe contraction and resolution convolution remain separate optional stages. The `describe(prepared)` and `describe(prediction)` methods expose a neutral inventory of the stages; they do not serialize entire numerical arrays or dynamically callable solver or probe objects.

## Validity boundaries

Batch 12 also corrects the geometric `max_range` evaluation for nonorthogonal cells: it minimizes the Cartesian distance over periodic lattice images, using a finite singular-value-derived bound rather than componentwise fractional wrapping. Ill-conditioned cells requiring an excessive exhaustive image search fail explicitly rather than silently using an incorrect cutoff.

The standard material-aware crystallographic path uses a three-dimensional geometry and one operator index per crystallographic site for its unambiguous automatic path. Richer orbital/spinor symmetries, vector-component phonon displacements, composite factors, and Majorana representations require explicit operator actions and local spaces. Crystallographic site permutations alone do not prove spin rotational invariance, axial-vector transformation, time-reversal symmetry, or the acoustic sum rule. Separate existing phonon solvers remain valid within their own tested domains, but do not constitute an automatic phonon invariant-basis generator.

A numerically small coefficient and structural absence remain distinct: the selected set of invariant directions fixes structure even when one selected coefficient equals zero. Likewise, physical Hamiltonians, representations and numerical truncations are independent objects. Identifiers describe declared structural content; they are not a proof of numerical convergence, experimental agreement, or global completeness over unbounded Hamiltonian families.

## Testing and documentation

`test/structural-forward-closure-validation.jl` exercises generated spin and fermion reference models, finite bosonic truncations, exact structural reconstruction, tampered identifiers, parameter constraints, and spin neutron-scattering observables before and after probe and resolution treatment. The regressions are intended to run as part of `Pkg.test()` along with the existing suite. Documentation is built with `julia --project=docs docs/make.jl`. External execution of Julia and numerical tests remains necessary to establish the batch as validated.

## API reference

```@docs; canonical=false
PreparedStructuralForward
prepare_forward
forward_parameters
forward_batch
reconstruct_selection
reconstruct_model
validate_structural_representation
structural_id
```
