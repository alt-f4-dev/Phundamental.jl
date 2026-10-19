```@meta
CurrentModule = Phundamental
```

# Physical Interaction Specification and Invariant Coupling Coordinates

The high-level physical-interaction layer declares physical Hamiltonian terms over explicit supports and delegates symmetry projection, Hermiticity, linear independence, admissibility, parameter constraints and model instantiation to the existing foundational machinery. It does not attempt to infer arbitrary internal symmetry actions or numerical representations from the material's name or chemical species.

## General physical interaction domain

An `OperatorOrbit` represents one **independent physical coupling channel** and a finite ordered set of supports; its operator builder is called with the members of each support as positional arguments. The builder must return an `AbstractOperatorExpr`. A support can have one, two, three or more indices, and an arbitrary compatible operator algebra can be used. The physical operator is the weighted *sum* over supports, not an average.

```julia
custom = OperatorOrbit(:three_body, [(1, 2, 3)]; operator=(i, j, k) -> Sz(i) * Sz(j) * Sz(k))
interactions = InteractionSpecification(custom)
result = generate_interaction_family(material, SpinAlgebra(3), interactions; dataset=dataset, max_body_order=3)
```

General `OperatorOrbit` builders may encode supported fermionic, bosonic, Majorana, coordinate/momentum, and composite symbolic interactions, provided the specified algebra and custom symmetry action correctly represent their indices and transformations. Operator builders remain Julia callbacks; neutral structural descriptions do not serialize their semantics.

## Named interaction channels

Spin exchange, local fields, hopping, density interactions and pairing have thin typed constructors. `ExchangeOrbit(:J, bonds)` creates `:J_xy` and `:J_zz` channels belonging to a single structural unit `:J`. The supported channels are `:xy`, `:zz`, `:xx` and `:yy`. `FieldOrbit(:h, sites)` generates the physical negative-Zeeman operator `-Σᵢ Sᶻᵢ`; change `sign` or `axis` explicitly if another convention is intended. `HoppingOrbit` and `DensityInteractionOrbit` accept `statistics=:fermion` or `:boson`; `PairingOrbit` builds a Hermitian spinless-fermion pairing channel from oriented bonds.

```julia
bonds = [(1, 2), (2, 3), (3, 4), (4, 1)]
interactions = InteractionSpecification(ExchangeOrbit(:nearest_neighbor, bonds), FieldOrbit(:field, 1:4))
result = generate_hamiltonian_family(material, SpinAlgebra(4); interactions=interactions, dataset=dataset, max_body_order=2, parameter_units=fill(:meV, 3))
family = structural_family(result)
selection = structural_selection(family, [:nearest_neighbor, :field])
```

The material, symmetry dataset and representation in these examples are assumed to be explicitly defined and consistent with the operator algebra. The generator uses the material overloads already established for primitive cells and periodic supercells; an explicit `dataset`, `symmetry_action` and `local_spaces` may be provided under the same physical restrictions as the lower-level API.

## Physical coupling coordinates

For each declared channel μ, let `O_μ` be its physical weighted operator sum. The candidate supplied to the crystallographic generator is `O_μ / N_μ`, where `N_μ` is the number of supplied supports; this convention does not assume that the generated invariant basis equals a simple bond average. The generator independently constructs Hermitian invariant directions `Φ_a`; the physical-to-invariant transformation `T` is then computed and checked from canonical complex operator-coordinate maps.

```math
O_\mu=\sum_a T_{a\mu}\Phi_a,\qquad\boldsymbol\theta=T\mathbf J.
```

The result preserves `generation`, `family`, `interactions`, and `coefficient_map`. For the physical coupling values, use a label-keyed mapping or `PhysicalCouplings`:

```julia
J = PhysicalCouplings(exchange=Dict(:nearest_neighbor => (3.5, 4.2)), fields=Dict(:field => 0.2), unit=:meV)
coefficients = physical_parameters(result, J)
model = instantiate_model(selection, representation, coefficients)
```

When a physical coupling unit is supplied, the generated parameter specifications must declare the same unit for every direction (for example with `parameter_units=fill(:meV, length_of_basis)`); no implicit unit conversion occurs. Bounds and prior distributions supplied through the generator apply to **invariant coordinates** `θ`, not automatically to the physical coordinates `J`. When `T` mixes channels, transporting a prior or bounded feasible region requires additional mathematics and is not performed implicitly.

## Symmetry and identifiability

The default `symmetry_policy=:require` verifies that each physical channel operator is already invariant under the declared symmetry action. In contrast, `symmetry_policy=:project` explicitly replaces the declared physical operator by its Reynolds projection. The former rejects symmetry-breaking interactions; the latter is a deliberate change of Hamiltonian, not an assertion of exact physical equivalence. Both operations use the existing generator's admissibility conditions and symmetry actions. A channel that vanishes or becomes linearly dependent after generation is rejected, because otherwise its structural-unit ownership and its independently specified coupling parameter would become ambiguous.

```julia
projected = generate_interaction_family(material, algebra, interactions; dataset=dataset, symmetry_policy=:project)
record = describe(projected)
```

The physical interaction domain remains finite and user-supplied. The generated basis is complete only within this declared candidate domain, not across all possible Hamiltonians. Arbitrary multiorbital/vector-component or spin-orbit symmetry operations require a user-supplied physically correct action; the standard crystallographic site-permutation action does not prove their validity.

## Spin-scattering problem setup

`spin_scattering_problem` constructs the local Cartesian spin-observable registry and Cartesian positions, a `SpinTensorStructureFactorRequest`, and an explicitly configured `MagneticNeutronProbe`. The default energy convention is meV with temperature in kelvin and the corresponding Boltzmann constant. No ionic form factors, absolute cross sections, magnetic domains, or detailed instrument models are silently inferred. The material lattice vectors and supplied wavevectors must use mutually compatible Cartesian length units (the defaults illustrated here assume Å and Å⁻¹).

```julia
representation = Representation(:exact_spin, SpinHilbertSpace(fill(1//2, 4)), SpinAlgebra(4), SpinProductBasis(collect(1:4)))
problem = spin_scattering_problem(material, representation, [[π / 2, 0.0, 0.0]], range(0.0, 15.0; length=301);
    resolution=GaussianResolution(0.16))
prepared = prepare_forward(selection, problem; solver=ExactDiagonalization())
response = forward_parameters(prepared, coefficients; parameter_domain=:parent)
```

`parameter_domain=:parent` accepts a vector, a keyed parent-coordinate mapping, or a parent `ParameterPoint`. By default, `strict=true` prohibits dropping nonzero coefficients from omitted structural units. Pass `strict=false` only when deliberately comparing a selected submodel with a broader nonzero parent parameter point; the omitted coefficients are then fixed to zero, and the remaining selected point must still satisfy all reduced parent constraints.

`validate_forward_equivalence(left, right; compare=(:spectrum, :intrinsic, :probe, :instrument))` reports a boolean for each requested forward stage; an independent reference Hamiltonian must still be constructed without reusing the tested physical-interaction expansion if the goal is to detect implementation errors in that expansion.
