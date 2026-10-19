# Model context and external Hamiltonian localization

Milestone A connects an authoritative crystallographic Hamiltonian generation result to its geometric and observable context, and recovers canonical invariant-basis coordinates from independently constructed Hamiltonians. This workflow does not extend the symmetry group representation, transform model specifications between algebras, or perform parameter inference.

## Typed physical context

`GeometryContext` stores Cartesian site positions, direct and reciprocal matrices, site labels, optional physical site masses in amu, and a certified operator-index mapping. `:one_index_per_site` retains the original spin and single-mode convention; `:site_component_layout` requires an explicit `ComponentIndexLayout` certifying the component count and ordering. Length units are explicitly recorded when known, and otherwise remain unspecified rather than being inferred from numeric coordinates. `PhysicalModelContext` retains the source material, optional supercell, crystallographic dataset, and either certified geometry or `nothing` when the algebra's operator index-to-site correspondence cannot be established. `ObservableContext` records the associated conventional observable registry and its naming convention. All three carrier types are defined in `PhundamentalCore` to avoid dependencies from the numerical solver layer on `Models`.

Generated models continue to use their existing `ManyBodyModel` fields. The typed context is available through `model_context(model)`, `geometry_context(model)`, and `observable_context(model)`; it is carried under the reserved `model.parameters[:physical_context]` key for backward compatibility. The familiar `:material`, `:cluster`, and `:positions` keys are populated from verified provenance when available, without fabricating `:hamiltonian_spec` or changing an independently specified material model. Existing handwritten models have no context unless supplied explicitly.

`instantiate_model(selection, representation, coefficients)` automatically retrieves the source `HamiltonianGenerationResult` from the family. For supported spin representations it registers `spin_x_i`, `spin_y_i`, and `spin_z_i`. For certified spinful-fermion families it registers `fermion_density_i`, `fermion_density_i_orbital`, and `fermion_spin_{x,y,z}_i`; for Cartesian vector families it registers `phonon_displacement_{x,y,z}_i` and `phonon_momentum_{x,y,z}_i`. All generated observables are associated with the authoritative physical site positions in the crystallographic ordering. If explicit user metadata disagrees with those certified positions or operators, construction fails rather than silently changing the momentum-dependent scattering phases. No internal spin rotation, ionic form factor, or experimental cross section is inferred.

```julia
selection = structural_selection(family, [:field, :exchange])
model = instantiate_model(selection, representation, Dict(:field => 0.6, :exchange => 0.8))
context = model_context(model)
positions = geometry_context(model).positions
```

`default_observable_context(family, representation)` inspects the derived registry without constructing a model. Certified `SpinfulFermionAction` families use site-major, orbital-major, spin-minor mode ordering, with spin `1` denoting up and `2` denoting down. Certified `VectorDisplacementAction` families use site-major Cartesian `(:x, :y, :z)` ordering for both displacement and conjugate momentum. `component_index(geometry, site, orbital, spin)` and `component_index(geometry, site, axis)` perform forward indexing; `component_site(geometry, mode)` and `component_coordinates(geometry, mode)` provide the inverse mapping. Other internal actions remain unsupported unless they certify their mapping; the API does not infer geometry from mode count alone.

The site-resolved generated fermion spin observables are physical spin-density bilinears, not independent spin operators. Their normalization is spin-½: $S_s^z = (n_{s\uparrow}-n_{s\downarrow})/2$, summed over orbitals, with matching $S_s^x$ and $S_s^y$. The phonon registry exposes unweighted canonical positions and momenta, without assuming a mode frequency or inventing a mass-weighted transformation. For certified Cartesian phonon families, `model.parameters[:masses]` contains the material's atomic masses replicated across Cartesian components; conflicting supplied masses are rejected. The forward interface can use these local registries through `registered_momentum_field` and `momentum_operator`, while numerical solver compatibility remains separately certified.

## External Hamiltonian coordinates

`structural_span_coordinates(basis, candidate, algebra)` uses canonical noncommutative operator coordinates and a real least-squares solve to recover coordinates in a Hermitian basis. The absolute residual measures mismatch in the joint real/imaginary coefficient vector; the tolerance test is relative to the candidate norm with an absolute floor. Nonnumeric symbolic coefficients return `missing` rather than false membership.

`locate_in_family(family, model_or_operator)` extends that operation to a complete `StructuralFamily`. It reports full parent coefficients, active structural units, residuals, constraint admissibility, and a selected structural subspace only when the Hamiltonian is representable and nonempty. By default, a constant identity shift not spanned by the family is considered out of span. With `project_identity=true`, a real scalar identity shift is removed from the candidate and returned as `location.identity_offset` and `describe(location)["identity_offset"]`; a family that already includes an identity basis direction cannot use that projection. Removing the offset establishes only equivalence modulo an overall energy origin, not literal Hamiltonian equality. Non-Hermitian imaginary identity shifts are rejected. The active-unit cutoff remains an explicit numerical interpretation choice: `zero_tolerance` is absolute, and `relative_zero_tolerance` optionally sets an additional cutoff relative to the Euclidean norm of the invariant-basis coordinates. The default preserves previous absolute-threshold behavior. Parameter constraints are still checked against all recovered parent coefficients, including coordinates below the active-unit threshold.

```julia
reference = ManyBodyModel(representation, 0.6 * (Sz(1) + Sz(2)) + 0.8 * Sz(1) * Sz(2))
location = locate_in_family(family, reference; tolerance=1e-10)
@assert location.in_span && location.valid_parameters
@assert !isnothing(location.selection)
record = describe(location)
shifted = locate_in_family(family, reference.hamiltonian + 1.2 * IdentityOperator(); project_identity=true)
@assert shifted.identity_offset ≈ 1.2
```

The canonical coordinate system is shared with the high-level physical-coupling transform. A successful location certifies membership only in the finite, supplied candidate family and its declared symmetry action. It does not establish completeness relative to all possible Hamiltonians, and does not replace exact operator equivalence and independent forward-physics regressions.

## Validation and continuous integration

`test/milestone-a-integration-validation.jl` covers nonidentity crystallographic translation symmetry, automatic context propagation, independently constructed magnetic-neutron scattering, valid and invalid structural localization, and cross-unit parameter equality. The CI workflow `.github/workflows/package-tests.yml` runs strict package precompilation and the complete regression suite against Julia 1.12 and the latest stable Julia line; a separate job builds the documentation without deploying it.
