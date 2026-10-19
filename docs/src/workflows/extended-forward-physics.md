# Extended Forward Physics: Structural Clusters, Embedding, Continuation, and Observables

Milestone D extends existing numerical solvers without replacing their core implementations. New routes are intentionally limited to physical cases where geometry, operator indexing, and approximation assumptions can be audited.

## Finite-reference structural spin clusters

`StructuralNLCEForwardContext(selection, max_order; bonds, bond_terms, ...)` accepts a material-aware structural spin model instantiated from a `StructuralSelection`. The source must retain its exact reference `Supercell` and structural-generation provenance. The `bonds` collection enumerates physical `Bond` links, including their periodic-image displacements. Each entry of `bond_terms` is `Bond => operator_expression`, representing the complete numerical two-site interaction carried by one physical link. Multiple spin channels acting on the same link must be summed in that expression. The expressions must act exactly on the declared endpoint indices; their canonical sum must reproduce all nonlocal terms of the instantiated periodic Hamiltonian. The remaining canonical terms must be site-local, and a nonzero identity offset is rejected. Site-local-only models may omit `bond_terms`.

A periodic supercell can fold geometrically distinct interactions onto the same finite operator pair. A periodic Hamiltonian term such as $(J_1+J_2) S^z_1 S^z_2$ does not determine the separate open-cluster values of $J_1$ and $J_2$. The API therefore does not infer those couplings: a source with nonlocal interactions and no explicit image-resolved `bond_terms` is rejected. The algebraic closure check proves consistency of the declared physical decomposition with the periodic source but cannot independently establish whether a user's attribution to particular bonds is physically correct. Bond displacements are checked against the source lattice and site positions; self-image bonds, duplicate images, non-spin operators, and unsupported mappings are rejected.

The induced open reference retains only those declared links whose periodic-image winding is zero. Each connected subcluster inherits its constituent site's onsite terms and only those bond terms whose two endpoints belong to the induced open cluster. Wrapped periodic-image links are not silently converted into open-cluster couplings. `regenerate_induced_spin_cluster(source, reference, subcluster; bonds, bond_terms)` exposes the same checked construction for individual subclusters. Finite-reference NLCE still has `infinite_lattice_claim=false`; correct open clusters alone do not certify a thermodynamic limit or convergence with reference size.

For example, a periodic two-site AB reference can contain a short intracell bond and a longer bond to a periodic image of the second site. Their distinct couplings must be provided explicitly:

```julia
request = ObservableRequest(:ground_energy, (_, solution) -> groundenergy(solution))
physical_bonds = [intra_bond, periodic_image_bond]
physical_terms = [intra_bond => J1 * Sz(1) * Sz(2), periodic_image_bond => J2 * Sz(1) * Sz(2)]
context = StructuralNLCEForwardContext(selection, 2; bonds=physical_bonds,
    bond_terms=physical_terms, allow_periodic_reference=true)
result = forward(ForwardProblem(model; solver=NLCESolver(), method_context=context, observable=request))
```

Here `model` must already be the certified structural selection instantiated with a periodic exchange coefficient equal to $J_1+J_2$; the symbols `intra_bond`, `periodic_image_bond`, `J1`, and `J2` denote separately constructed physical inputs. Structural NLCE remains restricted to supported finite-reference spin models. Generated fermionic, multiorbital, phonon, or continuum Hamiltonians require their own physically justified cluster generator.

## Prepared structural calculations with method contexts

`prepare_forward(selection, representation; solver, observable, method_context, ...)` now propagates a typed `AbstractForwardMethodContext` to the ordinary `ForwardProblem` executor for each selected coefficient point. The context must accompany an `AbstractObservable` request; it is never used for an untyped callable or silently omitted. `forward_parameters` validates the requested solver/model/observable/context combination at the current parameter point, and `forward_batch` retains the same context while regenerating each model independently. `describe(prepared)["method_context"]` records the selected context type without serializing solver configuration into a neutral execution payload.

The bond-resolved `StructuralNLCEForwardContext` from F1 can be used directly in prepared spin calculations. Each parameter point must remain consistent with the context's declared image-resolved bond terms; changing a nonlocal exchange coefficient while keeping fixed `bond_terms` raises a compatibility error instead of silently reusing outdated physical couplings. Onsite coefficients may vary while the certified bond decomposition remains fixed.

```julia
context = StructuralNLCEForwardContext(selection, 2; bonds=physical_bonds, bond_terms=physical_terms, allow_periodic_reference=true)
request = ObservableRequest(:ground_energy, (_, solution) -> groundenergy(solution))
prepared = prepare_forward(selection, representation; solver=NLCESolver(), observable=request, method_context=context)
results = forward_batch(prepared, parameter_points)
```

If `policy` is given together with a direct `method_context`, the prepared context becomes an explicit `ForwardRoute` candidate. The policy must authorize its approximation, and the chosen plan must retain that context; use `include_default_exact_routes=false` if a default exact solver would otherwise supersede it. The direct-context form does not accept additional competing `routes`. To compare or rank alternative methods, put the context on each candidate `ForwardRoute` and omit the top-level `method_context` keyword. Structural NLCE capability reports intentionally remain uncertified until open-cluster construction at execution, so automated planning requires `require_certified=false` for this route. This does not elevate finite-order NLCE to an infinite-lattice convergence claim.

## Single-band periodic embedding

`BravaisHopping(displacement, amplitude)` represents one directed Hamiltonian hopping between primitive cells, with its Hermitian reverse inferred. `bravais_momentum_lattice(material, shape, hoppings; onsite_energy=0)` constructs a scalar one-orbital `MomentumLattice` with the existing exact periodic momentum-shift maps. The momenta are reduced lattice radians; for Cartesian reciprocal momenta convert with the material reciprocal lattice. The dispersion is $\varepsilon(\mathbf k)=\varepsilon_0+2\operatorname{Re}\sum_{\delta}t_{\delta}e^{i\mathbf k\cdot\delta}$.

The model overload additionally verifies a supported scalar Hubbard source and the package Hamiltonian sign convention, under which each hopping matrix element is $-t$. For a generated structural selection with a certified single-orbital spinful-fermion component layout, F3 also constructs independent nearest-shell Hubbard hopping, onsite-interaction, and chemical-potential operator templates on the original finite `Supercell`. It recovers their unique real coefficients through canonical operator-space localization and rejects residual Hamiltonian terms, rank-deficient reference bases, missing source provenance, and multiorbital models. The generator must declare a finite physical `max_range`, and twice that range must be smaller than the minimum singular value of the reference-supercell translation matrix; otherwise periodic-image ambiguity is rejected. For this generated route, the explicitly supplied Bravais hopping directions must reproduce exactly the nearest-shell bond directions, with matching $-t$ amplitudes and no extra onsite-energy shift. This is a supported nearest-neighbor, one-orbital mapping, not a general interpretation of arbitrary fermion Hamiltonians or automatic $t'$ extraction. This establishes equivalence on the declared finite model and a geometrically separated interaction domain; it is not an inference of hidden longer-range bonds from aliased finite-site coefficients. A declared `max_range` is a physical modeling assumption, not independent evidence that an arbitrary external Hamiltonian has no longer-range periodic-image contributions. No DMFT/DΓA approximation is changed by this certification.

Geometry does not justify inventing missing orbitals, hopping lengths, spin-orbit terms, or an arbitrary band structure. The ordinary typed `HubbardModel` route continues to accept its established explicitly supplied geometric hoppings; the additional exact-shell check applies only to structurally generated models.

```julia
hops = [BravaisHopping((1,), -0.7)]
lattice = bravais_momentum_lattice(model, (64,), hops)
context = DMFTForwardContext(model, (64,), hops; temperature=0.5)
```

`LadderDGAForwardContext(model, shape, hops, bosonic_axis; ...)` uses the same geometrically derived `MomentumLattice`, including its momentum-addition maps. The DMFT and ladder-DΓA physical approximations remain explicit in their respective solver descriptions and forward policies.

## Explicit approximate analytic continuation

`continue_matsubara(axis, values, real_axis; method=RationalContinuation(...))` fits a scalar rational approximation to upper-half-plane fermionic Matsubara data and evaluates it at $\omega+i\eta$. A `GreenFunction` overload is also available. Fit residuals, numerical conditioning, and a minimum spectral density are reported. A small fit residual does not establish uniqueness, causality, or reliable spectral weights. In particular, no spectral-weight clipping, negative-density removal, or statistical uncertainty estimation is performed implicitly.

```julia
method = RationalContinuation(numerator_order=0, denominator_order=1, eta=0.05)
result = continue_matsubara(axis, green_values, collect(range(-3.0, 3.0; length=301)); method=method)
```

Continuation is a separate opt-in operation and is not silently applied inside Matsubara-domain DMFT or DΓA forward calculations. The built-in rational method is best suited to controlled low-pole or benchmark spectra; noisy experimental or Monte Carlo Green functions require regularization and independent error analysis not supplied by this routine.

## Typed static, thermal, multiphonon, and CPT forward calculations

`StaticStructureFactorRequest(field, qgrid; temperature, convention)` delegates to the existing equal-time structure-factor implementation. `ThermodynamicCurveRequest(temperatures; kB)` works with `ThermodynamicForwardSolver(method)`, where the method can be `ExactGibbs`, `ThermalLanczos`, or `CanonicalTPQ`. The planner includes exact thermodynamic curves by default; stochastic thermal sampling must be explicitly authorized with `ForwardExecutionPolicy(allowed_approximations=(:stochastic_thermal_sampling,))`.

`CPTSpectralFunctionRequest(momenta, energy_axis; eta, t_intercluster, spin)` uses an existing optimized fermion-sector eigenstate to construct the standard CPT spectral function. Intercluster hopping and line broadening are explicitly required, and CPT is not assigned to a generic, uncertified eigensystem solver.

```julia
static = StaticStructureFactorRequest(field, qgrid)
static_result = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=static))

thermal = ThermodynamicCurveRequest([0.1, 0.2, 0.5, 1.0])
thermal_result = forward(ForwardProblem(model; solver=ThermodynamicForwardSolver(ExactGibbs()), observable=thermal))
```

The high-level requests reuse the existing result types, with no new thermodynamic or CPT numerical kernel. Static structure factors require supported full eigensystems or quadratic bosonic modes; stochastic thermodynamic requests cannot be combined with unrelated outer solver transformations unless their compatibility is explicitly established.

`HarmonicMultiphononSliceRequest(probe; fixed_axis, fixed_value, horizontal_axis, horizontal_values, vertical_axis, vertical_values, qmesh_shape, energy_window, order=2, temperature=0, broadening=nothing)` lowers to `HarmonicPhononSolver` and then reuses `harmonic_neutron_slice` without a second instrumental probe contraction. Its current high-level domain requires three Cartesian dimensions. Supply `NuclearNeutronProbe` inside the request and do not also set `ForwardProblem.probe` or `ForwardProblem.resolution`, because integrated-slice instrument convolution is not currently defined. The existing numerical multiphonon engine provides `order=1`, `order=2`, and broadened higher-order/`:all` results; this request does not claim instrument-calibrated accuracy.
