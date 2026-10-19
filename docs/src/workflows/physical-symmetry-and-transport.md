# Physical symmetry and transformed generative specifications

Milestone B extends the existing crystallographic fixed-space generator to multicomponent fermionic and coordinate/momentum degrees of freedom. These operators remain subject to a finite supplied candidate domain; generating an invariant basis does not certify universal Hamiltonian completeness.

## Axial spins and time reversal

`standard_symmetry_action` already maps spin components with the Cartesian axial rotation `det(R)R`, applying an additional sign under `SpaceGroupOperation(...; time_reversal=true)`. The named `MagneticSymmetryAction()` wraps this implementation with a physical Cartesian-isometry check. An antiunitary operation conjugates scalar coefficients as well as transforming operators. The operation list must represent the complete intended finite symmetry group, not a handpicked subset of constraints.

For an orthogonal spatial rotation/reflection `R`, the spin operation is $S_i^\alpha \mapsto \det(R)\sum_\beta R_{\beta\alpha}S_{g(i)}^\beta$ (and a further minus sign under time reversal). A mirror plane containing a bond forbids DM components parallel to that plane, and inversion centered at a bond midpoint forbids all DM components. The Milestone B regression suite checks these independently against explicit bilinear expressions.

## Spinful fermions and multiorbital representations

`SpinfulFermionAction(; norbitals=1, orbital_representation=nothing, orbital_time_reversal=nothing)` declares a site-major, then orbital-major, then spin-major index layout. Each crystallographic site contributes `2 * norbitals` fermionic modes. The internal action maps annihilation operators using the spin-$1/2$ lift of the proper axial rotation and the supplied orbital representation; creation operators use the conjugate matrix. Orbital matrices must be unitary. The default identity orbital action asserts that orbitals behave as spatial scalars, which is appropriate only for a physically justified orbital basis. A multiorbital action containing time reversal requires an explicit orbital time-reversal matrix because a general orbital basis cannot be presumed real. Matrix unitarity alone does not certify the multiplication law of a complete crystallographic group; the caller must supply a consistent orbital representation.

The annihilation-field matrix is chosen as the adjoint of the active SU(2) spinor rotation, ensuring that the spin density $c^\dagger\boldsymbol\sigma c/2$ transforms with the same axial Cartesian matrix as the explicit spin-operator action. The lift of a spatial rotation to a spinor is projective: a $2\pi$ rotation acts as minus identity on a single spinor. The symmetry action is defined on both odd-parity field operators and even-parity operator products, so transformed annihilation/creation operators and their anticommutators can be evaluated. The **invariant Hamiltonian generator**, however, requires even-fermion-parity candidates, for which the central spinor sign cancels. Explicit pairing terms may be even parity and are supported by the same algebraic action. Full magnetic space-group corepresentations, momentum-dependent orbital Bloch phases, and site-dependent orbital gauges require custom actions.

```julia
specification = material_generative_specification(material, FermionAlgebra(2 * nsites); dataset=dataset,
    symmetry_action=SpinfulFermionAction())
```

Here `nsites` denotes the number of crystallographic sites in the material; the keyword is not a Julia variable supplied automatically. For multiorbital materials, set `norbitals` explicitly and supply the physical representation matrices.

## Vector displacement and momentum

`VectorDisplacementAction()` provides an unambiguous **three Cartesian components per crystallographic site** for `CoordinateMomentumAlgebra(3 * nsites)`. The primitive operators transform as $x_{i\alpha} \mapsto \sum_\beta R_{\beta\alpha}x_{g(i)\beta}$ and $p_{i\alpha} \mapsto (-1)^\eta\sum_\beta R_{\beta\alpha}p_{g(i)\beta}$, where $\eta=1$ denotes antiunitary time reversal. A full six-dimensional canonical phase-space map is not implied, and internal phonon modes in a non-Cartesian basis require a custom action.

## Acoustic sum rule as a coefficient constraint

For a harmonic potential $H_V=\frac12\sum_{i\alpha,j\beta} K_{i\alpha,j\beta}x_{i\alpha}x_{j\beta}$, translational invariance requires $\sum_j K_{i\alpha,j\beta}=0$ for every $i,\alpha,\beta$. Crystallographic invariance alone does not enforce this identity. `acoustic_sum_rule_constraints(space; nsites)` converts the row-sum identities into **independent named linear parameter constraints**, without filtering out individual basis directions needed to satisfy them collectively.

For a material-aware `CoordinateMomentumAlgebra(3 * nsites)` family, pass `acoustic_sum_rule=true` to `generate_hamiltonian_family(...)`. The generator attaches the resulting homogeneous `LinearParameterConstraint`s to the generated `ParameterSpace`. The default zero coefficients obey the constraint; nonzero coefficients must be checked with `ParameterPoint` or `parameter_constraint_report`. Only quadratic Cartesian displacement potentials plus optional quadratic kinetic operators are admitted for this rule; mixed and anharmonic terms are rejected rather than silently ignored.

## F4 crystallographic-group certification

`validate_crystallographic_group(crystal, dataset; atol=1e-7)` verifies the supplied finite operation list against the given crystallographic site geometry: integer unimodular fractional rotations, Cartesian metric preservation, bijective site permutations, an identity, unique operations modulo lattice translations, and closure under composition. Antiunitary flags compose by XOR. The check does not certify user-supplied orbital matrix representations or magnetic corepresentations. Pass `certify_group=true` to `material_generative_specification` or the material-aware `generate_hamiltonian_family` to request this independent validation before Reynolds projection. Existing caller-provided incomplete operation lists remain accepted without this opt-in for backward compatibility; such a list must not be interpreted as a full space group.

The F4 tests use a Spglib-discovered cubic B2 crystal (space group 221, 48 operations) whose two-site bilinear spin interaction has exactly one invariant Heisenberg exchange direction, and a fourfold screw structure (space group 76) for which four symmetry-related onsite spin fields project to one invariant sum. The screw example additionally verifies fractional translation and rejects a deliberately incomplete operation set.

## F4 rotational constraints on a finite open Hessian

For a finite open reference with Cartesian equilibrium positions $r_i$, an infinitesimal rigid rotation with axis $e_δ$ gives $u_j=e_δ\times(r_j-r_i)$. Its row-wise restoring force vanishes when $\sum_{jβ}K_{iα,jβ}[e_δ\times(r_j-r_i)]_β=0$. `rotational_sum_rule_constraints(space; nsites, positions, boundary=:open)` translates these homogeneous identities into independent `LinearParameterConstraint`s for a purely quadratic Cartesian-coordinate Hessian. It accepts a 3×N Cartesian position matrix, its transpose, or an iterable of Cartesian three-vectors. These constraints must be combined with the existing translational acoustic rule for a rigid-body-invariant finite mechanical system.

The material-aware generator accepts `rotational_sum_rule=true` **only** together with `acoustic_sum_rule=true`, explicit Cartesian `rotational_positions`, and `rotational_scope=:finite_site_hessian`. That scope is an explicit assertion that the generated finite-index Hessian is intended as a nonaliased, open-reference force-constant model. It is *not* a certification of infinite periodic-lattice Born–Huang rotational invariance. Periodic Hessians that have summed interactions over distinct lattice images cannot be reconstructed from the folded coefficient matrix: image-resolved force constants are required for a physical periodic rotational sum rule. Calls without the explicit finite-reference scope fail rather than silently interpreting periodic-site differences as physical bonds.

A direct regression contrasts longitudinal central-force and transverse pair springs along a two-site bond. Both satisfy the translational rule, but only the longitudinal spring is compatible with infinitesimal rigid rotation. The tests also verify invariance under shifting the Cartesian coordinate origin.

## Transporting specifications under representation changes

A transformed Hamiltonian does not generally inherit its source-representation symmetry action, local-state spaces, real-space interaction range, admissibility predicates, or operator body-order restriction. The generic `transform(T, model)` path now records these as `:not_transported` in the target `GenerativeSpecification.metadata`, clears unsupported action/local-space/range-evaluator fields, and records the source values for provenance. The old body-order bound is recorded but not asserted as a target bound. A declared source `max_range` is retained so that a subsequent admissibility audit explicitly reports it as unevaluated.

A transformed model remains usable for its valid numerical forward operations, but `admissibility_report(transformed.model)` is **incomplete** when physical invariance and constraints have not been transported. Applying `reynolds_project` to such a specification fails with an explanatory error. This does not assert that the mapped Hamiltonian violates symmetry; it says that symmetry membership cannot be certified by the transported specification. A future implementation of exact operator-level conjugation for particular representation maps may provide stronger transport guarantees.

## Validation boundary

Unit tests cover inversion, mirrors, explicit time reversal, spinor rotations, orbital mixing, coordinate/momentum rotations, parameter-space acoustic equalities, and a generated Jordan–Wigner target. These tests should be executed alongside the preexisting numerical transformation equivalence and Milestone A tests. Physical fidelity for spin-orbit-coupled noncollinear magnets, full magnetic corepresentations, and arbitrary hybrid tensor factors requires additional physical conventions and remains outside this milestone.
