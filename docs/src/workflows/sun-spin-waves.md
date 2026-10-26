# Site SU(N) generalized spin waves

The public `SpinWaveSolver` supports two explicitly selected formalisms: `:dipole` (the default, preserving the original noncollinear SU(2) solver) and `:SUN` (a site-based generalized multiboson harmonic expansion). The latter does not require the Hamiltonian to possess SU(N) symmetry: `N=2S+1` denotes the complete local spin-state dimension, and each magnetic site supplies `2S` tangent excitation flavors.

## References and API

Use `MagneticReferenceState` for an ordered dipolar reference in either formalism, or `SUNReferenceState` to specify arbitrary normalized local quantum amplitudes in the standard descending spin-projection basis $|S,S\rangle,|S,S-1\rangle,\ldots,|S,-S\rangle$. Supplying a `MagneticReferenceState` with `formalism=:SUN` automatically lifts its ordered directions to local highest-weight coherent states. All `SUNReferenceState` instances are site-product states and do not contain intersite entanglement.

```julia
reference = SUNReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
solver = SpinWaveSolver(reference=reference; formalism=:SUN)
result = solve(solver, model)
```

For a spin-1 large-$D$ reference with vanishing dipolar order, use `SUNReferenceState([[0,1,0], [0,1,0]])`; this reference is outside the sphere of fully polarized dipolar spin-coherent states. For the legacy approximation, use `SpinWaveSolver(reference=magnetic_reference)` or explicitly set `formalism=:dipole`.

The optional `tangent_bases` keyword accepts a complete unitary local basis for each site, with the reference in its first column. The frequencies and physical scattering tensors must remain invariant under unitary transformations of the remaining columns.

## Harmonic expansion

For a normalized state $|0_i\rangle$ and orthogonal local states $|\mu_i\rangle$, expand each local operator as $O_i=O_{00}+\sum_\mu(O_{\mu0}b_{i\mu}^\dagger+O_{0\mu}b_{i\mu})+\sum_{\mu\nu}(O_{\mu\nu}-O_{00}\delta_{\mu\nu})b_{i\mu}^\dagger b_{i\nu}+\cdots$. Substitution into the actual Hamiltonian's certified field, tensor-exchange, and symmetric single-ion anisotropy terms produces a Gaussian Hamiltonian $H_2=\sum_{pq}A_{pq}b_p^\dagger b_q+\frac12\sum_{pq}(B_{pq}b_p^\dagger b_q^\dagger+\mathrm{h.c.})$. On-site quadratic spin terms use exact finite-spin operator products in the full local matrix space, not a classical substitution or a dipolar finite-spin factor.

Stationarity is certified by projecting the local effective mean-field operator onto each independent tangent direction. A nonstationary or energetically unstable reference raises an error. The existing bosonic BdG backend is reused; paired singular Goldstone-mode problems still require explicit `zero_modes=:regularize`, and regulated eigenvectors cannot certify physical scattering intensities. Harmonic expansion validity and small tangent-boson depletion remain physical approximation conditions rather than universal guarantees.

## Mode and multipolar observables

`SpinWaveResult` retains the same public fields as in dipolar LSWT. Its `A`, `B`, `modes`, and `frequencies` are indexed by the total number of local flavors $M=\sum_i2S_i$ rather than the number of sites. Metadata includes `:formalism=>:SUN`, `:local_flavors`, `:flavor_offsets`, reference states and bases, local spin matrices, stationary-reference diagnostics, and the product-state approximation certificate.

`sun_operator_matrix_elements(result, i, O)` returns local reference, creation, annihilation, and tangent-sector matrix elements for any operator matrix `O` on site `i`, including quadrupoles such as `Sx*Sx-Sy*Sy`. `sun_mode_matrix_elements(result, [O1,O2,...])` projects a set of local operators onto the Gaussian modes. This provides multipolar excitation matrix elements but does not silently advertise a fully integrated multipolar neutron-scattering cross section.

The existing `SpinTensorStructureFactorRequest` and `spin_tensor_structure_factor` routes compute elastic and one-flavor dipolar Cartesian response from the SU(N) mode vectors in a finite-cell geometry. They do not compute connected longitudinal two-flavor continua or higher-multipolar cross sections. Periodic Bloch-mode dipolar spin tensors are available using the explicitly requested magnetic-supercell Bloch route below. The `renormalize_transverse` dipolar correction is deliberately unavailable in SU(N), where it has no universal meaning.

## Reference-state energy optimization

`optimize_spin_reference(model; formalism=:dipole)` minimizes the coherent-product expectation of the *actual* symbolic spin Hamiltonian on a product of direction spheres. The corresponding `formalism=:SUN` route minimizes the exact local-matrix product-state expectation over normalized local complex amplitudes, including tensor exchange, DM exchange, onsite anisotropy, and vector fields. Both formalisms use projected gradients, an energy-decreasing backtracking line search, and reproducible random multistarts; the default first initialization is the fully $+z$ polarized state. This is local constrained optimization, not certified global ground-state search, finite-temperature dynamics, or harmonic stability checking.

```julia
optimized = optimize_spin_reference(model; formalism=:dipole, starts=8, tol=1e-8, seed=3)
optimized.converged || error("reference optimizer did not reach the requested stationarity tolerance")
solver = SpinWaveSolver(reference=optimized.reference)
result = solve(solver, model)  # Independently checks harmonic stability and any zero modes.
```

A site-SU(N) reference can instead be selected with `formalism=:SUN`; set `initial=SUNReferenceState(...)` to guide the first start. The returned named tuple records `energy`, `residual`, `converged`, `iterations`, `restart`, all per-start diagnostics, `global_minimum_certified=false`, and `harmonic_stability=:not_checked`. A stationary state can still be a metastable local minimum or an unstable saddle, so run `solve` after optimization to test the quadratic problem independently. The optimizer does not modify the physical model and does not authorize a spin-wave approximation in the forward planner.

## Periodic Bloch scattering

`spinwave_dispersion(solver, model, qpoints)` returns a `SpinWaveDispersionResult` retaining the complete positive-norm Bloch vectors for both `:dipole` and `:SUN`. Its modes can now be passed directly to `spin_tensor_structure_factor(model, bands, field, qpoints, axis)` for laboratory-frame spin tensors. The existing `SpinTensorStructureFactorRequest` provides a higher-level route via an *opt-in* metadata key:

```julia
Q = [[0.0], [π / 3]]                     # Cartesian inverse-length momenta
energy = collect(range(-5.0, 5.0; length=501))
request = SpinTensorStructureFactorRequest(Q, energy; metadata=Dict(:spinwave_momentum_scope => :magnetic_supercell_bloch))
calculation = forward(ForwardProblem(model; solver, observable=request))
bands = calculation.solution             # SpinWaveDispersionResult
spectrum = calculation.observable_result # TensorSpectrumResult, 3x3 Cartesian response
```

A request without `:spinwave_momentum_scope` retains the original finite-cell path and returns `SpinWaveResult`. The Bloch request's `Q` must exactly match the calculated qpoints; momentum interpolation is not performed. The Bloch energy matrix uses physical bond-image displacements $\delta$ and phases $e^{i\mathbf q\cdot\delta}$, with the bosons defined in the position-phase gauge; one-magnon eigenvectors already include the magnetic-basis phases and must **not** be multiplied by site phases again. At nonzero temperature, the negative-frequency channel uses independently solved $-Q$ modes, which matters for nonreciprocal DM magnets. Regulated Goldstone modes do not provide certified scattering intensities and are rejected.

The available channels are the one-magnon response and an *elastic basis form factor*, not an infinite-lattice Bragg delta-function distribution. The harmonic result retains bare product-reference moments and does not evaluate thermal/zero-point moment renormalization by integration over the full Brillouin zone. Connected two-particle continua, form factors, polarization factors, domain averaging, powder averaging, and resolution are additional capabilities rather than hidden assumptions.

## Periodic bands and limitations

`spinwave_dispersion(solver, model, qpoints)` supports a periodic magnetic supercell if the model retains certified translated-image bond data in `:spinwave_bonds`. It constructs `A(q)`, `A(-q)` and `B(q)` in the expanded flavor space and uses the existing Bloch BdG diagonalizer. Folded same-index periodic bonds (which require an unfolded magnetic cell), onsite products beyond quadratic spin operators, three-spin and higher intersite terms, entangled-unit SU(D) states, non-Gaussian quantum corrections, and symmetry-unverified single-Q rotating-frame dispersions remain unsupported. Physical Hamiltonian admissibility is independent of any solver's approximation coverage.

## Validation

The regression file `test/sun-spin-wave-validation.jl` checks dipolar/SU(N) spin-half reduction, spin-1 axial anisotropy, zero-dipole large-D local excitations, quadrupolar matrix elements, tangent-space gauge changes, high-spin $(S=5/2)$ multiplets, typed finite-cell scattering, and magnetic-cell Bloch bands. The additional `test/bloch-reference-optimization-validation.jl` checks opt-in forward Bloch tensors, spin-half SU(2)/SU(N) equivalence, explicit qgrid consistency, and constrained product-state reference optimization. Execute both targeted files independently, then the full package suite and documentation build.

```bash
OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Phundamental; using Test; include("test/sun-spin-wave-validation.jl")'
OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Phundamental; using Test; include("test/bloch-reference-optimization-validation.jl")'
OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
```
