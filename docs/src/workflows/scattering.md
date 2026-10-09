# Scattering

Scattering calculations in Phundamental are organized as a sequence of physically distinct contractions. The recommended workflow computes an intrinsic correlation function first, applies the probe cross section second, and convolves the instrument response last.

$$\text{model/solution}\rightarrow S^{\alpha\beta}(\mathbf Q,\omega)\rightarrow I_{\mathrm{probe}}(\mathbf Q,\omega)\rightarrow I_{\mathrm{obs}}(\mathbf Q,\omega).$$

Keeping these stages separate prevents numerical line broadening, magnetic polarization factors, form factors, coherent scattering lengths, Debye--Waller factors, and experimental resolution from being hidden inside one undifferentiated intensity function.

## Register physical operators before a transformation

For a spin model, register the local spin tensor before applying a representation transformation:

```julia
model_with_spin, spin_field = register_spin_tensor(model, 1:N, positions; prefix=:spin, normalization=:sqrtN)
```

If the model is subsequently transformed, every registered operator is transformed with it. `spin_field` continues to refer to the correct physical spin operators through their registry keys.

This is preferable to constructing fresh source-representation `Sx`, `Sy`, or `Sz` operators after a Jordan--Wigner or other representation map.

## Solve the model

Choose a solver appropriate to the representation:

```julia
solution = solve(OptimizedLanczos(nev=32), working_model)
```

For a small validation system use `ExactDiagonalization`; for harmonic phonons use `HarmonicPhononSolver`; for an explicitly requested spin-wave approximation use `LinearSpinWaveSolver`.

The observable layer consumes the solver result through its public interface rather than depending on the internal storage backend.

## Define the spectral convention

A spectral calculation should make normalization and units explicit:

```julia
convention = SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=1.0)
```

The Fourier normalization in the `SpectrumConvention` must agree with the corresponding `MomentumField` or `SpinTensorField` normalization.

## Intrinsic line broadening

Finite exact systems have discrete Lehmann lines. To display or integrate them on a regular spectral grid, choose an intrinsic numerical line shape:

```julia
broadening = GaussianBroadening(0.05)
```

or

```julia
broadening = LorentzianBroadening(0.05)
```

This width describes how intrinsic delta functions are represented numerically. It is not an instrument resolution parameter.

## Generic scalar dynamical correlations

For a same-field correlation of one scalar field, use the one-field generic form:

```julia
Saa = dynamical_correlation(working_model, solution, field, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
```

For a coupled or cross-channel response, provide two independent momentum fields:

```julia
Sab = dynamical_correlation(working_model, solution, field_a, field_b, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
```

The cross-correlation is ordered as $S_{AB}(\mathbf q,x)$ with $A_{\mathbf q}$ and $B_{-\mathbf q}$; the API does not insert an adjoint or conjugation automatically. Both forms return a `SpectrumResult` containing the momentum grid, spectral grid, intrinsic intensity, and calculation metadata.

The same response can enter the high-level forward pipeline through `DynamicalCorrelationRequest`:

```julia
request = DynamicalCorrelationRequest(field_a, field_b, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
result = forward(ForwardProblem(working_model; solver=OptimizedLanczos(nev=32), observable=request))
```

## Arbitrary composite response

A coupled model with several registered physical channels can request the complete ordered response matrix without writing one request per pair. Construct labeled momentum fields and pass them as a `NamedTuple`:

```julia
spin_z = registered_momentum_field(model, [:spin_z_1, :spin_z_2])
displacement = registered_momentum_field(model, [:boson_displacement_1, :boson_displacement_2])
fields = (spin_z=spin_z, displacement=displacement)
request = CompositeDynamicalCorrelationRequest(fields, qgrid, energy_grid; convention=convention, broadening=broadening)
response = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=request)).observable_result
```

`response.intensity[iq, ix, a, b]` is the ordered $S_{ab}(\mathbf q,x)$ associated with `response.channels[a]` and `response.channels[b]`. The matrix therefore retains both cross directions rather than assuming that one is redundant.

For spin-phonon models whose observables follow the standard registry names created by the model builders, `SpinPhononCorrelationRequest` constructs the fields automatically:

```julia
request = SpinPhononCorrelationRequest(qgrid, energy_grid; spin_components=(:x, :y, :z), convention=convention, broadening=broadening)
response = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=request)).observable_result
```

The resulting channel order is the selected spin components followed by `:phonon_displacement`, so one intrinsic result contains the spin-spin, spin-phonon, phonon-spin, and phonon-phonon blocks before any probe-specific contraction.

## Scalar dynamic structure factor

The existing `dynamic_structure_factor` API is retained as the one-field structure-factor specialization:

```julia
Sqw = dynamic_structure_factor(working_model, solution, field, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
```

Its numerical spectrum is the same as the corresponding one-field `dynamical_correlation`; the legacy observable metadata remains `:dynamic_structure_factor`.

## Spin-tensor dynamic structure factor

Magnetic neutron scattering should normally start from the full Cartesian tensor:

```julia
S_tensor = spin_tensor_structure_factor(working_model, solution, spin_field, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
```

The output is `TensorSpectrumResult` with components $S^{\alpha\beta}(\mathbf Q,E)$.

Computing the tensor before the probe contraction preserves the information needed to change polarization geometry, $g$ tensors, or form factors without repeating the eigensystem calculation.

## Magnetic neutron contraction

Construct the probe separately:

```julia
probe = MagneticNeutronProbe(form_factor=Q -> 1.0, g_tensor=Matrix{Float64}(I, 3, 3), prefactor=1.0)
```

Then contract the tensor:

```julia
I_neutron = scattering_intensity(probe, S_tensor)
```

For nonzero scattering vector $\mathbf Q$, the magnetic contraction uses the transverse polarization tensor

$$P_{\alpha\beta}(\mathbf Q)=\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta,$$

and applies the supplied $g$ tensor and magnetic form factor. The absolute prefactor remains user-controlled because absolute cross-section units depend on the selected physical unit convention.

## Instrument resolution

Apply the instrumental response only after the intrinsic/probe response is available:

```julia
resolution = GaussianResolution(0.08)
I_observed = convolve_resolution(resolution, I_neutron)
```

The current optimized resolution convolution expects a strictly increasing uniform spectral grid. The finite-grid boundary is not renormalized per point, so intensity can leave a finite measurement window instead of being artificially returned to the edge.

## Sum-rule diagnostics

For a finite exact calculation, useful checks include:

```julia
weight = integrated_spectral_weight(Sqw)
residual = dynamic_static_sumrule_residual(...)
```

At finite temperature, also evaluate `detailed_balance_residual` when the input grid and convention permit the corresponding positive/negative-energy comparison.

Transformation equivalence can be tested with `validate_observable_equivalence` by comparing the same physical response before and after an exact representation map.

## Harmonic one-phonon scattering

A harmonic phonon workflow begins with a `HarmonicPhononSolver` result and a physical scattering vector $\mathbf Q$:

```julia
phonons = solve(HarmonicPhononSolver(), phonon_model)
probe = NuclearNeutronProbe()
```

For a q-resolved mode result, `one_phonon_neutron_lines` returns intrinsic creation/annihilation lines. The higher-level `one_phonon_neutron_intensity` route evaluates them on a grid.

The physical $\mathbf Q$ enters polarization projections and Debye--Waller factors. Crystal momentum is obtained from reciprocal-space decomposition when an extended-zone scattering vector is supplied.

## Zero-mode policy for bosons

Finite-temperature Bose factors require an explicit treatment of zero-energy modes. The default `RejectBosonZeroModes` raises an error rather than evaluating a divergent occupation.

If the omitted zero mode is physically known to be a condensate or Goldstone direction in a number-conserving formulation, `NumberConservingZeroModeProjection` can remove it from mode-native contractions. When the null direction is known before diagonalization, use `BosonSubspaceProjection` in the solver layer instead.

Do not use zero-mode projection merely to make an unstable phonon calculation return finite values; first determine whether the mode is physical, symmetry-protected, or a force-constant defect.

## Multiphonon scattering

For coherent harmonic multiphonon response, construct a reciprocal mesh and reusable workspace:

```julia
mesh = phonon_reciprocal_mesh(...)
workspace = harmonic_multiphonon_workspace(...)
```

`harmonic_coherent_intermediate_scattering` evaluates the intermediate scattering function, while `harmonic_multiphonon_neutron_intensity` produces the frequency-domain response. Explicit one- and two-phonon window functions are available when order-resolved contributions are required.

`harmonic_neutron_slice` and `converge_harmonic_neutron_slice` provide a higher-level route for a complete slice and convergence study.

## Classical static scattering

A classical spin ensemble produced by `MetropolisSampler` can be converted to `StaticTensorStructureFactorResult` with `spin_tensor_static_structure_factor` and then contracted with `MagneticNeutronProbe` through the same `scattering_intensity` interface.

The resulting static Monte Carlo intensity and a quantum dynamic spectrum share probe semantics but retain different result types and metadata, preventing the sampling approximation from being hidden.

## Recommended reporting

A reproducible scattering result should state the model and parameter values, representation/transformation, solver or sampling method, momentum convention, Fourier normalization, spectral-axis units, temperature and $k_B$, intrinsic line broadening, probe form factors/$g$ tensors/scattering lengths, Debye--Waller treatment, zero-mode policy if applicable, instrument resolution, and convergence parameters such as boson cutoff or reciprocal mesh.

## See also

- [Observables](../modules/observables.md) for the full intrinsic/probe/resolution API.
- [Models](../modules/models.md) for material scattering properties and phonon force constants.
- [Solvers](../modules/solvers.md) for exact, mode, spin-wave, and Monte Carlo backends.
- [Representation Transformations](representation-transformations.md) for preserving registered physical observables across exact maps.
- [Validation](../validation.md) for scattering and mode benchmark coverage.
