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

## Scalar dynamic structure factor

For a scalar field,

```julia
Sqw = dynamic_structure_factor(working_model, solution, field, qgrid, energy_grid; temperature=temperature, convention=convention, broadening=broadening)
```

The result is a `SpectrumResult` containing the momentum grid, spectral grid, intrinsic intensity, and calculation metadata.

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
