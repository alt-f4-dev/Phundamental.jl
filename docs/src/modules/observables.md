```@meta
CurrentModule = Phundamental.Observables
```

# Observables

`Phundamental.Observables` converts solved many-body states or mode calculations into physical correlation functions, spectra, structure factors, and probe-specific scattering intensities. The module preserves a central architectural separation:

$$\text{intrinsic observable}\rightarrow\text{probe contraction}\rightarrow\text{instrument response}.$$

An intrinsic dynamic structure factor is not the same object as a neutron cross section, and numerical line broadening is not the same operation as convolution with an experimental resolution function. The API keeps those stages distinct so each can be inspected and validated independently.

## Observable descriptors and registries

`AbstractObservable` is the root observable abstraction and `AbstractCorrelationObservable` identifies correlation-based quantities. `ObservableRef` addresses a symbolic operator stored in `model.observables`.

`register_observable` returns a copy of a model with a named operator added to its observable registry. `resolve_observable` retrieves the operator in the model's current representation.

This registry is the representation-safe route for transformed models. When an exact transformation maps a `ManyBodyModel`, registered operator observables are transformed with the Hamiltonian. Downstream scattering code can therefore keep referring to the same observable key instead of recreating a source-representation operator after the transformation.

## Momentum fields

`MomentumField` describes a set of local observables and their real-space positions. `momentum_operator` constructs

$$O_{\mathbf q}=a_N\sum_j e^{-i\mathbf q\cdot\mathbf r_j}O_j,$$

where the normalization $a_N$ is selected as `:sqrtN`, `:N`, or `:none`.

`SpinTensorField` stores the three Cartesian local spin fields needed for a full tensor response. `register_spin_tensor` registers $S^x_i$, $S^y_i$, and $S^z_i$ on a model and returns a field that remains valid after an exact representation transformation because the registry entries are mapped with the model.

## Spectrum conventions

`SpectrumConvention` makes Fourier normalization, the spectral-axis convention, $\hbar$, and $k_B$ explicit. The spectral axis may be energy or angular frequency. Any numerical broadening supplied to a spectrum must use the same units as the selected axis.

The convention exists to prevent otherwise silent factor errors. For example, a calculation performed on an energy axis should not apply a width specified in angular-frequency units unless the appropriate $\hbar$ conversion has already been made.

## Correlation functions

`thermal_probabilities` obtains Boltzmann weights for a solved eigensystem at the requested temperature. `eigenbasis_matrix` realizes an observable in the solver eigenbasis.

`correlation_function` evaluates time-domain correlations and returns a `CorrelationResult`. The correlation machinery is built on the common solver-result interface rather than on a particular eigensolver.

For finite-temperature calculations whose natural object is an imaginary-frequency propagator, use [Greens](greens.md) instead. The Observables correlation functions are aimed at physical real-time/equal-time and spectral response quantities.

## Lehmann lines and broadening

`LehmannLines` stores discrete transition centers and weights before numerical line broadening. This intermediate representation is useful for exact finite systems because it preserves the delta-function spectrum independently of how the spectrum is later plotted.

`lehmann_lines` computes the transitions. `AbstractBroadening` is the base for numerical line-shape models; v0.0.7 includes `LorentzianBroadening` and `GaussianBroadening`. `broaden_lines` evaluates the broadened spectrum on a requested axis.

Broadening here is a representation of discrete intrinsic spectral lines. It is not automatically interpreted as experimental resolution.

## Dynamic structure factors

`dynamic_structure_factor` computes a scalar dynamic structure factor from a momentum field or equivalent observable request. `spin_tensor_structure_factor` computes the full Cartesian tensor $S^{\alpha\beta}(\mathbf q,\omega)$.

`SpectrumResult` stores scalar spectra and `TensorSpectrumResult` stores tensor spectra indexed by momentum, spectral point, and Cartesian components. Metadata records conventions, broadening, temperature, and related calculation choices.

`detailed_balance_residual` measures the violation of the expected finite-temperature detailed-balance relation and is an important validation diagnostic.

## Static structure factors

`static_structure_factor` computes equal-time/static response and returns `StaticStructureFactorResult`. `StaticTensorStructureFactorResult` is the tensor counterpart used by spin correlations.

`dynamic_static_sumrule_residual` compares integrated dynamic spectral weight with the static/equal-time quantity when the assumptions of the selected convention apply. `integrated_spectral_weight` performs the corresponding spectral integration.

Static structure factors can also be computed from classical configuration ensembles through `spin_tensor_static_structure_factor`, allowing exact/quantum and Monte Carlo pathways to share output concepts while retaining method metadata.

## Electronic spectral functions

`spectral_density` computes a single-particle spectral density from the relevant fermionic solver result. `cluster_perturbation_spectral_function` evaluates the spectral function associated with the cluster-perturbation Green function produced by [Solvers](solvers.md).

The zero-temperature CPT Green-function workflow in Solvers and the finite-temperature Matsubara Green-function objects in [Greens](greens.md) are separate abstractions. Conversion to a real-frequency spectral function requires an appropriate real-frequency or continuation strategy; v0.0.7 does not silently analytically continue Matsubara data.

## Magnetic neutron scattering

`MagneticNeutronProbe` stores probe-level information used to contract a spin tensor into magnetic neutron intensity. `magnetic_neutron_intensity` applies the transverse polarization projector, site/material magnetic form factors, and the requested prefactor to an intrinsic spin response.

The separation of `SpinTensorField` from `MagneticNeutronProbe` means the same intrinsic tensor can be inspected independently of the neutron contraction or reused by another probe implementation.

## Nuclear and phonon neutron scattering

`NuclearNeutronProbe` describes coherent nuclear scattering lengths and associated probe metadata. `CoherentNeutronProbe` is provided as the coherent-neutron interface used by the harmonic scattering workflow.

`one_phonon_neutron_lines` computes intrinsic coherent one-phonon creation and annihilation lines from a q-resolved `PhononModeResult`. `one_phonon_neutron_intensity` evaluates those lines on a spectral grid.

The high-level full-$Q$ route decomposes the physical scattering vector into crystal momentum plus reciprocal-lattice translation using the reciprocal-space machinery, while retaining the physical $Q$ in polarization and Debye--Waller factors.

`phonon_reciprocal_mesh` constructs the mesh required for multiphonon contractions. `HarmonicMultiphononWorkspace` and `harmonic_multiphonon_workspace` cache the data used by coherent harmonic multiphonon calculations.

`harmonic_coherent_intermediate_scattering` evaluates the harmonic coherent intermediate scattering function. `harmonic_multiphonon_neutron_intensity` constructs the corresponding multiphonon response. Specialized one- and two-phonon window/line functions are provided when explicit low-order decompositions are desired.

`HarmonicNeutronSliceResult`, `harmonic_neutron_slice`, and `converge_harmonic_neutron_slice` provide a higher-level workflow for computing and convergence-testing harmonic neutron slices.

## Bosonic zero modes

Finite-temperature Bose occupations diverge as a mode energy approaches zero, so zero modes require an explicit physical policy rather than a numerical epsilon hidden inside an observable.

`RejectBosonZeroModes` is the default conservative policy and raises an error if a mode lies below the relative zero-mode threshold. `NumberConservingZeroModeProjection` explicitly removes the known condensate/Goldstone direction from mode-native contractions.

When the condensate orbital or null direction is known at the solver level, `BosonSubspaceProjection` in [Solvers](solvers.md) is preferable because it removes the direction before diagonalization. The observable-side projection is intended for a physically justified number-conserving interpretation, not as a generic way to suppress unstable small frequencies.

## Probe dispatch

`scattering_intensity` is the common probe-contraction entry point. It dispatches on the probe and intrinsic result type and returns `ScatteringResult` when a probe-specific intensity has been constructed.

This dispatch is what allows the top-level `forward` workflow to remain generic: the forward layer does not need to know the mathematical details of magnetic versus nuclear neutron contraction.

## Instrument resolution

`AbstractResolution` is the base instrument-response abstraction. `GaussianResolution` defines a normalized Gaussian response with width `sigma` and finite stencil radius controlled by `nsigma`.

`convolve_resolution` applies the instrument response after an intrinsic or probe-contracted spectrum has been calculated. For a uniform spectral grid, the discrete convolution approximates

$$I_{\mathrm{obs}}(x_i)=\int dx'\,R(x_i-x')I_{\mathrm{theory}}(x').$$

The finite-grid boundary is not renormalized per output point. Weight that would lie outside the requested measurement window is therefore lost from that window rather than artificially folded back into it.

!!! note
    Numerical line broadening and instrument resolution are separate operations. Use `broaden_lines` to represent intrinsic discrete lines on a grid and `convolve_resolution` to model a measurement response.

## Ensemble observables

Classical configuration ensembles produced by the solver module can be analyzed through `ensemble_expectation`, `ensemble_internal_energy`, `ensemble_heat_capacity`, and `ensemble_magnetization`. These functions propagate Monte Carlo sampling metadata so statistical uncertainty is not confused with exact quantum expectation values.

## Validation

`validate_observable_equivalence` compares observables calculated through mathematically equivalent representations. This is central to transformation validation because a correct Hamiltonian spectrum is insufficient if physical operators were mapped incorrectly.

`integrated_spectral_weight`, `dynamic_static_sumrule_residual`, and `detailed_balance_residual` provide independent spectral checks. Harmonic neutron workflows additionally expose convergence controls for reciprocal meshes and multiphonon truncation.

## Related modules

- [Models](models.md) supplies geometry, magnetic form factors, coherent scattering lengths, masses, and Debye--Waller information.
- [Solvers](solvers.md) supplies eigensystems, phonon modes, spin-wave modes, CPT Green functions, and classical ensembles.
- [Thermodynamics](thermodynamics.md) supplies finite-temperature state information.
- [Greens](greens.md) supplies finite-temperature imaginary-frequency propagators rather than real-frequency probe response.
- [Scattering Workflow](../workflows/scattering.md) shows the intrinsic/probe/instrument pipeline end to end.

## API reference

### Observable descriptors and results

```@docs
RejectBosonZeroModes
NumberConservingZeroModeProjection
ObservableRef
SpectrumConvention
MomentumField
SpinTensorField
LehmannLines
SpectrumResult
StaticStructureFactorResult
TensorSpectrumResult
CorrelationResult
ScatteringResult
resolve_observable
momentum_operator
register_observable
register_spin_tensor
ObservableRequest
```

### Correlations and spectra

```@docs
thermal_probabilities
eigenbasis_matrix
correlation_function
lehmann_lines
broaden_lines
dynamic_structure_factor
spin_tensor_structure_factor
static_structure_factor
spectral_density
cluster_perturbation_spectral_function
detailed_balance_residual
integrated_spectral_weight
```

### Magnetic neutron response

```@docs
MagneticNeutronProbe
magnetic_neutron_intensity
```

### Nuclear and phonon response

```@docs
NuclearNeutronProbe
one_phonon_neutron_lines
one_phonon_neutron_intensity
phonon_reciprocal_mesh
HarmonicMultiphononWorkspace
harmonic_multiphonon_workspace
harmonic_coherent_intermediate_scattering
harmonic_multiphonon_neutron_intensity
two_phonon_neutron_lines
one_phonon_neutron_window
two_phonon_neutron_window
two_phonon_neutron_intensity
harmonic_neutron_slice
converge_harmonic_neutron_slice
```

### Static ensemble response

```@docs
StaticTensorStructureFactorResult
spin_tensor_static_structure_factor
```

## Additional exported observable API

The remaining exports are the abstract observable/probe/broadening/resolution types, `LorentzianBroadening`, `GaussianBroadening`, `CoherentNeutronProbe`, `scattering_intensity`, `HarmonicNeutronSliceResult`, ensemble thermodynamic observables, `convolve_resolution`, and `dynamic_static_sumrule_residual`.
