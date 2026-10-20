```@meta
CurrentModule = Phundamental.Observables
```

# Observable API — Probe and Instrument Response

## Magnetic neutron response

```@docs
MagneticNeutronProbe
magnetic_neutron_intensity
```

## Nuclear and phonon response

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

## Static ensemble response

```@docs
StaticTensorStructureFactorResult
spin_tensor_static_structure_factor
```

## Additional exported observable API

The remaining exports are the abstract observable/probe/broadening/resolution types, `LorentzianBroadening`, `GaussianBroadening`, `CoherentNeutronProbe`, `scattering_intensity`, `HarmonicNeutronSliceResult`, ensemble thermodynamic observables, `convolve_resolution`, and `dynamic_static_sumrule_residual`.

## Generalized experimental resolution (Milestone C)

`EnergyDependentGaussianResolution`, `TabulatedGaussianResolution`, and `TabulatedEnergyResolution` extend the instrument stage to varying energy widths and tabulated response densities, using true-energy nonuniform quadrature. `GaussianQEResolution` and `SparseQEResolution` support covariance-coupled Q–energy broadening and supplied sparse response weights, respectively. `InstrumentCoverage` and `CoveredResolution` preserve detector validity as a mask instead of rewriting intensities at inaccessible points. See [Experimental Data and Instrument Resolution](../workflows/experimental-data-resolution.md) for conventions and limitations.

```@docs
EnergyDependentGaussianResolution
TabulatedGaussianResolution
TabulatedEnergyResolution
GaussianQEResolution
SparseQEResolution
InstrumentCoverage
CoveredResolution
```

## Extended typed forward requests

The `StaticStructureFactorRequest`, `ThermodynamicCurveRequest`, and `CPTSpectralFunctionRequest` constructors integrate pre-existing static, canonical-ensemble, and cluster-perturbation calculations with typed `ForwardProblem` requests. See [Extended Forward Physics](../workflows/extended-forward-physics.md) for the supported solver routes and physical restrictions.
