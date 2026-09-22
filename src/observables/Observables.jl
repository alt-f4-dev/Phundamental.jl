module Observables

using LinearAlgebra
using ..Core
using ..Solvers

include("ObservableTypes.jl")
include("CorrelationFunctions.jl")
include("DynamicStructureFactor.jl")
include("StaticStructureFactor.jl")
include("SpectralFunctions.jl")
include("ElectronSpectra.jl")
include("Scattering.jl")
include("HarmonicMultiphonon.jl")
include("Resolution.jl")
include("Validation.jl")

export AbstractObservable,
       AbstractCorrelationObservable,
       AbstractProbe,
       AbstractBroadening,
       AbstractResolution,
       AbstractBosonZeroModePolicy,
       RejectBosonZeroModes,
       NumberConservingZeroModeProjection,
       ObservableRef,
       SpectrumConvention,
       MomentumField,
       SpinTensorField,
       LehmannLines,
       SpectrumResult,
       StaticStructureFactorResult,
       TensorSpectrumResult,
       CorrelationResult,
       ScatteringResult,
       resolve_observable,
       register_observable,
       register_spin_tensor,
       momentum_operator,
       spin_tensor_operators,
       thermal_probabilities,
       eigenbasis_matrix,
       correlation_function,
       LorentzianBroadening,
       GaussianBroadening,
       lehmann_lines,
       broaden_lines,
       dynamic_structure_factor,
       spin_tensor_structure_factor,
       static_structure_factor,
       spectral_density,
       cluster_perturbation_spectral_function,
       detailed_balance_residual,
       MagneticNeutronProbe,
       magnetic_neutron_intensity,
       NuclearNeutronProbe,
       CoherentNeutronProbe,
       one_phonon_neutron_lines,
       one_phonon_neutron_intensity,
       phonon_reciprocal_mesh,
       HarmonicMultiphononWorkspace,
       harmonic_multiphonon_workspace,
       harmonic_coherent_intermediate_scattering,
       harmonic_multiphonon_neutron_intensity,
       one_phonon_neutron_window,
       two_phonon_neutron_lines,
       two_phonon_neutron_window,
       two_phonon_neutron_intensity,
       HarmonicNeutronSliceResult,
       harmonic_neutron_slice,
       converge_harmonic_neutron_slice,
       scattering_intensity,
       GaussianResolution,
       convolve_resolution,
       validate_observable_equivalence,
       integrated_spectral_weight,
       dynamic_static_sumrule_residual

end # module Observables
