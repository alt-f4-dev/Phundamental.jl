module Phundamental

# =============================================================================
# Phundamental.jl
#
# Top-level package module for the representation-aware many-body framework.
#
# Directory layout expected relative to this file:
#
#   Phundamental.jl        canonical entry point and public exports
#   core/                 foundational state spaces, algebra, representations, model spaces
#   algebra/              symbolic manipulation and canonicalization
#   crystallography/      space-group actions and invariant generation
#   transformations/      representation transformations and certification
#   solvers/              numerical solvers and optimized backends
#   observables/          correlation, scattering, and instrumental response
#   models/               geometry, materials, and Hamiltonian construction
#   thermodynamics/       thermal states and linked-cluster expansions
#   greens/               finite-temperature Green functions and analytic continuation
#   embedding/            impurity, DMFT, diagrammatic methods, geometric embedding
#   forward/              forward routes, solver compatibility, planning, and results
#   experimental/         portable predictions, datasets, and measurement comparison
#   api/                  cross-layer convenience constructors
#
# IMPORTANT:
# Julia already defines the built-in module `Core`.  Therefore the implementation
# files stored in the directory `core/` are loaded here into the submodule
# `PhundamentalCore`, rather than redefining a module named `Core`.
#
# `Foundation` is provided as a convenient alias:
#
#     Phundamental.Foundation === Phundamental.PhundamentalCore
#
# Implementation files are loaded under the authoritative inline submodules.
# Obsolete standalone directory wrappers were retired in Milestone E.
# =============================================================================


# =============================================================================
# 1. Fundamental mathematical objects
# =============================================================================

module PhundamentalCore

using LinearAlgebra

include("core/StateSpaces.jl")
include("core/Algebras.jl")
include("core/Operators.jl")
include("core/Representations.jl")
include("core/ModelStructures.jl")
include("core/ModelSpaces.jl")
include("core/ModelContexts.jl")
include("core/Models.jl")
include("core/ReciprocalSpace.jl")
include("core/Parameters.jl")

# State spaces
export AbstractStateSpace,
       AbstractHilbertSpace,
       AbstractFockSpace,
       SpinHilbertSpace,
       FermionFockSpace,
       BosonFockSpace,
       CoordinateHilbertSpace,
       CompositeStateSpace,
       BiorthogonalSpace,
       PhysicalSubspace,
       NumericalTruncation,
       spacedimension,
       physicaldimension

# Operator algebras
export AbstractOperatorAlgebra,
       SpinAlgebra,
       BosonAlgebra,
       FermionAlgebra,
       MajoranaAlgebra,
       CoordinateMomentumAlgebra,
       CompositeAlgebra,
       algebra_kind

# Noncommutative operator expressions
export AbstractOperatorExpr,
       AbstractPrimitiveOperator,
       IdentityOperator,
       OperatorSum,
       OperatorProduct,
       ScaledOperator,
       SpinX,
       SpinY,
       SpinZ,
       SpinPlus,
       SpinMinus,
       BosonAnnihilate,
       BosonCreate,
       FermionAnnihilate,
       FermionCreate,
       MajoranaOperator,
       PositionOperator,
       MomentumOperator,
       NumberOperator,
       Sx,
       Sy,
       Sz,
       Sp,
       Sm,
       b,
       c,
       γ,
       xop,
       pop,
       nb,
       nf,
       FactorOperator,
       atfactor,
       isprimitive,
       children,
       operator_degree,
       support_size,
       body_order

# Bases and representations
export AbstractBasis,
       SpinProductBasis,
       FermionOccupationBasis,
       BosonOccupationBasis,
       CoordinateBasis,
       CompositeBasis,
       BiorthogonalBasis,
       AbstractWaveVector,
       CartesianWaveVector,
       ReciprocalWaveVector,
       wavevector_coordinates,
       ReciprocalDecomposition,
       decompose_reciprocal_vector,
       GaugeStructure,
       Representation,
       isconstrained,
       istruncated,
       ambientdimension,
       representationdimension,
       physicalspace,
       FactorConstraint,
       FactorTruncation,
       factorconstraint,
       factortruncation

# Canonical Hamiltonian structures
export AbstractModelStructure,
       StructuralSupport,
       StructuralTerm,
       InteractionEdge,
       InteractionGraph,
       StructuralSignature,
       ModelStructure,
       representation_signature,
       structure,
       canonicalize_structure,
       structural_signature,
       interaction_graph,
       structurally_equivalent,
       structural_span_membership,
       structural_span_coordinates

# Hamiltonian model spaces and typed parameters
export AbstractModelSpace,
       AbstractAdmissibilityCondition,
       AdmissibilityCheck,
       AdmissibilityReport,
       PredicateAdmissibility,
       admissibility_check,
       admissibility_report,
       admissibility_reasons,
       GenerativeSpecification,
       HamiltonianBasis,
       HamiltonianSpace,
       admissible,
       model_space_dimension,
       generated_basis,
       generate_hamiltonian_space,
       hamiltonian_from_coefficients,
       ParameterSpec,
       AbstractParameterConstraint,
       FixedParameterConstraint,
       EqualParameterConstraint,
       LinearParameterConstraint,
       ParameterConstraintCheck,
       ParameterConstraintReport,
       ParameterBinding,
       ParameterSpace,
       ParameterPoint,
       parameter_space,
       parameter_point,
       parameter_names,
       parameter_bounds,
       parameter_units,
       parameter_vector,
       free_parameter_indices,
       parameter_constraints,
       parameter_constraint_check,
       parameter_constraint_report,
       parameter_constraint_reasons,
       valid_parameter_point,
       parameter_constraint_matrix,
       parameter_dof,
       parameter_bindings,
       parameter_binding,
       rebind_parameters,
       with_parameters

# Models
export ComponentIndexLayout,
       component_index,
       component_site,
       component_coordinates,
       GeometryContext,
       PhysicalModelContext,
       ObservableContext,
       observable_context,
       model_context,
       geometry_context,
       ManyBodyModel,
       with_provenance,
       remap_model,
       with_truncation,
       instantiate_model

end # module PhundamentalCore

# Public semantic alias for the fundamental layer.
const Foundation = PhundamentalCore


# =============================================================================
# 2. Operator algebra manipulation
# =============================================================================

module Algebra

using ..PhundamentalCore
import ..PhundamentalCore: canonical_model_expression

include("algebra/Commutators.jl")
include("algebra/RewriteRules.jl")
include("algebra/Canonicalization.jl")
include("algebra/StructuralCanonicalization.jl")

export commutator,
       anticommutator,
       primitive_commutator,
       primitive_anticommutator,
       rewrite_once,
       normal_order,
       canonicalize,
       operator_family

end # module Algebra


# =============================================================================
# 3. Crystallographic symmetry and invariant Hamiltonian generation
# =============================================================================

include("crystallography/Crystallography.jl")


# =============================================================================
# 4. Representation transformations
# =============================================================================

module Transformations

using LinearAlgebra
using ..PhundamentalCore
using ..Algebra

include("transformations/TransformationTypes.jl")
include("transformations/TransformationResult.jl")
include("transformations/Approximations.jl")

include("transformations/Fourier.jl")
include("transformations/Bogoliubov.jl")
include("transformations/ParticleHole.jl")
include("transformations/Majorana.jl")
include("transformations/JordanWigner.jl")
include("transformations/HolsteinPrimakoff.jl")
include("transformations/DysonMaleev.jl")
include("transformations/SchwingerBoson.jl")
include("transformations/AbrikosovFermion.jl")
include("transformations/SlaveParticle.jl")
include("transformations/CoordinateLadder.jl")
include("transformations/BosonicDisplacement.jl")
include("transformations/LangFirsov.jl")
include("transformations/SpinPolaron.jl")

include("transformations/Registry.jl")
include("transformations/TransformationGraph.jl")

export AbstractTransformation,
       RelationType,
       UnitaryEquivalence,
       CanonicalMap,
       AlgebraIsomorphism,
       ConstrainedEmbedding,
       SimilarityEquivalence,
       GaugeRedundantEmbedding,
       CompositeRelation,
       TransformationCertificate,
       TransformationResult,
       SqrtNumberFactor,
       BosonicDisplacementFactor,
       transformation_name,
       certificate,
       applicable,
       target_representation,
       transform_generator,
       transform_operator,
       transform_hamiltonian,
       transform_observable,
       transform,
       structurally_related,
       compose,
       CompositeTransformation,
       has_inverse,
       inverse,
       FourierTransformation,
       BogoliubovTransformation,
       ParticleHoleTransformation,
       MajoranaTransformation,
       JordanWigner,
       HolsteinPrimakoff,
       DysonMaleev,
       SchwingerBoson,
       AbrikosovFermion,
       SlaveParticleTransformation,
       CoordinateLadder,
       BosonicDisplacement,
       LangFirsov,
       SpinPolaron,
       TRANSFORMATION_REGISTRY,
       register_transformation!,
       transformation_constructor,
       make_transformation,
       available_transformations,
       AbstractApproximation,
       ApproximationCertificate,
       ApproximationResult,
       ModelSpaceProjection,
       approximation_name,
       approximation_certificate,
       approximation_applicable,
       apply_approximation,
       transformation_closure_defect,
       TransformationEdge,
       RepresentationGraph,
       add_representation!,
       add_transformation!,
       outgoing_edges,
       transformation_path,
       representation_class,
       transform_along

end # module Transformations


# =============================================================================
# 5. Numerical realization and many-body solvers
# =============================================================================

module Solvers

using LinearAlgebra
using SparseArrays
using Random
using ..PhundamentalCore
using ..Algebra
using ..Transformations

include("solvers/SolverTypes.jl")
include("solvers/SolverResult.jl")
include("solvers/BasisConstruction.jl")
include("solvers/NumericalCapabilities.jl")
include("solvers/MatrixRealization.jl")
include("solvers/ExactDiagonalization.jl")
include("solvers/Lanczos.jl")
include("solvers/TimeEvolution.jl")
include("solvers/Validation.jl")
include("solvers/optimized/Optimized.jl")
include("solvers/Classical/Classical.jl")

export AbstractSolver,
       solve,
       energy_spectrum,
       solver_name,
       SolverResult,
       EnergySpectrumResult,
       TimeEvolutionResult,
       nlevels,
       states,
       eigenstate,
       groundenergy,
       groundstate,
       isbiorthogonal,
       AbstractComputationalState,
       SpinBasisState,
       FermionBasisState,
       BosonBasisState,
       CompositeBasisState,
       ComputationalBasis,
       computational_basis,
       basis_dimension,
       basis_state,
       NumericalCapabilities,
       TransformationCapabilities,
       numerical_capabilities,
       transformation_capabilities,
       MatrixRealizationResult,
       realize,
       matrix,
       hamiltonian_matrix,
       SymbolicLinearOperator,
       ExactDiagonalization,
       LanczosSolver,
       TimeEvolutionSolver,
       OptimizedLanczos,
       AutoSolver,
       AbstractSector,
       SpinSector,
       FermionSector,
       QuantumNumber,
       QuantumNumberSector,
       CompositeSector,
       quantum_numbers,
       sector_value,
       has_quantum_number,
       ProjectedFermionResolvent,
       FermionGreenWorkspace,
       fermion_transition_vector,
       fermion_green_workspace,
       cluster_green_matrix,
       cpt_intercluster_hopping,
       cluster_perturbation_green_matrix,
       cpt_periodize,
       cluster_perturbation_green,
       AbstractOptimizedBackend,
       PackedSpinBackend,
       PackedFermionBackend,
       PackedBosonBackend,
       QuadraticFermionSolver,
       QuadraticBosonSolver,
       BosonSubspaceProjection,
       orthogonal_boson_subspace,
       HarmonicPhononSolver,
       LinearSpinWaveSolver,
       AbstractModeResult,
       QuadraticModeResult,
       PhononModeResult,
       PhononDispersionResult,
       SpinWaveResult,
       optimized_backend,
       supports_optimized_backend,
       optimized_matrix_free_operator,
       spin_sector_profile,
       spin_sector_plan,
       mode_energies,
       mode_vectors,
       dynamical_matrix,
       dynamical_matrix!,
       dynamical_gradient,
       dynamical_gradient!,
       dynamical_hessian,
       dynamical_hessian!,
       acoustic_sum_rule_residual,
       group_velocities,
       directional_group_velocities,
       phonon_dispersion,
       phonon_zero_point_energy,
       spinwave_zero_point_correction,
       spinwave_ground_energy,
       ContinuedFractionResult,
       continued_fraction,
       continued_fraction_value,
       continued_fraction_spectrum,
       exp_action,
       validate_optimized_against_generic,
       validate_packed_spin_implicit_basis,
       validate_packed_spin_rank_lookup,
       validate_spin_sector_decomposition,
       validate_phonon_modes,
       validate_phonon_dispersion,
       validate_mode_result,
       validate_spinwave_result,
       AbstractClassicalProblem,
       AbstractClassicalSampler,
       AbstractConfigurationEnsemble,
       IsingProblem,
       MetropolisSampler,
       ConfigurationEnsemble,
       sample,
       nsamples,
       configuration,
       ising_energy,
       ising_flip_delta,
       autocorrelation_time,
       effective_sample_size,
       correlated_stderr,
       validate_metropolis_detailed_balance,
       validate_sampling_against_exact_classical,
       evolve,
       validate_matrix_realization

end # module Solvers


# =============================================================================
# 5. Correlations, spectra, scattering, and resolution
# =============================================================================

module Observables

using LinearAlgebra
using ..PhundamentalCore
using ..Solvers

include("observables/ObservableTypes.jl")
include("observables/CorrelationFunctions.jl")
include("observables/DynamicStructureFactor.jl")
include("observables/CompositeResponses.jl")
include("observables/ForwardRequests.jl")
include("observables/ExtendedForwardRequests.jl")
include("observables/RequestRequirements.jl")
include("observables/SpinWaveStructureFactor.jl")
include("observables/StaticStructureFactor.jl")
include("observables/SpectralFunctions.jl")
include("observables/ElectronSpectra.jl")
include("observables/ResponseConversions.jl")
include("observables/Scattering.jl")
include("observables/HarmonicMultiphonon.jl")
include("observables/EnsembleObservables.jl")
include("observables/Resolution.jl")
include("observables/GeneralResolution.jl")
include("observables/Validation.jl")

export AbstractObservable,
       AbstractCorrelationObservable,
       AbstractProbe,
       AbstractBroadening,
       AbstractResolution,
       AbstractBosonZeroModePolicy,
       RejectBosonZeroModes,
       NumberConservingZeroModeProjection,
       ObservableRef,
       ObservableRequest,
       DynamicalCorrelationRequest,
       CompositeDynamicalCorrelationRequest,
       SpinPhononCorrelationRequest,
       SpinTensorStructureFactorRequest,
       StaticStructureFactorRequest,
       ThermodynamicCurveRequest,
       MagneticSusceptibilityRequest,
       MagnetizationCurveRequest,
       MagneticResponseCurve,
       CPTSpectralFunctionRequest,
       HarmonicMultiphononSliceRequest,
       LocalMatsubaraGreenFunctionRequest,
       LocalMatsubaraSelfEnergyRequest,
       MatsubaraSusceptibilityRequest,
       RequestRequirements,
       request_requirements,
       is_dynamical_request,
       SpectrumConvention,
       MomentumField,
       SpinTensorField,
       LehmannLines,
       SpectrumResult,
       StaticStructureFactorResult,
       TensorSpectrumResult,
       CompositeSpectrumResult,
       CorrelationResult,
       ScatteringResult,
       resolve_observable,
       registered_momentum_field,
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
       dynamical_correlation,
       composite_dynamical_correlation,
       spin_phonon_dynamical_correlation,
       dynamic_structure_factor,
       spin_tensor_structure_factor,
       static_structure_factor,
       spectral_density,
       spectral_function_from_retarded_green,
       commutator_spectral_density,
       correlation_spectrum_from_retarded_susceptibility,
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
       StaticTensorStructureFactorResult,
       ensemble_expectation,
       ensemble_internal_energy,
       ensemble_heat_capacity,
       ensemble_magnetization,
       spin_tensor_static_structure_factor,
       GaussianResolution,
       EnergyDependentGaussianResolution,
       TabulatedGaussianResolution,
       TabulatedEnergyResolution,
       GaussianQEResolution,
       SpatialGaussianQEResolution,
       SparseQEResolution,
       InstrumentCoverage,
       CoveredResolution,
       convolve_resolution,
       validate_observable_equivalence,
       integrated_spectral_weight,
       dynamic_static_sumrule_residual

end # module Observables


# =============================================================================
# 6. Crystal geometry, materials, and Hamiltonian constructors
# =============================================================================

module Models

using SHA

using LinearAlgebra
using ..PhundamentalCore
import ..PhundamentalCore: instantiate_model, structural_signature, structure, interaction_graph
import ..Algebra: canonicalize
import ..Crystallography
using ..Crystallography: SpaceGroupOperation, space_group_operations, expand_symmetry_orbit

# Geometry
include("models/Geometry/Lattices.jl")
include("models/Geometry/CrystalStructures.jl")
include("models/Geometry/MixedCrystalStructures.jl")
include("models/Geometry/CrystalBasisExpansion.jl")
include("models/Geometry/LocalFrames.jl")
include("models/Geometry/ReciprocalSpace.jl")
include("models/Geometry/NeighborLists.jl")

# Reusable long-range interaction laws and summation strategies
include("models/Interactions/Interactions.jl")

# Material metadata and scattering properties
include("models/Materials/AtomicData.jl")
include("models/Materials/Species.jl")
include("models/Materials/ScatteringProperties.jl")
include("models/Materials/Material.jl")
include("models/Materials/MixedMaterials.jl")

# Hamiltonian specifications and constructors
include("models/Hamiltonians/HamiltonianTypes.jl")
include("models/Hamiltonians/SpinModels.jl")
include("models/Hamiltonians/FermionModels.jl")
include("models/Hamiltonians/ImpurityModels.jl")
include("models/Hamiltonians/PhononModels.jl")
include("models/Hamiltonians/HybridModels.jl")
include("models/Hamiltonians/UniversalGeneration.jl")
include("models/Hamiltonians/AcousticSumRules.jl")
include("models/Hamiltonians/StructuralSearch.jl")
include("models/Hamiltonians/StructuralClosure.jl")
include("models/Hamiltonians/PhysicalInteractions.jl")
include("models/Hamiltonians/ModelContexts.jl")
include("models/Hamiltonians/StructuralLocalization.jl")

include("models/Validation.jl")

# Geometry
export BravaisLattice,
       lattice_dimension,
       direct_matrix,
       reciprocal_matrix,
       cell_measure,
       fractional_to_cartesian,
       cartesian_to_fractional,
       BasisSite,
       AbstractOccupancy,
       SpeciesOccupancy,
       MixedOccupancy,
       MixedBasisSite,
       CrystalStructure,
       MixedCrystalStructure,
       expand_crystal_basis,
       occupancy,
       CrystalSite,
       Supercell,
       disordered_supercell,
       nsites,
       site_position,
       site_positions,
       site_index,
       reciprocal_vector,
       reciprocal_coordinates,
       Bond,
       neighbor_bonds,
       bonds_in_shell,
       bond_cluster,
       coordination_numbers,
       LocalFrame,
       LocalFrameField,
       local_x,
       local_y,
       local_z,
       local_frame,
       local_frames,
       local_z_axes,
       global_vector,
       local_vector,
       global_tensor,
       local_tensor,
       global_pair_tensor,
       local_pair_tensor,
       frame_moment_vectors,
       validate_local_frames,
       AbstractLongRangeInteraction,
       AbstractInteractionSummation,
       CoulombInteraction,
       DipolarInteraction,
       RealSpaceCutoff,
       EwaldSummation,
       ScalarInteractionResult,
       TensorInteractionResult,
       interaction_matrix,
       interaction_tensor,
       interaction_energy,
       project_to_local_frames,
       ising_coupling_matrix,
       ising_zeeman_field,
       dipolar_hamiltonian,
       validate_long_range_result,
       validate_ewald_alpha_independence,
       AbstractHarmonicInteraction,
       HarmonicBondInteraction,
       HarmonicVectorCoordinate,
       HarmonicBendingInteraction,
       HarmonicAngleInteraction,
       harmonic_coordinate_vector,
       bending_force_constant_matrix,
       force_constants

# Materials
export AtomicSpecies,
       lookup_species,
       AbstractMagneticFormFactor,
       ConstantFormFactor,
       GaussianFormFactor,
       form_factor,
       DebyeWallerTensor,
       debye_waller_factor,
       SiteProperties,
       Material,
       atomic_species,
       basis_properties,
       site_properties,
       site_mass,
       site_spin,
       site_g_tensor,
       site_form_factor,
       site_scattering_length,
       mean_mass,
       mean_scattering_length,
       MeanSquareDisplacementResult,
       mean_square_displacement,
       phonon_debye_waller_tensors

# Hamiltonian models
export AbstractHamiltonianSpec,
       build_model,
       model_positions,
       model_cluster,
       mode_index,
       HeisenbergModel,
       XXZModel,
       HubbardModel,
       AndersonImpurityModel,
       impurity_bath_sites,
       ForceConstantBlock,
       RealSpaceForceConstants,
       project_force_constants,
       enforce_acoustic_sum_rule,
       PhononUnitConvention,
       HarmonicPhononModel,
       harmonic_model,
       LocalPhononModel,
       HolsteinModel,
       LongitudinalSpinBosonModel,
       HamiltonianGenerationResult,
       material_crystallographic_structure,
       material_generative_specification,
       acoustic_sum_rule_constraints,
       rotational_sum_rule_constraints,
       generate_hamiltonian_family,
       InteractionGroup,
       OperatorOrbit,
       InteractionSpecification,
       ExchangeOrbit,
       FieldOrbit,
       HoppingOrbit,
       DensityInteractionOrbit,
       PairingOrbit,
       physical_operator_sum,
       PhysicalCoefficientMap,
       InteractionGenerationResult,
       PhysicalCouplings,
       generate_interaction_family,
       physical_parameters,
       InvariantDirectionSignature,
       StructuralFamilySignature,
       StructuralUnit,
       StructuralFamily,
       StructuralSelectionSignature,
       StructuralSelection,
       StructuralMove,
       StructuralNeighborhood,
       StructuralCoordinateMap,
       ModelSpaceHierarchy,
       invariant_direction_signature,
       structural_family,
       structural_selection,
       structural_transition,
       structural_move,
       apply_structural_move,
       structural_neighbors,
       enumerate_structures,
       model_space_hierarchy,
       parent_spaces,
       child_spaces,
       is_subspace,
       structural_coordinate_map,
       embed_coordinates,
       project_coordinates,
       embed_parameters,
       project_parameters,
       parameter_subspace,
       validate_structural_representation,
       structural_id,
       describe,
       reconstruct_selection,
       reconstruct_model,
       generation_context,
       default_observable_context,
       StructuralLocation,
       locate_in_family,
       validate_lattice,
       validate_bonds,
       validate_material_model,
       ForceConstantValidationResult,
       validate_force_constants

end # module Models

include("api/ConvenienceAPI.jl")

# =============================================================================
# 7. Canonical thermodynamics, thermal sampling, and linked clusters
# =============================================================================

module Thermodynamics

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf

using ..PhundamentalCore
using ..Solvers
import ..Solvers: solve, solver_name
using ..Models
using ..Observables: ObservableRef, resolve_observable

include("thermodynamics/ThermalTypes.jl")
include("thermodynamics/Ensembles.jl")
include("thermodynamics/DensityMatrices.jl")
include("thermodynamics/PartitionFunctions.jl")
include("thermodynamics/FreeEnergy.jl")
include("thermodynamics/ThermalSampling.jl")
include("thermodynamics/SectorSampling.jl")
include("thermodynamics/ThermalObservables.jl")
include("thermodynamics/ClusterTypes.jl")
include("thermodynamics/ClusterGeneration.jl")
include("thermodynamics/LinkedCluster.jl")
include("thermodynamics/ClusterSampling.jl")
include("thermodynamics/nlce/NLCE.jl")
include("thermodynamics/Validation.jl")

export AbstractThermalMethod,
       AbstractThermalState,
       ExactGibbs,
       ThermalTypicality,
       ThermalLanczos,
       CanonicalTPQ,
       SectorDecomposition,
       ThermalWorkspace,
       SectorThermalWorkspace,
       thermal_workspace,
       ThermalEstimate,
       ThermodynamicPoint,
       ThermodynamicCurve,
       ExactGibbsState,
       SampledThermalState,
       thermal_state,
       thermal_curve,
       density_matrix,
       density_matrix_trace_error,
       partition_function,
       log_partition_function,
       free_energy,
       internal_energy,
       entropy,
       heat_capacity,
       expectation,
       thermal_variance,
       ThermalCluster,
       cluster_order,
       cluster_sites,
       cluster_bonds,
       connected_clusters,
       linked_cluster_weights,
       ClusterThermodynamicsResult,
       cluster_thermodynamics,
       AbstractClusterExpansion,
       FiniteReferenceExpansion,
       TopologicalClusterClass,
       TopologicalExpansion,
       NLCEProblem,
       NLCESolver,
       NLCEResult,
       linked_cluster_values,
       topological_nlce,
       expectation_nlce_problem,
       nlce_convergence,
       validate_nlce_result,
       NLCEClusterGraph,
       topological_expansion,
       site_nlce_expansion,
       TopologicalNLCEProblem,
       validate_embedding_stability,
       AbstractNLCEResummation,
       BareNLCE,
       WynnEpsilon,
       NLCEResummationResult,
       resum_nlce,
       nlce_sequence_residual,
       LiteratureBenchmark,
       LiteratureBenchmarkResult,
       compare_literature_benchmark,
       rigol_2007_square_site_counts,
       summarize_topological_expansion,
       validate_rigol_2007_square_site_counts,
       bulk_estimate,
       validate_thermal_state,
       validate_sampling_against_exact,
       validate_exact_spectrum_thermodynamics,
       validate_sector_thermodynamic_recombination,
       validate_linked_cluster_closure

end # module Thermodynamics


# =============================================================================
# 8. Frequency-dependent propagators, Nambu structure, and Dyson relations
# =============================================================================

module Greens

using LinearAlgebra
using Statistics
using ..PhundamentalCore
using ..Solvers: matrix
using ..Thermodynamics: ExactGibbs, ExactGibbsState, thermal_state

include("greens/FrequencyAxes.jl")
include("greens/GreenFunctionTypes.jl")
include("greens/Matsubara.jl")
include("greens/Lehmann.jl")
include("greens/Nambu.jl")
include("greens/Dyson.jl")
include("greens/Moments.jl")
include("greens/two_particle/TwoParticle.jl")
include("greens/Validation.jl")

export AbstractFrequencyAxis,
       FermionicMatsubaraAxis,
       matsubara_frequency,
       matsubara_index,
       inverse_temperature,
       axis_temperature,
       AbstractFrequencyMatrix,
       GreenFunction,
       NonInteractingGreenFunction,
       SelfEnergy,
       HybridizationFunction,
       frequency_axis,
       frequency_values,
       component_labels,
       matrix_dimension,
       scalar_frequency_values,
       noninteracting_green,
       zero_self_energy,
       lehmann_green,
       matsubara_green,
       NambuIndex,
       nambu_green,
       nambu_index,
       normal_block,
       anomalous_block,
       hole_block,
       self_energy,
       dyson_green,
       high_frequency_moment,
       high_frequency_moments,
       high_frequency_moment_residual,
       zeroth_moment_residual,
       causality_residual,
       matsubara_conjugation_residual,
       particle_hole_symmetry_residual,
       anomalous_norm,
       nambu_symmetry_residual,
       BosonicMatsubaraAxis,
       AbstractTwoParticleFrequencyObject,
       GeneralizedSusceptibility,
       IrreducibleVertex,
       TwoParticleInverseDiagnostics,
       irreducible_vertex,
       bethe_salpeter,
       physical_susceptibility,
       MomentumResolvedGreenFunction,
       MatrixMomentumResolvedGreenFunction,
       OrbitalLatticeBubble,
       orbital_lattice_bubble,
       momentum_green,
       periodic_momentum_shift_maps,
       lattice_bubble,
       local_bubble,
       AbstractImpurityTwoParticleBackend,
       TauGridFourPoint,
       ImpurityTwoParticleResult,
       impurity_two_particle,
       impurity_two_particle_residual,
       validate_green_function

end # module Greens


# =============================================================================
# 9. Self-consistent embedding and single-site ED-DMFT
# =============================================================================

module Embedding

using LinearAlgebra
using Statistics
using ..PhundamentalCore
using ..Models: AndersonImpurityModel, build_model
using ..Thermodynamics: AbstractThermalState, ExactGibbs, ExactGibbsState, ThermalEstimate, thermal_state
import ..Thermodynamics: expectation
using ..Greens
import ..Greens: momentum_green, _compatible_frequency_objects, _solve_inverse_matrix, _compatible_two_particle, _two_particle_values, _stable_inverse, _bubble_inverse, _common_active_two_particle_indices
using ..Observables: SusceptibilityResult
using ..Solvers: AbstractSolver, ComputationalBasis, ExactDiagonalization, FermionSector, matrix
import ..Solvers: _packed_fermion_basis, solve, solver_name

include("embedding/EmbeddingTypes.jl")
include("embedding/BathDiscretization.jl")
include("embedding/ImpurityProblems.jl")
include("embedding/BathFitting.jl")
include("embedding/ImpuritySolver.jl")
include("embedding/EDImpuritySectors.jl")
include("embedding/EDImpuritySolver.jl")
include("embedding/LatticeGreenFunctions.jl")
include("embedding/SelfConsistency.jl")
include("embedding/Mixing.jl")
include("embedding/DMFT.jl")
include("embedding/diagrammatic/DiagrammaticDMFT.jl")
include("embedding/Validation.jl")

export AbstractEmbeddingProblem,
       AbstractLatticeEmbedding,
       ImpuritySolverResult,
       DiscreteBath,
       hybridization_function,
       ImpurityProblem,
       BathFitOptions,
       BathFitResult,
       fit_bath,
       AbstractImpuritySolver,
       EDImpuritySolver,
       AtomicLattice,
       DiscreteLattice,
       BetheLattice,
       lattice_green,
       weiss_green,
       hybridization_from_weiss,
       AbstractMixing,
       LinearMixing,
       AdaptiveLinearMixing,
       mix_self_energy,
       self_energy_residual,
       DMFTProblem,
       DMFTConvergencePolicy,
       DMFTIteration,
       DMFTSolver,
       DMFTResult,
       AbstractDiagrammaticDMFTMethod,
       LadderDGA,
       MomentumLattice,
       MatrixMomentumLattice,
       periodic_momentum_lattice,
       MomentumResolvedSelfEnergy,
       FullVertex,
       full_vertex,
       schwinger_dyson_self_energy,
       dyson_momentum_green,
       nonlocal_self_energy_residual,
       DiagrammaticDMFTProblem,
       LadderDGAProblem,
       DiagrammaticExecutionPolicy,
       DiagrammaticDMFTSolver,
       DiagrammaticDMFTResult,
       LadderDGAResult,
       susceptibility,
       diagrammatic_convergence,
       validate_diagrammatic_dmft_result,
       SusceptibilityBenchmark,
       SusceptibilityBenchmarkResult,
       SelfEnergyBenchmark,
       SelfEnergyBenchmarkResult,
       compare_susceptibility_benchmark,
       compare_self_energy_benchmark,
       diagrammatic_dmft_benchmark_references,
       noninteracting_self_energy_residual,
       impurity_local_residual,
       atomic_hubbard_green,
       atomic_hubbard_self_energy,
       bath_discretization_residual,
       bath_particle_hole_residual,
       low_frequency_self_energy_residual,
       bethe_self_consistency_residual,
       quasiparticle_weight_estimate,
       validate_dmft_result

end # module Embedding


# =============================================================================
# 10. Top-level API bindings
# =============================================================================
#
# PhundamentalCore and Solvers are fully re-exported at the package root so common modeling and numerical workflows require only `using Phundamental`. The namespace modules remain available for explicit qualification and organization.

using .PhundamentalCore

using .Crystallography: CrystalStructure,
                        SpaceGroupOperation,
                        CrystallographicDataset,
                        CrystallographyOptions,
                        SymmetryOrbit,
                        space_group_operations,
                        expand_symmetry_orbit,
                        site_permutation,
                        validate_crystallographic_group,
                        discover_symmetry,
                        standard_symmetry_action,
                        MagneticSymmetryAction,
                        SpinfulFermionAction,
                        VectorDisplacementAction,
                        validate_internal_symmetry,
                        CrystallographicGenerator,
                        enumerate_operator_candidates,
                        reynolds_project,
                        generate_invariant_basis,
                        crystallographic_specification,
                        generate_crystallographic_hamiltonian_space

using .Transformations: AbstractTransformation,
                        TransformationResult,
                        transform,
                        structurally_related,
                        compose,
                        JordanWigner,
                        HolsteinPrimakoff,
                        DysonMaleev,
                        SchwingerBoson,
                        AbrikosovFermion,
                        FourierTransformation,
                        BogoliubovTransformation,
                        ParticleHoleTransformation,
                        MajoranaTransformation,
                        CoordinateLadder,
                        BosonicDisplacement,
                        LangFirsov,
                        SpinPolaron,
                        AbstractApproximation,
                        ApproximationCertificate,
                        ApproximationResult,
                        ModelSpaceProjection,
                        approximation_name,
                        approximation_certificate,
                        approximation_applicable,
                        apply_approximation,
                        transformation_closure_defect,
                        TransformationEdge,
                        RepresentationGraph,
                        add_representation!,
                        add_transformation!,
                        outgoing_edges,
                        transformation_path,
                        representation_class,
                        transform_along

using .Solvers
import .Solvers: solver_name

using .Observables: AbstractObservable,
                    AbstractProbe,
                    ObservableRequest,
                    DynamicalCorrelationRequest,
                    CompositeDynamicalCorrelationRequest,
                    SpinPhononCorrelationRequest,
                    SpinTensorStructureFactorRequest,
                    StaticStructureFactorRequest,
                    ThermodynamicCurveRequest,
                    MagneticSusceptibilityRequest,
                    MagnetizationCurveRequest,
                    MagneticResponseCurve,
                    CPTSpectralFunctionRequest,
                    HarmonicMultiphononSliceRequest,
                    LocalMatsubaraGreenFunctionRequest,
                    LocalMatsubaraSelfEnergyRequest,
                    MatsubaraSusceptibilityRequest,
                    RequestRequirements,
                    request_requirements,
                    is_dynamical_request,
                    SusceptibilityResult,
                    AbstractBroadening,
                    AbstractResolution,
                    AbstractBosonZeroModePolicy,
                    RejectBosonZeroModes,
                    NumberConservingZeroModeProjection,
                    MomentumField,
                    SpinTensorField,
                    registered_momentum_field,
                    SpectrumConvention,
                    LorentzianBroadening,
                    GaussianBroadening,
                    GaussianResolution,
                    EnergyDependentGaussianResolution,
                    TabulatedGaussianResolution,
                    TabulatedEnergyResolution,
                    GaussianQEResolution,
                    SpatialGaussianQEResolution,
                    SparseQEResolution,
                    InstrumentCoverage,
                    CoveredResolution,
                    MagneticNeutronProbe,
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
                    dynamical_correlation,
                    composite_dynamical_correlation,
                    spin_phonon_dynamical_correlation,
                    dynamic_structure_factor,
                    spin_tensor_structure_factor,
                    static_structure_factor,
                    spectral_density,
                    spectral_function_from_retarded_green,
                    commutator_spectral_density,
                    correlation_spectrum_from_retarded_susceptibility,
                    cluster_perturbation_spectral_function,
                    scattering_intensity,
                    convolve_resolution,
                    register_spin_tensor,
                    StaticTensorStructureFactorResult,
                    ensemble_expectation,
                    ensemble_internal_energy,
                    ensemble_heat_capacity,
                    ensemble_magnetization,
                    spin_tensor_static_structure_factor

using .Models: BravaisLattice,
               lattice_dimension,
               direct_matrix,
               reciprocal_matrix,
               cell_measure,
               fractional_to_cartesian,
               cartesian_to_fractional,
               BasisSite,
               AbstractOccupancy,
               SpeciesOccupancy,
               MixedOccupancy,
               MixedBasisSite,
               MixedCrystalStructure,
               expand_crystal_basis,
               occupancy,
               CrystalSite,
               nsites,
               site_position,
               site_positions,
               site_index,
               reciprocal_vector,
               reciprocal_coordinates,
               Bond,
               neighbor_bonds,
               bonds_in_shell,
               bond_cluster,
               coordination_numbers,
               AtomicSpecies,
               lookup_species,
               AbstractMagneticFormFactor,
               ConstantFormFactor,
               GaussianFormFactor,
               form_factor,
               DebyeWallerTensor,
               debye_waller_factor,
               SiteProperties,
               Material,
               atomic_species,
               basis_properties,
               site_properties,
               site_mass,
               site_spin,
               site_g_tensor,
               site_form_factor,
               site_scattering_length,
               Supercell,
               disordered_supercell,
               MeanSquareDisplacementResult,
               mean_square_displacement,
               phonon_debye_waller_tensors,
               mean_mass,
               mean_scattering_length,
               AbstractHamiltonianSpec,
               build_model,
               HeisenbergModel,
               XXZModel,
               HubbardModel,
               AndersonImpurityModel,
               impurity_bath_sites,
               AbstractHarmonicInteraction,
               HarmonicBondInteraction,
               HarmonicVectorCoordinate,
               HarmonicBendingInteraction,
               HarmonicAngleInteraction,
               harmonic_coordinate_vector,
               bending_force_constant_matrix,
               force_constants,
               ForceConstantBlock,
               RealSpaceForceConstants,
               project_force_constants,
               enforce_acoustic_sum_rule,
               PhononUnitConvention,
               HarmonicPhononModel,
               harmonic_model,
               LocalPhononModel,
               HolsteinModel,
               LongitudinalSpinBosonModel,
               HamiltonianGenerationResult,
               material_crystallographic_structure,
               material_generative_specification,
               acoustic_sum_rule_constraints,
       rotational_sum_rule_constraints,
               generate_hamiltonian_family,
       InteractionGroup,
       OperatorOrbit,
       InteractionSpecification,
       ExchangeOrbit,
       FieldOrbit,
       HoppingOrbit,
       DensityInteractionOrbit,
       PairingOrbit,
       physical_operator_sum,
       PhysicalCoefficientMap,
       InteractionGenerationResult,
       PhysicalCouplings,
       generate_interaction_family,
       physical_parameters,
               InvariantDirectionSignature,
               StructuralFamilySignature,
               StructuralUnit,
               StructuralFamily,
               StructuralSelectionSignature,
               StructuralSelection,
               StructuralMove,
               StructuralNeighborhood,
               StructuralCoordinateMap,
               ModelSpaceHierarchy,
               invariant_direction_signature,
               structural_family,
               structural_selection,
               structural_transition,
               structural_move,
               apply_structural_move,
               structural_neighbors,
               enumerate_structures,
               model_space_hierarchy,
               parent_spaces,
               child_spaces,
               is_subspace,
               structural_coordinate_map,
               embed_coordinates,
               project_coordinates,
               embed_parameters,
               project_parameters,
               parameter_subspace,
       validate_structural_representation,
       structural_id,
       describe,
       reconstruct_selection,
       reconstruct_model,
       generation_context,
       default_observable_context,
       StructuralLocation,
       locate_in_family,
               LocalFrame,
               LocalFrameField,
               local_x,
               local_y,
               local_z,
               local_frame,
               local_frames,
               local_z_axes,
               global_vector,
               local_vector,
               global_tensor,
               local_tensor,
               global_pair_tensor,
               local_pair_tensor,
               frame_moment_vectors,
               validate_local_frames,
               AbstractLongRangeInteraction,
               AbstractInteractionSummation,
               CoulombInteraction,
               DipolarInteraction,
               RealSpaceCutoff,
               EwaldSummation,
               ScalarInteractionResult,
               TensorInteractionResult,
               interaction_matrix,
               interaction_tensor,
               interaction_energy,
               project_to_local_frames,
               ising_coupling_matrix,
               ising_zeeman_field,
               dipolar_hamiltonian,
               validate_long_range_result,
               validate_ewald_alpha_independence,
               ForceConstantValidationResult,
               validate_force_constants

# `CrystalStructure` is already the crystallographic-generator type at the
# package root. Preserve that public binding and expose the lattice/material
# geometry type under an explicit backward-compatible alias.
const LatticeCrystalStructure = Models.CrystalStructure


using .Thermodynamics: AbstractThermalMethod,
                       AbstractThermalState,
                       ExactGibbs,
                       ThermalTypicality,
                       ThermalLanczos,
                       CanonicalTPQ,
                       SectorDecomposition,
                       ThermalWorkspace,
                       SectorThermalWorkspace,
                       thermal_workspace,
                       ThermalEstimate,
                       ThermodynamicPoint,
                       ThermodynamicCurve,
                       ExactGibbsState,
                       SampledThermalState,
                       thermal_state,
                       thermal_curve,
                       density_matrix,
                       partition_function,
                       log_partition_function,
                       free_energy,
                       internal_energy,
                       entropy,
                       heat_capacity,
                       expectation,
                       thermal_variance,
                       ThermalCluster,
                       ClusterThermodynamicsResult,
                       connected_clusters,
                       cluster_thermodynamics,
                       AbstractClusterExpansion,
                       FiniteReferenceExpansion,
                       TopologicalClusterClass,
                       TopologicalExpansion,
                       NLCEProblem,
                       NLCESolver,
                       NLCEResult,
                       linked_cluster_values,
                       topological_nlce,
                       expectation_nlce_problem,
                       nlce_convergence,
                       validate_nlce_result,
                       NLCEClusterGraph,
                       topological_expansion,
                       site_nlce_expansion,
                       TopologicalNLCEProblem,
                       validate_embedding_stability,
                       AbstractNLCEResummation,
                       BareNLCE,
                       WynnEpsilon,
                       NLCEResummationResult,
                       resum_nlce,
                       nlce_sequence_residual,
                       LiteratureBenchmark,
                       LiteratureBenchmarkResult,
                       compare_literature_benchmark,
                       rigol_2007_square_site_counts,
                       summarize_topological_expansion,
                       validate_rigol_2007_square_site_counts,
                       bulk_estimate,
                       validate_thermal_state,
                       validate_sampling_against_exact,
                       validate_exact_spectrum_thermodynamics,
                       validate_sector_thermodynamic_recombination,
                       validate_linked_cluster_closure

using .Greens
using .Embedding

include("forward/ForwardMethodContexts.jl")
include("embedding/GeometricEmbedding.jl")
include("greens/AnalyticContinuation.jl")
include("forward/ResultCapabilities.jl")
include("forward/ForwardCapabilityContracts.jl")
include("forward/ForwardCapabilities.jl")
include("forward/ForwardProblems.jl")
include("forward/StructuralClusterForward.jl")
include("forward/ExtendedForward.jl")
include("forward/ForwardPlanning.jl")

# =============================================================================
# 11. High-level forward-calculation orchestration
# =============================================================================

"""
    ForwardResult

Container for the successive stages of a complete forward calculation.

Fields
------
- `source_model`          : model constructed in its source representation
- `model`                 : model actually supplied to the solver
- `transformation_result` : `nothing` or the result of `transform`
- `solution`              : solver output
- `observable_result`     : intrinsic correlation/spectral result, or `nothing`
- `scattering_result`     : probe-contracted result, or `nothing`
- `result`                : final result after optional resolution convolution

The distinction between `observable_result`, `scattering_result`, and `result`
preserves the separation

    intrinsic spectrum
        -> probe cross section
        -> instrument response.
"""
struct ForwardResult{SM,M,TR,S,O,P,R}
    source_model::SM
    model::M
    transformation_result::TR
    solution::S
    observable_result::O
    scattering_result::P
    result::R
end

include("experimental/ExperimentalData.jl")

"""
    forward(model; transformation=nothing, solver=ExactDiagonalization(),
            solver_args=(), solver_kwargs=(;), method_context=nothing,
            observable=nothing, probe=nothing, resolution=nothing)

Execute the generic representation-aware forward pipeline

    ManyBodyModel
        -> optional representation transformation
        -> numerical solver
        -> optional observable calculation
        -> optional probe contraction
        -> optional instrument-resolution convolution.

`observable` may be a custom callable or a typed `AbstractObservable` request. Typed requests participate in request-aware computational compatibility and may use an explicit `method_context` when an already selected method requires numerical information that is not reconstructible from the finite source model. The context configures that method only; it does not rank solvers, determine physical/model-space admissibility, or authorize an approximation.

Example:

    obs = (model, solution) -> spin_tensor_structure_factor(
        model,
        solution,
        spinfield,
        Qgrid,
        ωgrid;
        temperature = 5.0,
        broadening = LorentzianBroadening(0.05),
    )

    result = forward(
        model;
        transformation = JordanWigner(1:N),
        solver = ExactDiagonalization(),
        observable = obs,
        probe = MagneticNeutronProbe(),
        resolution = GaussianResolution(0.08),
    )

The intrinsic numerical broadening used by the observable remains distinct
from `resolution`, which is applied only after the probe-specific intensity
has been constructed.
"""
function forward(
    model::ManyBodyModel;
    transformation=nothing,
    solver::AbstractSolver=ExactDiagonalization(),
    solver_args::Tuple=(),
    solver_kwargs::NamedTuple=(;),
    method_context=nothing,
    observable=nothing,
    probe=nothing,
    resolution=nothing,
)
    problem = ForwardProblem(model; transformation=transformation, solver=solver, solver_args=solver_args,
                             solver_kwargs=solver_kwargs, method_context=method_context, observable=observable, probe=probe, resolution=resolution)
    return forward(problem)
end

function forward(problem::ForwardProblem)
    !isnothing(problem.method_context) && throw(ArgumentError("method_context requires a typed AbstractObservable request; construct a solver-specific problem and call solve directly for expert-level method-only execution"))
    model = problem.model
    if isnothing(problem.transformation)
        transformation_result = nothing
        working_model = model
    else
        transformation_result = transform(problem.transformation, model)
        working_model = transformation_result.model
    end

    solution = solve(problem.solver, working_model, problem.solver_args...; problem.solver_kwargs...)
    observable_result = isnothing(problem.observable) ? nothing : evaluate_observable_request(problem.observable, working_model, solution)
    scattering_result = isnothing(problem.probe) ? observable_result : scattering_intensity(problem.probe, observable_result)
    final_result = isnothing(problem.resolution) ? scattering_result : convolve_resolution(problem.resolution, scattering_result)

    return ForwardResult(model, working_model, transformation_result, solution, observable_result, scattering_result, final_result)
end

"""
    forward(material, cluster, spec; model_kwargs=NamedTuple(), kwargs...)

Construct a `ManyBodyModel` from a material, finite supercell, and Hamiltonian
specification, then execute `forward(model; kwargs...)`.

Keyword arguments intended for `build_model` must be supplied through
`model_kwargs`; all remaining keyword arguments are passed to the model-level
`forward` method.

Example:

    result = forward(
        material,
        cluster,
        XXZModel(...);
        model_kwargs = (; bonds=bonds),
        transformation = JordanWigner(1:N),
        solver = ExactDiagonalization(),
        observable = obs,
        probe = MagneticNeutronProbe(),
        resolution = GaussianResolution(0.08),
    )
"""
function forward(
    material::Material,
    cluster::Supercell,
    spec::AbstractHamiltonianSpec;
    model_kwargs::NamedTuple=(;),
    kwargs...,
)
    model = build_model(material, cluster, spec; model_kwargs...)
    return forward(model; kwargs...)
end


include("forward/StructuralForward.jl")

# =============================================================================
# 12. Public exports
# =============================================================================

# Major namespaces
export ThermodynamicForwardSolver,
       RationalContinuation,
       ContinuationResult,
       continue_matsubara,
       BravaisHopping,
       OrbitalBravaisHopping,
       bravais_momentum_lattice,
       matrix_bravais_momentum_lattice,
       StructuralNLCEForwardContext,
       regenerate_induced_spin_cluster,
       PhundamentalCore,
       Foundation,
       Algebra,
       Crystallography,
       Transformations,
       Solvers,
       Observables,
       Models,
       Thermodynamics,
       Greens,
       Embedding

# Fundamental state spaces, algebras, operators, representations, model spaces, and models
export AbstractStateSpace,
       AbstractHilbertSpace,
       AbstractFockSpace,
       SpinHilbertSpace,
       FermionFockSpace,
       BosonFockSpace,
       CoordinateHilbertSpace,
       CompositeStateSpace,
       BiorthogonalSpace,
       PhysicalSubspace,
       NumericalTruncation,
       spacedimension,
       physicaldimension,
       AbstractOperatorAlgebra,
       SpinAlgebra,
       BosonAlgebra,
       FermionAlgebra,
       MajoranaAlgebra,
       CoordinateMomentumAlgebra,
       CompositeAlgebra,
       algebra_kind,
       AbstractOperatorExpr,
       AbstractPrimitiveOperator,
       IdentityOperator,
       OperatorSum,
       OperatorProduct,
       ScaledOperator,
       SpinX,
       SpinY,
       SpinZ,
       SpinPlus,
       SpinMinus,
       BosonAnnihilate,
       BosonCreate,
       FermionAnnihilate,
       FermionCreate,
       MajoranaOperator,
       PositionOperator,
       MomentumOperator,
       NumberOperator,
       Sx,
       Sy,
       Sz,
       Sp,
       Sm,
       b,
       c,
       γ,
       xop,
       pop,
       nb,
       nf,
       FactorOperator,
       atfactor,
       isprimitive,
       children,
       operator_degree,
       support_size,
       body_order,
       AbstractBasis,
       SpinProductBasis,
       FermionOccupationBasis,
       BosonOccupationBasis,
       CoordinateBasis,
       CompositeBasis,
       BiorthogonalBasis,
       AbstractWaveVector,
       CartesianWaveVector,
       ReciprocalWaveVector,
       wavevector_coordinates,
       ReciprocalDecomposition,
       decompose_reciprocal_vector,
       GaugeStructure,
       Representation,
       isconstrained,
       istruncated,
       ambientdimension,
       representationdimension,
       physicalspace,
       FactorConstraint,
       FactorTruncation,
       factorconstraint,
       factortruncation,
       AbstractModelStructure,
       StructuralSupport,
       StructuralTerm,
       InteractionEdge,
       InteractionGraph,
       StructuralSignature,
       ModelStructure,
       representation_signature,
       structure,
       canonicalize_structure,
       structural_signature,
       interaction_graph,
       structurally_equivalent,
       structural_span_membership,
       structural_span_coordinates,
       AbstractModelSpace,
       AbstractAdmissibilityCondition,
       AdmissibilityCheck,
       AdmissibilityReport,
       PredicateAdmissibility,
       admissibility_check,
       admissibility_report,
       admissibility_reasons,
       GenerativeSpecification,
       HamiltonianBasis,
       HamiltonianSpace,
       admissible,
       model_space_dimension,
       generated_basis,
       generate_hamiltonian_space,
       hamiltonian_from_coefficients,
       ParameterSpec,
       AbstractParameterConstraint,
       FixedParameterConstraint,
       EqualParameterConstraint,
       LinearParameterConstraint,
       ParameterConstraintCheck,
       ParameterConstraintReport,
       ParameterBinding,
       ParameterSpace,
       ParameterPoint,
       parameter_space,
       parameter_point,
       parameter_names,
       parameter_bounds,
       parameter_units,
       parameter_vector,
       free_parameter_indices,
       parameter_constraints,
       parameter_constraint_check,
       parameter_constraint_report,
       parameter_constraint_reasons,
       valid_parameter_point,
       parameter_constraint_matrix,
       parameter_dof,
       parameter_bindings,
       parameter_binding,
       rebind_parameters,
       with_parameters,
       ComponentIndexLayout,
       component_index,
       component_site,
       component_coordinates,
       GeometryContext,
       PhysicalModelContext,
       ObservableContext,
       observable_context,
       model_context,
       geometry_context,
       ManyBodyModel,
       with_provenance,
       remap_model,
       with_truncation,
       instantiate_model

# Transformations
export AbstractTransformation,
       TransformationResult,
       transform,
       structurally_related,
       compose,
       JordanWigner,
       HolsteinPrimakoff,
       DysonMaleev,
       SchwingerBoson,
       AbrikosovFermion,
       FourierTransformation,
       BogoliubovTransformation,
       ParticleHoleTransformation,
       MajoranaTransformation,
       CoordinateLadder,
       BosonicDisplacement,
       LangFirsov,
       SpinPolaron,
       AbstractApproximation,
       ApproximationCertificate,
       ApproximationResult,
       ModelSpaceProjection,
       approximation_name,
       approximation_certificate,
       approximation_applicable,
       apply_approximation,
       transformation_closure_defect,
       TransformationEdge,
       RepresentationGraph,
       add_representation!,
       add_transformation!,
       outgoing_edges,
       transformation_path,
       representation_class,
       transform_along

# Solvers and numerical realization
export AbstractSolver,
       solve,
       energy_spectrum,
       solver_name,
       SolverResult,
       EnergySpectrumResult,
       TimeEvolutionResult,
       nlevels,
       states,
       eigenstate,
       groundenergy,
       groundstate,
       isbiorthogonal,
       AbstractComputationalState,
       SpinBasisState,
       FermionBasisState,
       BosonBasisState,
       CompositeBasisState,
       ComputationalBasis,
       computational_basis,
       basis_dimension,
       basis_state,
       NumericalCapabilities,
       TransformationCapabilities,
       numerical_capabilities,
       transformation_capabilities,
       MatrixRealizationResult,
       realize,
       matrix,
       hamiltonian_matrix,
       SymbolicLinearOperator,
       ExactDiagonalization,
       LanczosSolver,
       TimeEvolutionSolver,
       OptimizedLanczos,
       AutoSolver,
       AbstractSector,
       SpinSector,
       FermionSector,
       QuantumNumber,
       QuantumNumberSector,
       CompositeSector,
       quantum_numbers,
       sector_value,
       has_quantum_number,
       ProjectedFermionResolvent,
       FermionGreenWorkspace,
       fermion_transition_vector,
       fermion_green_workspace,
       cluster_green_matrix,
       cpt_intercluster_hopping,
       cluster_perturbation_green_matrix,
       cpt_periodize,
       cluster_perturbation_green,
       AbstractOptimizedBackend,
       PackedSpinBackend,
       PackedFermionBackend,
       PackedBosonBackend,
       QuadraticFermionSolver,
       QuadraticBosonSolver,
       BosonSubspaceProjection,
       orthogonal_boson_subspace,
       HarmonicPhononSolver,
       LinearSpinWaveSolver,
       AbstractModeResult,
       QuadraticModeResult,
       PhononModeResult,
       PhononDispersionResult,
       SpinWaveResult,
       optimized_backend,
       supports_optimized_backend,
       optimized_matrix_free_operator,
       spin_sector_profile,
       spin_sector_plan,
       mode_energies,
       mode_vectors,
       dynamical_matrix,
       dynamical_matrix!,
       dynamical_gradient,
       dynamical_gradient!,
       dynamical_hessian,
       dynamical_hessian!,
       acoustic_sum_rule_residual,
       group_velocities,
       directional_group_velocities,
       phonon_dispersion,
       phonon_zero_point_energy,
       spinwave_zero_point_correction,
       spinwave_ground_energy,
       ContinuedFractionResult,
       continued_fraction,
       continued_fraction_value,
       continued_fraction_spectrum,
       exp_action,
       validate_optimized_against_generic,
       validate_packed_spin_implicit_basis,
       validate_packed_spin_rank_lookup,
       validate_spin_sector_decomposition,
       validate_phonon_modes,
       validate_phonon_dispersion,
       validate_mode_result,
       validate_spinwave_result,
       AbstractClassicalProblem,
       AbstractClassicalSampler,
       AbstractConfigurationEnsemble,
       IsingProblem,
       MetropolisSampler,
       ConfigurationEnsemble,
       sample,
       nsamples,
       configuration,
       ising_energy,
       ising_flip_delta,
       autocorrelation_time,
       effective_sample_size,
       correlated_stderr,
       validate_metropolis_detailed_balance,
       validate_sampling_against_exact_classical,
       evolve,
       validate_matrix_realization

# Observables / scattering
export AbstractProbe,
       AbstractBroadening,
       AbstractResolution,
       AbstractBosonZeroModePolicy,
       RejectBosonZeroModes,
       NumberConservingZeroModeProjection,
       ObservableRequest,
       DynamicalCorrelationRequest,
       CompositeDynamicalCorrelationRequest,
       SpinPhononCorrelationRequest,
       SpinTensorStructureFactorRequest,
       StaticStructureFactorRequest,
       ThermodynamicCurveRequest,
       MagneticSusceptibilityRequest,
       MagnetizationCurveRequest,
       MagneticResponseCurve,
       CPTSpectralFunctionRequest,
       HarmonicMultiphononSliceRequest,
       LocalMatsubaraGreenFunctionRequest,
       LocalMatsubaraSelfEnergyRequest,
       MatsubaraSusceptibilityRequest,
       SusceptibilityResult,
       MomentumField,
       SpinTensorField,
       SpectrumConvention,
       LorentzianBroadening,
       GaussianBroadening,
       GaussianResolution,
       EnergyDependentGaussianResolution,
       TabulatedGaussianResolution,
       TabulatedEnergyResolution,
       GaussianQEResolution,
       SpatialGaussianQEResolution,
       SparseQEResolution,
       InstrumentCoverage,
       CoveredResolution,
       MagneticNeutronProbe,
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
       dynamical_correlation,
       composite_dynamical_correlation,
       spin_phonon_dynamical_correlation,
       dynamic_structure_factor,
       spin_tensor_structure_factor,
       static_structure_factor,
       spectral_density,
       spectral_function_from_retarded_green,
       commutator_spectral_density,
       correlation_spectrum_from_retarded_susceptibility,
       cluster_perturbation_spectral_function,
       scattering_intensity,
       convolve_resolution,
       registered_momentum_field,
       register_spin_tensor,
       StaticTensorStructureFactorResult,
       ensemble_expectation,
       ensemble_internal_energy,
       ensemble_heat_capacity,
       ensemble_magnetization,
       spin_tensor_static_structure_factor

# Materials / Hamiltonians
export BravaisLattice,
       lattice_dimension,
       direct_matrix,
       reciprocal_matrix,
       cell_measure,
       fractional_to_cartesian,
       cartesian_to_fractional,
       BasisSite,
       AbstractOccupancy,
       SpeciesOccupancy,
       MixedOccupancy,
       MixedBasisSite,
       LatticeCrystalStructure,
       MixedCrystalStructure,
       expand_crystal_basis,
       occupancy,
       CrystalSite,
       nsites,
       site_position,
       site_positions,
       site_index,
       reciprocal_vector,
       reciprocal_coordinates,
       Bond,
       neighbor_bonds,
       bonds_in_shell,
       bond_cluster,
       coordination_numbers,
       AtomicSpecies,
       SiteProperties,
       Material,
       Supercell,
       disordered_supercell,
       lookup_species,
       AbstractMagneticFormFactor,
       ConstantFormFactor,
       GaussianFormFactor,
       form_factor,
       DebyeWallerTensor,
       debye_waller_factor,
       MeanSquareDisplacementResult,
       mean_square_displacement,
       phonon_debye_waller_tensors,
       site_g_tensor,
       site_form_factor,
       mean_mass,
       mean_scattering_length,
       AbstractHamiltonianSpec,
       build_model,
       HeisenbergModel,
       XXZModel,
       HubbardModel,
       AndersonImpurityModel,
       impurity_bath_sites,
       AbstractHarmonicInteraction,
       HarmonicBondInteraction,
       HarmonicVectorCoordinate,
       HarmonicBendingInteraction,
       HarmonicAngleInteraction,
       harmonic_coordinate_vector,
       bending_force_constant_matrix,
       force_constants,
       ForceConstantBlock,
       RealSpaceForceConstants,
       project_force_constants,
       enforce_acoustic_sum_rule,
       PhononUnitConvention,
       HarmonicPhononModel,
       harmonic_model,
       LocalPhononModel,
       HolsteinModel,
       LongitudinalSpinBosonModel,
       HamiltonianGenerationResult,
       material_crystallographic_structure,
       material_generative_specification,
       acoustic_sum_rule_constraints,
       rotational_sum_rule_constraints,
       generate_hamiltonian_family,
       InteractionGroup,
       OperatorOrbit,
       InteractionSpecification,
       ExchangeOrbit,
       FieldOrbit,
       HoppingOrbit,
       DensityInteractionOrbit,
       PairingOrbit,
       physical_operator_sum,
       PhysicalCoefficientMap,
       InteractionGenerationResult,
       PhysicalCouplings,
       generate_interaction_family,
       physical_parameters,
       InvariantDirectionSignature,
       StructuralFamilySignature,
       StructuralUnit,
       StructuralFamily,
       StructuralSelectionSignature,
       StructuralSelection,
       StructuralMove,
       StructuralNeighborhood,
       StructuralCoordinateMap,
       ModelSpaceHierarchy,
       invariant_direction_signature,
       structural_family,
       structural_selection,
       structural_transition,
       structural_move,
       apply_structural_move,
       structural_neighbors,
       enumerate_structures,
       model_space_hierarchy,
       parent_spaces,
       child_spaces,
       is_subspace,
       structural_coordinate_map,
       embed_coordinates,
       project_coordinates,
       embed_parameters,
       project_parameters,
       parameter_subspace,
       validate_structural_representation,
       structural_id,
       describe,
       reconstruct_selection,
       reconstruct_model,
       generation_context,
       default_observable_context,
       StructuralLocation,
       locate_in_family,
       LocalFrame,
       LocalFrameField,
       local_x,
       local_y,
       local_z,
       local_frame,
       local_frames,
       local_z_axes,
       global_vector,
       local_vector,
       global_tensor,
       local_tensor,
       global_pair_tensor,
       local_pair_tensor,
       frame_moment_vectors,
       validate_local_frames,
       AbstractLongRangeInteraction,
       AbstractInteractionSummation,
       CoulombInteraction,
       DipolarInteraction,
       RealSpaceCutoff,
       EwaldSummation,
       ScalarInteractionResult,
       TensorInteractionResult,
       interaction_matrix,
       interaction_tensor,
       interaction_energy,
       project_to_local_frames,
       ising_coupling_matrix,
       ising_zeeman_field,
       dipolar_hamiltonian,
       validate_long_range_result,
       validate_ewald_alpha_independence,
       ForceConstantValidationResult,
       validate_force_constants


# Thermodynamics / thermal sampling / linked clusters
export AbstractThermalMethod,
       AbstractThermalState,
       ExactGibbs,
       ThermalTypicality,
       ThermalLanczos,
       CanonicalTPQ,
       SectorDecomposition,
       ThermalWorkspace,
       SectorThermalWorkspace,
       thermal_workspace,
       ThermalEstimate,
       ThermodynamicPoint,
       ThermodynamicCurve,
       ExactGibbsState,
       SampledThermalState,
       thermal_state,
       thermal_curve,
       density_matrix,
       partition_function,
       log_partition_function,
       free_energy,
       internal_energy,
       entropy,
       heat_capacity,
       expectation,
       thermal_variance,
       ThermalCluster,
       ClusterThermodynamicsResult,
       connected_clusters,
       cluster_thermodynamics,
       AbstractClusterExpansion,
       FiniteReferenceExpansion,
       TopologicalClusterClass,
       TopologicalExpansion,
       NLCEProblem,
       NLCESolver,
       NLCEResult,
       linked_cluster_values,
       topological_nlce,
       expectation_nlce_problem,
       nlce_convergence,
       validate_nlce_result,
       NLCEClusterGraph,
       topological_expansion,
       site_nlce_expansion,
       TopologicalNLCEProblem,
       validate_embedding_stability,
       AbstractNLCEResummation,
       BareNLCE,
       WynnEpsilon,
       NLCEResummationResult,
       resum_nlce,
       nlce_sequence_residual,
       LiteratureBenchmark,
       LiteratureBenchmarkResult,
       compare_literature_benchmark,
       rigol_2007_square_site_counts,
       summarize_topological_expansion,
       validate_rigol_2007_square_site_counts,
       bulk_estimate,
       validate_thermal_state,
       validate_sampling_against_exact,
       validate_exact_spectrum_thermodynamics,
       validate_sector_thermodynamic_recombination,
       validate_linked_cluster_closure

# Green functions, Nambu structure, Dyson relations, and self-consistent embedding
export AbstractFrequencyAxis,
       FermionicMatsubaraAxis,
       matsubara_frequency,
       matsubara_index,
       inverse_temperature,
       axis_temperature,
       AbstractFrequencyMatrix,
       GreenFunction,
       NonInteractingGreenFunction,
       SelfEnergy,
       HybridizationFunction,
       frequency_axis,
       frequency_values,
       component_labels,
       matrix_dimension,
       scalar_frequency_values,
       noninteracting_green,
       zero_self_energy,
       lehmann_green,
       matsubara_green,
       NambuIndex,
       nambu_green,
       nambu_index,
       normal_block,
       anomalous_block,
       hole_block,
       self_energy,
       dyson_green,
       high_frequency_moment,
       high_frequency_moments,
       high_frequency_moment_residual,
       zeroth_moment_residual,
       causality_residual,
       matsubara_conjugation_residual,
       particle_hole_symmetry_residual,
       anomalous_norm,
       nambu_symmetry_residual,
       BosonicMatsubaraAxis,
       AbstractTwoParticleFrequencyObject,
       GeneralizedSusceptibility,
       IrreducibleVertex,
       TwoParticleInverseDiagnostics,
       irreducible_vertex,
       bethe_salpeter,
       physical_susceptibility,
       MomentumResolvedGreenFunction,
       MatrixMomentumResolvedGreenFunction,
       OrbitalLatticeBubble,
       orbital_lattice_bubble,
       momentum_green,
       periodic_momentum_shift_maps,
       lattice_bubble,
       local_bubble,
       AbstractImpurityTwoParticleBackend,
       TauGridFourPoint,
       ImpurityTwoParticleResult,
       impurity_two_particle,
       impurity_two_particle_residual,
       validate_green_function,
       AbstractEmbeddingProblem,
       AbstractLatticeEmbedding,
       ImpuritySolverResult,
       DiscreteBath,
       hybridization_function,
       ImpurityProblem,
       BathFitOptions,
       BathFitResult,
       fit_bath,
       AbstractImpuritySolver,
       EDImpuritySolver,
       AtomicLattice,
       DiscreteLattice,
       BetheLattice,
       lattice_green,
       weiss_green,
       hybridization_from_weiss,
       AbstractMixing,
       LinearMixing,
       AdaptiveLinearMixing,
       mix_self_energy,
       self_energy_residual,
       DMFTProblem,
       DMFTConvergencePolicy,
       DMFTIteration,
       DMFTSolver,
       DMFTResult,
       AbstractDiagrammaticDMFTMethod,
       LadderDGA,
       MomentumLattice,
       MatrixMomentumLattice,
       periodic_momentum_lattice,
       MomentumResolvedSelfEnergy,
       FullVertex,
       full_vertex,
       schwinger_dyson_self_energy,
       dyson_momentum_green,
       nonlocal_self_energy_residual,
       DiagrammaticDMFTProblem,
       LadderDGAProblem,
       DiagrammaticExecutionPolicy,
       DiagrammaticDMFTSolver,
       DiagrammaticDMFTResult,
       LadderDGAResult,
       susceptibility,
       diagrammatic_convergence,
       validate_diagrammatic_dmft_result,
       SusceptibilityBenchmark,
       SusceptibilityBenchmarkResult,
       SelfEnergyBenchmark,
       SelfEnergyBenchmarkResult,
       compare_susceptibility_benchmark,
       compare_self_energy_benchmark,
       diagrammatic_dmft_benchmark_references,
       noninteracting_self_energy_residual,
       impurity_local_residual,
       atomic_hubbard_green,
       atomic_hubbard_self_energy,
       bath_discretization_residual,
       bath_particle_hole_residual,
       low_frequency_self_energy_residual,
       bethe_self_consistency_residual,
       quasiparticle_weight_estimate,
       validate_dmft_result

# Crystallographic model-space generation
export CrystalStructure,
       SpaceGroupOperation,
       CrystallographicDataset,
       CrystallographyOptions,
       SymmetryOrbit,
       space_group_operations,
       expand_symmetry_orbit,
       site_permutation,
       validate_crystallographic_group,
       discover_symmetry,
       standard_symmetry_action,
       MagneticSymmetryAction,
       SpinfulFermionAction,
       VectorDisplacementAction,
       validate_internal_symmetry,
       CrystallographicGenerator,
       enumerate_operator_candidates,
       reynolds_project,
       generate_invariant_basis,
       crystallographic_specification,
       generate_crystallographic_hamiltonian_space

# Generalized forward/calculation and result-capability API
export ForwardProblem,
       CalculationPlan,
       ForwardResult,
       PreparedStructuralForward,
       SpinScatteringProblem,
       spin_observables,
       spin_scattering_problem,
       validate_forward_equivalence,
       prepare_forward,
       forward_parameters,
       forward_batch,
       AbstractForwardMethodContext,
       DMFTForwardContext,
       LadderDGAForwardContext,
       NLCEForwardContext,
       forward,
       evaluate_observable_request,
       RequestRequirements,
       request_requirements,
       is_dynamical_request,
       SolverCompatibility,
       CompatibilityIssue,
       CapabilityDecision,
       solver_model_capability,
       request_model_capability,
       solver_request_capability,
       solver_context_capability,
       solver_output_family,
       solver_approximation,
       compatibility_issues,
       solver_compatibility,
       solver_capabilities,
       compatible_solvers,
       ForwardExecutionPolicy,
       ForwardRoute,
       ForwardRouteAssessment,
       ForwardPlanningReport,
       ForwardPlanningError,
       plan_forward,
       planning_report,
       result_capabilities,
       supports,
       result_metadata,
       result_model,
       result_energies,
       result_frequencies,
       result_modes,
       result_basis,
       result_times,
       result_intensity,
       result_states,
       result_axes,
       result_values,
       result_uncertainty,
       result_provenance,
       EXPERIMENTAL_DATA_SCHEMA_VERSION,
       ExperimentalGrid,
       PortablePrediction,
       ExperimentalDataset,
       MeasurementComparison,
       portable_prediction,
       experimental_dataset,
       export_result,
       import_result,
       compare_measurement,
       response_capabilities,
       response_kind,
       frequency_domain,
       response_statistics,
       momentum_scope,
       operator_channels,
       response_order,
       response_convention,
       is_dynamical_response,
       requires_analytic_continuation,
       correlation_scope

end # module Phundamental
