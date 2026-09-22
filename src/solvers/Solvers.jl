module Solvers

using LinearAlgebra
using SparseArrays
using Random
using ..Core
using ..Algebra
using ..Transformations

include("SolverTypes.jl")
include("SolverResult.jl")
include("BasisConstruction.jl")
include("NumericalCapabilities.jl")
include("MatrixRealization.jl")
include("ExactDiagonalization.jl")
include("Lanczos.jl")
include("TimeEvolution.jl")
include("Validation.jl")
include("optimized/Optimized.jl")

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
       SpinSector,
       FermionSector,
       ProjectedFermionResolvent,
       FermionGreenWorkspace,
       fermion_transition_vector,
       fermion_green_workspace,
       cluster_green_matrix,
       cpt_intercluster_hopping,
       cluster_perturbation_green_matrix,
       cpt_periodize,
       cluster_perturbation_green,
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
       validate_phonon_modes,
       validate_phonon_dispersion,
       validate_mode_result,
       validate_spinwave_result,
       optimized_backend,
       supports_optimized_backend,
       optimized_matrix_free_operator,
       spin_sector_profile,
       spin_sector_plan,
       validate_optimized_against_generic,
       validate_packed_spin_implicit_basis,
       validate_packed_spin_rank_lookup,
       validate_spin_sector_decomposition,
       evolve,
       validate_matrix_realization

end # module Solvers
