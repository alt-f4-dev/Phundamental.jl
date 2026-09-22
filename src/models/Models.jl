module Models

using LinearAlgebra
using ..Core

# Geometry is deliberately independent of the Hamiltonian constructors.
include("Geometry/Lattices.jl")
include("Geometry/CrystalStructures.jl")
include("Geometry/ReciprocalSpace.jl")
include("Geometry/NeighborLists.jl")

# Harmonic and long-range interactions.
include("Interactions/Harmonic.jl")

# Material metadata and probe-relevant properties.
include("Materials/AtomicData.jl")
include("Materials/Species.jl")
include("Materials/ScatteringProperties.jl")
include("Materials/Material.jl")

# Physical Hamiltonian specifications and constructors.
include("Hamiltonians/HamiltonianTypes.jl")
include("Hamiltonians/SpinModels.jl")
include("Hamiltonians/FermionModels.jl")
include("Hamiltonians/ImpurityModels.jl")
include("Hamiltonians/PhononModels.jl")
include("Hamiltonians/HybridModels.jl")
include("Validation.jl")

# Geometry
export BravaisLattice,
       lattice_dimension,
       direct_matrix,
       reciprocal_matrix,
       cell_measure,
       fractional_to_cartesian,
       cartesian_to_fractional,
       BasisSite,
       CrystalStructure,
       CrystalSite,
       Supercell,
       nsites,
       site_position,
       site_positions,
       site_index,
       reciprocal_vector,
       reciprocal_coordinates,
       Bond,
       neighbor_bonds,
       bonds_in_shell,
       coordination_numbers

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
       MeanSquareDisplacementResult,
       mean_square_displacement,
       phonon_debye_waller_tensors

# Hamiltonian specifications and model construction
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
       AbstractHarmonicInteraction,
       HarmonicBondInteraction,
       HarmonicAngleInteraction,
       force_constants,
       ForceConstantBlock,
       RealSpaceForceConstants,
       project_force_constants,
       enforce_acoustic_sum_rule,
       PhononUnitConvention,
       HarmonicPhononModel,
       LocalPhononModel,
       HolsteinModel,
       LongitudinalSpinBosonModel,
       validate_lattice,
       validate_bonds,
       validate_material_model,
       ForceConstantValidationResult,
       validate_force_constants

end # module Models
