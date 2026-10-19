module Crystallography

using LinearAlgebra
using ..PhundamentalCore
using ..Algebra

include("SymmetryTypes.jl")
include("SpglibBackend.jl")
include("OrbitExpansion.jl")
include("SymmetryActions.jl")
include("InternalSymmetryActions.jl")
include("InvariantGeneration.jl")

export CrystalStructure,
       SpaceGroupOperation,
       CrystallographicDataset,
       CrystallographyOptions,
       SymmetryOrbit,
       space_group_operations,
       expand_symmetry_orbit,
       wrap_fractional,
       fractional_displacement,
       cartesian_rotation,
       axial_rotation,
       site_permutation,
       validate_crystallographic_group,
       spglib_cell,
       discover_symmetry,
       transform_primitive,
       transform_operator,
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

end
