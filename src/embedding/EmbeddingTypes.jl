# Shared embedding abstractions and result containers.

abstract type AbstractEmbeddingProblem end
abstract type AbstractLatticeEmbedding end

"""Result of one finite-temperature impurity solve."""
struct ImpuritySolverResult{M,G,G0,S,B,T}
    model::M
    green_function::G
    noninteracting_green_function::G0
    self_energy::S
    density::Float64
    double_occupancy::Float64
    bath::B
    thermal_state::T
    metadata::Dict{Symbol,Any}
end
