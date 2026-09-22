# Typed impurity problems independent of a particular impurity solver.

"""
    ImpurityProblem(interaction, axis; chemical_potential=0, impurity_energy=0, target_hybridization=nothing, bath=nothing)

Single-orbital spin-degenerate impurity problem. Exactly one of `target_hybridization` or `bath` must be supplied: the former requests bath fitting by a finite-bath solver, while the latter fixes the bath explicitly.
"""
struct ImpurityProblem{A<:FermionicMatsubaraAxis,H,B} <: AbstractEmbeddingProblem
    interaction::Float64
    chemical_potential::Float64
    impurity_energy::Float64
    axis::A
    target_hybridization::H
    bath::B
end

function ImpurityProblem(interaction::Real, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0, impurity_energy::Real=0.0,
                         target_hybridization=nothing, bath=nothing)
    xor(isnothing(target_hybridization), isnothing(bath)) || throw(ArgumentError("supply exactly one of target_hybridization or bath"))
    if !isnothing(target_hybridization)
        target_hybridization isa HybridizationFunction || throw(ArgumentError("target_hybridization must be a HybridizationFunction"))
        _compatible_frequency_objects(target_hybridization, HybridizationFunction(axis, zeros(ComplexF64, length(axis)); labels=[:impurity]))
        matrix_dimension(target_hybridization) == 1 || throw(DimensionMismatch("the first ED-DMFT milestone supports a scalar impurity hybridization"))
    end
    !isnothing(bath) && !(bath isa DiscreteBath) && throw(ArgumentError("bath must be a DiscreteBath"))
    return ImpurityProblem(Float64(interaction), Float64(chemical_potential), Float64(impurity_energy), axis, target_hybridization, bath)
end
