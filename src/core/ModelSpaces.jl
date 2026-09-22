# General Hamiltonian model-space objects implementing the manuscript-level
# generative specification G -> H(G). The actual invariant-basis generator is
# intentionally pluggable: this layer defines the mathematical API without
# claiming universal crystallographic completeness for every operator algebra.

abstract type AbstractAdmissibilityCondition end

"""
    PredicateAdmissibility(name, predicate; description="")

Named admissibility condition evaluated on a candidate Hamiltonian operator.
The predicate must return `Bool` when called on the candidate.
"""
struct PredicateAdmissibility{F} <: AbstractAdmissibilityCondition
    name::Symbol
    predicate::F
    description::String
end

PredicateAdmissibility(name::Symbol, predicate; description::AbstractString="") =
    PredicateAdmissibility(name, predicate, String(description))

admissible(condition::PredicateAdmissibility, candidate) = Bool(condition.predicate(candidate))
admissible(condition::AbstractAdmissibilityCondition, candidate) = throw(ArgumentError("no admissibility evaluator is defined for $(typeof(condition))"))

"""
    GenerativeSpecification

General system specification

    G = (G_space, w_alpha, H_alpha, A, U, C_adm)

used to define an admissible Hamiltonian model space. `space_group`,
`wyckoff_orbits`, and `symmetry_action` are intentionally generic so the core
package does not depend on one crystallography implementation. `max_body_order`
is interpreted as algebra-aware many-body rank rather than primitive operator degree.
"""
struct GenerativeSpecification{G,W,H,A<:AbstractOperatorAlgebra,U,C,M}
    space_group::G
    wyckoff_orbits::W
    local_spaces::H
    algebra::A
    symmetry_action::U
    admissibility_conditions::C
    max_range::Union{Nothing,Float64}
    max_body_order::Union{Nothing,Int}
    metadata::M
end

function GenerativeSpecification(space_group, wyckoff_orbits, local_spaces, algebra::A, symmetry_action;
                                 admissibility_conditions=(), max_range=nothing, max_body_order=nothing,
                                 metadata=Dict{Symbol,Any}()) where {A<:AbstractOperatorAlgebra}
    if !isnothing(max_range)
        isfinite(max_range) && max_range >= 0 || throw(ArgumentError("max_range must be finite and nonnegative"))
    end
    if !isnothing(max_body_order)
        Int(max_body_order) > 0 || throw(ArgumentError("max_body_order must be positive"))
    end
    return GenerativeSpecification(space_group, wyckoff_orbits, local_spaces, algebra, symmetry_action,
                                   Tuple(admissibility_conditions), isnothing(max_range) ? nothing : Float64(max_range),
                                   isnothing(max_body_order) ? nothing : Int(max_body_order), metadata)
end

"""
    HamiltonianBasis(operators; labels=nothing, metadata=Dict())

Linearly independent Hermitian operator basis `{Phi_a}` for one admissible
Hamiltonian space. Linear independence and completeness are mathematical
properties established by the generator/validator that created the object;
this container records those claims explicitly in `metadata` rather than
silently asserting them.
"""
struct HamiltonianBasis{O,L,M}
    operators::O
    labels::L
    metadata::M
end

function HamiltonianBasis(operators::AbstractVector{<:AbstractOperatorExpr}; labels=nothing, metadata=Dict{Symbol,Any}())
    isempty(operators) && throw(ArgumentError("HamiltonianBasis requires at least one operator"))
    ops = collect(operators)
    names = isnothing(labels) ? [Symbol("Phi_", i) for i in eachindex(ops)] : Symbol.(collect(labels))
    length(names) == length(ops) || throw(DimensionMismatch("basis labels and operators must have equal length"))
    length(unique(names)) == length(names) || throw(ArgumentError("HamiltonianBasis labels must be unique"))
    return HamiltonianBasis(ops, names, metadata)
end

Base.length(basis::HamiltonianBasis) = length(basis.operators)
Base.getindex(basis::HamiltonianBasis, i::Integer) = basis.operators[Int(i)]
Base.iterate(basis::HamiltonianBasis, state...) = iterate(basis.operators, state...)
model_space_dimension(basis::HamiltonianBasis) = length(basis)

"""
    HamiltonianSpace(specification, basis; metadata=Dict())

Finite-dimensional admissible Hamiltonian space `H(G)` represented by a real
basis of Hermitian operators. The object ties the generated basis to the
physical specification that defines its domain.
"""
struct HamiltonianSpace{S<:GenerativeSpecification,B<:HamiltonianBasis,M}
    specification::S
    basis::B
    metadata::M
end

HamiltonianSpace(specification::GenerativeSpecification, basis::HamiltonianBasis; metadata=Dict{Symbol,Any}()) =
    HamiltonianSpace(specification, basis, metadata)

Base.length(space::HamiltonianSpace) = length(space.basis)
model_space_dimension(space::HamiltonianSpace) = length(space)

function admissible(specification::GenerativeSpecification, candidate)
    if !isnothing(specification.max_body_order) && body_order(specification.algebra, candidate) > specification.max_body_order
        return false
    end
    for condition in specification.admissibility_conditions
        condition isa AbstractAdmissibilityCondition || throw(ArgumentError("admissibility conditions must subtype AbstractAdmissibilityCondition"))
        admissible(condition, candidate) || return false
    end
    return true
end

admissible(space::HamiltonianSpace, candidate) = admissible(space.specification, candidate)

"""
    hamiltonian_from_coefficients(space, theta)

Construct `H(theta) = sum_a theta[a] Phi_a` in the generated operator basis.
The coefficients are required to be real because `HamiltonianSpace` is a real
span of Hermitian basis operators.
"""
function hamiltonian_from_coefficients(space::HamiltonianSpace, theta::AbstractVector{<:Real})
    length(theta) == length(space) || throw(DimensionMismatch("parameter vector length does not match Hamiltonian-space dimension"))
    terms = AbstractOperatorExpr[]
    sizehint!(terms, length(theta))
    for (coefficient, operator) in zip(theta, space.basis.operators)
        iszero(coefficient) && continue
        push!(terms, coefficient * operator)
    end
    isempty(terms) && return 0 * IdentityOperator()
    return length(terms) == 1 ? terms[1] : OperatorSum(terms)
end

"""
    generated_basis(specification, generator)

Run a user- or backend-supplied invariant-basis generator. The generator must
return either a `HamiltonianBasis` or a vector of operator expressions. This
keeps the model-space API independent of a particular symmetry engine.
"""
function generated_basis(specification::GenerativeSpecification, generator)
    result = generator(specification)
    result isa HamiltonianBasis && return result
    result isa AbstractVector{<:AbstractOperatorExpr} && return HamiltonianBasis(result)
    throw(ArgumentError("basis generator must return HamiltonianBasis or a vector of operator expressions"))
end

function generate_hamiltonian_space(specification::GenerativeSpecification, generator; metadata=Dict{Symbol,Any}())
    return HamiltonianSpace(specification, generated_basis(specification, generator); metadata=metadata)
end

# Crystallographic generators intentionally enter through the existing callable-generator interface. The crystallography layer defines `CrystallographicGenerator` rather than coupling Core to Spglib.jl types.
