# Representation-level objects: basis descriptors, gauge metadata, and the
# complete mathematical representation R.

abstract type AbstractBasis end

struct SpinProductBasis <: AbstractBasis
    ordering::Vector{Int}
end

struct FermionOccupationBasis <: AbstractBasis
    ordering::Vector{Int}
end

struct BosonOccupationBasis <: AbstractBasis
    ordering::Vector{Int}
end

struct CoordinateBasis <: AbstractBasis
    ordering::Vector{Int}
end

struct CompositeBasis{B<:Tuple} <: AbstractBasis
    factors::B
end
CompositeBasis(bases::AbstractBasis...) = CompositeBasis(tuple(bases...))


# Explicit momentum-space coordinate wrappers. These are representation-level
# descriptors rather than lattice objects so numerical solvers can distinguish
# Cartesian and reciprocal-lattice coordinates without depending on the Models
# module.

abstract type AbstractWaveVector end

struct CartesianWaveVector{T<:Real} <: AbstractWaveVector
    coordinates::Vector{T}
end

struct ReciprocalWaveVector{T<:Real} <: AbstractWaveVector
    coordinates::Vector{T}
end

function CartesianWaveVector(q::AbstractVector{<:Real})
    isempty(q) && throw(ArgumentError("wavevector cannot be empty"))
    all(isfinite, q) || throw(ArgumentError("wavevector components must be finite"))
    T = float(eltype(q))
    return CartesianWaveVector{T}(T.(collect(q)))
end

function ReciprocalWaveVector(hkl::AbstractVector{<:Real})
    isempty(hkl) && throw(ArgumentError("wavevector cannot be empty"))
    all(isfinite, hkl) || throw(ArgumentError("wavevector components must be finite"))
    T = float(eltype(hkl))
    return ReciprocalWaveVector{T}(T.(collect(hkl)))
end

CartesianWaveVector(q::Tuple{Vararg{Real}}) = CartesianWaveVector(collect(q))
ReciprocalWaveVector(q::Tuple{Vararg{Real}}) = ReciprocalWaveVector(collect(q))

Base.length(q::AbstractWaveVector) = length(q.coordinates)
Base.getindex(q::AbstractWaveVector, i::Integer) = q.coordinates[Int(i)]
Base.iterate(q::AbstractWaveVector, state...) = iterate(q.coordinates, state...)
Base.eltype(::Type{CartesianWaveVector{T}}) where {T} = T
Base.eltype(::Type{ReciprocalWaveVector{T}}) where {T} = T
Base.collect(q::AbstractWaveVector) = collect(q.coordinates)

wavevector_coordinates(q::AbstractWaveVector) = copy(q.coordinates)

"""Descriptor for a left/right biorthogonal basis pair."""
struct BiorthogonalBasis{R,L} <: AbstractBasis
    right::R
    left::L
end

"""
    GaugeStructure(group, generators; description="")

Metadata describing an internal gauge redundancy of an enlarged
representation.  `generators` may contain symbolic constraint/gauge operators.
"""
struct GaugeStructure{G}
    group::Symbol
    generators::G
    description::String
end

GaugeStructure(group::Symbol, generators; description::AbstractString="") = GaugeStructure(group, generators, String(description))

"""
    Representation

Complete representation object

    R = (S, S_phys, A, B, C, G, |Omega>, ordering, truncation)

where `state_space` is the ambient mathematical state space and
`physical_subspace` is either `nothing` (the whole ambient space is physical)
or an exact constrained sector.  `truncation` is computational metadata and is
never interpreted as a physical constraint.
"""
struct Representation{S<:AbstractStateSpace,A<:AbstractOperatorAlgebra,B<:AbstractBasis,P,G,R,O,T}
    name::Symbol
    state_space::S
    physical_subspace::P
    algebra::A
    basis::B
    constraints::Tuple
    gauge_structure::G
    reference_state::R
    ordering::O
    truncation::T
end

function Representation(
    name::Symbol,
    state_space::S,
    algebra::A,
    basis::B;
    physical_subspace=nothing,
    constraints=(),
    gauge_structure=nothing,
    reference_state=nothing,
    ordering=nothing,
    truncation=nothing,
) where {S<:AbstractStateSpace,A<:AbstractOperatorAlgebra,B<:AbstractBasis}
    cs = Tuple(constraints)

    if !isnothing(physical_subspace)
        physical_subspace isa PhysicalSubspace ||
            throw(ArgumentError("physical_subspace must be a PhysicalSubspace or nothing"))
        physical_subspace.ambient == state_space ||
            throw(ArgumentError("physical_subspace must refer to the representation's ambient state space"))
    end

    if !isnothing(truncation)
        valid = truncation isa NumericalTruncation || truncation isa FactorTruncation ||
                (truncation isa Tuple && all(x -> x isa FactorTruncation, truncation)) ||
                (truncation isa AbstractVector && all(x -> x isa FactorTruncation, truncation))
        valid || throw(ArgumentError("truncation must be NumericalTruncation, FactorTruncation, a collection of FactorTruncation, or nothing"))
    end

    return Representation(
        name,
        state_space,
        physical_subspace,
        algebra,
        basis,
        cs,
        gauge_structure,
        reference_state,
        ordering,
        truncation,
    )
end

"""True when the mathematical representation uses a proper physical subspace."""
isconstrained(rep::Representation) = !isnothing(rep.physical_subspace) || !isempty(rep.constraints)

"""True when a numerical (not physical) truncation has been attached."""
istruncated(rep::Representation) = !isnothing(rep.truncation)

"""Ambient mathematical dimension, or `nothing` when infinite/unspecified."""
ambientdimension(rep::Representation) = spacedimension(rep.state_space)

"""Exact physical dimension when known, otherwise `nothing`."""
function representationdimension(rep::Representation)
    isnothing(rep.physical_subspace) && return spacedimension(rep.state_space)
    return physicaldimension(rep.physical_subspace)
end

"""Return the ambient space when all states are physical, otherwise the exact physical-sector descriptor."""
physicalspace(rep::Representation) = isnothing(rep.physical_subspace) ? rep.state_space : rep.physical_subspace

# ---------------------------------------------------------------------------
# Factor-local representation metadata
# ---------------------------------------------------------------------------

"""Attach one physical constraint to a specific tensor factor."""
struct FactorConstraint{C}
    factor::Int
    constraint::C
    function FactorConstraint(factor::Integer, constraint::C) where {C}
        factor > 0 || throw(ArgumentError("tensor-factor index must be positive"))
        new{C}(Int(factor), constraint)
    end
end

"""Attach one numerical truncation to a specific tensor factor."""
struct FactorTruncation{T<:NumericalTruncation}
    factor::Int
    truncation::T
    function FactorTruncation(factor::Integer, truncation::T) where {T<:NumericalTruncation}
        factor > 0 || throw(ArgumentError("tensor-factor index must be positive"))
        new{T}(Int(factor), truncation)
    end
end

factorconstraint(factor::Integer, constraint) = FactorConstraint(factor, constraint)
factortruncation(factor::Integer, truncation::NumericalTruncation) = FactorTruncation(factor, truncation)

"""
    with_truncation(rep, truncation)

Return a copy of `rep` with only its numerical truncation metadata replaced. `truncation` may be a `NumericalTruncation`, a `FactorTruncation`, a collection of `FactorTruncation`, or `nothing` to clear the truncation. Exact physical constraints are preserved unchanged.
"""
function with_truncation(rep::Representation, truncation)
    return Representation(
        rep.name,
        rep.state_space,
        rep.algebra,
        rep.basis;
        physical_subspace=rep.physical_subspace,
        constraints=rep.constraints,
        gauge_structure=rep.gauge_structure,
        reference_state=rep.reference_state,
        ordering=rep.ordering,
        truncation=truncation,
    )
end
