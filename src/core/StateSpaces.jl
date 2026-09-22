"""
    AbstractStateSpace

Root type for mathematical state spaces used by the package.

A state space is the ambient space in which a representation is defined.  The
physical sector and any numerical truncation are represented separately.
"""
abstract type AbstractStateSpace end

"""Finite or otherwise explicitly Hilbert-space-like state spaces."""
abstract type AbstractHilbertSpace <: AbstractStateSpace end

"""Bosonic or fermionic Fock spaces."""
abstract type AbstractFockSpace <: AbstractStateSpace end

"""
    SpinHilbertSpace(spins)

Tensor-product spin Hilbert space with one spin quantum number per site.
Each entry may be an integer or half-integer value such as `1//2`, `1`, or
`3//2`.
"""
struct SpinHilbertSpace{T<:Real} <: AbstractHilbertSpace
    spins::Vector{T}

    function SpinHilbertSpace(spins::AbstractVector{T}) where {T<:Real}
        isempty(spins) && throw(ArgumentError("at least one spin site is required"))
        any(s -> s < 0, spins) && throw(ArgumentError("spin quantum numbers must be nonnegative"))
        new{T}(collect(spins))
    end
end

"""Fermionic Fock space for `nmodes` canonical fermion modes."""
struct FermionFockSpace <: AbstractFockSpace
    nmodes::Int

    function FermionFockSpace(nmodes::Integer)
        nmodes < 0 && throw(ArgumentError("nmodes must be nonnegative"))
        new(Int(nmodes))
    end
end

"""Bosonic Fock space for `nmodes` canonical boson modes."""
struct BosonFockSpace <: AbstractFockSpace
    nmodes::Int

    function BosonFockSpace(nmodes::Integer)
        nmodes < 0 && throw(ArgumentError("nmodes must be nonnegative"))
        new(Int(nmodes))
    end
end

"""
    CoordinateHilbertSpace(ndof)

Canonical coordinate-momentum Hilbert space for `ndof` continuous degrees of
freedom.  Its exact dimension is infinite and is therefore reported as
`nothing` by `spacedimension`.
"""
struct CoordinateHilbertSpace <: AbstractHilbertSpace
    ndof::Int

    function CoordinateHilbertSpace(ndof::Integer)
        ndof < 0 && throw(ArgumentError("ndof must be nonnegative"))
        new(Int(ndof))
    end
end

"""Tensor-product state space composed of multiple component spaces."""
struct CompositeStateSpace{S<:Tuple} <: AbstractStateSpace
    factors::S

    function CompositeStateSpace(factors::S) where {S<:Tuple}
        all(x -> x isa AbstractStateSpace, factors) ||
            throw(ArgumentError("all composite factors must be state spaces"))
        new{S}(factors)
    end
end

CompositeStateSpace(spaces::AbstractStateSpace...) = CompositeStateSpace(tuple(spaces...))

"""
    BiorthogonalSpace(ambient, metric=nothing)

State-space wrapper for a representation whose natural left and right bases
are distinct, as in a nonunitary similarity representation.
"""
struct BiorthogonalSpace{S<:AbstractStateSpace,M} <: AbstractStateSpace
    ambient::S
    metric::M
end

BiorthogonalSpace(ambient::AbstractStateSpace) = BiorthogonalSpace(ambient, nothing)

"""
    PhysicalSubspace(ambient; constraints=(), projector=nothing, dimension=nothing)

Exact physical sector inside an ambient state space.  This object is for
physical representation constraints, not numerical cutoffs.
"""
struct PhysicalSubspace{S<:AbstractStateSpace,C,P}
    ambient::S
    constraints::C
    projector::P
    dimension::Union{Nothing,Int}
end

function PhysicalSubspace(
    ambient::AbstractStateSpace;
    constraints=(),
    projector=nothing,
    dimension::Union{Nothing,Integer}=nothing,
)
    dim = isnothing(dimension) ? nothing : Int(dimension)
    !isnothing(dim) && dim < 0 && throw(ArgumentError("dimension must be nonnegative"))
    return PhysicalSubspace(ambient, constraints, projector, dim)
end

"""
    NumericalTruncation(cutoffs; dimension=nothing, description="")

Computational truncation of an otherwise exact mathematical state space.
Numerical truncation is deliberately distinct from `PhysicalSubspace`.
"""
struct NumericalTruncation{C}
    cutoffs::C
    dimension::Union{Nothing,Int}
    description::String
end

function NumericalTruncation(
    cutoffs;
    dimension::Union{Nothing,Integer}=nothing,
    description::AbstractString="",
)
    dim = isnothing(dimension) ? nothing : Int(dimension)
    !isnothing(dim) && dim < 0 && throw(ArgumentError("dimension must be nonnegative"))
    return NumericalTruncation(cutoffs, dim, String(description))
end

"""
    spacedimension(space)

Return the exact mathematical dimension when finite and known.  Return
`nothing` for infinite-dimensional or unspecified spaces.
"""
spacedimension(::AbstractStateSpace) = nothing

function spacedimension(space::SpinHilbertSpace)
    dims = map(space.spins) do S
        d = 2S + 1
        isinteger(d) || throw(ArgumentError("spin values must be integer or half-integer"))
        Int(d)
    end
    return prod(dims)
end

spacedimension(space::FermionFockSpace) = 2^space.nmodes
spacedimension(::BosonFockSpace) = nothing
spacedimension(::CoordinateHilbertSpace) = nothing
spacedimension(space::BiorthogonalSpace) = spacedimension(space.ambient)

function spacedimension(space::CompositeStateSpace)
    dims = map(spacedimension, space.factors)
    any(isnothing, dims) && return nothing
    return prod(Int(d) for d in dims)
end

physicaldimension(space::PhysicalSubspace) =
    isnothing(space.dimension) ? spacedimension(space.ambient) : space.dimension
