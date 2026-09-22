# InteractionTypes.jl

abstract type AbstractLongRangeInteraction end
abstract type AbstractInteractionSummation end

"""
    CoulombInteraction(strength=1; neutralizing_background=false)

Periodic scalar interaction `strength/r`.

`strength` carries all unit-system prefactors. For example it may contain
`1/(4πϵ0)` in SI-derived units.

A fully periodic Coulomb cell must be charge neutral unless
`neutralizing_background=true`, in which case the standard uniform compensating
background term is included in the Ewald pair matrix.
"""
struct CoulombInteraction{T<:Real} <: AbstractLongRangeInteraction
    strength::T
    neutralizing_background::Bool
end

CoulombInteraction(
    strength::Real=1.0;
    neutralizing_background::Bool=false,
) = CoulombInteraction(float(strength), neutralizing_background)

"""
    DipolarInteraction(strength=1)

Tensor interaction

    strength * [I/r^3 - 3 rr'/r^5].

`strength` carries the physical unit prefactor. The Ewald implementation uses
conducting/tin-foil boundary conditions, for which the macroscopic surface
term vanishes.
"""
struct DipolarInteraction{T<:Real} <: AbstractLongRangeInteraction
    strength::T

    function DipolarInteraction{T}(strength) where {T<:Real}
        return new{T}(convert(T, strength))
    end
end

function DipolarInteraction(strength::Real=1.0)
    x = float(strength)
    return DipolarInteraction{typeof(x)}(x)
end

"""
    RealSpaceCutoff(cutoff)

Direct lattice-image truncation. For periodic Coulomb systems this is a
cutoff-dependent approximation and should not be confused with Ewald summation.
"""
struct RealSpaceCutoff <: AbstractInteractionSummation
    cutoff::Float64
end

function RealSpaceCutoff(cutoff::Real)
    cutoff > 0 || throw(ArgumentError("real-space cutoff must be positive"))
    return RealSpaceCutoff(Float64(cutoff))
end

"""
    EwaldSummation(; alpha=nothing, real_cutoff=nothing,
                     reciprocal_cutoff=nothing, tolerance=1e-9,
                     check_convergence=true, refinement_factor=1.35,
                     max_refinements=4, strict=true,
                     boundary=:tin_foil)

Three-dimensional Ewald summation with explicit real, reciprocal, self, and
background pieces.

Only fully periodic 3D cells and `boundary=:tin_foil` are accepted in this
implementation. This is deliberate: nonconducting dipolar surface terms depend
on macroscopic sample shape and should never be inserted implicitly.

When `check_convergence=true`, cutoffs are enlarged repeatedly until successive
pair matrices agree within `tolerance`, or `max_refinements` is reached.

Future optimization:
    Particle-mesh Ewald (PME/P3M) should be added as a separate summation
    strategy. This direct reciprocal-space implementation is the correctness
    reference backend and does not pretend to be PME.
"""
struct EwaldSummation <: AbstractInteractionSummation
    alpha::Union{Nothing,Float64}
    real_cutoff::Union{Nothing,Float64}
    reciprocal_cutoff::Union{Nothing,Float64}
    tolerance::Float64
    check_convergence::Bool
    refinement_factor::Float64
    max_refinements::Int
    strict::Bool
    boundary::Symbol
end

function EwaldSummation(
    ; alpha=nothing,
      real_cutoff=nothing,
      reciprocal_cutoff=nothing,
      tolerance::Real=1e-9,
      check_convergence::Bool=true,
      refinement_factor::Real=1.35,
      max_refinements::Integer=4,
      strict::Bool=true,
      boundary::Symbol=:tin_foil,
)
    !isnothing(alpha) && alpha <= 0 &&
        throw(ArgumentError("Ewald alpha must be positive"))
    !isnothing(real_cutoff) && real_cutoff <= 0 &&
        throw(ArgumentError("real_cutoff must be positive"))
    !isnothing(reciprocal_cutoff) && reciprocal_cutoff <= 0 &&
        throw(ArgumentError("reciprocal_cutoff must be positive"))
    0 < tolerance < 1 || throw(ArgumentError("tolerance must lie in (0,1)"))
    refinement_factor > 1 ||
        throw(ArgumentError("refinement_factor must exceed one"))
    max_refinements >= 1 ||
        throw(ArgumentError("max_refinements must be at least one"))
    boundary === :tin_foil ||
        throw(ArgumentError(
            "only boundary=:tin_foil is currently supported; " *
            "shape-dependent surface terms must be specified explicitly in a future extension"
        ))

    return EwaldSummation(
        isnothing(alpha) ? nothing : Float64(alpha),
        isnothing(real_cutoff) ? nothing : Float64(real_cutoff),
        isnothing(reciprocal_cutoff) ? nothing : Float64(reciprocal_cutoff),
        Float64(tolerance),
        check_convergence,
        Float64(refinement_factor),
        Int(max_refinements),
        strict,
        boundary,
    )
end

struct ScalarInteractionResult{T<:Real}
    total::Matrix{T}
    real_space::Matrix{T}
    reciprocal_space::Matrix{T}
    self::Matrix{T}
    background::Matrix{T}
    metadata::Dict{Symbol,Any}
end

struct TensorInteractionResult{T<:Real}
    total::Array{T,4}              # (i,j,alpha,beta)
    real_space::Array{T,4}
    reciprocal_space::Array{T,4}
    self::Array{T,4}
    surface::Array{T,4}
    metadata::Dict{Symbol,Any}
end

interaction_matrix(result::ScalarInteractionResult) = result.total
interaction_tensor(result::TensorInteractionResult) = result.total
