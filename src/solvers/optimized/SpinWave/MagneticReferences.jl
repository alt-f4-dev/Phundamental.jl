# Ordered SU(2) directions and arbitrary right-handed transverse gauges.

"""A unit vector at each spin site, represented as a 3-by-N real matrix."""
struct MagneticReferenceState
    directions::Matrix{Float64}
    frames::Vector{Matrix{Float64}}
end

function MagneticReferenceState(directions::AbstractMatrix{<:Real}; transverse_axes=nothing, tol::Real=1e-10)
    size(directions, 1) == 3 || throw(DimensionMismatch("magnetic directions must have size (3,N)"))
    size(directions, 2) > 0 || throw(ArgumentError("at least one magnetic direction is required"))
    all(isfinite, directions) || throw(ArgumentError("magnetic directions must be finite"))
    tol > 0 || throw(ArgumentError("reference tolerance must be positive"))
    n = Matrix{Float64}(directions)
    frames = Matrix{Float64}[]
    for i in axes(n, 2)
        ni = view(n, :, i)
        abs(norm(ni) - 1) <= tol || throw(ArgumentError("reference direction $i must be normalized"))
        hint = if isnothing(transverse_axes)
            seed = zeros(Float64, 3)
            seed[argmin(abs.(ni))] = 1.0
            seed
        else
            size(transverse_axes) == size(n) || throw(DimensionMismatch("transverse axes must have size (3,N)"))
            Float64.(transverse_axes[:, i])
        end
        ex = hint - dot(hint, ni) * ni
        norm(ex) > tol || throw(ArgumentError("transverse axis $i is parallel to the magnetic moment"))
        ex ./= norm(ex)
        ey = cross(ni, ex)
        push!(frames, hcat(ex, ey, ni))
    end
    return MagneticReferenceState(n, frames)
end

function MagneticReferenceState(directions::AbstractVector; kwargs...)
    all(direction -> length(direction) == 3, directions) || throw(DimensionMismatch("each magnetic direction must contain three components"))
    return MagneticReferenceState(hcat((Float64.(collect(d)) for d in directions)...); kwargs...)
end

"""General finite-cell harmonic spin-wave solver, with dipolar SU(2) as the default.

`formalism=:SUN` uses arbitrary normalized local states on CP^(2S) and the exact local spin matrices. The choice of formalism is explicit and cannot be inferred from the Hamiltonian.
"""
struct SpinWaveSolver{R} <: AbstractModeSolver
    reference::R
    formalism::Symbol
    tol::Float64
    stationarity_tol::Float64
    zero_modes::Symbol
    regularization::Float64
end

function SpinWaveSolver(; reference::Union{MagneticReferenceState,SUNReferenceState}, formalism::Symbol=:dipole,
                        tol::Real=1e-10, stationarity_tol::Real=1e-8, zero_modes::Symbol=:reject,
                        regularization::Real=1e-8)
    formalism in (:dipole, :SUN) || throw(ArgumentError("spin-wave formalism must be :dipole or :SUN"))
    formalism === :dipole && !(reference isa MagneticReferenceState) && throw(ArgumentError(":dipole requires MagneticReferenceState"))
    tol > 0 || throw(ArgumentError("spin-wave tolerance must be positive"))
    stationarity_tol > 0 || throw(ArgumentError("stationarity tolerance must be positive"))
    zero_modes in (:reject, :regularize) || throw(ArgumentError("zero_modes must be :reject or :regularize"))
    regularization > 0 || throw(ArgumentError("zero-mode regularization must be positive"))
    return SpinWaveSolver(reference, formalism, Float64(tol), Float64(stationarity_tol), zero_modes, Float64(regularization))
end

SpinWaveSolver(reference::Union{MagneticReferenceState,SUNReferenceState}; kwargs...) = SpinWaveSolver(; reference, kwargs...)
solver_name(::SpinWaveSolver) = :spin_wave

# Preserve the former generated five-argument constructor as a dipolar compatibility route.
SpinWaveSolver(reference::MagneticReferenceState, tol::Float64, stationarity_tol::Float64, zero_modes::Symbol, regularization::Float64) =
    SpinWaveSolver(reference, :dipole, tol, stationarity_tol, zero_modes, regularization)
