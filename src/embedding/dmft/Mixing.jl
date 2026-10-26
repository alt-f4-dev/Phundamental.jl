# Self-energy mixing policies for fixed-point embedding iterations.

abstract type AbstractMixing end

"""Linear fixed-point mixing `Σ ← (1-α)Σ_old + αΣ_new`."""
struct LinearMixing <: AbstractMixing
    alpha::Float64
end

function LinearMixing(alpha::Real=0.5)
    0 < alpha <= 1 || throw(ArgumentError("linear-mixing alpha must lie in (0,1]"))
    return LinearMixing(Float64(alpha))
end

"""
    AdaptiveLinearMixing(; initial_alpha=0.5, min_alpha=0.05, max_alpha=0.8,
                         decrease_factor=0.5, increase_factor=1.1,
                         improvement_ratio=0.8, worsening_ratio=1.05, recovery_window=3)

Adaptive linear self-energy mixing for difficult DMFT fixed points. The active mixing fraction is reduced when the fixed-point residual worsens, increased after substantial improvement, and also increased after `recovery_window` consecutive monotonic improvements. The update remains a convex linear mixture at every iteration.
"""
struct AdaptiveLinearMixing <: AbstractMixing
    initial_alpha::Float64
    min_alpha::Float64
    max_alpha::Float64
    decrease_factor::Float64
    increase_factor::Float64
    improvement_ratio::Float64
    worsening_ratio::Float64
    recovery_window::Int
end

function AdaptiveLinearMixing(; initial_alpha::Real=0.5, min_alpha::Real=0.05, max_alpha::Real=0.8,
                              decrease_factor::Real=0.5, increase_factor::Real=1.1,
                              improvement_ratio::Real=0.8, worsening_ratio::Real=1.05, recovery_window::Integer=3)
    0 < min_alpha <= max_alpha <= 1 || throw(ArgumentError("adaptive mixing requires 0 < min_alpha <= max_alpha <= 1"))
    min_alpha <= initial_alpha <= max_alpha || throw(ArgumentError("initial_alpha must lie within [min_alpha,max_alpha]"))
    0 < decrease_factor < 1 || throw(ArgumentError("decrease_factor must lie in (0,1)"))
    increase_factor > 1 || throw(ArgumentError("increase_factor must exceed 1"))
    0 < improvement_ratio < 1 || throw(ArgumentError("improvement_ratio must lie in (0,1)"))
    worsening_ratio > 1 || throw(ArgumentError("worsening_ratio must exceed 1"))
    recovery_window >= 2 || throw(ArgumentError("recovery_window must be at least 2"))
    return AdaptiveLinearMixing(Float64(initial_alpha), Float64(min_alpha), Float64(max_alpha), Float64(decrease_factor),
                                Float64(increase_factor), Float64(improvement_ratio), Float64(worsening_ratio), Int(recovery_window))
end

# Backward-compatible positional construction for the pre-recovery adaptive-mixing layout.
AdaptiveLinearMixing(initial_alpha::Float64, min_alpha::Float64, max_alpha::Float64, decrease_factor::Float64, increase_factor::Float64,
                     improvement_ratio::Float64, worsening_ratio::Float64) =
    AdaptiveLinearMixing(initial_alpha, min_alpha, max_alpha, decrease_factor, increase_factor, improvement_ratio, worsening_ratio, 3)

function _mix_self_energy(old::SelfEnergy, new::SelfEnergy, alpha::Float64, mode::Symbol)
    _compatible_frequency_objects(old, new)
    values = (1 - alpha) .* old.values .+ alpha .* new.values
    metadata = Dict{Symbol,Any}(:mixing => mode, :alpha => alpha)
    return SelfEnergy(old.axis, values; labels=old.labels, metadata=metadata)
end

mix_self_energy(mixing::LinearMixing, old::SelfEnergy, new::SelfEnergy) = _mix_self_energy(old, new, mixing.alpha, :linear)
mix_self_energy(mixing::AdaptiveLinearMixing, old::SelfEnergy, new::SelfEnergy) = _mix_self_energy(old, new, mixing.initial_alpha, :adaptive_linear)

function mix_self_energy(mixing::AdaptiveLinearMixing, old::SelfEnergy, new::SelfEnergy, alpha::Real)
    alpha_value = Float64(alpha)
    mixing.min_alpha <= alpha_value <= mixing.max_alpha || throw(ArgumentError("adaptive mixing alpha lies outside the configured bounds"))
    return _mix_self_energy(old, new, alpha_value, :adaptive_linear)
end

_mixing_alpha(mixing::LinearMixing) = mixing.alpha
_mixing_alpha(mixing::AdaptiveLinearMixing) = mixing.initial_alpha

_next_mixing_alpha(::LinearMixing, alpha::Float64, previous_residual::Float64, residual::Float64) = alpha

_next_mixing_state(::LinearMixing, alpha::Float64, previous_residual::Float64, residual::Float64, improving_streak::Int) = (alpha, 0)

function _next_mixing_state(mixing::AdaptiveLinearMixing, alpha::Float64, previous_residual::Float64, residual::Float64, improving_streak::Int)
    isfinite(residual) || return (mixing.min_alpha, 0)
    isfinite(previous_residual) || return (clamp(alpha, mixing.min_alpha, mixing.max_alpha), 0)
    previous_residual > 0 || return (clamp(alpha, mixing.min_alpha, mixing.max_alpha), 0)
    ratio = residual / previous_residual
    if ratio >= mixing.worsening_ratio
        return (max(mixing.min_alpha, alpha * mixing.decrease_factor), 0)
    elseif ratio <= mixing.improvement_ratio
        return (min(mixing.max_alpha, alpha * mixing.increase_factor), 0)
    elseif ratio < 1
        next_streak = improving_streak + 1
        if next_streak >= mixing.recovery_window
            return (min(mixing.max_alpha, alpha * mixing.increase_factor), 0)
        end
        return (clamp(alpha, mixing.min_alpha, mixing.max_alpha), next_streak)
    end
    return (clamp(alpha, mixing.min_alpha, mixing.max_alpha), 0)
end

function _next_mixing_alpha(mixing::AdaptiveLinearMixing, alpha::Float64, previous_residual::Float64, residual::Float64)
    next_alpha, _ = _next_mixing_state(mixing, alpha, previous_residual, residual, 0)
    return next_alpha
end

function self_energy_residual(old::SelfEnergy, new::SelfEnergy)
    _compatible_frequency_objects(old, new)
    numerator = maximum(abs.(new.values .- old.values); init=0.0)
    denominator = max(maximum(abs.(new.values); init=0.0), maximum(abs.(old.values); init=0.0), 1.0)
    return numerator / denominator
end
