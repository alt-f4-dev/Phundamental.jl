# Sequence acceleration and convergence diagnostics for order-resolved NLCE partial sums.

"""Abstract policy for transforming an order-resolved NLCE partial-sum sequence into a final estimate."""
abstract type AbstractNLCEResummation end

"""Return the bare highest-order NLCE partial sum without sequence acceleration."""
struct BareNLCE <: AbstractNLCEResummation end

"""Wynn epsilon sequence acceleration applied to NLCE partial sums."""
struct WynnEpsilon <: AbstractNLCEResummation
    cycles::Int
    denominator_tol::Float64
    function WynnEpsilon(; cycles::Integer=3, denominator_tol::Real=1e-14)
        cycles >= 1 || throw(ArgumentError("cycles must be positive"))
        denominator_tol > 0 || throw(ArgumentError("denominator_tol must be positive"))
        new(Int(cycles), Float64(denominator_tol))
    end
end

"""Result of an NLCE sequence-acceleration operation."""
struct NLCEResummationResult{M,E,H,D}
    method::M
    estimate::E
    history::H
    metadata::D
end

function _epsilon_inverse(value::Number, tol::Float64)
    abs(value) > tol || return complex(Inf)
    return inv(value)
end

function _epsilon_inverse(value::AbstractArray, tol::Float64)
    out = similar(complex.(value))
    for index in eachindex(value)
        out[index] = abs(value[index]) > tol ? inv(value[index]) : complex(Inf)
    end
    return out
end

function _epsilon_finite(value::Number)
    return isfinite(real(value)) && isfinite(imag(value))
end

_epsilon_finite(value::AbstractArray) = all(x -> isfinite(real(x)) && isfinite(imag(x)), value)

function _wynn_epsilon(sequence, cycles::Int, denominator_tol::Float64)
    values = [_linked_copy(value) for value in sequence]
    n = length(values)
    n >= 3 || throw(ArgumentError("Wynn epsilon acceleration requires at least three NLCE orders"))
    maximum_cycles = min(cycles, fld(n - 1, 2))
    previous = Any[_linked_scale(value, 0.0) for value in values]
    current = Any[_linked_copy(value) for value in values]
    approximants = Any[_linked_copy(values[end])]
    completed = 0

    for level in 1:(2 * maximum_cycles)
        count = length(current) - 1
        count >= 1 || break
        next = Vector{Any}(undef, count)
        for i in 1:count
            difference = _linked_subtract(current[i + 1], current[i])
            inverse_difference = _epsilon_inverse(difference, denominator_tol)
            base = level == 1 ? _linked_scale(values[1], 0.0) : previous[i + 1]
            next[i] = _linked_add(base, inverse_difference)
        end
        previous, current = current, next
        if iseven(level)
            candidate = current[end]
            _epsilon_finite(candidate) || break
            push!(approximants, _linked_copy(candidate))
            completed += 1
        end
    end
    return approximants[end], approximants, completed
end

"""
    resum_nlce(result::NLCEResult, method::AbstractNLCEResummation=BareNLCE())

Return an `NLCEResummationResult` obtained from the order-resolved partial sums in `result`. `BareNLCE()` returns the highest-order bare sum, while `WynnEpsilon` applies Wynn epsilon sequence acceleration and records the completed acceleration history and diagnostics.
"""
function resum_nlce(result::NLCEResult, ::BareNLCE=BareNLCE())
    estimate = bulk_estimate(result)
    metadata = Dict{Symbol,Any}(:method => :bare, :orders_used => length(result.partial_sums), :accelerated => false)
    return NLCEResummationResult(BareNLCE(), _linked_copy(estimate), Any[_linked_copy(estimate)], metadata)
end

function resum_nlce(result::NLCEResult, method::WynnEpsilon)
    estimate, history, completed = _wynn_epsilon(result.partial_sums, method.cycles, method.denominator_tol)
    metadata = Dict{Symbol,Any}(
        :method => :wynn_epsilon,
        :orders_used => length(result.partial_sums),
        :requested_cycles => method.cycles,
        :completed_cycles => completed,
        :denominator_tol => method.denominator_tol,
        :accelerated => completed > 0,
    )
    return NLCEResummationResult(method, estimate, history, metadata)
end

"""Absolute change between the last two available bare or resummed estimates."""
function nlce_sequence_residual(result::NLCEResult)
    length(result.partial_sums) >= 2 || return Inf
    return _linked_distance(result.partial_sums[end], result.partial_sums[end - 1])
end

function nlce_sequence_residual(result::NLCEResummationResult)
    length(result.history) >= 2 || return Inf
    return _linked_distance(result.history[end], result.history[end - 1])
end
