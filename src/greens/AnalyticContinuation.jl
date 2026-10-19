# Explicit approximate rational continuation of scalar fermionic Matsubara Green functions.
# No analytic continuation is performed implicitly by a forward solver.

using LinearAlgebra: svdvals, norm

"""A consciously approximate rational fit in the upper half-plane, with explicit conditioning and validation limits."""
struct RationalContinuation
    numerator_order::Int
    denominator_order::Int
    eta::Float64
    max_condition::Float64
    max_relative_fit_residual::Float64
end

function RationalContinuation(; numerator_order::Integer=1, denominator_order::Integer=1, eta::Real=0.05,
                               max_condition::Real=1e10, max_relative_fit_residual::Real=1e-4)
    numerator_order >= 0 && denominator_order >= 0 || throw(ArgumentError("rational orders must be nonnegative"))
    isfinite(eta) && eta > 0 || throw(ArgumentError("continuation broadening eta must be finite and positive"))
    isfinite(max_condition) && max_condition > 1 || throw(ArgumentError("maximum conditioning threshold must exceed one"))
    isfinite(max_relative_fit_residual) && max_relative_fit_residual >= 0 ||
        throw(ArgumentError("maximum relative fit residual must be finite and nonnegative"))
    return RationalContinuation(Int(numerator_order), Int(denominator_order), Float64(eta), Float64(max_condition),
                                Float64(max_relative_fit_residual))
end

"""Approximate upper-half-plane continuation, including fitting and causality diagnostics."""
struct ContinuationResult
    axis::Vector{Float64}
    retarded::Vector{ComplexF64}
    spectral_density::Vector{Float64}
    fit_residual::Float64
    condition_number::Float64
    minimum_spectral_density::Float64
    metadata::Dict{Symbol,Any}
end

"""
    continue_matsubara(axis, green_values, real_axis; method=RationalContinuation())

Fit `(a₀ + ⋯ + aₘzᵐ)/(1 + b₁z + ⋯ + bₙzⁿ)` on scalar fermionic Matsubara data and evaluate at `ω + iη`. Reject singular or inaccurate fits. A small residual does not prove a causal, accurate continuation; the returned spectral density is not silently clipped to nonnegative values.
"""
function continue_matsubara(axis::FermionicMatsubaraAxis, green_values::AbstractVector, real_axis;
                            method::RationalContinuation=RationalContinuation())
    length(green_values) == length(axis) || throw(DimensionMismatch("Matsubara data and frequency axis lengths differ"))
    data = ComplexF64.(collect(green_values))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), data) || throw(ArgumentError("Matsubara Green function must be finite"))
    frequency = Float64.(collect(real_axis))
    length(frequency) >= 2 && all(diff(frequency) .> 0) && all(isfinite, frequency) ||
        throw(ArgumentError("continuation real-energy axis must contain strictly increasing finite values"))
    sample = findall(z -> imag(z) > 0, axis.values)
    nparameters = method.numerator_order + method.denominator_order + 1
    length(sample) > nparameters || throw(ArgumentError("rational continuation requires more positive Matsubara samples than fitted coefficients"))
    z = ComplexF64[axis.values[i] for i in sample]
    G = ComplexF64[data[i] for i in sample]
    scale = maximum(abs, z)
    z ./= scale
    matrix = Matrix{ComplexF64}(undef, length(z), nparameters)
    for j in 0:method.numerator_order
        matrix[:, j + 1] .= z .^ j
    end
    for j in 1:method.denominator_order
        matrix[:, method.numerator_order + 1 + j] .= -G .* (z .^ j)
    end
    singular = svdvals(matrix)
    smallest = minimum(singular)
    condition_number = smallest > 0 ? maximum(singular) / smallest : Inf
    isfinite(condition_number) && condition_number <= method.max_condition ||
        throw(ArgumentError("rational continuation is ill-conditioned (condition=$condition_number)"))
    coefficients = matrix \ G
    residual = norm(matrix * coefficients - G) / max(norm(G), eps(Float64))
    residual <= method.max_relative_fit_residual ||
        throw(ArgumentError("rational continuation fit residual $residual exceeds the declared tolerance"))
    retarded = Vector{ComplexF64}(undef, length(frequency))
    for (i, omega) in enumerate(frequency)
        point = (omega + im * method.eta) / scale
        numerator = sum(coefficients[j + 1] * point^j for j in 0:method.numerator_order)
        denominator = 1 + sum(coefficients[method.numerator_order + 1 + j] * point^j for j in 1:method.denominator_order)
        abs(denominator) > 1e-12 || throw(ArgumentError("rational continuation encountered a pole near the retarded evaluation contour"))
        retarded[i] = numerator / denominator
    end
    density = -imag.(retarded) ./ π
    all(isfinite, density) || throw(ArgumentError("continued spectrum has nonfinite values"))
    metadata = Dict{Symbol,Any}(:method => :rational_matsubara_fit, :approximate => true,
                                :frequency_domain => :real, :input_frequency_domain => :matsubara,
                                :numerator_order => method.numerator_order, :denominator_order => method.denominator_order,
                                :broadening => method.eta, :sample_count => length(z),
                                :causality_certified => false, :numerical_uncertainty_certified => false)
    return ContinuationResult(frequency, retarded, density, residual, condition_number, minimum(density), metadata)
end

function continue_matsubara(green::GreenFunction, real_axis; method::RationalContinuation=RationalContinuation())
    axis = frequency_axis(green)
    axis isa FermionicMatsubaraAxis || throw(ArgumentError("only scalar fermionic Matsubara Green functions are supported"))
    matrix_dimension(green) == 1 || throw(ArgumentError("multiorbital matrix analytic continuation requires a separate method"))
    return continue_matsubara(axis, scalar_frequency_values(green), real_axis; method=method)
end
