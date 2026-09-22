# Observable-level validation and sum-rule diagnostics.

"""
Compare two scalar or tensor spectra generated through different exact
representation paths.  This is the observable-level form of representation
invariance: transformed Hamiltonians and transformed observables must produce
the same physical spectrum up to numerical tolerance.
"""
function validate_observable_equivalence(
    A::Union{SpectrumResult,TensorSpectrumResult,ScatteringResult},
    B::Union{SpectrumResult,TensorSpectrumResult,ScatteringResult};
    atol::Real=1e-10,
    rtol::Real=1e-8,
)
    length(A.axis) == length(B.axis) || return false
    length(A.q) == length(B.q) || return false
    all(isapprox(a, b; atol=atol, rtol=rtol) for (a, b) in zip(A.q, B.q)) || return false
    all(isapprox(a, b; atol=atol, rtol=rtol) for (a, b) in zip(A.axis, B.axis)) || return false
    size(A.intensity) == size(B.intensity) || return false
    return isapprox(A.intensity, B.intensity; atol=atol, rtol=rtol)
end

"""Trapezoidal integral along the spectral axis for each q value."""
function integrated_spectral_weight(spectrum::SpectrumResult)
    x = spectrum.axis
    y = spectrum.intensity
    out = zeros(ComplexF64, size(y, 1))
    for iq in axes(y, 1)
        total = 0.0 + 0im
        @inbounds for j in 1:length(x)-1
            total += 0.5 * (y[iq, j] + y[iq, j+1]) * (x[j+1] - x[j])
        end
        out[iq] = total
    end
    return out
end

function integrated_spectral_weight(spectrum::TensorSpectrumResult)
    x = spectrum.axis
    y = spectrum.intensity
    (size(y, 3) == 3 && size(y, 4) == 3) || throw(DimensionMismatch("spin tensor spectrum must have 3×3 Cartesian components"))
    out = zeros(ComplexF64, size(y, 1), 3, 3)

    for iq in axes(y, 1), α in 1:3, β in 1:3
        total = 0.0 + 0im

        @inbounds for j in 1:length(x)-1
            total += 0.5 * (y[iq, j, α, β] + y[iq, j + 1, α, β]) * (x[j + 1] - x[j])
        end

        out[iq, α, β] = total
    end

    return out
end

function dynamic_static_sumrule_residual(dynamic::SpectrumResult, static::StaticStructureFactorResult)
    integrated = integrated_spectral_weight(dynamic)
    target = static.intensity
    length(integrated) == length(target) || throw(DimensionMismatch("q grids differ"))
    denom = max.(abs.(target), eps(Float64))
    return norm((integrated .- target) ./ denom) / sqrt(length(target))
end
