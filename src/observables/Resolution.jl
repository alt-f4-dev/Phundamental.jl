# Instrument-resolution convolution. This is intentionally separate from the
# intrinsic/numerical line broadening used to represent delta functions.

struct GaussianResolution{T<:Real} <: AbstractResolution
    sigma::T
    nsigma::T
    function GaussianResolution(sigma::Real; nsigma::Real=5.0)
        sigma > 0 || throw(ArgumentError("resolution sigma must be positive"))
        nsigma > 0 || throw(ArgumentError("nsigma must be positive"))
        T = promote_type(typeof(float(sigma)), typeof(float(nsigma)))
        new{T}(T(sigma), T(nsigma))
    end
end

_resolution_kernel(R::GaussianResolution, Δ) =
    exp(-0.5 * (Δ / R.sigma)^2) / (sqrt(2π) * R.sigma)

function _uniform_step(x; rtol::Real=1e-8)
    length(x) >= 2 || throw(ArgumentError("resolution convolution requires at least two axis points"))
    d = diff(x)
    all(d .> 0) || throw(ArgumentError("axis must be strictly increasing"))
    dx = sum(d) / length(d)
    maximum(abs.(d .- dx)) <= rtol * max(abs(dx), eps(Float64)) ||
        throw(ArgumentError("optimized resolution convolution currently requires a uniform spectral grid"))
    return dx
end

function _resolution_stencil(axis, R::GaussianResolution)
    x = _check_axis_grid(axis)
    dx = _uniform_step(x)
    half = max(1, ceil(Int, R.nsigma * R.sigma / dx))
    offsets = -half:half
    # Multiplication by dx implements the quadrature measure in the
    # continuous convolution integral. No finite-window renormalization is
    # performed at the edges.
    kernel = ComplexF64[_resolution_kernel(R, k * dx) * dx for k in offsets]
    # Normalize the truncated stencil itself to unit area. This corrects only
    # the finite nsigma approximation; no per-output edge renormalization is
    # performed, so spectral weight can still leave a finite measurement window.
    kernel ./= sum(real.(kernel))
    return x, half, kernel
end

"""
Convolve a q-by-axis scalar spectrum with the Gaussian instrument response.
The discrete quadrature approximates

    I_obs(x_i) = integral dx' R(x_i-x') I_theory(x').

The kernel is *not* renormalized at finite-grid boundaries; response weight
outside the requested spectral window is lost from that window rather than
artificially folded back into it.
"""
function _convolve_axis2(values::AbstractMatrix, axis, R::GaussianResolution)
    x, half, kernel = _resolution_stencil(axis, R)
    size(values, 2) == length(x) || throw(DimensionMismatch("spectral axis length mismatch"))
    out = zeros(promote_type(eltype(values), ComplexF64), size(values))

    @inbounds for iq in axes(values, 1), ix in axes(values, 2)
        jlo = max(firstindex(x), ix - half)
        jhi = min(lastindex(x), ix + half)
        acc = zero(eltype(out))
        for j in jlo:jhi
            acc += kernel[(ix - j) + half + 1] * values[iq, j]
        end
        out[iq, ix] = acc
    end
    return out
end

"""Optimized convolution for q-by-axis-by-3-by-3 tensor spectra."""
function _convolve_axis2(values::AbstractArray{T,4}, axis, R::GaussianResolution) where {T}
    x, half, kernel = _resolution_stencil(axis, R)
    size(values, 2) == length(x) || throw(DimensionMismatch("spectral axis length mismatch"))
    out = zeros(promote_type(T, ComplexF64), size(values))

    @inbounds for iq in axes(values, 1), ix in axes(values, 2), α in axes(values, 3), β in axes(values, 4)
        jlo = max(firstindex(x), ix - half)
        jhi = min(lastindex(x), ix + half)
        acc = zero(eltype(out))
        for j in jlo:jhi
            acc += kernel[(ix - j) + half + 1] * values[iq, j, α, β]
        end
        out[iq, ix, α, β] = acc
    end
    return out
end

function convolve_resolution(R::GaussianResolution, spectrum::SpectrumResult)
    values = _convolve_axis2(spectrum.intensity, spectrum.axis, R)
    metadata = copy(spectrum.metadata)
    metadata[:resolution] = R
    metadata[:resolution_applied] = true
    metadata[:resolution_role] = :instrument_response
    return SpectrumResult(spectrum.q, spectrum.axis, values, metadata)
end

function convolve_resolution(R::GaussianResolution, spectrum::TensorSpectrumResult)
    values = _convolve_axis2(spectrum.intensity, spectrum.axis, R)
    metadata = copy(spectrum.metadata)
    metadata[:resolution] = R
    metadata[:resolution_applied] = true
    metadata[:resolution_role] = :instrument_response
    return TensorSpectrumResult(spectrum.q, spectrum.axis, values, metadata)
end

function convolve_resolution(R::GaussianResolution, spectrum::ScatteringResult)
    values = _convolve_axis2(spectrum.intensity, spectrum.axis, R)
    metadata = copy(spectrum.metadata)
    metadata[:resolution] = R
    metadata[:resolution_applied] = true
    metadata[:resolution_role] = :instrument_response
    # Probe intensity is physically real. Preserve a real array when numerical
    # convolution introduced only zero/tiny imaginary storage.
    if eltype(spectrum.intensity) <: Real
        values = real.(values)
    end
    return ScatteringResult(spectrum.q, spectrum.axis, values, metadata)
end
