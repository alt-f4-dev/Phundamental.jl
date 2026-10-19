# General instrument response on energy grids and explicitly weighted momentum-energy samples.

"""Energy-dependent Gaussian instrument width. `sigma(true_energy)` must return a finite positive value in spectral-axis units."""
struct EnergyDependentGaussianResolution{F} <: AbstractResolution
    sigma::F
    nsigma::Float64
    function EnergyDependentGaussianResolution(sigma::F; nsigma::Real=5.0) where {F}
        isfinite(nsigma) && nsigma > 0 || throw(ArgumentError("nsigma must be finite and positive"))
        new{F}(sigma, Float64(nsigma))
    end
end

"""Piecewise-linear Gaussian widths tabulated against true energy, with no extrapolation beyond the knots."""
struct TabulatedGaussianResolution <: AbstractResolution
    energy::Vector{Float64}
    sigma::Vector{Float64}
    nsigma::Float64
    function TabulatedGaussianResolution(energy, sigma; nsigma::Real=5.0)
        x = Float64.(collect(energy))
        s = Float64.(collect(sigma))
        length(x) == length(s) && length(x) >= 2 || throw(DimensionMismatch("tabulated energy and width arrays must agree and contain at least two points"))
        _check_axis_grid(x)
        all(v -> isfinite(v) && v > 0, s) || throw(ArgumentError("tabulated Gaussian widths must be finite and positive"))
        isfinite(nsigma) && nsigma > 0 || throw(ArgumentError("nsigma must be finite and positive"))
        new(x, s, Float64(nsigma))
    end
end

"""Energy transfer kernel `density[observed, true]`, tabulated on the input energy grid. Quadrature is applied to the true-energy axis."""
struct TabulatedEnergyResolution <: AbstractResolution
    energy::Vector{Float64}
    density::Matrix{Float64}
    function TabulatedEnergyResolution(energy, density)
        x = Float64.(collect(energy))
        _check_axis_grid(x)
        k = Matrix{Float64}(density)
        size(k) == (length(x), length(x)) || throw(DimensionMismatch("tabulated kernel must have shape (observed energy, true energy)"))
        all(v -> isfinite(v) && v >= 0, k) || throw(ArgumentError("resolution density must be finite and nonnegative"))
        new(x, k)
    end
end

function _energy_quadrature_weights(x)
    x = Float64.(_check_axis_grid(x))
    length(x) >= 2 || throw(ArgumentError("energy quadrature requires at least two points"))
    widths = diff(x)
    w = Vector{Float64}(undef, length(x))
    w[1] = widths[1] / 2
    w[end] = widths[end] / 2
    for j in 2:length(x)-1
        w[j] = (widths[j-1] + widths[j]) / 2
    end
    return w
end

function _tabulated_width(R::TabulatedGaussianResolution, x::Real)
    R.energy[1] <= x <= R.energy[end] || throw(ArgumentError("requested true energy lies outside tabulated width range"))
    j = searchsortedlast(R.energy, x)
    j == length(R.energy) && return R.sigma[end]
    t = (x - R.energy[j]) / (R.energy[j+1] - R.energy[j])
    return (1 - t) * R.sigma[j] + t * R.sigma[j+1]
end

_resolution_width(R::EnergyDependentGaussianResolution, x) = R.sigma(x)
_resolution_width(R::TabulatedGaussianResolution, x) = _tabulated_width(R, x)

function _energy_resolution_matrix(R::Union{EnergyDependentGaussianResolution,TabulatedGaussianResolution,TabulatedEnergyResolution}, axis)
    x = Float64.(_check_axis_grid(axis))
    weights = _energy_quadrature_weights(x)
    if R isa TabulatedEnergyResolution
        length(x) == length(R.energy) && all(isapprox.(x, R.energy; atol=1e-12, rtol=1e-12)) ||
            throw(ArgumentError("tabulated resolution and spectrum energy axes must match"))
        return R.density .* reshape(weights, 1, :)
    end
    K = Matrix{Float64}(undef, length(x), length(x))
    for j in eachindex(x)
        sigma = _resolution_width(R, x[j])
        sigma isa Real && isfinite(sigma) && sigma > 0 || throw(ArgumentError("resolution width must be finite, real and positive"))
        for i in eachindex(x)
            delta = x[i] - x[j]
            K[i, j] = abs(delta) <= R.nsigma * sigma ? weights[j] * exp(-0.5 * (delta / sigma)^2) / (sqrt(2π) * sigma) : 0.0
        end
    end
    return K
end

function _general_energy_convolution(values::AbstractArray, axis, R::Union{EnergyDependentGaussianResolution,TabulatedGaussianResolution,TabulatedEnergyResolution})
    ndims(values) in (2, 4) || throw(ArgumentError("energy convolution supports rank-2 and rank-4 response arrays"))
    K = _energy_resolution_matrix(R, axis)
    size(values, 2) == size(K, 2) || throw(DimensionMismatch("energy axis length mismatch"))
    out = zeros(promote_type(eltype(values), Float64), size(values))
    if ndims(values) == 2
        for iq in axes(values, 1), i in axes(values, 2), j in axes(values, 2)
            out[iq, i] += K[i, j] * values[iq, j]
        end
    else
        for iq in axes(values, 1), i in axes(values, 2), a in axes(values, 3), b in axes(values, 4), j in axes(values, 2)
            out[iq, i, a, b] += K[i, j] * values[iq, j, a, b]
        end
    end
    return out
end

"""Covariance Gaussian Q–energy resolution. Q coordinates and quadrature weights are supplied explicitly; covariance uses `[Q..., energy]` ordering."""
struct GaussianQEResolution <: AbstractResolution
    covariance::Matrix{Float64}
    inverse_covariance::Matrix{Float64}
    normalization::Float64
    q_coordinates::Matrix{Float64}
    q_weights::Vector{Float64}
    coordinate_frame::Symbol
    momentum_unit::String
    energy_unit::String
    nsigma::Float64
end

function GaussianQEResolution(covariance, q_coordinates, q_weights; coordinate_frame::Symbol, momentum_unit::AbstractString,
                              energy_unit::AbstractString, nsigma::Real=5.0)
    q = Matrix{Float64}(q_coordinates)
    size(q, 2) >= 1 && size(q, 1) >= 1 || throw(ArgumentError("momentum coordinates must have shape (nQ, dimension)"))
    all(isfinite, q) || throw(ArgumentError("momentum coordinates must be finite"))
    w = Float64.(collect(q_weights))
    length(w) == size(q, 1) && all(v -> isfinite(v) && v > 0, w) || throw(ArgumentError("momentum quadrature weights must be finite, positive and have length nQ"))
    coordinate_frame in (:cartesian, :reciprocal, :dimensionless) || throw(ArgumentError("coordinate_frame must be :cartesian, :reciprocal or :dimensionless"))
    isempty(momentum_unit) && throw(ArgumentError("momentum_unit must be explicit"))
    isempty(energy_unit) && throw(ArgumentError("energy_unit must be explicit"))
    C = Matrix{Float64}(covariance)
    size(C) == (size(q, 2) + 1, size(q, 2) + 1) || throw(DimensionMismatch("resolution covariance must cover each Q dimension and energy"))
    all(isfinite, C) || throw(ArgumentError("resolution covariance must be finite"))
    isapprox(C, transpose(C); rtol=1e-10, atol=1e-12) || throw(ArgumentError("resolution covariance must be symmetric"))
    F = cholesky(Hermitian(C); check=true)
    isfinite(nsigma) && nsigma > 0 || throw(ArgumentError("nsigma must be finite and positive"))
    norm = inv((2π)^(size(C, 1) / 2) * prod(diag(F.U)))
    return GaussianQEResolution(C, inv(C), norm, q, w, coordinate_frame, String(momentum_unit), String(energy_unit), Float64(nsigma))
end

function _check_resolution_q(R::GaussianQEResolution, q)
    length(q) == size(R.q_coordinates, 1) || throw(DimensionMismatch("momentum grid and resolution coordinates differ in length"))
    for (i, entry) in enumerate(q)
        coordinates = if entry isa Real
            [Float64(entry)]
        elseif entry isa Tuple || entry isa AbstractVector
            Float64.(collect(entry))
        elseif entry isa CartesianWaveVector
            R.coordinate_frame === :cartesian || throw(ArgumentError("resolution frame does not match Cartesian wavevectors"))
            Float64.(entry.coordinates)
        elseif entry isa ReciprocalWaveVector
            R.coordinate_frame === :reciprocal || throw(ArgumentError("resolution frame does not match reciprocal wavevectors"))
            Float64.(entry.coordinates)
        else
            throw(ArgumentError("unrecognized wavevector type; the covariance resolution requires verifiable Q coordinates"))
        end
        length(coordinates) == size(R.q_coordinates, 2) || throw(DimensionMismatch("momentum coordinate dimension mismatch"))
        isapprox(coordinates, vec(R.q_coordinates[i, :]); rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("resolution coordinates disagree with the numeric momentum grid"))
    end
end

function _qe_convolution(values::AbstractMatrix, q, axis, R::GaussianQEResolution)
    x = Float64.(_check_axis_grid(axis))
    _check_resolution_q(R, q)
    size(values) == (size(R.q_coordinates, 1), length(x)) || throw(DimensionMismatch("Q–energy intensity grid mismatch"))
    energy_weights = _energy_quadrature_weights(x)
    out = zeros(promote_type(eltype(values), Float64), size(values))
    d = size(R.q_coordinates, 2)
    sigma = sqrt.(diag(R.covariance))
    for iq in axes(values, 1), jq in axes(values, 1)
        dq = view(R.q_coordinates, iq, :) .- view(R.q_coordinates, jq, :)
        any(abs(dq[k]) > R.nsigma * sigma[k] for k in 1:d) && continue
        for ix in axes(values, 2)
            lower = searchsortedfirst(x, x[ix] - R.nsigma * sigma[end])
            upper = searchsortedlast(x, x[ix] + R.nsigma * sigma[end])
            for jx in lower:upper
                dw = x[ix] - x[jx]
                z = [dq; dw]
                mahalanobis = dot(z, R.inverse_covariance * z)
                mahalanobis > R.nsigma^2 && continue
                out[iq, ix] += R.normalization * exp(-mahalanobis / 2) * R.q_weights[jq] * energy_weights[jx] * values[jq, jx]
            end
        end
    end
    return out
end

"""Sparse discrete Q–energy response; supplied entries are already cell-integrated weights, not probability densities."""
struct SparseQEResolution <: AbstractResolution
    nQ::Int
    nE::Int
    observed::Vector{Int}
    source::Vector{Int}
    weights::Vector{Float64}
    function SparseQEResolution(nQ::Integer, nE::Integer, observed, source, weights)
        nQ >= 1 && nE >= 2 || throw(ArgumentError("discrete Q–energy resolution requires nQ >= 1 and nE >= 2"))
        dest = Int.(collect(observed))
        src = Int.(collect(source))
        w = Float64.(collect(weights))
        length(dest) == length(src) == length(w) || throw(DimensionMismatch("sparse resolution triples must agree in length"))
        all(i -> 1 <= i <= nQ * nE, dest) && all(i -> 1 <= i <= nQ * nE, src) || throw(ArgumentError("resolution index outside flattened grid"))
        all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("sparse resolution weights must be nonnegative and finite"))
        new(Int(nQ), Int(nE), dest, src, w)
    end
end

function _qe_convolution(values::AbstractMatrix, q, axis, R::SparseQEResolution)
    size(values) == (R.nQ, R.nE) && length(q) == R.nQ && length(axis) == R.nE ||
        throw(DimensionMismatch("sparse resolution dimensions do not match response"))
    out = zeros(promote_type(eltype(values), Float64), size(values))
    for k in eachindex(R.weights)
        out[R.observed[k]] += R.weights[k] * values[R.source[k]]
    end
    return out
end

const _GeneralEnergyResolution = Union{EnergyDependentGaussianResolution,TabulatedGaussianResolution,TabulatedEnergyResolution}
const _GeneralQEResolution = Union{GaussianQEResolution,SparseQEResolution}

function _convolved_result(spectrum::SpectrumResult, values, R)
    metadata = _general_resolution_metadata(spectrum.metadata, R)
    return SpectrumResult(spectrum.q, spectrum.axis, values, metadata)
end

function _convolved_result(spectrum::LocalSpectrumResult, values, R)
    metadata = _general_resolution_metadata(spectrum.metadata, R)
    return LocalSpectrumResult(spectrum.sites, spectrum.axis, values, metadata)
end

function _convolved_result(spectrum::TensorSpectrumResult, values, R)
    metadata = _general_resolution_metadata(spectrum.metadata, R)
    return TensorSpectrumResult(spectrum.q, spectrum.axis, values, metadata)
end

function _convolved_result(spectrum::CompositeSpectrumResult, values, R)
    metadata = _general_resolution_metadata(spectrum.metadata, R)
    return CompositeSpectrumResult(spectrum.channels, spectrum.q, spectrum.axis, values, metadata)
end

function _convolved_result(spectrum::ScatteringResult, values, R)
    metadata = _general_resolution_metadata(spectrum.metadata, R)
    return ScatteringResult(spectrum.q, spectrum.axis, values, metadata)
end

function _general_resolution_metadata(original, R)
    metadata = copy(original)
    metadata[:resolution] = R
    metadata[:resolution_applied] = true
    metadata[:resolution_role] = :instrument_response
    metadata[:resolution_quadrature] = R isa SparseQEResolution ? :discrete_cell_weights : :trapezoid_true_grid
    return metadata
end

function convolve_resolution(R::_GeneralEnergyResolution,
                             spectrum::Union{SpectrumResult,LocalSpectrumResult,TensorSpectrumResult,CompositeSpectrumResult,ScatteringResult})
    values = _general_energy_convolution(spectrum.intensity, spectrum.axis, R)
    if eltype(spectrum.intensity) <: Real
        values = real.(values)
    end
    return _convolved_result(spectrum, values, R)
end

function convolve_resolution(R::_GeneralQEResolution, spectrum::Union{SpectrumResult,ScatteringResult})
    values = _qe_convolution(spectrum.intensity, spectrum.q, spectrum.axis, R)
    return _convolved_result(spectrum, values, R)
end

"""Instrument coverage; `true` denotes a valid observed bin, never a statement that the predicted intensity is zero elsewhere."""
struct InstrumentCoverage{A<:AbstractArray{Bool}}
    valid::A
    function InstrumentCoverage(valid::AbstractArray{Bool})
        new{typeof(copy(valid))}(copy(valid))
    end
end

"""Instrument response combining a numerical resolution kernel with an independently specified detector/kinematic coverage mask."""
struct CoveredResolution{R<:AbstractResolution,C<:InstrumentCoverage} <: AbstractResolution
    kernel::R
    coverage::C
end

function convolve_resolution(R::CoveredResolution, spectrum)
    result = convolve_resolution(R.kernel, spectrum)
    size(R.coverage.valid) == size(result.intensity) || throw(DimensionMismatch("instrument coverage mask and predicted intensity differ in shape"))
    result.metadata[:coverage_mask] = copy(R.coverage.valid)
    result.metadata[:resolution] = R
    return result
end
