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

# The immutable point index belongs to the resolution grid and is shared by repeated convolutions.
struct _QEPointNode
    point::Int
    axis::Int
    left::Union{Nothing,_QEPointNode}
    right::Union{Nothing,_QEPointNode}
end

function _qe_spatial_index(q::Matrix{Float64})
    points = collect(axes(q, 1))
    d = size(q, 2)
    function build!(lo::Int, hi::Int, depth::Int)
        lo > hi && return nothing
        axis = mod1(depth, d)
        sort!(view(points, lo:hi); by=i -> q[i, axis])
        middle = (lo + hi) ÷ 2
        index = points[middle]
        return _QEPointNode(index, axis, build!(lo, middle - 1, depth + 1), build!(middle + 1, hi, depth + 1))
    end
    return build!(1, length(points), 1)
end

function _qe_visit_box(visit, node::Union{Nothing,_QEPointNode}, q, center, radius)
    isnothing(node) && return nothing
    axis = node.axis
    point = node.point
    coordinate = q[point, axis]
    inside = true
    @inbounds for k in eachindex(radius)
        if abs(q[point, k] - center[k]) > radius[k]
            inside = false
            break
        end
    end
    inside && visit(point)
    coordinate >= center[axis] - radius[axis] && _qe_visit_box(visit, node.left, q, center, radius)
    coordinate <= center[axis] + radius[axis] && _qe_visit_box(visit, node.right, q, center, radius)
    return nothing
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
    spatial_index::Union{Nothing,_QEPointNode}
end

# Retain the previous positional field constructor for callers that already supply a certified precision matrix.
function GaussianQEResolution(C::Matrix{Float64}, P::Matrix{Float64}, norm::Float64, q::Matrix{Float64},
                              w::Vector{Float64}, frame::Symbol, momentum_unit::String, energy_unit::String, nsigma::Float64)
    return GaussianQEResolution(C, P, norm, q, w, frame, momentum_unit, energy_unit, nsigma, _qe_spatial_index(q))
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
    return GaussianQEResolution(C, inv(C), norm, q, w, coordinate_frame, String(momentum_unit), String(energy_unit), Float64(nsigma), _qe_spatial_index(q))
end

"""Source-dependent Gaussian Q–E response. Set `storage=:lazy` to compute covariance factorization only for nonzero source bins.

A callable covariance provider uses `(q_vector, true_energy) -> covariance_matrix`. Its lazy form stores no source-bin covariance or inverse arrays. Array-backed lazy storage retains the supplied covariance but omits precision and normalization arrays.

Callable providers are evaluated and copied under a lock so they may safely reuse mutable internal buffers. In threaded convolution, the covariance and precision matrices are prepared in bounded batches before any worker sees them. Each worker reads immutable batch data and buffers its output contributions, which are merged under a lock.
"""
struct SpatialGaussianQEResolution{F} <: AbstractResolution
    covariance::Union{Nothing,Array{Float64,4}}
    inverse_covariance::Union{Nothing,Array{Float64,4}}
    normalization::Union{Nothing,Matrix{Float64}}
    covariance_function::F
    q_coordinates::Matrix{Float64}
    energy::Vector{Float64}
    q_weights::Vector{Float64}
    coordinate_frame::Symbol
    momentum_unit::String
    energy_unit::String
    nsigma::Float64
    spatial_index::Union{Nothing,_QEPointNode}
end

# Compatibility with the original positional storage constructor; normal use should prefer keyword constructors.
function SpatialGaussianQEResolution(covariance::Union{Nothing,Array{Float64,4}},
                                     inverse_covariance::Union{Nothing,Array{Float64,4}},
                                     normalization::Union{Nothing,Matrix{Float64}}, covariance_function::F,
                                     q::Matrix{Float64}, energy::Vector{Float64}, weights::Vector{Float64},
                                     frame::Symbol, momentum_unit::String, energy_unit::String, nsigma::Float64) where {F}
    return SpatialGaussianQEResolution(covariance, inverse_covariance, normalization, covariance_function,
                                       q, energy, weights, frame, momentum_unit, energy_unit, nsigma, _qe_spatial_index(q))
end

function _spatial_resolution_grid(q_coordinates, energy, q_weights; coordinate_frame, momentum_unit, energy_unit, nsigma)
    q = Matrix{Float64}(q_coordinates)
    size(q, 1) >= 1 && size(q, 2) >= 1 && all(isfinite, q) ||
        throw(ArgumentError("momentum coordinates must be nonempty, finite, and have shape (nQ, dimension)"))
    x = Float64.(_check_axis_grid(energy))
    w = Float64.(collect(q_weights))
    length(w) == size(q, 1) && all(v -> isfinite(v) && v > 0, w) ||
        throw(ArgumentError("momentum quadrature weights must be finite, positive, and have length nQ"))
    coordinate_frame in (:cartesian, :reciprocal, :dimensionless) || throw(ArgumentError("invalid coordinate frame"))
    isempty(momentum_unit) && throw(ArgumentError("momentum_unit must be explicit"))
    isempty(energy_unit) && throw(ArgumentError("energy_unit must be explicit"))
    isfinite(nsigma) && nsigma > 0 || throw(ArgumentError("nsigma must be finite and positive"))
    return q, x, w
end

function _qe_source_factorization(C, d::Int)
    size(C) == (d, d) || throw(DimensionMismatch("source covariance matrix dimension differs from Q–E coordinates"))
    all(isfinite, C) && isapprox(C, transpose(C); rtol=1e-10, atol=1e-12) ||
        throw(ArgumentError("source covariance must be finite and symmetric"))
    F = cholesky(Hermitian(C); check=true)
    return inv(F), inv((2π)^(d / 2) * prod(diag(F.U)))
end

function SpatialGaussianQEResolution(covariances::AbstractArray{<:Real,4}, q_coordinates, energy, q_weights;
                                     coordinate_frame::Symbol, momentum_unit::AbstractString, energy_unit::AbstractString,
                                     nsigma::Real=5.0, storage::Symbol=:dense)
    storage in (:dense, :lazy) || throw(ArgumentError("storage must be :dense or :lazy"))
    q, x, w = _spatial_resolution_grid(q_coordinates, energy, q_weights; coordinate_frame, momentum_unit, energy_unit, nsigma)
    d = size(q, 2) + 1
    size(covariances) == (d, d, size(q, 1), length(x)) ||
        throw(DimensionMismatch("covariance array must have shape (dimension+1, dimension+1, nQ, nE)"))
    cov = storage === :lazy && covariances isa Array{Float64,4} ? covariances : Float64.(Array(covariances))
    precision = storage === :dense ? similar(cov) : nothing
    normalization = storage === :dense ? Matrix{Float64}(undef, size(q, 1), length(x)) : nothing
    if storage === :dense
        for ix in eachindex(x), iq in axes(q, 1)
            P, amplitude = _qe_source_factorization(view(cov, :, :, iq, ix), d)
            precision[:, :, iq, ix] .= P
            normalization[iq, ix] = amplitude
        end
    end
    return SpatialGaussianQEResolution(cov, precision, normalization, nothing, q, x, w, coordinate_frame,
                                       String(momentum_unit), String(energy_unit), Float64(nsigma), _qe_spatial_index(q))
end

function SpatialGaussianQEResolution(covariance_function::Function, q_coordinates, energy, q_weights;
                                     coordinate_frame::Symbol, momentum_unit::AbstractString, energy_unit::AbstractString,
                                     nsigma::Real=5.0, storage::Symbol=:lazy)
    storage in (:lazy, :dense) || throw(ArgumentError("storage must be :lazy or :dense"))
    q, x, w = _spatial_resolution_grid(q_coordinates, energy, q_weights; coordinate_frame, momentum_unit, energy_unit, nsigma)
    if storage === :dense
        d = size(q, 2) + 1
        cov = Array{Float64}(undef, d, d, size(q, 1), length(x))
        for ix in eachindex(x), iq in axes(q, 1)
            C = Matrix{Float64}(covariance_function(view(q, iq, :), x[ix]))
            size(C) == (d, d) || throw(DimensionMismatch("covariance function returned a matrix with incorrect shape"))
            cov[:, :, iq, ix] .= C
        end
        return SpatialGaussianQEResolution(cov, q, x, w; coordinate_frame, momentum_unit, energy_unit, nsigma, storage)
    end
    return SpatialGaussianQEResolution(nothing, nothing, nothing, covariance_function, q, x, w, coordinate_frame,
                                       String(momentum_unit), String(energy_unit), Float64(nsigma), _qe_spatial_index(q))
end

function _qe_source_covariance(R::SpatialGaussianQEResolution, jq::Int, jx::Int)
    C = isnothing(R.covariance) ? Matrix{Float64}(R.covariance_function(view(R.q_coordinates, jq, :), R.energy[jx])) :
                                 view(R.covariance, :, :, jq, jx)
    if isnothing(R.inverse_covariance)
        P, amplitude = _qe_source_factorization(C, size(R.q_coordinates, 2) + 1)
        return C, P, amplitude
    end
    return C, view(R.inverse_covariance, :, :, jq, jx), R.normalization[jq, jx]
end

const _GaussianQEResolution = Union{GaussianQEResolution,SpatialGaussianQEResolution}

function _check_resolution_q(R::_GaussianQEResolution, q)
    length(q) == size(R.q_coordinates, 1) || throw(DimensionMismatch("momentum grid and resolution coordinates differ in length"))
    for (i, entry) in enumerate(q)
        coordinates = if entry isa Real
            [Float64(entry)]
        elseif entry isa AbstractVector
            entry
        elseif entry isa Tuple
            collect(entry)
        elseif entry isa CartesianWaveVector
            R.coordinate_frame === :cartesian || throw(ArgumentError("resolution frame does not match Cartesian wavevectors"))
            entry.coordinates
        elseif entry isa ReciprocalWaveVector
            R.coordinate_frame === :reciprocal || throw(ArgumentError("resolution frame does not match reciprocal wavevectors"))
            entry.coordinates
        else
            throw(ArgumentError("unrecognized wavevector type; the covariance resolution requires verifiable Q coordinates"))
        end
        length(coordinates) == size(R.q_coordinates, 2) || throw(DimensionMismatch("momentum coordinate dimension mismatch"))
        isapprox(coordinates, view(R.q_coordinates, i, :); rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("resolution coordinates disagree with the numeric momentum grid"))
    end
end

# The source-dependent path owns independent workspace arrays for each work partition.
# A provider may return a borrowed mutable buffer, so evaluation and copying are one critical section.
# Factorization and scattering occur outside this lock using partition-owned matrices.
const _qe_callable_covariance_lock = ReentrantLock()

# The small-matrix Cholesky, triangular inverse, and precision product run entirely in place.
struct _QECovarianceWorkspace
    covariance::Matrix{Float64}
    factored::Matrix{Float64}
    precision::Matrix{Float64}
    scratch::Matrix{Float64}
end

function _qe_workspace(dimension::Int)
    return _QECovarianceWorkspace(zeros(dimension, dimension), zeros(dimension, dimension),
                                  zeros(dimension, dimension), zeros(dimension, dimension))
end

# Form C = L*L' and C^-1 = L^-T*L^-1 without allocating matrices or calling a shared LAPACK workspace.
# For realistic Q–E resolutions dimension is three or four; this path also handles higher dimensions.
function _qe_factorize_source!(work::_QECovarianceWorkspace)
    C = work.covariance
    L = work.factored
    inverse_lower = work.precision
    precision = work.scratch
    dimension = size(C, 1)
    diagonal_product = 1.0
    @inbounds for i in 1:dimension
        for j in 1:i-1
            entry = C[i, j]
            for k in 1:j-1
                entry -= L[i, k] * L[j, k]
            end
            L[i, j] = entry / L[j, j]
        end
        pivot = C[i, i]
        for k in 1:i-1
            pivot -= L[i, k]^2
        end
        isfinite(pivot) && pivot > 0 || throw(LinearAlgebra.PosDefException(i))
        L[i, i] = sqrt(pivot)
        diagonal_product *= L[i, i]
    end
    @inbounds for column in 1:dimension, row in 1:dimension
        entry = row == column ? 1.0 : 0.0
        for k in 1:row-1
            entry -= L[row, k] * inverse_lower[k, column]
        end
        inverse_lower[row, column] = entry / L[row, row]
    end
    @inbounds for column in 1:dimension, row in 1:dimension
        entry = 0.0
        for k in max(row, column):dimension
            entry += inverse_lower[k, row] * inverse_lower[k, column]
        end
        precision[row, column] = entry
    end
    copyto!(inverse_lower, precision)
    amplitude = inv((2π)^(dimension / 2) * diagonal_product)
    isfinite(amplitude) || throw(ArgumentError("source covariance has a nonfinite Gaussian normalization"))
    return C, inverse_lower, amplitude
end

function _qe_source_properties!(::Nothing, R::GaussianQEResolution, jq::Int, jx::Int)
    return R.covariance, R.inverse_covariance, R.normalization
end

function _qe_source_properties!(work::_QECovarianceWorkspace, R::SpatialGaussianQEResolution, jq::Int, jx::Int)
    if !isnothing(R.inverse_covariance)
        return view(R.covariance, :, :, jq, jx), view(R.inverse_covariance, :, :, jq, jx), R.normalization[jq, jx]
    end
    if isnothing(R.covariance)
        lock(_qe_callable_covariance_lock)
        try
            source = R.covariance_function(view(R.q_coordinates, jq, :), R.energy[jx])
            size(source) == size(work.covariance) ||
                throw(DimensionMismatch("source covariance at Q index $jq and energy index $jx has an incorrect shape"))
            copyto!(work.covariance, source)
        finally
            unlock(_qe_callable_covariance_lock)
        end
    else
        source = view(R.covariance, :, :, jq, jx)
        copyto!(work.covariance, source)
    end
    C = work.covariance
    dimension = size(C, 1)
    @inbounds for b in 1:dimension, a in 1:dimension
        isfinite(C[a, b]) || throw(ArgumentError("source covariance at Q index $jq and energy index $jx must be finite"))
        isapprox(C[a, b], C[b, a]; rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("source covariance at Q index $jq and energy index $jx must be symmetric"))
    end
    try
        return _qe_factorize_source!(work)
    catch error
        if error isa LinearAlgebra.PosDefException
            throw(ArgumentError("source covariance at Q index $jq and energy index $jx is not positive definite (pivot $(error.info))"))
        end
        rethrow()
    end
end

# Accumulate a bounded number of source contributions privately before merging them under one lock.
# No thread reads or writes shared output during Gaussian kernel evaluation.
const _QE_UPDATE_BUFFER_LIMIT = 16_384

struct _QEUpdateBuffer{T}
    entries::Vector{Tuple{Int,Int,T}}
    target::Matrix{T}
    merge_lock::ReentrantLock
end

function _qe_flush_updates!(buffer::_QEUpdateBuffer)
    isempty(buffer.entries) && return nothing
    lock(buffer.merge_lock)
    try
        for (iq, ix, contribution) in buffer.entries
            @inbounds buffer.target[iq, ix] += contribution
        end
    finally
        unlock(buffer.merge_lock)
    end
    empty!(buffer.entries)
    return nothing
end

@inline function _qe_record_update!(out::AbstractMatrix, ::Nothing, iq::Int, ix::Int, contribution)
    @inbounds out[iq, ix] += contribution
    return nothing
end

@inline function _qe_record_update!(out::AbstractMatrix, buffer::_QEUpdateBuffer, iq::Int, ix::Int, contribution)
    push!(buffer.entries, (iq, ix, contribution))
    length(buffer.entries) >= _QE_UPDATE_BUFFER_LIMIT && _qe_flush_updates!(buffer)
    return nothing
end

# Exact spatial Mahalanobis support, followed by solving the correlated energy quadratic.
# Restricting the energy roots cannot alter the integral; it only excludes points outside the nsigma ellipsoid.
# Read-only source properties are never modified while their Gaussian contributions are evaluated.
# The serial path obtains properties from its private workspace; the threaded path prepares them
# before scheduling workers. This separates all mutable factorization from parallel scattering.
function _qe_scatter_prepared_source!(out::AbstractMatrix, R::_GaussianQEResolution, index::_QEPointNode, x::Vector{Float64},
                                      weights::Vector{Float64}, values::AbstractMatrix, jq::Int, jx::Int,
                                      C::AbstractMatrix, P::AbstractMatrix, amplitude::Float64, updates)
    source_value = values[jq, jx]
    iszero(source_value) && return nothing
    d = size(R.q_coordinates, 2)
    sigma_e = sqrt(C[d+1, d+1])
    lower_e = searchsortedfirst(x, x[jx] - R.nsigma * sigma_e)
    upper_e = searchsortedlast(x, x[jx] + R.nsigma * sigma_e)
    lower_e > upper_e && return nothing
    radius = ntuple(k -> R.nsigma * sqrt(C[k, k]), d)
    center = ntuple(k -> R.q_coordinates[jq, k], d)
    factor = amplitude * R.q_weights[jq] * weights[jx] * source_value
    cutoff2 = R.nsigma^2
    a = P[d+1, d+1]
    _qe_visit_box(index, R.q_coordinates, center, radius) do iq
        qq = 0.0
        cross = 0.0
        @inbounds for u in 1:d
            du = R.q_coordinates[iq, u] - center[u]
            cross += du * P[u, d+1]
            for v in 1:d
                qq += du * P[u, v] * (R.q_coordinates[iq, v] - center[v])
            end
        end
        discriminant = cross^2 + a * (cutoff2 - qq)
        if discriminant < 0.0
            discriminant < -64eps(Float64) * (cross^2 + abs(a * (cutoff2 - qq)) + cutoff2) && return nothing
            # Keep a boundary pixel when cancellation makes the discriminant slightly negative.
            lower, upper = lower_e, upper_e
        else
            width = sqrt(discriminant)
            lower_bound = x[jx] + (-cross - width) / a
            upper_bound = x[jx] + (-cross + width) / a
            lower = max(lower_e, searchsortedfirst(x, prevfloat(lower_bound)))
            upper = min(upper_e, searchsortedlast(x, nextfloat(upper_bound)))
        end
        lower > upper && return nothing
        @inbounds for ix in lower:upper
            dw = x[ix] - x[jx]
            distance2 = qq + 2 * cross * dw + a * dw^2
            distance2 > cutoff2 && continue
            contribution = factor * exp(-0.5 * max(distance2, 0.0))
            _qe_record_update!(out, updates, iq, ix, contribution)
        end
        return nothing
    end
    return nothing
end

function _qe_scatter_source!(out::AbstractMatrix, R::_GaussianQEResolution, index::_QEPointNode, x::Vector{Float64},
                             weights::Vector{Float64}, values::AbstractMatrix, jq::Int, jx::Int, work, updates)
    iszero(values[jq, jx]) && return nothing
    C, P, amplitude = _qe_source_properties!(work, R, jq, jx)
    return _qe_scatter_prepared_source!(out, R, index, x, weights, values, jq, jx, C, P, amplitude, updates)
end

# This bound applies to INPUT bins, not momentum points. Covariance and precision buffers
# therefore remain O(batch_size * (dimension+1)^2), even for dense TOF spectra.
const _QE_THREAD_SOURCE_BATCH = 16_384

function _qe_threaded_convolution!(out::Matrix, values::AbstractMatrix, x::Vector{Float64},
                                   energy_weights::Vector{Float64}, R::_GaussianQEResolution, index::_QEPointNode)
    nQ, nE = size(values)
    nworkers = min(Threads.nthreads(), nQ)
    nworkers <= 1 && throw(ArgumentError("threaded Q–E convolution requires more than one worker"))
    d = size(R.q_coordinates, 2) + 1
    batch_capacity = min(_QE_THREAD_SOURCE_BATCH, length(values))
    covariance_batch = Array{Float64}(undef, d, d, batch_capacity)
    precision_batch = similar(covariance_batch)
    amplitude_batch = Vector{Float64}(undef, batch_capacity)
    index_batch = Vector{Int}(undef, batch_capacity)
    preparation_workspace = R isa SpatialGaussianQEResolution ? _qe_workspace(d) : nothing
    merge_lock = ReentrantLock()
    buffers = [_QEUpdateBuffer(Tuple{Int,Int,eltype(out)}[], out, merge_lock) for _ in 1:nworkers]

    for first_bin in 1:_QE_THREAD_SOURCE_BATCH:length(values)
        last_bin = min(first_bin + _QE_THREAD_SOURCE_BATCH - 1, length(values))
        active = 0
        # All mutable covariance evaluation/factorization happens before workers start.
        # Copying is required: lazy properties point into a reusable factorization workspace.
        for linear in first_bin:last_bin
            iszero(values[linear]) && continue
            jq = mod1(linear, nQ)
            jx = cld(linear, nQ)
            C, P, amplitude = _qe_source_properties!(preparation_workspace, R, jq, jx)
            active += 1
            copyto!(view(covariance_batch, :, :, active), C)
            copyto!(view(precision_batch, :, :, active), P)
            amplitude_batch[active] = amplitude
            index_batch[active] = linear
        end
        active == 0 && continue

        # The batch is frozen until the barrier at the end of @threads. No worker ever
        # accesses mutable preparation_workspace or writes directly into the output.
        Threads.@threads :static for worker in 1:nworkers
            updates = buffers[worker]
            for slot in worker:nworkers:active
                linear = index_batch[slot]
                jq = mod1(linear, nQ)
                jx = cld(linear, nQ)
                C = view(covariance_batch, :, :, slot)
                P = view(precision_batch, :, :, slot)
                _qe_scatter_prepared_source!(out, R, index, x, energy_weights, values,
                                             jq, jx, C, P, amplitude_batch[slot], updates)
                _qe_flush_updates!(updates)
            end
        end
    end
    return out
end

function _qe_convolution(values::AbstractMatrix, q, axis, R::_GaussianQEResolution; threaded::Bool=false)
    x = Float64.(_check_axis_grid(axis))
    _check_resolution_q(R, q)
    size(values) == (size(R.q_coordinates, 1), length(x)) || throw(DimensionMismatch("Q–energy intensity grid mismatch"))
    if R isa SpatialGaussianQEResolution
        length(x) == length(R.energy) && all(isapprox.(x, R.energy; rtol=1e-12, atol=1e-12)) ||
            throw(ArgumentError("spatial resolution and spectrum true-energy axes must agree"))
    end
    out = zeros(promote_type(eltype(values), Float64), size(values))
    energy_weights = _energy_quadrature_weights(x)
    index = R.spatial_index
    isnothing(index) && return out
    if threaded && Threads.nthreads() > 1
        _qe_threaded_convolution!(out, values, x, energy_weights, R, index)
    else
        work = R isa SpatialGaussianQEResolution ? _qe_workspace(size(R.q_coordinates, 2) + 1) : nothing
        for jq in axes(values, 1), jx in axes(values, 2)
            _qe_scatter_source!(out, R, index, x, energy_weights, values, jq, jx, work, nothing)
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
const _GeneralQEResolution = Union{GaussianQEResolution,SpatialGaussianQEResolution,SparseQEResolution}

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

function convolve_resolution(R::_GeneralQEResolution, spectrum::Union{SpectrumResult,ScatteringResult}; threaded::Bool=false)
    values = R isa SparseQEResolution ? _qe_convolution(spectrum.intensity, spectrum.q, spectrum.axis, R) :
             _qe_convolution(spectrum.intensity, spectrum.q, spectrum.axis, R; threaded)
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
