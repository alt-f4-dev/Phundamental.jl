# Versioned, explicit scientific numerical exchange for model inference and measurement comparisons.

using LinearAlgebra: Hermitian, cholesky, eigvals, diag

const EXPERIMENTAL_DATA_SCHEMA_VERSION = 1

"""Explicit momentum/site and spectral coordinate contract. Numerical momentum rows are indexed `(iQ, component)`."""
struct ExperimentalGrid
    positions::Matrix{Float64}
    spectral::Vector{Float64}
    coordinate_role::Symbol
    coordinate_frame::Symbol
    coordinate_unit::String
    spectral_kind::Symbol
    spectral_unit::String
    function ExperimentalGrid(positions, spectral; coordinate_role::Symbol=:momentum, coordinate_frame::Symbol,
                              coordinate_unit::AbstractString, spectral_kind::Symbol=:energy, spectral_unit::AbstractString)
        q = Matrix{Float64}(positions)
        x = Float64.(collect(spectral))
        size(q, 1) >= 1 && size(q, 2) >= 1 || throw(ArgumentError("coordinates require at least one sample and component"))
        all(isfinite, q) || throw(ArgumentError("coordinates must be finite"))
        length(x) >= 2 && all(isfinite, x) && all(diff(x) .> 0) || throw(ArgumentError("spectral axis must be finite, strictly increasing and contain at least two points"))
        coordinate_role in (:momentum, :site) || throw(ArgumentError("coordinate_role must be :momentum or :site"))
        coordinate_frame in (:cartesian, :reciprocal, :dimensionless, :site) || throw(ArgumentError("invalid coordinate frame"))
        coordinate_role === :site && coordinate_frame !== :site && throw(ArgumentError("site coordinates require :site frame"))
        coordinate_role === :momentum && coordinate_frame === :site && throw(ArgumentError("momentum coordinates cannot use :site frame"))
        spectral_kind in (:energy, :omega) || throw(ArgumentError("spectral_kind must be :energy or :omega"))
        isempty(coordinate_unit) && throw(ArgumentError("coordinate unit must be explicit"))
        isempty(spectral_unit) && throw(ArgumentError("spectral unit must be explicit"))
        new(q, x, coordinate_role, coordinate_frame, String(coordinate_unit), spectral_kind, String(spectral_unit))
    end
end

"""Neutral prediction retaining original Julia column-major intensity dimensions and explicit physical axes."""
struct PortablePrediction{T,N}
    grid::ExperimentalGrid
    intensity::Array{T,N}
    intensity_unit::String
    normalization::String
    stage::Symbol
    response::Dict{String,Any}
    provenance::Dict{String,Any}
    coverage::BitArray{N}
    uncertainty::Union{Nothing,Array{Float64,N}}
    covariance::Union{Nothing,Matrix{Float64}}
end

"""Measured values on the same explicit grid contract; `valid=true` identifies measured detector bins."""
struct ExperimentalDataset{N}
    grid::ExperimentalGrid
    intensity::Array{Float64,N}
    uncertainty::Union{Nothing,Array{Float64,N}}
    covariance::Union{Nothing,Matrix{Float64}}
    valid::BitArray{N}
    intensity_unit::String
    normalization::String
    provenance::Dict{String,Any}
    likelihood::Symbol
end

# Preserve the original positional measurement constructor for callers with pre-F5 datasets.
ExperimentalDataset(grid::ExperimentalGrid, intensity::Array{Float64,N}, uncertainty::Union{Nothing,Array{Float64,N}},
                    covariance::Union{Nothing,Matrix{Float64}}, valid::BitArray{N}, intensity_unit::String,
                    normalization::String, provenance::Dict{String,Any}) where {N} =
    ExperimentalDataset(grid, intensity, uncertainty, covariance, valid, intensity_unit, normalization, provenance, :gaussian)

"""Measurement comparison diagnostics using only valid bins and either diagonal uncertainties or a full covariance."""
struct MeasurementComparison
    chi_squared::Float64
    nvalid::Int
    residual::Vector{Float64}
    whitened_residual::Vector{Float64}
    likelihood::Symbol
    deviance::Float64
    log_likelihood::Float64
end

MeasurementComparison(chi_squared::Real, nvalid::Int, residual::Vector{Float64}, whitened::Vector{Float64}) =
    MeasurementComparison(Float64(chi_squared), nvalid, residual, whitened, :gaussian, Float64(chi_squared), NaN)

function _portable_metadata(value)
    if value isa Symbol || value isa AbstractString
        return string(value)
    elseif value isa Bool || value === nothing
        return value
    elseif value isa Real
        isfinite(value) || throw(ArgumentError("portable metadata must not contain nonfinite numbers"))
        return value
    elseif value isa Tuple || value isa AbstractVector
        return [_portable_metadata(item) for item in value]
    elseif value isa NamedTuple || value isa AbstractDict
        return Dict{String,Any}(string(key) => _portable_metadata(item) for (key, item) in pairs(value))
    end
    throw(ArgumentError("metadata of type $(typeof(value)) cannot be represented without losing meaning; provide a neutral provenance record"))
end

function _portable_grid(grid::ExperimentalGrid)
    return Dict{String,Any}(
        "coordinates" => [Float64.(collect(row)) for row in eachrow(grid.positions)],
        "spectral" => copy(grid.spectral), "coordinate_role" => string(grid.coordinate_role),
        "coordinate_frame" => string(grid.coordinate_frame), "coordinate_unit" => grid.coordinate_unit,
        "spectral_kind" => string(grid.spectral_kind), "spectral_unit" => grid.spectral_unit,
    )
end

function _import_grid(record)
    rows = record["coordinates"]
    isempty(rows) && throw(ArgumentError("grid contains no coordinate rows"))
    dimension = length(first(rows))
    all(row -> length(row) == dimension, rows) || throw(DimensionMismatch("momentum coordinate row lengths disagree"))
    positions = Matrix{Float64}(undef, length(rows), dimension)
    for i in eachindex(rows)
        positions[i, :] .= rows[i]
    end
    return ExperimentalGrid(positions, record["spectral"]; coordinate_role=Symbol(record["coordinate_role"]),
                            coordinate_frame=Symbol(record["coordinate_frame"]), coordinate_unit=record["coordinate_unit"],
                            spectral_kind=Symbol(record["spectral_kind"]), spectral_unit=record["spectral_unit"])
end

function _portable_values(values; mask=nothing)
    isbitstype(eltype(values)) || throw(ArgumentError("export requires dense numeric intensity arrays"))
    eltype(values) <: Number || throw(ArgumentError("intensity arrays must be numerical"))
    isnothing(mask) || size(mask) == size(values) || throw(DimensionMismatch("coverage and intensity shapes differ"))
    real_values = Vector{Union{Nothing,Float64}}(undef, length(values))
    imag_values = eltype(values) <: Complex ? similar(real_values) : nothing
    for (i, value) in enumerate(values)
        valid = isnothing(mask) || mask[i]
        rv = real(value)
        iv = imag(value)
        valid && (!isfinite(rv) || !isfinite(iv)) && throw(ArgumentError("unmasked intensity contains nonfinite values"))
        real_values[i] = valid ? Float64(rv) : nothing
        !isnothing(imag_values) && (imag_values[i] = valid ? Float64(iv) : nothing)
    end
    return Dict{String,Any}("shape" => collect(size(values)), "order" => "column_major",
                            "real" => real_values, "imag" => imag_values)
end

function _import_values(record, valid)
    record["order"] == "column_major" || throw(ArgumentError("only Julia column-major arrays are supported"))
    shape = Tuple(Int.(record["shape"]))
    length(shape) in (2, 4) || throw(ArgumentError("only rank-2 and rank-4 experimental arrays are supported"))
    prod(shape) == length(record["real"]) == length(valid) || throw(DimensionMismatch("encoded intensity length mismatch"))
    has_imag = !isnothing(record["imag"])
    has_imag && length(record["imag"]) != prod(shape) && throw(DimensionMismatch("encoded imaginary array length mismatch"))
    values = has_imag ? Vector{ComplexF64}(undef, prod(shape)) : Vector{Float64}(undef, prod(shape))
    for i in eachindex(values)
        if valid[i]
            rv = record["real"][i]
            rv isa Real && isfinite(rv) || throw(ArgumentError("valid intensity entry is missing or nonfinite"))
            iv = has_imag ? record["imag"][i] : 0.0
            iv isa Real && isfinite(iv) || throw(ArgumentError("valid imaginary intensity entry is missing or nonfinite"))
            values[i] = has_imag ? ComplexF64(rv, iv) : Float64(rv)
        else
            values[i] = has_imag ? ComplexF64(NaN, NaN) : NaN
        end
    end
    return reshape(values, shape)
end

function _portable_prediction(grid::ExperimentalGrid, values; intensity_unit, normalization, stage=:intrinsic,
                              response=Dict{String,Any}(), provenance=Dict{String,Any}(), coverage=trues(size(values)),
                              uncertainty=nothing, covariance=nothing)
    values isa AbstractArray && ndims(values) in (2, 4) || throw(ArgumentError("prediction must provide a rank-2 or rank-4 intensity array"))
    size(values, 1) == size(grid.positions, 1) && size(values, 2) == length(grid.spectral) || throw(DimensionMismatch("prediction axes and intensity disagree"))
    isempty(intensity_unit) && throw(ArgumentError("intensity unit must be explicit"))
    isempty(normalization) && throw(ArgumentError("normalization must be explicit"))
    size(coverage) == size(values) || throw(DimensionMismatch("prediction coverage shape mismatch"))
    valid = BitArray(coverage)
    _portable_values(values; mask=valid)
    stderr = if isnothing(uncertainty)
        nothing
    else
        size(uncertainty) == size(values) || throw(DimensionMismatch("prediction uncertainty shape mismatch"))
        errors = Float64.(Array(uncertainty))
        all(isfinite(errors[i]) && errors[i] >= 0 for i in eachindex(errors) if valid[i]) ||
            throw(ArgumentError("valid prediction uncertainty must be finite and nonnegative"))
        errors
    end
    cov = if isnothing(covariance)
        nothing
    else
        C = Matrix{Float64}(covariance)
        size(C) == (length(values), length(values)) || throw(DimensionMismatch("prediction covariance shape mismatch"))
        all(isfinite, C) && isapprox(C, transpose(C); rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("prediction covariance must be finite and symmetric"))
        eigenvalues = eigvals(Hermitian(C))
        minimum(eigenvalues) >= -1e-10 * max(maximum(abs, eigenvalues), 1.0) ||
            throw(ArgumentError("prediction covariance must be positive semidefinite"))
        C
    end
    data = Array(values)
    return PortablePrediction(grid, data, String(intensity_unit), String(normalization), Symbol(stage),
                              _portable_metadata(response), _portable_metadata(provenance), valid, stderr, cov)
end

function _inferred_coordinates(result; momentum_coordinates=nothing)
    if result isa Observables.LocalSpectrumResult
        sites = Float64.(collect(result.sites))
        return reshape(sites, :, 1), :site
    end
    hasproperty(result, :q) || throw(ArgumentError("only momentum-resolved and local spectral results can be exported"))
    q = collect(result.q)
    if !isnothing(momentum_coordinates)
        positions = Matrix{Float64}(momentum_coordinates)
        length(q) == size(positions, 1) || throw(DimensionMismatch("explicit momentum coordinates and result Q count disagree"))
        return positions, :momentum
    end
    isempty(q) && throw(ArgumentError("result has no momentum samples"))
    if all(point -> point isa Real, q)
        return reshape(Float64.(q), :, 1), :momentum
    elseif all(point -> point isa Tuple || point isa AbstractVector, q)
        d = length(q[1])
        all(point -> length(point) == d && all(v -> v isa Real, point), q) ||
            throw(ArgumentError("momentum coordinate dimensions must agree and be numeric"))
        matrix = Matrix{Float64}(undef, length(q), d)
        for i in eachindex(q)
            matrix[i, :] .= q[i]
        end
        return matrix, :momentum
    end
    throw(ArgumentError("wavevector objects require explicit momentum_coordinates and coordinate_frame to prevent silent frame conversion"))
end

"""Create a neutral portable prediction from a supported numerical spectrum or a selected `ForwardResult` stage."""
function portable_prediction(result; stage::Symbol=:auto, momentum_coordinates=nothing, coordinate_frame::Symbol,
                             coordinate_unit::AbstractString, spectral_unit::AbstractString,
                             intensity_unit::AbstractString, normalization::AbstractString, provenance=Dict{String,Any}(),
                             spectral_kind::Union{Nothing,Symbol}=nothing, include_forward_description::Bool=true,
                             uncertainty=nothing, covariance=nothing)
    selected_stage = stage === :auto ? (result isa ForwardResult ? :instrument : :intrinsic) : stage
    selected = if result isa ForwardResult
        selected_stage === :instrument ? result.result : selected_stage === :probe ? result.scattering_result :
        selected_stage === :intrinsic ? result.observable_result : throw(ArgumentError("stage must be :instrument, :probe, or :intrinsic"))
    else
        result
    end
    isnothing(selected) && throw(ArgumentError("requested forward stage has no numerical spectrum"))
    supported = Union{Observables.SpectrumResult,Observables.LocalSpectrumResult,Observables.TensorSpectrumResult,
                      Observables.CompositeSpectrumResult,Observables.ScatteringResult,Observables.SusceptibilityResult}
    selected isa supported || throw(ArgumentError("portable prediction does not support $(typeof(selected)); select a real-frequency spectral stage"))
    frequency_domain(selected) === :matsubara && throw(ArgumentError("Matsubara responses require an explicit continuation before experimental spectral comparison"))
    coordinate, role = _inferred_coordinates(selected; momentum_coordinates=momentum_coordinates)
    frame = role === :site ? :site : coordinate_frame
    metadata_kind = get(_result_metadata_or_empty(selected), :spectral_axis, nothing)
    isnothing(spectral_kind) && isnothing(metadata_kind) && throw(ArgumentError("spectral_kind must be explicit when the result does not declare :spectral_axis"))
    xkind = isnothing(spectral_kind) ? Symbol(metadata_kind) : spectral_kind
    if !isnothing(metadata_kind) && !isnothing(spectral_kind) && xkind != Symbol(metadata_kind)
        throw(ArgumentError("spectral_kind conflicts with the original result convention"))
    end
    xkind in (:energy, :omega) || throw(ArgumentError("spectral axis must declare :energy or :omega"))
    grid = ExperimentalGrid(coordinate, selected.axis; coordinate_role=role, coordinate_frame=frame,
                            coordinate_unit=coordinate_unit, spectral_kind=xkind, spectral_unit=spectral_unit)
    values = hasproperty(selected, :intensity) ? selected.intensity : selected.values
    mask = get(_result_metadata_or_empty(selected), :coverage_mask, trues(size(values)))
    response = response_convention(selected)
    response[:source_result_type] = string(typeof(selected))
    provenance_record = _portable_metadata(provenance)
    if result isa ForwardResult && include_forward_description
        haskey(provenance_record, "forward") && throw(ArgumentError("provenance key forward is reserved for the source calculation"))
        provenance_record["forward"] = _portable_metadata(Models.describe(result))
    end
    selected_uncertainty = isnothing(uncertainty) && hasproperty(selected, :uncertainty) ? selected.uncertainty : uncertainty
    selected_covariance = isnothing(covariance) && hasproperty(selected, :covariance) ? selected.covariance : covariance
    return _portable_prediction(grid, values; intensity_unit=intensity_unit, normalization=normalization,
                                stage=selected_stage, response=response, provenance=provenance_record, coverage=mask,
                                uncertainty=selected_uncertainty, covariance=selected_covariance)
end

"""Construct a measured dataset with optional per-bin uncertainty, covariance, and explicit detector-validity mask."""
function experimental_dataset(grid::ExperimentalGrid, intensities; uncertainty=nothing, covariance=nothing, likelihood::Symbol=:gaussian,
                              valid=trues(size(intensities)), intensity_unit::AbstractString, normalization::AbstractString,
                              provenance=Dict{String,Any}())
    values = Float64.(Array(intensities))
    ndims(values) in (2, 4) || throw(ArgumentError("measurements must be rank-2 or rank-4 arrays"))
    size(values, 1) == size(grid.positions, 1) && size(values, 2) == length(grid.spectral) ||
        throw(DimensionMismatch("measurement axes and intensity disagree"))
    isempty(intensity_unit) && throw(ArgumentError("intensity unit must be explicit"))
    isempty(normalization) && throw(ArgumentError("normalization must be explicit"))
    size(valid) == size(values) || throw(DimensionMismatch("measurement mask and intensity differ in shape"))
    mask = BitArray(valid)
    any(mask) || throw(ArgumentError("experimental dataset must contain at least one valid measurement bin"))
    all(isfinite(values[i]) for i in eachindex(values) if mask[i]) || throw(ArgumentError("valid measurement intensities must be finite"))
    likelihood in (:gaussian, :poisson) || throw(ArgumentError("likelihood must be :gaussian or :poisson"))
    if likelihood === :poisson
        intensity_unit == "counts" && normalization == "raw_counts" ||
            throw(ArgumentError("Poisson likelihood requires intensity_unit=counts and normalization=raw_counts"))
        isnothing(uncertainty) && isnothing(covariance) ||
            throw(ArgumentError("raw-count Poisson measurements must not supply Gaussian uncertainties or covariance"))
        all(v -> v >= 0 && v == trunc(v), values[mask]) ||
            throw(ArgumentError("Poisson observations must be nonnegative integer raw counts"))
    end
    err = if isnothing(uncertainty)
        nothing
    else
        size(uncertainty) == size(values) || throw(DimensionMismatch("measurement uncertainty shape mismatch"))
        candidate = Float64.(Array(uncertainty))
        all(isfinite(candidate[i]) && candidate[i] > 0 for i in eachindex(candidate) if mask[i]) ||
            throw(ArgumentError("valid measurement standard deviations must be positive and finite"))
        candidate
    end
    cov = if isnothing(covariance)
        nothing
    else
        C = Matrix{Float64}(covariance)
        size(C) == (length(values), length(values)) || throw(DimensionMismatch("covariance must cover the entire flattened measurement grid"))
        all(isfinite, C) && isapprox(C, transpose(C); rtol=1e-10, atol=1e-12) ||
            throw(ArgumentError("measurement covariance must be finite and symmetric"))
        valid_idx = findall(vec(mask))
        cholesky(Hermitian(C[valid_idx, valid_idx]); check=true)
        C
    end
    likelihood === :gaussian && isnothing(err) && isnothing(cov) &&
        throw(ArgumentError("Gaussian measurements require standard deviations or covariance for statistical comparison"))
    return ExperimentalDataset(grid, values, err, cov, mask, String(intensity_unit), String(normalization), _portable_metadata(provenance), likelihood)
end

function _portable_record(kind::String, grid, values, mask, intensity_unit, normalization, provenance; response=nothing, stage=nothing,
                          uncertainty=nothing, covariance=nothing, likelihood=:gaussian)
    record = Dict{String,Any}("schema" => "phundamental.experimental", "version" => EXPERIMENTAL_DATA_SCHEMA_VERSION,
                              "kind" => kind, "grid" => _portable_grid(grid), "intensity" => _portable_values(values; mask=mask),
                              "valid" => collect(vec(mask)), "intensity_unit" => intensity_unit,
                              "normalization" => normalization, "provenance" => _portable_metadata(provenance))
    record["uncertainty"] = isnothing(uncertainty) ? nothing : _portable_values(uncertainty; mask=mask)
    record["covariance"] = isnothing(covariance) ? nothing : _portable_values(covariance)
    if kind == "measurement"
        record["likelihood"] = string(likelihood)
    end
    if kind == "prediction"
        record["response"] = _portable_metadata(response)
        record["stage"] = string(stage)
    end
    return record
end

"""Return a JSON-compatible schema-v1 dictionary with explicit `column_major` array order and no implicit unit conversion."""
export_result(result::PortablePrediction) = _portable_record("prediction", result.grid, result.intensity, result.coverage,
    result.intensity_unit, result.normalization, result.provenance; response=result.response, stage=result.stage,
    uncertainty=result.uncertainty, covariance=result.covariance)

export_result(dataset::ExperimentalDataset) = _portable_record("measurement", dataset.grid, dataset.intensity, dataset.valid,
    dataset.intensity_unit, dataset.normalization, dataset.provenance; uncertainty=dataset.uncertainty, covariance=dataset.covariance,
    likelihood=dataset.likelihood)

export_result(result; kwargs...) = export_result(portable_prediction(result; kwargs...))

"""Validate and reconstruct schema-v1 prediction or measurement records. Only explicit neutral dictionaries are accepted."""
function import_result(record::AbstractDict)
    record["schema"] == "phundamental.experimental" || throw(ArgumentError("unsupported experimental schema"))
    record["version"] == EXPERIMENTAL_DATA_SCHEMA_VERSION || throw(ArgumentError("unsupported experimental schema version"))
    grid = _import_grid(record["grid"])
    shape = Tuple(Int.(record["intensity"]["shape"]))
    all(dimension -> dimension > 0, shape) || throw(ArgumentError("encoded intensity dimensions must be positive"))
    length(record["valid"]) == prod(shape) || throw(DimensionMismatch("measurement coverage length mismatch"))
    all(value -> value isa Bool, record["valid"]) || throw(ArgumentError("coverage flags must be booleans"))
    mask = reshape(Bool.(collect(record["valid"])), shape)
    values = _import_values(record["intensity"], mask)
    uncertainty = isnothing(record["uncertainty"]) ? nothing : _import_values(record["uncertainty"], mask)
    covariance = isnothing(record["covariance"]) ? nothing : _import_values(record["covariance"], trues(length(record["covariance"]["real"])))
    if record["kind"] == "prediction"
        return _portable_prediction(grid, values; coverage=mask, intensity_unit=record["intensity_unit"],
                                    normalization=record["normalization"], stage=Symbol(record["stage"]),
                                    response=record["response"], provenance=record["provenance"],
                                    uncertainty=uncertainty, covariance=covariance)
    elseif record["kind"] == "measurement"
        any(v -> !isreal(v), values[mask]) && throw(ArgumentError("measurements must be real"))
        return experimental_dataset(grid, real.(values); valid=mask, uncertainty=uncertainty, covariance=covariance,
                                    likelihood=Symbol(get(record, "likelihood", "gaussian")),
                                    intensity_unit=record["intensity_unit"], normalization=record["normalization"],
                                    provenance=record["provenance"])
    end
    throw(ArgumentError("experimental record kind must be prediction or measurement"))
end

function _same_experimental_grid(a::ExperimentalGrid, b::ExperimentalGrid)
    return a.coordinate_role == b.coordinate_role && a.coordinate_frame == b.coordinate_frame &&
           a.coordinate_unit == b.coordinate_unit && a.spectral_unit == b.spectral_unit &&
           a.spectral_kind == b.spectral_kind && a.positions == b.positions && a.spectral == b.spectral
end

"""Compare consistent predictions and measurements using Gaussian chi-square or raw-count Poisson deviance.

Poisson `deviance` is twice the log likelihood ratio relative to the saturated model (Cash deviance).
Its `log_likelihood` omits the data-only log-factorial normalization; `chi_squared` is `NaN` in Poisson mode.
"""
function compare_measurement(prediction::PortablePrediction, measurement::ExperimentalDataset; imaginary_tolerance::Real=1e-10)
    _same_experimental_grid(prediction.grid, measurement.grid) || throw(ArgumentError("prediction and measurement grids, units, or frames differ"))
    prediction.intensity_unit == measurement.intensity_unit || throw(ArgumentError("prediction and measurement intensity units differ"))
    prediction.normalization == measurement.normalization || throw(ArgumentError("prediction and measurement normalizations differ"))
    size(prediction.intensity) == size(measurement.intensity) || throw(DimensionMismatch("prediction and measurement shapes differ"))
    isfinite(imaginary_tolerance) && imaginary_tolerance >= 0 || throw(ArgumentError("imaginary_tolerance must be finite and nonnegative"))
    mask = prediction.coverage .& measurement.valid
    idx = findall(vec(mask))
    isempty(idx) && throw(ArgumentError("no simultaneously valid predicted and measured bins"))
    predicted = vec(prediction.intensity)[idx]
    maximum(abs, imag.(predicted)) <= imaginary_tolerance ||
        throw(ArgumentError("comparison requires real predictions; complex response tensors must be contracted first"))
    expected = Float64.(real.(predicted))
    observed = vec(measurement.intensity)[idx]
    residual = expected .- observed
    if measurement.likelihood === :poisson
        all(v -> isfinite(v) && v >= 0, expected) || throw(ArgumentError("Poisson expected counts must be finite and nonnegative"))
        deviance = 0.0
        log_likelihood = 0.0
        standardized = Vector{Float64}(undef, length(idx))
        for i in eachindex(idx)
            mu = expected[i]
            n = observed[i]
            term = if n == 0
                mu
            elseif mu == 0
                Inf
            else
                mu - n + n * log(n / mu)
            end
            term = max(term, 0.0)
            deviance += 2 * term
            log_likelihood += mu == 0 ? (n == 0 ? 0.0 : -Inf) : n * log(mu) - mu
            standardized[i] = sign(residual[i]) * sqrt(2 * term)
        end
        return MeasurementComparison(NaN, length(idx), residual, standardized, :poisson, deviance, log_likelihood)
    end
    if !isnothing(measurement.covariance)
        F = cholesky(Hermitian(measurement.covariance[idx, idx]); check=true)
        whitened = F.L \ residual
        logdet = 2 * sum(log, diag(F.L))
    else
        isnothing(measurement.uncertainty) && throw(ArgumentError("Gaussian comparison requires covariance or standard deviation"))
        sigma = vec(measurement.uncertainty)[idx]
        whitened = residual ./ sigma
        logdet = 2 * sum(log, sigma)
    end
    chi2 = sum(abs2, whitened)
    ll = -0.5 * (chi2 + logdet + length(idx) * log(2π))
    return MeasurementComparison(chi2, length(idx), residual, collect(whitened), :gaussian, chi2, ll)
end
