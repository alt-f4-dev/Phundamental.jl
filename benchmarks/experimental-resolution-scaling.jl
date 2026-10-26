using Phundamental
using LinearAlgebra
using Printf
using Statistics

# Benchmarks the Phase I multidimensional Q–E resolution backend without changing the package's numerical kernels.
# From benchmarks/: OPENBLAS_NUM_THREADS=1 julia -t auto --project=. experimental-resolution-scaling.jl
# Profiles: PHUNDAMENTAL_QE_PROFILE=quick (default), medium, tof, or tof_full.
# In medium and tof, PHUNDAMENTAL_QE_ADAPTIVE=true scales momentum counts down when needed.
# Actual and requested dimensions are reported separately; scaled cases do not prove full-size throughput.
# Set PHUNDAMENTAL_QE_ADAPTIVE=false or use tof_full to retain strict, possibly skipped workloads.
# PHUNDAMENTAL_QE_THREADED=true enables source-parallel scatter for comparisons.
# PHUNDAMENTAL_QE_VERIFY_PARALLEL=true checks every measured threaded output against the serial kernel.
# Never infer speedup from the thread count; recorded runtimes determine which path is preferable.

const QE_MIB = 2.0^20
const QE_PRIMES = (2, 3, 5)

function _qe_env_int(name::AbstractString, default::Int)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || throw(ArgumentError("$name must be a positive integer"))
    return value
end

function _qe_env_float(name::AbstractString, default::Real)
    value = parse(Float64, get(ENV, name, string(default)))
    isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    return value
end

function _qe_env_bool(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    value in ("true", "1", "yes") && return true
    value in ("false", "0", "no") && return false
    throw(ArgumentError("$name must be true/false, 1/0, or yes/no"))
end

function _qe_env_choices(name::AbstractString, default::AbstractString, allowed::Tuple)
    selections = Symbol.(strip.(split(get(ENV, name, default), ',')))
    !isempty(selections) && all(choice -> choice in allowed, selections) ||
        throw(ArgumentError("$name must contain comma-separated values from $(join(allowed, ", "))"))
    return unique(selections)
end

function _qe_radical_inverse(index::Int, base::Int)
    factor = inv(Float64(base))
    value = 0.0
    while index > 0
        index, digit = divrem(index, base)
        value += digit * factor
        factor /= base
    end
    return value
end

function _qe_coordinates(nQ::Int, dimension::Int)
    dimension in (2, 3) || throw(ArgumentError("momentum dimension must be two or three"))
    coordinates = Matrix{Float64}(undef, nQ, dimension)
    for i in 1:nQ, axis in 1:dimension
        coordinates[i, axis] = _qe_radical_inverse(i, QE_PRIMES[axis])
    end
    return coordinates
end

function _qe_source_values(coordinates, energies, density::Symbol)
    density in (:sparse, :dense) || throw(ArgumentError("density must be :sparse or :dense"))
    nQ, dimension = size(coordinates)
    values = Matrix{Float64}(undef, nQ, length(energies))
    nonzero = 0
    for j in eachindex(energies), i in 1:nQ
        if density === :sparse && mod(i + 17 * j, 31) != 0
            values[i, j] = 0.0
            continue
        end
        qmean = sum(@view coordinates[i, :]) / dimension
        center = 8.0 + 7.0 * qmean
        values[i, j] = 0.10 + exp(-0.5 * ((energies[j] - center) / 1.8)^2) + 0.04 * qmean
        nonzero += 1
    end
    return values, nonzero
end

function _qe_correlation_matrix(dimension::Int)
    d = dimension + 1
    correlation = Matrix{Float64}(I, d, d)
    correlation[1, d] = correlation[d, 1] = 0.28
    correlation[2, d] = correlation[d, 2] = -0.20
    correlation[1, 2] = correlation[2, 1] = 0.10
    if dimension == 3
        correlation[3, d] = correlation[d, 3] = 0.12
    end
    isposdef(Symmetric(correlation)) || error("benchmark covariance correlation matrix is not positive definite")
    return correlation
end

function _qe_covariance_provider(dimension::Int, kind::Symbol)
    kind in (:constant, :varying) || throw(ArgumentError("covariance must be :constant or :varying"))
    correlation = _qe_correlation_matrix(dimension)
    covariance_dimension = size(correlation, 1)
    covariance_dimension == dimension + 1 || throw(DimensionMismatch("Q–E covariance requires one energy coordinate in addition to momentum"))
    if kind === :constant
        widths = vcat(fill(0.021, dimension), 0.65)
        constant_covariance = Diagonal(widths) * correlation * Diagonal(widths)
        return (q, energy) -> constant_covariance
    end
    return function(q, energy)
        length(q) == dimension || throw(DimensionMismatch("source momentum coordinate dimension differs from covariance provider"))
        qphase = 2π * q[1]
        widths = Vector{Float64}(undef, covariance_dimension)
        @inbounds for component in 1:dimension
            widths[component] = 0.019 + 0.009 * (0.5 + 0.5 * sin(qphase + component / 2))
        end
        widths[end] = 0.48 + 0.20 * (0.5 + 0.5 * cos(0.3 * energy + qphase))
        return Diagonal(widths) * correlation * Diagonal(widths)
    end
end

function _qe_validate_covariance_providers()
    for dimension in (2, 3), kind in (:constant, :varying)
        provider = _qe_covariance_provider(dimension, kind)
        for momentum in (fill(0.0, dimension), fill(0.25, dimension), fill(0.75, dimension)), energy in (-5.0, 10.0, 25.0)
            covariance = provider(momentum, energy)
            size(covariance) == (dimension + 1, dimension + 1) ||
                error("invalid Q–E covariance dimensions for d=$dimension $kind Q=$momentum E=$energy")
            isposdef(Hermitian(covariance)) ||
                error("non-positive-definite Q–E covariance for d=$dimension $kind Q=$momentum E=$energy")
        end
    end
    return nothing
end

function _qe_case(dimension::Int, nQ::Int, nE::Int, density::Symbol, covariance_kind::Symbol)
    coordinates = _qe_coordinates(nQ, dimension)
    q = [collect(@view coordinates[i, :]) for i in 1:nQ]
    energies = collect(range(-5.0, 25.0; length=nE))
    values, nonzero = _qe_source_values(coordinates, energies, density)
    weights = fill(inv(Float64(nQ)), nQ)
    spectrum = Phundamental.Observables.ScatteringResult(q, energies, values, Dict{Symbol,Any}())
    provider = _qe_covariance_provider(dimension, covariance_kind)
    keywords = (; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=3.0)
    return (; spectrum, coordinates, energies, values, weights, provider, keywords, nonzero)
end

function _qe_energy_weights(energies)
    weights = Vector{Float64}(undef, length(energies))
    weights[1] = (energies[2] - energies[1]) / 2
    weights[end] = (energies[end] - energies[end-1]) / 2
    for j in 2:length(energies)-1
        weights[j] = (energies[j+1] - energies[j-1]) / 2
    end
    return weights
end

# Independent O((nQ*nE)^2) reference: it neither calls the package convolution nor uses its neighbor-search tree.
# Restrict to deliberately small cases; it is not suitable for TOF-scale dimensions.
function _qe_direct_reference(sample)
    values = sample.values
    coordinates = sample.coordinates
    energies = sample.energies
    nQ, nE = size(values)
    d = size(coordinates, 2)
    energy_weights = _qe_energy_weights(energies)
    output = zeros(Float64, nQ, nE)
    for jq in 1:nQ, jx in 1:nE
        intensity = values[jq, jx]
        iszero(intensity) && continue
        covariance = sample.provider(view(coordinates, jq, :), energies[jx])
        precision = inv(covariance)
        amplitude = inv(sqrt((2π)^(d + 1) * det(covariance)))
        factor = intensity * sample.weights[jq] * energy_weights[jx] * amplitude
        for iq in 1:nQ, ix in 1:nE
            displacement = [coordinates[iq, k] - coordinates[jq, k] for k in 1:d]
            push!(displacement, energies[ix] - energies[jx])
            distance2 = dot(displacement, precision * displacement)
            distance2 <= sample.keywords.nsigma^2 || continue
            output[iq, ix] += factor * exp(-0.5 * max(distance2, 0.0))
        end
    end
    return output
end

function _qe_maximum_error(actual, reference)
    maximum_error = maximum(abs.(actual .- reference))
    relative_error = maximum_error / max(maximum(abs, reference), eps(Float64))
    return maximum_error, relative_error
end

function _qe_validate_small(sample, lazy_output)
    nQ, nE = size(sample.values)
    nQ * nE <= 512 || return (NaN, NaN, NaN, NaN)
    d = size(sample.coordinates, 2) + 1
    covariance_array = Array{Float64}(undef, d, d, nQ, nE)
    for j in 1:nE, i in 1:nQ
        covariance_array[:, :, i, j] .= sample.provider(view(sample.coordinates, i, :), sample.energies[j])
    end
    dense = SpatialGaussianQEResolution(covariance_array, sample.coordinates, sample.energies, sample.weights;
                                        sample.keywords..., storage=:dense)
    array_lazy = SpatialGaussianQEResolution(covariance_array, sample.coordinates, sample.energies, sample.weights;
                                             sample.keywords..., storage=:lazy)
    dense_output = convolve_resolution(dense, sample.spectrum).intensity
    array_lazy_output = convolve_resolution(array_lazy, sample.spectrum).intensity
    independent_output = _qe_direct_reference(sample)
    dense_abs, dense_rel = _qe_maximum_error(lazy_output, dense_output)
    array_abs, array_rel = _qe_maximum_error(lazy_output, array_lazy_output)
    brute_abs, brute_rel = _qe_maximum_error(lazy_output, independent_output)
    all(isapprox.(lazy_output, dense_output; atol=1e-11, rtol=1e-10)) || error("callable-lazy and dense covariance outputs differ")
    all(isapprox.(lazy_output, array_lazy_output; atol=1e-11, rtol=1e-10)) || error("callable-lazy and array-lazy outputs differ")
    all(isapprox.(lazy_output, independent_output; atol=1e-10, rtol=1e-9)) || error("Q–E convolution differs from the independent reference")
    return (dense_abs, max(dense_rel, array_rel), brute_abs, brute_rel)
end

function _qe_memory_bytes(label::AbstractString)
    isfile("/proc/self/status") || return NaN
    for line in eachline("/proc/self/status")
        startswith(line, label * ":") || continue
        return parse(Float64, split(line)[2]) * 1024.0
    end
    return NaN
end

function _qe_profiles(profile::Symbol)
    if profile === :quick
        return [(2, 16, 24), (2, 64, 24), (2, 144, 24), (3, 64, 24), (3, 512, 24), (3, 1_728, 24)]
    elseif profile === :medium
        return [(2, 256, 64), (2, 1_024, 128), (2, 4_096, 200), (3, 512, 64), (3, 1_728, 128), (3, 4_096, 200)]
    elseif profile === :tof
        return [(2, 256, 64), (2, 1_024, 128), (2, 10_000, 200),
                (3, 512, 64), (3, 2_048, 128), (3, 10_000, 200), (3, 100_000, 290)]
    elseif profile === :tof_full
        return [(2, 10_000, 200), (3, 10_000, 200), (3, 100_000, 290)]
    end
    throw(ArgumentError("PHUNDAMENTAL_QE_PROFILE must be quick, medium, tof, or tof_full"))
end

# Rough workload estimate for budget gating, not an observed operation count or asymptotic complexity claim.
function _qe_estimated_pairs(dimension::Int, nQ::Int, nE::Int, density::Symbol)
    fraction = density === :dense ? 1.0 : inv(31.0)
    source_bins = Float64(nQ) * nE * fraction
    q_neighbors = min(Float64(nQ), max(1.0, Float64(nQ) * (2 * 3.0 * 0.03)^dimension))
    energy_neighbors = min(Float64(nE), max(1.0, Float64(nE) * (2 * 3.0 * 0.70 / 30.0)))
    return source_bins * q_neighbors * energy_neighbors
end

# Select the largest momentum grid meeting both work and input-size limits.
# The number of energy bins and the sparse/dense input distribution are never changed.
function _qe_adjust_grid(dimension::Int, requested_nQ::Int, nE::Int, density::Symbol, settings)
    original_work = _qe_estimated_pairs(dimension, requested_nQ, nE, density)
    original_mib = 8.0 * Float64(requested_nQ) * nE / QE_MIB
    if settings.force || (original_work <= settings.budget && original_mib <= settings.max_input_mib)
        return requested_nQ
    end
    settings.adaptive || return requested_nQ
    low = 16
    if low > requested_nQ || _qe_estimated_pairs(dimension, low, nE, density) > settings.budget ||
       8.0 * low * nE / QE_MIB > settings.max_input_mib
        return requested_nQ
    end
    high = requested_nQ
    while low + 1 < high
        candidate = low + (high - low) ÷ 2
        candidate_work = _qe_estimated_pairs(dimension, candidate, nE, density)
        candidate_mib = 8.0 * Float64(candidate) * nE / QE_MIB
        if candidate_work <= settings.budget && candidate_mib <= settings.max_input_mib
            low = candidate
        else
            high = candidate
        end
    end
    return low
end

function _qe_settings()
    profile = Symbol(lowercase(strip(get(ENV, "PHUNDAMENTAL_QE_PROFILE", "quick"))))
    profile in (:quick, :medium, :tof, :tof_full) || throw(ArgumentError("unrecognized PHUNDAMENTAL_QE_PROFILE"))
    densities = _qe_env_choices("PHUNDAMENTAL_QE_DENSITIES", "sparse,dense", (:sparse, :dense))
    covariances = _qe_env_choices("PHUNDAMENTAL_QE_COVARIANCES", "constant,varying", (:constant, :varying))
    repetitions = _qe_env_int("PHUNDAMENTAL_QE_REPEATS", profile === :quick ? 3 : 1)
    budget = _qe_env_float("PHUNDAMENTAL_QE_MAX_CANDIDATES", 2.0e8)
    max_input_mib = _qe_env_float("PHUNDAMENTAL_QE_MAX_INPUT_MIB", 1024.0)
    force = _qe_env_bool("PHUNDAMENTAL_QE_FORCE")
    threaded = _qe_env_bool("PHUNDAMENTAL_QE_THREADED")
    adaptive = _qe_env_bool("PHUNDAMENTAL_QE_ADAPTIVE", profile in (:medium, :tof))
    verify_parallel = _qe_env_bool("PHUNDAMENTAL_QE_VERIFY_PARALLEL", true)
    energy_override = get(ENV, "PHUNDAMENTAL_QE_SCALING_ENERGIES", "")
    nenergy = isempty(energy_override) ? nothing : _qe_env_int("PHUNDAMENTAL_QE_SCALING_ENERGIES", 24)
    output = strip(get(ENV, "PHUNDAMENTAL_QE_OUTPUT", ""))
    return (; profile, densities, covariances, repetitions, budget, max_input_mib, force, threaded, adaptive, verify_parallel, nenergy, output)
end

function _qe_benchmark_case(dimension, requested_nQ, requested_nE, density, covariance_kind, settings)
    nE = requested_nE
    nQ = _qe_adjust_grid(dimension, requested_nQ, nE, density, settings)
    predicted = _qe_estimated_pairs(dimension, nQ, nE, density)
    intensity_mib = 8.0 * Float64(nQ) * nE / QE_MIB
    scaled = nQ != requested_nQ
    if scaled
        @printf("SCALE d=%d Q=%d -> %d E=%d %s %s; estimated work=%.3e <= %.3e pairs\n",
                dimension, requested_nQ, nQ, nE, string(density), string(covariance_kind), predicted, settings.budget)
    end
    if !settings.force && (predicted > settings.budget || intensity_mib > settings.max_input_mib)
        reason = predicted > settings.budget ? "estimated-work" : "input-memory"
        @printf("SKIP d=%d nQ=%d nE=%d density=%s covariance=%s reason=%s predicted=%.3e pairs input=%.1f MiB\n",
                dimension, nQ, nE, string(density), string(covariance_kind), reason, predicted, intensity_mib)
        return (; status="SKIP", dimension, nQ, nE, density, covariance_kind, threaded=settings.threaded, predicted, nonzero=0,
                construction_s=NaN, construction_alloc=NaN, resolution_mib=NaN, convolution_s=NaN,
                convolution_alloc=NaN, output_mib=NaN, rss_mib=NaN, rss_peak_mib=NaN,
                dense_abs=NaN, dense_rel=NaN, brute_abs=NaN, brute_rel=NaN,
                requested_nQ, requested_nE, parallel_abs=NaN, parallel_rel=NaN)
    end

    sample = _qe_case(dimension, nQ, nE, density, covariance_kind)
    GC.gc()
    built = @timed SpatialGaussianQEResolution(sample.provider, sample.coordinates, sample.energies, sample.weights;
                                                sample.keywords..., storage=:lazy)
    resolution = built.value
    GC.gc()
    # Repeated measurements exclude input generation and resolution construction; construction is measured above.
    times = Float64[]
    allocations = Float64[]
    last_output = nothing
    for _ in 1:settings.repetitions
        GC.gc()
        timed = @timed convolve_resolution(resolution, sample.spectrum; threaded=settings.threaded)
        push!(times, timed.time)
        push!(allocations, timed.bytes)
        last_output = timed.value.intensity
    end
    all(isfinite, last_output) || error("nonfinite convolution output")
    minimum(last_output) >= -1e-12 || error("convolution produced negative output for a nonnegative spectrum")
    dense_abs, dense_rel, brute_abs, brute_rel = _qe_validate_small(sample, last_output)
    parallel_abs, parallel_rel = NaN, NaN
    # This independent call does not contribute to reported convolution time or allocations.
    # It validates the actual benchmark grid, not only its small warm-up cases.
    if settings.threaded && settings.verify_parallel
        reference = convolve_resolution(resolution, sample.spectrum; threaded=false).intensity
        parallel_abs, parallel_rel = _qe_maximum_error(last_output, reference)
        all(isapprox.(last_output, reference; atol=1e-10, rtol=1e-9)) ||
            error("threaded and serial kernels disagree for d=$dimension nQ=$nQ nE=$nE density=$density covariance=$covariance_kind: max|Δ|=$parallel_abs")
    end
    result = (; status=(scaled ? "SCALED" : "OK"), dimension, nQ, nE, density, covariance_kind, threaded=settings.threaded, predicted,
              nonzero=sample.nonzero, construction_s=built.time, construction_alloc=built.bytes / QE_MIB,
              resolution_mib=Base.summarysize(resolution) / QE_MIB, convolution_s=median(times),
              convolution_alloc=median(allocations) / QE_MIB, output_mib=Base.summarysize(last_output) / QE_MIB,
              rss_mib=_qe_memory_bytes("VmRSS") / QE_MIB, rss_peak_mib=_qe_memory_bytes("VmHWM") / QE_MIB,
              dense_abs, dense_rel, brute_abs, brute_rel, requested_nQ, requested_nE, parallel_abs, parallel_rel)
    @printf("d=%d nQ=%d nE=%d %-6s %-8s threaded=%s nonzero=%d build=%.3f s conv=%.4f s alloc=%.2f MiB R=%.2f MiB RSS=%.1f MiB peakRSS=%.1f MiB status=%s\n",
            dimension, nQ, nE, string(density), string(covariance_kind), string(settings.threaded), sample.nonzero,
            result.construction_s, result.convolution_s, result.convolution_alloc, result.resolution_mib,
            result.rss_mib, result.rss_peak_mib, result.status)
    if !isnan(result.brute_abs)
        @printf("  independent max|Δ|=%.3e relative=%.3e; dense max|Δ|=%.3e relative=%.3e\n",
                result.brute_abs, result.brute_rel, result.dense_abs, result.dense_rel)
    end
    if !isnan(result.parallel_abs)
        @printf("  verified against serial: max|Δ|=%.3e relative=%.3e\n", result.parallel_abs, result.parallel_rel)
    end
    return result
end

function _qe_write_csv(filename, results)
    columns = (:status, :dimension, :nQ, :nE, :density, :covariance_kind, :threaded, :predicted, :nonzero,
               :construction_s, :construction_alloc, :resolution_mib, :convolution_s,
               :convolution_alloc, :output_mib, :rss_mib, :rss_peak_mib,
               :dense_abs, :dense_rel, :brute_abs, :brute_rel,
               :requested_nQ, :requested_nE, :parallel_abs, :parallel_rel)
    mkpath(dirname(filename))
    open(filename, "w") do io
        println(io, join(string.(columns), ","))
        for result in results
            println(io, join((string(getproperty(result, column)) for column in columns), ","))
        end
    end
    return filename
end

function _qe_warmup(threaded::Bool)
    for dimension in (2, 3), kind in (:constant, :varying), density in (:sparse, :dense)
        sample = _qe_case(dimension, 8, 8, density, kind)
        resolution = SpatialGaussianQEResolution(sample.provider, sample.coordinates, sample.energies, sample.weights;
                                                  sample.keywords..., storage=:lazy)
        try
            reference = convolve_resolution(resolution, sample.spectrum; threaded=false).intensity
            for attempt in 1:(threaded ? 10 : 1)
                output = convolve_resolution(resolution, sample.spectrum; threaded).intensity
                all(isapprox.(output, reference; atol=1e-11, rtol=1e-10)) ||
                    error("Q–E warm-up disagrees with serial for dimension=$dimension covariance=$kind density=$density attempt=$attempt")
            end
        catch error
            @error "Q–E warm-up failed" dimension covariance=kind density threaded exception=(error, catch_backtrace())
            rethrow()
        end
    end
    return nothing
end

function main()
    settings = _qe_settings()
    _qe_validate_covariance_providers()
    _qe_warmup(settings.threaded)
    @printf("Q–E scaling profile=%s Julia=%s threads=%d BLAS=%d threaded=%s repeats=%d max-candidates=%.2e adaptive=%s verify-parallel=%s force=%s\n",
            string(settings.profile), string(VERSION), Threads.nthreads(), BLAS.get_num_threads(),
            string(settings.threaded), settings.repetitions, settings.budget,
            string(settings.adaptive), string(settings.verify_parallel), string(settings.force))
    println("Synthetic quasi-uniform Q samples in [0,1]^d; E in [-5,25] meV; signed Q–E covariance correlations.")
    println("Convolution times exclude separate warm-up, input generation, and resolution construction; allocations are cumulative, not peak memory.")
    println("VmHWM is the process-wide Linux high-water RSS and is not a per-case peak; estimated pairs are heuristic budget guards.")
    println("Resolution construction includes one spatial-index build. Threaded lazy source properties are prepared in bounded batches.")
    println("Adaptive scaling preserves the requested energy-bin count and source density; SCALED is not a full-size TOF measurement.")
    println("Serial cross-checks are outside reported convolution timings; peak RSS may include verification when enabled.")
    results = NamedTuple[]
    for (dimension, nQ, default_nE) in _qe_profiles(settings.profile)
        nE = isnothing(settings.nenergy) ? default_nE : settings.nenergy
        nE >= 2 || throw(ArgumentError("the number of energy bins must be at least two"))
        for density in settings.densities, kind in settings.covariances
            push!(results, _qe_benchmark_case(dimension, nQ, nE, density, kind, settings))
        end
    end
    if !isempty(settings.output)
        _qe_write_csv(settings.output, results)
        println("CSV: ", abspath(settings.output))
    end
    println("Completed: ", count(result -> result.status != "SKIP", results),
            " cases (", count(result -> result.status == "SCALED", results), " resized); skipped: ",
            count(result -> result.status == "SKIP", results), ".")
    return results
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
