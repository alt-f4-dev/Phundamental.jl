# Reproduce and characterize small-grid Q–E serial/threaded disagreement.
# From repository root:
# OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmarks benchmarks/diagnose-experimental-resolution-threading.jl
# This script does not modify the package or call the benchmark's main() function.

using Phundamental
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "experimental-resolution-scaling.jl"))

function _diagnostic_resolution(sample)
    return SpatialGaussianQEResolution(sample.provider, sample.coordinates, sample.energies, sample.weights;
                                       sample.keywords..., storage=:lazy)
end

function _diagnostic_spectrum(sample, values)
    return Phundamental.Observables.ScatteringResult(sample.spectrum.q, sample.energies, values, Dict{Symbol,Any}())
end

function _diagnostic_nonzero_sources(values)
    sources = Tuple{Int,Int}[]
    for jx in axes(values, 2), jq in axes(values, 1)
        iszero(values[jq, jx]) || push!(sources, (jq, jx))
    end
    return sources
end

function _diagnostic_compare(label, serial, threaded; atol=1e-11, rtol=1e-10)
    difference = abs.(threaded .- serial)
    maximum_difference, maximum_index = findmax(difference)
    criterion = isapprox.(threaded, serial; atol, rtol)
    mismatched = findall(!, criterion)
    @printf("%-28s max|Δ|=%.6e mismatches=%d/%d serial_sum=%.12e threaded_sum=%.12e\n",
            label, maximum_difference, length(mismatched), length(criterion), sum(serial), sum(threaded))
    if !isempty(mismatched)
        row, column = Tuple(maximum_index)
        @printf("  largest mismatch at (%d,%d): serial=%.17e threaded=%.17e\n",
                row, column, serial[maximum_index], threaded[maximum_index])
        for index in Iterators.take(mismatched, 8)
            row, column = Tuple(index)
            @printf("  mismatch (%d,%d) serial=%.17e threaded=%.17e |Δ|=%.3e\n",
                    row, column, serial[index], threaded[index], difference[index])
        end
    end
    return isempty(mismatched)
end

function _diagnostic_run_case(dimension::Int, nQ::Int, nE::Int, density::Symbol, covariance_kind::Symbol; repetitions::Int=5)
    sample = _qe_case(dimension, nQ, nE, density, covariance_kind)
    resolution = _diagnostic_resolution(sample)
    sources = _diagnostic_nonzero_sources(sample.values)
    println("\nCASE d=$dimension nQ=$nQ nE=$nE density=$density covariance=$covariance_kind source_count=$(length(sources))")
    if length(sources) <= 16
        println("  nonzero source (Q index, energy index): ", sources)
    end
    serial = convolve_resolution(resolution, sample.spectrum; threaded=false).intensity
    for iteration in 1:repetitions
        threaded = convolve_resolution(resolution, sample.spectrum; threaded=true).intensity
        _diagnostic_compare("combined run $iteration", serial, threaded)
    end
    if nQ <= 8 && nE <= 8
        direct = _qe_direct_reference(sample)
        _diagnostic_compare("serial vs independent", direct, serial)
        _diagnostic_compare("threaded vs independent", direct,
                            convolve_resolution(resolution, sample.spectrum; threaded=true).intensity)
    end
    if length(sources) <= 16
        for (jq, jx) in sources
            isolated = zeros(size(sample.values))
            isolated[jq, jx] = sample.values[jq, jx]
            isolated_spectrum = _diagnostic_spectrum(sample, isolated)
            isolated_serial = convolve_resolution(resolution, isolated_spectrum; threaded=false).intensity
            isolated_threaded = convolve_resolution(resolution, isolated_spectrum; threaded=true).intensity
            _diagnostic_compare("isolated Q=$jq E=$jx", isolated_serial, isolated_threaded)
        end
    end
    return nothing
end

function main()
    println("Q–E threaded diagnostic; Julia=$VERSION threads=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    println("Only small synthetic inputs are evaluated. No benchmark output files are modified.")
    _qe_validate_covariance_providers()
    _diagnostic_run_case(2, 8, 8, :sparse, :varying)
    _diagnostic_run_case(2, 8, 8, :sparse, :constant)
    _diagnostic_run_case(3, 8, 8, :sparse, :varying)
    _diagnostic_run_case(2, 64, 8, :sparse, :varying)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
