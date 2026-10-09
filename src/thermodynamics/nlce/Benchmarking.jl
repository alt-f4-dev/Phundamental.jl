# Peer-reviewed benchmark metadata and comparison utilities for NLCE validation.

"""Published reference series with explicit provenance, model parameters, values, and uncertainties."""
struct LiteratureBenchmark{X,Y,U,P,M}
    name::Symbol
    citation::String
    doi::String
    observable::Symbol
    x::X
    values::Y
    uncertainty::U
    parameters::P
    metadata::M
end

function LiteratureBenchmark(name::Symbol, citation::AbstractString, doi::AbstractString, observable::Symbol, x, values;
                             uncertainty=nothing, parameters=Dict{Symbol,Any}(), metadata=Dict{Symbol,Any}())
    xs = collect(x)
    ys = collect(values)
    length(xs) == length(ys) || throw(DimensionMismatch("benchmark coordinate and value vectors must have equal lengths"))
    uncertainty_values = isnothing(uncertainty) ? fill(NaN, length(ys)) : Float64.(collect(uncertainty))
    length(uncertainty_values) == length(ys) || throw(DimensionMismatch("benchmark uncertainty vector must match the values"))
    return LiteratureBenchmark(name, String(citation), String(doi), observable, xs, ys, uncertainty_values,
                               Dict{Symbol,Any}(parameters), Dict{Symbol,Any}(metadata))
end

"""Machine-readable residual report for a calculation compared with a published benchmark."""
struct LiteratureBenchmarkResult{B,P,R,N,M}
    benchmark::B
    predicted::P
    residuals::R
    normalized_residuals::N
    passed::Bool
    metadata::M
end

"""Compare predicted values with a peer-reviewed benchmark using absolute, relative, and reported-uncertainty tolerances."""
function compare_literature_benchmark(benchmark::LiteratureBenchmark, predicted; atol::Real=0.0, rtol::Real=0.0,
                                      uncertainty_sigma::Real=3.0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    uncertainty_sigma > 0 || throw(ArgumentError("uncertainty_sigma must be positive"))
    predictions = collect(predicted)
    length(predictions) == length(benchmark.values) || throw(DimensionMismatch("prediction count must match benchmark values"))
    residuals = Float64[abs(Float64(real(predictions[i] - benchmark.values[i]))) for i in eachindex(predictions)]
    normalized = fill(NaN, length(residuals))
    passed = true
    for i in eachindex(residuals)
        scale = atol + rtol * max(abs(Float64(real(benchmark.values[i]))), abs(Float64(real(predictions[i]))), 1.0)
        uncertainty = benchmark.uncertainty[i]
        if isfinite(uncertainty) && uncertainty > 0
            normalized[i] = residuals[i] / uncertainty
            scale = max(scale, uncertainty_sigma * uncertainty)
        end
        residuals[i] <= scale || (passed = false)
    end
    metadata = Dict{Symbol,Any}(
        :max_absolute_residual => isempty(residuals) ? 0.0 : maximum(residuals),
        :max_normalized_residual => all(isnan, normalized) ? NaN : maximum(filter(x -> !isnan(x), normalized)),
        :atol => Float64(atol),
        :rtol => Float64(rtol),
        :uncertainty_sigma => Float64(uncertainty_sigma),
    )
    return LiteratureBenchmarkResult(benchmark, predictions, residuals, normalized, passed, metadata)
end

"""
    rigol_2007_square_site_counts(; max_order=16)

Published site-based square-lattice NLCE graph-count benchmark from Table II of Rigol, Bryant, and Singh, Phys. Rev. E 75, 061118 (2007), DOI 10.1103/PhysRevE.75.061118. The lattice constants are sums of embeddings per lattice site for all topological classes at a given site order.
"""
function rigol_2007_square_site_counts(; max_order::Integer=16)
    1 <= max_order <= 16 || throw(ArgumentError("Rigol square-site benchmark is tabulated for orders 1 through 16"))
    topological = Int[1, 1, 1, 3, 4, 10, 19, 51, 112, 300, 746, 2042, 5450, 15197, 42192, 119561]
    lattice_constants = Int[1, 2, 6, 19, 63, 216, 760, 2725, 9910, 36446, 135268, 505861, 1903890, 7204874, 27394666, 104592937]
    orders = collect(1:Int(max_order))
    metadata = Dict{Symbol,Any}(:table => :II, :lattice => :square, :expansion => :site_based, :peer_reviewed => true)
    return (
        topological_clusters=LiteratureBenchmark(:rigol_2007_square_site_topologies,
            "M. Rigol, T. Bryant, and R. R. P. Singh, Phys. Rev. E 75, 061118 (2007)", "10.1103/PhysRevE.75.061118",
            :topological_cluster_count, orders, topological[orders]; metadata=metadata),
        lattice_constants=LiteratureBenchmark(:rigol_2007_square_site_lattice_constants,
            "M. Rigol, T. Bryant, and R. R. P. Singh, Phys. Rev. E 75, 061118 (2007)", "10.1103/PhysRevE.75.061118",
            :sum_lattice_constants, orders, lattice_constants[orders]; metadata=metadata),
    )
end

"""Summarize topological-cluster counts and the sum of lattice constants at every retained site order."""
function summarize_topological_expansion(expansion::TopologicalExpansion)
    max_order = maximum(cluster.order for cluster in expansion.clusters)
    counts = zeros(Int, max_order)
    lattice_constants = zeros(Float64, max_order)
    for cluster in expansion.clusters
        counts[cluster.order] += 1
        lattice_constants[cluster.order] += cluster.multiplicity
    end
    return (orders=collect(1:max_order), topological_cluster_count=counts, sum_lattice_constants=lattice_constants)
end

"""Compare an automatically generated square-lattice site expansion with the peer-reviewed Rigol-Bryant-Singh Table II counts."""
function validate_rigol_2007_square_site_counts(expansion::TopologicalExpansion; max_order::Integer=maximum(c.order for c in expansion.clusters), atol::Real=1e-10)
    reference = rigol_2007_square_site_counts(; max_order=max_order)
    summary = summarize_topological_expansion(expansion)
    topology = compare_literature_benchmark(reference.topological_clusters, summary.topological_cluster_count[1:max_order]; atol=0.0, rtol=0.0)
    multiplicity = compare_literature_benchmark(reference.lattice_constants, summary.sum_lattice_constants[1:max_order]; atol=atol, rtol=0.0)
    return (passed=topology.passed && multiplicity.passed, topological_clusters=topology, lattice_constants=multiplicity)
end
