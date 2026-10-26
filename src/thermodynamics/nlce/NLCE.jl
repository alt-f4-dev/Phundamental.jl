# Generic numerical linked-cluster expansion built on the existing finite-reference cluster machinery.

"""Abstract expansion geometry used by finite-reference and topological numerical linked-cluster reconstructions."""
abstract type AbstractClusterExpansion end

"""Finite-reference embedded-cluster expansion preserving the historical linked-cluster semantics."""
struct FiniteReferenceExpansion <: AbstractClusterExpansion
    reference_sites::Int
    function FiniteReferenceExpansion(reference_sites::Integer)
        reference_sites > 0 || throw(ArgumentError("reference_sites must be positive"))
        new(Int(reference_sites))
    end
end

"""
    TopologicalClusterClass(key, order, multiplicity; subcluster_multiplicities=Dict(), metadata=Dict())

One linked-cluster equivalence class. `multiplicity` is the number of embeddings per lattice site. `subcluster_multiplicities[s]` is the number of embeddings of proper connected class `s` inside one representative of this class, which is the multiplicity required by topological NLCE inclusion-exclusion.
"""
struct TopologicalClusterClass{K,S,T}
    key::K
    order::Int
    multiplicity::Float64
    subcluster_multiplicities::S
    metadata::T
    function TopologicalClusterClass(key, order::Integer, multiplicity::Real; subcluster_multiplicities=Dict{Any,Int}(), metadata=Dict{Symbol,Any}())
        order > 0 || throw(ArgumentError("cluster order must be positive"))
        multiplicity >= 0 || throw(ArgumentError("cluster multiplicity must be nonnegative"))
        subclusters = Dict{Any,Int}(subcluster_multiplicities)
        all(>(0), values(subclusters)) || throw(ArgumentError("subcluster multiplicities must be positive"))
        haskey(subclusters, key) && throw(ArgumentError("a cluster cannot contain itself as a proper subcluster"))
        new{typeof(key),typeof(subclusters),typeof(metadata)}(key, Int(order), Float64(multiplicity), subclusters, metadata)
    end
end

"""Thermodynamic-limit topology-based expansion using one representative per cluster equivalence class."""
struct TopologicalExpansion{C,M} <: AbstractClusterExpansion
    clusters::C
    metadata::M
    function TopologicalExpansion(clusters; metadata=Dict{Symbol,Any}())
        classes = collect(clusters)
        isempty(classes) && throw(ArgumentError("a topological NLCE requires at least one cluster class"))
        all(c -> c isa TopologicalClusterClass, classes) || throw(ArgumentError("TopologicalExpansion requires TopologicalClusterClass entries"))
        cluster_keys = [c.key for c in classes]
        length(unique(cluster_keys)) == length(cluster_keys) || throw(ArgumentError("topological cluster keys must be unique"))
        order_by_key = Dict(c.key => c.order for c in classes)
        for cluster in classes, subkey in Base.keys(cluster.subcluster_multiplicities)
            haskey(order_by_key, subkey) || throw(KeyError("topological cluster $(cluster.key) references unknown subcluster class $subkey"))
            order_by_key[subkey] < cluster.order || throw(ArgumentError("subcluster class $subkey must have lower order than $(cluster.key)"))
        end
        new{typeof(classes),typeof(metadata)}(classes, metadata)
    end
end

TopologicalExpansion(clusters, metadata::AbstractDict) = TopologicalExpansion(clusters; metadata=Dict{Symbol,Any}(metadata))

"""Generic NLCE problem for the existing material/reference/model construction pipeline."""
struct NLCEProblem{M,R,S,E,B,T,MB,F}
    material::M
    reference::R
    specification::S
    evaluator::E
    max_order::Int
    bonds::B
    min_order::Int
    max_clusters::Int
    model_transform::T
    model_builder::MB
    allow_periodic_reference::Bool
    embedding_factor::F
end

function NLCEProblem(material::Material, reference::Supercell, specification::AbstractHamiltonianSpec, evaluator;
                     max_order::Integer, bonds=nothing, min_order::Integer=1, max_clusters::Integer=100_000,
                     model_transform=nothing, model_builder=nothing, allow_periodic_reference::Bool=false, embedding_factor=nothing)
    1 <= min_order <= max_order || throw(ArgumentError("require 1 <= min_order <= max_order"))
    max_clusters > 0 || throw(ArgumentError("max_clusters must be positive"))
    return NLCEProblem(material, reference, specification, evaluator, Int(max_order), bonds, Int(min_order), Int(max_clusters),
                       model_transform, model_builder, allow_periodic_reference, embedding_factor)
end

"""Solver descriptor for finite-reference NLCE reconstruction."""
struct NLCESolver <: AbstractSolver
    normalize_per_site::Bool
end

NLCESolver(; normalize_per_site::Bool=true) = NLCESolver(normalize_per_site)
solver_name(::NLCESolver) = :nlce

"""Order-resolved result for a generic finite-reference numerical linked-cluster expansion."""
struct NLCEResult{C,R,W,P,D}
    clusters::C
    raw_values::R
    weights::W
    partial_sums::P
    convergence::Vector{Float64}
    metadata::D
end


_linked_copy(value::Number) = value
_linked_copy(value::AbstractArray) = copy(value)
_linked_copy(value) = deepcopy(value)

_linked_subtract(a::Number, b::Number) = a - b
function _linked_subtract(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("linked-cluster values have incompatible array shapes"))
    return a .- b
end
_linked_subtract(a, b) = a - b

_linked_add(a::Number, b::Number) = a + b
function _linked_add(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("linked-cluster values have incompatible array shapes"))
    return a .+ b
end
_linked_add(a, b) = a + b

_linked_scale(value::Number, factor::Real) = factor * value
_linked_scale(value::AbstractArray, factor::Real) = factor .* value
_linked_scale(value, factor::Real) = factor * value

_linked_distance(a::Number, b::Number) = Float64(abs(a - b))
function _linked_distance(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("linked-cluster values have incompatible array shapes"))
    return Float64(norm(a .- b))
end
_linked_distance(a, b) = Float64(norm(a - b))

_cluster_key(cluster::ThermalCluster) = cluster.key
_cluster_order(cluster::ThermalCluster) = cluster.order
_cluster_key(cluster::TopologicalClusterClass) = cluster.key
_cluster_order(cluster::TopologicalClusterClass) = cluster.order

function _linked_subset_keys(key)
    key isa Tuple{Vararg{Int}} || throw(ArgumentError("automatic subset subtraction currently requires Tuple{Vararg{Int}} cluster keys"))
    return _proper_subset_keys(key)
end

"""
    linked_cluster_values(raw_values, clusters)

Generic inclusion-exclusion over scalar or array-like cluster values. The raw values and cluster keys must describe the same embedded connected-subcluster hierarchy used by `connected_clusters`.
"""
function linked_cluster_values(raw_values::AbstractDict, clusters)
    sorted_clusters = sort(collect(clusters), by=c -> (_cluster_order(c), _cluster_key(c)))
    weights = Dict{Any,Any}()
    for cluster in sorted_clusters
        key = _cluster_key(cluster)
        haskey(raw_values, key) || throw(KeyError("missing raw linked-cluster value for cluster $key"))
        weight = _linked_copy(raw_values[key])
        for subkey in _linked_subset_keys(key)
            haskey(weights, subkey) || continue
            weight = _linked_subtract(weight, weights[subkey])
        end
        weights[key] = weight
    end
    return weights
end

function _finite_reference_partial_sums(weights, clusters, max_order::Int, reference_sites::Int, normalize_per_site::Bool; embedding_factor=nothing)
    partial = Vector{Any}(undef, max_order)
    current = nothing
    normalization = normalize_per_site ? inv(reference_sites) : 1.0
    for order in 1:max_order
        for cluster in clusters
            _cluster_order(cluster) == order || continue
            factor = isnothing(embedding_factor) ? 1.0 : embedding_factor(cluster)
            isfinite(factor) && factor > 0 || throw(ArgumentError("NLCE embedding factors must be finite and positive"))
            term = _linked_scale(weights[_cluster_key(cluster)], normalization * factor)
            current = isnothing(current) ? _linked_copy(term) : _linked_add(current, term)
        end
        isnothing(current) && throw(ArgumentError("no linked-cluster contribution exists through order $order"))
        partial[order] = _linked_copy(current)
    end
    return partial
end

function _topological_linked_weights(raw_values::AbstractDict, expansion::TopologicalExpansion)
    sorted_clusters = sort(expansion.clusters, by=c -> (c.order, string(c.key)))
    weights = Dict{Any,Any}()
    for cluster in sorted_clusters
        haskey(raw_values, cluster.key) || throw(KeyError("missing raw topological NLCE value for cluster $(cluster.key)"))
        weight = _linked_copy(raw_values[cluster.key])
        for (subkey, multiplicity) in cluster.subcluster_multiplicities
            haskey(weights, subkey) || throw(KeyError("topological cluster $(cluster.key) references unavailable lower-order class $subkey"))
            weight = _linked_subtract(weight, _linked_scale(weights[subkey], multiplicity))
        end
        weights[cluster.key] = weight
    end
    return weights
end

function _topological_partial_sums(weights, expansion::TopologicalExpansion)
    max_order = maximum(c.order for c in expansion.clusters)
    partial = Vector{Any}(undef, max_order)
    current = nothing
    for order in 1:max_order
        for cluster in expansion.clusters
            cluster.order == order || continue
            term = _linked_scale(weights[cluster.key], cluster.multiplicity)
            current = isnothing(current) ? _linked_copy(term) : _linked_add(current, term)
        end
        isnothing(current) && throw(ArgumentError("no topological NLCE contribution exists through order $order"))
        partial[order] = _linked_copy(current)
    end
    return partial
end

function _order_convergence(partial)
    out = fill(NaN, length(partial))
    for order in 2:length(partial)
        out[order] = _linked_distance(partial[order], partial[order - 1])
    end
    return out
end

function _nlce_reference_bonds(problem::NLCEProblem)
    if !isnothing(problem.bonds) && !isnothing(problem.model_builder)
        bonds = collect(problem.bonds)
        all(bond -> bond isa Bond, bonds) || throw(ArgumentError("explicit NLCE reference bonds must be Bond objects"))
        return bonds
    end
    reference_model = isnothing(problem.bonds) ? build_model(problem.material, problem.reference, problem.specification) :
                      build_model(problem.material, problem.reference, problem.specification; bonds=collect(problem.bonds))
    ref_bonds = isnothing(problem.bonds) ? get(reference_model.parameters, :bonds, nothing) : collect(problem.bonds)
    isnothing(ref_bonds) && throw(ArgumentError("Hamiltonian builder did not expose `parameters[:bonds]`; supply bonds explicitly"))
    return ref_bonds
end

function _nlce_builder(problem::NLCEProblem)
    isnothing(problem.model_builder) && return (mat, cl, sp, bonds) -> build_model(mat, cl, sp; bonds=bonds)
    return problem.model_builder
end

"""
    solve(solver::NLCESolver, problem::NLCEProblem)

Evaluate a generic cluster quantity on every connected embedded cluster, recursively subtract proper connected-subcluster weights, and return order-resolved finite-reference linked-cluster sums. The evaluator is called as `evaluator(model, cluster, cluster_index)`.
"""
function solve(solver::NLCESolver, problem::NLCEProblem)
    problem.min_order == 1 || throw(ArgumentError("linked-cluster inclusion-exclusion requires min_order=1"))
    ref_bonds = _nlce_reference_bonds(problem)
    clusters = connected_clusters(problem.reference, ref_bonds; max_order=problem.max_order, min_order=problem.min_order,
                                  max_clusters=problem.max_clusters, allow_periodic_reference=problem.allow_periodic_reference)
    builder = _nlce_builder(problem)
    raw = Dict{Any,Any}()
    for (index, cluster) in enumerate(clusters)
        model = builder(problem.material, cluster.supercell, problem.specification, cluster.bonds)
        model isa ManyBodyModel || throw(ArgumentError("model_builder must return ManyBodyModel"))
        working = _apply_cluster_model_transform(model, problem.model_transform)
        raw[cluster.key] = problem.evaluator(working, cluster, index)
    end
    weights = linked_cluster_values(raw, clusters)
    partial = _finite_reference_partial_sums(weights, clusters, problem.max_order, nsites(problem.reference), solver.normalize_per_site; embedding_factor=problem.embedding_factor)
    convergence = _order_convergence(partial)
    metadata = Dict{Symbol,Any}(
        :method => :finite_reference_nlce,
        :correlation_scope => :short_intermediate_spatial,
        :max_order => problem.max_order,
        :cluster_count => length(clusters),
        :reference_sites => nsites(problem.reference),
        :per_site_normalization => solver.normalize_per_site,
        :infinite_lattice_claim => !isnothing(problem.embedding_factor),
        :translation_embedding_corrected => !isnothing(problem.embedding_factor),
        :interpretation => isnothing(problem.embedding_factor) ? :convergence_before_reference_boundary_required : :certified_periodic_translations_through_max_order,
    )
    return NLCEResult(clusters, raw, weights, partial, convergence, metadata)
end

"""Construct an order-resolved thermodynamic-limit NLCE result from pre-evaluated topological cluster classes."""
function topological_nlce(raw_values::AbstractDict, expansion::TopologicalExpansion)
    weights = _topological_linked_weights(raw_values, expansion)
    partial = _topological_partial_sums(weights, expansion)
    metadata = Dict{Symbol,Any}(
        :method => :topological_nlce,
        :correlation_scope => :short_intermediate_spatial,
        :max_order => length(partial),
        :cluster_count => length(expansion.clusters),
        :per_site_normalization => true,
        :infinite_lattice_claim => get(expansion.metadata, :infinite_lattice_claim, true),
        :multiplicities_supplied => true,
        :assumption => get(expansion.metadata, :infinite_lattice_claim, true) ? :complete_cluster_catalog_through_max_order : :embedding_completeness_not_asserted,
        :expansion_metadata => copy(expansion.metadata),
    )
    return NLCEResult(expansion.clusters, Dict(raw_values), weights, partial, _order_convergence(partial), metadata)
end

"""Highest-order NLCE estimate."""
bulk_estimate(result::NLCEResult) = result.partial_sums[end]

"""NLCE estimate through a specified expansion order."""
function bulk_estimate(result::NLCEResult, order::Integer)
    1 <= order <= length(result.partial_sums) || throw(BoundsError(result.partial_sums, order))
    return result.partial_sums[Int(order)]
end

"""Bare order-to-order NLCE convergence metric."""
nlce_convergence(result::NLCEResult) = copy(result.convergence)

"""Convenience constructor for an equal-temperature cluster expectation-value NLCE evaluator."""
function expectation_nlce_problem(material::Material, reference::Supercell, specification::AbstractHamiltonianSpec, observable;
                                  max_order::Integer, temperature::Real, method::AbstractThermalMethod=ExactGibbs(), kB::Real=1.0, kwargs...)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    evaluator = function (model, _cluster, index)
        local_method = _cluster_method(method, index)
        state = thermal_state(model, local_method; temperature=temperature, kB=kB)
        return expectation(state, observable).value
    end
    return NLCEProblem(material, reference, specification, evaluator; max_order=max_order, kwargs...)
end

_linked_norm(value::Number) = Float64(abs(value))
_linked_norm(value::AbstractArray) = Float64(norm(value))
_linked_norm(value) = Float64(norm(value))

function _nlce_reconstructed_raw(result::NLCEResult, cluster::ThermalCluster)
    value = _linked_copy(result.weights[cluster.key])
    for subkey in _proper_subset_keys(cluster.key)
        haskey(result.weights, subkey) || continue
        value = _linked_add(value, result.weights[subkey])
    end
    return value
end

function _nlce_reconstructed_raw(result::NLCEResult, cluster::TopologicalClusterClass)
    value = _linked_copy(result.weights[cluster.key])
    for (subkey, multiplicity) in cluster.subcluster_multiplicities
        value = _linked_add(value, _linked_scale(result.weights[subkey], multiplicity))
    end
    return value
end

"""Validate linked-cluster closure and result dimensions without assuming a specific physical observable."""
function validate_nlce_result(result::NLCEResult; atol::Real=1e-10, rtol::Real=1e-8)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    isempty(result.clusters) && return (valid=false, max_closure_residual=Inf, checked_clusters=0)
    residual_max = 0.0
    for cluster in result.clusters
        key = _cluster_key(cluster)
        haskey(result.raw_values, key) || return (valid=false, max_closure_residual=Inf, checked_clusters=0)
        haskey(result.weights, key) || return (valid=false, max_closure_residual=Inf, checked_clusters=0)
        reconstructed = _nlce_reconstructed_raw(result, cluster)
        raw = result.raw_values[key]
        residual = _linked_distance(reconstructed, raw)
        residual_max = max(residual_max, residual)
        residual <= atol + rtol * max(_linked_norm(raw), 1.0) || return (valid=false, max_closure_residual=residual_max, checked_clusters=length(result.clusters))
    end
    dimensions_ok = length(result.partial_sums) == length(result.convergence)
    return (valid=dimensions_ok, max_closure_residual=residual_max, checked_clusters=length(result.clusters))
end


include("GraphNLCE.jl")
include("Resummation.jl")
include("Benchmarking.jl")
