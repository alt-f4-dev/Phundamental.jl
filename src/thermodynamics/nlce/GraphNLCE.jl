# Automatic site-based topological NLCE construction from embedded connected clusters.

"""Colored multigraph representation used internally for exact topological-cluster equivalence tests."""
struct NLCEClusterGraph
    vertex_labels::Vector{Any}
    edge_labels::Matrix{Any}
end

Base.length(graph::NLCEClusterGraph) = length(graph.vertex_labels)

function _default_nlce_vertex_label(site)
    return (site.basis_index, site.species)
end

function _default_nlce_edge_label(bond)
    return (bond.shell, round(Float64(bond.distance); sigdigits=12))
end

function _cluster_graph(cluster::ThermalCluster; vertex_label=_default_nlce_vertex_label, edge_label=_default_nlce_edge_label)
    n = cluster.order
    labels = Any[vertex_label(site) for site in cluster.supercell.sites]
    edges = Matrix{Any}(undef, n, n)
    fill!(edges, nothing)
    grouped = Dict{Tuple{Int,Int},Vector{Any}}()
    for bond in cluster.bonds
        bond.i == bond.j && throw(ArgumentError("site-based topological NLCE does not support self-image bonds inside an open cluster representative"))
        i, j = minmax(bond.i, bond.j)
        push!(get!(grouped, (i, j), Any[]), edge_label(bond))
    end
    for ((i, j), values) in grouped
        canonical = Tuple(sort!(values; by=string))
        edges[i, j] = canonical
        edges[j, i] = canonical
    end
    return NLCEClusterGraph(labels, edges)
end

function _induced_graph(graph::NLCEClusterGraph, vertices::AbstractVector{<:Integer})
    selected = Int.(collect(vertices))
    labels = graph.vertex_labels[selected]
    edges = graph.edge_labels[selected, selected]
    return NLCEClusterGraph(Any[labels...], Matrix{Any}(edges))
end

_graph_edge(graph::NLCEClusterGraph, i::Int, j::Int) = graph.edge_labels[i, j]
_graph_degree(graph::NLCEClusterGraph, i::Int) = count(j -> j != i && !isnothing(_graph_edge(graph, i, j)), 1:length(graph))

function _graph_local_signature(graph::NLCEClusterGraph, i::Int)
    incident = Any[]
    for j in 1:length(graph)
        j == i && continue
        edge = _graph_edge(graph, i, j)
        isnothing(edge) || push!(incident, edge)
    end
    sort!(incident; by=string)
    return (graph.vertex_labels[i], length(incident), Tuple(incident))
end

function _graph_invariant(graph::NLCEClusterGraph)
    locals = [_graph_local_signature(graph, i) for i in 1:length(graph)]
    sort!(locals; by=string)
    edges = Any[]
    for i in 1:length(graph)-1, j in i+1:length(graph)
        edge = _graph_edge(graph, i, j)
        isnothing(edge) || push!(edges, edge)
    end
    sort!(edges; by=string)
    return (length(graph), Tuple(sort(copy(graph.vertex_labels); by=string)), Tuple(locals), Tuple(edges))
end

function _graph_connected(graph::NLCEClusterGraph)
    n = length(graph)
    n <= 1 && return true
    visited = falses(n)
    stack = Int[1]
    visited[1] = true
    while !isempty(stack)
        i = pop!(stack)
        for j in 1:n
            j == i && continue
            isnothing(_graph_edge(graph, i, j)) && continue
            visited[j] && continue
            visited[j] = true
            push!(stack, j)
        end
    end
    return all(visited)
end

function _graph_isomorphic(a::NLCEClusterGraph, b::NLCEClusterGraph)
    length(a) == length(b) || return false
    _graph_invariant(a) == _graph_invariant(b) || return false
    n = length(a)
    source_signatures = [_graph_local_signature(a, i) for i in 1:n]
    target_signatures = [_graph_local_signature(b, i) for i in 1:n]
    candidates = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        candidates[i] = Int[j for j in 1:n if target_signatures[j] == source_signatures[i]]
        isempty(candidates[i]) && return false
    end
    order = sortperm(1:n; by=i -> (length(candidates[i]), -_graph_degree(a, i), string(source_signatures[i])))
    mapping = zeros(Int, n)
    used = falses(n)

    function consistent(source::Int, target::Int)
        for other in 1:n
            mapped = mapping[other]
            iszero(mapped) && continue
            _graph_edge(a, source, other) == _graph_edge(b, target, mapped) || return false
        end
        return true
    end

    function assign!(position::Int)
        position > n && return true
        source = order[position]
        for target in candidates[source]
            used[target] && continue
            consistent(source, target) || continue
            mapping[source] = target
            used[target] = true
            assign!(position + 1) && return true
            mapping[source] = 0
            used[target] = false
        end
        return false
    end

    return assign!(1)
end

function _connected_vertex_subsets(graph::NLCEClusterGraph)
    n = length(graph)
    n <= 1 && return Vector{Vector{Int}}()
    n <= 62 || throw(ArgumentError("exact subcluster enumeration currently supports at most 62 vertices"))
    subsets = Vector{Vector{Int}}()
    fullmask = (UInt64(1) << n) - UInt64(1)
    for mask in UInt64(1):(fullmask - UInt64(1))
        vertices = Int[i for i in 1:n if !iszero(mask & (UInt64(1) << (i - 1)))]
        _graph_connected(_induced_graph(graph, vertices)) || continue
        push!(subsets, vertices)
    end
    return subsets
end

function _find_graph_class(graph::NLCEClusterGraph, classes)
    invariant = _graph_invariant(graph)
    for (index, entry) in enumerate(classes)
        entry.invariant == invariant || continue
        _graph_isomorphic(graph, entry.graph) && return index
    end
    return nothing
end

"""
    topological_expansion(clusters, reference_sites; vertex_label=_default_nlce_vertex_label, edge_label=_default_nlce_edge_label, metadata=Dict())

Group embedded connected site clusters into exact colored-graph isomorphism classes, derive proper connected-subcluster multiplicities from one representative per class, and convert embedded-cluster counts into lattice constants per reference site. For a periodic reference that is large enough to contain every retained cluster without topological wrapping, the resulting multiplicities are the site-based NLCE lattice constants.
"""
function topological_expansion(clusters, reference_sites::Integer; vertex_label=_default_nlce_vertex_label,
                               edge_label=_default_nlce_edge_label, metadata=Dict{Symbol,Any}())
    reference_sites > 0 || throw(ArgumentError("reference_sites must be positive"))
    embedded = sort(collect(clusters); by=c -> (c.order, c.key))
    isempty(embedded) && throw(ArgumentError("topological expansion requires at least one embedded cluster"))
    groups = NamedTuple[]
    class_of_key = Dict{Any,Int}()

    for cluster in embedded
        graph = _cluster_graph(cluster; vertex_label=vertex_label, edge_label=edge_label)
        invariant = _graph_invariant(graph)
        same_order = findall(entry -> entry.order == cluster.order && entry.invariant == invariant, groups)
        class_index = nothing
        for index in same_order
            if _graph_isomorphic(graph, groups[index].graph)
                class_index = index
                break
            end
        end
        if isnothing(class_index)
            push!(groups, (order=cluster.order, graph=graph, invariant=invariant, representative=cluster, members=ThermalCluster[cluster]))
            class_index = length(groups)
        else
            push!(groups[class_index].members, cluster)
        end
        class_of_key[cluster.key] = class_index
    end

    keys_by_group = [(:site_graph, groups[index].order, index) for index in eachindex(groups)]
    classes = TopologicalClusterClass[]
    for (index, group) in enumerate(groups)
        submultiplicities = Dict{Any,Int}()
        for vertices in _connected_vertex_subsets(group.graph)
            subgraph = _induced_graph(group.graph, vertices)
            candidate_indices = findall(entry -> entry.order == length(vertices) && entry.invariant == _graph_invariant(subgraph), groups)
            match = nothing
            for candidate in candidate_indices
                if _graph_isomorphic(subgraph, groups[candidate].graph)
                    match = candidate
                    break
                end
            end
            isnothing(match) && throw(ArgumentError("proper connected subcluster of topological class $(keys_by_group[index]) is absent from the cluster catalog"))
            subkey = keys_by_group[match]
            submultiplicities[subkey] = get(submultiplicities, subkey, 0) + 1
        end
        info = Dict{Symbol,Any}(
            :representative => group.representative,
            :embedding_count => length(group.members),
            :graph => group.graph,
            :graph_invariant => group.invariant,
            :embedded_keys => [member.key for member in group.members],
        )
        merge!(info, metadata)
        push!(classes, TopologicalClusterClass(keys_by_group[index], group.order, length(group.members) / reference_sites;
                                               subcluster_multiplicities=submultiplicities, metadata=info))
    end

    expansion_metadata = Dict{Symbol,Any}(
        :construction => :embedded_site_graph_classification,
        :reference_sites => Int(reference_sites),
        :embedded_cluster_count => length(embedded),
        :topological_cluster_count => length(classes),
        :lattice_constants_from_embedding_counts => true,
        :infinite_lattice_claim => false,
    )
    merge!(expansion_metadata, metadata)
    return TopologicalExpansion(classes; metadata=expansion_metadata)
end

"""
    site_nlce_expansion(reference, bonds; max_order, max_clusters=1_000_000, allow_periodic_reference=true, assume_embedding_complete=false, ...)

Construct a site-based topological expansion from a finite reference supercell. A periodic reference is normally preferred because embedding counts then converge rapidly to the thermodynamic lattice constants. `assume_embedding_complete=true` records an explicit user assertion; it does not bypass any graph or subcluster checks.
"""
function site_nlce_expansion(reference::Supercell, bonds::AbstractVector{<:Bond}; max_order::Integer, max_clusters::Integer=1_000_000,
                             allow_periodic_reference::Bool=true, assume_embedding_complete::Bool=false,
                             vertex_label=_default_nlce_vertex_label, edge_label=_default_nlce_edge_label)
    max_order >= 1 || throw(ArgumentError("max_order must be positive"))
    clusters = connected_clusters(reference, bonds; max_order=max_order, min_order=1, max_clusters=max_clusters,
                                  allow_periodic_reference=allow_periodic_reference)
    metadata = Dict{Symbol,Any}(
        :reference_periodic => collect(reference.periodic),
        :max_order => Int(max_order),
        :embedding_complete_asserted => assume_embedding_complete,
        :infinite_lattice_claim => assume_embedding_complete,
    )
    expansion = topological_expansion(clusters, nsites(reference); vertex_label=vertex_label, edge_label=edge_label, metadata=metadata)
    return expansion
end

"""Generic end-to-end topological NLCE problem evaluated once per graph-equivalence class."""
struct TopologicalNLCEProblem{M,S,E,F,T,MB}
    material::M
    specification::S
    expansion::E
    evaluator::F
    model_transform::T
    model_builder::MB
end

function TopologicalNLCEProblem(material::Material, specification::AbstractHamiltonianSpec, expansion::TopologicalExpansion, evaluator;
                                model_transform=nothing, model_builder=nothing)
    return TopologicalNLCEProblem(material, specification, expansion, evaluator, model_transform, model_builder)
end

function _topological_nlce_builder(problem::TopologicalNLCEProblem)
    isnothing(problem.model_builder) && return (material, cluster, specification, bonds) -> build_model(material, cluster, specification; bonds=bonds)
    return problem.model_builder
end

function solve(solver::NLCESolver, problem::TopologicalNLCEProblem)
    solver.normalize_per_site || throw(ArgumentError("topological NLCE multiplicities are defined per lattice site; normalize_per_site=false is not supported"))
    builder = _topological_nlce_builder(problem)
    raw = Dict{Any,Any}()
    for (index, cluster_class) in enumerate(sort(problem.expansion.clusters; by=c -> (c.order, string(c.key))))
        representative = get(cluster_class.metadata, :representative, nothing)
        representative isa ThermalCluster || throw(ArgumentError("topological cluster $(cluster_class.key) does not carry a ThermalCluster representative"))
        model = builder(problem.material, representative.supercell, problem.specification, representative.bonds)
        model isa ManyBodyModel || throw(ArgumentError("model_builder must return ManyBodyModel"))
        working = _apply_cluster_model_transform(model, problem.model_transform)
        raw[cluster_class.key] = problem.evaluator(working, representative, index)
    end
    result = topological_nlce(raw, problem.expansion)
    result.metadata[:automatic_topology] = true
    result.metadata[:expansion_metadata] = copy(problem.expansion.metadata)
    result.metadata[:infinite_lattice_claim] = get(problem.expansion.metadata, :infinite_lattice_claim, false)
    return result
end

function _match_topological_classes(reference::TopologicalExpansion, candidate::TopologicalExpansion)
    matches = Dict{Any,Any}()
    for source in reference.clusters
        source_graph = get(source.metadata, :graph, nothing)
        source_graph isa NLCEClusterGraph || continue
        candidates = [target for target in candidate.clusters if target.order == source.order]
        match = findfirst(target -> begin
            target_graph = get(target.metadata, :graph, nothing)
            target_graph isa NLCEClusterGraph && _graph_isomorphic(source_graph, target_graph)
        end, candidates)
        isnothing(match) || (matches[source.key] = candidates[match].key)
    end
    return matches
end

"""
    validate_embedding_stability(reference, candidate; atol=1e-12, rtol=1e-10)

Compare lattice constants for graph-isomorphic classes obtained from two independent embedding references. Equality through a retained order is the operational check required before a finite-reference graph catalog is promoted to an infinite-lattice NLCE catalog.
"""
function validate_embedding_stability(reference::TopologicalExpansion, candidate::TopologicalExpansion; atol::Real=1e-12, rtol::Real=1e-10)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    matches = _match_topological_classes(reference, candidate)
    missing = Any[]
    multiplicity_residuals = Dict{Any,Float64}()
    for source in reference.clusters
        if !haskey(matches, source.key)
            push!(missing, source.key)
            continue
        end
        target_key = matches[source.key]
        target = only(filter(c -> c.key == target_key, candidate.clusters))
        residual = abs(source.multiplicity - target.multiplicity)
        multiplicity_residuals[source.key] = residual
    end
    extra_count = length(candidate.clusters) - length(values(matches))
    stable = isempty(missing) && extra_count == 0 && all(source -> begin
        residual = multiplicity_residuals[source.key]
        target = only(filter(c -> c.key == matches[source.key], candidate.clusters))
        residual <= atol + rtol * max(abs(source.multiplicity), abs(target.multiplicity), 1.0)
    end, reference.clusters)
    return (stable=stable, missing_classes=missing, extra_class_count=extra_count,
            max_multiplicity_residual=isempty(multiplicity_residuals) ? Inf : maximum(values(multiplicity_residuals)), matches=matches)
end
