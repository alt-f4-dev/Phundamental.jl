# Representation graph over exact transformations and approximations.

"""
    TransformationEdge(source, target, map; ...)

Directed edge between representation nodes. `parameter_map` maps parameter
metadata/coordinates along the forward edge. `inverse_parameter_map` is
required for executable reverse traversal unless the forward map is `identity`.
"""
struct TransformationEdge{M,P,I}
    source::Symbol
    target::Symbol
    map::M
    exact::Bool
    invertible::Bool
    parameter_map::P
    inverse_parameter_map::I
    metadata::Dict{Symbol,Any}
end

function TransformationEdge(source::Symbol, target::Symbol, map; exact=nothing, invertible=nothing,
                            parameter_map=identity, inverse_parameter_map=nothing, metadata=Dict{Symbol,Any}())
    if map isa AbstractTransformation
        cert = certificate(map)
        edge_exact = isnothing(exact) ? cert.exact && !cert.approximate : Bool(exact)
        edge_invertible = isnothing(invertible) ? cert.invertible : Bool(invertible)
    elseif map isa AbstractApproximation
        edge_exact = isnothing(exact) ? false : Bool(exact)
        edge_invertible = isnothing(invertible) ? false : Bool(invertible)
    else
        throw(ArgumentError("graph edge map must be AbstractTransformation or AbstractApproximation"))
    end
    reverse_map = isnothing(inverse_parameter_map) && parameter_map === identity ? identity : inverse_parameter_map
    edge_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return TransformationEdge(source, target, map, edge_exact, edge_invertible, parameter_map, reverse_map, edge_metadata)
end

mutable struct RepresentationGraph
    representations::Dict{Symbol,Representation}
    edges::Vector{TransformationEdge}
end

RepresentationGraph() = RepresentationGraph(Dict{Symbol,Representation}(), TransformationEdge[])

function add_representation!(graph::RepresentationGraph, representation::Representation)
    graph.representations[representation.name] = representation
    return representation
end

function add_transformation!(graph::RepresentationGraph, source::Symbol, target::Symbol, map; kwargs...)
    haskey(graph.representations, source) || throw(KeyError("source representation $source is not registered"))
    haskey(graph.representations, target) || throw(KeyError("target representation $target is not registered"))
    edge = TransformationEdge(source, target, map; kwargs...)
    push!(graph.edges, edge)
    return edge
end

function outgoing_edges(graph::RepresentationGraph, source::Symbol; exact_only::Bool=false)
    return [edge for edge in graph.edges if edge.source === source && (!exact_only || edge.exact)]
end

function _reverse_edge(edge::TransformationEdge)
    edge.map isa AbstractTransformation || return nothing
    edge.invertible || return nothing
    has_inverse(edge.map) || return nothing
    isnothing(edge.inverse_parameter_map) && return nothing
    return TransformationEdge(edge.target, edge.source, inverse(edge.map); exact=edge.exact, invertible=true,
                              parameter_map=edge.inverse_parameter_map, inverse_parameter_map=edge.parameter_map,
                              metadata=merge(edge.metadata, Dict{Symbol,Any}(:reverse_of => (edge.source, edge.target))))
end

function _traversable_edges(graph::RepresentationGraph, source::Symbol; exact_only::Bool=false)
    edges = outgoing_edges(graph, source; exact_only=exact_only)
    for edge in graph.edges
        edge.target === source || continue
        reverse = _reverse_edge(edge)
        isnothing(reverse) && continue
        (!exact_only || reverse.exact) && push!(edges, reverse)
    end
    return edges
end

"""Breadth-first executable path of graph edges from `source` to `target`."""
function transformation_path(graph::RepresentationGraph, source::Symbol, target::Symbol; exact_only::Bool=false)
    source === target && return TransformationEdge[]
    haskey(graph.representations, source) || throw(KeyError(source))
    haskey(graph.representations, target) || throw(KeyError(target))
    queue = Symbol[source]
    visited = Set{Symbol}([source])
    parent = Dict{Symbol,Tuple{Symbol,TransformationEdge}}()
    while !isempty(queue)
        current = popfirst!(queue)
        for edge in _traversable_edges(graph, current; exact_only=exact_only)
            next = edge.target
            next in visited && continue
            push!(visited, next)
            parent[next] = (current, edge)
            if next === target
                path = TransformationEdge[]
                cursor = target
                while cursor !== source
                    previous, used = parent[cursor]
                    pushfirst!(path, used)
                    cursor = previous
                end
                return path
            end
            push!(queue, next)
        end
    end
    throw(ArgumentError("no executable transformation path exists from $source to $target"))
end

"""Exact mathematical representation orbit reachable through invertible exact edges."""
function representation_class(graph::RepresentationGraph, source::Symbol)
    haskey(graph.representations, source) || throw(KeyError(source))
    visited = Set{Symbol}([source])
    queue = Symbol[source]
    while !isempty(queue)
        current = popfirst!(queue)
        for edge in graph.edges
            if edge.exact && edge.invertible && edge.source === current && !(edge.target in visited)
                push!(visited, edge.target)
                push!(queue, edge.target)
            end
            if edge.exact && edge.invertible && edge.target === current && !(edge.source in visited)
                push!(visited, edge.source)
                push!(queue, edge.source)
            end
        end
    end
    return sort!(collect(visited))
end

function _map_edge_parameters(edge::TransformationEdge, model::ManyBodyModel)
    source = isnothing(model.parameterization) ? model.parameters : model.parameterization
    mapped = edge.parameter_map(source)
    mapped === source && return model
    if mapped isa ParameterPoint
        return with_parameters(model, mapped; rebuild=false)
    elseif mapped isa ParameterSpace
        return ManyBodyModel(model.representation, model.hamiltonian; parameters=model.parameters, observables=model.observables,
                             provenance=model.provenance, specification=model.specification, model_space=model.model_space,
                             parameterization=mapped)
    end
    return ManyBodyModel(model.representation, model.hamiltonian; parameters=mapped, observables=model.observables,
                         provenance=model.provenance, specification=model.specification, model_space=model.model_space,
                         parameterization=model.parameterization)
end

function transform_along(graph::RepresentationGraph, model::ManyBodyModel, target::Symbol; exact_only::Bool=false)
    path = transformation_path(graph, model.representation.name, target; exact_only=exact_only)
    current = model
    for edge in path
        if edge.map isa AbstractTransformation
            current = transform(edge.map, current).model
        else
            current = apply_approximation(edge.map, current).model
        end
        current = _map_edge_parameters(edge, current)
    end
    return current
end
