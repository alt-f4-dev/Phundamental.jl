# Batch 12: immutable structural forward request and repeat-evaluation boundary.
# No caches or mutable solver workspaces are shared between parameter points.

"""
    PreparedStructuralForward

Bound structural hypothesis, representation, observable, and numerical route. A prepared request does not cache a model or promise computational compatibility; each parameter point is validated by the authoritative Julia schema and forward executor.
"""
struct PreparedStructuralForward{S,R,C}
    selection::S
    representation::R
    configuration::C
end

"""
    prepare_forward(selection, representation; kwargs...)

Bind a finite structural selection to an explicit compatible representation and forward route. Supply an `AbstractObservable` for request-aware planning with `policy`; otherwise the explicit solver is used. Numerical truncation remains part of `representation`, never part of physical structural identity.
"""
function prepare_forward(selection::StructuralSelection, representation::Representation;
                         transformation=nothing, solver::AbstractSolver=ExactDiagonalization(), solver_args::Tuple=(),
                         solver_kwargs::NamedTuple=(;), observable=nothing, probe=nothing, resolution=nothing,
                         policy=nothing, routes=(), method_context=nothing,
                         observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}())
    validate_structural_representation(selection, representation)
    isnothing(selection.family.parameter_space) && throw(ArgumentError("structural forward evaluation requires a parent ParameterSpace"))
    !isnothing(method_context) && throw(ArgumentError("method-specific contexts are configured via typed request-aware forward planning, not the direct prepared interface"))
    !isnothing(policy) && !(policy isa ForwardExecutionPolicy) &&
        throw(ArgumentError("policy must be a ForwardExecutionPolicy or nothing"))
    !isnothing(policy) && !(observable isa Observables.AbstractObservable) &&
        throw(ArgumentError("a forward execution policy requires a typed AbstractObservable request"))
    !isnothing(probe) && isnothing(observable) && throw(ArgumentError("probe requires an observable"))
    !isnothing(resolution) && isnothing(observable) && throw(ArgumentError("resolution requires an observable"))
    !isnothing(policy) && !isnothing(transformation) &&
        throw(ArgumentError("request-aware planning owns the transformation; provide a ForwardRoute instead"))
    config = (transformation=transformation, solver=solver, solver_args=solver_args, solver_kwargs=solver_kwargs,
              observable=observable, probe=probe, resolution=resolution, policy=policy, routes=Tuple(routes),
              observables=Dict{Symbol,Any}(observables), metadata_parameters=Dict{Symbol,Any}(metadata_parameters))
    return PreparedStructuralForward(selection, representation, config)
end

"""
    forward_parameters(prepared, values)

Evaluate one parameter point. Structural identity and omitted parent coefficients remain fixed; new `ParameterSpace`, `ManyBodyModel`, and solver objects are constructed through the established APIs. No mutable result state is reused.
"""
function forward_parameters(prepared::PreparedStructuralForward, values)
    config = prepared.configuration
    model = instantiate_model(prepared.selection, prepared.representation, values;
                              observables=copy(config.observables), metadata_parameters=copy(config.metadata_parameters))
    if isnothing(config.policy)
        if isnothing(config.transformation) && config.observable isa Observables.AbstractObservable
            support = solver_compatibility(config.solver, model, config.observable)
            support.compatible || throw(ArgumentError("selected solver does not support the requested observable: $(join(support.reasons, "; "))"))
        end
        problem = ForwardProblem(model; transformation=config.transformation, solver=config.solver, solver_args=config.solver_args,
                                 solver_kwargs=config.solver_kwargs, observable=config.observable, probe=config.probe, resolution=config.resolution,
                                 metadata=Dict{Symbol,Any}(:structural_family_id => structural_id(prepared.selection.family),
                                                           :structural_selection_id => structural_id(prepared.selection)))
        return forward(problem)
    end
    return forward(model, config.observable; policy=config.policy, routes=config.routes, probe=config.probe, resolution=config.resolution,
                   metadata=Dict{Symbol,Any}(:structural_family_id => structural_id(prepared.selection.family),
                                             :structural_selection_id => structural_id(prepared.selection)))
end

"""Evaluate an ordered batch without structural caching or hidden parameter-schema changes; reject an empty batch."""
function forward_batch(prepared::PreparedStructuralForward, batch)
    entries = collect(batch)
    isempty(entries) && throw(ArgumentError("forward batch must contain at least one parameter point"))
    return [forward_parameters(prepared, values) for values in entries]
end

"""Convenience structural forward call with all physical operations delegated to `prepare_forward` and `forward_parameters`."""
function forward(selection::StructuralSelection, representation::Representation, values; kwargs...)
    return forward_parameters(prepare_forward(selection, representation; kwargs...), values)
end

"""Produce a neutral stage inventory without requiring Python to deserialize Julia solver results or physical operators."""
function Models.describe(result::ForwardResult)
    selected = Models._structural_provenance(result.source_model)
    stage(value) = isnothing(value) ? nothing : string(nameof(typeof(value)))
    record = Dict{String,Any}("schema_version" => Models.STRUCTURAL_SCHEMA_VERSION, "kind" => "forward_result",
                              "source_model" => describe(result.source_model), "working_model" => describe(result.model),
                              "transformation" => stage(result.transformation_result), "solution" => stage(result.solution),
                              "intrinsic_observable" => stage(result.observable_result), "probe_response" => stage(result.scattering_result),
                              "instrument_result" => stage(result.result),
                              "family_id" => isnothing(selected) ? nothing : selected.family_id,
                              "selection_id" => isnothing(selected) ? nothing : selected.selection_id,
                              "approximation_or_truncation" => isnothing(result.model.representation.truncation) ? nothing :
                                  string(nameof(typeof(result.model.representation.truncation))))
    return record
end

"""Expose a prepared Julia forward handle as a neutral, non-executable interface description."""
function Models.describe(prepared::PreparedStructuralForward)
    config = prepared.configuration
    return Dict{String,Any}("schema_version" => Models.STRUCTURAL_SCHEMA_VERSION, "kind" => "prepared_structural_forward",
                            "family_id" => structural_id(prepared.selection.family), "selection_id" => structural_id(prepared.selection),
                            "representation" => String(prepared.representation.name),
                            "representation_signature" => Models._structural_digest("representation", PhundamentalCore.representation_signature(prepared.representation)),
                            "numerical_truncation" => isnothing(prepared.representation.truncation) ? nothing : string(prepared.representation.truncation),
                            "solver" => string(nameof(typeof(config.solver))),
                            "observable" => isnothing(config.observable) ? nothing : string(nameof(typeof(config.observable))),
                            "probe" => isnothing(config.probe) ? nothing : string(nameof(typeof(config.probe))),
                            "resolution" => isnothing(config.resolution) ? nothing : string(nameof(typeof(config.resolution))),
                            "prepared_cache" => false)
end
