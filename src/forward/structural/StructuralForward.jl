# Batch 12: immutable structural forward request and repeat-evaluation boundary.
# No caches or mutable solver workspaces are shared between parameter points.

using LinearAlgebra: I

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

Bind a finite structural selection to an explicit compatible representation and forward route. An explicit `method_context` is forwarded unchanged to `ForwardProblem` and must accompany a typed observable request. With a `policy`, it becomes one explicit route; policy selection must choose that route, and alternative routes must instead carry their own contexts through `ForwardRoute`. Numerical truncation remains part of `representation`, never part of physical structural identity.
"""
function prepare_forward(selection::StructuralSelection, representation::Representation;
                         transformation=nothing, solver::AbstractSolver=ExactDiagonalization(), solver_args::Tuple=(),
                         solver_kwargs::NamedTuple=(;), observable=nothing, probe=nothing, resolution=nothing,
                         policy=nothing, routes=(), method_context=nothing,
                         observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}())
    validate_structural_representation(selection, representation)
    isnothing(selection.family.parameter_space) && throw(ArgumentError("structural forward evaluation requires a parent ParameterSpace"))
    !isnothing(method_context) && !(method_context isa AbstractForwardMethodContext) &&
        throw(ArgumentError("method_context must be AbstractForwardMethodContext or nothing"))
    !isnothing(method_context) && !(observable isa Observables.AbstractObservable) &&
        throw(ArgumentError("an explicit method_context requires a typed AbstractObservable request"))
    !isnothing(method_context) && !isnothing(policy) && !isempty(routes) &&
        throw(ArgumentError("supply either an explicit method_context or policy routes, not both; use ForwardRoute for alternative contexts"))
    !isnothing(policy) && !(policy isa ForwardExecutionPolicy) &&
        throw(ArgumentError("policy must be a ForwardExecutionPolicy or nothing"))
    !isnothing(policy) && !(observable isa Observables.AbstractObservable) &&
        throw(ArgumentError("a forward execution policy requires a typed AbstractObservable request"))
    !isnothing(probe) && isnothing(observable) && throw(ArgumentError("probe requires an observable"))
    !isnothing(resolution) && isnothing(observable) && throw(ArgumentError("resolution requires an observable"))
    !isnothing(policy) && !isnothing(transformation) &&
        throw(ArgumentError("request-aware planning owns the transformation; provide a ForwardRoute instead"))
    config = (transformation=transformation, solver=solver, solver_args=solver_args, solver_kwargs=solver_kwargs,
              method_context=method_context, observable=observable, probe=probe, resolution=resolution, policy=policy, routes=Tuple(routes),
              observables=Dict{Symbol,Any}(observables), metadata_parameters=Dict{Symbol,Any}(metadata_parameters))
    return PreparedStructuralForward(selection, representation, config)
end

"""
    forward_parameters(prepared, values)

Evaluate one parameter point. Structural identity and omitted parent coefficients remain fixed; new `ParameterSpace`, `ManyBodyModel`, and solver objects are constructed through the established APIs. No mutable result state is reused.
"""
function _forward_selected_parameters(prepared::PreparedStructuralForward, values)
    config = prepared.configuration
    model = instantiate_model(prepared.selection, prepared.representation, values;
                              observables=copy(config.observables), metadata_parameters=copy(config.metadata_parameters))
    if isnothing(config.policy)
        if isnothing(config.transformation) && config.observable isa Observables.AbstractObservable
            support = solver_compatibility(config.solver, model, config.observable, config.method_context)
            support.compatible || throw(ArgumentError("selected solver does not support the requested observable: $(join(support.reasons, "; "))"))
        end
        problem = ForwardProblem(model; transformation=config.transformation, solver=config.solver, solver_args=config.solver_args,
                                 solver_kwargs=config.solver_kwargs, method_context=config.method_context,
                                 observable=config.observable, probe=config.probe, resolution=config.resolution,
                                 metadata=Dict{Symbol,Any}(:structural_family_id => structural_id(prepared.selection.family),
                                                           :structural_selection_id => structural_id(prepared.selection)))
        return forward(problem)
    end
    metadata = Dict{Symbol,Any}(:structural_family_id => structural_id(prepared.selection.family),
                                :structural_selection_id => structural_id(prepared.selection))
    if isnothing(config.method_context)
        return forward(model, config.observable; policy=config.policy, routes=config.routes,
                       probe=config.probe, resolution=config.resolution, metadata=metadata)
    end
    explicit_route = ForwardRoute(config.solver; method_context=config.method_context, solver_args=config.solver_args,
                                  solver_kwargs=config.solver_kwargs, label=:prepared_explicit_context)
    plan = plan_forward(model, config.observable; policy=config.policy, routes=(explicit_route,),
                        probe=config.probe, resolution=config.resolution, metadata=metadata)
    plan.method_context === config.method_context || throw(ArgumentError(
        "the forward execution policy selected a different method than the explicitly prepared context; set include_default_exact_routes=false or supply policy routes instead"))
    return forward(plan)
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
                            "method_context" => isnothing(config.method_context) ? nothing : string(nameof(typeof(config.method_context))),
                            "observable" => isnothing(config.observable) ? nothing : string(nameof(typeof(config.observable))),
                            "probe" => isnothing(config.probe) ? nothing : string(nameof(typeof(config.probe))),
                            "resolution" => isnothing(config.resolution) ? nothing : string(nameof(typeof(config.resolution))),
                            "prepared_cache" => false)
end

# Higher-level spin scattering retains explicit experimental conventions and delegates to the same forward pathway.

"""Validated spin-scattering setup with registered local operators, Cartesian positions and experimental response settings."""
struct SpinScatteringProblem{R,Q,P,Z,O,X}
    representation::R
    request::Q
    probe::P
    resolution::Z
    observables::O
    positions::X
end

"""Construct local Cartesian spin operator keys and positions from a one-spin-index-per-site material/cluster."""
function spin_observables(material::Material, algebra::SpinAlgebra; cluster=nothing)
    crystal = isnothing(cluster) ? material_crystallographic_structure(material) :
        material_crystallographic_structure(material, cluster)
    algebra.nsites == length(crystal) || throw(DimensionMismatch("spin algebra must contain one index per crystallographic site"))
    positions = [collect(crystal.lattice * crystal.positions[:, i]) for i in 1:length(crystal)]
    operators = Dict{Symbol,Any}()
    for i in 1:algebra.nsites
        operators[Symbol("spin_x_", i)] = Sx(i)
        operators[Symbol("spin_y_", i)] = Sy(i)
        operators[Symbol("spin_z_", i)] = Sz(i)
    end
    return (; observables=operators, positions)
end

"""
    spin_scattering_problem(material, representation, Q, energy; kwargs...)

Prepare the existing spin-tensor observable, magnetic neutron probe and optional resolution with explicit normalization, temperature, broadening, and probe settings. This constructor does not infer ionic form factors, absolute cross sections, or internal spin rotations.
"""
function spin_scattering_problem(material::Material, representation::Representation, Q, energy;
                                 cluster=nothing, temperature::Real=0.0,
                                 convention::SpectrumConvention=SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy,
                                     hbar=1.0, kB=8.617333262e-2),
                                 broadening::Observables.AbstractBroadening=LorentzianBroadening(0.12),
                                 g_tensor=2.0 * Matrix{Float64}(I, 3, 3), form_factor=(Q -> 1.0),
                                 prefactor::Real=1.0, resolution=nothing, sites=nothing)
    representation.algebra isa SpinAlgebra || throw(ArgumentError("spin scattering requires a SpinAlgebra representation"))
    registered = spin_observables(material, representation.algebra; cluster=cluster)
    request = SpinTensorStructureFactorRequest(Q, energy; temperature=temperature, convention=convention,
                                                broadening=broadening, sites=sites)
    probe = MagneticNeutronProbe(g_tensor=g_tensor, form_factor=form_factor, prefactor=prefactor)
    !isnothing(resolution) && !(resolution isa Observables.AbstractResolution) &&
        throw(ArgumentError("instrument resolution must subtype AbstractResolution"))
    return SpinScatteringProblem(representation, request, probe, resolution, registered.observables, registered.positions)
end

"""Bind a structural selection to an explicitly configured spin-scattering calculation."""
function prepare_forward(selection::StructuralSelection, problem::SpinScatteringProblem; kwargs...)
    return prepare_forward(selection, problem.representation; observable=problem.request, probe=problem.probe,
                           resolution=problem.resolution, observables=problem.observables,
                           metadata_parameters=Dict{Symbol,Any}(:positions => problem.positions), kwargs...)
end

"""Evaluate selected coefficients or explicitly projected parent-domain coefficients (never silently discard omitted terms)."""
function forward_parameters(prepared::PreparedStructuralForward, values; parameter_domain::Symbol=:selected, strict::Bool=true)
    if parameter_domain === :selected
        return _forward_selected_parameters(prepared, values)
    elseif parameter_domain !== :parent
        throw(ArgumentError("parameter_domain must be :selected or :parent"))
    end
    selection = prepared.selection
    schema = selection.family.parameter_space
    isnothing(schema) && throw(ArgumentError("parent-domain forward evaluation requires a parent parameter schema"))
    vector = if values isa ParameterPoint
        values.space === schema || throw(ArgumentError("parent ParameterPoint belongs to a different parameter schema"))
        values.values
    elseif values isa AbstractVector{<:Real}
        collect(values)
    elseif values isa AbstractDict || values isa NamedTuple
        supplied = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(values))
        names = parameter_names(schema)
        Set(keys(supplied)) == Set(names) || throw(ArgumentError("parent coefficient names must match the parent schema exactly"))
        all(value -> value isa Real && isfinite(value), Base.values(supplied)) ||
            throw(ArgumentError("parent coefficients must be finite real values"))
        Float64[supplied[name] for name in names]
    else
        throw(ArgumentError("parent coefficients require a real vector, named mapping, or parent ParameterPoint"))
    end
    reduced = project_parameters(selection, vector; strict=strict)
    embed_parameters(selection, reduced)
    return _forward_selected_parameters(prepared, reduced)
end

function _forward_stage_equivalent(a, b; atol::Real, rtol::Real)
    isnothing(a) && return false
    isnothing(b) && return false
    if (a isa Number && b isa Number) || (a isa AbstractArray && b isa AbstractArray)
        return isapprox(a, b; atol=atol, rtol=rtol)
    end
    applicable(Observables.validate_observable_equivalence, a, b) || return false
    return Observables.validate_observable_equivalence(a, b; atol=atol, rtol=rtol)
end

"""Compare selected solver and observable stages of two forward results without assuming an eigenspectrum exists.

`:spectrum` compares finite eigenenergies only; other result kinds should request the observable stages explicitly. A missing stage on either side is reported as `false`, not treated as equivalent. Unsupported stages and invalid tolerances are rejected.
"""
function validate_forward_equivalence(left::ForwardResult, right::ForwardResult;
                                      compare=(:spectrum, :intrinsic, :probe, :instrument), atol::Real=1e-8, rtol::Real=1e-9)
    isfinite(atol) && atol >= 0 && isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("forward equivalence tolerances must be finite and nonnegative"))
    stages = Tuple(compare)
    isempty(stages) && throw(ArgumentError("at least one equivalence stage is required"))
    length(unique(stages)) == length(stages) || throw(ArgumentError("forward equivalence stages must not be repeated"))
    permitted = (:spectrum, :intrinsic, :probe, :instrument)
    all(stage -> stage in permitted, stages) || throw(ArgumentError("unknown forward equivalence stage"))
    checks = Dict{Symbol,Bool}()
    for stage in stages
        if stage === :spectrum
            if supports(left.solution, :energies) && supports(right.solution, :energies)
                a = result_energies(left.solution)
                b = result_energies(right.solution)
                checks[stage] = length(a) == length(b) && isapprox(sort(collect(a)), sort(collect(b)); atol=atol, rtol=rtol)
            else
                checks[stage] = false
            end
        else
            field = stage === :intrinsic ? :observable_result : stage === :probe ? :scattering_result : :result
            a = getproperty(left, field)
            b = getproperty(right, field)
            checks[stage] = _forward_stage_equivalent(a, b; atol=atol, rtol=rtol)
        end
    end
    return (; valid=all(values(checks)), checks, atol, rtol)
end
