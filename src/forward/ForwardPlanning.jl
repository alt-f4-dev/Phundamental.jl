# Deterministic policy-governed execution planning for the high-level forward API.
# Planning selects among computationally compatible routes without redefining physical/model-space admissibility.

const _DEFAULT_FORWARD_SOLVER_PREFERENCE = (
    :harmonic_phonon,
    :quadratic_fermion,
    :quadratic_boson,
    :optimized_lanczos,
    :exact_diagonalization,
    :thermal_curve,
    :lanczos,
    :auto_solver,
    :linear_spin_wave,
    :spin_wave,
    :single_site_dmft,
    :ladder_dga,
    :nlce,
)

"""
    ForwardExecutionPolicy(; allowed_approximations=(), require_certified=true, include_default_exact_routes=true, solver_preference=_DEFAULT_FORWARD_SOLVER_PREFERENCE)

Policy governing automatic forward-route selection. Exact numerical routes are permitted by default. Physical approximations must be named explicitly in `allowed_approximations`. `require_certified=true` rejects routes whose solver-compatibility report cannot be statically certified. The policy never determines whether a Hamiltonian is physically admissible.
"""
struct ForwardExecutionPolicy
    allowed_approximations::Tuple
    require_certified::Bool
    include_default_exact_routes::Bool
    solver_preference::Tuple
end

function _symbol_tuple(values)
    isnothing(values) && return ()
    values isa Symbol && return (values,)
    symbols = Symbol[]
    for value in values
        push!(symbols, value isa Symbol ? value : Symbol(value))
    end
    return Tuple(unique(symbols))
end

function ForwardExecutionPolicy(; allowed_approximations=(), require_certified::Bool=true, include_default_exact_routes::Bool=true,
                                solver_preference=_DEFAULT_FORWARD_SOLVER_PREFERENCE)
    approximations = Tuple(approximation for approximation in _symbol_tuple(allowed_approximations) if approximation !== :none)
    preference = _symbol_tuple(solver_preference)
    return ForwardExecutionPolicy(approximations, require_certified, include_default_exact_routes, preference)
end

"""
    ForwardRoute(solver; transformation=nothing, method_context=nothing, solver_args=(), solver_kwargs=(;), label=nothing)

Concrete candidate computational route considered by `plan_forward`. The route carries all information needed to construct a `ForwardProblem` if selected. Automatic planning accepts only certified exact outer representation transformations; approximate transformations remain explicit user-level calculations outside the planner.
"""
struct ForwardRoute{S,T,C,A,K}
    solver::S
    transformation::T
    method_context::C
    solver_args::A
    solver_kwargs::K
    label::Symbol

    function ForwardRoute(solver::S, transformation::T, method_context::C, solver_args::A, solver_kwargs::K,
                          label::Symbol) where {S,T,C,A,K}
        solver isa AbstractSolver || throw(ArgumentError("forward route solver must be an AbstractSolver"))
        (isnothing(transformation) || transformation isa AbstractTransformation) ||
            throw(ArgumentError("transformation must be AbstractTransformation or nothing"))
        (isnothing(method_context) || method_context isa AbstractForwardMethodContext) ||
            throw(ArgumentError("method_context must be AbstractForwardMethodContext or nothing"))
        solver_args isa Tuple || throw(ArgumentError("solver_args must be a Tuple"))
        solver_kwargs isa NamedTuple || throw(ArgumentError("solver_kwargs must be a NamedTuple"))
        return new{S,T,C,A,K}(solver, transformation, method_context, solver_args, solver_kwargs, label)
    end
end

function ForwardRoute(solver::AbstractSolver; transformation=nothing, method_context=nothing, solver_args::Tuple=(), solver_kwargs::NamedTuple=(;),
                      label::Union{Nothing,Symbol}=nothing)
    route_label = isnothing(label) ? solver_name(solver) : label
    return ForwardRoute(solver, transformation, method_context, solver_args, solver_kwargs, route_label)
end

"""Machine-readable assessment of one candidate forward route."""
struct ForwardRouteAssessment
    label::Symbol
    solver::Symbol
    compatible::Bool
    permitted::Bool
    certified::Bool
    approximations::Tuple
    transformation::Symbol
    method_context::Symbol
    score::Union{Nothing,NTuple{4,Int}}
    reasons::Tuple{Vararg{String}}
end

"""Inspectable planning record retained in a planned `ForwardProblem` and in `ForwardPlanningError`."""
struct ForwardPlanningReport{P,R,A}
    policy::P
    requirements::R
    selected_label::Union{Nothing,Symbol}
    selected_solver::Union{Nothing,Symbol}
    selected_approximations::Tuple
    selected_certified::Bool
    assessments::A
end

"""Error raised when no policy-permitted computational route can satisfy a forward request."""
struct ForwardPlanningError{R} <: Exception
    report::R
end

function Base.showerror(io::IO, error::ForwardPlanningError)
    report = error.report
    println(io, "no policy-permitted computational route satisfies request $(report.requirements.request_kind)")
    for assessment in report.assessments
        status = assessment.permitted ? "candidate" : assessment.compatible ? "policy-rejected" : "incompatible"
        detail = isempty(assessment.reasons) ? "no additional diagnostic" : join(assessment.reasons, "; ")
        println(io, "  $(assessment.label) [$(assessment.solver), $status]: $detail")
    end
end

"""Return the `ForwardPlanningReport` attached to a planned `ForwardProblem` or `ForwardPlanningError`, or `nothing` for an unplanned problem."""
planning_report(error::ForwardPlanningError) = error.report
planning_report(problem::ForwardProblem) = problem.metadata isa AbstractDict ? get(problem.metadata, :forward_planning_report, nothing) : nothing

function _default_exact_forward_routes()
    return (
        ForwardRoute(HarmonicPhononSolver()),
        ForwardRoute(QuadraticFermionSolver()),
        ForwardRoute(QuadraticBosonSolver()),
        ForwardRoute(ExactDiagonalization()),
        ForwardRoute(ThermodynamicForwardSolver()),
    )
end

_normalize_forward_route(route::ForwardRoute) = route
_normalize_forward_route(solver::AbstractSolver) = ForwardRoute(solver)

function _collect_forward_routes(policy::ForwardExecutionPolicy, routes)
    output = ForwardRoute[]
    policy.include_default_exact_routes && append!(output, _default_exact_forward_routes())
    supplied = (routes isa ForwardRoute || routes isa AbstractSolver) ? (routes,) : routes
    for route in supplied
        push!(output, _normalize_forward_route(route))
    end
    return output
end

_context_approximations(::Nothing) = ()
_context_approximations(::AbstractForwardMethodContext) = ()

function _context_approximations(context::LadderDGAForwardContext)
    approximation = _solver_approximation(context.dmft_solver)
    return approximation === :none ? () : (approximation,)
end

function _context_approximations(context::NLCEForwardContext)
    inner = Symbol[_context_approximations(context.inner_method_context)...]
    approximation = _solver_approximation(context.inner_solver)
    approximation === :none || push!(inner, approximation)
    return Tuple(unique(inner))
end

function _route_approximations(route::ForwardRoute)
    approximations = Symbol[_context_approximations(route.method_context)...]
    approximation = _solver_approximation(route.solver)
    approximation === :none || push!(approximations, approximation)
    return Tuple(unique(approximations))
end

function _route_transformation_name(route::ForwardRoute)
    isnothing(route.transformation) && return :none
    name = Transformations.transformation_name(route.transformation)
    return name isa Symbol ? name : Symbol(name)
end

function _route_context_name(route::ForwardRoute)
    isnothing(route.method_context) && return :none
    return nameof(typeof(route.method_context))
end

function _planning_preference_rank(policy::ForwardExecutionPolicy, solver::AbstractSolver)
    name = solver_name(solver)
    index = findfirst(==(name), policy.solver_preference)
    return isnothing(index) ? length(policy.solver_preference) + 1 : index
end

function _automatic_transformation_support(route::ForwardRoute, model::ManyBodyModel)
    isnothing(route.transformation) && return true, model, (), true
    _forward_allows_outer_transformation(route.solver) || return false, model, ("selected method does not permit an outer representation transformation",), true
    transformation_certificate = try
        Transformations.certificate(route.transformation)
    catch error
        return false, model, ("automatic planning requires a transformation certificate: $(sprint(showerror, error))",), false
    end
    transformation_certificate.exact && !transformation_certificate.approximate || return false, model, ("automatic planning accepts only certified exact outer representation transformations",), true
    transformed = try
        Transformations.transform(route.transformation, model).model
    catch error
        return false, model, ("exact transformation could not be applied to the source model: $(sprint(showerror, error))",), true
    end
    return true, transformed, (), true
end

function _route_policy_support(policy::ForwardExecutionPolicy, approximations::Tuple)
    forbidden = [approximation for approximation in approximations if !(approximation in policy.allowed_approximations)]
    isempty(forbidden) && return true, ()
    return false, ("physical approximation(s) not authorized by the execution policy: $(join(string.(forbidden), ", "))",)
end

function _assess_forward_route(route::ForwardRoute, model::ManyBodyModel, request::Observables.AbstractObservable,
                               policy::ForwardExecutionPolicy, index::Int)
    transformation_supported, working_model, transformation_reasons, transformation_certified = _automatic_transformation_support(route, model)
    approximations = _route_approximations(route)
    policy_supported, policy_reasons = _route_policy_support(policy, approximations)

    if transformation_supported
        compatibility = solver_compatibility(route.solver, working_model, request, route.method_context)
        compatible = compatibility.compatible
        certified = compatibility.certified && transformation_certified
        compatibility_reasons = compatibility.reasons
    else
        compatible = false
        certified = transformation_certified
        compatibility_reasons = ()
    end

    certification_supported = !policy.require_certified || certified
    certification_reasons = certification_supported ? () : ("execution policy requires a statically certified route",)
    permitted = compatible && policy_supported && certification_supported
    reasons = _compatibility_reasons(transformation_reasons, compatibility_reasons, policy_reasons, certification_reasons)
    approximation_tier = isempty(approximations) ? 0 : 1
    certification_tier = certified ? 0 : 1
    score = permitted ? (approximation_tier, certification_tier, _planning_preference_rank(policy, route.solver), index) : nothing
    assessment = ForwardRouteAssessment(route.label, solver_name(route.solver), compatible, permitted, certified, approximations,
                                        _route_transformation_name(route), _route_context_name(route), score, reasons)
    return assessment
end

function _planning_metadata(metadata)
    metadata isa AbstractDict && return Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    metadata isa NamedTuple && return Dict{Symbol,Any}(pairs(metadata))
    throw(ArgumentError("planning metadata must be an AbstractDict or NamedTuple"))
end

"""
    plan_forward(model, request; policy=ForwardExecutionPolicy(), routes=(), probe=nothing, resolution=nothing, metadata=Dict())

Construct an inspectable `ForwardProblem` by deterministically selecting the preferred policy-permitted computational route for `model` and `request`. Default candidates contain only context-free exact numerical routes. Approximation-bearing routes such as LSWT, DMFT, ladder DΓA, or NLCE must be supplied explicitly through `routes` and authorized by `policy`.
"""
function plan_forward(model::ManyBodyModel, request::Observables.AbstractObservable; policy::ForwardExecutionPolicy=ForwardExecutionPolicy(), routes=(),
                      probe=nothing, resolution=nothing, metadata=Dict{Symbol,Any}())
    candidates = _collect_forward_routes(policy, routes)
    assessments = ForwardRouteAssessment[]
    for (index, route) in enumerate(candidates)
        push!(assessments, _assess_forward_route(route, model, request, policy, index))
    end

    permitted_indices = [index for (index, assessment) in enumerate(assessments) if assessment.permitted]
    if isempty(permitted_indices)
        report = ForwardPlanningReport(policy, request_requirements(request), nothing, nothing, (), false, assessments)
        throw(ForwardPlanningError(report))
    end

    sort!(permitted_indices; by=index -> something(assessments[index].score))
    selected_index = first(permitted_indices)
    selected_route = candidates[selected_index]
    selected_assessment = assessments[selected_index]
    report = ForwardPlanningReport(policy, request_requirements(request), selected_assessment.label, selected_assessment.solver,
                                   selected_assessment.approximations, selected_assessment.certified, assessments)
    plan_metadata = _planning_metadata(metadata)
    plan_metadata[:forward_planning_report] = report
    plan_metadata[:forward_planned] = true
    plan_metadata[:planned_solver] = selected_assessment.solver
    plan_metadata[:planned_approximations] = selected_assessment.approximations
    plan_metadata[:planned_transformation] = selected_assessment.transformation
    return ForwardProblem(model; transformation=selected_route.transformation, solver=selected_route.solver, solver_args=selected_route.solver_args,
                          solver_kwargs=selected_route.solver_kwargs, method_context=selected_route.method_context, observable=request,
                          probe=probe, resolution=resolution, metadata=plan_metadata)
end

"""Construct the model from material data and return `plan_forward(model, request; ...)`."""
function plan_forward(material::Material, cluster::Supercell, specification::AbstractHamiltonianSpec, request::Observables.AbstractObservable;
                      model_kwargs::NamedTuple=(;), kwargs...)
    model = build_model(material, cluster, specification; model_kwargs...)
    return plan_forward(model, request; kwargs...)
end

"""Plan and execute a typed high-level forward request using the conservative default execution policy unless another policy is supplied."""
function forward(model::ManyBodyModel, request::Observables.AbstractObservable; policy::ForwardExecutionPolicy=ForwardExecutionPolicy(), routes=(),
                 probe=nothing, resolution=nothing, metadata=Dict{Symbol,Any}())
    plan = plan_forward(model, request; policy=policy, routes=routes, probe=probe, resolution=resolution, metadata=metadata)
    return forward(plan)
end
"""Construct a model, plan a typed request under `ForwardExecutionPolicy`, and execute the selected route."""
function forward(material::Material, cluster::Supercell, specification::AbstractHamiltonianSpec, request::Observables.AbstractObservable;
                 model_kwargs::NamedTuple=(;), kwargs...)
    plan = plan_forward(material, cluster, specification, request; model_kwargs=model_kwargs, kwargs...)
    return forward(plan)
end
