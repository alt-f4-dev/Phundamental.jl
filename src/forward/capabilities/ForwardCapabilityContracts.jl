# Extensible, solver-independent capability contracts for high-level forward planning.
# Solver implementations may specialize these public hooks without editing the package's built-in dispatch tables.

"""A stage-specific diagnostic for computational compatibility. Physical admissibility is a separate contract."""
struct CompatibilityIssue
    stage::Symbol
    code::Symbol
    message::String

    function CompatibilityIssue(stage::Symbol, code::Symbol, message::AbstractString)
        stage in (:model, :request_model, :solver_request, :method_context) || throw(ArgumentError("unsupported compatibility stage $stage"))
        isempty(message) && throw(ArgumentError("compatibility issue message cannot be empty"))
        return new(stage, code, String(message))
    end
end

"""A typed result returned by a public solver/request capability extension hook."""
struct CapabilityDecision
    supported::Bool
    certified::Bool
    issues::Tuple{Vararg{CompatibilityIssue}}
end

function CapabilityDecision(supported::Bool; certified::Bool=true, issues=())
    all(issue -> issue isa CompatibilityIssue, issues) || throw(ArgumentError("capability issues must be CompatibilityIssue objects"))
    return CapabilityDecision(supported, certified, Tuple(issues))
end

_capability_tuple(decision::CapabilityDecision) = (decision.supported, Tuple(issue.message for issue in decision.issues), decision.certified)

"""Return a model-support decision for a solver. Define a specialized method to integrate a new `AbstractSolver` subtype."""
solver_model_capability(::AbstractSolver, ::ManyBodyModel) = CapabilityDecision(false; issues=(
    CompatibilityIssue(:model, :unregistered, "no static model-capability rule is registered for this solver"),))

"""Return the result family produced by a solver on a model, or `:unknown` if not declared."""
solver_output_family(::AbstractSolver, ::ManyBodyModel) = :unknown

"""Return the explicit physical approximation identity of a solver; `:none` denotes no physical approximation."""
solver_approximation(::AbstractSolver) = :none

"""Return a physical model/request decision. Custom typed requests can specialize this hook."""
request_model_capability(::AbstractObservable, ::ManyBodyModel) = CapabilityDecision(true; certified=false)

"""Return solver/request compatibility. New solver implementations may override this method independently of result-family matching."""
function solver_request_capability(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable)
    supported, reasons, certified = _request_family_support(request, _solver_output_family(solver, model))
    issues = Tuple(CompatibilityIssue(:solver_request, supported ? :deferred : :unsupported, reason) for reason in reasons)
    return CapabilityDecision(supported; certified=certified, issues=issues)
end

"""Return a method-context compatibility decision, including the certification status of runtime-only checks."""
solver_context_capability(::AbstractSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) = CapabilityDecision(true)

function solver_context_capability(::AbstractSolver, ::ManyBodyModel, ::AbstractObservable, context::AbstractForwardMethodContext)
    issue = CompatibilityIssue(:method_context, :unsupported, "the selected solver does not accept forward method context $(nameof(typeof(context)))")
    return CapabilityDecision(false; issues=(issue,))
end

function solver_context_capability(::AbstractSolver, ::ManyBodyModel, ::AbstractObservable, context)
    issue = CompatibilityIssue(:method_context, :invalid_context, "method_context must be AbstractForwardMethodContext or nothing; got $(typeof(context))")
    return CapabilityDecision(false; issues=(issue,))
end

function _stage_issues(stage::Symbol, supported::Bool, certified::Bool, reasons)
    code = !supported ? :unsupported : certified ? :note : :deferred
    issues = CompatibilityIssue[CompatibilityIssue(stage, code, reason) for reason in reasons]
    if !supported && isempty(issues)
        push!(issues, CompatibilityIssue(stage, :unsupported, "the capability rule rejected this stage without a detailed reason"))
    elseif supported && !certified && isempty(issues)
        push!(issues, CompatibilityIssue(stage, :deferred, "this stage requires checks at execution"))
    end
    return Tuple(issues)
end

