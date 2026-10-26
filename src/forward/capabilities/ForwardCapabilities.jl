# Computational compatibility for the high-level forward API.
# Physical/model-space admissibility is intentionally outside this layer.

"""
    SolverCompatibility

Static compatibility report for a concrete numerical solver, a `ManyBodyModel`, and optionally a typed observable request. `compatible` means that the current Julia implementation has a known computational route for the supplied combination; it does not imply that the Hamiltonian is physically admissible under a future generative specification.
"""
struct SolverCompatibility{R,I}
    solver::Symbol
    model_supported::Bool
    request_supported::Bool
    compatible::Bool
    certified::Bool
    approximation::Symbol
    requirements::R
    reasons::Tuple{Vararg{String}}
    issues::I
end

# Preserve positional construction used before typed diagnostic stages were introduced.
SolverCompatibility(solver::Symbol, model_supported::Bool, request_supported::Bool, compatible::Bool, certified::Bool,
                    approximation::Symbol, requirements, reasons::Tuple{Vararg{String}}) =
    SolverCompatibility(solver, model_supported, request_supported, compatible, certified, approximation, requirements, reasons, ())

_solver_approximation(solver::AbstractSolver) = solver_approximation(solver)
_solver_approximation(::LinearSpinWaveSolver) = :linear_spin_wave
_solver_approximation(::DMFTSolver) = :single_site_dmft
_solver_approximation(::DiagrammaticDMFTSolver) = :one_shot_ladder_dga
_solver_approximation(::NLCESolver) = :finite_order_nlce

function _compatibility_reasons(reasons...)
    output = String[]
    for group in reasons
        append!(output, group)
    end
    return Tuple(unique(output))
end

function _finite_basis_solver_support(model::ManyBodyModel)
    capabilities = numerical_capabilities(model)
    return capabilities.numerically_realizable, capabilities.reasons
end

_solver_model_support(solver::AbstractSolver, model::ManyBodyModel) = _capability_tuple(solver_model_capability(solver, model))

function _field_model_support(field::MomentumField, model::ManyBodyModel)
    reasons = String[]
    for key in field.keys
        try
            Observables.resolve_observable(model, key)
        catch error
            push!(reasons, sprint(showerror, error))
        end
    end
    return isempty(reasons), Tuple(unique(reasons))
end

_request_model_support(request::AbstractObservable, model::ManyBodyModel) = _capability_tuple(request_model_capability(request, model))

function _request_model_support(request::ObservableRequest, ::ManyBodyModel)
    return true, ("ObservableRequest evaluator is opaque; request compatibility cannot be statically certified",), false
end

function _request_model_support(request::DynamicalCorrelationRequest, model::ManyBodyModel)
    left_ok, left_reasons = _field_model_support(request.field_a, model)
    right_ok, right_reasons = _field_model_support(request.field_b, model)
    return left_ok && right_ok, _compatibility_reasons(left_reasons, right_reasons), true
end

function _request_model_support(request::CompositeDynamicalCorrelationRequest, model::ManyBodyModel)
    reasons = String[]
    for field in values(request.fields)
        supported, field_reasons = _field_model_support(field, model)
        supported || append!(reasons, field_reasons)
    end
    return isempty(reasons), Tuple(unique(reasons)), true
end

function _request_model_support(request::SpinPhononCorrelationRequest, model::ManyBodyModel)
    try
        Observables._spin_phonon_request_fields(request, model)
        return true, (), true
    catch error
        return false, (sprint(showerror, error),), true
    end
end

function _request_model_support(request::SpinTensorStructureFactorRequest, model::ManyBodyModel)
    try
        Observables._spin_tensor_request_field(request, model)
        return true, (), true
    catch error
        return false, (sprint(showerror, error),), true
    end
end

function _contains_fermion_algebra(algebra::AbstractOperatorAlgebra)
    algebra isa FermionAlgebra && return true
    algebra isa CompositeAlgebra || return false
    return any(_contains_fermion_algebra, algebra.factors)
end

function _request_model_support(::Observables.LocalElectronSpectralFunctionRequest, model::ManyBodyModel)
    _contains_fermion_algebra(model.representation.algebra) || return false, ("local electronic spectral request requires a fermionic algebra factor",), true
    return true, (), true
end

_request_model_support(::Observables.LocalMatsubaraGreenFunctionRequest, model::ManyBodyModel) = _hubbard_forward_model_support(model)

_request_model_support(::Observables.LocalMatsubaraSelfEnergyRequest, model::ManyBodyModel) = _hubbard_forward_model_support(model)

_request_model_support(::Observables.MatsubaraSusceptibilityRequest, model::ManyBodyModel) = _hubbard_forward_model_support(model)

_request_model_support(request::AbstractObservable, model::ManyBodyModel, ::AbstractSolver) = _request_model_support(request, model)

function _request_model_support(request::Observables.PhononStructureFactorRequest, model::ManyBodyModel)
    try
        Observables._phonon_structure_solver_wavevectors(request, model)
        return true, (), true
    catch error
        return false, (sprint(showerror, error),), true
    end
end

_solver_output_family(solver::AbstractSolver, model::ManyBodyModel) = solver_output_family(solver, model)

function _request_family_support(::ObservableRequest, ::Symbol)
    return true, ("ObservableRequest evaluator is opaque; solver-result compatibility cannot be statically certified",), false
end

function _request_family_support(::DynamicalCorrelationRequest, family::Symbol)
    supported = family in (:eigensystem, :quadratic_boson_mode)
    reasons = supported ? () : ("DynamicalCorrelationRequest currently requires an eigensystem or bosonic QuadraticModeResult",)
    return supported, reasons, true
end

function _request_family_support(::CompositeDynamicalCorrelationRequest, family::Symbol)
    supported = family in (:eigensystem, :quadratic_boson_mode)
    reasons = supported ? () : ("CompositeDynamicalCorrelationRequest currently requires an eigensystem or bosonic QuadraticModeResult",)
    return supported, reasons, true
end

function _request_family_support(::SpinPhononCorrelationRequest, family::Symbol)
    supported = family in (:eigensystem, :quadratic_boson_mode)
    reasons = supported ? () : ("SpinPhononCorrelationRequest currently requires an eigensystem or bosonic QuadraticModeResult",)
    return supported, reasons, true
end

function _request_family_support(::SpinTensorStructureFactorRequest, family::Symbol)
    supported = family in (:eigensystem, :spin_wave)
    reasons = supported ? () : ("SpinTensorStructureFactorRequest currently requires an eigensystem or SpinWaveResult",)
    return supported, reasons, true
end

function _request_family_support(::Observables.LocalElectronSpectralFunctionRequest, ::Symbol)
    return false, ("LocalElectronSpectralFunctionRequest requires a spin-resolved FermionSector SolverResult from the packed fermion pathway",), true
end

function _request_family_support(::Observables.PhononStructureFactorRequest, family::Symbol)
    supported = family === :phonon_dispersion
    reasons = supported ? () : ("PhononStructureFactorRequest currently requires a request-aware harmonic phonon dispersion route",)
    return supported, reasons, true
end

function _request_family_support(::Observables.LocalMatsubaraGreenFunctionRequest, family::Symbol)
    supported = family === :dmft
    reasons = supported ? () : ("LocalMatsubaraGreenFunctionRequest currently requires the single-site DMFT forward route",)
    return supported, reasons, true
end

function _request_family_support(::Observables.LocalMatsubaraSelfEnergyRequest, family::Symbol)
    supported = family === :dmft
    reasons = supported ? () : ("LocalMatsubaraSelfEnergyRequest currently requires the single-site DMFT forward route",)
    return supported, reasons, true
end

function _request_family_support(::Observables.MatsubaraSusceptibilityRequest, family::Symbol)
    supported = family === :ladder_dga
    reasons = supported ? () : ("MatsubaraSusceptibilityRequest currently requires the ladder-DΓA forward route",)
    return supported, reasons, true
end

function _request_family_support(::AbstractObservable, ::Symbol)
    return false, ("no solver-result compatibility rule is registered for this observable request type",), true
end

_solver_request_support(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable) =
    _capability_tuple(solver_request_capability(solver, model, request))

function _resolved_fermion_sector(model::ManyBodyModel, sector)
    resolved = try
        Solvers.specialize_sector(model, sector)
    catch
        nothing
    end
    return resolved isa FermionSector
end

function _request_context_temperature_support(request::AbstractObservable, temperature::Float64)
    requested = request_requirements(request).temperature
    isnothing(requested) && return true, ()
    isapprox(requested, temperature; rtol=1e-12, atol=1e-12 * max(abs(temperature), 1.0)) && return true, ()
    return false, ("request temperature does not match the selected embedding method context",)
end

_forward_context_support(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, ::Nothing) =
    _capability_tuple(solver_context_capability(solver, model, request, nothing))

_forward_context_support(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, context::AbstractForwardMethodContext) =
    _capability_tuple(solver_context_capability(solver, model, request, context))

_forward_context_support(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, context) =
    _capability_tuple(solver_context_capability(solver, model, request, context))

include("BuiltinSolverCapabilities.jl")

"""
    solver_compatibility(solver, model)
    solver_compatibility(solver, model, request)
    solver_compatibility(solver, model, request, method_context)

Inspect whether the current numerical implementation can evaluate a model, optionally for a particular typed observable request and explicit method context. This is a computational compatibility query, not a physical/model-space admissibility test.
"""
function solver_compatibility(solver::AbstractSolver, model::ManyBodyModel)
    model_supported, model_reasons, certified = _solver_model_support(solver, model)
    issues = _stage_issues(:model, model_supported, certified, model_reasons)
    return SolverCompatibility(solver_name(solver), model_supported, true, model_supported, certified, _solver_approximation(solver), nothing,
                               Tuple(unique(model_reasons)), issues)
end

solver_compatibility(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable) =
    solver_compatibility(solver, model, request, nothing)

function solver_compatibility(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, method_context)
    requirements = request_requirements(request)
    model_supported, model_reasons, model_certified = _solver_model_support(solver, model)
    request_model_supported, request_model_reasons, request_model_certified = _request_model_support(request, model, solver)
    route_supported, route_reasons, route_certified = _solver_request_support(solver, model, request)
    context_supported, context_reasons, context_certified = _forward_context_support(solver, model, request, method_context)
    request_supported = request_model_supported && route_supported && context_supported
    compatible = model_supported && request_supported
    reasons = _compatibility_reasons(model_reasons, request_model_reasons, route_reasons, context_reasons)
    certified = model_certified && request_model_certified && route_certified && context_certified
    issues = (_stage_issues(:model, model_supported, model_certified, model_reasons)...,
              _stage_issues(:request_model, request_model_supported, request_model_certified, request_model_reasons)...,
              _stage_issues(:solver_request, route_supported, route_certified, route_reasons)...,
              _stage_issues(:method_context, context_supported, context_certified, context_reasons)...)
    return SolverCompatibility(solver_name(solver), model_supported, request_supported, compatible, certified, _solver_approximation(solver),
                               requirements, reasons, issues)
end

"""Return the typed stage diagnostics attached to a compatibility report without modifying its legacy reason strings."""
compatibility_issues(report::SolverCompatibility) = report.issues

solver_capabilities(solver::AbstractSolver, model::ManyBodyModel) = solver_compatibility(solver, model)
solver_capabilities(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable) = solver_compatibility(solver, model, request)
solver_capabilities(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, method_context) =
    solver_compatibility(solver, model, request, method_context)

supports(solver::AbstractSolver, model::ManyBodyModel) = solver_compatibility(solver, model).compatible
supports(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable) = solver_compatibility(solver, model, request).compatible
supports(solver::AbstractSolver, model::ManyBodyModel, request::AbstractObservable, method_context) =
    solver_compatibility(solver, model, request, method_context).compatible

"""Return the subset of caller-supplied solver candidates that are computationally compatible with `model` and `request`. This function does not rank or select among them."""
function compatible_solvers(model::ManyBodyModel, request::AbstractObservable, candidates)
    return [solver for solver in candidates if supports(solver, model, request)]
end
