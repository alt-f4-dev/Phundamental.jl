# Explicit matrix-DMFT forward route; it never infers local interactions from unrelated model metadata.

"""
    MultiOrbitalDMFTForwardContext(model, problem)

Bind a `MultiOrbitalDMFTProblem` to its declaring `ManyBodyModel` for explicit
Matsubara forward execution. Caller-supplied numerical data are authoritative;
physical equality to a general symbolic Hamiltonian is not automatically certified.
This route is never automatically preferred over exact numerical solvers.
"""
struct MultiOrbitalDMFTForwardContext{M<:ManyBodyModel,P<:MultiOrbitalDMFTProblem} <: AbstractForwardMethodContext
    source_model::M
    problem::P
end

_solver_model_support(::MultiOrbitalDMFTSolver, model::ManyBodyModel) =
    _contains_fermion_algebra(model.representation.algebra) ?
    (true, ("the multiorbital impurity interaction is declared explicitly by the forward context",), false) :
    (false, ("matrix DMFT requires a fermionic model representation",), true)

_solver_output_family(::MultiOrbitalDMFTSolver, ::ManyBodyModel) = :dmft
_solver_approximation(::MultiOrbitalDMFTSolver) = :single_site_multi_orbital_dmft
_forward_allows_outer_transformation(::MultiOrbitalDMFTSolver) = false

# General Matsubara observables are meaningful for any fermionic model; the chosen
# solver retains responsibility for checking whether it can actually compute them.
_request_model_support(::Observables.LocalMatsubaraGreenFunctionRequest, model::ManyBodyModel, ::MultiOrbitalDMFTSolver) =
    _contains_fermion_algebra(model.representation.algebra) ? (true, (), true) :
    (false, ("a Matsubara fermionic Green request needs a fermionic representation",), true)

_request_model_support(::Observables.LocalMatsubaraSelfEnergyRequest, model::ManyBodyModel, ::MultiOrbitalDMFTSolver) =
    _contains_fermion_algebra(model.representation.algebra) ? (true, (), true) :
    (false, ("a Matsubara self-energy request needs a fermionic representation",), true)

function _forward_context_support(::MultiOrbitalDMFTSolver, model::ManyBodyModel, request::AbstractObservable,
                                  context::MultiOrbitalDMFTForwardContext)
    context.source_model === model || return false, ("multiorbital DMFT context must be bound to the supplied model",), true
    supported, reasons = _request_context_temperature_support(request, context.problem.temperature)
    return supported, (reasons..., "explicit impurity and lattice data are not automatically certified as a lowering of the symbolic Hamiltonian"), false
end

_forward_context_support(::MultiOrbitalDMFTSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("multiorbital DMFT requires an explicit MultiOrbitalDMFTForwardContext",), true)

function _prepare_solver_input(::MultiOrbitalDMFTSolver, model::ManyBodyModel, ::Observables.AbstractObservable,
                               args::Tuple, kwargs::NamedTuple, context::MultiOrbitalDMFTForwardContext)
    isempty(args) || throw(ArgumentError("multiorbital DMFT input comes from method_context; solver_args must be empty"))
    context.source_model === model || throw(ArgumentError("context model differs from the requested forward model"))
    return _ProblemSolverInput(context.problem, (), kwargs)
end

function _execute_solver_input(solver::MultiOrbitalDMFTSolver, input::_ProblemSolverInput{P}) where {P<:MultiOrbitalDMFTProblem}
    result = solve(solver, input.problem, input.args...; input.kwargs...)
    result.metadata[:forward_route] = :single_site_multi_orbital_dmft
    result.metadata[:forward_frequency_domain] = :matsubara
    result.metadata[:symbolic_lowering_certified] = false
    return result
end

# Explicit opt-in high-level forward route for the screened particle-hole approximation.

"""Bind a screened-ladder problem to a specific physical source model for typed Matsubara execution."""
struct OrbitalScreenedLadderForwardContext{M<:ManyBodyModel,P<:OrbitalScreenedLadderProblem} <: AbstractForwardMethodContext
    source_model::M
    problem::P
end

_solver_model_support(::OrbitalScreenedLadderSolver, model::ManyBodyModel) =
    _contains_fermion_algebra(model.representation.algebra) ?
    (true, ("lattice and local interaction supplied explicitly by context",), false) :
    (false, ("screened-ladder execution requires a fermionic model representation",), true)

_solver_output_family(::OrbitalScreenedLadderSolver, ::ManyBodyModel) = :dmft
_solver_approximation(::OrbitalScreenedLadderSolver) = :one_shot_screened_particle_hole
_forward_allows_outer_transformation(::OrbitalScreenedLadderSolver) = false

_request_model_support(::Observables.LocalMatsubaraSelfEnergyRequest, model::ManyBodyModel, ::OrbitalScreenedLadderSolver) =
    _contains_fermion_algebra(model.representation.algebra) ? (true, (), true) :
    (false, ("local Matsubara self-energy requires a fermionic model",), true)

function _forward_context_support(::OrbitalScreenedLadderSolver, model::ManyBodyModel,
                                  request::AbstractObservable, context::OrbitalScreenedLadderForwardContext)
    context.source_model === model || return false, ("forward context belongs to a different model",), true
    supported, reasons = _request_context_temperature_support(request, context.problem.dmft.problem.temperature)
    return supported, (reasons..., "symbolic-to-lattice lowering is caller-supplied and not certified"), false
end

_forward_context_support(::OrbitalScreenedLadderSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("screened-ladder solver requires OrbitalScreenedLadderForwardContext",), true)

function _prepare_solver_input(::OrbitalScreenedLadderSolver, model::ManyBodyModel, ::AbstractObservable,
                               args::Tuple, kwargs::NamedTuple, context::OrbitalScreenedLadderForwardContext)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("screened-ladder inputs belong to method_context"))
    context.source_model === model || throw(ArgumentError("forward model and context differ"))
    return _ProblemSolverInput(context.problem, (), (;))
end

function _execute_solver_input(solver::OrbitalScreenedLadderSolver, input::_ProblemSolverInput{P}) where {P<:OrbitalScreenedLadderProblem}
    result = solve(solver, input.problem)
    result.metadata[:forward_route] = :one_shot_screened_particle_hole
    result.metadata[:symbolic_lowering_certified] = false
    return result
end

"""Request the *momentum-resolved* Matsubara self-energy from an explicit screened-ladder forward route."""
struct MomentumMatsubaraSelfEnergyRequest <: Observables.AbstractObservable
    metadata::Dict{Symbol,Any}
end

MomentumMatsubaraSelfEnergyRequest(; metadata=Dict{Symbol,Any}()) =
    MomentumMatsubaraSelfEnergyRequest(Dict{Symbol,Any}(metadata))

(::MomentumMatsubaraSelfEnergyRequest)(::ManyBodyModel, solution::OrbitalScreenedLadderResult) = solution.nonlocal_self_energy

function Observables.request_requirements(request::MomentumMatsubaraSelfEnergyRequest)
    return Observables.RequestRequirements(request_kind=:momentum_matsubara_self_energy, response_kind=:self_energy,
        frequency_domain=:matsubara, statistics=:fermionic, momentum_scope=:momentum_resolved,
        response_order=2, metadata=request.metadata)
end

_request_model_support(::MomentumMatsubaraSelfEnergyRequest, model::ManyBodyModel, ::OrbitalScreenedLadderSolver) =
    _contains_fermion_algebra(model.representation.algebra) ? (true, (), true) :
    (false, ("momentum Matsubara self-energy requires a fermionic model",), true)

function _request_family_support(::MomentumMatsubaraSelfEnergyRequest, family::Symbol)
    supported = family === :dmft
    return supported, supported ? () : ("the momentum Matsubara self-energy needs a matrix embedding result",), true
end

# Explicit high-level forward context for the finite-support SU(2) ladder calculation.

"""Bind a prepared multiorbital ladder workflow to its declared physical source model.

No symbolic-to-numerical multiorbital lowering is inferred from unrelated model
metadata; the caller explicitly supplies the converged numerical DMFT input.
"""
struct OrbitalLadderWorkflowForwardContext{M<:ManyBodyModel,P<:OrbitalLadderWorkflowProblem} <: AbstractForwardMethodContext
    source_model::M
    problem::P
end

_solver_model_support(::OrbitalLadderWorkflowSolver, model::ManyBodyModel) =
    _contains_fermion_algebra(model.representation.algebra) ?
    (true, ("a converged SU(2)-invariant multiorbital DMFT reference must be supplied",), false) :
    (false, ("the orbital ladder requires a fermionic model representation",), true)

_solver_output_family(::OrbitalLadderWorkflowSolver, ::ManyBodyModel) = :dmft
_solver_approximation(::OrbitalLadderWorkflowSolver) = :finite_support_one_shot_su2_ladder
_forward_allows_outer_transformation(::OrbitalLadderWorkflowSolver) = false

_request_model_support(::Observables.LocalMatsubaraSelfEnergyRequest, ::ManyBodyModel, ::OrbitalLadderWorkflowSolver) =
    (false, ("use MomentumMatsubaraSelfEnergyRequest for the nonlocal ladder self-energy",), true)

_request_model_support(::MomentumMatsubaraSelfEnergyRequest, model::ManyBodyModel, ::OrbitalLadderWorkflowSolver) =
    _contains_fermion_algebra(model.representation.algebra) ? (true, (), true) :
    (false, ("momentum-resolved fermionic self-energy requires a fermionic model",), true)

function _forward_context_support(::OrbitalLadderWorkflowSolver, model::ManyBodyModel,
    request::AbstractObservable, context::OrbitalLadderWorkflowForwardContext)
    context.source_model === model || return false, ("workflow context is bound to a different model",), true
    supported, reasons = _request_context_temperature_support(request, context.problem.dmft.problem.temperature)
    return supported, (reasons..., "symbolic-to-lattice lowering is explicitly caller-supplied",), false
end

_forward_context_support(::OrbitalLadderWorkflowSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("the ladder solver requires OrbitalLadderWorkflowForwardContext",), true)

function _prepare_solver_input(::OrbitalLadderWorkflowSolver, model::ManyBodyModel, ::AbstractObservable,
    args::Tuple, kwargs::NamedTuple, context::OrbitalLadderWorkflowForwardContext)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("ladder solver inputs belong to method_context"))
    context.source_model === model || throw(ArgumentError("forward model differs from context source model"))
    return _ProblemSolverInput(context.problem, (), (;))
end

function _execute_solver_input(solver::OrbitalLadderWorkflowSolver, input::_ProblemSolverInput{P}) where {P<:OrbitalLadderWorkflowProblem}
    result = solve(solver, input.problem)
    result.metadata[:forward_route] = :one_shot_su2_ladder
    result.metadata[:symbolic_lowering_certified] = false
    return result
end

(::MomentumMatsubaraSelfEnergyRequest)(::ManyBodyModel, result::OrbitalLadderWorkflowResult) = result.nonlocal_self_energy
