# High-level routing for already implemented equal-time, thermal, and cluster-perturbation physics.

"""Solver descriptor that delegates thermodynamics to an explicit exact or stochastic thermal method."""
struct ThermodynamicForwardSolver{M<:AbstractThermalMethod} <: AbstractSolver
    method::M
end

ThermodynamicForwardSolver() = ThermodynamicForwardSolver(ExactGibbs())
solver_name(::ThermodynamicForwardSolver) = :thermal_curve
_solver_approximation(solver::ThermodynamicForwardSolver) = solver.method isa ExactGibbs ? :none : :stochastic_thermal_sampling

function _solver_model_support(solver::ThermodynamicForwardSolver, model::ManyBodyModel)
    supported, reasons = _finite_basis_solver_support(model)
    if numerical_capabilities(model).biorthogonal
        return false, ("canonical thermodynamics requires a Hermitian, non-biorthogonal physical Hamiltonian",), true
    end
    return supported, reasons, true
end

# A validated thermal request is statically inspectable; model and solver admissibility remain separate checks.
_request_model_support(::ThermodynamicCurveRequest, ::ManyBodyModel) = (true, (), true)

_solver_output_family(::ThermodynamicForwardSolver, ::ManyBodyModel) = :thermodynamic_curve

function _request_family_support(::ThermodynamicCurveRequest, family::Symbol)
    supported = family === :thermodynamic_curve
    return supported, supported ? () : ("thermodynamic curves require ThermodynamicForwardSolver",), true
end

function _request_family_support(::StaticStructureFactorRequest, family::Symbol)
    supported = family in (:eigensystem, :quadratic_boson_mode)
    return supported, supported ? () : ("static structure factors require a full eigensystem or quadratic-boson mode result",), true
end

function _request_family_support(::CPTSpectralFunctionRequest, family::Symbol)
    return false, ("CPT requires an optimized fermion-sector solver and certified single-particle workspace",), true
end

function _solver_request_support(::ThermodynamicForwardSolver, ::ManyBodyModel, ::ThermodynamicCurveRequest)
    return true, (), true
end

function _solver_request_support(solver::OptimizedLanczos, model::ManyBodyModel, ::CPTSpectralFunctionRequest)
    _resolved_fermion_sector(model, solver.sector) ||
        return false, ("CPT requires OptimizedLanczos with a resolved fermionic ground sector",), true
    return true, (), true
end

function _solver_request_support(solver::AutoSolver, model::ManyBodyModel, ::CPTSpectralFunctionRequest)
    _solver_output_family(solver, model) === :eigensystem && _resolved_fermion_sector(model, solver.sector) ||
        return false, ("CPT requires AutoSolver to select an eigensystem with a resolved fermion sector",), true
    return true, (), true
end

struct _ThermalCurveSolverInput{M,R}
    model::M
    request::R
end

function _prepare_solver_input(::ThermodynamicForwardSolver, model::ManyBodyModel, request::ThermodynamicCurveRequest,
                               args::Tuple, kwargs::NamedTuple, ::Nothing)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("thermal solver configuration belongs to ThermodynamicForwardSolver, not solver_args"))
    return _ThermalCurveSolverInput(model, request)
end

function _execute_solver_input(solver::ThermodynamicForwardSolver, input::_ThermalCurveSolverInput)
    curve = thermal_curve(input.model, solver.method, input.request.temperatures; kB=input.request.kB)
    curve.metadata[:forward_route] = solver_name(solver)
    curve.metadata[:forward_approximation] = _solver_approximation(solver)
    curve.metadata[:observable_request] = :thermodynamic_curve
    return _SolvedObservable(curve, curve)
end

function _request_family_support(::HarmonicMultiphononSliceRequest, family::Symbol)
    supported = family === :phonon_mode
    return supported, supported ? () : ("harmonic multiphonon slices require PhononDispersionResult",), true
end

function _solver_request_support(::HarmonicPhononSolver, model::ManyBodyModel, ::HarmonicMultiphononSliceRequest)
    supported, reasons = _harmonic_phonon_model_support(model)
    supported || return false, reasons, true
    get(model.parameters, :spatial_dimension, nothing) == 3 &&
        get(model.parameters, :displacement_dimension, nothing) == 3 ||
        return false, ("the coherent harmonic neutron slice currently requires three Cartesian displacement and reciprocal dimensions",), true
    return true, (), true
end

function _prepare_solver_input(::HarmonicPhononSolver, model::ManyBodyModel, request::HarmonicMultiphononSliceRequest,
                               args::Tuple, kwargs::NamedTuple, ::Nothing)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("multiphonon q mesh is selected by the request, not solver_args"))
    return _PhononDispersionInput(model, phonon_reciprocal_mesh(request.qmesh_shape))
end

# Partial eigenspectra do not satisfy the exact equal-time thermal trace or its sum rules.
_solver_request_support(::OptimizedLanczos, ::ManyBodyModel, ::StaticStructureFactorRequest) =
    (false, ("static equal-time structure factors require a complete eigensystem; optimized Lanczos is partial",), true)
_solver_request_support(::LanczosSolver, ::ManyBodyModel, ::StaticStructureFactorRequest) =
    (false, ("static equal-time structure factors require a complete eigensystem; Lanczos is partial",), true)
_solver_request_support(::AutoSolver, ::ManyBodyModel, ::StaticStructureFactorRequest) =
    (false, ("automatic solver eigenstate completeness is not certified for static equal-time structure factors",), true)
