# Typed forward-calculation specification. This makes the full calculation inspectable and serializable while preserving the existing `forward(...)` convenience API.

"""
    ForwardProblem(model; transformation=nothing, solver=ExactDiagonalization(), solver_args=(), solver_kwargs=(;), method_context=nothing, observable=nothing, probe=nothing, resolution=nothing, metadata=Dict())

Complete specification of a forward calculation, including representation map, numerical solver, optional explicit method-specific context, intrinsic observable request, probe contraction, and instrument resolution. `method_context` configures an already selected computational method; it never determines physical/model-space admissibility or ranks solvers.
"""
struct ForwardProblem{M,T,S,A,K,C,O,P,R,D}
    model::M
    transformation::T
    solver::S
    solver_args::A
    solver_kwargs::K
    method_context::C
    observable::O
    probe::P
    resolution::R
    metadata::D

    # Inner validation also applies to the full positional constructor. Without it,
    # field-exact arguments could bypass the checked keyword constructor.
    function ForwardProblem(model::M, transformation::T, solver::S, solver_args::A, solver_kwargs::K,
                            method_context::C, observable::O, probe::P, resolution::R, metadata::D) where {M,T,S,A,K,C,O,P,R,D}
        model isa ManyBodyModel || throw(ArgumentError("forward model must be a ManyBodyModel"))
        solver isa AbstractSolver || throw(ArgumentError("forward solver must be an AbstractSolver"))
        solver_args isa Tuple || throw(ArgumentError("solver_args must be a Tuple"))
        solver_kwargs isa NamedTuple || throw(ArgumentError("solver_kwargs must be a NamedTuple"))
        (isnothing(transformation) || transformation isa AbstractTransformation) ||
            throw(ArgumentError("transformation must be AbstractTransformation or nothing"))
        (isnothing(method_context) || method_context isa AbstractForwardMethodContext) ||
            throw(ArgumentError("method_context must be AbstractForwardMethodContext or nothing"))
        (!isnothing(observable) || isnothing(probe)) || throw(ArgumentError("probe requires an observable calculation"))
        (!isnothing(observable) || isnothing(resolution)) || throw(ArgumentError("resolution requires an observable calculation"))
        return new{M,T,S,A,K,C,O,P,R,D}(model, transformation, solver, solver_args, solver_kwargs,
                                         method_context, observable, probe, resolution, metadata)
    end
end

function ForwardProblem(model::ManyBodyModel; transformation=nothing, solver::AbstractSolver=ExactDiagonalization(), solver_args::Tuple=(),
                        solver_kwargs::NamedTuple=(;), method_context=nothing, observable=nothing, probe=nothing, resolution=nothing,
                        metadata=Dict{Symbol,Any}())
    return ForwardProblem(model, transformation, solver, solver_args, solver_kwargs, method_context, observable, probe, resolution, metadata)
end

# Backward-compatible positional construction for the pre-Batch-5 field layout.
ForwardProblem(model, transformation, solver, solver_args, solver_kwargs, observable, probe, resolution, metadata) =
    ForwardProblem(model, transformation, solver, solver_args, solver_kwargs, nothing, observable, probe, resolution, metadata)

const CalculationPlan = ForwardProblem

function evaluate_observable_request(observable, model::ManyBodyModel, solution)
    applicable(observable, model, solution) || throw(ArgumentError("observable request must be callable as observable(model, solution)"))
    return observable(model, solution)
end

struct _DirectSolverInput{M,A,K}
    model::M
    args::A
    kwargs::K
end

struct _ProblemSolverInput{P,A,K}
    problem::P
    args::A
    kwargs::K
end

struct _PhononDispersionInput{M,Q}
    model::M
    qpath::Q
end

struct _SpinWaveDispersionInput{M,Q}
    model::M
    qpoints::Q
end

struct _NLCEProblemSolverInput{P,R,C}
    problem::P
    requirements::R
    context::C
end

struct _SolvedObservable{S,O}
    solution::S
    observable_result::O
end

_prepare_solver_input(solver::AbstractSolver, model::ManyBodyModel, observable::Observables.AbstractObservable, args::Tuple, kwargs::NamedTuple) =
    _prepare_solver_input(solver, model, observable, args, kwargs, nothing)

_prepare_solver_input(::AbstractSolver, model::ManyBodyModel, ::Observables.AbstractObservable, args::Tuple, kwargs::NamedTuple, ::Nothing) =
    _DirectSolverInput(model, args, kwargs)

function _prepare_solver_input(::HarmonicPhononSolver, model::ManyBodyModel, observable::Observables.PhononStructureFactorRequest,
                               args::Tuple, kwargs::NamedTuple, ::Nothing)
    isempty(args) || throw(ArgumentError("PhononStructureFactorRequest determines the harmonic q path; solver_args must be empty"))
    isempty(kwargs) || throw(ArgumentError("HarmonicPhononSolver is configured by the solver object; solver_kwargs must be empty for PhononStructureFactorRequest"))
    qpath = Observables._phonon_structure_solver_wavevectors(observable, model)
    return _PhononDispersionInput(model, qpath)
end

function _prepare_solver_input(::SpinWaveSolver, model::ManyBodyModel, observable::Observables.SpinTensorStructureFactorRequest,
                               args::Tuple, kwargs::NamedTuple, ::Nothing)
    scope = get(observable.metadata, :spinwave_momentum_scope, :finite_cell_form_factor)
    scope in (:finite_cell_form_factor, :magnetic_supercell_bloch) || throw(ArgumentError("unrecognized spin-wave momentum scope $scope"))
    if scope === :magnetic_supercell_bloch
        isempty(args) && isempty(kwargs) || throw(ArgumentError("Bloch spin-wave requests determine qpoints; solver_args and solver_kwargs must be empty"))
        return _SpinWaveDispersionInput(model, observable.qgrid)
    end
    return _DirectSolverInput(model, args, kwargs)
end

function _prepare_solver_input(::DMFTSolver, model::ManyBodyModel, ::Observables.AbstractObservable, args::Tuple, kwargs::NamedTuple,
                               context::DMFTForwardContext)
    isempty(args) || throw(ArgumentError("high-level DMFT lowering constructs DMFTProblem from the model and method_context; solver_args must be empty"))
    parameters = _single_band_hubbard_forward_parameters(model)
    problem = DMFTProblem(context.lattice, parameters.interaction; temperature=context.temperature,
                          chemical_potential=parameters.chemical_potential, impurity_energy=context.impurity_energy,
                          kB=context.kB, axis=context.axis)
    return _ProblemSolverInput(problem, (), kwargs)
end

function _prepare_solver_input(::DiagrammaticDMFTSolver, model::ManyBodyModel, ::Observables.MatsubaraSusceptibilityRequest,
                               args::Tuple, kwargs::NamedTuple, context::LadderDGAForwardContext)
    isempty(args) || throw(ArgumentError("high-level ladder-DΓA lowering constructs LadderDGAProblem from method_context; solver_args must be empty"))
    isempty(kwargs) || throw(ArgumentError("DiagrammaticDMFTSolver is configured by the solver object; solver_kwargs must be empty for the high-level ladder-DΓA route"))
    parameters = _single_band_hubbard_forward_parameters(model)
    dmft_problem = DMFTProblem(context.lattice, parameters.interaction; temperature=context.temperature,
                               chemical_potential=parameters.chemical_potential, impurity_energy=context.impurity_energy,
                               kB=context.kB, axis=context.fermionic_axis)
    dmft_result = solve(context.dmft_solver, dmft_problem; context.dmft_solver_kwargs...)
    dmft_result.converged || throw(ArgumentError("ladder-DΓA forward execution requires a converged DMFT reference; termination=$(get(dmft_result.metadata, :termination, :unknown)), residual=$(dmft_result.residual)"))
    problem = LadderDGAProblem(dmft_result, context.lattice, context.bosonic_axis; impurity_backend=context.impurity_backend,
                               impurity_two_particle=context.impurity_two_particle)
    return _ProblemSolverInput(problem, (), (;))
end

function _nlce_additive_observable(value)
    value isa Number && return value
    value isa AbstractArray && return copy(value)
    throw(ArgumentError("NLCE requires an additive scalar, array, ThermodynamicCurve, or MagneticResponseCurve; got $(typeof(value))"))
end

# Return extensive channels in fixed column order; the linked-cluster algorithm acts on values, not presentation metadata.
function _nlce_additive_observable(curve::ThermodynamicCurve)
    size = length(curve.temperatures)
    all(length(values) == size for values in (curve.free_energy, curve.internal_energy, curve.entropy, curve.heat_capacity)) ||
        throw(DimensionMismatch("thermodynamic channels must share one temperature grid"))
    return hcat(curve.free_energy, curve.internal_energy, curve.entropy, curve.heat_capacity)
end

_nlce_additive_observable(curve::MagneticResponseCurve) = copy(curve.values)

function _nlce_cluster_evaluator(observable::Observables.AbstractObservable, context::NLCEForwardContext)
    return function(cluster_model, cluster, cluster_index)
        cluster_metadata = Dict{Symbol,Any}(:nlce_cluster_key => cluster.key, :nlce_cluster_order => cluster.order, :nlce_cluster_index => cluster_index)
        cluster_problem = ForwardProblem(cluster_model; transformation=context.inner_transformation, solver=context.inner_solver,
                                         solver_args=context.inner_solver_args, solver_kwargs=context.inner_solver_kwargs,
                                         method_context=context.inner_method_context, observable=observable, metadata=cluster_metadata)
        return _nlce_additive_observable(forward(cluster_problem).observable_result)
    end
end

function _prepare_solver_input(::NLCESolver, model::ManyBodyModel, observable::Observables.ObservableRequest, args::Tuple,
                               kwargs::NamedTuple, context::NLCEForwardContext)
    isempty(args) || throw(ArgumentError("high-level NLCE lowering constructs NLCEProblem from method_context; solver_args must be empty"))
    isempty(kwargs) || throw(ArgumentError("NLCESolver is configured by the solver object; solver_kwargs must be empty for the high-level NLCE route"))
    components = _nlce_forward_model_components(model)
    evaluator = _nlce_cluster_evaluator(observable, context)
    bonds = isnothing(context.bonds) ? get(model.parameters, :bonds, nothing) : context.bonds
    problem = NLCEProblem(components.material, components.reference, components.specification, evaluator; max_order=context.max_order,
                          bonds=bonds, min_order=context.min_order, max_clusters=context.max_clusters,
                          model_transform=context.model_transform, model_builder=context.model_builder,
                          allow_periodic_reference=context.allow_periodic_reference)
    return _NLCEProblemSolverInput(problem, request_requirements(observable), context)
end

function _prepare_solver_input(::NLCESolver, model::ManyBodyModel,
                               observable::Union{ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest},
                               args::Tuple, kwargs::NamedTuple, context::NLCEForwardContext)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("NLCE solver arguments belong to NLCEForwardContext"))
    components = _nlce_forward_model_components(model)
    bonds = isnothing(context.bonds) ? get(model.parameters, :bonds, nothing) : context.bonds
    problem = NLCEProblem(components.material, components.reference, components.specification,
                          _nlce_cluster_evaluator(observable, context); max_order=context.max_order, bonds=bonds,
                          min_order=context.min_order, max_clusters=context.max_clusters,
                          model_transform=context.model_transform, model_builder=context.model_builder,
                          allow_periodic_reference=context.allow_periodic_reference)
    return _NLCEProblemSolverInput(problem, request_requirements(observable), context)
end


_execute_solver_input(solver::AbstractSolver, input::_DirectSolverInput) = solve(solver, input.model, input.args...; input.kwargs...)
_execute_solver_input(solver::AbstractSolver, input::_ProblemSolverInput) = solve(solver, input.problem, input.args...; input.kwargs...)
_execute_solver_input(solver::HarmonicPhononSolver, input::_PhononDispersionInput) = phonon_dispersion(solver, input.model, input.qpath)
_execute_solver_input(solver::SpinWaveSolver, input::_SpinWaveDispersionInput) = spinwave_dispersion(solver, input.model, input.qpoints)

function _execute_solver_input(solver::DMFTSolver, input::_ProblemSolverInput{P}) where {P<:DMFTProblem}
    result = solve(solver, input.problem, input.args...; input.kwargs...)
    result.metadata[:forward_route] = :single_site_dmft
    result.metadata[:forward_frequency_domain] = :matsubara
    return result
end

function _execute_solver_input(solver::DiagrammaticDMFTSolver, input::_ProblemSolverInput{P}) where {P<:LadderDGAProblem}
    result = solve(solver, input.problem, input.args...; input.kwargs...)
    result.metadata[:forward_route] = :ladder_dga
    result.metadata[:forward_frequency_domain] = :matsubara
    result.metadata[:dmft_reference_preserved] = true
    return result
end

function _execute_solver_input(solver::NLCESolver, input::_NLCEProblemSolverInput)
    solution = solve(solver, input.problem)
    solution.metadata[:forward_route] = :finite_reference_nlce
    if input.problem.specification isa InducedStructuralSpinSpec
        solution.metadata[:structural_family_id] = input.problem.specification.family_id
        solution.metadata[:structural_selection_id] = input.problem.specification.selection_id
        solution.metadata[:cluster_regeneration] = :induced_spin_terms
    end
    solution.metadata[:observable_request] = input.requirements.request_kind
    solution.metadata[:observable_components] = if input.requirements.request_kind === :thermodynamic_curve
        (:free_energy, :internal_energy, :entropy, :heat_capacity)
    elseif input.requirements.request_kind === :magnetic_susceptibility
        (:susceptibility,)
    elseif input.requirements.request_kind === :magnetization_curve
        (:magnetization,)
    else
        ()
    end
    solution.metadata[:inner_solver] = solver_name(input.context.inner_solver)
    solution.metadata[:inner_approximation] = _solver_approximation(input.context.inner_solver)
    solution.metadata[:inner_method_context] = isnothing(input.context.inner_method_context) ? :none : Symbol(nameof(typeof(input.context.inner_method_context)))
    return _SolvedObservable(solution, bulk_estimate(solution))
end

function _solver_compatibility_error(report::SolverCompatibility)
    detail = isempty(report.reasons) ? "no compatible computational route is registered" : join(report.reasons, "; ")
    return ArgumentError("solver $(report.solver) is not computationally compatible with this model/request: $detail")
end

function _attach_planning_metadata!(value, problem::ForwardProblem)
    problem.metadata isa AbstractDict || return value
    report = get(problem.metadata, :forward_planning_report, nothing)
    isnothing(report) && return value
    hasproperty(value, :metadata) || return value
    metadata = getproperty(value, :metadata)
    metadata isa AbstractDict || return value
    metadata[:forward_planned] = true
    metadata[:planned_solver] = report.selected_solver
    metadata[:planned_approximations] = report.selected_approximations
    metadata[:planned_certified] = report.selected_certified
    return value
end

"""Solve the numerical problem required by a typed observable request after checking computational compatibility. Method-specific input construction is isolated in `_prepare_solver_input`; this boundary does not authorize physical approximations or alter model-space admissibility."""
function solve_for_observable(solver::AbstractSolver, model::ManyBodyModel, observable::Observables.AbstractObservable, args::Tuple,
                              kwargs::NamedTuple, method_context=nothing)
    report = solver_compatibility(solver, model, observable, method_context)
    report.compatible || throw(_solver_compatibility_error(report))
    input = _prepare_solver_input(solver, model, observable, args, kwargs, method_context)
    return _execute_solver_input(solver, input)
end

_forward_allows_outer_transformation(::AbstractSolver) = true
_forward_allows_outer_transformation(::DMFTSolver) = false
_forward_allows_outer_transformation(::DiagrammaticDMFTSolver) = false
_forward_allows_outer_transformation(::NLCESolver) = false

function Observables.scattering_intensity(probe::Observables.NuclearNeutronProbe, spectrum::Observables.PhononStructureFactorResult)
    model = spectrum.model
    D = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    ndof = nbasis * D
    size(spectrum.intensity, 3) == ndof || throw(DimensionMismatch("phonon structure-factor displacement dimension is inconsistent with the model"))
    size(spectrum.intensity, 4) == ndof || throw(DimensionMismatch("phonon structure-factor displacement dimension is inconsistent with the model"))
    lengths = Observables._basis_coherent_scattering_lengths(model)
    length(lengths) == nbasis || throw(DimensionMismatch("coherent scattering-length metadata does not match the phonon basis"))

    nq, nx = size(spectrum.intensity, 1), size(spectrum.intensity, 2)
    length(spectrum.q) == nq || throw(DimensionMismatch("phonon structure-factor Q grid does not match the intensity tensor"))
    out = zeros(Float64, nq, nx)

    for iq in 1:nq
        Q = Float64.(spectrum.q[iq])
        length(Q) == D || throw(DimensionMismatch("nuclear neutron Q dimension must match the phonon displacement dimension"))
        coefficients = Vector{ComplexF64}(undef, ndof)
        for site in 1:nbasis, alpha in 1:D
            index = (site - 1) * D + alpha
            coefficients[index] = lengths[site] * Q[alpha]
        end

        for ix in 1:nx
            value = 0.0 + 0.0im
            @inbounds for a in 1:ndof, b in 1:ndof
                value += coefficients[a] * spectrum.intensity[iq, ix, a, b] * conj(coefficients[b])
            end
            scale = max(abs(real(value)), 1.0)
            abs(imag(value)) <= 1e-9 * scale || throw(ArgumentError("nuclear phonon contraction produced a significant imaginary intensity"))
            out[iq, ix] = probe.prefactor * real(value)
        end
    end

    material = model.parameters[:material]
    basis_units = [get(material.species[site.species].metadata, :nuclear_scattering_length_unit, :unspecified) for site in material.crystal.basis]
    metadata = copy(spectrum.metadata)
    metadata[:probe] = :coherent_nuclear_neutron
    metadata[:coherent_scattering_lengths] = lengths
    metadata[:coherent_scattering_length_unit] = length(unique(basis_units)) == 1 ? first(basis_units) : :mixed
    metadata[:coherent_scattering_lengths_applied] = true
    metadata[:neutron_Q_projection_applied] = true
    metadata[:debye_waller_applied] = false
    metadata[:absolute_prefactor] = probe.prefactor
    metadata[:resolution_applied] = false
    return Observables.ScatteringResult(spectrum.q, spectrum.axis, out, metadata)
end

"""Request-aware forward path for typed observables. The generic `forward(problem::ForwardProblem)` method remains available for untyped callables and calculations without an observable request."""
function forward(problem::ForwardProblem{M,T,S,A,K,C,O,P,R,D}) where {M,T,S,A,K,C,O<:Observables.AbstractObservable,P,R,D}
    model = problem.model
    if !isnothing(problem.transformation) && !_forward_allows_outer_transformation(problem.solver)
        throw(ArgumentError("$(solver_name(problem.solver)) high-level lowering derives its computational problem from the physical source model; outer representation transformations are not supported. Configure any method-internal or cluster transformation explicitly in method_context."))
    end

    if isnothing(problem.transformation)
        transformation_result = nothing
        working_model = model
    else
        transformation_result = transform(problem.transformation, model)
        working_model = transformation_result.model
    end

    solved = solve_for_observable(problem.solver, working_model, problem.observable, problem.solver_args, problem.solver_kwargs, problem.method_context)
    if solved isa _SolvedObservable
        solution = solved.solution
        observable_result = solved.observable_result
    else
        solution = solved
        observable_result = evaluate_observable_request(problem.observable, working_model, solution)
    end
    _attach_planning_metadata!(solution, problem)
    _attach_planning_metadata!(observable_result, problem)
    scattering_result = isnothing(problem.probe) ? observable_result : scattering_intensity(problem.probe, observable_result)
    _attach_planning_metadata!(scattering_result, problem)
    final_result = isnothing(problem.resolution) ? scattering_result : convolve_resolution(problem.resolution, scattering_result)
    _attach_planning_metadata!(final_result, problem)
    return ForwardResult(model, working_model, transformation_result, solution, observable_result, scattering_result, final_result)
end

const PhononStructureFactorRequest = Observables.PhononStructureFactorRequest
const PhononStructureFactorResult = Observables.PhononStructureFactorResult

export PhononStructureFactorRequest, PhononStructureFactorResult
