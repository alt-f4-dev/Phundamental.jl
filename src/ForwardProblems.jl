# Typed forward-calculation specification. This makes the full calculation inspectable and serializable while preserving the existing `forward(...)` convenience API.

"""
    ForwardProblem(model; transformation=nothing, solver=ExactDiagonalization(), solver_args=(), solver_kwargs=(;), observable=nothing, probe=nothing, resolution=nothing, metadata=Dict())

Complete specification of a forward calculation, including representation map, numerical solver, intrinsic observable request, probe contraction, and instrument resolution.
"""
struct ForwardProblem{M,T,S,A,K,O,P,R,D}
    model::M
    transformation::T
    solver::S
    solver_args::A
    solver_kwargs::K
    observable::O
    probe::P
    resolution::R
    metadata::D
end

function ForwardProblem(model::ManyBodyModel; transformation=nothing, solver::AbstractSolver=ExactDiagonalization(), solver_args::Tuple=(),
                        solver_kwargs::NamedTuple=(;), observable=nothing, probe=nothing, resolution=nothing, metadata=Dict{Symbol,Any}())
    !isnothing(transformation) && !(transformation isa AbstractTransformation) && throw(ArgumentError("transformation must be AbstractTransformation or nothing"))
    !isnothing(probe) && isnothing(observable) && throw(ArgumentError("probe requires an observable calculation"))
    !isnothing(resolution) && isnothing(observable) && throw(ArgumentError("resolution requires an observable calculation"))
    return ForwardProblem(model, transformation, solver, solver_args, solver_kwargs, observable, probe, resolution, metadata)
end

const CalculationPlan = ForwardProblem

function evaluate_observable_request(observable, model::ManyBodyModel, solution)
    applicable(observable, model, solution) || throw(ArgumentError("observable request must be callable as observable(model, solution)"))
    return observable(model, solution)
end

"""Solve the numerical problem required by a typed observable request. The default preserves the ordinary `solve(solver, model, args...; kwargs...)` behavior."""
solve_for_observable(solver::AbstractSolver, model::ManyBodyModel, observable, args::Tuple, kwargs::NamedTuple) = solve(solver, model, args...; kwargs...)

function solve_for_observable(solver::HarmonicPhononSolver, model::ManyBodyModel, observable::Observables.PhononStructureFactorRequest,
                              args::Tuple, kwargs::NamedTuple)
    isempty(args) || throw(ArgumentError("PhononStructureFactorRequest determines the harmonic q path; solver_args must be empty"))
    isempty(kwargs) || throw(ArgumentError("HarmonicPhononSolver is configured by the solver object; solver_kwargs must be empty for PhononStructureFactorRequest"))
    qpath = Observables._phonon_structure_solver_wavevectors(observable, model)
    return phonon_dispersion(solver, model, qpath)
end

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
function forward(problem::ForwardProblem{M,T,S,A,K,O,P,R,D}) where {M,T,S,A,K,O<:Observables.AbstractObservable,P,R,D}
    model = problem.model
    if isnothing(problem.transformation)
        transformation_result = nothing
        working_model = model
    else
        transformation_result = transform(problem.transformation, model)
        working_model = transformation_result.model
    end

    solution = solve_for_observable(problem.solver, working_model, problem.observable, problem.solver_args, problem.solver_kwargs)
    observable_result = evaluate_observable_request(problem.observable, working_model, solution)
    scattering_result = isnothing(problem.probe) ? observable_result : scattering_intensity(problem.probe, observable_result)
    final_result = isnothing(problem.resolution) ? scattering_result : convolve_resolution(problem.resolution, scattering_result)
    return ForwardResult(model, working_model, transformation_result, solution, observable_result, scattering_result, final_result)
end

const PhononStructureFactorRequest = Observables.PhononStructureFactorRequest
const PhononStructureFactorResult = Observables.PhononStructureFactorResult

export PhononStructureFactorRequest, PhononStructureFactorResult
