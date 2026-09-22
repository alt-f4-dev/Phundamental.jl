# Typed forward-calculation specification. This makes the full calculation
# inspectable and serializable while preserving the existing `forward(...)`
# convenience API.

"""
    ForwardProblem(model; transformation=nothing, solver=ExactDiagonalization(), solver_args=(), solver_kwargs=(;),
                   observable=nothing, probe=nothing, resolution=nothing, metadata=Dict())

Complete specification of a forward calculation, including representation
map, numerical solver, intrinsic observable request, probe contraction, and
instrument resolution.
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

function ForwardProblem(model::ManyBodyModel; transformation=nothing, solver::AbstractSolver=ExactDiagonalization(),
                        solver_args::Tuple=(), solver_kwargs::NamedTuple=(;), observable=nothing, probe=nothing,
                        resolution=nothing, metadata=Dict{Symbol,Any}())
    !isnothing(transformation) && !(transformation isa AbstractTransformation) &&
        throw(ArgumentError("transformation must be AbstractTransformation or nothing"))
    !isnothing(probe) && isnothing(observable) && throw(ArgumentError("probe requires an observable calculation"))
    !isnothing(resolution) && isnothing(observable) && throw(ArgumentError("resolution requires an observable calculation"))
    return ForwardProblem(model, transformation, solver, solver_args, solver_kwargs, observable, probe, resolution, metadata)
end

const CalculationPlan = ForwardProblem

function evaluate_observable_request(observable, model::ManyBodyModel, solution)
    applicable(observable, model, solution) || throw(ArgumentError("observable request must be callable as observable(model, solution)"))
    return observable(model, solution)
end
