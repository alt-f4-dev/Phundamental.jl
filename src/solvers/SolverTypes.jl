# Shared solver abstractions.

"""Root type for numerical many-body solvers."""
abstract type AbstractSolver end

"""
    solve(solver, model, args...; kwargs...)

Common solver entry point. Concrete solvers provide methods specialized on
`AbstractSolver` subtypes.
"""
function solve(solver::AbstractSolver, model::ManyBodyModel, args...; kwargs...)
    throw(MethodError(solve, (solver, model, args...)))
end

"""Return a short symbolic name for a solver implementation."""
solver_name(solver::AbstractSolver) = Symbol(nameof(typeof(solver)))
