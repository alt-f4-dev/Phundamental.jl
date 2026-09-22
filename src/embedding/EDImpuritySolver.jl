# Exact-diagonalization impurity solver with finite-bath discretization.

"""
    EDImpuritySolver(; bath_sites=4, bath_fit=BathFitOptions(), ed_solver=ExactDiagonalization(), kB=1, backend=:auto, threading=:serial)

Finite-temperature single-orbital Anderson impurity solver. A target hybridization is fitted to `bath_sites` spin-degenerate bath orbitals. `backend=:auto` uses spin-resolved `(N_up,N_down)` sector exact diagonalization for the normal Anderson model and retains `backend=:dense` as an independent small-system reference path. `threading=:serial` preserves deterministic single-thread execution, while `:threads` and `:auto` use Julia tasks for independent sector diagonalizations and Lehmann transition blocks when more than one Julia thread is available.
"""
struct EDImpuritySolver{O<:BathFitOptions,E<:ExactDiagonalization} <: AbstractImpuritySolver
    bath_sites::Int
    bath_fit::O
    ed_solver::E
    kB::Float64
    backend::Symbol
    threading::Symbol
end

function EDImpuritySolver(; bath_sites::Integer=4, bath_fit::BathFitOptions=BathFitOptions(), ed_solver::ExactDiagonalization=ExactDiagonalization(),
                          kB::Real=1.0, backend::Symbol=:auto, threading::Symbol=:serial)
    bath_sites >= 0 || throw(ArgumentError("bath_sites must be nonnegative"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    backend in (:auto, :dense, :sectorized) || throw(ArgumentError("backend must be :auto, :dense, or :sectorized"))
    _validate_impurity_threading(threading)
    return EDImpuritySolver(Int(bath_sites), bath_fit, ed_solver, Float64(kB), backend, threading)
end

# Backward-compatible positional construction for the pre-optimization solver layouts.
EDImpuritySolver(bath_sites::Int, bath_fit::O, ed_solver::E, kB::Float64) where {O<:BathFitOptions,E<:ExactDiagonalization} =
    EDImpuritySolver(bath_sites, bath_fit, ed_solver, kB, :auto, :serial)

EDImpuritySolver(bath_sites::Int, bath_fit::O, ed_solver::E, kB::Float64, backend::Symbol) where {O<:BathFitOptions,E<:ExactDiagonalization} =
    EDImpuritySolver(bath_sites, bath_fit, ed_solver, kB, backend, :serial)

solver_name(::EDImpuritySolver) = :ed_impurity

function _selected_impurity_backend(solver::EDImpuritySolver)
    solver.backend === :dense && return :dense
    solver.backend === :sectorized && solver.ed_solver.hermitian === :no &&
        throw(ArgumentError("backend=:sectorized requires a Hermitian exact-diagonalization configuration"))
    solver.backend === :sectorized && return :sectorized
    return solver.ed_solver.hermitian === :no ? :dense : :sectorized
end

function _impurity_noninteracting_green(problem::ImpurityProblem, bath::DiscreteBath)
    delta = hybridization_function(bath, problem.axis; chemical_potential=problem.chemical_potential)
    values = ComplexF64[
        inv(problem.axis[i] + problem.chemical_potential - problem.impurity_energy - delta[1, 1, i])
        for i in 1:length(problem.axis)
    ]
    metadata = Dict{Symbol,Any}(:kind => :anderson_weiss, :chemical_potential => problem.chemical_potential,
                                :impurity_energy => problem.impurity_energy, :bath_sites => length(bath))
    return NonInteractingGreenFunction(problem.axis, values; labels=[:impurity], metadata=metadata), delta
end

function _paramagnetic_impurity_green(state::ExactGibbsState, axis::FermionicMatsubaraAxis)
    up = matsubara_green(state, axis; modes=[1], labels=[:impurity_up])
    down = matsubara_green(state, axis; modes=[2], labels=[:impurity_down])
    values = 0.5 .* (up.values .+ down.values)
    spin_difference = maximum(abs.(up.values .- down.values); init=0.0)
    green = GreenFunction(axis, values; labels=[:impurity],
                          metadata=Dict{Symbol,Any}(:method => :exact_lehmann, :paramagnetic_average => true, :spin_difference => spin_difference))
    return green, spin_difference
end

function _dense_impurity_state(model::ManyBodyModel, solver::EDImpuritySolver, temperature::Float64)
    method = ExactGibbs(solver=solver.ed_solver)
    return thermal_state(model, method; temperature=temperature, kB=solver.kB)
end

function _sectorized_impurity_state(model::ManyBodyModel, specification::AndersonImpurityModel, solver::EDImpuritySolver, temperature::Float64)
    return _sectorized_impurity_state(model, specification; temperature=temperature, kB=solver.kB, threading=solver.threading)
end

function _solve_impurity_state(model::ManyBodyModel, specification::AndersonImpurityModel, solver::EDImpuritySolver, temperature::Float64)
    backend = _selected_impurity_backend(solver)
    state = backend === :dense ? _dense_impurity_state(model, solver, temperature) : _sectorized_impurity_state(model, specification, solver, temperature)
    return state, backend
end

"""
    solve(solver::EDImpuritySolver, problem::ImpurityProblem; initial_bath=nothing)

Solve a typed impurity problem and return its Green function, self-energy, density, double occupancy, fitted bath, and metadata. `initial_bath` is used only when fitting a target hybridization and provides a warm start for successive DMFT iterations.
"""
function solve(solver::EDImpuritySolver, problem::ImpurityProblem; initial_bath=nothing)
    !isnothing(initial_bath) && !(initial_bath isa DiscreteBath) && throw(ArgumentError("initial_bath must be a DiscreteBath or nothing"))
    bath_fit = nothing
    bath = if isnothing(problem.bath)
        bath_fit = fit_bath(problem.target_hybridization, solver.bath_sites; chemical_potential=problem.chemical_potential,
                            options=solver.bath_fit, initial_bath=initial_bath)
        bath_fit.bath
    else
        problem.bath
    end

    specification = AndersonImpurityModel(problem.interaction; chemical_potential=problem.chemical_potential,
                                          impurity_energy=problem.impurity_energy, bath_energies=bath.energies,
                                          hybridizations=bath.hybridizations)
    model = build_model(specification)
    temperature = axis_temperature(problem.axis; kB=solver.kB)
    state, backend = _solve_impurity_state(model, specification, solver, temperature)
    green, spin_difference = _paramagnetic_impurity_green(state, problem.axis)
    green0, delta = _impurity_noninteracting_green(problem, bath)
    sigma = self_energy(green0, green)
    density = Float64(real(expectation(state, :impurity_density).value))
    double_occupancy = Float64(real(expectation(state, :impurity_double_occupancy).value))

    full_dimension = state isa SectorizedImpurityThermalState ? state.workspace.full_dimension : length(state.solution.basis)
    max_sector_dimension = state isa SectorizedImpurityThermalState ? state.workspace.max_sector_dimension : full_dimension
    metadata = Dict{Symbol,Any}(
        :solver => :ed_impurity,
        :backend => backend,
        :bath_sites => length(bath),
        :temperature => temperature,
        :beta => problem.axis.beta,
        :chemical_potential => problem.chemical_potential,
        :interaction => problem.interaction,
        :spin_difference => spin_difference,
        :hybridization => delta,
        :full_hilbert_dimension => full_dimension,
        :max_sector_dimension => max_sector_dimension,
        :threading_requested => solver.threading,
        :threading_effective => state isa SectorizedImpurityThermalState ? get(state.metadata, :threading_effective, :serial) : :serial,
        :julia_threads => Threads.nthreads(),
        :bath_fit_converged => isnothing(bath_fit) ? nothing : bath_fit.converged,
        :bath_fit_objective => isnothing(bath_fit) ? nothing : bath_fit.objective,
        :bath_fit_iterations => isnothing(bath_fit) ? nothing : bath_fit.iterations,
        :bath_fit_metadata => isnothing(bath_fit) ? nothing : copy(bath_fit.metadata),
        :warm_start_used => isnothing(bath_fit) ? false : get(bath_fit.metadata, :warm_start_used, false),
    )
    return ImpuritySolverResult(model, green, green0, sigma, density, double_occupancy, bath, state, metadata)
end
