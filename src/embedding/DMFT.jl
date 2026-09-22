# Normal-state single-site dynamical mean-field theory.

"""
    DMFTProblem(lattice, interaction; temperature, chemical_potential=0, impurity_energy=0, nfrequencies=128, kB=1, axis=nothing)

Normal-state single-site, single-orbital DMFT problem on the fermionic Matsubara axis. The first milestone assumes a local scalar self-energy and a spin-degenerate paramagnetic impurity.
"""
struct DMFTProblem{L<:AbstractLatticeEmbedding,A<:FermionicMatsubaraAxis} <: AbstractEmbeddingProblem
    lattice::L
    interaction::Float64
    temperature::Float64
    chemical_potential::Float64
    impurity_energy::Float64
    kB::Float64
    axis::A
end

function DMFTProblem(lattice::AbstractLatticeEmbedding, interaction::Real; temperature::Real, chemical_potential::Real=0.0,
                     impurity_energy::Real=0.0, nfrequencies::Integer=128, kB::Real=1.0, axis=nothing)
    temperature > 0 || throw(ArgumentError("DMFT currently requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    beta = inv(Float64(kB) * Float64(temperature))
    frequency_axis = isnothing(axis) ? FermionicMatsubaraAxis(beta, nfrequencies) : axis
    frequency_axis isa FermionicMatsubaraAxis || throw(ArgumentError("axis must be a FermionicMatsubaraAxis"))
    isapprox(frequency_axis.beta, beta; rtol=1e-12, atol=1e-12 * max(abs(beta), 1.0)) ||
        throw(ArgumentError("provided Matsubara axis does not match temperature and kB"))
    return DMFTProblem(lattice, Float64(interaction), Float64(temperature), Float64(chemical_potential),
                       Float64(impurity_energy), Float64(kB), frequency_axis)
end

"""
    DMFTConvergencePolicy(; stagnation_window=8, stagnation_rtol=1e-3,
                          oscillation_window=8, oscillation_fraction=0.75,
                          divergence_factor=1e6)

Classification policy for a DMFT run that exhausts its iteration budget. The policy does not prematurely stop finite residual trajectories; it classifies them as `:stagnated`, `:oscillatory`, `:diverged`, or `:maxiter` after the fixed-point loop unless a nonfinite residual forces immediate termination.
"""
struct DMFTConvergencePolicy
    stagnation_window::Int
    stagnation_rtol::Float64
    oscillation_window::Int
    oscillation_fraction::Float64
    divergence_factor::Float64
end

function DMFTConvergencePolicy(; stagnation_window::Integer=8, stagnation_rtol::Real=1e-3,
                               oscillation_window::Integer=8, oscillation_fraction::Real=0.75,
                               divergence_factor::Real=1e6)
    stagnation_window >= 3 || throw(ArgumentError("stagnation_window must be at least 3"))
    stagnation_rtol > 0 || throw(ArgumentError("stagnation_rtol must be positive"))
    oscillation_window >= 4 || throw(ArgumentError("oscillation_window must be at least 4"))
    0 < oscillation_fraction <= 1 || throw(ArgumentError("oscillation_fraction must lie in (0,1]"))
    divergence_factor > 1 || throw(ArgumentError("divergence_factor must exceed 1"))
    return DMFTConvergencePolicy(Int(stagnation_window), Float64(stagnation_rtol), Int(oscillation_window),
                                 Float64(oscillation_fraction), Float64(divergence_factor))
end

"""One recorded DMFT fixed-point iteration with numerical convergence diagnostics."""
struct DMFTIteration
    iteration::Int
    residual::Float64
    density::Float64
    double_occupancy::Float64
    bath_fit_objective::Union{Nothing,Float64}
    bath_fit_converged::Union{Nothing,Bool}
    weiss_residual::Float64
    hybridization_residual::Float64
    bath_fit_residual::Float64
    mixing_alpha::Float64
end

DMFTIteration(iteration::Int, residual::Float64, density::Float64, double_occupancy::Float64,
              bath_fit_objective::Union{Nothing,Float64}, bath_fit_converged::Union{Nothing,Bool}) =
    DMFTIteration(iteration, residual, density, double_occupancy, bath_fit_objective, bath_fit_converged, NaN, NaN, NaN, NaN)

"""
    DMFTSolver(; impurity_solver=EDImpuritySolver(), mixing=LinearMixing(0.5), tol=1e-8, maxiter=100,
               require_bath_fit_convergence=false, verify_closure=true, convergence_policy=DMFTConvergencePolicy())

Single-site DMFT fixed-point driver. Successive ED impurity solves warm-start their finite-bath fit from the preceding iteration. `verify_closure=true` preserves the independent final impurity solve used by the validation suite. `AdaptiveLinearMixing` can be supplied to adjust the linear mixing fraction from the observed residual trajectory.
"""
struct DMFTSolver{I<:AbstractImpuritySolver,M<:AbstractMixing,C<:DMFTConvergencePolicy} <: AbstractSolver
    impurity_solver::I
    mixing::M
    tol::Float64
    maxiter::Int
    require_bath_fit_convergence::Bool
    verify_closure::Bool
    convergence_policy::C
end

function DMFTSolver(; impurity_solver::AbstractImpuritySolver=EDImpuritySolver(), mixing::AbstractMixing=LinearMixing(0.5),
                    tol::Real=1e-8, maxiter::Integer=100, require_bath_fit_convergence::Bool=false, verify_closure::Bool=true,
                    convergence_policy::DMFTConvergencePolicy=DMFTConvergencePolicy())
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    return DMFTSolver(impurity_solver, mixing, Float64(tol), Int(maxiter), require_bath_fit_convergence, verify_closure, convergence_policy)
end

# Backward-compatible positional construction for pre-convergence-control solver layouts.
DMFTSolver(impurity_solver::I, mixing::M, tol::Float64, maxiter::Int, require_bath_fit_convergence::Bool) where {I<:AbstractImpuritySolver,M<:AbstractMixing} =
    DMFTSolver(impurity_solver, mixing, tol, maxiter, require_bath_fit_convergence, true, DMFTConvergencePolicy())

DMFTSolver(impurity_solver::I, mixing::M, tol::Float64, maxiter::Int, require_bath_fit_convergence::Bool,
           verify_closure::Bool) where {I<:AbstractImpuritySolver,M<:AbstractMixing} =
    DMFTSolver(impurity_solver, mixing, tol, maxiter, require_bath_fit_convergence, verify_closure, DMFTConvergencePolicy())

solver_name(::DMFTSolver) = :single_site_dmft

"""Result of a single-site DMFT fixed-point calculation."""
struct DMFTResult{P,S,SE,G,W,H,I}
    problem::P
    solver::S
    self_energy::SE
    local_green_function::G
    weiss_green_function::W
    hybridization::H
    impurity_result::I
    converged::Bool
    iterations::Int
    residual::Float64
    history::Vector{DMFTIteration}
    metadata::Dict{Symbol,Any}
end

function _dmft_initial_self_energy(problem::DMFTProblem, initial_self_energy)
    if isnothing(initial_self_energy)
        return zero_self_energy(problem.axis, 1; labels=[:impurity])
    end
    initial_self_energy isa SelfEnergy || throw(ArgumentError("initial_self_energy must be a SelfEnergy"))
    _compatible_frequency_objects(initial_self_energy, zero_self_energy(problem.axis, 1; labels=[:impurity]))
    return initial_self_energy
end

_solve_dmft_impurity(solver::AbstractImpuritySolver, problem::ImpurityProblem, previous_bath) = solve(solver, problem)
_solve_dmft_impurity(solver::EDImpuritySolver, problem::ImpurityProblem, previous_bath) = solve(solver, problem; initial_bath=previous_bath)

function _frequency_object_residual(previous, current)
    _compatible_frequency_objects(previous, current)
    numerator = maximum(abs.(current.values .- previous.values); init=0.0)
    denominator = max(maximum(abs.(current.values); init=0.0), maximum(abs.(previous.values); init=0.0), 1.0)
    return numerator / denominator
end

function _dmft_bath_fit_residual(target::HybridizationFunction, bath::DiscreteBath, chemical_potential::Float64)
    fitted = hybridization_function(bath, target.axis; chemical_potential=chemical_potential)
    _compatible_frequency_objects(target, fitted)
    numerator = norm(fitted.values - target.values)
    return numerator / max(norm(target.values), eps(Float64))
end

function _classify_dmft_termination(history::Vector{DMFTIteration}, tol::Float64, policy::DMFTConvergencePolicy)
    isempty(history) && return :maxiter
    residuals = Float64[entry.residual for entry in history]
    any(residual -> !isfinite(residual), residuals) && return :diverged
    best_residual = minimum(residuals)
    if residuals[end] >= policy.divergence_factor * max(best_residual, tol)
        return :diverged
    end

    if length(residuals) >= policy.stagnation_window
        tail = residuals[end-policy.stagnation_window+1:end]
        scale = max(maximum(abs.(tail)), tol)
        (maximum(tail) - minimum(tail)) / scale <= policy.stagnation_rtol && return :stagnated
    end

    if length(residuals) >= policy.oscillation_window
        tail = residuals[end-policy.oscillation_window+1:end]
        differences = diff(tail)
        sign_changes = 0
        comparisons = 0
        for i in 2:length(differences)
            if !iszero(differences[i - 1]) && !iszero(differences[i])
                comparisons += 1
                signbit(differences[i - 1]) != signbit(differences[i]) && (sign_changes += 1)
            end
        end
        comparisons > 0 && sign_changes / comparisons >= policy.oscillation_fraction && return :oscillatory
    end

    return :maxiter
end

function _dmft_best_history_point(history::Vector{DMFTIteration})
    isempty(history) && return (residual=Inf, iteration=0)
    residuals = Float64[entry.residual for entry in history]
    index = argmin(residuals)
    return (residual=residuals[index], iteration=history[index].iteration)
end

"""
    solve(solver::DMFTSolver, problem::DMFTProblem; initial_self_energy=nothing, initial_bath=nothing)

Run the normal-state single-site DMFT fixed-point iteration. `initial_self_energy` enables continuation between nearby physical points or bath sizes. `initial_bath` additionally warm-starts an ED bath fit when the source and target impurity solvers use the same number of bath sites.
"""
function solve(solver::DMFTSolver, problem::DMFTProblem; initial_self_energy=nothing, initial_bath=nothing)
    sigma = _dmft_initial_self_energy(problem, initial_self_energy)
    history = DMFTIteration[]
    converged = false
    impurity_result = nothing
    residual = Inf
    previous_residual = Inf
    previous_weiss = nothing
    previous_target = nothing
    previous_bath = initial_bath
    mixing_alpha = _mixing_alpha(solver.mixing)
    improving_streak = 0
    termination = :maxiter
    closure_residual = NaN
    closure_verified = false
    closure_attempts = 0
    closure_retry_started = false

    for iteration in 1:solver.maxiter
        lattice_green_function = lattice_green(problem.lattice, problem.axis, sigma; chemical_potential=problem.chemical_potential)
        weiss = weiss_green(lattice_green_function, sigma)
        target = hybridization_from_weiss(weiss; chemical_potential=problem.chemical_potential, impurity_energy=problem.impurity_energy)
        impurity_problem = ImpurityProblem(problem.interaction, problem.axis; chemical_potential=problem.chemical_potential,
                                           impurity_energy=problem.impurity_energy, target_hybridization=target)
        impurity_result = _solve_dmft_impurity(solver.impurity_solver, impurity_problem, previous_bath)
        previous_bath = impurity_result.bath
        candidate = impurity_result.self_energy
        residual = self_energy_residual(sigma, candidate)
        fit_converged = get(impurity_result.metadata, :bath_fit_converged, nothing)
        fit_objective = get(impurity_result.metadata, :bath_fit_objective, nothing)
        weiss_residual = isnothing(previous_weiss) ? NaN : _frequency_object_residual(previous_weiss, weiss)
        hybridization_residual = isnothing(previous_target) ? NaN : _frequency_object_residual(previous_target, target)
        bath_fit_residual = _dmft_bath_fit_residual(target, impurity_result.bath, problem.chemical_potential)
        mixing_alpha, improving_streak = _next_mixing_state(solver.mixing, mixing_alpha, previous_residual, residual, improving_streak)
        push!(history, DMFTIteration(iteration, residual, impurity_result.density, impurity_result.double_occupancy,
                                     fit_objective, fit_converged, weiss_residual, hybridization_residual,
                                     bath_fit_residual, mixing_alpha))

        if !isfinite(residual)
            termination = :diverged
            break
        end

        bath_ok = !solver.require_bath_fit_convergence || fit_converged === true || fit_converged === nothing
        if residual <= solver.tol && bath_ok
            if !solver.verify_closure
                converged = true
                termination = :converged
                break
            end

            closure_attempts += 1
            closure_problem = ImpurityProblem(problem.interaction, problem.axis; chemical_potential=problem.chemical_potential,
                                              impurity_energy=problem.impurity_energy, target_hybridization=target)
            closure_result = _solve_dmft_impurity(solver.impurity_solver, closure_problem, previous_bath)
            trial_closure_residual = self_energy_residual(sigma, closure_result.self_energy)
            closure_fit_converged = get(closure_result.metadata, :bath_fit_converged, nothing)
            closure_bath_ok = !solver.require_bath_fit_convergence || closure_fit_converged === true || closure_fit_converged === nothing

            if !isfinite(trial_closure_residual)
                impurity_result = closure_result
                previous_bath = closure_result.bath
                closure_residual = trial_closure_residual
                closure_verified = true
                residual = closure_residual
                termination = :diverged
                break
            elseif trial_closure_residual <= solver.tol && closure_bath_ok
                impurity_result = closure_result
                previous_bath = closure_result.bath
                closure_residual = trial_closure_residual
                closure_verified = true
                residual = closure_residual
                converged = true
                termination = :converged
                break
            elseif iteration == solver.maxiter
                impurity_result = closure_result
                previous_bath = closure_result.bath
                closure_residual = trial_closure_residual
                closure_verified = true
                residual = closure_residual
                termination = :closure_failed
                break
            end

            closure_retry_started = true
        end

        sigma = solver.mixing isa AdaptiveLinearMixing ? mix_self_energy(solver.mixing, sigma, candidate, mixing_alpha) :
                                                        mix_self_energy(solver.mixing, sigma, candidate)
        previous_residual = residual
        previous_weiss = weiss
        previous_target = target
    end

    isnothing(impurity_result) && throw(ErrorException("DMFT iteration produced no impurity solution"))
    lattice_green_function = lattice_green(problem.lattice, problem.axis, sigma; chemical_potential=problem.chemical_potential)
    weiss = weiss_green(lattice_green_function, sigma)
    target = hybridization_from_weiss(weiss; chemical_potential=problem.chemical_potential, impurity_energy=problem.impurity_energy)
    need_final_closure = !closure_verified && (solver.verify_closure || !converged)

    if need_final_closure
        closure_attempts += 1
        final_problem = ImpurityProblem(problem.interaction, problem.axis; chemical_potential=problem.chemical_potential,
                                        impurity_energy=problem.impurity_energy, target_hybridization=target)
        final_result = _solve_dmft_impurity(solver.impurity_solver, final_problem, previous_bath)
        closure_residual = self_energy_residual(sigma, final_result.self_energy)
        final_fit_converged = get(final_result.metadata, :bath_fit_converged, nothing)
        final_bath_ok = !solver.require_bath_fit_convergence || final_fit_converged === true || final_fit_converged === nothing
        closure_verified = true
        impurity_result = final_result
        previous_bath = final_result.bath
        converged = termination !== :diverged && isfinite(closure_residual) && closure_residual <= solver.tol && final_bath_ok
        residual = closure_residual
        if converged
            termination = :converged
        elseif !isfinite(closure_residual)
            termination = :diverged
        elseif closure_retry_started
            termination = :closure_failed
        elseif termination !== :diverged
            termination = _classify_dmft_termination(history, solver.tol, solver.convergence_policy)
        end
    elseif closure_verified
        residual = closure_residual
    else
        residual = isempty(history) ? Inf : history[end].residual
        termination = converged ? :converged : _classify_dmft_termination(history, solver.tol, solver.convergence_policy)
    end

    best = _dmft_best_history_point(history)
    metadata = Dict{Symbol,Any}(
        :method => :single_site_dmft,
        :normal_state => true,
        :single_orbital => true,
        :paramagnetic => true,
        :converged => converged,
        :termination => termination,
        :iterations => length(history),
        :residual => residual,
        :final_residual => residual,
        :final_iteration_residual => isempty(history) ? Inf : history[end].residual,
        :best_residual => best.residual,
        :best_iteration_residual => best.residual,
        :iteration_of_best_residual => best.iteration,
        :closure_residual => closure_residual,
        :closure_verified => closure_verified,
        :closure_attempts => closure_attempts,
        :closure_retry_started => closure_retry_started,
        :tolerance => solver.tol,
        :chemical_potential => problem.chemical_potential,
        :temperature => problem.temperature,
        :bath_warm_start => solver.impurity_solver isa EDImpuritySolver,
        :continuation_self_energy => !isnothing(initial_self_energy),
        :continuation_bath => !isnothing(initial_bath) && solver.impurity_solver isa EDImpuritySolver,
        :mixing => solver.mixing isa AdaptiveLinearMixing ? :adaptive_linear : :linear,
        :final_mixing_alpha => isempty(history) ? _mixing_alpha(solver.mixing) : history[end].mixing_alpha,
    )
    return DMFTResult(problem, solver, sigma, lattice_green_function, weiss, target, impurity_result, converged,
                      length(history), residual, history, metadata)
end
