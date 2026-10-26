# Normal-state multiorbital single-site ED-DMFT on an explicit matrix momentum lattice.

"""
    MultiOrbitalDMFTProblem(lattice, interaction; temperature, chemical_potential=0, one_body=nothing,
                            kB=1, nfrequencies=128, axis=nothing, onsite_tolerance=1e-9)

A local spin-orbital interaction and a Hermitian matrix Bloch Hamiltonian sharing the
same ordered component labels. The impurity onsite matrix is the normalized
Brillouin-zone mean of the lattice Hamiltonian; an explicitly supplied `one_body`
must agree with that mean, preventing silent local-level double counting.

Only the periodic Bloch gauge is accepted. The interaction is applied once, to
the local impurity, and no analytic continuation is performed.
"""
struct MultiOrbitalDMFTProblem{L<:MatrixMomentumLattice,A<:FermionicMatsubaraAxis} <: AbstractEmbeddingProblem
    lattice::L
    interaction::SpinOrbitalInteraction
    one_body::Matrix{ComplexF64}
    temperature::Float64
    chemical_potential::Float64
    kB::Float64
    axis::A
    labels::Vector{Symbol}
end

function MultiOrbitalDMFTProblem(lattice::MatrixMomentumLattice, interaction::SpinOrbitalInteraction;
                                  temperature::Real, chemical_potential::Real=0.0, one_body=nothing,
                                  kB::Real=1.0, nfrequencies::Integer=128, axis=nothing, onsite_tolerance::Real=1e-9)
    isfinite(temperature) && temperature > 0 || throw(ArgumentError("DMFT temperature must be finite and positive"))
    isfinite(kB) && kB > 0 || throw(ArgumentError("Boltzmann constant must be finite and positive"))
    isfinite(chemical_potential) || throw(ArgumentError("chemical potential must be finite"))
    isfinite(onsite_tolerance) && onsite_tolerance >= 0 || throw(ArgumentError("onsite_tolerance must be finite and nonnegative"))
    nfrequencies > 0 || throw(ArgumentError("nfrequencies must be positive"))
    lattice.gauge === :periodic || throw(ArgumentError("matrix DMFT currently requires the periodic Bloch gauge"))
    n = size(lattice.hamiltonians, 1)
    size(interaction.tensor, 1) == n || throw(DimensionMismatch("interaction and lattice spin-orbital dimensions differ"))
    lattice.labels == interaction.labels || throw(ArgumentError("lattice and interaction spin-orbital labels differ"))
    onsite = zeros(ComplexF64, n, n)
    for ik in eachindex(lattice.weights)
        onsite .+= lattice.weights[ik] .* view(lattice.hamiltonians, :, :, ik)
    end
    onsite = _impurity_hermitian(onsite, "Brillouin-zone averaged local Hamiltonian")
    if !isnothing(one_body)
        given = _impurity_hermitian(one_body, "supplied impurity onsite Hamiltonian")
        size(given) == (n, n) || throw(DimensionMismatch("impurity onsite matrix and lattice dimensions differ"))
        isapprox(given, onsite; atol=onsite_tolerance, rtol=onsite_tolerance) ||
            throw(ArgumentError("impurity onsite matrix must equal the weighted local lattice Hamiltonian; otherwise local energies may be double counted"))
    end
    beta = inv(Float64(kB) * Float64(temperature))
    frequency_axis = isnothing(axis) ? FermionicMatsubaraAxis(beta, nfrequencies) : axis
    frequency_axis isa FermionicMatsubaraAxis || throw(ArgumentError("axis must be a FermionicMatsubaraAxis"))
    isapprox(frequency_axis.beta, beta; atol=1e-12 * max(beta, 1.0), rtol=1e-12) ||
        throw(ArgumentError("Matsubara axis beta disagrees with temperature and kB"))
    return MultiOrbitalDMFTProblem(lattice, interaction, onsite, Float64(temperature), Float64(chemical_potential),
                                   Float64(kB), frequency_axis, copy(lattice.labels))
end

"""Recover the full matrix hybridization `Δ = (z+μ)I-h_local-𝒢₀⁻¹`."""
function hybridization_from_weiss(weiss::NonInteractingGreenFunction, impurity_one_body::AbstractMatrix; chemical_potential::Real=0.0)
    n = matrix_dimension(weiss)
    h = _impurity_hermitian(impurity_one_body, "impurity onsite Hamiltonian")
    size(h) == (n, n) || throw(DimensionMismatch("onsite Hamiltonian and Weiss dimensions differ"))
    isfinite(chemical_potential) || throw(ArgumentError("chemical potential must be finite"))
    identity = Matrix{ComplexF64}(I, n, n)
    values = Array{ComplexF64}(undef, n, n, length(weiss))
    for i in 1:length(weiss)
        values[:, :, i] = (weiss.axis[i] + chemical_potential) .* identity .- h .- _solve_inverse_matrix(weiss[i])
    end
    return HybridizationFunction(weiss.axis, values; labels=weiss.labels,
                                 metadata=Dict{Symbol,Any}(:kind => :matrix_dmft_target, :chemical_potential => Float64(chemical_potential)))
end

"""One matrix-DMFT iteration; bath-fit and closure errors are recorded separately."""
struct MultiOrbitalDMFTIteration
    iteration::Int
    self_energy_residual::Float64
    weiss_residual::Float64
    hybridization_residual::Float64
    bath_fit_residual::Float64
    bath_fit_objective::Union{Nothing,Float64}
    bath_fit_converged::Union{Nothing,Bool}
    mixing_alpha::Float64
end

"""The complete matrix-DMFT fixed-point driver. The scalar `DMFTSolver` is unchanged."""
struct MultiOrbitalDMFTSolver{I<:MultiOrbitalEDSolver,M<:AbstractMixing,C<:DMFTConvergencePolicy} <: AbstractSolver
    impurity_solver::I
    mixing::M
    tol::Float64
    maxiter::Int
    require_bath_fit_convergence::Bool
    convergence_policy::C
end

function MultiOrbitalDMFTSolver(; impurity_solver::MultiOrbitalEDSolver=MultiOrbitalEDSolver(),
                                 mixing::AbstractMixing=LinearMixing(0.5), tol::Real=1e-8, maxiter::Integer=100,
                                 require_bath_fit_convergence::Bool=false,
                                 convergence_policy::DMFTConvergencePolicy=DMFTConvergencePolicy())
    isfinite(tol) && tol > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    return MultiOrbitalDMFTSolver(impurity_solver, mixing, Float64(tol), Int(maxiter),
                                   require_bath_fit_convergence, convergence_policy)
end

solver_name(::MultiOrbitalDMFTSolver) = :multi_orbital_dmft

"""Matrix-DMFT result with final self-energy, lattice and Weiss propagators, impurity solution and closure metadata."""
struct MultiOrbitalDMFTResult{P,S,E,G,W,H,I}
    problem::P
    solver::S
    self_energy::E
    local_green_function::G
    weiss_green_function::W
    target_hybridization::H
    impurity_result::I
    converged::Bool
    iterations::Int
    residual::Float64
    history::Vector{MultiOrbitalDMFTIteration}
    metadata::Dict{Symbol,Any}
end

function _multi_dmft_self_energy(problem::MultiOrbitalDMFTProblem, initial_self_energy)
    expected = zero_self_energy(problem.axis, length(problem.labels); labels=problem.labels)
    if isnothing(initial_self_energy)
        return expected
    end
    initial_self_energy isa SelfEnergy || throw(ArgumentError("initial_self_energy must be SelfEnergy or nothing"))
    _compatible_frequency_objects(initial_self_energy, expected)
    initial_self_energy.labels == problem.labels || throw(ArgumentError("initial self-energy labels differ from the lattice"))
    return SelfEnergy(problem.axis, copy(initial_self_energy.values); labels=problem.labels)
end

# Stream k blocks, avoiding the N_orb² * N_k * N_frequency materialization used by momentum-resolved Green functions.
function _multi_dmft_local_green(problem::MultiOrbitalDMFTProblem, sigma::SelfEnergy)
    n = length(problem.labels)
    _compatible_frequency_objects(sigma, zero_self_energy(problem.axis, n; labels=problem.labels))
    sigma.labels == problem.labels || throw(ArgumentError("self-energy component labels differ from the lattice"))
    lattice = problem.lattice
    values = zeros(ComplexF64, n, n, length(problem.axis))
    kernel = Matrix{ComplexF64}(undef, n, n)
    for i in 1:length(problem.axis)
        for ik in eachindex(lattice.weights)
            kernel .= -view(lattice.hamiltonians, :, :, ik) .- sigma[i]
            for a in 1:n
                kernel[a, a] += problem.axis[i] + problem.chemical_potential
            end
            values[:, :, i] .+= lattice.weights[ik] .* _solve_inverse_matrix(kernel)
        end
    end
    return GreenFunction(problem.axis, values; labels=problem.labels,
                         metadata=Dict{Symbol,Any}(:lattice => :matrix_momentum, :nk => length(lattice.weights),
                                                   :integration => :streaming, :bloch_gauge => lattice.gauge))
end

function _multi_dmft_fields(problem::MultiOrbitalDMFTProblem, sigma::SelfEnergy)
    local_green = _multi_dmft_local_green(problem, sigma)
    weiss = weiss_green(local_green, sigma)
    target = hybridization_from_weiss(weiss, problem.one_body; chemical_potential=problem.chemical_potential)
    return local_green, weiss, target
end

function _multi_dmft_impurity(problem::MultiOrbitalDMFTProblem, target::HybridizationFunction)
    return MultiOrbitalImpurityProblem(problem.one_body, problem.interaction, problem.axis;
                                      chemical_potential=problem.chemical_potential, target_hybridization=target)
end

# The relative fit error is ill-conditioned when the exact hybridization is zero.
# Retain it for nonzero targets, but also report the absolute Frobenius error.
function _multi_dmft_bath_errors(target::HybridizationFunction, bath::MatrixDiscreteBath, mu::Float64)
    fitted = hybridization_function(bath, target.axis; chemical_potential=mu)
    _compatible_frequency_objects(target, fitted)
    absolute = norm(fitted.values - target.values)
    target_norm = norm(target.values)
    return (; relative=absolute / max(target_norm, eps(Float64)), absolute, target_norm)
end

_multi_dmft_bath_residual(target::HybridizationFunction, bath::MatrixDiscreteBath, mu::Float64) =
    _multi_dmft_bath_errors(target, bath, mu).relative

function _multi_dmft_classification(history::Vector{MultiOrbitalDMFTIteration}, solver::MultiOrbitalDMFTSolver)
    translated = DMFTIteration[DMFTIteration(h.iteration, h.self_energy_residual, NaN, NaN, h.bath_fit_objective,
                                             h.bath_fit_converged, h.weiss_residual, h.hybridization_residual,
                                             h.bath_fit_residual, h.mixing_alpha) for h in history]
    return _classify_dmft_termination(translated, solver.tol, solver.convergence_policy)
end

"""
    solve(solver::MultiOrbitalDMFTSolver, problem::MultiOrbitalDMFTProblem;
          initial_self_energy=nothing, initial_bath=nothing)

Iterate the full complex local matrix self-energy. Every returned result is
recomputed at the final self-energy; an independent closure solve is mandatory
for a `converged=true` result. Optimizer convergence and the relative matrix
bath-fit discrepancy are independent from the DMFT fixed-point residual.
"""
function solve(solver::MultiOrbitalDMFTSolver, problem::MultiOrbitalDMFTProblem;
               initial_self_energy=nothing, initial_bath=nothing)
    if !isnothing(initial_bath)
        initial_bath isa MatrixDiscreteBath || throw(ArgumentError("initial_bath must be a MatrixDiscreteBath"))
        length(initial_bath) == solver.impurity_solver.bath_sites || throw(DimensionMismatch("initial bath has an incompatible number of sites"))
        initial_bath.labels == problem.labels || throw(ArgumentError("initial bath labels disagree with the lattice"))
    end
    # Fail before any expensive lattice or fit computation if the ED basis is too large.
    _multi_ed_dimensions(length(problem.labels) + solver.impurity_solver.bath_sites, solver.impurity_solver)
    beta_expected = inv(problem.kB * problem.temperature)
    isapprox(problem.axis.beta, beta_expected; atol=1e-12 * max(beta_expected, 1.0), rtol=1e-12) ||
        throw(ArgumentError("DMFT axis and impurity ED temperature are inconsistent"))
    isapprox(solver.impurity_solver.kB, problem.kB; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("DMFT and impurity solvers must use the same Boltzmann constant"))
    sigma = _multi_dmft_self_energy(problem, initial_self_energy)
    previous_bath = initial_bath
    previous_weiss = nothing
    previous_target = nothing
    previous_residual = Inf
    mixing_alpha = _mixing_alpha(solver.mixing)
    improving_streak = 0
    history = MultiOrbitalDMFTIteration[]
    termination = :maxiter
    closure_attempts = 0
    converged = false

    for iteration in 1:solver.maxiter
        local_green, weiss, target = _multi_dmft_fields(problem, sigma)
        impurity = solve(solver.impurity_solver, _multi_dmft_impurity(problem, target); initial_bath=previous_bath)
        previous_bath = impurity.bath
        candidate = impurity.self_energy
        residual = self_energy_residual(sigma, candidate)
        fit_converged = get(impurity.metadata, :bath_fit_converged, nothing)
        fit_objective = get(impurity.metadata, :bath_fit_objective, nothing)
        bath_residual = _multi_dmft_bath_residual(target, impurity.bath, problem.chemical_potential)
        weiss_residual = isnothing(previous_weiss) ? NaN : _frequency_object_residual(previous_weiss, weiss)
        hybridization_residual = isnothing(previous_target) ? NaN : _frequency_object_residual(previous_target, target)
        mixing_alpha, improving_streak = _next_mixing_state(solver.mixing, mixing_alpha, previous_residual, residual, improving_streak)
        push!(history, MultiOrbitalDMFTIteration(iteration, residual, weiss_residual, hybridization_residual,
                                                 bath_residual, fit_objective, fit_converged, mixing_alpha))
        if !isfinite(residual)
            termination = :diverged
            break
        end
        fit_ok = !solver.require_bath_fit_convergence || fit_converged === true
        if residual <= solver.tol && fit_ok
            termination = :candidate_converged
            # Retain the current self-energy so that final closure checks exactly this fixed point.
            break
        end
        if iteration < solver.maxiter
            sigma = solver.mixing isa AdaptiveLinearMixing ? mix_self_energy(solver.mixing, sigma, candidate, mixing_alpha) :
                                                            mix_self_energy(solver.mixing, sigma, candidate)
        end
        previous_weiss = weiss
        previous_target = target
        previous_residual = residual
    end

    # Reconstruct all reported propagators at the final self-energy, never at a stale iteration.
    local_green, weiss, target = _multi_dmft_fields(problem, sigma)
    closure_attempts += 1
    impurity = solve(solver.impurity_solver, _multi_dmft_impurity(problem, target); initial_bath=previous_bath)
    closure_residual = self_energy_residual(sigma, impurity.self_energy)
    fit_converged = get(impurity.metadata, :bath_fit_converged, nothing)
    bath_errors = _multi_dmft_bath_errors(target, impurity.bath, problem.chemical_potential)
    bath_residual = bath_errors.relative
    bath_ok = !solver.require_bath_fit_convergence || fit_converged === true
    converged = isfinite(closure_residual) && closure_residual <= solver.tol && bath_ok
    if converged
        termination = :converged
    elseif !isfinite(closure_residual)
        termination = :diverged
    elseif termination === :candidate_converged
        termination = :closure_failed
    elseif termination !== :diverged
        termination = _multi_dmft_classification(history, solver)
    end
    residual = closure_residual
    best_index = argmin([entry.self_energy_residual for entry in history])
    best = history[best_index]
    metadata = Dict{Symbol,Any}(
        :method => :single_site_multi_orbital_dmft,
        :normal_state => true,
        :correlation_scope => :local,
        :spin_orbital_labels => copy(problem.labels),
        :converged => converged,
        :termination => termination,
        :iterations => length(history),
        :residual => residual,
        :closure_residual => closure_residual,
        :closure_verified => true,
        :closure_attempts => closure_attempts,
        :final_iteration_residual => history[end].self_energy_residual,
        :best_residual => best.self_energy_residual,
        :best_iteration => best.iteration,
        :bath_fit_converged => fit_converged,
        :bath_fit_objective => get(impurity.metadata, :bath_fit_objective, nothing),
        :bath_fit_residual => bath_residual,
        :bath_fit_absolute_residual => bath_errors.absolute,
        :bath_target_norm => bath_errors.target_norm,
        :bath_sites => length(impurity.bath),
        :mixing => solver.mixing isa AdaptiveLinearMixing ? :adaptive_linear : :linear,
        :final_mixing_alpha => history[end].mixing_alpha,
        :continuation_self_energy => !isnothing(initial_self_energy),
        :continuation_bath => !isnothing(initial_bath),
        :temperature => problem.temperature,
        :chemical_potential => problem.chemical_potential,
        :kB => problem.kB,
    )
    return MultiOrbitalDMFTResult(problem, solver, sigma, local_green, weiss, target, impurity,
                                   converged, length(history), residual, history, metadata)
end

"""Return the relative difference between the final interacting impurity and local lattice Green functions."""
function impurity_local_residual(result::MultiOrbitalDMFTResult)
    impurity = result.impurity_result.green_function
    local_green = result.local_green_function
    _compatible_frequency_objects(impurity, local_green)
    return norm(impurity.values - local_green.values) / max(norm(impurity.values), norm(local_green.values), 1.0)
end

"""Verify matrix Dyson, Weiss, bath fit, independent closure and self-energy convergence."""
function validate_dmft_result(result::MultiOrbitalDMFTResult; atol::Real=1e-9, rtol::Real=1e-8)
    isfinite(atol) && atol >= 0 && isfinite(rtol) && rtol >= 0 || throw(ArgumentError("tolerances must be finite and nonnegative"))
    problem = result.problem
    local_green, weiss, target = _multi_dmft_fields(problem, result.self_energy)
    fields_ok = isapprox(local_green.values, result.local_green_function.values; atol=atol, rtol=rtol) &&
                isapprox(weiss.values, result.weiss_green_function.values; atol=atol, rtol=rtol) &&
                isapprox(target.values, result.target_hybridization.values; atol=atol, rtol=rtol)
    closure_residual = self_energy_residual(result.self_energy, result.impurity_result.self_energy)
    bath_errors = _multi_dmft_bath_errors(target, result.impurity_result.bath, problem.chemical_potential)
    closure_ok = isapprox(closure_residual, result.metadata[:closure_residual]; atol=atol, rtol=rtol)
    bath_ok = isapprox(bath_errors.relative, result.metadata[:bath_fit_residual]; atol=atol, rtol=rtol)
    bath_absolute_ok = isapprox(bath_errors.absolute, result.metadata[:bath_fit_absolute_residual]; atol=atol, rtol=rtol)
    bath_target_ok = isapprox(bath_errors.target_norm, result.metadata[:bath_target_norm]; atol=atol, rtol=rtol)
    accepted = fields_ok && closure_ok && bath_ok && bath_absolute_ok && bath_target_ok &&
               result.metadata[:closure_verified] === true &&
               (!result.converged || closure_residual <= result.solver.tol &&
                (!result.solver.require_bath_fit_convergence || result.metadata[:bath_fit_converged] === true))
    return (; valid=accepted, fields_ok, closure_ok, bath_ok, bath_absolute_ok, bath_target_ok, closure_residual,
            bath_fit_residual=bath_errors.relative, bath_fit_absolute_residual=bath_errors.absolute,
            bath_target_norm=bath_errors.target_norm, impurity_local_residual=impurity_local_residual(result))
end
