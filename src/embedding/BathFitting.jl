# Deterministic nonlinear least-squares fitting of a Matsubara hybridization to a finite ED bath.

"""
    BathFitOptions(; maxiter=200, tol=1e-10, damping=1e-4, weight_power=1, energy_window=nothing, hybridization_max=Inf, multistart=3, symmetry=:none)

Configuration for finite-bath fitting. The objective uses frequency weights proportional to `|ωₙ|^(-weight_power)` so low Matsubara frequencies can be emphasized without changing the target hybridization itself. `symmetry=:particle_hole` constrains an even bath to energy pairs about the supplied chemical potential with equal hybridization magnitudes inside each pair.
"""
struct BathFitOptions
    maxiter::Int
    tol::Float64
    damping::Float64
    weight_power::Float64
    energy_window::Union{Nothing,Tuple{Float64,Float64}}
    hybridization_max::Float64
    multistart::Int
    symmetry::Symbol
end

function BathFitOptions(; maxiter::Integer=200, tol::Real=1e-10, damping::Real=1e-4, weight_power::Real=1.0,
                        energy_window=nothing, hybridization_max::Real=Inf, multistart::Integer=3, symmetry::Symbol=:none)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isfinite(damping) && damping > 0 || throw(ArgumentError("damping must be finite and positive"))
    isfinite(weight_power) && weight_power >= 0 || throw(ArgumentError("weight_power must be finite and nonnegative"))
    multistart > 0 || throw(ArgumentError("multistart must be positive"))
    symmetry in (:none, :particle_hole) || throw(ArgumentError("symmetry must be :none or :particle_hole"))
    (isfinite(hybridization_max) || hybridization_max == Inf) && hybridization_max > 0 ||
        throw(ArgumentError("hybridization_max must be positive and finite or Inf"))
    window = if isnothing(energy_window)
        nothing
    else
        lo, hi = Float64(energy_window[1]), Float64(energy_window[2])
        isfinite(lo) && isfinite(hi) || throw(ArgumentError("energy_window endpoints must be finite"))
        lo < hi || throw(ArgumentError("energy_window must satisfy lower < upper"))
        (lo, hi)
    end
    return BathFitOptions(Int(maxiter), Float64(tol), Float64(damping), Float64(weight_power), window,
                          Float64(hybridization_max), Int(multistart), symmetry)
end

BathFitOptions(maxiter::Int, tol::Float64, damping::Float64, weight_power::Float64,
               energy_window::Union{Nothing,Tuple{Float64,Float64}}, hybridization_max::Float64, multistart::Int) =
    BathFitOptions(maxiter, tol, damping, weight_power, energy_window, hybridization_max, multistart, :none)

"""Finite-bath fit result including the optimized bath and numerical convergence metadata."""
struct BathFitResult
    bath::DiscreteBath
    objective::Float64
    converged::Bool
    iterations::Int
    metadata::Dict{Symbol,Any}
end

function _bath_fit_window(target::HybridizationFunction, chemical_potential::Float64, options::BathFitOptions)
    !isnothing(options.energy_window) && return options.energy_window
    data = scalar_frequency_values(target)
    nprobe = min(length(data), 8)
    scale = maximum(abs.(view(data, 1:nprobe)))
    span = max(1.0, 4scale)
    return (chemical_potential - span, chemical_potential + span)
end

function _bath_fit_weights(target::HybridizationFunction, options::BathFitOptions)
    omega = abs.(imag.(target.axis.values))
    floor_frequency = minimum(omega)
    weights = (max.(omega, floor_frequency) ./ floor_frequency) .^ (-options.weight_power)
    weights ./= sum(weights)
    return weights
end


function _particle_hole_offset_limit(chemical_potential::Float64, window::Tuple{Float64,Float64})
    lo, hi = window
    limit = min(chemical_potential - lo, hi - chemical_potential)
    isfinite(limit) && limit > 0 || throw(ArgumentError("particle-hole-symmetric bath fitting requires the chemical potential to lie strictly inside energy_window"))
    return limit
end

function _particle_hole_initial_parameters(target::HybridizationFunction, nbath::Int, chemical_potential::Float64,
                                           window::Tuple{Float64,Float64}, start::Int, nstarts::Int)
    iseven(nbath) || throw(ArgumentError("particle-hole-symmetric bath fitting requires an even number of bath sites"))
    npairs = nbath ÷ 2
    npairs == 0 && return Float64[]
    limit = _particle_hole_offset_limit(chemical_potential, window)
    phase = nstarts == 1 ? 0.0 : (start - 1) / nstarts
    offsets = Vector{Float64}(undef, npairs)
    for pair in 1:npairs
        base = (pair - 0.5) / npairs
        shifted = mod(base + 0.35 * phase / max(npairs, 1), 1.0)
        offsets[pair] = shifted * limit
    end
    sort!(offsets)

    order = sortperm(collect(1:length(target)); by=i -> abs(target.axis[i]), rev=true)
    ntail = min(length(order), 8)
    tail = order[1:ntail]
    delta = scalar_frequency_values(target)
    sumv2 = sum(max(real(target.axis[i] * delta[i]), 0.0) for i in tail) / ntail
    if !(isfinite(sumv2) && sumv2 > 0)
        sumv2 = max(maximum(abs.(delta)) * max(2limit, 1.0), 1e-8)
    end
    coupling = sqrt(sumv2 / nbath)
    return vcat(offsets, fill(coupling, npairs))
end

function _particle_hole_initial_parameters(bath::DiscreteBath, nbath::Int, chemical_potential::Float64,
                                           window::Tuple{Float64,Float64}, hybridization_max::Float64)
    length(bath) == nbath || throw(DimensionMismatch("initial bath has $(length(bath)) sites but fit requests $nbath"))
    iseven(nbath) || throw(ArgumentError("particle-hole-symmetric bath fitting requires an even number of bath sites"))
    npairs = nbath ÷ 2
    npairs == 0 && return Float64[]
    permutation = sortperm(bath.energies)
    offsets = Vector{Float64}(undef, npairs)
    couplings = Vector{Float64}(undef, npairs)
    for pair in 1:npairs
        opposite = nbath + 1 - pair
        lower = permutation[pair]
        upper = permutation[opposite]
        offsets[pair] = max(0.0, 0.5 * (bath.energies[upper] - bath.energies[lower]))
        couplings[pair] = 0.5 * (abs(bath.hybridizations[lower]) + abs(bath.hybridizations[upper]))
    end
    order = sortperm(offsets)
    parameters = vcat(offsets[order], couplings[order])
    all(isfinite, parameters) || throw(ArgumentError("initial bath contains nonfinite parameters"))
    return _project_particle_hole_parameters!(parameters, chemical_potential, window, hybridization_max)
end

function _particle_hole_residual_jacobian(parameters::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64,
                                          weights::Vector{Float64})
    npairs = length(parameters) ÷ 2
    offsets = view(parameters, 1:npairs)
    couplings = view(parameters, npairs+1:2npairs)
    target_values = scalar_frequency_values(target)
    nfreq = length(target)
    residual = zeros(Float64, 2nfreq)
    jacobian = zeros(Float64, 2nfreq, 2npairs)
    derivatives_offset = Vector{ComplexF64}(undef, npairs)
    derivatives_coupling = Vector{ComplexF64}(undef, npairs)

    @inbounds for i in 1:nfreq
        z = target.axis[i]
        predicted = 0.0 + 0.0im
        for pair in 1:npairs
            offset = offsets[pair]
            coupling = couplings[pair]
            denominator_lower = z + offset
            denominator_upper = z - offset
            predicted += coupling^2 / denominator_lower + coupling^2 / denominator_upper
            derivatives_offset[pair] = coupling^2 * (inv(denominator_upper^2) - inv(denominator_lower^2))
            derivatives_coupling[pair] = 2coupling * (inv(denominator_lower) + inv(denominator_upper))
        end
        error = predicted - target_values[i]
        scale = sqrt(weights[i])
        row_re = 2i - 1
        row_im = 2i
        residual[row_re] = scale * real(error)
        residual[row_im] = scale * imag(error)
        for pair in 1:npairs
            dxi = scale * derivatives_offset[pair]
            dv = scale * derivatives_coupling[pair]
            jacobian[row_re, pair] = real(dxi)
            jacobian[row_im, pair] = imag(dxi)
            jacobian[row_re, npairs+pair] = real(dv)
            jacobian[row_im, npairs+pair] = imag(dv)
        end
    end
    return residual, jacobian
end

function _particle_hole_objective(parameters::Vector{Float64}, target::HybridizationFunction, weights::Vector{Float64})
    npairs = length(parameters) ÷ 2
    offsets = view(parameters, 1:npairs)
    couplings = view(parameters, npairs+1:2npairs)
    target_values = scalar_frequency_values(target)
    objective = 0.0
    @inbounds for i in eachindex(target_values)
        z = target.axis[i]
        predicted = 0.0 + 0.0im
        for pair in 1:npairs
            offset = offsets[pair]
            coupling = couplings[pair]
            predicted += coupling^2 / (z + offset) + coupling^2 / (z - offset)
        end
        objective += weights[i] * abs2(predicted - target_values[i])
    end
    return objective
end

function _project_particle_hole_parameters!(parameters::Vector{Float64}, chemical_potential::Float64,
                                            window::Tuple{Float64,Float64}, hybridization_max::Float64)
    npairs = length(parameters) ÷ 2
    limit = _particle_hole_offset_limit(chemical_potential, window)
    @inbounds for pair in 1:npairs
        parameters[pair] = clamp(abs(parameters[pair]), 0.0, limit)
        parameters[npairs+pair] = clamp(abs(parameters[npairs+pair]), 0.0, hybridization_max)
    end
    return parameters
end

function _particle_hole_bath(parameters::Vector{Float64}, chemical_potential::Float64)
    npairs = length(parameters) ÷ 2
    offsets = view(parameters, 1:npairs)
    couplings = view(parameters, npairs+1:2npairs)
    order = sortperm(offsets)
    energies = Vector{Float64}(undef, 2npairs)
    hybridizations = Vector{ComplexF64}(undef, 2npairs)
    cursor = 1
    for index in reverse(order)
        energies[cursor] = chemical_potential - offsets[index]
        hybridizations[cursor] = ComplexF64(couplings[index])
        cursor += 1
    end
    for index in order
        energies[cursor] = chemical_potential + offsets[index]
        hybridizations[cursor] = ComplexF64(couplings[index])
        cursor += 1
    end
    return DiscreteBath(energies, hybridizations)
end

function _bath_initial_parameters(target::HybridizationFunction, nbath::Int, chemical_potential::Float64,
                                  window::Tuple{Float64,Float64}, start::Int, nstarts::Int)
    nbath == 0 && return Float64[]
    lo, hi = window
    width = hi - lo
    phase = nstarts == 1 ? 0.0 : (start - 1) / nstarts
    energies = Vector{Float64}(undef, nbath)
    for p in 1:nbath
        base = (p - 0.5) / nbath
        shifted = mod(base + 0.35 * phase / max(nbath, 1), 1.0)
        energies[p] = lo + shifted * width
    end
    sort!(energies)

    order = sortperm(collect(1:length(target)); by=i -> abs(target.axis[i]), rev=true)
    ntail = min(length(order), 8)
    tail = order[1:ntail]
    delta = scalar_frequency_values(target)
    sumv2 = sum(max(real(target.axis[i] * delta[i]), 0.0) for i in tail) / ntail
    if !(isfinite(sumv2) && sumv2 > 0)
        sumv2 = max(maximum(abs.(delta)) * max(width, 1.0), 1e-8)
    end
    coupling = sqrt(sumv2 / nbath)
    couplings = fill(coupling, nbath)
    return vcat(energies, couplings)
end

function _bath_initial_parameters(bath::DiscreteBath, nbath::Int, window::Tuple{Float64,Float64}, hybridization_max::Float64)
    length(bath) == nbath || throw(DimensionMismatch("initial bath has $(length(bath)) sites but fit requests $nbath"))
    parameters = vcat(copy(bath.energies), abs.(bath.hybridizations))
    all(isfinite, parameters) || throw(ArgumentError("initial bath contains nonfinite parameters"))
    return _project_bath_parameters!(parameters, window, hybridization_max)
end

function _bath_residual_jacobian(parameters::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64,
                                 weights::Vector{Float64})
    nbath = length(parameters) ÷ 2
    energies = view(parameters, 1:nbath)
    couplings = view(parameters, nbath+1:2nbath)
    target_values = scalar_frequency_values(target)
    nfreq = length(target)
    residual = zeros(Float64, 2nfreq)
    jacobian = zeros(Float64, 2nfreq, 2nbath)
    derivatives_energy = Vector{ComplexF64}(undef, nbath)
    derivatives_coupling = Vector{ComplexF64}(undef, nbath)

    @inbounds for i in 1:nfreq
        z = target.axis[i]
        predicted = 0.0 + 0.0im
        for p in 1:nbath
            denominator = z + chemical_potential - energies[p]
            v = couplings[p]
            predicted += v^2 / denominator
            derivatives_energy[p] = v^2 / denominator^2
            derivatives_coupling[p] = 2v / denominator
        end
        error = predicted - target_values[i]
        scale = sqrt(weights[i])
        row_re = 2i - 1
        row_im = 2i
        residual[row_re] = scale * real(error)
        residual[row_im] = scale * imag(error)
        for p in 1:nbath
            de = scale * derivatives_energy[p]
            dv = scale * derivatives_coupling[p]
            jacobian[row_re, p] = real(de)
            jacobian[row_im, p] = imag(de)
            jacobian[row_re, nbath+p] = real(dv)
            jacobian[row_im, nbath+p] = imag(dv)
        end
    end
    return residual, jacobian
end

function _bath_objective(parameters::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64, weights::Vector{Float64})
    nbath = length(parameters) ÷ 2
    energies = view(parameters, 1:nbath)
    couplings = view(parameters, nbath+1:2nbath)
    target_values = scalar_frequency_values(target)
    objective = 0.0
    @inbounds for i in eachindex(target_values)
        z = target.axis[i]
        predicted = 0.0 + 0.0im
        for p in 1:nbath
            predicted += couplings[p]^2 / (z + chemical_potential - energies[p])
        end
        objective += weights[i] * abs2(predicted - target_values[i])
    end
    return objective
end

function _project_bath_parameters!(parameters::Vector{Float64}, window::Tuple{Float64,Float64}, hybridization_max::Float64)
    nbath = length(parameters) ÷ 2
    lo, hi = window
    @inbounds for p in 1:nbath
        parameters[p] = clamp(parameters[p], lo, hi)
        parameters[nbath+p] = clamp(abs(parameters[nbath+p]), 0.0, hybridization_max)
    end
    return parameters
end

function _fit_residual_jacobian(parameters::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64,
                                weights::Vector{Float64}, symmetry::Symbol)
    symmetry === :none && return _bath_residual_jacobian(parameters, target, chemical_potential, weights)
    symmetry === :particle_hole && return _particle_hole_residual_jacobian(parameters, target, chemical_potential, weights)
    throw(ArgumentError("unsupported bath-fit symmetry $symmetry"))
end

function _fit_objective(parameters::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64,
                        weights::Vector{Float64}, symmetry::Symbol)
    symmetry === :none && return _bath_objective(parameters, target, chemical_potential, weights)
    symmetry === :particle_hole && return _particle_hole_objective(parameters, target, weights)
    throw(ArgumentError("unsupported bath-fit symmetry $symmetry"))
end

function _project_fit_parameters!(parameters::Vector{Float64}, chemical_potential::Float64, window::Tuple{Float64,Float64},
                                  hybridization_max::Float64, symmetry::Symbol)
    symmetry === :none && return _project_bath_parameters!(parameters, window, hybridization_max)
    symmetry === :particle_hole && return _project_particle_hole_parameters!(parameters, chemical_potential, window, hybridization_max)
    throw(ArgumentError("unsupported bath-fit symmetry $symmetry"))
end

function _fit_bath_start(initial::Vector{Float64}, target::HybridizationFunction, chemical_potential::Float64, weights::Vector{Float64},
                         window::Tuple{Float64,Float64}, options::BathFitOptions)
    parameters = copy(initial)
    damping = options.damping
    damping_limit = inv(eps(Float64))
    objective = _fit_objective(parameters, target, chemical_potential, weights, options.symmetry)
    isfinite(objective) || throw(ArgumentError("initial bath-fit objective is not finite"))
    best = copy(parameters)
    best_objective = objective
    converged = objective <= options.tol^2
    iterations = 0
    termination = converged ? :objective_tolerance : :max_iterations

    for iteration in 1:options.maxiter
        iterations = iteration
        residual, jacobian = _fit_residual_jacobian(parameters, target, chemical_potential, weights, options.symmetry)
        if !(all(isfinite, residual) && all(isfinite, jacobian))
            termination = :nonfinite_model
            break
        end

        gradient = transpose(jacobian) * residual
        hessian = transpose(jacobian) * jacobian
        if !(all(isfinite, gradient) && all(isfinite, hessian))
            termination = :nonfinite_normal_equations
            break
        end
        if norm(gradient, Inf) <= options.tol * max(1.0, sqrt(objective))
            converged = true
            termination = :gradient_tolerance
            break
        end

        diagonal_scale = max.(diag(hessian), 1.0)
        if damping >= damping_limit
            termination = :damping_limit
            break
        end
        system = hessian + damping * Diagonal(diagonal_scale)
        if !all(isfinite, system)
            termination = :nonfinite_system
            break
        end

        step = try
            -(system \ gradient)
        catch error
            if error isa LinearAlgebra.SingularException || error isa LinearAlgebra.LAPACKException
                damping = min(10damping, damping_limit)
                continue
            end
            rethrow()
        end
        if !all(isfinite, step)
            damping = min(10damping, damping_limit)
            continue
        end
        if norm(step) <= options.tol * max(norm(parameters), 1.0)
            converged = true
            termination = :step_tolerance
            break
        end

        candidate = parameters + step
        if !all(isfinite, candidate)
            damping = min(10damping, damping_limit)
            continue
        end
        _project_fit_parameters!(candidate, chemical_potential, window, options.hybridization_max, options.symmetry)
        if !all(isfinite, candidate)
            damping = min(10damping, damping_limit)
            continue
        end

        candidate_objective = _fit_objective(candidate, target, chemical_potential, weights, options.symmetry)
        if isfinite(candidate_objective) && candidate_objective < objective
            previous_objective = objective
            parameters = candidate
            objective = candidate_objective
            damping = max(damping / 3, eps(Float64))
            if objective < best_objective
                best = copy(parameters)
                best_objective = objective
            end
            if objective <= options.tol^2
                converged = true
                termination = :objective_tolerance
                break
            end
            if previous_objective - objective <= options.tol * max(previous_objective, 1.0)
                converged = true
                termination = :objective_stagnation
                break
            end
        else
            damping = min(10damping, damping_limit)
        end
    end
    return best, best_objective, converged, iterations, termination
end

"""
    fit_bath(target, nbath; chemical_potential=0, options=BathFitOptions(), initial_bath=nothing)

Fit a scalar Matsubara hybridization to a finite real-coupling ED bath using deterministic damped Gauss-Newton iterations with analytic derivatives. When `initial_bath` is supplied, that bath is evaluated first as a warm start. A deterministic safeguard start is then evaluated when available; the remaining configured multistarts are evaluated only if the safeguard improves the warm-start objective or the warm start does not converge. With `options.symmetry=:particle_hole`, the optimizer uses a reduced parameterization with bath energies paired about `chemical_potential` and equal hybridization magnitudes within each pair.
"""
function fit_bath(target::HybridizationFunction, nbath::Integer; chemical_potential::Real=0.0, options::BathFitOptions=BathFitOptions(),
                  initial_bath=nothing)
    matrix_dimension(target) == 1 || throw(DimensionMismatch("finite-bath fitting currently supports a scalar hybridization"))
    nbath >= 0 || throw(ArgumentError("nbath must be nonnegative"))
    options.symmetry === :particle_hole && isodd(nbath) &&
        throw(ArgumentError("particle-hole-symmetric bath fitting requires an even number of bath sites"))
    mu = Float64(chemical_potential)
    isfinite(mu) || throw(ArgumentError("chemical_potential must be finite"))
    target_values = scalar_frequency_values(target)
    all(isfinite, target_values) || throw(ArgumentError("target hybridization contains nonfinite values"))
    !isnothing(initial_bath) && !(initial_bath isa DiscreteBath) && throw(ArgumentError("initial_bath must be a DiscreteBath or nothing"))
    weights = _bath_fit_weights(target, options)

    if nbath == 0
        !isnothing(initial_bath) && length(initial_bath) != 0 && throw(DimensionMismatch("initial bath must contain zero sites"))
        bath = DiscreteBath(Float64[], ComplexF64[])
        objective = sum(weights .* abs2.(scalar_frequency_values(target)))
        metadata = Dict{Symbol,Any}(:bath_sites => 0, :starts => 1, :starts_executed => 1, :selected_start => 1, :iterations => 0,
                                    :weight_power => options.weight_power, :symmetry => options.symmetry, :parameter_count => 0,
                                    :start_objectives => [objective], :start_converged => [objective <= options.tol^2],
                                    :start_iterations => [0], :start_termination => [:objective_tolerance], :start_kinds => [:zero_bath],
                                    :warm_start_used => !isnothing(initial_bath), :objective_spread => 0.0)
        return BathFitResult(bath, objective, objective <= options.tol^2, 0, metadata)
    end

    n = Int(nbath)
    window = _bath_fit_window(target, mu, options)
    options.symmetry === :particle_hole && _particle_hole_offset_limit(mu, window)
    best_parameters = Float64[]
    best_objective = Inf
    best_converged = false
    best_iterations = 0
    best_start = 0
    start_objectives = Float64[]
    start_converged = Bool[]
    start_iterations = Int[]
    start_termination = Symbol[]
    start_kinds = Symbol[]

    function evaluate_start!(initial::Vector{Float64}, kind::Symbol)
        parameters, objective, converged, iterations, termination = _fit_bath_start(initial, target, mu, weights, window, options)
        push!(start_objectives, objective)
        push!(start_converged, converged)
        push!(start_iterations, iterations)
        push!(start_termination, termination)
        push!(start_kinds, kind)
        if objective < best_objective
            best_parameters = parameters
            best_objective = objective
            best_converged = converged
            best_iterations = iterations
            best_start = length(start_objectives)
        end
        return objective, converged
    end

    if isnothing(initial_bath)
        for start_index in 1:options.multistart
            initial = options.symmetry === :particle_hole ?
                      _particle_hole_initial_parameters(target, n, mu, window, start_index, options.multistart) :
                      _bath_initial_parameters(target, n, mu, window, start_index, options.multistart)
            evaluate_start!(initial, :deterministic_multistart)
        end
    else
        warm_initial = options.symmetry === :particle_hole ?
                       _particle_hole_initial_parameters(initial_bath, n, mu, window, options.hybridization_max) :
                       _bath_initial_parameters(initial_bath, n, window, options.hybridization_max)
        warm_objective, warm_converged = evaluate_start!(warm_initial, :warm_start)
        deterministic_count = options.multistart
        if deterministic_count > 0
            safeguard_initial = options.symmetry === :particle_hole ?
                                _particle_hole_initial_parameters(target, n, mu, window, 1, deterministic_count) :
                                _bath_initial_parameters(target, n, mu, window, 1, deterministic_count)
            safeguard_objective, safeguard_converged = evaluate_start!(safeguard_initial, :warm_start_safeguard)
            significant_improvement = safeguard_objective < warm_objective - max(options.tol * max(warm_objective, 1.0), 32eps(Float64))
            if !warm_converged || !safeguard_converged || significant_improvement
                for start_index in 2:deterministic_count
                    initial = options.symmetry === :particle_hole ?
                              _particle_hole_initial_parameters(target, n, mu, window, start_index, deterministic_count) :
                              _bath_initial_parameters(target, n, mu, window, start_index, deterministic_count)
                    evaluate_start!(initial, :deterministic_fallback)
                end
            end
        end
    end

    isempty(best_parameters) && throw(ErrorException("bath fitting did not produce a finite candidate"))
    bath = options.symmetry === :particle_hole ?
           _particle_hole_bath(best_parameters, mu) :
           DiscreteBath(best_parameters[1:n], ComplexF64.(best_parameters[n+1:2n]))
    metadata = Dict{Symbol,Any}(
        :bath_sites => n,
        :starts => options.multistart,
        :starts_executed => length(start_objectives),
        :selected_start => best_start,
        :iterations => best_iterations,
        :weight_power => options.weight_power,
        :energy_window => window,
        :symmetry => options.symmetry,
        :parameter_count => length(best_parameters),
        :particle_hole_center => options.symmetry === :particle_hole ? mu : nothing,
        :start_objectives => start_objectives,
        :start_converged => start_converged,
        :start_iterations => start_iterations,
        :start_termination => start_termination,
        :start_kinds => start_kinds,
        :warm_start_used => !isnothing(initial_bath),
        :objective_spread => maximum(start_objectives) - minimum(start_objectives),
    )
    return BathFitResult(bath, best_objective, best_converged, best_iterations, metadata)
end
