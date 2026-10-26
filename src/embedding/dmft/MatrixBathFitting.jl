# Phase II.A.1: deterministic complex matrix-hybridization fitting with diagonal bath energies.

"""Matrix bath-fit result; the original scalar `BathFitResult` remains unchanged."""
struct MatrixBathFitResult
    bath::MatrixDiscreteBath
    objective::Float64
    converged::Bool
    iterations::Int
    metadata::Dict{Symbol,Any}
end

function _matrix_bath_weights(target::HybridizationFunction, options::BathFitOptions)
    omega = abs.(imag.(target.axis.values))
    floor_frequency = minimum(omega)
    weights = (omega ./ floor_frequency) .^ (-options.weight_power)
    weights ./= sum(weights)
    return weights
end

function _matrix_bath_window(target::HybridizationFunction, mu::Float64, options::BathFitOptions)
    !isnothing(options.energy_window) && return options.energy_window
    nfreq = min(length(target), 8)
    scale = maximum(abs, view(target.values, :, :, 1:nfreq))
    span = max(1.0, 4 * scale)
    return (mu - span, mu + span)
end

function _matrix_bath_encode(bath::MatrixDiscreteBath)
    V = bath.hybridizations
    return vcat(copy(bath.energies), vec(real.(V)), vec(imag.(V)))
end

function _matrix_bath_decode(parameters::Vector{Float64}, n::Int, nbath::Int, names::Vector{Symbol})
    offset = nbath
    real_part = reshape(view(parameters, offset+1:offset+n*nbath), n, nbath)
    imaginary_part = reshape(view(parameters, offset+n*nbath+1:offset+2 * n * nbath), n, nbath)
    return MatrixDiscreteBath(parameters[1:nbath], complex.(real_part, imaginary_part); labels=names)
end

function _matrix_bath_project!(parameters::Vector{Float64}, n::Int, nbath::Int, window, maximum_coupling::Float64)
    lo, hi = window
    @inbounds for p in 1:nbath
        parameters[p] = clamp(parameters[p], lo, hi)
        norm2 = 0.0
        for a in 1:n
            re = parameters[nbath + (p-1)*n + a]
            im = parameters[nbath + n*nbath + (p-1)*n + a]
            norm2 += re^2 + im^2
        end
        if norm2 > maximum_coupling^2
            factor = maximum_coupling / sqrt(norm2)
            for a in 1:n
                parameters[nbath + (p-1)*n + a] *= factor
                parameters[nbath + n*nbath + (p-1)*n + a] *= factor
            end
        end
    end
    return parameters
end

function _matrix_bath_initial(target::HybridizationFunction, nbath::Int, mu::Float64, window, start::Int, nstarts::Int)
    n = matrix_dimension(target)
    lo, hi = window
    phase = nstarts == 1 ? 0.0 : (start - 1) / nstarts
    energies = [lo + (hi - lo) * mod((p - 0.5) / nbath + 0.35 * phase / nbath, 1.0) for p in 1:nbath]
    # The high-frequency moment is positive semidefinite for any causal finite bath.
    order = sortperm(collect(1:length(target)); by=i -> abs(target.axis[i]), rev=true)
    ntail = min(length(order), 8)
    moment = zeros(ComplexF64, n, n)
    for index in view(order, 1:ntail)
        moment .+= (target.axis[index] + mu) .* view(target.values, :, :, index)
    end
    moment ./= ntail
    eig = eigen(Hermitian(0.5 .* (moment + moment')))
    strengths = max.(real.(eig.values), 0.0)
    if maximum(strengths) <= 1e-14
        strengths .= max(maximum(abs, target.values), 1e-8)
    end
    V = zeros(ComplexF64, n, nbath)
    eigen_order = sortperm(strengths; rev=true)
    for p in 1:nbath
        component = eigen_order[mod(p + start - 2, n) + 1]
        phase_factor = cis(2π * phase * (p - 1) / max(nbath, 1))
        V[:, p] .= sqrt(strengths[component] * n / nbath) * phase_factor .* eig.vectors[:, component]
    end
    return _matrix_bath_encode(MatrixDiscreteBath(energies, V; labels=target.labels))
end

function _matrix_bath_objective(parameters::Vector{Float64}, target::HybridizationFunction, mu::Float64, weights::Vector{Float64})
    n = matrix_dimension(target)
    nbath = length(parameters) ÷ (2 * n + 1)
    offset = nbath
    objective = 0.0
    @inbounds for i in 1:length(target), b in 1:n, a in 1:n
        predicted = 0.0 + 0.0im
        for p in 1:nbath
            va = complex(parameters[offset + (p-1)*n + a], parameters[offset + n*nbath + (p-1)*n + a])
            vb = complex(parameters[offset + (p-1)*n + b], parameters[offset + n*nbath + (p-1)*n + b])
            predicted += va * conj(vb) / (target.axis[i] + mu - parameters[p])
        end
        objective += weights[i] * abs2(predicted - target[a, b, i])
    end
    return objective
end

function _matrix_bath_residual_jacobian(parameters::Vector{Float64}, target::HybridizationFunction, mu::Float64, weights::Vector{Float64})
    n = matrix_dimension(target)
    nbath = length(parameters) ÷ (2 * n + 1)
    offset = nbath
    residual = Vector{Float64}(undef, 2 * n * n * length(target))
    jacobian = zeros(Float64, length(residual), length(parameters))
    @inbounds for i in 1:length(target)
        scale = sqrt(weights[i])
        for b in 1:n, a in 1:n
            row = 2 * ((i - 1)*n*n + (b - 1)*n + a) - 1
            predicted = 0.0 + 0.0im
            for p in 1:nbath
                va = complex(parameters[offset + (p-1)*n + a], parameters[offset + n*nbath + (p-1)*n + a])
                vb = complex(parameters[offset + (p-1)*n + b], parameters[offset + n*nbath + (p-1)*n + b])
                factor = inv(target.axis[i] + mu - parameters[p])
                predicted += va * conj(vb) * factor
                derivative_energy = scale * va * conj(vb) * factor^2
                jacobian[row, p] = real(derivative_energy)
                jacobian[row+1, p] = imag(derivative_energy)
                for c in (a, b)
                    real_index = offset + (p-1)*n + c
                    imag_index = offset + n*nbath + (p-1)*n + c
                    real_derivative = scale * factor * ((a == c ? conj(vb) : 0.0im) + (b == c ? va : 0.0im))
                    imag_derivative = scale * factor * ((a == c ? im*conj(vb) : 0.0im) - (b == c ? im*va : 0.0im))
                    # At a == b, both derivative contributions must be added once, not twice.
                    if a == b && c == b
                        jacobian[row, real_index] = real(real_derivative)
                        jacobian[row+1, real_index] = imag(real_derivative)
                        jacobian[row, imag_index] = real(imag_derivative)
                        jacobian[row+1, imag_index] = imag(imag_derivative)
                        break
                    end
                    jacobian[row, real_index] = real(real_derivative)
                    jacobian[row+1, real_index] = imag(real_derivative)
                    jacobian[row, imag_index] = real(imag_derivative)
                    jacobian[row+1, imag_index] = imag(imag_derivative)
                end
            end
            error = scale * (predicted - target[a, b, i])
            residual[row] = real(error)
            residual[row+1] = imag(error)
        end
    end
    return residual, jacobian
end

function _matrix_bath_start(initial::Vector{Float64}, target::HybridizationFunction, mu::Float64,
                            weights::Vector{Float64}, window, options::BathFitOptions)
    n = matrix_dimension(target)
    nbath = length(initial) ÷ (2 * n + 1)
    parameters = _matrix_bath_project!(copy(initial), n, nbath, window, options.hybridization_max)
    best_objective = _matrix_bath_objective(parameters, target, mu, weights)
    damping = options.damping
    iterations = 0
    converged = best_objective <= options.tol^2
    for iteration in 1:options.maxiter
        converged && break
        iterations = iteration
        residual, J = _matrix_bath_residual_jacobian(parameters, target, mu, weights)
        gradient = J' * residual
        maximum(abs, gradient) <= options.tol * max(1.0, sqrt(best_objective)) && (converged = true; break)
        normal = J' * J
        accepted = false
        for _ in 1:12
            trial_step = -(normal + damping * I) \ gradient
            all(isfinite, trial_step) || break
            trial = _matrix_bath_project!(parameters + trial_step, n, nbath, window, options.hybridization_max)
            trial_objective = _matrix_bath_objective(trial, target, mu, weights)
            if trial_objective < best_objective
                improvement = best_objective - trial_objective
                parameters = trial
                best_objective = trial_objective
                damping = max(damping * 0.35, 1e-14)
                accepted = true
                converged = best_objective <= options.tol^2 || improvement <= options.tol * max(best_objective, 1e-12)
                break
            end
            damping = min(damping * 8.0, 1e14)
        end
        accepted || break
    end
    return parameters, best_objective, converged, iterations
end

"""
    fit_matrix_bath(target, nbath; chemical_potential=0, options=BathFitOptions(), initial_bath=nothing)

Deterministic finite-bath fit for complex matrix-valued Matsubara hybridizations.
Minimizes frequency-weighted Frobenius residuals using analytic real/imaginary
Jacobian and damped Gauss–Newton steps. Bath levels are diagonal and real, and
coupling columns are complex; causality is guaranteed by the bath parameterization.
The nonconvex fit is local, and the result reports its attained residual rather
than guaranteeing recovery of a global minimum. `symmetry=:particle_hole` is not
currently certified for matrix-valued baths and is rejected explicitly.
"""
function fit_matrix_bath(target::HybridizationFunction, nbath::Integer; chemical_potential::Real=0.0,
                         options::BathFitOptions=BathFitOptions(), initial_bath=nothing)
    nbath >= 0 || throw(ArgumentError("nbath must be nonnegative"))
    nbath <= 64 || throw(ArgumentError("matrix bath fitting currently supports at most 64 bath sites"))
    options.symmetry === :none || throw(ArgumentError("matrix bath fitting does not yet certify particle-hole constrained baths"))
    mu = Float64(chemical_potential)
    isfinite(mu) || throw(ArgumentError("chemical_potential must be finite"))
    all(_isfinite_complex, target.values) || throw(ArgumentError("target hybridization must be finite"))
    n = matrix_dimension(target)
    weights = _matrix_bath_weights(target, options)
    if !isnothing(initial_bath)
        initial_bath isa MatrixDiscreteBath || throw(ArgumentError("initial_bath must be a MatrixDiscreteBath"))
        length(initial_bath) == nbath || throw(DimensionMismatch("warm-start bath size differs from nbath"))
        size(initial_bath.hybridizations, 1) == n || throw(DimensionMismatch("warm-start impurity dimension differs from target"))
        initial_bath.labels == target.labels || throw(ArgumentError("warm-start component labels differ from target"))
    end
    if nbath == 0
        bath = MatrixDiscreteBath(Float64[], zeros(ComplexF64, n, 0); labels=target.labels)
        objective = sum(weights[i] * sum(abs2, view(target.values, :, :, i)) for i in 1:length(target))
        return MatrixBathFitResult(bath, objective, objective <= options.tol^2, 0,
                                   Dict{Symbol,Any}(:bath_sites => 0, :start_objectives => [objective], :selected_start => 1, :warm_start_used => !isnothing(initial_bath)))
    end
    window = _matrix_bath_window(target, mu, options)
    best_parameters = Float64[]
    best_objective = Inf
    best_converged = false
    best_iterations = 0
    selected_start = 0
    objectives = Float64[]
    kinds = Symbol[]
    starts = isnothing(initial_bath) ? options.multistart : options.multistart + 1
    for start in 1:starts
        warm = !isnothing(initial_bath) && start == 1
        initial = warm ? _matrix_bath_encode(initial_bath) :
                  _matrix_bath_initial(target, Int(nbath), mu, window, warm ? 1 : start - Int(!isnothing(initial_bath)), options.multistart)
        parameters, objective, converged, iterations = _matrix_bath_start(initial, target, mu, weights, window, options)
        push!(objectives, objective)
        push!(kinds, warm ? :warm_start : :deterministic_multistart)
        if objective < best_objective
            best_parameters = parameters
            best_objective = objective
            best_converged = converged
            best_iterations = iterations
            selected_start = start
        end
    end
    permutation = sortperm(view(best_parameters, 1:Int(nbath)))
    bath = _matrix_bath_decode(best_parameters, n, Int(nbath), target.labels)
    bath = MatrixDiscreteBath(bath.energies[permutation], bath.hybridizations[:, permutation]; labels=target.labels)
    return MatrixBathFitResult(bath, best_objective, best_converged, best_iterations,
                               Dict{Symbol,Any}(:bath_sites => Int(nbath), :parameter_count => length(best_parameters),
                                                :energy_window => window, :weight_power => options.weight_power,
                                                :start_objectives => objectives, :start_kinds => kinds,
                                                :selected_start => selected_start, :starts_executed => length(objectives),
                                                :warm_start_used => !isnothing(initial_bath), :iterations => best_iterations,
                                                :symmetry => :none, :objective_tolerance_reached => best_objective <= options.tol^2,
                                                :objective_spread => maximum(objectives) - minimum(objectives)))
end
