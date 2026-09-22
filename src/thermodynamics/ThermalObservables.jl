# Thermal expectation values from exact or sampled ensembles.

function _thermal_resolve_operator(model::ManyBodyModel, op)
    if op isa Symbol
        haskey(model.observables, op) || throw(KeyError("observable $op is not registered in this model"))
        return model.observables[op]
    elseif op isa ObservableRef
        return resolve_observable(model, op)
    end
    return op
end

function _complex_jackknife_stderr(values::Vector{ComplexF64})
    n = length(values)
    n <= 1 && return NaN
    center = mean(values)
    return sqrt((n - 1) / n * sum(abs2, values .- center))
end

function _real_jackknife_stderr(values::Vector{Float64})
    n = length(values)
    n <= 1 && return NaN
    center = mean(values)
    return sqrt((n - 1) / n * sum(abs2, values .- center))
end

"""
    expectation(state, observable; hbar=1)

Canonical expectation value. Exact Gibbs states use left/right eigenvectors consistently; sampled states use retained thermally filtered typical vectors. For sampled states, the reported uncertainty is a delete-one jackknife error for the ratio estimator and is `NaN` when only one realization is available.
"""
function expectation(state::ExactGibbsState, observable; hbar::Real=get(state.model.parameters, :hbar, 1.0))
    op = _thermal_resolve_operator(state.model, observable)
    O = matrix(op, state.solution.basis; sparse=false, hbar=hbar)
    Oe = adjoint(state.solution.left_states) * O * state.solution.right_states
    value = sum(state.probabilities .* diag(Oe))
    return ThermalEstimate(value, 0.0)
end

function expectation(state::SampledThermalState, observable; hbar::Real=get(state.model.parameters, :hbar, 1.0))
    V = state.filtered_vectors
    isnothing(V) && throw(ArgumentError("arbitrary expectation values require retained thermal vectors; use ThermalTypicality(store_vectors=true) or CanonicalTPQ(store_vectors=true)"))
    isnothing(state.basis) && throw(ArgumentError("retained thermal vectors require a materialized computational basis"))
    op = _thermal_resolve_operator(state.model, observable)
    O = matrix(op, state.basis; sparse=true, hbar=hbar)

    R = size(V, 2)
    nums = zeros(ComplexF64, R)
    dens = zeros(Float64, R)
    tmp = zeros(ComplexF64, length(state.basis))

    for r in 1:R
        ψ = view(V, :, r)
        mul!(tmp, O, ψ)
        scale = state.sample_scales[r]
        nums[r] = scale * dot(ψ, tmp)
        dens[r] = scale * state.sample_norms[r]
    end

    nsum = sum(nums)
    dsum = sum(dens)
    dsum > 0 || throw(ArgumentError("sampled thermal normalization vanished"))
    value = nsum / dsum

    stderr = if R <= 1
        NaN
    else
        loo = Vector{ComplexF64}(undef, R)
        @inbounds for r in 1:R
            denom = dsum - dens[r]
            denom > 0 || throw(ArgumentError("leave-one-out sampled thermal normalization vanished"))
            loo[r] = (nsum - nums[r]) / denom
        end
        _complex_jackknife_stderr(loo)
    end
    return ThermalEstimate(value, Float64(stderr))
end

"""
    thermal_variance(state, observable)

Return `<O†O> - |<O>|²`, using the representation-correct operator matrix. For sampled states, numerator, second moment, and normalization are jackknifed jointly so their covariance is retained.
"""
function thermal_variance(state::ExactGibbsState, observable; hbar::Real=get(state.model.parameters, :hbar, 1.0))
    op = _thermal_resolve_operator(state.model, observable)
    meanO = expectation(state, op; hbar=hbar)
    meanO2 = expectation(state, adjoint(op) * op; hbar=hbar)
    value = real(meanO2.value - abs2(meanO.value))
    return ThermalEstimate(max(value, 0.0), 0.0)
end

function thermal_variance(state::SampledThermalState, observable; hbar::Real=get(state.model.parameters, :hbar, 1.0))
    V = state.filtered_vectors
    isnothing(V) && throw(ArgumentError("thermal variance requires retained thermal vectors; use ThermalTypicality(store_vectors=true) or CanonicalTPQ(store_vectors=true)"))
    isnothing(state.basis) && throw(ArgumentError("retained thermal vectors require a materialized computational basis"))
    op = _thermal_resolve_operator(state.model, observable)
    O = matrix(op, state.basis; sparse=true, hbar=hbar)

    R = size(V, 2)
    nums = zeros(ComplexF64, R)
    second = zeros(Float64, R)
    dens = zeros(Float64, R)
    tmp = zeros(ComplexF64, length(state.basis))

    for r in 1:R
        ψ = view(V, :, r)
        mul!(tmp, O, ψ)
        scale = state.sample_scales[r]
        nums[r] = scale * dot(ψ, tmp)
        second[r] = scale * real(dot(tmp, tmp))
        dens[r] = scale * state.sample_norms[r]
    end

    nsum = sum(nums)
    ssum = sum(second)
    dsum = sum(dens)
    dsum > 0 || throw(ArgumentError("sampled thermal normalization vanished"))
    meanO = nsum / dsum
    meanO2 = ssum / dsum
    value = max(real(meanO2 - abs2(meanO)), 0.0)

    stderr = if R <= 1
        NaN
    else
        loo = Vector{Float64}(undef, R)
        @inbounds for r in 1:R
            denom = dsum - dens[r]
            denom > 0 || throw(ArgumentError("leave-one-out sampled thermal normalization vanished"))
            μ = (nsum - nums[r]) / denom
            μ2 = (ssum - second[r]) / denom
            loo[r] = max(real(μ2 - abs2(μ)), 0.0)
        end
        _real_jackknife_stderr(loo)
    end
    return ThermalEstimate(value, Float64(stderr))
end
