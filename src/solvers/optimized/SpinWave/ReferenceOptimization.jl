# Local product-state reference minimization shared by dipolar and site-SU(N) spin waves.
# This routine reports a local stationary point, not a certified global ground state.

function _spinwave_reference_objective(data::_SpinWaveInteractions, ref::MagneticReferenceState)
    N = length(data.spins)
    n = ref.directions
    size(n) == (3, N) || throw(DimensionMismatch("dipolar reference size differs from model"))
    gradient = zeros(Float64, 3, N)
    energy = data.constant
    for i in 1:N
        S = data.spins[i]
        ni = view(n, :, i)
        K = data.anisotropy[i]
        h = view(data.fields, :, i)
        energy += -S * dot(h, ni) + S * (S - 0.5) * dot(ni, K * ni) + S / 2 * tr(K)
        gradient[:, i] .+= S .* (-h + (2 * S - 1) .* (K * ni))
    end
    for ((i, j), J) in data.exchange
        Si, Sj = data.spins[i], data.spins[j]
        ni, nj = view(n, :, i), view(n, :, j)
        energy += Si * Sj * dot(ni, J * nj)
        gradient[:, i] .+= Si * Sj .* (J * nj)
        gradient[:, j] .+= Si * Sj .* (transpose(J) * ni)
    end
    gradient .-= n .* reshape(vec(sum(n .* gradient; dims=1)), 1, N)
    return energy, gradient
end

function _spinwave_reference_objective(data::_SpinWaveInteractions, ref::SUNReferenceState)
    N = length(data.spins)
    length(ref.amplitudes) == N || throw(DimensionMismatch("SU(N) reference size differs from model"))
    ops = [_sun_spin_matrices(S) for S in data.spins]
    states = ref.amplitudes
    moments = zeros(Float64, 3, N)
    fields = [zeros(ComplexF64, length(z), length(z)) for z in states]
    energy = data.constant
    for i in 1:N
        z = states[i]
        length(z) == round(Int, 2 * data.spins[i]) + 1 || throw(DimensionMismatch("SU(N) spin dimension mismatch at site $i"))
        for α in 1:3
            moments[α, i] = real(dot(z, ops[i][α] * z))
            fields[i] .-= data.fields[α, i] .* ops[i][α]
        end
        for α in 1:3, β in 1:3
            fields[i] .+= data.anisotropy[i][α, β] .* (ops[i][α] * ops[i][β])
        end
        fields[i] .= (fields[i] + adjoint(fields[i])) / 2
        energy += real(dot(z, fields[i] * z))
    end
    for ((i, j), J) in data.exchange
        energy += dot(view(moments, :, i), J * view(moments, :, j))
        for α in 1:3
            fields[i] .+= (J * view(moments, :, j))[α] .* ops[i][α]
            fields[j] .+= (transpose(J) * view(moments, :, i))[α] .* ops[j][α]
        end
    end
    gradient = Vector{ComplexF64}[]
    for i in 1:N
        z = states[i]
        g = fields[i] * z
        g .-= dot(z, g) .* z
        push!(gradient, g)
    end
    return energy, gradient
end

_spinwave_gradient_norm(gradient::AbstractMatrix) = norm(gradient)
_spinwave_gradient_norm(gradient::AbstractVector{<:AbstractVector}) = sqrt(sum(abs2, (g for v in gradient for g in v)))

function _spinwave_reference_step(ref::MagneticReferenceState, gradient::AbstractMatrix, step::Real)
    n = ref.directions .- step .* gradient
    for i in axes(n, 2)
        n[:, i] ./= norm(view(n, :, i))
    end
    return MagneticReferenceState(n)
end

function _spinwave_reference_step(ref::SUNReferenceState, gradient::AbstractVector, step::Real)
    states = Vector{ComplexF64}[]
    for (z, g) in zip(ref.amplitudes, gradient)
        trial = z .- step .* g
        trial ./= norm(trial)
        push!(states, trial)
    end
    return SUNReferenceState(states)
end

function _spinwave_random_reference(spins, formalism::Symbol, rng)
    if formalism === :dipole
        n = randn(rng, 3, length(spins))
        for i in axes(n, 2)
            n[:, i] ./= norm(view(n, :, i))
        end
        return MagneticReferenceState(n)
    end
    return SUNReferenceState([begin
        z = randn(rng, ComplexF64, round(Int, 2 * S) + 1)
        z ./= norm(z)
    end for S in spins])
end

"""Minimize the physical product-reference energy on (S²)^N or a product of CP^(2 * S) manifolds.

`formalism=:dipole` uses coherent spin directions; `:SUN` uses arbitrary normalized local spin states. The returned named tuple contains `reference`, `energy`, `residual`, `converged`, `iterations`, `restart`, and per-start diagnostics. A converged result is a *local* stationary solution, not proof of the global ground state or harmonic stability. `initial` may supply a reference of the selected formalism; remaining multistarts are seeded random references.
"""
function optimize_spin_reference(model::ManyBodyModel; formalism::Symbol=:dipole, initial=nothing, starts::Integer=4,
                                 maxiter::Integer=2000, tol::Real=1e-8, step::Real=1.0, seed::Integer=0)
    formalism in (:dipole, :SUN) || throw(ArgumentError("formalism must be :dipole or :SUN"))
    starts > 0 || throw(ArgumentError("starts must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    tol > 0 && step > 0 || throw(ArgumentError("tol and step must be positive"))
    data = _spinwave_extract_interactions(model; tol=1e-10)
    if !isnothing(initial)
        formalism === :dipole && !(initial isa MagneticReferenceState) && throw(ArgumentError(":dipole initial reference must be MagneticReferenceState"))
        formalism === :SUN && !(initial isa Union{SUNReferenceState,MagneticReferenceState}) && throw(ArgumentError(":SUN initial reference must be SUNReferenceState or MagneticReferenceState"))
    end
    rng = Random.MersenneTwister(seed)
    diagnostics = NamedTuple[]
    best = nothing
    for restart in 1:starts
        ref = if restart == 1 && !isnothing(initial)
            formalism === :SUN ? _sun_resolved_reference(initial, data.spins) : initial
        elseif restart == 1 && formalism === :dipole
            MagneticReferenceState(repeat(reshape([0.0, 0.0, 1.0], 3, 1), 1, length(data.spins)))
        elseif restart == 1 && formalism === :SUN
            SUNReferenceState([vcat(1.0 + 0im, zeros(ComplexF64, round(Int, 2 * S))) for S in data.spins])
        else
            _spinwave_random_reference(data.spins, formalism, rng)
        end
        energy, gradient = _spinwave_reference_objective(data, ref)
        residual = _spinwave_gradient_norm(gradient)
        converged = residual <= tol
        iterations = 0
        while !converged && iterations < maxiter
            rate = Float64(step)
            candidate = nothing
            next_energy = energy
            next_gradient = gradient
            for _ in 1:35
                trial = _spinwave_reference_step(ref, gradient, rate)
                trial_energy, trial_gradient = _spinwave_reference_objective(data, trial)
                if trial_energy <= energy - 1e-4 * rate * residual^2
                    candidate = trial
                    next_energy, next_gradient = trial_energy, trial_gradient
                    break
                end
                rate /= 2
            end
            isnothing(candidate) && break
            ref = candidate
            energy, gradient = next_energy, next_gradient
            residual = _spinwave_gradient_norm(gradient)
            iterations += 1
            converged = residual <= tol
        end
        push!(diagnostics, (; restart, energy, residual, converged, iterations))
        if isnothing(best) || (converged && !best.converged) || (converged == best.converged && energy < best.energy)
            best = (; reference=ref, energy, residual, converged, iterations, restart)
        end
    end
    return merge(best, (; formalism, starts, diagnostics, global_minimum_certified=false, harmonic_stability=:not_checked))
end
