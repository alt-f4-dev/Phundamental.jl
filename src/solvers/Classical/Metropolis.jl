# Metropolis.jl
#
# Detailed-balance-safe random-site Ising Metropolis kernel.

@inline function _accept_log_metropolis(rng, beta::Real, deltaE::Real)
    if deltaE <= 0
        return true
    end
    # log(rand) avoids overflow/underflow in exp(-beta*deltaE).
    return log(rand(rng)) < -beta * deltaE
end

function _cache_consistency!(
    problem::IsingProblem,
    state,
    cached_energy::Real,
    local_product,
    tolerance::Real,
)
    exact = ising_energy(problem, state)
    escale = max(abs(exact), abs(cached_energy), 1.0)
    eerr = abs(exact - cached_energy) / escale

    exact_local = problem.J * state
    lerr = norm(exact_local - local_product) / max(norm(exact_local), 1.0)

    max(eerr, lerr) <= tolerance || throw(ErrorException(
        "classical Metropolis cache drift detected: energy error=$eerr, local-field error=$lerr"
    ))
    return exact
end

function _metropolis_attempt!(
    problem::IsingProblem,
    state,
    local_product,
    energy::Float64,
    beta::Float64,
    rng,
)
    N = length(problem)
    i = rand(rng, 1:N)  # exactly uniform symmetric proposal
    deltaE = Float64(ising_flip_delta(problem, state, local_product, i))
    if _accept_log_metropolis(rng, beta, deltaE)
        old = state[i]
        delta_spin = -2 * old
        state[i] = -old
        _update_local_product!(local_product, problem.J, i, delta_spin)
        return energy + deltaE, true
    end
    return energy, false
end

function _metropolis_sweep!(
    problem::IsingProblem,
    state,
    local_product,
    energy::Float64,
    beta::Float64,
    rng,
)
    accepted = 0
    for _ in 1:length(problem)
        energy, ok = _metropolis_attempt!(
            problem, state, local_product, energy, beta, rng,
        )
        accepted += ok
    end
    return energy, accepted
end
