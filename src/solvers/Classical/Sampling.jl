# Sampling.jl

@inline function _checked_metropolis_sweep!(
    problem::IsingProblem,
    state,
    local_product,
    energy::Float64,
    beta::Float64,
    rng,
    sweeps_done::Int,
    sampler::MetropolisSampler,
)
    energy, accepted = _metropolis_sweep!(
        problem, state, local_product, energy, beta, rng,
    )
    sweeps_done += 1
    if sampler.energy_check_interval > 0 &&
       sweeps_done % sampler.energy_check_interval == 0
        energy = _cache_consistency!(
            problem, state, energy, local_product, sampler.cache_tolerance,
        )
    end
    return energy, accepted, sweeps_done
end

"""
    sample(sampler, problem; temperature, kB=1)

Generate a canonical Ising configuration ensemble.

Stored configurations are separated by `sweeps_per_sample` full random-site
sweeps after `burnin_sweeps`. The sampler periodically recomputes both the
energy and `J*s` cache from scratch and aborts on inconsistency.
"""
function sample(
    sampler::MetropolisSampler,
    problem::IsingProblem;
    temperature::Real,
    kB::Real=1.0,
)
    temperature > 0 ||
        throw(ArgumentError("Metropolis canonical sampling requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    beta = inv(Float64(kB) * Float64(temperature))

    rng = MersenneTwister(sampler.seed)
    state = _initial_ising_state(problem, sampler.initial, rng)
    local_product = Vector{Float64}(problem.J * state)
    energy = Float64(ising_energy(problem, state))

    total_attempts = 0
    accepted_attempts = 0
    sweeps_done = 0
    N = length(problem)

    for _ in 1:sampler.burnin_sweeps
        energy, accepted, sweeps_done = _checked_metropolis_sweep!(
            problem, state, local_product, energy, beta, rng,
            sweeps_done, sampler,
        )
        accepted_attempts += accepted
        total_attempts += N
    end

    configs = Matrix{Int8}(undef, N, sampler.samples)
    energies = Vector{Float64}(undef, sampler.samples)

    for sample_index in 1:sampler.samples
        for _ in 1:sampler.sweeps_per_sample
            energy, accepted, sweeps_done = _checked_metropolis_sweep!(
                problem, state, local_product, energy, beta, rng,
                sweeps_done, sampler,
            )
            accepted_attempts += accepted
            total_attempts += N
        end
        @views configs[:, sample_index] .= state
        energies[sample_index] = energy
    end

    # Mandatory final consistency check independent of interval settings.
    energy = _cache_consistency!(
        problem, state, energy, local_product, sampler.cache_tolerance,
    )
    energies[end] = energy

    tauE = autocorrelation_time(energies)
    neffE = min(length(energies), length(energies) / (2tauE))

    metadata = Dict{Symbol,Any}(
        :sampler => :metropolis,
        :problem => :ising,
        :beta => beta,
        :samples => sampler.samples,
        :burnin_sweeps => sampler.burnin_sweeps,
        :sweeps_per_sample => sampler.sweeps_per_sample,
        :seed => sampler.seed,
        :acceptance_rate => total_attempts == 0 ? NaN : accepted_attempts / total_attempts,
        :energy_tau_int => tauE,
        :energy_effective_samples => neffE,
        :energy_stderr => correlated_stderr(energies),
        :proposal => :uniform_random_single_site_flip,
        :proposal_symmetric => true,
        :acceptance_rule => :log_metropolis,
        :detailed_balance => :exact_for_symmetric_proposal,
        :cache_checked => true,
        :parallel_tempering => :future,
    )

    return ConfigurationEnsemble(
        problem,
        configs,
        energies,
        Float64(temperature),
        Float64(kB),
        metadata,
    )
end
