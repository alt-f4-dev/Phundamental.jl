# Validation.jl

"""
    validate_metropolis_detailed_balance(problem; temperature, ...)

Check the one-step transition relation

    pi(s) P(s->s') = pi(s') P(s'->s)

for deterministic pseudorandom Ising configurations and uniformly selected
single-site flips. Computation is performed in log space.
"""
function validate_metropolis_detailed_balance(
    problem::IsingProblem;
    temperature::Real,
    kB::Real=1.0,
    trials::Integer=1000,
    seed::Integer=99173,
    tolerance::Real=1e-12,
)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    trials > 0 || throw(ArgumentError("trials must be positive"))
    beta = inv(Float64(kB) * Float64(temperature))
    rng = MersenneTwister(seed)
    N = length(problem)
    logproposal = -log(N)

    worst = 0.0
    for _ in 1:trials
        s = Int8[rand(rng, Bool) ? 1 : -1 for _ in 1:N]
        i = rand(rng, 1:N)
        E = ising_energy(problem, s)
        local_product = problem.J * s
        dE = Float64(ising_flip_delta(problem, s, local_product, i))

        # Forward and reverse Metropolis transition logs. The unknown partition
        # function cancels exactly.
        log_forward = -beta * E + logproposal + min(0.0, -beta * dE)
        E2 = E + dE
        log_reverse = -beta * E2 + logproposal + min(0.0, beta * dE)
        worst = max(worst, abs(log_forward - log_reverse))
    end

    return Dict{Symbol,Any}(
        :passed => worst <= tolerance,
        :max_log_balance_error => worst,
        :trials => Int(trials),
        :proposal_symmetric => true,
    )
end

function _exact_ising_statistics(
    problem::IsingProblem,
    temperature::Real,
    kB::Real,
)
    N = length(problem)
    N <= 24 || throw(ArgumentError(
        "exact Ising enumeration is limited to N<=24 by this validator"
    ))
    beta = inv(Float64(kB) * Float64(temperature))
    nstates = Int(1) << N

    energies = Vector{Float64}(undef, nstates)
    mags = Vector{Float64}(undef, nstates)
    state = Vector{Int8}(undef, N)

    for mask in 0:(nstates - 1)
        @inbounds for i in 1:N
            state[i] = ((mask >> (i - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
        energies[mask + 1] = ising_energy(problem, state)
        mags[mask + 1] = sum(state) / N
    end

    Emin = minimum(energies)
    w = exp.(-beta .* (energies .- Emin))
    Zs = sum(w)
    p = w ./ Zs

    U = dot(p, energies)
    E2 = dot(p, energies .^ 2)
    M = dot(p, mags)
    Cv = (E2 - U^2) / (Float64(kB) * Float64(temperature)^2)

    return Dict{Symbol,Float64}(
        :internal_energy => U,
        :heat_capacity => max(Cv, 0.0),
        :magnetization_per_site => M,
    )
end

"""
    validate_sampling_against_exact_classical(problem; ...)

Small-system independent validation against exact enumeration.
"""
function validate_sampling_against_exact_classical(
    problem::IsingProblem;
    temperature::Real,
    kB::Real=1.0,
    sampler::MetropolisSampler=MetropolisSampler(
        samples=20_000,
        burnin_sweeps=5_000,
        sweeps_per_sample=2,
        seed=2026,
    ),
    sigma_tolerance::Real=6.0,
    absolute_floor::Real=5e-3,
)
    exact = _exact_ising_statistics(problem, temperature, kB)
    ensemble = sample(sampler, problem; temperature=temperature, kB=kB)

    U = mean(ensemble.energies)
    Uerr = correlated_stderr(ensemble.energies)
    Cv = var(ensemble.energies; corrected=false) /
         (Float64(kB) * Float64(temperature)^2)

    mags = [
        sum(view(ensemble.configurations, :, r)) / length(problem)
        for r in 1:nsamples(ensemble)
    ]
    M = mean(mags)
    Merr = correlated_stderr(mags)

    Uok = abs(U - exact[:internal_energy]) <=
          max(sigma_tolerance * Uerr, absolute_floor)
    Mok = abs(M - exact[:magnetization_per_site]) <=
          max(sigma_tolerance * Merr, absolute_floor)

    # Heat-capacity uncertainty is more delicate because it is a variance.
    # Use a conservative relative/absolute gate for this built-in validator.
    Cscale = max(abs(exact[:heat_capacity]), 1.0)
    Cok = abs(Cv - exact[:heat_capacity]) <= max(0.08 * Cscale, absolute_floor)

    return Dict{Symbol,Any}(
        :passed => Uok && Mok && Cok,
        :ensemble => ensemble,
        :exact => exact,
        :sampled => Dict(
            :internal_energy => U,
            :heat_capacity => Cv,
            :magnetization_per_site => M,
        ),
        :stderr => Dict(
            :internal_energy => Uerr,
            :magnetization_per_site => Merr,
        ),
        :internal_energy_passed => Uok,
        :heat_capacity_passed => Cok,
        :magnetization_passed => Mok,
    )
end
