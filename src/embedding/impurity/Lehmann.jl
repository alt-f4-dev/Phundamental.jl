# ============================================================================
# OrbitalBilinearLehmann.jl: preserved implementation from src/embedding/OrbitalBilinearLehmann.jl
# ============================================================================


# Exact finite-bath orbital-bilinear response: equal-time asymptotic inputs.
# This is not a reconstruction of the complete three-frequency fermionic vertex.

"""Exact finite-bath Lehmann data for orbital bilinears `B_ab = d†_a d_b`.

The connected bosonic response uses `χ_AB(iΩ) = ∫₀ᵝ dτ exp(iΩτ)
(⟨A(τ)B(0)⟩ - ⟨A⟩⟨B⟩)`. Bilinear ordering follows `pairs`.
"""
struct OrbitalBilinearLehmann
    beta::Float64
    labels::Vector{Symbol}
    pairs::Vector{Tuple{Int,Int}}
    energies::Vector{Float64}
    probabilities::Vector{Float64}
    operators::Array{ComplexF64,3}
    expectations::Vector{ComplexF64}
    spectral_radius::Float64
    metadata::Dict{Symbol,Any}
end

"""Exact bosonic spectral moments with elementwise certified finite-bath bounds."""
struct OrbitalBilinearMoments
    beta::Float64
    labels::Vector{Symbol}
    pairs::Vector{Tuple{Int,Int}}
    coefficients::Array{ComplexF64,3}
    absolute_strengths::Matrix{Float64}
    spectral_radius::Float64
    metadata::Dict{Symbol,Any}
end

"""Construct equal-time two-particle inputs from the complete interacting ED eigensystem.

The returned operator tensor uses the exact ED eigenbasis and includes every
particle-number sector. `max_storage_bytes` is checked before operator allocation.
This is the two-bilinear (bosonic) correlation needed for high-frequency K1
construction, but neither the K1 interaction contraction nor three-leg K2 is inferred.
"""
function orbital_bilinear_lehmann(state::MultiOrbitalEDThermalState; pairs=nothing,
                                 max_storage_bytes::Integer=268435456)
    n = state.hamiltonian.impurity_dimension
    chosen = _orbital_pair_selection(n, pairs)
    dim = state.metadata[:full_hilbert_dimension]
    np = length(chosen)
    required = big(16) * (big(np) + 2 * big(n)) * dim^2 + 3big(8) * dim
    max_storage_bytes > 0 && required <= max_storage_bytes ||
        throw(ArgumentError("orbital-bilinear ED operators exceed max_storage_bytes"))
    modes = sort!(unique(vcat([a for (a, b) in chosen], [b for (a, b) in chosen])))
    annihilation, energies = _orbital_pair_operators(state, modes)
    probabilities = reduce(vcat, (sector.probabilities for sector in state.sectors))
    operators = Array{ComplexF64}(undef, dim, dim, np)
    for (ip, (a, b)) in enumerate(chosen)
        mul!(@view(operators[:, :, ip]), adjoint(annihilation[a]), annihilation[b])
    end
    means = ComplexF64[sum(probabilities[s] * operators[s, s, ip] for s in 1:dim) for ip in 1:np]
    radius = maximum(energies) - minimum(energies)
    all(isfinite, operators) && all(isfinite, means) && isapprox(sum(probabilities), 1.0; atol=1e-12) ||
        throw(ErrorException("invalid finite-bath bilinear Lehmann representation"))
    metadata = Dict{Symbol,Any}(:method => :exact_finite_bath_bilinear_lehmann,
        :estimated_operator_storage_bytes => Int(required), :exact_equal_time_k1_input => true,
        :two_particle_vertex_tail_certified => false, :three_leg_k2_available => false)
    return OrbitalBilinearLehmann(state.beta, Symbol.(state.hamiltonian.labels[1:n]), chosen,
        energies, probabilities, operators, means, radius, metadata)
end

"""Evaluate the connected bosonic bilinear response, including degenerate transitions.

The static component includes `β p_s` for every degenerate pair of eigenstates
and subtracts `β⟨A⟩⟨B⟩`. No imaginary-time Fourier quadrature is used.
"""
function orbital_bilinear_susceptibility(data::OrbitalBilinearLehmann, axis::BosonicMatsubaraAxis;
                                         degeneracy_tolerance::Real=1e-12, max_output_bytes::Integer=268435456)
    isapprox(axis.beta, data.beta; atol=1e-12 * max(data.beta, 1.0), rtol=1e-12) ||
        throw(ArgumentError("bosonic axis temperature differs from the ED state"))
    isfinite(degeneracy_tolerance) && degeneracy_tolerance >= 0 || throw(ArgumentError("invalid degeneracy tolerance"))
    np, dim, nb = length(data.pairs), length(data.energies), length(axis)
    required = big(16) * np^2 * nb
    max_output_bytes > 0 && required <= max_output_bytes ||
        throw(ArgumentError("bilinear response exceeds max_output_bytes"))
    values = zeros(ComplexF64, np, np, nb)
    @inbounds for (im, m) in enumerate(axis.indices)
        z = axis.values[im]
        for t in 1:dim, s in 1:dim
            delta = data.energies[t] - data.energies[s]
            weight = if abs(delta) <= degeneracy_tolerance
                iszero(m) ? data.beta * data.probabilities[s] : 0.0
            else
                (data.probabilities[s] - data.probabilities[t]) / (delta - z)
            end
            iszero(weight) && continue
            for j in 1:np, i in 1:np
                values[i, j, im] += weight * data.operators[s, t, i] * data.operators[t, s, j]
            end
        end
        if iszero(m)
            for j in 1:np, i in 1:np
                values[i, j, im] -= data.beta * data.expectations[i] * data.expectations[j]
            end
        end
    end
    return (values=values, pairs=copy(data.pairs), labels=copy(data.labels), axis=axis,
        metadata=Dict{Symbol,Any}(:method => :exact_finite_bath_bilinear_response,
            :connected => true, :bosonic_high_frequency_kernel_input => true,
            :two_particle_vertex_tail_certified => false))
end

"""Compute rigorous `1/(iΩ)` moments of the *connected bosonic bilinear response*.

For nonzero Ω, `χ_AB(z) = Σ_{st} (p_s-p_t) A_st B_ts/(E_t-E_s-z)`.
The coefficient of `z^(-k-1)` is `-Σ(p_s-p_t)(E_t-E_s)^k A_st B_ts`.
The absolute transition strength bounds its remainder elementwise. These are
not the full fermionic vertex K1/K2 kernels.
"""
function orbital_bilinear_moments(data::OrbitalBilinearLehmann; order::Integer=3,
                                  max_output_bytes::Integer=268435456)
    order > 0 || throw(ArgumentError("moment order must be positive"))
    np = length(data.pairs)
    required = big(16) * np^2 * order + big(8) * np^2
    max_output_bytes > 0 && required <= max_output_bytes ||
        throw(ArgumentError("bilinear moments exceed max_output_bytes"))
    coeff = zeros(ComplexF64, np, np, Int(order))
    strengths = zeros(Float64, np, np)
    dim = length(data.energies)
    @inbounds for t in 1:dim, s in 1:dim
        gap = data.energies[t] - data.energies[s]
        population = data.probabilities[s] - data.probabilities[t]
        iszero(population) && continue
        for j in 1:np, i in 1:np
            amplitude = population * data.operators[s, t, i] * data.operators[t, s, j]
            strengths[i, j] += abs(amplitude)
            power = -amplitude
            for k in 1:Int(order)
                coeff[i, j, k] += power
                power *= gap
            end
        end
    end
    return OrbitalBilinearMoments(data.beta, copy(data.labels), copy(data.pairs), coeff, strengths,
        data.spectral_radius, Dict{Symbol,Any}(:method => :exact_bosonic_bilinear_moments,
            :certified_for => :finite_bath_bilinear_response_only,
            :two_particle_vertex_tail_certified => false, :three_leg_k2_available => false))
end

"""Evaluate high-frequency bilinear moments with elementwise certified remainder bounds.

At nonzero Matsubara frequencies with `|z| > R`, the bound is
`S_AB * (R/|z|)^order / (|z|-R)`, with `S_AB` the absolute spectral weight.
"""
function orbital_bilinear_tail(moments::OrbitalBilinearMoments, axis::BosonicMatsubaraAxis;
                               max_output_bytes::Integer=268435456)
    isapprox(axis.beta, moments.beta; atol=1e-12 * max(axis.beta, 1.0), rtol=1e-12) ||
        throw(ArgumentError("bosonic axis temperature differs from the Lehmann moments"))
    np, _, order = size(moments.coefficients)
    required = big(16 + 8) * np^2 * length(axis)
    max_output_bytes > 0 && required <= max_output_bytes ||
        throw(ArgumentError("bilinear tail and bound arrays exceed max_output_bytes"))
    values = zeros(ComplexF64, np, np, length(axis))
    bounds = zeros(Float64, np, np, length(axis))
    radius = moments.spectral_radius
    for (im, z) in enumerate(axis.values)
        abs(z) > radius || throw(ArgumentError("the bosonic tail requires |iΩ| > the ED spectral radius"))
        invz = inv(z)
        power = invz
        for k in 1:order
            @views values[:, :, im] .+= power .* moments.coefficients[:, :, k]
            power *= invz
        end
        @views bounds[:, :, im] .= moments.absolute_strengths .* ((radius / abs(z))^order / (abs(z) - radius))
    end
    return (values=values, error_bounds=bounds, axis=axis,
        certified_for=:finite_bath_bilinear_response_only,
        two_particle_vertex_tail_certified=false)
end

"""Project exact bilinear correlations onto longitudinal SU(2) density/magnetic channels.

`pairs` must contain the complete spin-interleaved spin-orbital pair space.
The returned density and magnetic values use `χd = χ↑↑ + χ↑↓` and
`χm = χ↑↑ - χ↑↓`. Longitudinal equalities are tested, but the full transverse
SU(2) Ward identity and vertex asymptotics are not inferred.
"""
function orbital_bilinear_spin_channels(data::OrbitalBilinearLehmann, axis::BosonicMatsubaraAxis;
                                        atol::Real=1e-9, rtol::Real=1e-7,
                                        max_output_bytes::Integer=268435456)
    nspin = length(data.labels)
    iseven(nspin) || throw(ArgumentError("SU(2) projection requires spin-interleaved spin-orbitals"))
    data.pairs == [(a, b) for a in 1:nspin for b in 1:nspin] ||
        throw(ArgumentError("SU(2) bilinear projection requires the complete canonical pair space"))
    all(x -> isfinite(x) && x >= 0, (atol, rtol)) || throw(ArgumentError("invalid spin-channel tolerances"))
    norb = nspin ÷ 2
    np = norb^2
    required = big(16) * (nspin^4 + 2 * big(np)^2) * length(axis)
    max_output_bytes > 0 && required <= max_output_bytes ||
        throw(ArgumentError("SU(2) bilinear responses exceed max_output_bytes"))
    response = orbital_bilinear_susceptibility(data, axis; max_output_bytes=max_output_bytes)
    charge = zeros(ComplexF64, np, np, length(axis))
    spin = similar(charge)
    residual = 0.0
    scale = 0.0
    for d in 1:norb, c in 1:norb, b in 1:norb, a in 1:norb
        i = (a - 1) * norb + b
        j = (c - 1) * norb + d
        up_left = (2a - 2) * nspin + 2b - 1
        down_left = (2a - 1) * nspin + 2b
        up_right = (2c - 2) * nspin + 2d - 1
        down_right = (2c - 1) * nspin + 2d
        for im in eachindex(axis.indices)
            uu = response.values[up_left, up_right, im]
            ud = response.values[up_left, down_right, im]
            dd = response.values[down_left, down_right, im]
            du = response.values[down_left, up_right, im]
            residual = max(residual, abs(uu - dd), abs(ud - du))
            scale = max(scale, abs(uu), abs(ud), abs(dd), abs(du))
            charge[i, j, im] = uu + ud
            spin[i, j, im] = uu - ud
        end
    end
    bound = atol + rtol * scale
    residual <= bound || throw(ArgumentError("exact bilinear response fails longitudinal SU(2) equality"))
    return (density=charge, magnetic=spin, axis=axis,
        spatial_pairs=[(a, b) for a in 1:norb for b in 1:norb],
        metadata=Dict{Symbol,Any}(:longitudinal_residual => residual,
            :longitudinal_tolerance => bound, :exact_finite_bath => true,
            :full_su2_ward_identity_verified => false, :two_particle_vertex_tail_certified => false))
end

# ============================================================================
# OrbitalLehmannTail.jl: preserved implementation from src/embedding/OrbitalLehmannTail.jl
# ============================================================================


# Certified high-frequency expansion of the *one-particle* finite-bath Lehmann Green function.
# This does not provide the K1/K2 kernels of the two-particle vertex.

"""Exact finite-bath spectral moments and a rigorous spectral-radius bound for the one-particle remainder."""
struct OrbitalLehmannMoments
    coefficients::Array{ComplexF64,3}
    beta::Float64
    spectral_radius::Float64
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

"""Compute the first `order` Lehmann coefficients of the impurity matrix Green function.

For the finite-bath ED spectral representation `G(z) = Σ_j A_j/(z + δ_j)`, the
coefficient of `z^(-k-1)` is `M_k = Σ_j (-δ_j)^k A_j`. Each `A_j` is
positive semidefinite. No high-frequency extrapolation or fitted tail is used.
"""
function orbital_impurity_lehmann_moments(state::MultiOrbitalEDThermalState;
                                            order::Integer=3, max_workspace_bytes::Integer=268435456)
    order >= 1 || throw(ArgumentError("order must be positive"))
    max_workspace_bytes > 0 || throw(ArgumentError("max_workspace_bytes must be positive"))
    n = state.hamiltonian.impurity_dimension
    largest_sector = maximum(length(sector.masks) for sector in state.sectors)
    budget = big(16) * (big(n)^2 * order + 6 * big(n) * largest_sector^2)
    budget <= max_workspace_bytes || throw(ArgumentError("Lehmann moment workspace exceeds max_workspace_bytes"))
    coefficients = zeros(ComplexF64, n, n, Int(order))
    maximum_gap = 0.0
    for source_index in eachindex(state.sectors)
        target_index = _multi_ed_target(state, source_index)
        target_index == 0 && continue
        source = state.sectors[source_index]
        target = state.sectors[target_index]
        transitions = [_multi_ed_transition(source, target, a) for a in 1:n]
        for s in eachindex(source.energies), t in eachindex(target.energies)
            weight = source.probabilities[s] + target.probabilities[t]
            iszero(weight) && continue
            gap = target.energies[t] - source.energies[s]
            maximum_gap = max(maximum_gap, abs(gap))
            coefficient = weight
            for k in 1:order
                for b in 1:n, a in 1:n
                    coefficients[a, b, k] += coefficient * transitions[a][t, s] * conj(transitions[b][t, s])
                end
                coefficient *= -gap
            end
        end
    end
    zeroth = Matrix(@view coefficients[:, :, 1])
    residual = opnorm(zeroth - Matrix{ComplexF64}(I, n, n), 2)
    all(isfinite, coefficients) && isfinite(maximum_gap) || throw(ErrorException("nonfinite finite-bath spectral moments"))
    metadata = Dict{Symbol,Any}(:method => :exact_finite_bath_lehmann_moments,
        :remainder_bound => :operator_norm_spectral_radius, :zeroth_moment_residual => residual,
        :maximum_excitation_energy => maximum_gap, :moment_count => Int(order),
        :estimated_workspace_bytes => Int(budget), :one_particle_only => true,
        :two_particle_vertex_tail_certified => false)
    return OrbitalLehmannMoments(coefficients, state.beta, maximum_gap,
        Symbol.(state.hamiltonian.labels[1:n]), metadata)
end

"""Evaluate a truncated matrix Lehmann series and its *rigorous finite-bath* operator-norm error bound.

The bound applies only when `|z| > spectral_radius`. For `p` coefficients,
`||G(z)-G_p(z)||₂ ≤ ||M₀||₂ (R/|z|)^p / (|z|-R)`, with `R` the
maximum retained Lehmann excitation energy. It is a one-particle bound and
must not be interpreted as a bound on a multiorbital two-particle vertex.
"""
function orbital_impurity_lehmann_tail(moments::OrbitalLehmannMoments, axis::FermionicMatsubaraAxis)
    isapprox(axis.beta, moments.beta; atol=1e-12 * max(axis.beta, 1.0), rtol=1e-12) ||
        throw(ArgumentError("Matsubara axis and Lehmann moments have different temperatures"))
    n, _, count = size(moments.coefficients)
    values = zeros(ComplexF64, n, n, length(axis))
    bounds = Vector{Float64}(undef, length(axis))
    R = moments.spectral_radius
    mass = opnorm(@view(moments.coefficients[:, :, 1]), 2)
    for (i, z) in enumerate(axis.values)
        magnitude = abs(z)
        magnitude > R || throw(ArgumentError("a rigorous Lehmann tail requires |iω| > the largest transition energy"))
        reciprocal = inv(z)
        power = reciprocal
        for k in 1:count
            @views values[:, :, i] .+= power .* moments.coefficients[:, :, k]
            power *= reciprocal
        end
        bounds[i] = mass * (R / magnitude)^count / (magnitude - R)
    end
    metadata = Dict{Symbol,Any}(:method => :finite_bath_spectral_moment_expansion,
        :one_particle_tail_operator_norm_bounded => true,
        :two_particle_vertex_tail_certified => false, :moment_count => count)
    return (green=GreenFunction(axis, values; labels=moments.labels, metadata=metadata),
        error_bounds=bounds, certified_for=:one_particle_finite_bath_only)
end
