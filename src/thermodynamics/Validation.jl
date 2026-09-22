# Thermodynamic consistency and convergence diagnostics.

"""
    validate_thermal_state(state; atol=1e-10, rtol=1e-8)

Basic thermodynamic consistency checks. Returns a diagnostic dictionary rather than throwing for ordinary numerical residuals.
"""
function validate_thermal_state(state::AbstractThermalState; atol::Real=1e-10, rtol::Real=1e-8)
    p = state.point
    finite_scalars = all(isfinite, (p.free_energy, p.internal_energy, p.entropy, p.heat_capacity))
    nonnegative_cv = p.heat_capacity >= -atol

    thermo_identity = iszero(p.temperature) ? abs(p.free_energy - p.internal_energy) : abs(p.free_energy - (p.internal_energy - p.temperature * p.entropy))
    scale = max(abs(p.free_energy), abs(p.internal_energy), 1.0)
    identity_ok = thermo_identity <= atol + rtol * scale

    trace_error = NaN
    if state isa ExactGibbsState
        trace_error = abs(sum(state.probabilities) - 1)
    elseif state isa SampledThermalState && !isnothing(state.filtered_vectors) && !isnothing(state.basis)
        rho = length(state.basis) <= 512 ? density_matrix(state; max_dimension=512) : nothing
        !isnothing(rho) && (trace_error = density_matrix_trace_error(rho))
    end

    return Dict{Symbol,Any}(
        :finite_scalars => finite_scalars,
        :nonnegative_heat_capacity => nonnegative_cv,
        :thermodynamic_identity_residual => thermo_identity,
        :thermodynamic_identity_passed => identity_ok,
        :density_trace_error => trace_error,
        :passed => finite_scalars && nonnegative_cv && identity_ok && (isnan(trace_error) || trace_error <= atol + rtol),
    )
end

"""
    validate_sampling_against_exact(model; temperature, sampled_method, kB=1)

Compare sampled F, U, S, and Cv against exact Gibbs thermodynamics for a small validation model.
"""
function validate_sampling_against_exact(model::ManyBodyModel; temperature::Real, sampled_method::Union{ThermalTypicality,ThermalLanczos,CanonicalTPQ}=ThermalLanczos(), kB::Real=1.0)
    exact = thermal_state(model, ExactGibbs(); temperature=temperature, kB=kB)
    sampled = thermal_state(model, sampled_method; temperature=temperature, kB=kB)
    return Dict{Symbol,Any}(
        :exact => exact.point,
        :sampled => sampled.point,
        :absolute_error => Dict(
            :free_energy => abs(free_energy(sampled) - free_energy(exact)),
            :internal_energy => abs(internal_energy(sampled) - internal_energy(exact)),
            :entropy => abs(entropy(sampled) - entropy(exact)),
            :heat_capacity => abs(heat_capacity(sampled) - heat_capacity(exact)),
        ),
    )
end

"""
    validate_exact_spectrum_thermodynamics(model, temperatures; kB=1, atol=1e-10)

Verify that the eigenvalue-only exact path reproduces thermodynamics from the complete eigensystem. Intended for small regression models.
"""
function validate_exact_spectrum_thermodynamics(model::ManyBodyModel, temperatures; kB::Real=1.0, atol::Real=1e-10)
    solver = ExactDiagonalization()
    spectrum = energy_spectrum(solver, model)
    eigensystem = solve(solver, model)
    curve_spectrum = thermal_curve(spectrum, temperatures; kB=kB)
    curve_eigensystem = thermal_curve(eigensystem, temperatures; kB=kB)
    errors = Dict{Symbol,Float64}(
        :logZ => maximum(abs.(curve_spectrum.logZ .- curve_eigensystem.logZ)),
        :free_energy => maximum(abs.(curve_spectrum.free_energy .- curve_eigensystem.free_energy)),
        :internal_energy => maximum(abs.(curve_spectrum.internal_energy .- curve_eigensystem.internal_energy)),
        :entropy => maximum(abs.(curve_spectrum.entropy .- curve_eigensystem.entropy)),
        :heat_capacity => maximum(abs.(curve_spectrum.heat_capacity .- curve_eigensystem.heat_capacity)),
    )
    return Dict{Symbol,Any}(:passed => maximum(values(errors)) <= atol, :errors => errors, :dimension => spectrum.dimension)
end

"""
    validate_linked_cluster_closure(result; property=:free_energy)

When the expansion includes the entire connected reference graph, the sum of all embedded linked weights should reproduce the raw full-reference extensive property, divided by the number of reference sites.
"""
function validate_linked_cluster_closure(result::ClusterThermodynamicsResult; property::Symbol=:free_energy)
    fullkey = Tuple(1:result.reference_sites)
    haskey(result.raw_curves, fullkey) || return Dict{Symbol,Any}(:available => false, :reason => :full_reference_cluster_not_included)

    raw = _curve_property(result.raw_curves[fullkey], property) ./ result.reference_sites
    linked = bulk_estimate(result, property)
    residual = maximum(abs.(raw .- linked))
    return Dict{Symbol,Any}(
        :available => true,
        :property => property,
        :max_abs_residual => residual,
        :raw_reference_per_site => raw,
        :linked_per_site => linked,
    )
end

"""
    validate_sector_thermodynamic_recombination(model; temperatures=[0.25, 0.5, 1.0], ...)

For a small spin-1/2 model, validate the exact thermodynamic sector-recombination algebra independently of stochastic TPQ sampling. Dense spectra are computed in every representative fixed-`Sᶻ` sector, weighted by exact symmetry multiplicity, and compared with the full `ExactGibbs` curve.
"""
function validate_sector_thermodynamic_recombination(model::ManyBodyModel; temperatures=[0.25, 0.5, 1.0], kB::Real=1.0, exploit_equivalent_sectors::Bool=true, hbar::Real=get(model.parameters, :hbar, 1.0), atol::Real=1e-10, max_total_dimension::Integer=100_000)
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperature grid cannot be empty"))
    all(>(0), Ts) || throw(ArgumentError("validation temperatures must be positive"))
    plan = Solvers.spin_sector_plan(model; exploit_equivalent_sectors=exploit_equivalent_sectors, hbar=hbar)
    plan.total_dimension <= max_total_dimension || throw(ArgumentError("full validation dimension $(plan.total_dimension) exceeds max_total_dimension=$max_total_dimension"))

    sector_energies = Vector{Vector{Float64}}(undef, length(plan.sectors))
    for i in eachindex(plan.sectors)
        packed = Solvers.optimized_matrix_free_operator(model; sector=plan.sectors[i], hbar=hbar, max_basis_dimension=max_total_dimension, materialize_basis=true, threaded=false)
        realized = realize(model.hamiltonian, packed.basis; hbar=hbar, sparse=false, check_physical_closure=false)
        sector_energies[i] = Float64.(eigvals(Hermitian(0.5 .* (realized.matrix .+ adjoint(realized.matrix)))))
    end

    reference = thermal_curve(model, ExactGibbs(solver=ExactDiagonalization(hbar=hbar)), Ts; kB=kB)
    logZ = zeros(Float64, length(Ts))
    F = similar(logZ)
    U = similar(logZ)
    S = similar(logZ)
    Cv = similar(logZ)

    global_eref = minimum(minimum(E) for E in sector_energies)
    for (iT, T) in enumerate(Ts)
        beta = inv(Float64(kB) * T)
        zsum = 0.0
        usum = 0.0
        h2sum = 0.0
        for s in eachindex(sector_energies)
            mult = Float64(plan.multiplicities[s])
            for energy in sector_energies[s]
                weight = mult * exp(-beta * (energy - global_eref))
                zsum += weight
                usum += weight * energy
                h2sum += weight * energy^2
            end
        end
        logZ[iT] = -beta * global_eref + log(zsum)
        U[iT] = usum / zsum
        variance = max(h2sum / zsum - U[iT]^2, 0.0)
        F[iT] = -Float64(kB) * T * logZ[iT]
        S[iT] = (U[iT] - F[iT]) / T
        Cv[iT] = variance / (Float64(kB) * T^2)
    end

    errors = Dict{Symbol,Float64}(
        :logZ => maximum(abs.(logZ .- reference.logZ)),
        :free_energy => maximum(abs.(F .- reference.free_energy)),
        :internal_energy => maximum(abs.(U .- reference.internal_energy)),
        :entropy => maximum(abs.(S .- reference.entropy)),
        :heat_capacity => maximum(abs.(Cv .- reference.heat_capacity)),
    )
    return Dict{Symbol,Any}(
        :passed => maximum(values(errors)) <= atol,
        :errors => errors,
        :dimension => plan.total_dimension,
        :sectors_evaluated => length(plan.sectors),
        :equivalent_sector_pairing => plan.equivalence_used,
    )
end
