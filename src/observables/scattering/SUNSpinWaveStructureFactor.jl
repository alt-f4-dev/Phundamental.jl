# Dipolar laboratory-frame tensor computed from site SU(N) product-state excitations.

function _sun_spin_tensor_structure_factor(model::ManyBodyModel, result::SpinWaveResult, field::SpinTensorField, qgrid, axis;
                                           temperature::Real, convention::SpectrumConvention, broadening::AbstractBroadening,
                                           degeneracy_tol::Real, weight_tol::Real, channels, renormalize_transverse::Bool,
                                           spectral_accumulation::Symbol)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    !renormalize_transverse || throw(ArgumentError("SU(N) has no universal dipolar transverse-moment renormalization"))
    spectral_accumulation in (:direct, :auto) || throw(ArgumentError("SU(N) tensor supports direct spectral accumulation"))
    selected = Tuple(Symbol(ch) for ch in channels)
    !isempty(selected) && all(ch -> ch in (:elastic, :one_magnon), selected) || throw(ArgumentError("SU(N) Gaussian response supports :elastic and :one_magnon only"))
    haskey(result.metadata, :zero_mode_regularization) && throw(ArgumentError("regulated SU(N) Goldstone modes do not have certified scattering intensities"))
    sites, positions, normalization = _lswt_field_geometry(model, field)
    convention.fourier_normalization == normalization || throw(ArgumentError("mismatched Fourier normalizations"))
    model.representation.algebra isa SpinAlgebra || throw(ArgumentError("SU(N) tensor requires SpinAlgebra"))
    N = length(result.metadata[:local_bases])
    all(i -> 1 <= i <= N, sites) || throw(BoundsError(result.metadata[:local_bases], sites))
    offsets = result.metadata[:flavor_offsets]
    M = offsets[end]
    frequencies = Float64.(result.frequencies)
    nmode = length(frequencies)
    U = @view result.modes[1:M, :]
    V = size(result.modes, 1) == 2M ? @view(result.modes[M+1:2M, :]) : nothing
    occupations = _lswt_bose_occupations(frequencies, temperature, convention.kB; tol=result.solver.tol)
    hbar = get(model.parameters, :hbar, 1.0)
    amplitude = zeros(ComplexF64, length(sites), nmode, 3)
    moments = zeros(Float64, 3, length(sites))
    depletion = zeros(Float64, length(sites))
    for (a, site) in enumerate(sites)
        local_ops = result.metadata[:local_spin_matrices][site]
        d = offsets[site+1] - offsets[site]
        density = zeros(ComplexF64, d, d)
        for μ in 1:d, ν in 1:d, mode in 1:nmode
            p, q = offsets[site] + μ, offsets[site] + ν
            density[μ, ν] += conj(U[p, mode]) * U[q, mode] * occupations[mode]
            if !isnothing(V)
                density[μ, ν] += V[p, mode] * conj(V[q, mode]) * (occupations[mode] + 1)
            end
        end
        depletion[a] = real(tr(density))
        depletion[a] <= 1 + 100 * result.solver.tol || throw(ArgumentError("SU(N) local boson depletion exceeds one: harmonic expansion is invalid"))
        for α in 1:3
            entries = sun_operator_matrix_elements(result, site, local_ops[α])
            corrected = entries.expectation * (1 - tr(density)) + sum(entries.tangent[μ, ν] * density[μ, ν] for μ in 1:d, ν in 1:d)
            abs(imag(corrected)) <= 100 * result.solver.tol * max(abs(corrected), 1.0) || throw(ArgumentError("SU(N) dipole expectation is not real"))
            moments[α, a] = hbar * real(corrected)
            for mode in 1:nmode, μ in 1:d
                p = offsets[site] + μ
                amplitude[a, mode, α] += hbar * entries.annihilation[μ] * U[p, mode]
                isnothing(V) || (amplitude[a, mode, α] += hbar * entries.creation[μ] * V[p, mode])
            end
        end
    end
    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    output = zeros(ComplexF64, length(qs), length(x), 3, 3)
    scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    phases = zeros(ComplexF64, length(sites))
    for (iq, q) in enumerate(qs)
        _lswt_phase_vector!(phases, q, positions, normalization)
        if :elastic in selected
            elastic = ComplexF64[sum(phases[a] * moments[α, a] for a in eachindex(sites)) for α in 1:3]
            for α in 1:3, β in 1:3
                weight = elastic[α] * conj(elastic[β])
                abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, 0.0, weight, broadening, α, β)
            end
        end
        if :one_magnon in selected
            for mode in 1:nmode
                emission = ComplexF64[sum(phases[a] * amplitude[a, mode, α] for a in eachindex(sites)) for α in 1:3]
                absorption = ComplexF64[sum(phases[a] * conj(amplitude[a, mode, α]) for a in eachindex(sites)) for α in 1:3]
                for α in 1:3, β in 1:3
                    weight = (occupations[mode] + 1) * emission[α] * conj(emission[β])
                    abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, scale * frequencies[mode], weight, broadening, α, β)
                    if occupations[mode] > 0
                        reverse_weight = occupations[mode] * absorption[α] * conj(absorption[β])
                        abs(reverse_weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, -scale * frequencies[mode], reverse_weight, broadening, α, β)
                    end
                end
            end
        end
    end
    metadata = Dict{Symbol,Any}(
        :observable => :spin_tensor_dynamic_structure_factor,
        :response_kind => :correlation_spectrum,
        :frequency_domain => :real,
        :statistics => :bosonic,
        :momentum_scope => :finite_cell_form_factor,
        :response_order => 2,
        :components => (:x, :y, :z),
        :temperature => float(temperature),
        :fourier_normalization => normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :solver => :spin_wave,
        :formalism => :SUN,
        :tensor_definition => :S_alpha_beta,
        :approximation => :gaussian_multiboson_spin_wave,
        :channels => selected,
        :field_sites => copy(sites),
        :zero_point_or_thermal_depletion => depletion,
        :ordered_moments => moments,
        :spectral_accumulation => :direct_kernel_evaluation,
        :two_magnon_channel => :not_implemented,
    )
    return TensorSpectrumResult(qs, x, output, metadata)
end
