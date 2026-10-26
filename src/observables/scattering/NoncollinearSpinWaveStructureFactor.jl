# Laboratory-frame elastic and one-magnon scattering for noncollinear SU(2) references.

"""Cartesian spin tensor for a generalized finite-cell SU(2) spin-wave solution.

The elastic and one-magnon sectors are supported. The connected two-magnon longitudinal sector is not implemented and is rejected explicitly. Regulated Goldstone modes are not used to claim certified scattering intensities.
"""
function spin_tensor_structure_factor(model::ManyBodyModel, result::SpinWaveResult{<:SpinWaveSolver}, field::SpinTensorField, qgrid, axis;
                                      temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.x.normalization),
                                      broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                      weight_tol::Real=0.0, channels=(:elastic, :one_magnon), renormalize_transverse::Bool=false,
                                      spectral_accumulation::Symbol=:direct)
    result.solver.formalism === :SUN && return _sun_spin_tensor_structure_factor(model, result, field, qgrid, axis;
        temperature, convention, broadening, degeneracy_tol, weight_tol, channels, renormalize_transverse, spectral_accumulation)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    selected = Tuple(Symbol(ch) for ch in channels)
    !isempty(selected) && all(ch -> ch in (:elastic, :one_magnon), selected) || throw(ArgumentError(
        "generalized SU(2) LSWT currently supports :elastic and :one_magnon, but not :two_magnon"
    ))
    spectral_accumulation in (:direct, :auto) || throw(ArgumentError("generalized spin-wave tensor currently supports direct spectral accumulation"))
    haskey(result.metadata, :zero_mode_regularization) && throw(ArgumentError(
        "regulated Goldstone-mode vectors cannot provide certified scattering intensities; use a finite-gap reference or explicit infrared treatment"
    ))
    sites, positions, normalization = _lswt_field_geometry(model, field)
    convention.fourier_normalization == normalization || throw(ArgumentError("mismatched Fourier normalizations"))
    space = model.representation.state_space isa BiorthogonalSpace ? model.representation.state_space.ambient : model.representation.state_space
    spins = Float64.(space.spins)
    hbar = get(model.parameters, :hbar, 1.0)
    frames = result.metadata[:reference_frames]
    directions = result.metadata[:reference_directions]
    N = length(spins)
    size(directions) == (3, N) || throw(DimensionMismatch("spin-wave reference has wrong site count"))
    frequencies = Float64.(result.frequencies)
    nmode = length(frequencies)
    U = view(result.modes, 1:N, :)
    V = size(result.modes, 1) == 2N ? view(result.modes, N+1:2N, :) : nothing
    occupations = _lswt_bose_occupations(frequencies, temperature, convention.kB; tol=result.solver.tol)
    depletion = _lswt_local_depletion(U, V, occupations, sites)
    moment = spins[sites] .- depletion
    minimum(moment) >= -100 * result.solver.tol || throw(ArgumentError("spin-wave depletion exceeds the reference moment"))
    moment = max.(moment, 0.0)
    local_scale = renormalize_transverse ? moment : spins[sites]
    amplitude = zeros(ComplexF64, length(sites), nmode, 3)
    for (a, site) in enumerate(sites), ν in 1:nmode, α in 1:3
        ex = frames[site][α, 1]
        ey = frames[site][α, 2]
        u = U[site, ν]
        v = isnothing(V) ? 0.0 + 0.0im : V[site, ν]
        amplitude[a, ν, α] = hbar * sqrt(local_scale[a] / 2) * ((ex - im * ey) * u + (ex + im * ey) * v)
    end
    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    output = zeros(ComplexF64, length(qs), length(x), 3, 3)
    scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    phases = zeros(ComplexF64, length(sites))
    for (iq, q) in enumerate(qs)
        _lswt_phase_vector!(phases, q, positions, normalization)
        if :elastic in selected
            elastic = ComplexF64[sum(phases[a] * hbar * moment[a] * directions[α, sites[a]] for a in eachindex(sites)) for α in 1:3]
            for α in 1:3, β in 1:3
                weight = elastic[α] * conj(elastic[β])
                abs(weight) <= weight_tol && continue
                _lswt_add_scalar_line!(output, iq, x, 0.0, weight, broadening, α, β)
            end
        end
        if :one_magnon in selected
            for ν in 1:nmode
                emission = ComplexF64[sum(phases[a] * amplitude[a, ν, α] for a in eachindex(sites)) for α in 1:3]
                absorption = ComplexF64[sum(phases[a] * conj(amplitude[a, ν, α]) for a in eachindex(sites)) for α in 1:3]
                for α in 1:3, β in 1:3
                    weight = (occupations[ν] + 1) * emission[α] * conj(emission[β])
                    abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, scale * frequencies[ν], weight, broadening, α, β)
                    if occupations[ν] > 0
                        reverse_weight = occupations[ν] * absorption[α] * conj(absorption[β])
                        abs(reverse_weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, -scale * frequencies[ν], reverse_weight, broadening, α, β)
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
        :tensor_definition => :S_alpha_beta,
        :approximation => :gaussian_linear_spin_wave,
        :channels => selected,
        :field_sites => copy(sites),
        :reference_directions => copy(directions),
        :zero_point_or_thermal_depletion => depletion,
        :ordered_moments => moment,
        :transverse_normalization => renormalize_transverse ? :ordered_moment : :bare_spin,
        :spectral_accumulation => :direct_kernel_evaluation,
        :two_magnon_channel => :not_implemented,
    )
    return TensorSpectrumResult(qs, x, output, metadata)
end
