# One-magnon laboratory-frame Bloch spin tensors, shared by dipolar and site-SU(N) references.

function _spinwave_bloch_local_amplitudes(model::ManyBodyModel, result::SpinWaveDispersionResult)
    solver = result.solver
    spins = Float64.((model.representation.state_space isa BiorthogonalSpace ? model.representation.state_space.ambient : model.representation.state_space).spins)
    N = length(spins)
    if solver.formalism === :SUN
        ref = Solvers._sun_resolved_reference(solver.reference, spins)
        local_amplitudes = Matrix{ComplexF64}[]
        moments = zeros(Float64, 3, N)
        for i in 1:N
            ops = Solvers._sun_spin_matrices(spins[i])
            U = ref.bases[i]
            d = size(U, 2) - 1
            f = zeros(ComplexF64, 3, d)
            for α in 1:3
                moments[α, i] = real(dot(ref.amplitudes[i], ops[α] * ref.amplitudes[i]))
                for μ in 1:d
                    f[α, μ] = dot(view(U, :, 1), ops[α] * view(U, :, μ + 1))
                end
            end
            push!(local_amplitudes, f)
        end
        return local_amplitudes, moments
    end
    ref = solver.reference
    size(ref.directions) == (3, N) || throw(DimensionMismatch("Bloch reference and model disagree"))
    local_amplitudes = Matrix{ComplexF64}[]
    for i in 1:N
        ex, ey = view(ref.frames[i], :, 1), view(ref.frames[i], :, 2)
        push!(local_amplitudes, reshape(sqrt(spins[i] / 2) .* (ex .- im .* ey), 3, 1))
    end
    return local_amplitudes, ref.directions .* reshape(spins, 1, N)
end

function _spinwave_bloch_amplitudes(result::SpinWaveDispersionResult, local_amplitudes, sites, iq, normalization::Symbol, hbar::Real)
    modes = result.modes[iq]
    M = sum(size(f, 2) for f in local_amplitudes)
    size(modes, 1) == 2M || throw(DimensionMismatch("Bloch spin-wave eigenvectors must have 2M Nambu components"))
    nmode = size(modes, 2)
    nmode == size(result.frequencies, 2) || throw(DimensionMismatch("Bloch mode columns and frequency count disagree"))
    offsets = cumsum(vcat(0, [size(f, 2) for f in local_amplitudes]))
    factor = hbar * _fourier_amplitude(normalization, length(sites))
    amplitude = zeros(ComplexF64, 3, nmode)
    for site in sites, α in 1:3, ν in 1:nmode, μ in 1:size(local_amplitudes[site], 2)
        p = offsets[site] + μ
        f = local_amplitudes[site][α, μ]
        amplitude[α, ν] += factor * (f * modes[p, ν] + conj(f) * modes[M+p, ν])
    end
    return amplitude
end

function _spinwave_bloch_same_grid(expected, supplied; tol::Real)
    length(expected) == length(supplied) || return false
    return all(eachindex(expected)) do i
        q1, q2 = expected[i], supplied[i]
        length(q1) == length(q2) && norm(q1 .- q2) <= tol * max(norm(q1), norm(q2), 1.0)
    end
end

"""Bloch-cell elastic basis form factor and one-magnon spin tensor for both spin-wave formalisms.

The `qgrid` must match the previously solved Bloch momenta exactly within numerical tolerance. Interactions use physical bond displacements in `cis(q⋅δ)`, so Bloch eigenvectors are in the position-phase gauge: **do not apply basis-site Fourier phases a second time to one-magnon amplitudes**. The elastic channel is a magnetic-cell basis factor; reciprocal-lattice Bragg delta functions are not constructed. This is a harmonic, single-cell-normalized response, not finite-sample absolute intensity.
"""
function spin_tensor_structure_factor(model::ManyBodyModel, result::SpinWaveDispersionResult, field::SpinTensorField, qgrid, axis;
                                      temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.x.normalization),
                                      broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                      weight_tol::Real=0.0, channels=(:elastic, :one_magnon), renormalize_transverse::Bool=false,
                                      spectral_accumulation::Symbol=:direct)
    result.model === model || throw(ArgumentError("Bloch scattering requires the unchanged source model used for the dispersion"))
    get(result.metadata, :momentum_scope, nothing) === :magnetic_supercell_bloch || throw(ArgumentError("expected magnetic-supercell Bloch modes"))
    temperature >= 0 && degeneracy_tol >= 0 && weight_tol >= 0 || throw(ArgumentError("temperature and tolerances must be nonnegative"))
    renormalize_transverse && throw(ArgumentError("Bloch scattering does not implement self-consistent ordered-moment renormalization"))
    spectral_accumulation in (:direct, :auto) || throw(ArgumentError("Bloch scattering requires direct spectral accumulation"))
    selected = Tuple(Symbol(ch) for ch in channels)
    !isempty(selected) && all(ch -> ch in (:elastic, :one_magnon), selected) || throw(ArgumentError("Bloch scattering supports only :elastic and :one_magnon"))
    any(haskey(meta, :zero_mode_regularization) for meta in result.metadata[:per_momentum_diagnostics]) && throw(ArgumentError(
        "regulated Bloch Goldstone modes cannot supply certified spectral weights"
    ))
    sites, positions, normalization = _lswt_field_geometry(model, field)
    convention.fourier_normalization == normalization || throw(ArgumentError("mismatched Fourier normalization"))
    q = [Float64.(collect(x)) for x in qgrid]
    _spinwave_bloch_same_grid(result.qpoints, q; tol=result.solver.tol) || throw(ArgumentError(
        "Bloch scattering Q grid must match spinwave_dispersion qpoints; q interpolation is not implemented"
    ))
    local_amplitudes, moments = _spinwave_bloch_local_amplitudes(model, result)
    all(i -> 1 <= i <= length(local_amplitudes), sites) || throw(BoundsError(local_amplitudes, sites))
    frequencies = result.frequencies
    nQ, nmodes = size(frequencies)
    x = _check_axis_grid(axis)
    output = zeros(ComplexF64, nQ, length(x), 3, 3)
    hbar = get(model.parameters, :hbar, 1.0)
    scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    minus = if temperature > 0 && :one_magnon in selected
        Solvers.spinwave_dispersion(result.solver, model, [-v for v in q])
    else
        nothing
    end
    if !isnothing(minus)
        any(haskey(meta, :zero_mode_regularization) for meta in minus.metadata[:per_momentum_diagnostics]) && throw(ArgumentError(
            "negative-Q thermal Bloch modes require unregularized BdG vectors"
        ))
    end
    occupations = zeros(Float64, nQ, nmodes)
    for iq in 1:nQ
        if :elastic in selected
            phases = zeros(ComplexF64, length(sites))
            _lswt_phase_vector!(phases, q[iq], positions, normalization)
            elastic = ComplexF64[hbar * sum(phases[a] * moments[α, sites[a]] for a in eachindex(sites)) for α in 1:3]
            for α in 1:3, β in 1:3
                weight = elastic[α] * conj(elastic[β])
                abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, 0.0, weight, broadening, α, β)
            end
        end
        :one_magnon in selected || continue
        ω = vec(frequencies[iq, :])
        n = _lswt_bose_occupations(ω, temperature, convention.kB; tol=result.solver.tol)
        occupations[iq, :] .= n
        emission = _spinwave_bloch_amplitudes(result, local_amplitudes, sites, iq, normalization, hbar)
        for ν in eachindex(ω), α in 1:3, β in 1:3
            weight = (n[ν] + 1) * emission[α, ν] * conj(emission[β, ν])
            abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, scale * ω[ν], weight, broadening, α, β)
        end
        if !isnothing(minus)
            reverse_ω = vec(minus.frequencies[iq, :])
            reverse_n = _lswt_bose_occupations(reverse_ω, temperature, convention.kB; tol=result.solver.tol)
            reverse_amplitude = conj.(_spinwave_bloch_amplitudes(minus, local_amplitudes, sites, iq, normalization, hbar))
            for ν in eachindex(reverse_ω), α in 1:3, β in 1:3
                weight = reverse_n[ν] * reverse_amplitude[α, ν] * conj(reverse_amplitude[β, ν])
                abs(weight) > weight_tol && _lswt_add_scalar_line!(output, iq, x, -scale * reverse_ω[ν], weight, broadening, α, β)
            end
        end
    end
    metadata = Dict{Symbol,Any}(
        :observable => :spin_tensor_dynamic_structure_factor,
        :response_kind => :correlation_spectrum,
        :frequency_domain => :real,
        :statistics => :bosonic,
        :momentum_scope => :magnetic_supercell_bloch,
        :momentum_convention => :inverse_cartesian_length,
        :bloch_phase_convention => :physical_displacement_position_gauge,
        :response_order => 2,
        :components => (:x, :y, :z),
        :formalism => result.solver.formalism,
        :temperature => float(temperature),
        :fourier_normalization => normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :solver => :spin_wave,
        :approximation => :gaussian_linear_spin_wave,
        :channels => selected,
        :field_sites => copy(sites),
        :elastic_channel => :basis_form_factor_without_bragg_delta,
        :one_magnon_channel => :periodic_bloch_modes,
        :ordered_moments => moments[:, sites],
        :depletion => :not_computed_from_finite_momentum_grid,
        :zero_mode_scattering => :reject_regularization,
        :two_magnon_channel => :not_implemented,
    )
    return TensorSpectrumResult(q, x, output, metadata)
end
