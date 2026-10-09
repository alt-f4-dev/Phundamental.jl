# Labeled multi-channel dynamical response for coupled and composite models.

function _composite_field_data(fields::NamedTuple)
    labels = keys(fields)
    isempty(labels) && throw(ArgumentError("composite dynamical response requires at least one channel"))
    channel_fields = values(fields)
    all(field -> field isa MomentumField, channel_fields) ||
        throw(ArgumentError("every composite dynamical-response channel must be a MomentumField"))
    normalization = first(channel_fields).normalization
    all(field -> field.normalization == normalization, channel_fields) ||
        throw(ArgumentError("all composite dynamical-response fields must use the same Fourier normalization"))
    return labels, channel_fields, normalization
end

function _composite_spectrum_metadata(model::ManyBodyModel, result, labels, normalization::Symbol, temperature::Real,
                                      convention::SpectrumConvention, broadening::AbstractBroadening)
    return Dict{Symbol,Any}(
        :observable => :composite_dynamical_correlation,
        :response_kind => :correlation_spectrum,
        :frequency_domain => :real,
        :statistics => :unspecified,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :operator_channels => Tuple(labels),
        :definition => :S_AB_matrix,
        :correlator => :A_q_t_B_minus_q_0,
        :operator_order => :row_channel_then_column_channel,
        :channels => labels,
        :temperature => float(temperature),
        :fourier_normalization => normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :representation => model.representation.name,
        :solver => get(result.metadata, :solver, :unknown),
    )
end

"""
    composite_dynamical_correlation(model, result, fields, qgrid, axis; ...)

Compute the complete ordered dynamical-correlation matrix for a labeled `NamedTuple` of `MomentumField`s. If the channels are `(A=field_a, B=field_b, ...)`, the returned `CompositeSpectrumResult.intensity` is indexed `(iq, ix, a, b)` and stores `S_{ab}(q,x)` for every ordered channel pair. No Hermitian, reciprocal, or detailed-balance relation is imposed between matrix entries.

All fields must use the same Fourier normalization. The result remains intrinsic: probe contraction and instrument resolution are separate stages.
"""
function composite_dynamical_correlation(model::ManyBodyModel, result::SolverResult, fields::NamedTuple, qgrid, axis;
                                         temperature::Real=0.0,
                                         convention::SpectrumConvention=SpectrumConvention(fourier_normalization=_composite_field_data(fields)[3]),
                                         broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                         weight_tol::Real=0.0)
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    labels, channel_fields, normalization = _composite_field_data(fields)
    convention.fourier_normalization == normalization ||
        throw(ArgumentError("SpectrumConvention Fourier normalization must match all composite response fields"))
    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("qgrid must not be empty"))
    x = _check_axis_grid(axis)
    nchannels = length(channel_fields)
    values = zeros(ComplexF64, length(qs), length(x), nchannels, nchannels)
    energies = _real_energies(result)
    probabilities = thermal_probabilities(result; temperature=temperature, kB=convention.kB, degeneracy_tol=degeneracy_tol)
    axis_scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))

    for (iq, q) in enumerate(qs)
        Aq = [eigenbasis_matrix(model, result, momentum_operator(model, field, q); hbar=convention.hbar) for field in channel_fields]
        Bm = [eigenbasis_matrix(model, result, momentum_operator(model, field, _negate_q(q)); hbar=convention.hbar) for field in channel_fields]

        @inbounds for m in eachindex(energies)
            pm = probabilities[m]
            pm <= weight_tol && continue
            for n in eachindex(energies)
                center = (energies[n] - energies[m]) * axis_scale
                kernel = _kernel(broadening, x .- center)
                for a in 1:nchannels, b in 1:nchannels
                    weight = pm * Aq[a][m, n] * Bm[b][n, m]
                    abs(weight) <= weight_tol && continue
                    @views values[iq, :, a, b] .+= weight .* kernel
                end
            end
        end
    end

    metadata = _composite_spectrum_metadata(model, result, labels, normalization, temperature, convention, broadening)
    return CompositeSpectrumResult(labels, qs, x, values, metadata)
end

"""
    composite_dynamical_correlation(model, result::QuadraticModeResult, fields, qgrid, axis; ...)

Compute the complete ordered channel matrix directly from a Gaussian bosonic mode solution. The same channel and ordering conventions as the generic solver-result method apply, including explicit handling of the selected bosonic zero-mode policy.
"""
function composite_dynamical_correlation(model::ManyBodyModel, result::QuadraticModeResult, fields::NamedTuple, qgrid, axis;
                                         temperature::Real=0.0,
                                         convention::SpectrumConvention=SpectrumConvention(fourier_normalization=_composite_field_data(fields)[3]),
                                         broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                         weight_tol::Real=0.0, zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes())
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    labels, channel_fields, normalization = _composite_field_data(fields)
    convention.fourier_normalization == normalization ||
        throw(ArgumentError("SpectrumConvention Fourier normalization must match all composite response fields"))
    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("qgrid must not be empty"))
    x = _check_axis_grid(axis)
    nchannels = length(channel_fields)
    values = zeros(ComplexF64, length(qs), length(x), nchannels, nchannels)
    observable_order = :affine_linear
    projected_zero_modes = Set{Int}()

    for (iq, q) in enumerate(qs)
        projected_a = [_quadratic_mode_projection(model, result, _momentum_boson_quadratic_coefficients(model, field, q)) for field in channel_fields]
        projected_b = [_quadratic_mode_projection(model, result, _momentum_boson_quadratic_coefficients(model, field, _negate_q(q))) for field in channel_fields]

        for a in 1:nchannels, b in 1:nchannels
            lines = _quadratic_mode_lehmann_lines(result, projected_a[a], projected_b[b]; temperature=temperature, convention=convention,
                                                  degeneracy_tol=degeneracy_tol, weight_tol=weight_tol, zero_mode_policy=zero_mode_policy)
            @views values[iq, :, a, b] .= broaden_lines(lines, x, broadening)
            lines.metadata[:observable_order] === :quadratic && (observable_order = :quadratic)
            union!(projected_zero_modes, lines.metadata[:projected_zero_modes])
        end
    end

    metadata = _composite_spectrum_metadata(model, result, labels, normalization, temperature, convention, broadening)
    metadata[:mode_native] = true
    metadata[:mode_space] = get(result.metadata, :mode_space, :unknown)
    metadata[:observable_class] = observable_order === :quadratic ? :gaussian_boson_quadratic : :linear_boson
    metadata[:observable_order] = observable_order
    metadata[:zero_mode_policy] = Symbol(nameof(typeof(zero_mode_policy)))
    metadata[:projected_zero_modes] = sort!(collect(projected_zero_modes))
    return CompositeSpectrumResult(labels, qs, x, values, metadata)
end

function _normalize_spin_components(components)
    selected = components isa Symbol ? (components,) : Tuple(Symbol(string(component)) for component in components)
    isempty(selected) && throw(ArgumentError("spin_components must contain at least one Cartesian component"))
    all(component -> component in (:x, :y, :z), selected) || throw(ArgumentError("spin_components may contain only :x, :y, and :z"))
    length(unique(selected)) == length(selected) || throw(ArgumentError("spin_components must not contain duplicates"))
    return selected
end

_spin_component_field(field::SpinTensorField, component::Symbol) = component === :x ? field.x : component === :y ? field.y : field.z

function _spin_phonon_fields(spin_field::SpinTensorField, phonon_field::MomentumField, spin_components)
    components = _normalize_spin_components(spin_components)
    labels = Symbol[Symbol("spin_", String(component)) for component in components]
    push!(labels, :phonon_displacement)
    fields = MomentumField[_spin_component_field(spin_field, component) for component in components]
    push!(fields, phonon_field)
    return NamedTuple{Tuple(labels)}(Tuple(fields))
end

"""
    spin_phonon_dynamical_correlation(model, result, spin_field, phonon_field, qgrid, axis; spin_components=(:x, :y, :z), ...)

Compute the coupled spin-phonon response as a labeled submatrix of the general composite dynamical response. The selected Cartesian spin fields are followed by the phonon-displacement field, so the result contains spin-spin, spin-phonon, phonon-spin, and phonon-phonon ordered correlations in one `CompositeSpectrumResult`.
"""
function spin_phonon_dynamical_correlation(model::ManyBodyModel, result, spin_field::SpinTensorField, phonon_field::MomentumField,
                                           qgrid, axis; spin_components=(:x, :y, :z), kwargs...)
    fields = _spin_phonon_fields(spin_field, phonon_field, spin_components)
    response = composite_dynamical_correlation(model, result, fields, qgrid, axis; kwargs...)
    metadata = copy(response.metadata)
    metadata[:observable] = :spin_phonon_dynamical_correlation
    metadata[:statistics] = :bosonic
    metadata[:response_family] = :spin_phonon
    metadata[:spin_components] = _normalize_spin_components(spin_components)
    return CompositeSpectrumResult(response.channels, response.q, response.axis, response.intensity, metadata)
end
