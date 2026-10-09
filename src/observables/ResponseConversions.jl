# Explicit conversions between causal real-frequency response functions and physical spectral observables.

function _retarded_green_normalization(normalization::Symbol)
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) ||
        throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
    return normalization
end

function _retarded_green_intensity(values, normalization::Symbol)
    _retarded_green_normalization(normalization)
    if normalization === :senechal
        return -2 .* imag.(values)
    end
    return -imag.(values) ./ pi
end

function _response_metadata_copy(metadata)
    return metadata isa AbstractDict ? Dict{Symbol,Any}(metadata) : Dict{Symbol,Any}(pairs(metadata))
end

"""
    spectral_function_from_retarded_green(axis, values; ...)
    spectral_function_from_retarded_green(qgrid, axis, values; ...)

Convert a scalar retarded Green function into a real-frequency spectral function. The local form accepts one complex value per spectral point. The momentum-resolved form accepts a matrix indexed `(iq, ix)`.

For `normalization=:spectral_density` or `:minus_im_over_pi`, the conversion is `A = -Im(G^R)/pi`. `normalization=:senechal` returns the `-2 Im(G^R)` convention used by the CPT benchmark API. This operation requires retarded real-frequency data; Matsubara data must be analytically continued before calling this function.
"""
function spectral_function_from_retarded_green(axis, values::AbstractVector{<:Number};
                                               convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                               normalization::Symbol=:spectral_density, metadata=Dict{Symbol,Any}())
    x = _check_axis_grid(axis)
    length(values) == length(x) || throw(DimensionMismatch("retarded Green-function value count must match the spectral axis"))
    intensity = reshape(Float64.(_retarded_green_intensity(values, normalization)), 1, length(x))
    response_metadata = _response_metadata_copy(metadata)
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :spectral_function_from_retarded_green,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :local,
        :response_order => 2,
        :source_response => :retarded_green_function,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => normalization,
        :resolution_applied => false,
    ))
    return SpectrumResult([nothing], x, intensity, response_metadata)
end

function spectral_function_from_retarded_green(qgrid, axis, values::AbstractMatrix{<:Number};
                                               convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                               normalization::Symbol=:spectral_density, metadata=Dict{Symbol,Any}())
    q = collect(qgrid)
    x = _check_axis_grid(axis)
    size(values) == (length(q), length(x)) || throw(DimensionMismatch("retarded Green-function values must have shape (nq, nfrequency)"))
    intensity = Float64.(_retarded_green_intensity(values, normalization))
    response_metadata = _response_metadata_copy(metadata)
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :spectral_function_from_retarded_green,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :source_response => :retarded_green_function,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => normalization,
        :resolution_applied => false,
    ))
    return SpectrumResult(q, x, intensity, response_metadata)
end

function _real_susceptibility_axis(result::SusceptibilityResult)
    raw_axis = collect(result.axis)
    isempty(raw_axis) && throw(ArgumentError("susceptibility axis must not be empty"))
    all(value -> value isa Real, raw_axis) || throw(ArgumentError(
        "conversion from susceptibility to a real-frequency spectrum requires a real axis; Matsubara susceptibility data require explicit analytic continuation first"
    ))
    return _check_axis_grid(raw_axis)
end

function _hermitian_response_channel(result::SusceptibilityResult, hermitian_channel)
    metadata = result.metadata isa AbstractDict ? result.metadata : Dict{Symbol,Any}(pairs(result.metadata))
    declared = get(metadata, :hermitian_channel, nothing)
    flag = isnothing(hermitian_channel) ? declared : hermitian_channel
    flag === true || throw(ArgumentError(
        "scalar -Im(chi^R) conversion is valid here only for a Hermitian response channel; set metadata[:hermitian_channel]=true or pass hermitian_channel=true when that condition is known to hold"
    ))
    return true
end

function _commutator_axis_prefactor(convention::SpectrumConvention)
    return convention.spectral_axis === :energy ? 1.0 : Float64(convention.hbar)
end

"""
    commutator_spectral_density(result::SusceptibilityResult; convention=..., hermitian_channel=nothing)

Convert a scalar Hermitian retarded susceptibility into its commutator spectral density. With the package Fourier conventions, `rho_AB(x) = -s Im(chi^R_AB(x))/pi`, where `s=1` on an energy-transfer axis and `s=hbar` on an angular-frequency axis.

The input axis must already be real. The function intentionally rejects Matsubara susceptibilities because analytic continuation is a separate, method-dependent operation.
"""
function commutator_spectral_density(result::SusceptibilityResult;
                                     convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                     hermitian_channel=nothing, metadata=Dict{Symbol,Any}())
    x = _real_susceptibility_axis(result)
    _hermitian_response_channel(result, hermitian_channel)
    size(result.values) == (length(result.q), length(x)) || throw(DimensionMismatch("susceptibility values must have shape (nq, nfrequency)"))
    prefactor = _commutator_axis_prefactor(convention)
    values = -prefactor .* imag.(result.values) ./ pi
    response_metadata = _response_metadata_copy(result.metadata)
    merge!(response_metadata, _response_metadata_copy(metadata))
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :commutator_spectral_density,
        :response_kind => :commutator_spectral_density,
        :frequency_domain => :real,
        :statistics => :bosonic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :source_response => :retarded_susceptibility,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => :unit_integral_delta,
        :resolution_applied => false,
    ))
    return SpectrumResult(copy(result.q), x, Float64.(values), response_metadata)
end

function _fdt_energy(x::Real, convention::SpectrumConvention)
    return convention.spectral_axis === :energy ? Float64(x) : Float64(convention.hbar) * Float64(x)
end

function _fdt_denominator(energy::Float64, temperature::Float64, kB::Float64)
    if iszero(temperature)
        energy > 0 && return 1.0
        energy < 0 && return -Inf
        return 0.0
    end
    beta = inv(kB * temperature)
    return -expm1(-beta * energy)
end

function _resolve_zero_frequency_value(zero_frequency_value, q, x, chi_value)
    isnothing(zero_frequency_value) && throw(ArgumentError(
        "the fluctuation-dissipation factor is singular at zero transfer; supply zero_frequency_value explicitly or evaluate the zero-frequency limit with a continuation/interpolation method appropriate to the response"
    ))
    return zero_frequency_value isa Function ? zero_frequency_value(q, x, chi_value) : zero_frequency_value
end

"""
    correlation_spectrum_from_retarded_susceptibility(result; temperature, convention=..., hermitian_channel=nothing, zero_frequency_value=nothing)

Convert a scalar Hermitian retarded susceptibility into the ordered equilibrium correlation spectrum `S_AB` using the fluctuation-dissipation relation. In the package normalization,

`S_AB(x) = rho_AB(x) / (1 - exp(-beta*x_E))`,

where `x_E=x` for an energy axis and `x_E=hbar*x` for an angular-frequency axis. `rho_AB` is the commutator spectral density returned by `commutator_spectral_density`.

The input must already be a real-frequency retarded susceptibility. Analytic continuation from Matsubara frequencies is deliberately not performed implicitly. At exactly zero transfer the Bose factor requires a limiting procedure; provide `zero_frequency_value` explicitly when the requested grid contains zero.
"""
function correlation_spectrum_from_retarded_susceptibility(result::SusceptibilityResult; temperature::Real,
                                                            convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                                            hermitian_channel=nothing, zero_frequency_value=nothing,
                                                            zero_frequency_tol::Real=0.0, metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    zero_frequency_tol >= 0 || throw(ArgumentError("zero_frequency_tol must be nonnegative"))
    rho = commutator_spectral_density(result; convention=convention, hermitian_channel=hermitian_channel)
    values = Matrix{Float64}(undef, size(rho.intensity))
    temperature_value = Float64(temperature)
    kB = Float64(convention.kB)

    for iq in axes(values, 1), ix in axes(values, 2)
        x = Float64(rho.axis[ix])
        energy = _fdt_energy(x, convention)
        if abs(energy) <= zero_frequency_tol
            values[iq, ix] = Float64(real(_resolve_zero_frequency_value(zero_frequency_value, rho.q[iq], x, result.values[iq, ix])))
            continue
        end
        denominator = _fdt_denominator(energy, temperature_value, kB)
        values[iq, ix] = isinf(denominator) ? 0.0 : rho.intensity[iq, ix] / denominator
    end

    response_metadata = copy(rho.metadata)
    merge!(response_metadata, _response_metadata_copy(metadata))
    response_metadata[:observable] = :correlation_spectrum_from_retarded_susceptibility
    response_metadata[:response_kind] = :correlation_spectrum
    response_metadata[:source_response] = :retarded_susceptibility
    response_metadata[:fluctuation_dissipation_applied] = true
    response_metadata[:temperature] = temperature_value
    response_metadata[:zero_frequency_policy] = isnothing(zero_frequency_value) ? :error : :explicit_value
    return SpectrumResult(copy(rho.q), copy(rho.axis), values, response_metadata)
end
