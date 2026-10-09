# Capability-based result introspection shared across solver, observable, scattering, Green-function, and thermodynamic result containers. Structural capabilities describe what data a result exposes, while response capabilities describe the physical semantics of dynamical objects without requiring downstream concrete-type switches.

const _RESULT_CAPABILITY_FIELDS = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :energies => (:energies,),
    :frequencies => (:frequencies,),
    :states => (:right_states, :states),
    :modes => (:modes,),
    :basis => (:basis,),
    :times => (:times,),
    :momentum => (:q, :qpoints),
    :spectral_axis => (:axis,),
    :intensity => (:intensity,),
    :values => (:values,),
    :metadata => (:metadata,),
    :model => (:model,),
    :uncertainty => (:uncertainty, :stderr, :covariance),
)

function _result_metadata_or_empty(result)
    hasproperty(result, :metadata) || return Dict{Symbol,Any}()
    metadata = getproperty(result, :metadata)
    metadata isa AbstractDict && return metadata
    metadata isa NamedTuple && return Dict{Symbol,Any}(pairs(metadata))
    return Dict{Symbol,Any}()
end

function _metadata_symbol(metadata, key::Symbol, default::Symbol=:unspecified)
    value = get(metadata, key, default)
    return value isa Symbol ? value : Symbol(string(value))
end

function _axis_frequency_domain(axis)
    axis isa Greens.FermionicMatsubaraAxis && return :matsubara
    axis isa Greens.BosonicMatsubaraAxis && return :matsubara
    values = try
        collect(axis)
    catch
        return :unspecified
    end
    isempty(values) && return :unspecified
    all(value -> value isa Real, values) && return :real
    return :unspecified
end

"""Return the semantic kind of a dynamical response or frequency-dependent object."""
function response_kind(result)
    metadata = _result_metadata_or_empty(result)
    haskey(metadata, :response_kind) && return _metadata_symbol(metadata, :response_kind)

    result isa Observables.SusceptibilityResult && return :susceptibility
    result isa Observables.CorrelationResult && return :time_correlation
    result isa Observables.ScatteringResult && return :scattering_intensity
    result isa Greens.GreenFunction && return :green_function
    result isa Greens.NonInteractingGreenFunction && return :green_function
    result isa Greens.SelfEnergy && return :self_energy
    result isa Greens.HybridizationFunction && return :hybridization_function
    result isa Greens.GeneralizedSusceptibility && return :generalized_susceptibility

    observable = get(metadata, :observable, nothing)
    observable in (:dynamical_correlation, :dynamic_structure_factor, :spin_tensor_dynamic_structure_factor,
                   :composite_dynamical_correlation, :spin_phonon_dynamical_correlation,
                   :correlation_spectrum_from_retarded_susceptibility) && return :correlation_spectrum
    observable === :spectral_density && return :spectral_density
    observable in (:cluster_perturbation_spectral_function, :local_electron_spectral_function, :spectral_function_from_retarded_green) && return :spectral_function
    observable === :commutator_spectral_density && return :commutator_spectral_density
    return :unspecified
end

"""Return `:real`, `:matsubara`, `:time`, `:static`, or `:unspecified` for a response-like result."""
function frequency_domain(result)
    metadata = _result_metadata_or_empty(result)
    haskey(metadata, :frequency_domain) && return _metadata_symbol(metadata, :frequency_domain)

    result isa Observables.CorrelationResult && return :time
    result isa Observables.StaticStructureFactorResult && return :static
    isdefined(Observables, :StaticTensorStructureFactorResult) && result isa Observables.StaticTensorStructureFactorResult && return :static
    result isa Greens.AbstractFrequencyMatrix && return _axis_frequency_domain(result.axis)
    result isa Greens.AbstractTwoParticleFrequencyObject && return :matsubara
    hasproperty(result, :axis) && return _axis_frequency_domain(getproperty(result, :axis))
    return :unspecified
end

"""Return the exchange statistics associated with a response, or `:unspecified` when the result does not declare them."""
function response_statistics(result)
    metadata = _result_metadata_or_empty(result)
    haskey(metadata, :statistics) && return _metadata_symbol(metadata, :statistics)

    result isa Greens.FrequencyMatrix && return :fermionic
    result isa Greens.AbstractTwoParticleFrequencyObject && return :bosonic
    if hasproperty(result, :axis)
        axis = getproperty(result, :axis)
        axis isa Greens.FermionicMatsubaraAxis && return :fermionic
        axis isa Greens.BosonicMatsubaraAxis && return :bosonic
    end
    response_kind(result) === :susceptibility && return :bosonic
    return :unspecified
end

"""Return `:local`, `:momentum_resolved`, or `:unspecified` for the spatial scope of a response."""
function momentum_scope(result)
    metadata = _result_metadata_or_empty(result)
    haskey(metadata, :momentum_scope) && return _metadata_symbol(metadata, :momentum_scope)

    result isa Observables.LocalSpectrumResult && return :local
    if hasproperty(result, :q)
        q = getproperty(result, :q)
        values = try
            collect(q)
        catch
            return :momentum_resolved
        end
        isempty(values) && return :unspecified
        all(isnothing, values) && return :local
        return :momentum_resolved
    end
    hasproperty(result, :qpoints) && return :momentum_resolved
    return :unspecified
end

"""Return stable operator/channel labels for a response when they are available, otherwise an empty tuple."""
function operator_channels(result)
    result isa Observables.CompositeSpectrumResult && return Tuple(result.channels)
    metadata = _result_metadata_or_empty(result)
    for key in (:operator_channels, :channels, :components)
        if haskey(metadata, key)
            value = metadata[key]
            return value isa Tuple ? value : Tuple(collect(value))
        end
    end
    haskey(metadata, :channel) && return (metadata[:channel],)
    hasproperty(result, :channel) && return (getproperty(result, :channel),)
    return ()
end

"""Return the physical operator-response order when declared or inferable, otherwise `0`. Two-point spectra and susceptibilities return `2`."""
function response_order(result)
    metadata = _result_metadata_or_empty(result)
    haskey(metadata, :response_order) && return Int(metadata[:response_order])
    kind = response_kind(result)
    kind === :generalized_susceptibility && return 4
    kind in (:correlation_spectrum, :spectral_density, :spectral_function, :commutator_spectral_density, :susceptibility, :time_correlation, :green_function) && return 2
    return 0
end

"""Return whether the object carries a dynamical/time/frequency response rather than a static quantity."""
function is_dynamical_response(result)
    domain = frequency_domain(result)
    kind = response_kind(result)
    return domain in (:real, :matsubara, :time) && kind !== :unspecified
end

"""Return whether a frequency-domain result requires analytic continuation before it can represent a real-frequency response."""
requires_analytic_continuation(result) = frequency_domain(result) === :matsubara

"""Return a compact dictionary of the common dynamical-response semantics inferred from result type, axis, and metadata."""
function response_convention(result)
    metadata = _result_metadata_or_empty(result)
    convention = Dict{Symbol,Any}(
        :response_kind => response_kind(result),
        :frequency_domain => frequency_domain(result),
        :statistics => response_statistics(result),
        :momentum_scope => momentum_scope(result),
        :operator_channels => operator_channels(result),
        :response_order => response_order(result),
    )
    for key in (:spectral_axis, :normalization, :definition, :operator_order, :correlator, :hbar, :kB, :temperature, :channel)
        haskey(metadata, key) && (convention[key] = metadata[key])
    end
    return convention
end

"""Return the semantic capabilities provided by a response-like result."""
function response_capabilities(result)
    capabilities = Set{Symbol}()
    is_dynamical_response(result) && push!(capabilities, :dynamical_response)
    response_kind(result) !== :unspecified && push!(capabilities, :response_kind)
    frequency_domain(result) !== :unspecified && push!(capabilities, :frequency_domain)
    response_statistics(result) !== :unspecified && push!(capabilities, :response_statistics)
    momentum_scope(result) !== :unspecified && push!(capabilities, :momentum_scope)
    !isempty(operator_channels(result)) && push!(capabilities, :operator_channels)
    response_order(result) > 0 && push!(capabilities, :response_order)
    requires_analytic_continuation(result) && push!(capabilities, :requires_analytic_continuation)
    return capabilities
end

function result_capabilities(result)
    capabilities = Set{Symbol}()
    for (capability, fields) in _RESULT_CAPABILITY_FIELDS
        any(field -> hasproperty(result, field), fields) && push!(capabilities, capability)
    end
    hasproperty(result, :model) && hasproperty(getproperty(result, :model), :provenance) && push!(capabilities, :provenance)
    union!(capabilities, response_capabilities(result))
    return capabilities
end

supports(result, capability::Symbol) = capability in result_capabilities(result)

function _first_result_property(result, fields::Tuple{Vararg{Symbol}}, capability::Symbol)
    for field in fields
        hasproperty(result, field) && return getproperty(result, field)
    end
    throw(ArgumentError("result does not support capability :$capability"))
end

result_metadata(result) = _first_result_property(result, (:metadata,), :metadata)
result_model(result) = _first_result_property(result, (:model,), :model)
result_energies(result) = _first_result_property(result, (:energies,), :energies)
result_frequencies(result) = _first_result_property(result, (:frequencies,), :frequencies)
result_modes(result) = _first_result_property(result, (:modes,), :modes)
result_basis(result) = _first_result_property(result, (:basis,), :basis)
result_times(result) = _first_result_property(result, (:times,), :times)
result_intensity(result) = _first_result_property(result, (:intensity,), :intensity)

function result_states(result)
    return _first_result_property(result, (:right_states, :states), :states)
end

function result_axes(result)
    axes = Dict{Symbol,Any}()
    hasproperty(result, :q) && (axes[:momentum] = getproperty(result, :q))
    hasproperty(result, :qpoints) && (axes[:momentum] = getproperty(result, :qpoints))
    hasproperty(result, :axis) && (axes[:spectral] = getproperty(result, :axis))
    hasproperty(result, :times) && (axes[:time] = getproperty(result, :times))
    return axes
end

function result_values(result)
    for field in (:intensity, :values, :energies, :frequencies, :states, :right_states, :modes)
        hasproperty(result, field) && return getproperty(result, field)
    end
    throw(ArgumentError("result exposes no standard value capability"))
end

function result_uncertainty(result)
    return _first_result_property(result, (:uncertainty, :stderr, :covariance), :uncertainty)
end

function result_provenance(result)
    model = result_model(result)
    hasproperty(model, :provenance) || throw(ArgumentError("result model carries no provenance"))
    return getproperty(model, :provenance)
end

"""Return the declared physical correlation scope carried by a result, or `:unspecified` when the producing method did not classify it."""
function correlation_scope(result)
    hasproperty(result, :metadata) || return :unspecified
    metadata = getproperty(result, :metadata)
    metadata isa AbstractDict || return :unspecified
    return get(metadata, :correlation_scope, :unspecified)
end
