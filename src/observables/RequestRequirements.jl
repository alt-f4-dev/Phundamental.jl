# Declarative physical requirements for typed high-level observable requests.
# These descriptors state what a request asks the forward pipeline to produce;
# they do not select a solver, authorize an approximation, or determine whether
# a Hamiltonian is physically admissible.

"""
    RequestRequirements

Machine-readable physical requirements carried by a high-level observable request. The descriptor is intentionally independent of solver implementation and result-container type.
"""
struct RequestRequirements
    request_kind::Symbol
    response_kind::Symbol
    frequency_domain::Symbol
    statistics::Symbol
    momentum_scope::Symbol
    response_order::Int
    operator_channels::Tuple
    temperature::Union{Nothing,Float64}
    metadata::Dict{Symbol,Any}
end

function RequestRequirements(; request_kind::Symbol, response_kind::Symbol=:unspecified, frequency_domain::Symbol=:unspecified,
                             statistics::Symbol=:unspecified, momentum_scope::Symbol=:unspecified, response_order::Integer=0,
                             operator_channels=(), temperature=nothing, metadata=Dict{Symbol,Any}())
    response_order >= 0 || throw(ArgumentError("response_order must be nonnegative"))
    T = isnothing(temperature) ? nothing : Float64(temperature)
    !isnothing(T) && T < 0 && throw(ArgumentError("temperature must be nonnegative"))
    channels = operator_channels isa Tuple ? operator_channels : Tuple(collect(operator_channels))
    return RequestRequirements(request_kind, response_kind, frequency_domain, statistics, momentum_scope, Int(response_order), channels, T,
                               Dict{Symbol,Any}(metadata))
end

function _request_metadata_dict(metadata)
    metadata isa AbstractDict && return Dict{Symbol,Any}(metadata)
    metadata isa NamedTuple && return Dict{Symbol,Any}(pairs(metadata))
    return Dict{Symbol,Any}()
end

function _request_metadata_symbol(metadata, key::Symbol, default::Symbol=:unspecified)
    value = get(metadata, key, default)
    return value isa Symbol ? value : Symbol(string(value))
end

function _request_metadata_channels(metadata)
    value = get(metadata, :operator_channels, ())
    value === nothing && return ()
    value isa Tuple && return value
    value isa AbstractVector && return Tuple(value)
    return (value,)
end

function _request_metadata_temperature(metadata)
    value = get(metadata, :temperature, nothing)
    return isnothing(value) ? nothing : Float64(value)
end

"""Return declarative physical requirements for a typed observable request."""
function request_requirements(request::AbstractObservable)
    return RequestRequirements(request_kind=Symbol(nameof(typeof(request))))
end

function request_requirements(request::ObservableRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=request.name,
        response_kind=_request_metadata_symbol(metadata, :response_kind),
        frequency_domain=_request_metadata_symbol(metadata, :frequency_domain),
        statistics=_request_metadata_symbol(metadata, :statistics),
        momentum_scope=_request_metadata_symbol(metadata, :momentum_scope),
        response_order=Int(get(metadata, :response_order, 0)),
        operator_channels=_request_metadata_channels(metadata),
        temperature=_request_metadata_temperature(metadata),
        metadata=metadata,
    )
end

function request_requirements(request::DynamicalCorrelationRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:dynamical_correlation,
        response_kind=:correlation_spectrum,
        frequency_domain=:real,
        statistics=_request_metadata_symbol(metadata, :statistics),
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=_request_metadata_channels(metadata),
        temperature=request.temperature,
        metadata=metadata,
    )
end

function request_requirements(request::CompositeDynamicalCorrelationRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:composite_dynamical_correlation,
        response_kind=:correlation_spectrum,
        frequency_domain=:real,
        statistics=_request_metadata_symbol(metadata, :statistics),
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=Tuple(keys(request.fields)),
        temperature=request.temperature,
        metadata=metadata,
    )
end

function request_requirements(request::SpinPhononCorrelationRequest)
    metadata = _request_metadata_dict(request.metadata)
    channels = (Tuple(request.spin_components)..., :phonon_displacement)
    return RequestRequirements(
        request_kind=:spin_phonon_dynamical_correlation,
        response_kind=:correlation_spectrum,
        frequency_domain=:real,
        statistics=:bosonic,
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=channels,
        temperature=request.temperature,
        metadata=metadata,
    )
end

function request_requirements(request::SpinTensorStructureFactorRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:spin_tensor_structure_factor,
        response_kind=:correlation_spectrum,
        frequency_domain=:real,
        statistics=:bosonic,
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=(:x, :y, :z),
        temperature=request.temperature,
        metadata=metadata,
    )
end

function request_requirements(request::LocalElectronSpectralFunctionRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:local_electron_spectral_function,
        response_kind=:spectral_function,
        frequency_domain=:real,
        statistics=:fermionic,
        momentum_scope=:local,
        response_order=2,
        operator_channels=(request.spin,),
        temperature=0.0,
        metadata=metadata,
    )
end

function request_requirements(request::PhononStructureFactorRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:phonon_structure_factor,
        response_kind=:correlation_spectrum,
        frequency_domain=:real,
        statistics=:bosonic,
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=(:displacement,),
        temperature=request.temperature,
        metadata=metadata,
    )
end

function request_requirements(request::LocalMatsubaraGreenFunctionRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:local_matsubara_green_function,
        response_kind=:green_function,
        frequency_domain=:matsubara,
        statistics=:fermionic,
        momentum_scope=:local,
        response_order=2,
        operator_channels=_request_metadata_channels(metadata),
        temperature=_request_metadata_temperature(metadata),
        metadata=metadata,
    )
end

function request_requirements(request::LocalMatsubaraSelfEnergyRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:local_matsubara_self_energy,
        response_kind=:self_energy,
        frequency_domain=:matsubara,
        statistics=:fermionic,
        momentum_scope=:local,
        response_order=2,
        operator_channels=_request_metadata_channels(metadata),
        temperature=_request_metadata_temperature(metadata),
        metadata=metadata,
    )
end

function request_requirements(request::MatsubaraSusceptibilityRequest)
    metadata = _request_metadata_dict(request.metadata)
    return RequestRequirements(
        request_kind=:matsubara_susceptibility,
        response_kind=:susceptibility,
        frequency_domain=:matsubara,
        statistics=:bosonic,
        momentum_scope=:momentum_resolved,
        response_order=2,
        operator_channels=(request.channel,),
        temperature=_request_metadata_temperature(metadata),
        metadata=metadata,
    )
end

"""Return whether a request explicitly asks for a time- or frequency-dependent response."""
is_dynamical_request(request::AbstractObservable) = request_requirements(request).frequency_domain in (:real, :matsubara, :time)
is_dynamical_request(requirements::RequestRequirements) = requirements.frequency_domain in (:real, :matsubara, :time)

function request_requirements(request::StaticStructureFactorRequest)
    return RequestRequirements(request_kind=:static_structure_factor, response_kind=:equal_time_correlation,
                               frequency_domain=:static, statistics=:bosonic, momentum_scope=:momentum_resolved,
                               response_order=2, temperature=request.temperature, metadata=copy(request.metadata))
end

function request_requirements(request::ThermodynamicCurveRequest)
    return RequestRequirements(request_kind=:thermodynamic_curve, response_kind=:thermodynamic_curve,
                               frequency_domain=:static, statistics=:canonical, momentum_scope=:bulk,
                               response_order=0, metadata=copy(request.metadata))
end

function request_requirements(request::CPTSpectralFunctionRequest)
    return RequestRequirements(request_kind=:cluster_perturbation_spectral_function, response_kind=:spectral_function,
                               frequency_domain=:real, statistics=:fermionic, momentum_scope=:momentum_resolved,
                               response_order=2, operator_channels=(request.spin,), temperature=0.0,
                               metadata=copy(request.metadata))
end

function request_requirements(request::HarmonicMultiphononSliceRequest)
    return RequestRequirements(request_kind=:harmonic_multiphonon_slice, response_kind=:integrated_correlation_spectrum,
                               frequency_domain=:integrated, statistics=:bosonic, momentum_scope=:momentum_resolved,
                               response_order=2, operator_channels=(:displacement,), temperature=request.temperature,
                               metadata=copy(request.metadata))
end
