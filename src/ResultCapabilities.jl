# Capability-based result introspection shared across solver, observable,
# scattering, and thermodynamic result containers. The protocol intentionally
# uses field capabilities rather than concrete result types so new backends can
# participate without editing a central union type.

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

function result_capabilities(result)
    capabilities = Set{Symbol}()
    for (capability, fields) in _RESULT_CAPABILITY_FIELDS
        any(field -> hasproperty(result, field), fields) && push!(capabilities, capability)
    end
    hasproperty(result, :model) && hasproperty(getproperty(result, :model), :provenance) && push!(capabilities, :provenance)
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
