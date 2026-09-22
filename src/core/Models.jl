"""
    ManyBodyModel

Complete Hamiltonian model carried through the framework. The legacy fields
`representation`, `hamiltonian`, `parameters`, `observables`, and `provenance`
remain unchanged. `specification`, `model_space`, and `parameterization`
optionally connect a selected Hamiltonian to the generalized construction
`G -> H(G) -> H(theta)` without overloading the legacy parameter/metadata map.
"""
struct ManyBodyModel{R<:Representation,H,P,O,V,G,S,Q}
    representation::R
    hamiltonian::H
    parameters::P
    observables::O
    provenance::V
    specification::G
    model_space::S
    parameterization::Q
end

function ManyBodyModel(representation::Representation, hamiltonian; parameters=Dict{Symbol,Any}(), observables=Dict{Symbol,Any}(),
                       provenance=Any[], specification=nothing, model_space=nothing, parameterization=nothing)
    if !isnothing(model_space)
        model_space isa HamiltonianSpace || throw(ArgumentError("model_space must be HamiltonianSpace or nothing"))
        if isnothing(specification)
            specification = model_space.specification
        elseif specification !== model_space.specification && specification != model_space.specification
            throw(ArgumentError("specification must agree with model_space.specification"))
        end
    end
    !isnothing(specification) && !(specification isa GenerativeSpecification) &&
        throw(ArgumentError("specification must be GenerativeSpecification or nothing"))
    !isnothing(parameterization) && !(parameterization isa ParameterPoint || parameterization isa ParameterSpace) &&
        throw(ArgumentError("parameterization must be ParameterPoint, ParameterSpace, or nothing"))
    return ManyBodyModel(representation, hamiltonian, parameters, observables, collect(provenance), specification, model_space, parameterization)
end

# Backward-compatible positional constructor matching the pre-extension layout.
ManyBodyModel(representation::Representation, hamiltonian, parameters, observables, provenance) =
    ManyBodyModel(representation, hamiltonian; parameters=parameters, observables=observables, provenance=provenance)

"""Return a copy of a model with an additional provenance record."""
function with_provenance(model::ManyBodyModel, record)
    history = Any[model.provenance...]
    push!(history, record)
    return ManyBodyModel(model.representation, model.hamiltonian; parameters=model.parameters, observables=model.observables,
                         provenance=history, specification=model.specification, model_space=model.model_space,
                         parameterization=model.parameterization)
end

"""Return a copy of a model with a replacement representation and Hamiltonian."""
function remap_model(model::ManyBodyModel, representation::Representation, hamiltonian; observables=model.observables,
                     parameters=model.parameters, provenance=model.provenance, specification=model.specification,
                     model_space=model.model_space, parameterization=model.parameterization)
    return ManyBodyModel(representation, hamiltonian; parameters=parameters, observables=observables, provenance=provenance,
                         specification=specification, model_space=model_space, parameterization=parameterization)
end

"""
    with_truncation(model, truncation; record=true)

Return a copy of `model` whose representation carries `truncation`. This is a numerical-basis choice only: the Hamiltonian, exact physical constraints, model-space metadata, parameters, and observables are unchanged. Set `truncation=nothing` to clear a numerical truncation.
"""
function with_truncation(model::ManyBodyModel, truncation; record::Bool=true)
    representation = with_truncation(model.representation, truncation)
    provenance = Any[model.provenance...]
    if record
        push!(provenance, (
            numerical_truncation=isnothing(truncation) ? :cleared : :set,
            representation=representation.name,
            truncation=truncation,
        ))
    end
    return ManyBodyModel(representation, model.hamiltonian; parameters=model.parameters, observables=model.observables,
                         provenance=provenance, specification=model.specification, model_space=model.model_space,
                         parameterization=model.parameterization)
end

"""
    instantiate_model(space, representation, parameters; observables=Dict(), metadata_parameters=Dict(), provenance=Any[])

Instantiate one selected Hamiltonian from a generated `HamiltonianSpace`.
`parameters` may be a `ParameterPoint` or a real coefficient vector. The
legacy `model.parameters` map remains available for solver metadata and is
populated with the selected named coefficients plus `metadata_parameters`.
"""
function instantiate_model(space::HamiltonianSpace, representation::Representation, parameters;
                           observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}(), provenance=Any[])
    point = if parameters isa ParameterPoint
        parameters
    elseif parameters isa AbstractVector{<:Real}
        ParameterPoint(ParameterSpace(space; values=parameters), parameters)
    else
        throw(ArgumentError("parameters must be ParameterPoint or a real coefficient vector"))
    end
    length(point.values) == length(space) || throw(DimensionMismatch("parameter dimension does not match HamiltonianSpace"))
    point = rebind_parameters(point, space)
    hamiltonian = hamiltonian_from_coefficients(space, point.values)
    legacy = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata_parameters))
    for (name, value) in zip(parameter_names(point), point.values)
        legacy[name] = value
    end
    return ManyBodyModel(representation, hamiltonian; parameters=legacy, observables=observables, provenance=provenance,
                         specification=space.specification, model_space=space, parameterization=point)
end
