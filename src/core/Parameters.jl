# Typed parameter-space support for H(theta) within a generated Hamiltonian
# basis. Existing dictionary-style model parameters remain accepted.

"""
    ParameterSpec(name, value; lower=-Inf, upper=Inf, unit=nothing, fixed=false, prior=nothing, metadata=Dict())

One real Hamiltonian coefficient together with inference metadata.
"""
struct ParameterSpec{T<:Real,U,P,M}
    name::Symbol
    value::T
    lower::T
    upper::T
    unit::U
    fixed::Bool
    prior::P
    metadata::M
end

function ParameterSpec(name::Symbol, value::T; lower=-Inf, upper=Inf, unit=nothing, fixed::Bool=false,
                       prior=nothing, metadata=Dict{Symbol,Any}()) where {T<:Real}
    V = promote_type(typeof(float(value)), typeof(float(lower)), typeof(float(upper)))
    lo = V(lower)
    hi = V(upper)
    val = V(value)
    lo <= hi || throw(ArgumentError("parameter lower bound must not exceed upper bound"))
    lo <= val <= hi || throw(ArgumentError("initial parameter value lies outside its bounds"))
    return ParameterSpec{V,typeof(unit),typeof(prior),typeof(metadata)}(name, val, lo, hi, unit, fixed, prior, metadata)
end

"""Ordered parameter space associated with an optional generated Hamiltonian space."""
struct ParameterSpace{S<:AbstractVector{<:ParameterSpec},H,M}
    specifications::S
    hamiltonian_space::H
    metadata::M
end

function ParameterSpace(specifications::AbstractVector{<:ParameterSpec}; hamiltonian_space=nothing, metadata=Dict{Symbol,Any}())
    specs = collect(specifications)
    isempty(specs) && throw(ArgumentError("ParameterSpace requires at least one parameter"))
    names = getfield.(specs, :name)
    length(unique(names)) == length(names) || throw(ArgumentError("parameter names must be unique"))
    if !isnothing(hamiltonian_space)
        hamiltonian_space isa HamiltonianSpace || throw(ArgumentError("hamiltonian_space must be HamiltonianSpace or nothing"))
        length(hamiltonian_space) == length(specs) || throw(DimensionMismatch("ParameterSpace dimension must match HamiltonianSpace"))
    end
    return ParameterSpace(specs, hamiltonian_space, metadata)
end

function ParameterSpace(space::HamiltonianSpace; values=zeros(Float64, length(space)), bounds=nothing, units=nothing,
                        fixed=nothing, priors=nothing, metadata=Dict{Symbol,Any}())
    length(values) == length(space) || throw(DimensionMismatch("values must have one entry per Hamiltonian basis operator"))
    labels = space.basis.labels
    bounds_vec = isnothing(bounds) ? [(-Inf, Inf) for _ in labels] : collect(bounds)
    units_vec = isnothing(units) ? fill(nothing, length(labels)) : collect(units)
    fixed_vec = isnothing(fixed) ? fill(false, length(labels)) : Bool.(collect(fixed))
    priors_vec = isnothing(priors) ? fill(nothing, length(labels)) : collect(priors)
    length(bounds_vec) == length(labels) || throw(DimensionMismatch("bounds must have one entry per parameter"))
    length(units_vec) == length(labels) || throw(DimensionMismatch("units must have one entry per parameter"))
    length(fixed_vec) == length(labels) || throw(DimensionMismatch("fixed must have one entry per parameter"))
    length(priors_vec) == length(labels) || throw(DimensionMismatch("priors must have one entry per parameter"))
    specs = ParameterSpec[]
    for i in eachindex(labels)
        lo, hi = bounds_vec[i]
        push!(specs, ParameterSpec(labels[i], values[i]; lower=lo, upper=hi, unit=units_vec[i], fixed=fixed_vec[i], prior=priors_vec[i]))
    end
    return ParameterSpace(specs; hamiltonian_space=space, metadata=metadata)
end

Base.length(space::ParameterSpace) = length(space.specifications)
parameter_names(space::ParameterSpace) = getfield.(space.specifications, :name)
parameter_bounds(space::ParameterSpace) = [(spec.lower, spec.upper) for spec in space.specifications]
parameter_units(space::ParameterSpace) = getfield.(space.specifications, :unit)
free_parameter_indices(space::ParameterSpace) = findall(spec -> !spec.fixed, space.specifications)

"""A concrete point theta in a `ParameterSpace`."""
struct ParameterPoint{S<:ParameterSpace,T<:Real}
    space::S
    values::Vector{T}
end

function ParameterPoint(space::ParameterSpace, values::AbstractVector{<:Real})
    length(values) == length(space) || throw(DimensionMismatch("parameter point dimension does not match ParameterSpace"))
    vals = float.(collect(values))
    for (value, spec) in zip(vals, space.specifications)
        spec.lower <= value <= spec.upper || throw(ArgumentError("parameter $(spec.name)=$value lies outside [$(spec.lower), $(spec.upper)]"))
    end
    return ParameterPoint(space, vals)
end

ParameterPoint(space::ParameterSpace) = ParameterPoint(space, getfield.(space.specifications, :value))
parameter_vector(point::ParameterPoint) = copy(point.values)
parameter_names(point::ParameterPoint) = parameter_names(point.space)

function Base.getindex(point::ParameterPoint, name::Symbol)
    idx = findfirst(==(name), parameter_names(point.space))
    isnothing(idx) && throw(KeyError(name))
    return point.values[idx]
end

function parameter_point(space::ParameterSpace; kwargs...)
    values = getfield.(space.specifications, :value)
    names = parameter_names(space)
    for (name, value) in pairs(kwargs)
        idx = findfirst(==(name), names)
        isnothing(idx) && throw(KeyError(name))
        values[idx] = value
    end
    return ParameterPoint(space, values)
end

parameter_space(model::ManyBodyModel) = model.parameterization isa ParameterPoint ? model.parameterization.space :
    model.parameterization isa ParameterSpace ? model.parameterization : nothing

parameter_vector(model::ManyBodyModel) = model.parameterization isa ParameterPoint ? parameter_vector(model.parameterization) :
    throw(ArgumentError("model carries no typed ParameterPoint"))

parameter_names(model::ManyBodyModel) = model.parameterization isa ParameterPoint ? parameter_names(model.parameterization) :
    model.parameterization isa ParameterSpace ? parameter_names(model.parameterization) : collect(keys(model.parameters))

function rebind_parameters(parameters, target_space::Union{Nothing,HamiltonianSpace})
    isnothing(target_space) && return parameters
    if parameters isa ParameterPoint
        source = parameters.space
        rebound = ParameterSpace(source.specifications; hamiltonian_space=target_space, metadata=source.metadata)
        return ParameterPoint(rebound, parameters.values)
    elseif parameters isa ParameterSpace
        return ParameterSpace(parameters.specifications; hamiltonian_space=target_space, metadata=parameters.metadata)
    end
    return parameters
end

"""
    with_parameters(model, point; rebuild=true)

Return a model at a new parameter point. When the model carries a
`HamiltonianSpace`, `rebuild=true` reconstructs `H(theta)` in that basis.
"""
function with_parameters(model::ManyBodyModel, point::ParameterPoint; rebuild::Bool=true)
    hamiltonian = model.hamiltonian
    rebound = point
    if !isnothing(model.model_space) && point.space.hamiltonian_space !== model.model_space
        rebound = rebind_parameters(point, model.model_space)
    end
    if rebuild
        isnothing(model.model_space) && throw(ArgumentError("cannot rebuild H(theta): model has no HamiltonianSpace"))
        hamiltonian = hamiltonian_from_coefficients(model.model_space, rebound.values)
    end
    legacy = model.parameters isa AbstractDict ? Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(model.parameters)) : model.parameters
    if legacy isa AbstractDict
        for (name, value) in zip(parameter_names(rebound), rebound.values)
            legacy[name] = value
        end
    end
    return ManyBodyModel(model.representation, hamiltonian; parameters=legacy, observables=model.observables,
                         provenance=model.provenance, specification=model.specification, model_space=model.model_space,
                         parameterization=rebound)
end
