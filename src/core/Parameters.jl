# Constraint-bearing typed parameter schemas for H(theta) within a generated Hamiltonian basis. Existing dictionary-style model parameters remain accepted.

"""
    ParameterSpec(name, value; lower=-Inf, upper=Inf, unit=nothing, fixed=false, prior=nothing, metadata=Dict())

One real Hamiltonian coefficient together with inference metadata. `fixed=true` now has an enforced schema meaning: every `ParameterPoint` in the containing `ParameterSpace` must retain this specification's reference value.
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

"""Root type for exact equality constraints defining an admissible real parameter schema."""
abstract type AbstractParameterConstraint end

"""
    FixedParameterConstraint(name, value; atol=1e-10, rtol=1e-10, metadata=Dict())

Require one named parameter to remain equal to a specified real value within the declared numerical tolerance.
"""
struct FixedParameterConstraint{T<:Real,M} <: AbstractParameterConstraint
    parameter::Symbol
    value::T
    atol::Float64
    rtol::Float64
    metadata::M
end

function FixedParameterConstraint(parameter::Symbol, value::T; atol::Real=1e-10, rtol::Real=1e-10, metadata=Dict{Symbol,Any}()) where {T<:Real}
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    isfinite(value) || throw(ArgumentError("fixed parameter values must be finite"))
    return FixedParameterConstraint(parameter, value, Float64(atol), Float64(rtol), metadata)
end

"""
    EqualParameterConstraint(left, right; atol=1e-10, rtol=1e-10, metadata=Dict())

Require two named real parameters to be equal within the declared numerical tolerance.
"""
struct EqualParameterConstraint{M} <: AbstractParameterConstraint
    left::Symbol
    right::Symbol
    atol::Float64
    rtol::Float64
    metadata::M
end

function EqualParameterConstraint(left::Symbol, right::Symbol; atol::Real=1e-10, rtol::Real=1e-10, metadata=Dict{Symbol,Any}())
    left != right || throw(ArgumentError("equal-parameter constraint requires two distinct parameter names"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    return EqualParameterConstraint(left, right, Float64(atol), Float64(rtol), metadata)
end

"""
    LinearParameterConstraint(coefficients, rhs; name=:linear_parameter_constraint, atol=1e-10, rtol=1e-10, metadata=Dict())

Require a real affine equality `sum(coefficients[p] * theta[p]) = rhs`. Coefficients may be supplied as an `AbstractDict`, `NamedTuple`, or collection of `Pair`s. Unit conversion, when required, must already be represented in the supplied numerical coefficients; Foundation does not perform dimensional algebra on `ParameterSpec.unit` metadata.
"""
struct LinearParameterConstraint{M} <: AbstractParameterConstraint
    name::Symbol
    coefficients::Dict{Symbol,Float64}
    rhs::Float64
    atol::Float64
    rtol::Float64
    metadata::M
end

function LinearParameterConstraint(coefficients, rhs::Real; name::Symbol=:linear_parameter_constraint, atol::Real=1e-10, rtol::Real=1e-10,
                                   metadata=Dict{Symbol,Any}())
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    isfinite(rhs) || throw(ArgumentError("linear-constraint rhs must be finite"))
    stored = Dict{Symbol,Float64}()
    iterator = coefficients isa AbstractDict || coefficients isa NamedTuple ? pairs(coefficients) : coefficients
    for entry in iterator
        entry isa Pair || throw(ArgumentError("linear-constraint coefficients must be supplied as named pairs"))
        parameter = Symbol(first(entry))
        coefficient = Float64(last(entry))
        isfinite(coefficient) || throw(ArgumentError("linear-constraint coefficients must be finite"))
        stored[parameter] = get(stored, parameter, 0.0) + coefficient
    end
    filter!(pair -> !iszero(last(pair)), stored)
    isempty(stored) && throw(ArgumentError("LinearParameterConstraint requires at least one nonzero coefficient"))
    return LinearParameterConstraint(name, stored, Float64(rhs), Float64(atol), Float64(rtol), metadata)
end

"""One evaluated bound or parameter-equality condition."""
struct ParameterConstraintCheck
    name::Symbol
    satisfied::Bool
    residual::Float64
    reason::String
    metadata::Dict{Symbol,Any}
end

function ParameterConstraintCheck(name::Symbol, satisfied::Bool; residual::Real=0.0, reason::AbstractString="", metadata=Dict{Symbol,Any}())
    stored_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return ParameterConstraintCheck(name, satisfied, Float64(residual), String(reason), stored_metadata)
end

"""Inspectable result of evaluating one concrete parameter vector against a `ParameterSpace`."""
struct ParameterConstraintReport
    valid::Bool
    checks::Vector{ParameterConstraintCheck}
    metadata::Dict{Symbol,Any}
end

function ParameterConstraintReport(checks; metadata=Dict{Symbol,Any}())
    collected = ParameterConstraintCheck[checks...]
    stored_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return ParameterConstraintReport(all(check -> check.satisfied, collected), collected, stored_metadata)
end

"""Stable association between one named scalar coefficient and one basis direction of a `HamiltonianSpace`."""
struct ParameterBinding
    name::Symbol
    basis_index::Int
    basis_label::Symbol
    structural_signature::StructuralSignature
end

_constraint_name(constraint::FixedParameterConstraint) = Symbol("fixed_", constraint.parameter)
_constraint_name(constraint::EqualParameterConstraint) = Symbol("equal_", constraint.left, "_", constraint.right)
_constraint_name(constraint::LinearParameterConstraint) = constraint.name
_constraint_parameters(constraint::FixedParameterConstraint) = (constraint.parameter,)
_constraint_parameters(constraint::EqualParameterConstraint) = (constraint.left, constraint.right)
_constraint_parameters(constraint::LinearParameterConstraint) = Tuple(keys(constraint.coefficients))
_constraint_parameters(::AbstractParameterConstraint) = ()

function _implicit_parameter_constraints(specifications)
    constraints = AbstractParameterConstraint[]
    for specification in specifications
        if specification.fixed || specification.lower == specification.upper
            source = specification.fixed ? :fixed_flag : :degenerate_bound
            metadata = Dict{Symbol,Any}(:implicit => true, :source => source)
            push!(constraints, FixedParameterConstraint(specification.name, specification.value; metadata=metadata))
        end
    end
    return Tuple(constraints)
end

function _parameter_bindings(space::HamiltonianSpace, names)
    length(names) == length(space) || throw(DimensionMismatch("parameter names must have one entry per Hamiltonian basis operator"))
    bindings = ParameterBinding[]
    sizehint!(bindings, length(space))
    for i in eachindex(space.basis.operators)
        basis_structure = structure(space.basis.operators[i], space.specification.algebra)
        push!(bindings, ParameterBinding(Symbol(names[i]), Int(i), space.basis.labels[i], structural_signature(basis_structure)))
    end
    return Tuple(bindings)
end

"""
    ParameterSpace(specifications; hamiltonian_space=nothing, constraints=(), metadata=Dict())

Ordered real parameter schema associated with an optional `HamiltonianSpace`. Explicit constraints are stored independently of ordinary bounds. When a Hamiltonian space is attached, every parameter is deterministically bound by index to one basis direction and its canonical structural signature.
"""
struct ParameterSpace{S<:AbstractVector{<:ParameterSpec},H,C,B,M}
    specifications::S
    hamiltonian_space::H
    constraints::C
    bindings::B
    metadata::M
end

function ParameterSpace(specifications::AbstractVector{<:ParameterSpec}; hamiltonian_space=nothing, constraints=(), metadata=Dict{Symbol,Any}())
    specs = collect(specifications)
    isempty(specs) && throw(ArgumentError("ParameterSpace requires at least one parameter"))
    names = getfield.(specs, :name)
    length(unique(names)) == length(names) || throw(ArgumentError("parameter names must be unique"))
    if !isnothing(hamiltonian_space)
        hamiltonian_space isa HamiltonianSpace || throw(ArgumentError("hamiltonian_space must be HamiltonianSpace or nothing"))
        length(hamiltonian_space) == length(specs) || throw(DimensionMismatch("ParameterSpace dimension must match HamiltonianSpace"))
    end

    explicit_constraints = constraints isa AbstractParameterConstraint ? (constraints,) : Tuple(constraints)
    all(constraint -> constraint isa AbstractParameterConstraint, explicit_constraints) ||
        throw(ArgumentError("parameter constraints must subtype AbstractParameterConstraint"))
    known_names = Set(names)
    for constraint in explicit_constraints
        for parameter in _constraint_parameters(constraint)
            parameter in known_names || throw(ArgumentError("parameter constraint references unknown parameter :$(parameter)"))
        end
        if constraint isa EqualParameterConstraint
            left_index = findfirst(==(constraint.left), names)
            right_index = findfirst(==(constraint.right), names)
            left_unit = specs[left_index].unit
            right_unit = specs[right_index].unit
            if !isnothing(left_unit) && !isnothing(right_unit) && !isequal(left_unit, right_unit)
                throw(ArgumentError("equal-parameter constraint requires matching declared units for :$(constraint.left) and :$(constraint.right)"))
            end
        end
    end

    bindings = isnothing(hamiltonian_space) ? () : _parameter_bindings(hamiltonian_space, names)
    space = ParameterSpace{typeof(specs),typeof(hamiltonian_space),typeof(explicit_constraints),typeof(bindings),typeof(metadata)}(
        specs,
        hamiltonian_space,
        explicit_constraints,
        bindings,
        metadata,
    )
    defaults = getfield.(specs, :value)
    report = parameter_constraint_report(space, defaults)
    report.valid || throw(ArgumentError("initial parameter values violate the parameter schema: $(join(parameter_constraint_reasons(report), "; "))"))
    return space
end

# Backward-compatible positional constructor matching the pre-Batch-9 field layout.
ParameterSpace(specifications::AbstractVector{<:ParameterSpec}, hamiltonian_space, metadata) =
    ParameterSpace(specifications; hamiltonian_space=hamiltonian_space, metadata=metadata)

function ParameterSpace(space::HamiltonianSpace; values=zeros(Float64, length(space)), bounds=nothing, units=nothing,
                        fixed=nothing, priors=nothing, constraints=(), metadata=Dict{Symbol,Any}())
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
    return ParameterSpace(specs; hamiltonian_space=space, constraints=constraints, metadata=metadata)
end

Base.length(space::ParameterSpace) = length(space.specifications)
parameter_names(space::ParameterSpace) = getfield.(space.specifications, :name)
parameter_bounds(space::ParameterSpace) = [(spec.lower, spec.upper) for spec in space.specifications]
parameter_units(space::ParameterSpace) = getfield.(space.specifications, :unit)
free_parameter_indices(space::ParameterSpace) = findall(spec -> !spec.fixed, space.specifications)

"""Return the parameter constraints declared by `space`. By default the result also includes implicit fixed-value constraints induced by `ParameterSpec.fixed=true` or a degenerate bound `lower == upper`."""
function parameter_constraints(space::ParameterSpace; include_implicit::Bool=true)
    implicit = include_implicit ? _implicit_parameter_constraints(space.specifications) : ()
    return (implicit..., space.constraints...)
end

"""Return the stable structural bindings carried by a typed parameter schema."""
parameter_bindings(space::ParameterSpace) = space.bindings
parameter_bindings(space::HamiltonianSpace) = _parameter_bindings(space, space.basis.labels)

"""Return the structural binding associated with one named parameter."""
function parameter_binding(space::ParameterSpace, name::Symbol)
    index = findfirst(binding -> binding.name == name, space.bindings)
    isnothing(index) && throw(KeyError(name))
    return space.bindings[index]
end

function _parameter_index(space::ParameterSpace, name::Symbol)
    index = findfirst(==(name), parameter_names(space))
    isnothing(index) && throw(KeyError(name))
    return index
end

"""
    parameter_constraint_check(constraint, space, values)

Evaluate one named equality constraint at a concrete parameter vector and return a `ParameterConstraintCheck`.
"""
function parameter_constraint_check(constraint::FixedParameterConstraint, space::ParameterSpace, values::AbstractVector{<:Real})
    index = _parameter_index(space, constraint.parameter)
    value = values[index]
    residual = Float64(value - constraint.value)
    satisfied = isapprox(value, constraint.value; atol=constraint.atol, rtol=constraint.rtol)
    reason = satisfied ? "" : "parameter :$(constraint.parameter)=$(value) must equal $(constraint.value)"
    metadata = Dict{Symbol,Any}(:parameter => constraint.parameter, :target => constraint.value, :atol => constraint.atol, :rtol => constraint.rtol)
    return ParameterConstraintCheck(_constraint_name(constraint), satisfied; residual=residual, reason=reason, metadata=metadata)
end

function parameter_constraint_check(constraint::EqualParameterConstraint, space::ParameterSpace, values::AbstractVector{<:Real})
    left = values[_parameter_index(space, constraint.left)]
    right = values[_parameter_index(space, constraint.right)]
    residual = Float64(left - right)
    satisfied = isapprox(left, right; atol=constraint.atol, rtol=constraint.rtol)
    reason = satisfied ? "" : "parameters :$(constraint.left)=$(left) and :$(constraint.right)=$(right) must be equal"
    metadata = Dict{Symbol,Any}(:left => constraint.left, :right => constraint.right, :atol => constraint.atol, :rtol => constraint.rtol)
    return ParameterConstraintCheck(_constraint_name(constraint), satisfied; residual=residual, reason=reason, metadata=metadata)
end

function parameter_constraint_check(constraint::LinearParameterConstraint, space::ParameterSpace, values::AbstractVector{<:Real})
    lhs = 0.0
    for (name, coefficient) in constraint.coefficients
        lhs += coefficient * values[_parameter_index(space, name)]
    end
    residual = lhs - constraint.rhs
    satisfied = isapprox(lhs, constraint.rhs; atol=constraint.atol, rtol=constraint.rtol)
    reason = satisfied ? "" : "linear parameter constraint :$(constraint.name) has lhs=$(lhs), expected rhs=$(constraint.rhs)"
    metadata = Dict{Symbol,Any}(:lhs => lhs, :rhs => constraint.rhs, :coefficients => copy(constraint.coefficients), :atol => constraint.atol, :rtol => constraint.rtol)
    return ParameterConstraintCheck(constraint.name, satisfied; residual=residual, reason=reason, metadata=metadata)
end

parameter_constraint_check(constraint::AbstractParameterConstraint, ::ParameterSpace, ::AbstractVector{<:Real}) =
    throw(ArgumentError("no parameter-constraint evaluator is defined for $(typeof(constraint))"))

"""
    parameter_constraint_report(space, values)
    parameter_constraint_report(point)

Evaluate one concrete real coefficient vector against parameter bounds, fixed values, and all explicit equality constraints in the schema.
"""
function parameter_constraint_report(space::ParameterSpace, values::AbstractVector{<:Real})
    length(values) == length(space) || throw(DimensionMismatch("parameter vector length does not match ParameterSpace"))
    checks = ParameterConstraintCheck[]
    for (value, specification) in zip(values, space.specifications)
        satisfied = specification.lower <= value <= specification.upper
        violation = if satisfied
            0.0
        elseif value < specification.lower
            Float64(specification.lower - value)
        elseif value > specification.upper
            Float64(value - specification.upper)
        else
            Inf
        end
        reason = satisfied ? "" : "parameter :$(specification.name)=$(value) lies outside [$(specification.lower), $(specification.upper)]"
        metadata = Dict{Symbol,Any}(:parameter => specification.name, :value => value, :lower => specification.lower, :upper => specification.upper)
        push!(checks, ParameterConstraintCheck(Symbol("bounds_", specification.name), satisfied; residual=violation, reason=reason, metadata=metadata))
    end
    for constraint in parameter_constraints(space)
        push!(checks, parameter_constraint_check(constraint, space, values))
    end
    metadata = Dict{Symbol,Any}(:parameter_count => length(space), :constraint_count => length(parameter_constraints(space)), :bound_count => length(space))
    return ParameterConstraintReport(checks; metadata=metadata)
end

"""Return explanatory messages for violated bounds or parameter constraints."""
parameter_constraint_reasons(report::ParameterConstraintReport) = [check.reason for check in report.checks if !check.satisfied && !isempty(check.reason)]

"""Return true when a concrete vector satisfies the complete `ParameterSpace` schema."""
valid_parameter_point(space::ParameterSpace, values::AbstractVector{<:Real}) = parameter_constraint_report(space, values).valid

function _constraint_linear_row(constraint::FixedParameterConstraint, space::ParameterSpace)
    row = zeros(Float64, length(space))
    row[_parameter_index(space, constraint.parameter)] = 1.0
    return row, Float64(constraint.value)
end

function _constraint_linear_row(constraint::EqualParameterConstraint, space::ParameterSpace)
    row = zeros(Float64, length(space))
    row[_parameter_index(space, constraint.left)] = 1.0
    row[_parameter_index(space, constraint.right)] = -1.0
    return row, 0.0
end

function _constraint_linear_row(constraint::LinearParameterConstraint, space::ParameterSpace)
    row = zeros(Float64, length(space))
    for (name, coefficient) in constraint.coefficients
        row[_parameter_index(space, name)] += coefficient
    end
    return row, constraint.rhs
end

_constraint_linear_row(::AbstractParameterConstraint, ::ParameterSpace) = nothing

"""
    parameter_constraint_matrix(space)

Return `(A, b)` for the exact affine equality constraints `A * theta = b` implied by fixed parameters, degenerate bounds, and explicit parameter-schema constraints. Ordinary inequality bounds are not included.
"""
function parameter_constraint_matrix(space::ParameterSpace)
    constraints = parameter_constraints(space)
    A = zeros(Float64, length(constraints), length(space))
    b = zeros(Float64, length(constraints))
    for (row_index, constraint) in enumerate(constraints)
        row_data = _constraint_linear_row(constraint, space)
        isnothing(row_data) && throw(ArgumentError("constraint $(typeof(constraint)) has no linear equality representation"))
        row, rhs = row_data
        A[row_index, :] .= row
        b[row_index] = rhs
    end
    return A, b
end

"""Return the dimension of the affine equality-constrained parameter manifold. Inequality bounds do not reduce this count unless they are degenerate (`lower == upper`)."""
function parameter_dof(space::ParameterSpace; tolerance::Real=1e-10)
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative"))
    A, b = parameter_constraint_matrix(space)
    isempty(A) && return length(space)
    rank_A = rank(A; atol=tolerance, rtol=tolerance)
    rank_augmented = rank(hcat(A, b); atol=tolerance, rtol=tolerance)
    rank_augmented == rank_A || throw(ArgumentError("parameter equality constraints define an inconsistent affine system"))
    return length(space) - rank_A
end

"""A concrete point theta in a `ParameterSpace`. Construction validates the complete parameter schema."""
struct ParameterPoint{S<:ParameterSpace,T<:Real}
    space::S
    values::Vector{T}

    # Define the Vector constructor explicitly so Julia does not generate a
    # more-specific field constructor that bypasses parameter-schema checks.
    function ParameterPoint(space::S, values::Vector{T}) where {S<:ParameterSpace,T<:Real}
        length(values) == length(space) || throw(DimensionMismatch("parameter point dimension does not match ParameterSpace"))
        vals = float.(values)
        report = parameter_constraint_report(space, vals)
        report.valid || throw(ArgumentError("parameter point violates the parameter schema: $(join(parameter_constraint_reasons(report), "; "))"))
        V = eltype(vals)
        return new{S,V}(space, vals)
    end
end

ParameterPoint(space::ParameterSpace, values::AbstractVector{<:Real}) = ParameterPoint(space, collect(values))

ParameterPoint(space::ParameterSpace) = ParameterPoint(space, getfield.(space.specifications, :value))
parameter_vector(point::ParameterPoint) = copy(point.values)
parameter_names(point::ParameterPoint) = parameter_names(point.space)
parameter_constraint_report(point::ParameterPoint) = parameter_constraint_report(point.space, point.values)
valid_parameter_point(point::ParameterPoint) = parameter_constraint_report(point).valid
parameter_constraints(point::ParameterPoint; kwargs...) = parameter_constraints(point.space; kwargs...)
parameter_bindings(point::ParameterPoint) = parameter_bindings(point.space)
parameter_binding(point::ParameterPoint, name::Symbol) = parameter_binding(point.space, name)

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

function parameter_constraint_report(model::ManyBodyModel)
    schema = parameter_space(model)
    isnothing(schema) && throw(ArgumentError("model carries no typed ParameterSpace"))
    values = model.parameterization isa ParameterPoint ? model.parameterization.values : getfield.(schema.specifications, :value)
    return parameter_constraint_report(schema, values)
end

valid_parameter_point(model::ManyBodyModel) = parameter_constraint_report(model).valid
function parameter_constraints(model::ManyBodyModel; kwargs...)
    schema = parameter_space(model)
    isnothing(schema) && throw(ArgumentError("model carries no typed ParameterSpace"))
    return parameter_constraints(schema; kwargs...)
end
function parameter_bindings(model::ManyBodyModel)
    schema = parameter_space(model)
    isnothing(schema) && throw(ArgumentError("model carries no typed ParameterSpace"))
    return parameter_bindings(schema)
end
function parameter_binding(model::ManyBodyModel, name::Symbol)
    schema = parameter_space(model)
    isnothing(schema) && throw(ArgumentError("model carries no typed ParameterSpace"))
    return parameter_binding(schema, name)
end

function rebind_parameters(parameters, target_space::Union{Nothing,HamiltonianSpace})
    isnothing(target_space) && return parameters
    if parameters isa ParameterPoint
        source = parameters.space
        rebound = ParameterSpace(source.specifications; hamiltonian_space=target_space, constraints=source.constraints, metadata=source.metadata)
        return ParameterPoint(rebound, parameters.values)
    elseif parameters isa ParameterSpace
        return ParameterSpace(parameters.specifications; hamiltonian_space=target_space, constraints=parameters.constraints, metadata=parameters.metadata)
    end
    return parameters
end

"""
    with_parameters(model, point; rebuild=true)

Return a model at a new parameter point. If the model already carries a typed parameter schema, that schema is authoritative: caller-supplied values are rebound to it before the Hamiltonian is rebuilt. This prevents a foreign `ParameterPoint` from bypassing the model's declared bounds or coupling constraints.
"""
function with_parameters(model::ManyBodyModel, point::ParameterPoint; rebuild::Bool=true)
    hamiltonian = model.hamiltonian
    schema = parameter_space(model)
    rebound = if !isnothing(schema)
        parameter_names(point) == parameter_names(schema) || throw(ArgumentError("parameter point names do not match the model's parameter schema"))
        ParameterPoint(schema, point.values)
    elseif !isnothing(model.model_space)
        rebind_parameters(point, model.model_space)
    else
        point
    end
    if rebuild
        isnothing(model.model_space) && throw(ArgumentError("cannot rebuild H(theta): model has no HamiltonianSpace"))
        hamiltonian = hamiltonian_from_coefficients(model.model_space, rebound.values)
        _require_complete_admissibility(admissibility_report(model.model_space, hamiltonian), "reparameterized Hamiltonian")
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
