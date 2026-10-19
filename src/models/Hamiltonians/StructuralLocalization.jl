# Locate independent external Hamiltonians in an already certified generated structural family.

"""Numerical parent coordinates and structural support of an external Hamiltonian, including failure diagnostics."""
struct StructuralLocation
    family::StructuralFamily
    selection::Union{Nothing,StructuralSelection}
    parent_coefficients::Vector{Float64}
    selected_coefficients::Vector{Float64}
    active_units::Vector{Symbol}
    residual::Float64
    relative_residual::Float64
    in_span::Bool
    valid_parameters::Bool
    reasons::Vector{String}
    identity_offset::Float64
    identity_projected::Bool
end

StructuralLocation(family::StructuralFamily, selection::Union{Nothing,StructuralSelection}, parent::Vector{Float64},
                   selected::Vector{Float64}, active::Vector{Symbol}, residual::Float64, relative::Float64,
                   in_span::Bool, valid::Bool, reasons::Vector{String}) =
    StructuralLocation(family, selection, parent, selected, active, residual, relative, in_span, valid, reasons, 0.0, false)

function _external_identity_projection(algebra::AbstractOperatorAlgebra, operator::AbstractOperatorExpr; tolerance::Real)
    entries = PhundamentalCore._numeric_linear_terms(algebra, operator)
    isnothing(entries) && throw(ArgumentError("identity projection requires numerically evaluable canonical coefficients"))
    removed = AbstractOperatorExpr[]
    offset = 0.0
    for (key, coefficient) in first(entries)
        key isa Tuple && !isempty(key) && first(key) === :identity || continue
        abs(imag(coefficient)) <= tolerance * max(1.0, abs(real(coefficient))) ||
            throw(ArgumentError("an imaginary identity coefficient cannot be discarded as a physical energy offset"))
        push!(removed, real(coefficient) * last(entries)[key])
        offset += real(coefficient)
    end
    return isempty(removed) ? operator : operator - reduce(+, removed), offset
end

"""
    locate_in_family(family, model_or_operator; tolerance=1e-10, zero_tolerance=tolerance, relative_zero_tolerance=0, project_identity=false)

Recover real invariant-basis coordinates from canonical operator coefficients. The strict default treats identity offsets as out-of-span unless included in the family. `project_identity=true` removes only a real scalar identity offset and records it separately; it is rejected for families whose basis already contains an identity direction. The optional relative active-unit threshold is applied in addition to the original absolute threshold. A zero nonidentity Hamiltonian has no nonempty structural selection.
"""
function locate_in_family(family::StructuralFamily, operator::AbstractOperatorExpr;
                          tolerance::Real=1e-10, zero_tolerance::Real=tolerance,
                          relative_zero_tolerance::Real=0.0, project_identity::Bool=false)
    isfinite(zero_tolerance) && zero_tolerance >= 0 || throw(ArgumentError("zero_tolerance must be finite and nonnegative"))
    isfinite(relative_zero_tolerance) && relative_zero_tolerance >= 0 ||
        throw(ArgumentError("relative_zero_tolerance must be finite and nonnegative"))
    basis = family.parent.basis.operators
    algebra = family.parent.specification.algebra
    offset = 0.0
    candidate = operator
    if project_identity
        has_identity_direction = any(basis) do direction
            terms = PhundamentalCore._numeric_linear_terms(algebra, direction)
            isnothing(terms) && throw(ArgumentError("identity projection requires numeric invariant-basis coefficients"))
            any(key -> key isa Tuple && !isempty(key) && first(key) === :identity, keys(first(terms)))
        end
        has_identity_direction && throw(ArgumentError("cannot project an identity offset from a family that already contains an identity direction"))
        candidate, offset = _external_identity_projection(algebra, operator; tolerance=tolerance)
    end
    solution = PhundamentalCore.structural_span_coordinates(basis, candidate, algebra; tolerance=tolerance)
    solution === missing && throw(ArgumentError("external Hamiltonian requires numerically evaluable canonical operator coefficients"))
    coefficients = solution.coefficients
    reasons = String[]
    !solution.within_span && push!(reasons, "external Hamiltonian lies outside the generated invariant span")
    threshold = max(zero_tolerance, relative_zero_tolerance * norm(coefficients))
    active = Symbol[unit.id for unit in family.units if any(i -> abs(coefficients[i]) > threshold, unit.basis_indices)]
    isempty(active) && push!(reasons, "zero Hamiltonian has no nonempty structural selection")
    parent_valid = true
    if !isnothing(family.parameter_space)
        report = parameter_constraint_report(family.parameter_space, coefficients)
        parent_valid = report.valid
        parent_valid || append!(reasons, parameter_constraint_reasons(report))
    end
    if !solution.within_span || isempty(active) || !parent_valid
        return StructuralLocation(family, nothing, coefficients, Float64[], active, solution.residual,
                                  solution.relative_residual, solution.within_span, parent_valid, reasons, offset, project_identity)
    end
    selection = structural_selection(family, active)
    values = Float64[coefficients[i] for i in selection.basis_indices]
    try
        if !isnothing(family.parameter_space)
            point = ParameterPoint(parameter_subspace(selection, values))
            embed_parameters(selection, point.values)
        end
    catch error
        if error isa ArgumentError || error isa DimensionMismatch
            push!(reasons, sprint(showerror, error))
            return StructuralLocation(family, nothing, coefficients, Float64[], active, solution.residual,
                                      solution.relative_residual, true, false, reasons, offset, project_identity)
        end
        rethrow()
    end
    return StructuralLocation(family, selection, coefficients, values, active, solution.residual,
                              solution.relative_residual, true, true, reasons, offset, project_identity)
end

function locate_in_family(family::StructuralFamily, model::ManyBodyModel; kwargs...)
    PhundamentalCore._same_model_algebra(family.parent.specification.algebra, model.representation.algebra) ||
        throw(ArgumentError("external model's operator algebra does not match the generated family"))
    model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("external model must contain a symbolic operator Hamiltonian"))
    return locate_in_family(family, model.hamiltonian; kwargs...)
end

"""A neutral diagnostic record; no serialized operator expression is trusted for reconstruction."""
function describe(location::StructuralLocation)
    return Dict{String,Any}(
        "schema_version" => STRUCTURAL_SCHEMA_VERSION,
        "kind" => "structural_location",
        "family_id" => structural_id(location.family),
        "selection_id" => isnothing(location.selection) ? nothing : structural_id(location.selection),
        "active_units" => String.(location.active_units),
        "parent_coefficients" => copy(location.parent_coefficients),
        "selected_coefficients" => copy(location.selected_coefficients),
        "residual" => location.residual,
        "relative_residual" => location.relative_residual,
        "in_span" => location.in_span,
        "valid_parameters" => location.valid_parameters,
        "identity_offset" => location.identity_offset,
        "identity_projected" => location.identity_projected,
        "reasons" => copy(location.reasons),
    )
end
