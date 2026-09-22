# Lightweight numerical-closure diagnostics for representation-aware models.
# These checks inspect representation metadata without enumerating the basis.

"""
    NumericalCapabilities

Metadata describing whether a representation/Hamiltonian can enter the generic finite-basis matrix and eigensolver pipeline without changing its physical sector.
"""
struct NumericalCapabilities
    symbolic_hamiltonian::Bool
    operator_realization_supported::Bool
    finite_basis::Bool
    basis_constructible::Bool
    numerically_realizable::Bool
    requires_truncation::Bool
    unresolved_constraints::Bool
    transported_truncation::Bool
    coordinate_discretization_required::Bool
    biorthogonal::Bool
    hermiticity::Symbol
    compatible_solvers::Tuple{Vararg{Symbol}}
    reasons::Tuple{Vararg{String}}
end

"""
    TransformationCapabilities

Mathematical transformation metadata together with numerical-closure information for the actual transformed target model.
"""
struct TransformationCapabilities
    name::Symbol
    applicable::Bool
    exact::Bool
    invertible::Bool
    relation::RelationType
    requires_constraints::Bool
    gauge_redundant::Bool
    target_representation::Union{Nothing,Symbol}
    numerical::Union{Nothing,NumericalCapabilities}
    reasons::Tuple{Vararg{String}}
end

function _contains_biorthogonal_space(space::AbstractStateSpace)
    space isa BiorthogonalSpace && return true
    space isa CompositeStateSpace || return false
    return any(_contains_biorthogonal_space, space.factors)
end

function _contains_coordinate_algebra(algebra::AbstractOperatorAlgebra)
    algebra isa CoordinateMomentumAlgebra && return true
    algebra isa CompositeAlgebra || return false
    return any(_contains_coordinate_algebra, algebra.factors)
end

function _transported_truncation_metadata(truncation)
    isnothing(truncation) && return false
    if truncation isa NumericalTruncation
        value = truncation.cutoffs
        return value isa NamedTuple && haskey(value, :kind) && value.kind === :transported_subspace
    elseif truncation isa FactorTruncation
        return _transported_truncation_metadata(truncation.truncation)
    elseif truncation isa Tuple || truncation isa AbstractVector
        return any(_transported_truncation_metadata, truncation)
    end
    return false
end

function _truncation_known_dimension(truncation)
    isnothing(truncation) && return nothing
    if truncation isa NumericalTruncation
        return truncation.dimension
    elseif truncation isa FactorTruncation
        return truncation.truncation.dimension
    elseif truncation isa Tuple || truncation isa AbstractVector
        dims = [entry.truncation.dimension for entry in truncation if entry isa FactorTruncation]
        isempty(dims) && return nothing
        any(isnothing, dims) && return nothing
        return prod(Int(dim) for dim in dims)
    end
    return nothing
end

_numeric_operator_supported(::IdentityOperator) = true
_numeric_operator_supported(::SpinX) = true
_numeric_operator_supported(::SpinY) = true
_numeric_operator_supported(::SpinZ) = true
_numeric_operator_supported(::SpinPlus) = true
_numeric_operator_supported(::SpinMinus) = true
_numeric_operator_supported(::BosonAnnihilate) = true
_numeric_operator_supported(::BosonCreate) = true
_numeric_operator_supported(::FermionAnnihilate) = true
_numeric_operator_supported(::FermionCreate) = true
_numeric_operator_supported(::MajoranaOperator) = true
_numeric_operator_supported(::NumberOperator) = true
_numeric_operator_supported(::SqrtNumberFactor) = true
_numeric_operator_supported(::BosonicDisplacementFactor) = true
_numeric_operator_supported(::Union{PositionOperator,MomentumOperator}) = false
_numeric_operator_supported(op::FactorOperator) = _numeric_operator_supported(op.operator)
_numeric_operator_supported(op::ScaledOperator) = _numeric_operator_supported(op.operator)
_numeric_operator_supported(op::OperatorSum) = all(_numeric_operator_supported, op.terms)
_numeric_operator_supported(op::OperatorProduct) = all(_numeric_operator_supported, op.factors)
_numeric_operator_supported(::AbstractPrimitiveOperator) = false
_numeric_operator_supported(::AbstractOperatorExpr) = false
_numeric_operator_supported(::AbstractMatrix) = true
_numeric_operator_supported(::Any) = false

function _bounded_boson_modes(factor::FactorLayout, constraints)
    bounded = falses(factor.state_space.nmodes)
    for constraint in constraints
        kind = _constraint_kind(constraint)
        if kind === :occupation_bounds
            position = get(factor.positions, Int(constraint.mode), 0)
            position == 0 && continue
            bounded[position] = true
        elseif kind === :schwinger_occupancy
            for mode in constraint.modes
                position = get(factor.positions, Int(mode), 0)
                position == 0 || (bounded[position] = true)
            end
        end
    end
    return bounded
end

function _representation_basis_capabilities(rep::Representation)
    reasons = String[]
    constraints = _all_physical_constraints(rep)
    unresolved_constraints = false
    transported_truncation = _transported_truncation_metadata(rep.truncation)
    coordinate_required = _contains_coordinate_algebra(rep.algebra)
    factor_constructible = !coordinate_required
    factor_finite = !coordinate_required
    requires_truncation = false

    if coordinate_required
        push!(reasons, "coordinate/momentum degrees of freedom require CoordinateLadder or a dedicated discretization backend before generic matrix realization")
    end
    if transported_truncation
        push!(reasons, "the representation carries source-basis truncation metadata that must be replaced or cleared explicitly in the target basis")
    end

    factors = if coordinate_required
        FactorLayout[]
    else
        try
            _representation_factors(rep)
        catch error
            factor_constructible = false
            factor_finite = false
            push!(reasons, sprint(showerror, error))
            FactorLayout[]
        end
    end

    for constraint in constraints
        raw = constraint isa FactorConstraint ? constraint.constraint : constraint
        kind = _constraint_kind(raw)
        if kind === :transported_constraint || kind === :transported_projector || kind === :transported_physical_subspace
            unresolved_constraints = true
            via = raw isa NamedTuple && haskey(raw, :via) ? raw.via : :unknown
            push!(reasons, "exact physical-sector metadata transported through $via requires an explicit target-basis projector")
        elseif _constraint_family(raw) === :unknown
            unresolved_constraints = true
            push!(reasons, "unsupported physical constraint $(repr(raw))")
        end
    end

    for (factor_index, factor) in enumerate(factors)
        if factor.algebra isa FermionAlgebra && factor.algebra.nmodes > 64
            factor_constructible = false
            push!(reasons, "fermion factor $factor_index exceeds the current 64-mode computational-basis implementation limit")
        elseif factor.algebra isa MajoranaAlgebra && factor.algebra.ngenerators ÷ 2 > 64
            factor_constructible = false
            push!(reasons, "Majorana factor $factor_index exceeds the current 64-complex-mode computational-basis implementation limit")
        end

        factor.family === :boson || continue

        family_constraints = try
            _constraints_for_factor(constraints, factor.family, factor_index, factors)
        catch error
            unresolved_constraints = true
            factor_constructible = false
            push!(reasons, sprint(showerror, error))
            Any[]
        end

        truncation = _truncation_for_factor(rep.truncation, factor_index)
        if _transported_truncation_metadata(truncation)
            factor_constructible = false
            factor_finite = false
            requires_truncation = true
            push!(reasons, "boson factor $factor_index carries a source-basis truncation as provenance; select a target-basis cutoff with with_truncation")
            continue
        end

        if !isnothing(truncation)
            cutoffs = try
                _cutoffs_from_truncation(truncation, factor.state_space.nmodes)
            catch error
                factor_constructible = false
                push!(reasons, sprint(showerror, error))
                nothing
            end
            if !isnothing(cutoffs)
                continue
            end
        end

        bounded = _bounded_boson_modes(factor, family_constraints)
        if !all(bounded)
            factor_constructible = false
            factor_finite = false
            requires_truncation = true
            missing = [factor.ordering[i] for i in eachindex(bounded) if !bounded[i]]
            push!(reasons, "boson factor $factor_index has no finite occupation bound for modes $(missing)")
        end
    end

    exact_dimension = representationdimension(rep)
    truncation_dimension = _truncation_known_dimension(rep.truncation)
    finite_basis = !isnothing(exact_dimension) || !isnothing(truncation_dimension) || factor_finite
    basis_constructible = factor_constructible && !unresolved_constraints && !transported_truncation

    return (
        finite_basis=finite_basis,
        basis_constructible=basis_constructible,
        requires_truncation=requires_truncation,
        unresolved_constraints=unresolved_constraints,
        transported_truncation=transported_truncation,
        coordinate_discretization_required=coordinate_required,
        reasons=reasons,
    )
end

"""
    numerical_capabilities(model)
    numerical_capabilities(rep)

Inspect whether a representation can be converted to the finite numerical basis used by the generic matrix and eigensolver stack. The check is metadata-only and does not enumerate the basis.
"""
function numerical_capabilities(rep::Representation; operator=nothing)
    basis = _representation_basis_capabilities(rep)
    operator_supported = isnothing(operator) ? true : _numeric_operator_supported(operator)
    reasons = copy(basis.reasons)
    operator_supported || push!(reasons, "the Hamiltonian contains operator primitives without a finite matrix realization")

    biorthogonal = _contains_biorthogonal_space(rep.state_space)
    hermiticity = biorthogonal ? :biorthogonal_nonhermitian : :runtime_check
    numerically_realizable = basis.basis_constructible && operator_supported
    compatible = if !numerically_realizable
        ()
    elseif biorthogonal
        (:exact_diagonalization, :time_evolution)
    else
        (:exact_diagonalization, :lanczos, :time_evolution, :auto)
    end

    return NumericalCapabilities(
        operator isa AbstractOperatorExpr,
        operator_supported,
        basis.finite_basis,
        basis.basis_constructible,
        numerically_realizable,
        basis.requires_truncation,
        basis.unresolved_constraints,
        basis.transported_truncation,
        basis.coordinate_discretization_required,
        biorthogonal,
        hermiticity,
        compatible,
        Tuple(unique(reasons)),
    )
end

numerical_capabilities(model::ManyBodyModel) = numerical_capabilities(model.representation; operator=model.hamiltonian)

"""
    transformation_capabilities(T, model)

Report mathematical and numerical closure for applying transformation `T` to `model`. The transformation is evaluated symbolically when its source-domain checks pass so that target-basis and operator-realization requirements are assessed on the actual transformed model.
"""
function transformation_capabilities(T::AbstractTransformation, model::ManyBodyModel)
    cert = certificate(T)
    source_validation = Transformations.validate_source(T, model)
    source_ok = all(values(source_validation))
    if !source_ok
        reasons = Tuple("source validation failed: $(key)" for (key, value) in pairs(source_validation) if !value)
        return TransformationCapabilities(
            transformation_name(T), false, cert.exact, cert.invertible, cert.relation,
            cert.requires_constraints, cert.gauge_redundant, nothing, nothing, reasons,
        )
    end

    transformed = transform(T, model).model
    numerical = numerical_capabilities(transformed)
    return TransformationCapabilities(
        transformation_name(T), true, cert.exact, cert.invertible, cert.relation,
        cert.requires_constraints, cert.gauge_redundant, transformed.representation.name,
        numerical, numerical.reasons,
    )
end

function transformation_capabilities(T::CompositeTransformation, model::ManyBodyModel)
    cert = certificate(T)
    transformed = try
        transform(T, model).model
    catch error
        return TransformationCapabilities(
            transformation_name(T), false, cert.exact, cert.invertible, cert.relation,
            cert.requires_constraints, cert.gauge_redundant, nothing, nothing,
            (sprint(showerror, error),),
        )
    end
    numerical = numerical_capabilities(transformed)
    return TransformationCapabilities(
        transformation_name(T), true, cert.exact, cert.invertible, cert.relation,
        cert.requires_constraints, cert.gauge_redundant, transformed.representation.name,
        numerical, numerical.reasons,
    )
end
