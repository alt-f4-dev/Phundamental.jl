# General Hamiltonian model-space objects implementing the manuscript-level generative specification G -> H(G). The actual invariant-basis generator remains pluggable: this layer defines the physical admissibility contract without claiming universal crystallographic completeness for every operator algebra.

abstract type AbstractAdmissibilityCondition end

"""Root type for physical Hamiltonian model-space containers. Model-space admissibility is independent of numerical solver compatibility."""
abstract type AbstractModelSpace end

"""
    AdmissibilityCheck(name, passed; reason="", metadata=Dict())

One inspectable physical-admissibility check. `passed` is `true` for an accepted criterion, `false` for a rejected criterion, and `missing` when a declared criterion cannot be evaluated from the supplied model-space information.
"""
struct AdmissibilityCheck
    name::Symbol
    passed::Union{Bool,Missing}
    reason::String
    metadata::Dict{Symbol,Any}
end

function AdmissibilityCheck(
    name::Symbol,
    passed::Union{Bool,Missing};
    reason::AbstractString="",
    metadata=Dict{Symbol,Any}(),
)
    stored_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return AdmissibilityCheck(name, passed, String(reason), stored_metadata)
end

"""
    AdmissibilityReport(checks; metadata=Dict())

Structured physical-admissibility result. `complete` is true only when every declared criterion was evaluated, and `admissible` is true only when the report is complete and every check passed.
"""
struct AdmissibilityReport
    admissible::Bool
    complete::Bool
    checks::Vector{AdmissibilityCheck}
    metadata::Dict{Symbol,Any}
end

function AdmissibilityReport(checks; metadata=Dict{Symbol,Any}())
    collected = AdmissibilityCheck[checks...]
    complete = all(check -> !ismissing(check.passed), collected)
    accepted = complete && all(check -> check.passed === true, collected)
    stored_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return AdmissibilityReport(accepted, complete, collected, stored_metadata)
end

admissible(report::AdmissibilityReport) = report.admissible

"""Return the explanatory messages for failed or unevaluated checks in an `AdmissibilityReport`."""
admissibility_reasons(report::AdmissibilityReport) = [check.reason for check in report.checks if check.passed !== true && !isempty(check.reason)]

"""
    PredicateAdmissibility(name, predicate; description="")

Named admissibility condition evaluated on a candidate Hamiltonian operator. The predicate must return `Bool` when called on the candidate.
"""
struct PredicateAdmissibility{F} <: AbstractAdmissibilityCondition
    name::Symbol
    predicate::F
    description::String
end

PredicateAdmissibility(name::Symbol, predicate; description::AbstractString="") = PredicateAdmissibility(name, predicate, String(description))

admissible(condition::PredicateAdmissibility, candidate) = Bool(condition.predicate(candidate))
admissible(condition::AbstractAdmissibilityCondition, candidate) = throw(ArgumentError("no admissibility evaluator is defined for $(typeof(condition))"))

"""
    admissibility_check(condition, candidate)

Evaluate one `AbstractAdmissibilityCondition` and return an inspectable `AdmissibilityCheck`. Custom condition types may specialize this method when they need richer diagnostic metadata; implementing only `admissible(condition, candidate)` remains supported.
"""
function admissibility_check(condition::AbstractAdmissibilityCondition, candidate)
    passed = admissible(condition, candidate)
    name = condition isa PredicateAdmissibility ? condition.name : nameof(typeof(condition))
    description = condition isa PredicateAdmissibility ? condition.description : ""
    reason = passed ? "" : (isempty(description) ? "admissibility condition :$(name) rejected the candidate" : description)
    metadata = Dict{Symbol,Any}(:condition_type => typeof(condition))
    !isempty(description) && (metadata[:description] = description)
    return AdmissibilityCheck(name, passed; reason=reason, metadata=metadata)
end

"""
    GenerativeSpecification

General physical specification `G = (G_space, w_alpha, H_alpha, A, U, C_adm, R_max, k_max)` used to define an admissible Hamiltonian model space. `space_group`, `wyckoff_orbits`, and `symmetry_action` remain generic so Foundation does not depend on one crystallography implementation. `max_body_order` is interpreted as algebra-aware many-body rank rather than primitive operator degree. When `max_range` is declared, `range_evaluator(candidate)` must return the candidate's nonnegative finite physical interaction range; without an evaluator, range admissibility is explicitly reported as unevaluated rather than silently accepted.
"""
struct GenerativeSpecification{G,W,H,A<:AbstractOperatorAlgebra,U,C,R,M}
    space_group::G
    wyckoff_orbits::W
    local_spaces::H
    algebra::A
    symmetry_action::U
    admissibility_conditions::C
    max_range::Union{Nothing,Float64}
    max_body_order::Union{Nothing,Int}
    range_evaluator::R
    metadata::M
end

function GenerativeSpecification(
    space_group,
    wyckoff_orbits,
    local_spaces,
    algebra::A,
    symmetry_action;
    admissibility_conditions=(),
    max_range=nothing,
    max_body_order=nothing,
    range_evaluator=nothing,
    metadata=Dict{Symbol,Any}(),
) where {A<:AbstractOperatorAlgebra}
    if !isnothing(max_range)
        isfinite(max_range) && max_range >= 0 || throw(ArgumentError("max_range must be finite and nonnegative"))
    end
    if !isnothing(max_body_order)
        Int(max_body_order) > 0 || throw(ArgumentError("max_body_order must be positive"))
    end
    conditions = Tuple(admissibility_conditions)
    all(condition -> condition isa AbstractAdmissibilityCondition, conditions) ||
        throw(ArgumentError("admissibility conditions must subtype AbstractAdmissibilityCondition"))
    resolved_range = isnothing(max_range) ? nothing : Float64(max_range)
    resolved_body_order = isnothing(max_body_order) ? nothing : Int(max_body_order)
    return GenerativeSpecification(
        space_group,
        wyckoff_orbits,
        local_spaces,
        algebra,
        symmetry_action,
        conditions,
        resolved_range,
        resolved_body_order,
        range_evaluator,
        metadata,
    )
end

# Backward-compatible positional constructor matching the pre-Batch-7 field layout.
function GenerativeSpecification(
    space_group,
    wyckoff_orbits,
    local_spaces,
    algebra::A,
    symmetry_action,
    admissibility_conditions,
    max_range,
    max_body_order,
    metadata,
) where {A<:AbstractOperatorAlgebra}
    return GenerativeSpecification(
        space_group,
        wyckoff_orbits,
        local_spaces,
        algebra,
        symmetry_action;
        admissibility_conditions=admissibility_conditions,
        max_range=max_range,
        max_body_order=max_body_order,
        metadata=metadata,
    )
end

_operator_compatible(algebra::AbstractOperatorAlgebra, operator::AbstractPrimitiveOperator) = applicable(body_order, algebra, operator)
_operator_compatible(::SpinAlgebra, ::IdentityOperator) = true
_operator_compatible(::BosonAlgebra, ::IdentityOperator) = true
_operator_compatible(::FermionAlgebra, ::IdentityOperator) = true
_operator_compatible(::MajoranaAlgebra, ::IdentityOperator) = true
_operator_compatible(::CoordinateMomentumAlgebra, ::IdentityOperator) = true
_operator_compatible(::CompositeAlgebra, ::IdentityOperator) = true
_operator_compatible(algebra::SpinAlgebra, operator::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}) = 1 <= operator.site <= algebra.nsites
_operator_compatible(algebra::BosonAlgebra, operator::Union{BosonAnnihilate,BosonCreate}) = 1 <= operator.mode <= algebra.nmodes
_operator_compatible(algebra::BosonAlgebra, operator::NumberOperator) = operator.kind == :boson && 1 <= operator.mode <= algebra.nmodes
_operator_compatible(algebra::FermionAlgebra, operator::Union{FermionAnnihilate,FermionCreate}) = 1 <= operator.mode <= algebra.nmodes
_operator_compatible(algebra::FermionAlgebra, operator::NumberOperator) = operator.kind == :fermion && 1 <= operator.mode <= algebra.nmodes
_operator_compatible(algebra::MajoranaAlgebra, operator::MajoranaOperator) = 1 <= operator.index <= algebra.ngenerators
_operator_compatible(algebra::CoordinateMomentumAlgebra, operator::Union{PositionOperator,MomentumOperator}) = 1 <= operator.mode <= algebra.ndof
_operator_compatible(::SpinAlgebra, ::NumberOperator) = false
_operator_compatible(::MajoranaAlgebra, ::NumberOperator) = false
_operator_compatible(::CoordinateMomentumAlgebra, ::NumberOperator) = false
_operator_compatible(::SpinAlgebra, ::AbstractPrimitiveOperator) = false
_operator_compatible(::BosonAlgebra, ::AbstractPrimitiveOperator) = false
_operator_compatible(::FermionAlgebra, ::AbstractPrimitiveOperator) = false
_operator_compatible(::MajoranaAlgebra, ::AbstractPrimitiveOperator) = false
_operator_compatible(::CoordinateMomentumAlgebra, ::AbstractPrimitiveOperator) = false
_operator_compatible(algebra::AbstractOperatorAlgebra, expression::ScaledOperator) = _operator_compatible(algebra, expression.operator)
_operator_compatible(algebra::AbstractOperatorAlgebra, expression::OperatorSum) = all(term -> _operator_compatible(algebra, term), expression.terms)
_operator_compatible(algebra::AbstractOperatorAlgebra, expression::OperatorProduct) = all(factor -> _operator_compatible(algebra, factor), expression.factors)

function _operator_compatible(algebra::CompositeAlgebra, operator::FactorOperator)
    1 <= operator.factor <= length(algebra.factors) || return false
    return _operator_compatible(algebra.factors[operator.factor], operator.operator)
end

function _operator_compatible(algebra::CompositeAlgebra, operator::AbstractPrimitiveOperator)
    operator isa IdentityOperator && return true
    factor = try
        _composite_factor_index(algebra, operator)
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
    factor == 0 && return true
    return _operator_compatible(algebra.factors[factor], _unwrap_factor(operator))
end

_operator_compatible(algebra::CompositeAlgebra, expression::ScaledOperator) = _operator_compatible(algebra, expression.operator)
_operator_compatible(algebra::CompositeAlgebra, expression::OperatorSum) = all(term -> _operator_compatible(algebra, term), expression.terms)
_operator_compatible(algebra::CompositeAlgebra, expression::OperatorProduct) = all(factor -> _operator_compatible(algebra, factor), expression.factors)

function _range_check(specification::GenerativeSpecification, candidate::AbstractOperatorExpr)
    isnothing(specification.max_range) && return nothing
    if isnothing(specification.range_evaluator)
        return AdmissibilityCheck(
            :max_range,
            missing;
            reason="max_range=$(specification.max_range) is declared, but no physical range_evaluator is available",
            metadata=Dict(:max_range => specification.max_range),
        )
    end
    applicable(specification.range_evaluator, candidate) || throw(ArgumentError("range_evaluator must accept one candidate Hamiltonian operator"))
    range = specification.range_evaluator(candidate)
    if isnothing(range) || ismissing(range)
        return AdmissibilityCheck(
            :max_range,
            missing;
            reason="range_evaluator did not resolve a physical interaction range for the candidate",
            metadata=Dict(:max_range => specification.max_range),
        )
    end
    range isa Real || throw(ArgumentError("range_evaluator must return a real interaction range, missing, or nothing"))
    isfinite(range) && range >= 0 || throw(ArgumentError("range_evaluator returned a nonfinite or negative interaction range"))
    value = Float64(range)
    passed = value <= specification.max_range
    reason = passed ? "" : "physical interaction range $(value) exceeds max_range=$(specification.max_range)"
    metadata = Dict{Symbol,Any}(:interaction_range => value, :max_range => specification.max_range)
    return AdmissibilityCheck(:max_range, passed; reason=reason, metadata=metadata)
end

function _algebra_label(algebra::AbstractOperatorAlgebra)
    return applicable(algebra_kind, algebra) ? algebra_kind(algebra) : nameof(typeof(algebra))
end

"""
    admissibility_report(specification, candidate)
    admissibility_report(space, candidate)

Evaluate physical/model-space admissibility without consulting any numerical solver. The report always checks compatibility with the declared operator algebra, then evaluates configured body-order, physical-range, and custom admissibility conditions. A declared but unevaluable criterion makes the report incomplete and therefore not admissible.
"""
function admissibility_report(specification::GenerativeSpecification, candidate::AbstractOperatorExpr)
    checks = AdmissibilityCheck[]
    algebra_label = _algebra_label(specification.algebra)
    algebra_ok = _operator_compatible(specification.algebra, candidate)
    algebra_reason = algebra_ok ? "" : "candidate operator is incompatible with the declared $(algebra_label) algebra or its index bounds"
    push!(
        checks,
        AdmissibilityCheck(:operator_algebra, algebra_ok; reason=algebra_reason, metadata=Dict(:algebra => specification.algebra)),
    )

    order = nothing
    if algebra_ok && applicable(body_order, specification.algebra, candidate)
        order = body_order(specification.algebra, candidate)
    end
    if !isnothing(specification.max_body_order)
        if isnothing(order)
            push!(
                checks,
                AdmissibilityCheck(
                    :max_body_order,
                    missing;
                    reason="body order cannot be evaluated for the declared algebra and candidate",
                    metadata=Dict(:max_body_order => specification.max_body_order),
                ),
            )
        else
            passed = order <= specification.max_body_order
            reason = passed ? "" : "many-body order $(order) exceeds max_body_order=$(specification.max_body_order)"
            metadata = Dict{Symbol,Any}(:body_order => order, :max_body_order => specification.max_body_order)
            push!(checks, AdmissibilityCheck(:max_body_order, passed; reason=reason, metadata=metadata))
        end
    end

    if !isnothing(specification.max_range)
        if algebra_ok
            push!(checks, _range_check(specification, candidate))
        else
            push!(
                checks,
                AdmissibilityCheck(
                    :max_range,
                    missing;
                    reason="physical range is not evaluated because operator-algebra compatibility failed",
                    metadata=Dict(:max_range => specification.max_range),
                ),
            )
        end
    end

    transport = specification.metadata isa AbstractDict || specification.metadata isa NamedTuple ? specification.metadata : Dict{Symbol,Any}()
    for (key, name) in ((:symmetry_action_transport, :transported_symmetry),
                        (:local_spaces_transport, :transported_local_spaces),
                        (:admissibility_conditions_transport, :transported_conditions))
        if get(transport, key, nothing) === :not_transported
            push!(checks, AdmissibilityCheck(name, missing;
                reason="source $(name) has not been transported into this operator representation",
                metadata=Dict(:transport_status => :not_transported)))
        end
    end

    for condition in specification.admissibility_conditions
        if algebra_ok
            push!(checks, admissibility_check(condition, candidate))
        else
            name = condition isa PredicateAdmissibility ? condition.name : nameof(typeof(condition))
            push!(
                checks,
                AdmissibilityCheck(
                    name,
                    missing;
                    reason="condition :$(name) is not evaluated because operator-algebra compatibility failed",
                    metadata=Dict(:condition_type => typeof(condition)),
                ),
            )
        end
    end

    metadata = Dict{Symbol,Any}(
        :scope => :generative_specification,
        :candidate_type => typeof(candidate),
        :algebra_kind => algebra_label,
        :max_range => specification.max_range,
        :max_body_order => specification.max_body_order,
    )
    return AdmissibilityReport(checks; metadata=metadata)
end

function admissibility_report(specification::GenerativeSpecification, candidate)
    reason = "candidate must subtype AbstractOperatorExpr to belong to a Hamiltonian model space"
    check = AdmissibilityCheck(:operator_expression, false; reason=reason, metadata=Dict(:candidate_type => typeof(candidate)))
    metadata = Dict{Symbol,Any}(
        :scope => :generative_specification,
        :candidate_type => typeof(candidate),
        :algebra_kind => _algebra_label(specification.algebra),
        :max_range => specification.max_range,
        :max_body_order => specification.max_body_order,
    )
    return AdmissibilityReport((check,); metadata=metadata)
end

admissible(specification::GenerativeSpecification, candidate) = admissibility_report(specification, candidate).admissible

"""
    HamiltonianBasis(operators; labels=nothing, metadata=Dict())

Linearly independent Hermitian operator basis `{Phi_a}` for one admissible Hamiltonian space. Linear independence, Hermiticity, symmetry invariance, and completeness are mathematical properties established by the generator/validator that created the object; this container records those claims explicitly in `metadata` rather than silently asserting them.
"""
struct HamiltonianBasis{O,L,M}
    operators::O
    labels::L
    metadata::M
end

function HamiltonianBasis(operators::AbstractVector{<:AbstractOperatorExpr}; labels=nothing, metadata=Dict{Symbol,Any}())
    isempty(operators) && throw(ArgumentError("HamiltonianBasis requires at least one operator"))
    ops = collect(operators)
    names = isnothing(labels) ? [Symbol("Phi_", i) for i in eachindex(ops)] : Symbol.(collect(labels))
    length(names) == length(ops) || throw(DimensionMismatch("basis labels and operators must have equal length"))
    length(unique(names)) == length(names) || throw(ArgumentError("HamiltonianBasis labels must be unique"))
    return HamiltonianBasis(ops, names, metadata)
end

Base.length(basis::HamiltonianBasis) = length(basis.operators)
Base.getindex(basis::HamiltonianBasis, i::Integer) = basis.operators[Int(i)]
Base.iterate(basis::HamiltonianBasis, state...) = iterate(basis.operators, state...)
model_space_dimension(basis::HamiltonianBasis) = length(basis)

"""
    HamiltonianSpace(specification, basis; metadata=Dict())

Finite-dimensional Hamiltonian space `H(G)` represented by a real basis of Hermitian operators. The object ties the generated basis to the physical specification that defines its domain; direct construction records this relationship but does not by itself certify Hermiticity, independence, symmetry completeness, or basis admissibility.
"""
struct HamiltonianSpace{S<:GenerativeSpecification,B<:HamiltonianBasis,M} <: AbstractModelSpace
    specification::S
    basis::B
    metadata::M
end

HamiltonianSpace(specification::GenerativeSpecification, basis::HamiltonianBasis; metadata=Dict{Symbol,Any}()) = HamiltonianSpace(specification, basis, metadata)

Base.length(space::HamiltonianSpace) = length(space.basis)
model_space_dimension(space::HamiltonianSpace) = length(space)

function _basis_span_check(space::HamiltonianSpace, candidate::AbstractOperatorExpr)
    tolerance = space.basis.metadata isa AbstractDict ? get(space.basis.metadata, :linear_tolerance, 1e-10) : 1e-10
    membership = structural_span_membership(space.basis.operators, candidate, space.specification.algebra; tolerance=tolerance)
    if membership === true
        return AdmissibilityCheck(:basis_span, true; metadata=Dict(:certification => :canonical_linear_span, :tolerance => tolerance))
    elseif membership === false
        return AdmissibilityCheck(
            :basis_span,
            false;
            reason="candidate does not belong to the real canonical span of the HamiltonianBasis",
            metadata=Dict(:certification => :canonical_linear_span, :tolerance => tolerance),
        )
    end
    return AdmissibilityCheck(
        :basis_span,
        missing;
        reason="candidate span membership cannot be certified because canonical linear coordinates contain unsupported nonnumeric coefficients",
        metadata=Dict(:certification => :unresolved_symbolic_coefficients),
    )
end

function _basis_span_check(space::HamiltonianSpace, candidate)
    return AdmissibilityCheck(
        :basis_span,
        false;
        reason="candidate must subtype AbstractOperatorExpr to belong to a Hamiltonian model space",
        metadata=Dict(:candidate_type => typeof(candidate)),
    )
end

"""Extract the coefficient-independent canonical structure represented by a `HamiltonianSpace` basis."""
function structure(space::HamiltonianSpace)
    context = (:algebra, _freeze_structural_value(space.specification.algebra))
    metadata = Dict{Symbol,Any}(:scope => :hamiltonian_space, :basis_labels => Tuple(space.basis.labels), :model_space_dimension => length(space))
    return _model_structure(context, space.specification.algebra, space.basis.operators; metadata=metadata)
end

canonicalize_structure(space::HamiltonianSpace) = structure(space)
structural_signature(space::HamiltonianSpace) = structural_signature(structure(space))
interaction_graph(space::HamiltonianSpace) = interaction_graph(structure(space))

function admissibility_report(space::HamiltonianSpace, candidate)
    report = admissibility_report(space.specification, candidate)
    checks = AdmissibilityCheck[report.checks...]
    push!(checks, _basis_span_check(space, candidate))
    metadata = copy(report.metadata)
    metadata[:scope] = :hamiltonian_space
    metadata[:model_space_dimension] = length(space)
    return AdmissibilityReport(checks; metadata=metadata)
end

admissible(space::HamiltonianSpace, candidate) = admissibility_report(space, candidate).admissible

"""
    admissibility_report(space)

Evaluate every basis direction of a `HamiltonianSpace` against its physical specification. This is an admissibility audit only; it does not certify linear independence, Hermiticity, symmetry completeness, or universal generator completeness.
"""
function admissibility_report(space::HamiltonianSpace)
    checks = AdmissibilityCheck[]
    for (index, (label, operator)) in enumerate(zip(space.basis.labels, space.basis.operators))
        report = admissibility_report(space.specification, operator)
        passed = report.complete ? report.admissible : missing
        reasons = admissibility_reasons(report)
        reason = if passed === true
            ""
        elseif isempty(reasons)
            "basis direction :$(label) is not certified admissible"
        else
            join(reasons, "; ")
        end
        metadata = Dict{Symbol,Any}(:basis_index => index, :basis_label => label, :operator_report => report)
        push!(checks, AdmissibilityCheck(label, passed; reason=reason, metadata=metadata))
    end
    return AdmissibilityReport(checks; metadata=Dict(:scope => :hamiltonian_basis, :model_space_dimension => length(space)))
end

admissible(space::HamiltonianSpace) = admissibility_report(space).admissible

"""
    hamiltonian_from_coefficients(space, theta)

Construct `H(theta) = sum_a theta[a] Phi_a` in the generated operator basis. The coefficients are required to be real because `HamiltonianSpace` is a real span of Hermitian basis operators.
"""
function hamiltonian_from_coefficients(space::HamiltonianSpace, theta::AbstractVector{<:Real})
    length(theta) == length(space) || throw(DimensionMismatch("parameter vector length does not match Hamiltonian-space dimension"))
    terms = AbstractOperatorExpr[]
    sizehint!(terms, length(theta))
    for (coefficient, operator) in zip(theta, space.basis.operators)
        iszero(coefficient) && continue
        push!(terms, coefficient * operator)
    end
    isempty(terms) && return 0 * IdentityOperator()
    return length(terms) == 1 ? terms[1] : OperatorSum(terms)
end

"""
    generated_basis(specification, generator)

Run a user- or backend-supplied invariant-basis generator. The generator must return either a `HamiltonianBasis` or a vector of operator expressions. This keeps the model-space API independent of a particular symmetry engine.
"""
function generated_basis(specification::GenerativeSpecification, generator)
    result = generator(specification)
    result isa HamiltonianBasis && return result
    result isa AbstractVector{<:AbstractOperatorExpr} && return HamiltonianBasis(result)
    throw(ArgumentError("basis generator must return HamiltonianBasis or a vector of operator expressions"))
end

function _require_complete_admissibility(report::AdmissibilityReport, subject::AbstractString)
    reasons = join(admissibility_reasons(report), "; ")
    report.complete || throw(ArgumentError("$(subject) has unevaluated physical admissibility criteria: $(reasons)"))
    report.admissible || throw(ArgumentError("$(subject) violates the declared physical model space: $(reasons)"))
    return report
end

function generate_hamiltonian_space(specification::GenerativeSpecification, generator; metadata=Dict{Symbol,Any}())
    space = HamiltonianSpace(specification, generated_basis(specification, generator); metadata=metadata)
    _require_complete_admissibility(admissibility_report(space), "generated Hamiltonian basis")
    return space
end

# Crystallographic generators intentionally enter through the existing callable-generator interface. The crystallography layer defines `CrystallographicGenerator` rather than coupling Foundation to Spglib.jl types.
