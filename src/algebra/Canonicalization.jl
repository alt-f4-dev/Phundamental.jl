# Fixed-point canonicalization for noncommutative operator expressions.
#
# The canonicalizer performs structural normalization, exact local rewrite
# rules, and like-term collection.  It deliberately does not perform
# transformation-specific substitutions; those belong to Transformations/.

# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

_iszeroexpr(expr::AbstractOperatorExpr) =
    expr isa ScaledOperator && iszero(expr.coefficient)

function _expr_equal(A::AbstractOperatorExpr, B::AbstractOperatorExpr)
    typeof(A) === typeof(B) || return false

    if A isa IdentityOperator
        return A.label == B.label
    elseif A isa FactorOperator
        return A.factor == B.factor && _expr_equal(A.operator, B.operator)
    elseif A isa SpinX || A isa SpinY || A isa SpinZ || A isa SpinPlus || A isa SpinMinus
        return A.site == B.site
    elseif A isa BosonAnnihilate || A isa BosonCreate ||
           A isa FermionAnnihilate || A isa FermionCreate ||
           A isa PositionOperator || A isa MomentumOperator
        return A.mode == B.mode
    elseif A isa MajoranaOperator
        return A.index == B.index
    elseif A isa NumberOperator
        return A.kind == B.kind && A.mode == B.mode
    elseif A isa ScaledOperator
        return A.coefficient == B.coefficient && _expr_equal(A.operator, B.operator)
    elseif A isa OperatorSum
        length(A.terms) == length(B.terms) || return false
        return all(_expr_equal(a, b) for (a, b) in zip(A.terms, B.terms))
    elseif A isa OperatorProduct
        length(A.factors) == length(B.factors) || return false
        return all(_expr_equal(a, b) for (a, b) in zip(A.factors, B.factors))
    end

    return false
end

function _split_coefficient(expr::AbstractOperatorExpr)
    if expr isa ScaledOperator
        return expr.coefficient, expr.operator
    end
    return 1, expr
end

function _scale(coefficient::Number, expr::AbstractOperatorExpr)
    iszero(coefficient) && return _zero_operator()
    coefficient == 1 && return expr

    if expr isa ScaledOperator
        return _scale(coefficient * expr.coefficient, expr.operator)
    elseif expr isa OperatorSum
        return OperatorSum(AbstractOperatorExpr[
            _scale(coefficient, term) for term in expr.terms
        ])
    end

    return ScaledOperator(coefficient, expr)
end

function _collect_like_terms(terms::Vector{AbstractOperatorExpr})
    cores = AbstractOperatorExpr[]
    coefficients = Number[]

    for term in terms
        _iszeroexpr(term) && continue
        coefficient, core = _split_coefficient(term)

        match_index = findfirst(existing -> _expr_equal(existing, core), cores)
        if match_index === nothing
            push!(cores, core)
            push!(coefficients, coefficient)
        else
            coefficients[match_index] += coefficient
        end
    end

    result = AbstractOperatorExpr[]
    for (coefficient, core) in zip(coefficients, cores)
        iszero(coefficient) || push!(result, _scale(coefficient, core))
    end
    return result
end

function _normalized_sum(terms::Vector{AbstractOperatorExpr})
    flat = AbstractOperatorExpr[]
    for term in terms
        normalized = _normalize_structure(term)
        if normalized isa OperatorSum
            append!(flat, normalized.terms)
        elseif !_iszeroexpr(normalized)
            push!(flat, normalized)
        end
    end

    collected = _collect_like_terms(flat)
    isempty(collected) && return _zero_operator()
    length(collected) == 1 && return collected[1]
    return OperatorSum(collected)
end

function _distribute_product(factors::Vector{AbstractOperatorExpr})
    # Expand the first sum encountered; recursive normalization handles any
    # additional sums introduced by the expansion.
    sum_index = findfirst(factor -> factor isa OperatorSum, factors)
    sum_index === nothing && return nothing

    sum_factor = factors[sum_index]::OperatorSum
    expanded = AbstractOperatorExpr[]

    for term in sum_factor.terms
        new_factors = copy(factors)
        new_factors[sum_index] = term
        push!(expanded, _normalize_structure(_make_product(new_factors)))
    end

    return _normalized_sum(expanded)
end

function _normalized_product(factors::Vector{AbstractOperatorExpr})
    flat = AbstractOperatorExpr[]
    coefficient::Number = 1

    for factor in factors
        normalized = _normalize_structure(factor)

        if _iszeroexpr(normalized)
            return _zero_operator()
        elseif normalized isa ScaledOperator
            coefficient *= normalized.coefficient
            push!(flat, normalized.operator)
        elseif normalized isa OperatorProduct
            append!(flat, normalized.factors)
        elseif normalized isa IdentityOperator
            continue
        else
            push!(flat, normalized)
        end
    end

    iszero(coefficient) && return _zero_operator()

    distributed = _distribute_product(flat)
    distributed === nothing || return _scale(coefficient, distributed)

    core = _make_product(flat)
    return _scale(coefficient, core)
end

_normalize_structure(expr::AbstractPrimitiveOperator) = expr

function _normalize_structure(expr::ScaledOperator)
    child = _normalize_structure(expr.operator)
    return _scale(expr.coefficient, child)
end

function _normalize_structure(expr::OperatorSum)
    return _normalized_sum(expr.terms)
end

function _normalize_structure(expr::OperatorProduct)
    return _normalized_product(expr.factors)
end

# ---------------------------------------------------------------------------
# Canonicalization public interface
# ---------------------------------------------------------------------------

"""
    canonicalize(algebra, expr; hbar=1, max_iterations=256)

Return a canonical expression obtained by repeatedly applying the exact
relations of `algebra` until a fixed point is reached.

The routine performs:

1. flattening of nested sums and products,
2. scalar-coefficient collection,
3. distribution of products over sums,
4. identity and zero removal,
5. algebra-specific normal ordering,
6. CCR/CAR/Clifford/spin/canonical-coordinate rewrites,
7. collection of like terms.

`hbar` defaults to one so that the algebra layer is usable in dimensionless
many-body calculations while still supporting the dimensional convention used
by the mathematical formalism.
"""
function canonicalize(
    algebra::AbstractOperatorAlgebra,
    expr::AbstractOperatorExpr;
    hbar::Number = 1,
    max_iterations::Integer = 256,
)
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))

    current = _normalize_structure(expr)

    for _ in 1:max_iterations
        rewritten = rewrite_once(algebra, current; hbar=hbar)
        next = _normalize_structure(rewritten)

        _expr_equal(current, next) && return next
        current = next
    end

    throw(ErrorException(
        "canonicalization did not converge after $(max_iterations) iterations"
    ))
end

"""
    normal_order(algebra, expr; kwargs...)

Alias for `canonicalize` emphasizing the operator-ordering use case.
"""
normal_order(
    algebra::AbstractOperatorAlgebra,
    expr::AbstractOperatorExpr;
    kwargs...,
) = canonicalize(algebra, expr; kwargs...)
