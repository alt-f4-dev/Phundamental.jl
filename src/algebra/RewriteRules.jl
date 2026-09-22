# Local algebra rewrite rules.
#
# This file knows individual exact identities, but not the global fixed-point
# simplification strategy.  Canonicalization.jl supplies that strategy.

# ---------------------------------------------------------------------------
# Canonical-order keys
# ---------------------------------------------------------------------------

# Primitive operator family order for tensor-product/composite algebras.
function _composite_family_index(algebra::CompositeAlgebra, op::AbstractOperatorExpr)
    family = operator_family(op)
    for (i, factor) in enumerate(algebra.factors)
        algebra_kind(factor) == family && return i
    end
    return typemax(Int)
end

_boson_rank(::BosonCreate) = 1
_boson_rank(::NumberOperator) = 2
_boson_rank(::BosonAnnihilate) = 3

_fermion_rank(::FermionCreate) = 1
_fermion_rank(::NumberOperator) = 2
_fermion_rank(::FermionAnnihilate) = 3

_spin_rank(::SpinPlus) = 1
_spin_rank(::SpinX) = 2
_spin_rank(::SpinY) = 3
_spin_rank(::SpinZ) = 4
_spin_rank(::SpinMinus) = 5

_operator_index(op::Union{BosonCreate,BosonAnnihilate}) = op.mode
_operator_index(op::Union{FermionCreate,FermionAnnihilate}) = op.mode
_operator_index(op::NumberOperator) = op.mode
_operator_index(op::_SpinPrimitive) = op.site
_operator_index(op::MajoranaOperator) = op.index
_operator_index(op::Union{PositionOperator,MomentumOperator}) = op.mode

# ---------------------------------------------------------------------------
# Pair rewrites
# ---------------------------------------------------------------------------

"""Return an exact replacement for `A*B`, or `nothing` if no rewrite is needed."""
_rewrite_pair(
    ::AbstractOperatorAlgebra,
    ::AbstractOperatorExpr,
    ::AbstractOperatorExpr;
    hbar::Number = 1,
) = nothing

# Identity factors.
_rewrite_pair(::AbstractOperatorAlgebra, ::IdentityOperator, B::AbstractOperatorExpr; hbar::Number=1) = B
_rewrite_pair(::AbstractOperatorAlgebra, A::AbstractOperatorExpr, ::IdentityOperator; hbar::Number=1) = A

# Bosonic normal ordering: creation < number < annihilation.
function _rewrite_pair(
    ::BosonAlgebra,
    A::Union{BosonCreate,BosonAnnihilate,NumberOperator},
    B::Union{BosonCreate,BosonAnnihilate,NumberOperator};
    hbar::Number = 1,
)
    A isa NumberOperator && A.kind != :boson && return nothing
    B isa NumberOperator && B.kind != :boson && return nothing

    # b_i^† b_i = n_i.
    if A isa BosonCreate && B isa BosonAnnihilate && A.mode == B.mode
        return nb(A.mode)
    end

    # b_i b_j^† = b_j^† b_i + delta_ij.
    if A isa BosonAnnihilate && B isa BosonCreate
        ordered = B * A
        return A.mode == B.mode ? IdentityOperator() + ordered : ordered
    end

    # [n_i,b_j^†] = delta_ij b_j^†.
    if A isa NumberOperator && B isa BosonCreate
        ordered = B * A
        return A.mode == B.mode ? ordered + B : ordered
    end

    # b_i n_j = n_j b_i + delta_ij b_i.
    if A isa BosonAnnihilate && B isa NumberOperator
        ordered = B * A
        return A.mode == B.mode ? ordered + A : ordered
    end

    # Operators of the same type commute; sort their mode labels.
    if typeof(A) == typeof(B) && _operator_index(A) > _operator_index(B)
        return B * A
    end

    return nothing
end

# Fermionic normal ordering: creation < number < annihilation.
function _rewrite_pair(
    ::FermionAlgebra,
    A::Union{FermionCreate,FermionAnnihilate,NumberOperator},
    B::Union{FermionCreate,FermionAnnihilate,NumberOperator};
    hbar::Number = 1,
)
    A isa NumberOperator && A.kind != :fermion && return nothing
    B isa NumberOperator && B.kind != :fermion && return nothing

    # Same-mode fermionic nilpotency and projector identities.
    if A isa FermionCreate && B isa FermionCreate && A.mode == B.mode
        return _zero_operator()
    elseif A isa FermionAnnihilate && B isa FermionAnnihilate && A.mode == B.mode
        return _zero_operator()
    elseif A isa NumberOperator && B isa NumberOperator && A.mode == B.mode
        return nf(A.mode)
    elseif A isa FermionCreate && B isa NumberOperator && A.mode == B.mode
        return _zero_operator()
    elseif A isa NumberOperator && B isa FermionAnnihilate && A.mode == B.mode
        return _zero_operator()
    elseif A isa NumberOperator && B isa FermionCreate && A.mode == B.mode
        return B
    elseif A isa FermionAnnihilate && B isa NumberOperator && A.mode == B.mode
        return A
    end

    # c_i^† c_i = n_i.
    if A isa FermionCreate && B isa FermionAnnihilate && A.mode == B.mode
        return nf(A.mode)
    end

    # c_i c_j^† = delta_ij - c_j^† c_i.
    if A isa FermionAnnihilate && B isa FermionCreate
        ordered = -(B * A)
        return A.mode == B.mode ? IdentityOperator() + ordered : ordered
    end

    # Number operators are even: [n_i,c_j^†] = delta_ij c_j^†.
    if A isa NumberOperator && B isa FermionCreate
        ordered = B * A
        return A.mode == B.mode ? ordered + B : ordered
    end

    # c_i n_j = n_j c_i + delta_ij c_i.
    if A isa FermionAnnihilate && B isa NumberOperator
        ordered = B * A
        return A.mode == B.mode ? ordered + A : ordered
    end

    # Equal-species fermionic generators anticommute across distinct modes.
    if A isa FermionCreate && B isa FermionCreate && A.mode > B.mode
        return -(B * A)
    elseif A isa FermionAnnihilate && B isa FermionAnnihilate && A.mode > B.mode
        return -(B * A)
    end

    # Number operators commute and are sorted by mode.
    if A isa NumberOperator && B isa NumberOperator && A.mode > B.mode
        return B * A
    end

    return nothing
end

# Reduce same-mode fermionic structures that are separated by other modes
# after ordinary creation < number < annihilation ordering has converged.
function _rewrite_nonlocal_fermion_product(
    algebra::FermionAlgebra,
    factors::Vector{AbstractOperatorExpr},
)
    for mode in 1:algebra.nmodes
        create_index = findfirst(factor -> factor isa FermionCreate && factor.mode == mode, factors)
        number_index = findfirst(
            factor -> factor isa NumberOperator && factor.kind == :fermion && factor.mode == mode,
            factors,
        )
        annihilate_index = findfirst(factor -> factor isa FermionAnnihilate && factor.mode == mode, factors)

        # In canonical creation < number ordering, c_i^† ... n_i = 0 because
        # n_i commutes with all intervening different-mode operators.
        if create_index !== nothing && number_index !== nothing
            return _zero_operator()
        end

        # In canonical number < annihilation ordering, n_i ... c_i = 0.
        if number_index !== nothing && annihilate_index !== nothing
            return _zero_operator()
        end

        # Reduce a non-adjacent c_i^† ... c_i pair to n_i. Moving c_i across
        # every intervening odd fermionic generator contributes one minus sign;
        # number operators are even and therefore contribute no sign.
        if create_index !== nothing && annihilate_index !== nothing
            create_index < annihilate_index || continue

            crossings = count(
                factor -> factor isa FermionCreate || factor isa FermionAnnihilate,
                factors[create_index+1:annihilate_index-1],
            )

            reduced = AbstractOperatorExpr[]
            sizehint!(reduced, length(factors) - 1)

            for i in eachindex(factors)
                if i == create_index
                    push!(reduced, nf(mode))
                elseif i != annihilate_index
                    push!(reduced, factors[i])
                end
            end

            product = _make_product(reduced)
            return isodd(crossings) ? -product : product
        end
    end

    return nothing
end

# Majorana Clifford ordering: increasing generator index.
function _rewrite_pair(
    ::MajoranaAlgebra,
    A::MajoranaOperator,
    B::MajoranaOperator;
    hbar::Number = 1,
)
    A.index == B.index && return IdentityOperator()
    A.index > B.index && return -(B * A)
    return nothing
end

# Spin operators on distinct sites commute and are sorted by site.  On a
# single site we use AB = BA + [A,B] whenever a directly known spin
# commutator can move an inverted pair toward the chosen order.
function _rewrite_pair(
    algebra::SpinAlgebra,
    A::_SpinPrimitive,
    B::_SpinPrimitive;
    hbar::Number = 1,
)
    if A.site != B.site
        return A.site > B.site ? B * A : nothing
    end

    rankA = _spin_rank(A)
    rankB = _spin_rank(B)
    rankA <= rankB && return nothing

    bracket = primitive_commutator(algebra, A, B; hbar=hbar)
    bracket === nothing && return nothing
    return B * A + bracket
end

# Coordinate-momentum ordering: x before p, with modes sorted inside each set.
function _rewrite_pair(
    ::CoordinateMomentumAlgebra,
    A::Union{PositionOperator,MomentumOperator},
    B::Union{PositionOperator,MomentumOperator};
    hbar::Number = 1,
)
    if A isa MomentumOperator && B isa PositionOperator
        ordered = B * A
        return A.mode == B.mode ? ordered - (im * hbar) * IdentityOperator() : ordered
    end

    if typeof(A) == typeof(B) && A.mode > B.mode
        return B * A
    end

    return nothing
end

# Composite algebras use the algebra relations within each factor and commute
# primitive operators belonging to distinct tensor-product factors.
function _rewrite_pair(
    algebra::CompositeAlgebra,
    A::AbstractPrimitiveOperator,
    B::AbstractPrimitiveOperator;
    hbar::Number = 1,
)
    subA = _matching_subalgebra(algebra, A)
    subB = _matching_subalgebra(algebra, B)

    (subA === nothing || subB === nothing) && return nothing

    if subA === subB
        return _rewrite_pair(subA, A, B; hbar=hbar)
    end

    idxA = _composite_family_index(algebra, A)
    idxB = _composite_family_index(algebra, B)
    return idxA > idxB ? B * A : nothing
end

# ---------------------------------------------------------------------------
# One-step recursive rewrite
# ---------------------------------------------------------------------------

function _make_product(factors::Vector{AbstractOperatorExpr})
    isempty(factors) && return IdentityOperator()
    length(factors) == 1 && return factors[1]
    return OperatorProduct(factors)
end

function _replace_pair(
    factors::Vector{AbstractOperatorExpr},
    i::Int,
    replacement::AbstractOperatorExpr,
)
    prefix = i > 1 ? factors[1:i-1] : AbstractOperatorExpr[]
    suffix = i + 1 < length(factors) ? factors[i+2:end] : AbstractOperatorExpr[]
    return _make_product(AbstractOperatorExpr[prefix...; replacement; suffix...])
end

"""
    rewrite_once(algebra, expr; hbar=1)

Apply at most one local algebra rewrite to `expr`, recursing into sums,
products, and scaled expressions.  Repeated application is managed by
`canonicalize`.
"""
rewrite_once(
    ::AbstractOperatorAlgebra,
    expr::AbstractPrimitiveOperator;
    hbar::Number = 1,
) = expr

function rewrite_once(
    algebra::AbstractOperatorAlgebra,
    expr::ScaledOperator;
    hbar::Number = 1,
)
    child = rewrite_once(algebra, expr.operator; hbar=hbar)
    return ScaledOperator(expr.coefficient, child)
end

function rewrite_once(
    algebra::AbstractOperatorAlgebra,
    expr::OperatorSum;
    hbar::Number = 1,
)
    return OperatorSum(AbstractOperatorExpr[
        rewrite_once(algebra, term; hbar=hbar) for term in expr.terms
    ])
end

function rewrite_once(
    algebra::AbstractOperatorAlgebra,
    expr::OperatorProduct;
    hbar::Number = 1,
)
    factors = copy(expr.factors)

    # First descend into compound factors.
    for i in eachindex(factors)
        factor = factors[i]
        if !(factor isa AbstractPrimitiveOperator)
            rewritten = rewrite_once(algebra, factor; hbar=hbar)
            if rewritten !== factor
                factors[i] = rewritten
                return _make_product(factors)
            end
        end
    end

    # Then inspect adjacent primitive pairs.
    for i in 1:(length(factors) - 1)
        A = factors[i]
        B = factors[i + 1]
        replacement = _rewrite_pair(algebra, A, B; hbar=hbar)
        replacement === nothing || return _replace_pair(factors, i, replacement)
    end

    # Fermionic creation < number < annihilation ordering can leave exact
    # same-mode reductions separated by different-mode operators. Apply one
    # such reduction after ordinary adjacent rewriting has reached local order.
    if algebra isa FermionAlgebra
        replacement = _rewrite_nonlocal_fermion_product(algebra, factors)
        replacement === nothing || return replacement
    end

    return expr
end
