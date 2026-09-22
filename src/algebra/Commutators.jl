# Algebra-aware commutators and anticommutators.
#
# The unevaluated two-argument forms construct noncommutative expressions.
# The algebra-aware forms canonicalize those expressions using the defining
# relations of the supplied operator algebra.

"""Return the formal commutator `[A,B] = A*B - B*A`."""
commutator(A::AbstractOperatorExpr, B::AbstractOperatorExpr) = A * B - B * A

"""Return the formal anticommutator `{A,B} = A*B + B*A`."""
anticommutator(A::AbstractOperatorExpr, B::AbstractOperatorExpr) = A * B + B * A

"""
    commutator(algebra, A, B; hbar=1)

Evaluate and canonicalize the commutator in `algebra`.
"""
function commutator(
    algebra::AbstractOperatorAlgebra,
    A::AbstractOperatorExpr,
    B::AbstractOperatorExpr;
    hbar::Number = 1,
)
    direct = primitive_commutator(algebra, A, B; hbar=hbar)
    direct !== nothing && return canonicalize(algebra, direct; hbar=hbar)
    return canonicalize(algebra, commutator(A, B); hbar=hbar)
end

"""
    anticommutator(algebra, A, B; hbar=1)

Evaluate and canonicalize the anticommutator in `algebra`.
"""
function anticommutator(
    algebra::AbstractOperatorAlgebra,
    A::AbstractOperatorExpr,
    B::AbstractOperatorExpr;
    hbar::Number = 1,
)
    direct = primitive_anticommutator(algebra, A, B; hbar=hbar)
    direct !== nothing && return canonicalize(algebra, direct; hbar=hbar)
    return canonicalize(algebra, anticommutator(A, B); hbar=hbar)
end

_zero_operator() = 0 * IdentityOperator()

# ---------------------------------------------------------------------------
# Operator-family introspection
# ---------------------------------------------------------------------------

operator_family(::SpinX) = :spin
operator_family(::SpinY) = :spin
operator_family(::SpinZ) = :spin
operator_family(::SpinPlus) = :spin
operator_family(::SpinMinus) = :spin
operator_family(::BosonAnnihilate) = :boson
operator_family(::BosonCreate) = :boson
operator_family(::FermionAnnihilate) = :fermion
operator_family(::FermionCreate) = :fermion
operator_family(::MajoranaOperator) = :majorana
operator_family(::PositionOperator) = :coordinate_momentum
operator_family(::MomentumOperator) = :coordinate_momentum
operator_family(op::NumberOperator) = op.kind
operator_family(::IdentityOperator) = :identity
operator_family(::AbstractOperatorExpr) = :composite_expression

function _matching_subalgebra(algebra::CompositeAlgebra, op::AbstractOperatorExpr)
    family = operator_family(op)
    matches = [factor for factor in algebra.factors if algebra_kind(factor) == family]
    isempty(matches) && return nothing
    length(matches) == 1 && return only(matches)
    throw(ArgumentError("operator family $family occurs in multiple tensor factors; use atfactor(factor, operator)"))
end

# ---------------------------------------------------------------------------
# Primitive commutators
# ---------------------------------------------------------------------------

"""
    primitive_commutator(algebra, A, B; hbar=1)

Return a directly known primitive commutator, or `nothing` when no dedicated
primitive rule is defined.  A returned zero is represented as `0*I`.
"""
primitive_commutator(
    ::AbstractOperatorAlgebra,
    ::AbstractOperatorExpr,
    ::AbstractOperatorExpr;
    hbar::Number = 1,
) = nothing

# Bosonic CCR.
function primitive_commutator(
    ::BosonAlgebra,
    A::Union{BosonAnnihilate,BosonCreate},
    B::Union{BosonAnnihilate,BosonCreate};
    hbar::Number = 1,
)
    A.mode == B.mode || return _zero_operator()

    if A isa BosonAnnihilate && B isa BosonCreate
        return IdentityOperator()
    elseif A isa BosonCreate && B isa BosonAnnihilate
        return -IdentityOperator()
    end
    return _zero_operator()
end

# Number-operator commutators for bosons.
function primitive_commutator(
    ::BosonAlgebra,
    A::NumberOperator,
    B::Union{BosonAnnihilate,BosonCreate};
    hbar::Number = 1,
)
    A.kind == :boson || return nothing
    A.mode == B.mode || return _zero_operator()
    return B isa BosonCreate ? B : -B
end

function primitive_commutator(
    algebra::BosonAlgebra,
    A::Union{BosonAnnihilate,BosonCreate},
    B::NumberOperator;
    hbar::Number = 1,
)
    value = primitive_commutator(algebra, B, A; hbar=hbar)
    return value === nothing ? nothing : -value
end

# Spin algebra.  Operators on distinct sites commute.
const _SpinPrimitive = Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}

function primitive_commutator(
    ::SpinAlgebra,
    A::_SpinPrimitive,
    B::_SpinPrimitive;
    hbar::Number = 1,
)
    A.site == B.site || return _zero_operator()
    typeof(A) == typeof(B) && return _zero_operator()
    i = A.site

    # Cartesian su(2) relations.
    A isa SpinX && B isa SpinY && return (im * hbar) * Sz(i)
    A isa SpinY && B isa SpinX && return (-im * hbar) * Sz(i)
    A isa SpinY && B isa SpinZ && return (im * hbar) * Sx(i)
    A isa SpinZ && B isa SpinY && return (-im * hbar) * Sx(i)
    A isa SpinZ && B isa SpinX && return (im * hbar) * Sy(i)
    A isa SpinX && B isa SpinZ && return (-im * hbar) * Sy(i)

    # Ladder relations.
    A isa SpinZ && B isa SpinPlus  && return hbar * Sp(i)
    A isa SpinPlus && B isa SpinZ  && return -hbar * Sp(i)
    A isa SpinZ && B isa SpinMinus && return -hbar * Sm(i)
    A isa SpinMinus && B isa SpinZ && return hbar * Sm(i)
    A isa SpinPlus && B isa SpinMinus && return (2 * hbar) * Sz(i)
    A isa SpinMinus && B isa SpinPlus && return (-2 * hbar) * Sz(i)

    return nothing
end

# Coordinate-momentum algebra.
function primitive_commutator(
    ::CoordinateMomentumAlgebra,
    A::Union{PositionOperator,MomentumOperator},
    B::Union{PositionOperator,MomentumOperator};
    hbar::Number = 1,
)
    A.mode == B.mode || return _zero_operator()

    A isa PositionOperator && B isa MomentumOperator &&
        return (im * hbar) * IdentityOperator()
    A isa MomentumOperator && B isa PositionOperator &&
        return (-im * hbar) * IdentityOperator()

    return _zero_operator()
end

# Composite tensor-product algebras.  Operators belonging to distinct algebra
# factors commute.  Operators in the same factor use that factor's relations.
function primitive_commutator(
    algebra::CompositeAlgebra,
    A::AbstractPrimitiveOperator,
    B::AbstractPrimitiveOperator;
    hbar::Number = 1,
)
    subA = _matching_subalgebra(algebra, A)
    subB = _matching_subalgebra(algebra, B)

    (subA === nothing || subB === nothing) && return nothing
    subA === subB || return _zero_operator()
    return primitive_commutator(subA, A, B; hbar=hbar)
end

# ---------------------------------------------------------------------------
# Primitive anticommutators
# ---------------------------------------------------------------------------

"""
    primitive_anticommutator(algebra, A, B; hbar=1)

Return a directly known primitive anticommutator, or `nothing` when no
primitive relation is defined.
"""
primitive_anticommutator(
    ::AbstractOperatorAlgebra,
    ::AbstractOperatorExpr,
    ::AbstractOperatorExpr;
    hbar::Number = 1,
) = nothing

# Fermionic CAR.
function primitive_anticommutator(
    ::FermionAlgebra,
    A::Union{FermionAnnihilate,FermionCreate},
    B::Union{FermionAnnihilate,FermionCreate};
    hbar::Number = 1,
)
    if A isa FermionAnnihilate && B isa FermionCreate
        return A.mode == B.mode ? IdentityOperator() : _zero_operator()
    elseif A isa FermionCreate && B isa FermionAnnihilate
        return A.mode == B.mode ? IdentityOperator() : _zero_operator()
    end
    return _zero_operator()
end

# Majorana Clifford algebra: {gamma_mu,gamma_nu} = 2 delta_mu_nu I.
function primitive_anticommutator(
    ::MajoranaAlgebra,
    A::MajoranaOperator,
    B::MajoranaOperator;
    hbar::Number = 1,
)
    return A.index == B.index ? 2 * IdentityOperator() : _zero_operator()
end

# Number-operator commutators for fermions.  The number operator is even and
# obeys the same number-changing commutator as in the bosonic case.
function primitive_commutator(
    ::FermionAlgebra,
    A::NumberOperator,
    B::Union{FermionAnnihilate,FermionCreate};
    hbar::Number = 1,
)
    A.kind == :fermion || return nothing
    A.mode == B.mode || return _zero_operator()
    return B isa FermionCreate ? B : -B
end

function primitive_commutator(
    algebra::FermionAlgebra,
    A::Union{FermionAnnihilate,FermionCreate},
    B::NumberOperator;
    hbar::Number = 1,
)
    value = primitive_commutator(algebra, B, A; hbar=hbar)
    return value === nothing ? nothing : -value
end

function primitive_anticommutator(
    algebra::CompositeAlgebra,
    A::AbstractPrimitiveOperator,
    B::AbstractPrimitiveOperator;
    hbar::Number = 1,
)
    subA = _matching_subalgebra(algebra, A)
    subB = _matching_subalgebra(algebra, B)
    (subA === nothing || subB === nothing || subA !== subB) && return nothing
    return primitive_anticommutator(subA, A, B; hbar=hbar)
end

# ---------------------------------------------------------------------------
# Tensor-factor-qualified primitives
# ---------------------------------------------------------------------------

operator_family(op::FactorOperator) = operator_family(op.operator)

function _factor_subalgebra(algebra::CompositeAlgebra, factor::Int)
    1 <= factor <= length(algebra.factors) || throw(BoundsError(algebra.factors, factor))
    return algebra.factors[factor]
end

function _qualify_factor_result(factor::Int, expr::AbstractOperatorExpr)
    expr isa IdentityOperator && return expr
    expr isa AbstractPrimitiveOperator && return FactorOperator(factor, expr)
    expr isa ScaledOperator && return ScaledOperator(expr.coefficient, _qualify_factor_result(factor, expr.operator))
    expr isa OperatorSum && return OperatorSum(AbstractOperatorExpr[_qualify_factor_result(factor, term) for term in expr.terms])
    expr isa OperatorProduct && return OperatorProduct(AbstractOperatorExpr[_qualify_factor_result(factor, term) for term in expr.factors])
    return expr
end

function primitive_commutator(algebra::CompositeAlgebra, A::FactorOperator, B::FactorOperator; hbar::Number=1)
    if A.factor == B.factor
        result = primitive_commutator(_factor_subalgebra(algebra, A.factor), A.operator, B.operator; hbar=hbar)
        return isnothing(result) ? nothing : _qualify_factor_result(A.factor, result)
    end
    fa = operator_family(A.operator)
    fb = operator_family(B.operator)
    if fa in (:fermion, :majorana) && fb in (:fermion, :majorana)
        return nothing
    end
    return _zero_operator()
end

function primitive_anticommutator(algebra::CompositeAlgebra, A::FactorOperator, B::FactorOperator; hbar::Number=1)
    if A.factor == B.factor
        result = primitive_anticommutator(_factor_subalgebra(algebra, A.factor), A.operator, B.operator; hbar=hbar)
        return isnothing(result) ? nothing : _qualify_factor_result(A.factor, result)
    end
    fa = operator_family(A.operator)
    fb = operator_family(B.operator)
    if fa in (:fermion, :majorana) && fb in (:fermion, :majorana)
        return _zero_operator()
    end
    return nothing
end
