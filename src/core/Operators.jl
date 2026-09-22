# Noncommutative operator expression system.
#
# The package deliberately owns this small expression tree instead of assuming
# ordinary commutative symbolic multiplication.  Symbolic scalar coefficients
# remain generic and may later be Float64, ComplexF64, Symbolics.Num, etc.

abstract type AbstractOperatorExpr end
abstract type AbstractPrimitiveOperator <: AbstractOperatorExpr end

"""Identity operator, optionally tagged by a state-space/algebra label."""
struct IdentityOperator <: AbstractPrimitiveOperator
    label::Symbol
end
IdentityOperator() = IdentityOperator(:I)

# Spin generators.
struct SpinX <: AbstractPrimitiveOperator
    site::Int
end
struct SpinY <: AbstractPrimitiveOperator
    site::Int
end
struct SpinZ <: AbstractPrimitiveOperator
    site::Int
end
struct SpinPlus <: AbstractPrimitiveOperator
    site::Int
end
struct SpinMinus <: AbstractPrimitiveOperator
    site::Int
end

# Bosonic generators.
struct BosonAnnihilate <: AbstractPrimitiveOperator
    mode::Int
end
struct BosonCreate <: AbstractPrimitiveOperator
    mode::Int
end

# Fermionic generators.
struct FermionAnnihilate <: AbstractPrimitiveOperator
    mode::Int
end
struct FermionCreate <: AbstractPrimitiveOperator
    mode::Int
end

# Majorana, coordinate, and momentum generators.
struct MajoranaOperator <: AbstractPrimitiveOperator
    index::Int
end
struct PositionOperator <: AbstractPrimitiveOperator
    mode::Int
end
struct MomentumOperator <: AbstractPrimitiveOperator
    mode::Int
end

"""
    NumberOperator(kind, mode)

Derived occupation operator.  `kind` is conventionally `:boson`, `:fermion`,
or another representation-specific tag.
"""
struct NumberOperator <: AbstractPrimitiveOperator
    kind::Symbol
    mode::Int
end

"""Ordered sum of noncommutative operator expressions."""
struct OperatorSum <: AbstractOperatorExpr
    terms::Vector{AbstractOperatorExpr}

    function OperatorSum(terms::AbstractVector{<:AbstractOperatorExpr})
        flat = AbstractOperatorExpr[]
        for term in terms
            if term isa OperatorSum
                append!(flat, term.terms)
            else
                push!(flat, term)
            end
        end
        new(flat)
    end
end

"""Ordered noncommutative product of operator expressions."""
struct OperatorProduct <: AbstractOperatorExpr
    factors::Vector{AbstractOperatorExpr}

    function OperatorProduct(factors::AbstractVector{<:AbstractOperatorExpr})
        flat = AbstractOperatorExpr[]
        for factor in factors
            if factor isa OperatorProduct
                append!(flat, factor.factors)
            else
                push!(flat, factor)
            end
        end
        new(flat)
    end
end

"""Scalar coefficient multiplying a noncommutative operator expression."""
struct ScaledOperator{C,O<:AbstractOperatorExpr} <: AbstractOperatorExpr
    coefficient::C
    operator::O
end

# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

Sx(i::Integer) = SpinX(Int(i))
Sy(i::Integer) = SpinY(Int(i))
Sz(i::Integer) = SpinZ(Int(i))
Sp(i::Integer) = SpinPlus(Int(i))
Sm(i::Integer) = SpinMinus(Int(i))

b(i::Integer) = BosonAnnihilate(Int(i))
c(i::Integer) = FermionAnnihilate(Int(i))

# Legacy compatibility aliases. Creation operators are canonically written as b(i)' and c(i)' and these aliases are intentionally not exported.
bd(i::Integer) = b(i)'
cd(i::Integer) = c(i)'
γ(i::Integer)  = MajoranaOperator(Int(i))
xop(i::Integer) = PositionOperator(Int(i))
pop(i::Integer) = MomentumOperator(Int(i))
nb(i::Integer) = NumberOperator(:boson, Int(i))
nf(i::Integer) = NumberOperator(:fermion, Int(i))

# ---------------------------------------------------------------------------
# Algebraic construction
# ---------------------------------------------------------------------------

Base.:+(a::AbstractOperatorExpr, b::AbstractOperatorExpr) =
    OperatorSum(AbstractOperatorExpr[a, b])

Base.:-(a::AbstractOperatorExpr) = ScaledOperator(-1, a)
Base.:-(a::AbstractOperatorExpr, b::AbstractOperatorExpr) = a + (-b)

Base.:*(a::AbstractOperatorExpr, b::AbstractOperatorExpr) =
    OperatorProduct(AbstractOperatorExpr[a, b])

Base.:*(α::Number, op::AbstractOperatorExpr) = ScaledOperator(α, op)
Base.:*(op::AbstractOperatorExpr, α::Number) = ScaledOperator(α, op)
Base.:/(op::AbstractOperatorExpr, α::Number) = ScaledOperator(inv(α), op)

# Addition of scalar constants is represented as a scalar times identity.
Base.:+(α::Number, op::AbstractOperatorExpr) = α * IdentityOperator() + op
Base.:+(op::AbstractOperatorExpr, α::Number) = op + α * IdentityOperator()
Base.:-(op::AbstractOperatorExpr, α::Number) = op + (-α) * IdentityOperator()
Base.:-(α::Number, op::AbstractOperatorExpr) = α * IdentityOperator() - op

# ---------------------------------------------------------------------------
# Hermitian conjugation
# ---------------------------------------------------------------------------

Base.adjoint(op::IdentityOperator) = op
Base.adjoint(op::SpinX) = op
Base.adjoint(op::SpinY) = op
Base.adjoint(op::SpinZ) = op
Base.adjoint(op::SpinPlus) = SpinMinus(op.site)
Base.adjoint(op::SpinMinus) = SpinPlus(op.site)
Base.adjoint(op::BosonAnnihilate) = BosonCreate(op.mode)
Base.adjoint(op::BosonCreate) = BosonAnnihilate(op.mode)
Base.adjoint(op::FermionAnnihilate) = FermionCreate(op.mode)
Base.adjoint(op::FermionCreate) = FermionAnnihilate(op.mode)
Base.adjoint(op::MajoranaOperator) = op
Base.adjoint(op::PositionOperator) = op
Base.adjoint(op::MomentumOperator) = op
Base.adjoint(op::NumberOperator) = op
Base.adjoint(op::ScaledOperator) = ScaledOperator(conj(op.coefficient), adjoint(op.operator))
Base.adjoint(op::OperatorSum) = OperatorSum(AbstractOperatorExpr[adjoint(x) for x in op.terms])
Base.adjoint(op::OperatorProduct) =
    OperatorProduct(AbstractOperatorExpr[adjoint(x) for x in reverse(op.factors)])

# ---------------------------------------------------------------------------
# Basic introspection
# ---------------------------------------------------------------------------

isprimitive(::AbstractOperatorExpr) = false
isprimitive(::AbstractPrimitiveOperator) = true

children(::AbstractPrimitiveOperator) = ()
children(op::ScaledOperator) = (op.operator,)
children(op::OperatorSum) = op.terms
children(op::OperatorProduct) = op.factors

# Lightweight display methods make REPL inspection readable without imposing a
# full symbolic pretty-printer on Core.
function Base.show(io::IO, op::IdentityOperator)
    print(io, op.label)
end
Base.show(io::IO, op::SpinX) = print(io, "Sx(", op.site, ")")
Base.show(io::IO, op::SpinY) = print(io, "Sy(", op.site, ")")
Base.show(io::IO, op::SpinZ) = print(io, "Sz(", op.site, ")")
Base.show(io::IO, op::SpinPlus) = print(io, "S+(", op.site, ")")
Base.show(io::IO, op::SpinMinus) = print(io, "S-(", op.site, ")")
Base.show(io::IO, op::BosonAnnihilate) = print(io, "b(", op.mode, ")")
Base.show(io::IO, op::BosonCreate) = print(io, "b†(", op.mode, ")")
Base.show(io::IO, op::FermionAnnihilate) = print(io, "c(", op.mode, ")")
Base.show(io::IO, op::FermionCreate) = print(io, "c†(", op.mode, ")")
Base.show(io::IO, op::MajoranaOperator) = print(io, "γ(", op.index, ")")
Base.show(io::IO, op::PositionOperator) = print(io, "x(", op.mode, ")")
Base.show(io::IO, op::MomentumOperator) = print(io, "p(", op.mode, ")")
Base.show(io::IO, op::NumberOperator) = print(io, "n_", op.kind, "(", op.mode, ")")

function Base.show(io::IO, op::ScaledOperator)
    print(io, "(", op.coefficient, ")*")
    show(io, op.operator)
end

function Base.show(io::IO, op::OperatorSum)
    print(io, "(")
    for (i, term) in enumerate(op.terms)
        i > 1 && print(io, " + ")
        show(io, term)
    end
    print(io, ")")
end

function Base.show(io::IO, op::OperatorProduct)
    print(io, "(")
    for (i, factor) in enumerate(op.factors)
        i > 1 && print(io, " * ")
        show(io, factor)
    end
    print(io, ")")
end

# ---------------------------------------------------------------------------
# Tensor-factor-qualified primitive operators
# ---------------------------------------------------------------------------

"""
    FactorOperator(factor, operator)
    atfactor(factor, operator)

Address a primitive operator to one explicit tensor factor of a composite
representation. This removes the ambiguity of expressions such as `c(1)` when
multiple tensor factors carry the same operator family.
"""
struct FactorOperator{O<:AbstractPrimitiveOperator} <: AbstractPrimitiveOperator
    factor::Int
    operator::O
    function FactorOperator(factor::Integer, operator::O) where {O<:AbstractPrimitiveOperator}
        factor > 0 || throw(ArgumentError("tensor-factor index must be positive"))
        new{O}(Int(factor), operator)
    end
end

atfactor(factor::Integer, operator::AbstractPrimitiveOperator) = FactorOperator(factor, operator)

Base.adjoint(op::FactorOperator) = FactorOperator(op.factor, adjoint(op.operator))
children(op::FactorOperator) = (op.operator,)
isprimitive(::FactorOperator) = true

function Base.show(io::IO, op::FactorOperator)
    print(io, "factor[", op.factor, "]::")
    show(io, op.operator)
end

# ---------------------------------------------------------------------------
# Operator complexity and algebra-aware many-body rank
# ---------------------------------------------------------------------------

"""
    operator_degree(expression)

Return the syntactic degree of an operator expression, defined as the largest number of primitive operator factors in any additive term. This quantity is representation dependent and is distinct from physical many-body rank.
"""
operator_degree(::IdentityOperator) = 0
operator_degree(::AbstractPrimitiveOperator) = 1
operator_degree(expression::ScaledOperator) = operator_degree(expression.operator)
operator_degree(expression::OperatorSum) = maximum(operator_degree(term) for term in expression.terms; init=0)
operator_degree(expression::OperatorProduct) = sum(operator_degree(factor) for factor in expression.factors; init=0)

_primitive_support_key(::IdentityOperator) = nothing
_primitive_support_key(operator::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}) = (:spin, operator.site)
_primitive_support_key(operator::Union{BosonAnnihilate,BosonCreate}) = (:boson, operator.mode)
_primitive_support_key(operator::Union{FermionAnnihilate,FermionCreate}) = (:fermion, operator.mode)
_primitive_support_key(operator::MajoranaOperator) = (:majorana, operator.index)
_primitive_support_key(operator::Union{PositionOperator,MomentumOperator}) = (:coordinate_momentum, operator.mode)
_primitive_support_key(operator::NumberOperator) = (operator.kind, operator.mode)
_primitive_support_key(operator::FactorOperator) = begin
    key = _primitive_support_key(operator.operator)
    isnothing(key) ? nothing : (:factor, operator.factor, key)
end

function _support_keys(expression::AbstractPrimitiveOperator)
    key = _primitive_support_key(expression)
    return isnothing(key) ? Set{Any}() : Set{Any}((key,))
end

_support_keys(expression::ScaledOperator) = _support_keys(expression.operator)
function _support_keys(expression::OperatorSum)
    keys = Set{Any}()
    for term in expression.terms
        union!(keys, _support_keys(term))
    end
    return keys
end

function _support_keys(expression::OperatorProduct)
    keys = Set{Any}()
    for factor in expression.factors
        union!(keys, _support_keys(factor))
    end
    return keys
end

"""
    support_size(expression)

Return the maximum number of distinct local supports acted on by any additive term. `support_size` is useful for diagnostics but is not identical to many-body rank for particle algebras.
"""
support_size(expression::AbstractPrimitiveOperator) = length(_support_keys(expression))
support_size(expression::ScaledOperator) = support_size(expression.operator)
support_size(expression::OperatorProduct) = length(_support_keys(expression))
support_size(expression::OperatorSum) = maximum(support_size(term) for term in expression.terms; init=0)

_operator_kind(::IdentityOperator) = :identity
_operator_kind(::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}) = :spin
_operator_kind(::Union{BosonAnnihilate,BosonCreate}) = :boson
_operator_kind(::Union{FermionAnnihilate,FermionCreate}) = :fermion
_operator_kind(::MajoranaOperator) = :majorana
_operator_kind(::Union{PositionOperator,MomentumOperator}) = :coordinate_momentum
_operator_kind(operator::NumberOperator) = operator.kind
_operator_kind(operator::FactorOperator) = _operator_kind(operator.operator)

_particle_legs(::IdentityOperator, ::Symbol) = 0
_particle_legs(operator::NumberOperator, kind::Symbol) = operator.kind == kind ? 2 : throw(ArgumentError("number operator $(operator.kind) is incompatible with $kind algebra"))
_particle_legs(::Union{BosonAnnihilate,BosonCreate}, kind::Symbol) = kind == :boson ? 1 : throw(ArgumentError("bosonic operator is incompatible with $kind algebra"))
_particle_legs(::Union{FermionAnnihilate,FermionCreate}, kind::Symbol) = kind == :fermion ? 1 : throw(ArgumentError("fermionic operator is incompatible with $kind algebra"))
_particle_legs(expression::ScaledOperator, kind::Symbol) = _particle_legs(expression.operator, kind)
_particle_legs(expression::OperatorSum, kind::Symbol) = maximum(_particle_legs(term, kind) for term in expression.terms; init=0)
_particle_legs(expression::OperatorProduct, kind::Symbol) = sum(_particle_legs(factor, kind) for factor in expression.factors; init=0)

_spin_support(::IdentityOperator) = Set{Int}()
_spin_support(operator::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}) = Set((operator.site,))
_spin_support(expression::ScaledOperator) = _spin_support(expression.operator)
function _spin_support(expression::OperatorSum)
    sites = Set{Int}()
    for term in expression.terms
        union!(sites, _spin_support(term))
    end
    return sites
end
function _spin_support(expression::OperatorProduct)
    sites = Set{Int}()
    for factor in expression.factors
        union!(sites, _spin_support(factor))
    end
    return sites
end

_coordinate_support(::IdentityOperator) = Set{Int}()
_coordinate_support(operator::Union{PositionOperator,MomentumOperator}) = Set((operator.mode,))
_coordinate_support(expression::ScaledOperator) = _coordinate_support(expression.operator)
function _coordinate_support(expression::OperatorSum)
    modes = Set{Int}()
    for term in expression.terms
        union!(modes, _coordinate_support(term))
    end
    return modes
end
function _coordinate_support(expression::OperatorProduct)
    modes = Set{Int}()
    for factor in expression.factors
        union!(modes, _coordinate_support(factor))
    end
    return modes
end

_majorana_legs(::IdentityOperator) = 0
_majorana_legs(::MajoranaOperator) = 1
_majorana_legs(expression::ScaledOperator) = _majorana_legs(expression.operator)
_majorana_legs(expression::OperatorSum) = maximum(_majorana_legs(term) for term in expression.terms; init=0)
_majorana_legs(expression::OperatorProduct) = sum(_majorana_legs(factor) for factor in expression.factors; init=0)

function _composite_factor_index(algebra::CompositeAlgebra, operator::AbstractPrimitiveOperator)
    operator isa IdentityOperator && return 0
    operator isa FactorOperator && return operator.factor
    kind = _operator_kind(operator)
    matches = findall(factor -> algebra_kind(factor) == kind, algebra.factors)
    isempty(matches) && throw(ArgumentError("operator family $kind does not occur in the composite algebra"))
    length(matches) == 1 || throw(ArgumentError("operator family $kind occurs in multiple tensor factors; use atfactor(factor, operator)"))
    return only(matches)
end

_unwrap_factor(operator::FactorOperator) = operator.operator
_unwrap_factor(operator::AbstractPrimitiveOperator) = operator

function _body_order_composite_term(algebra::CompositeAlgebra, expression::AbstractOperatorExpr)
    factors = expression isa OperatorProduct ? expression.factors : AbstractOperatorExpr[expression]
    grouped = Dict{Int,Vector{AbstractOperatorExpr}}()
    for factor in factors
        factor isa AbstractPrimitiveOperator || throw(ArgumentError("canonical composite body-order analysis requires primitive product factors"))
        index = _composite_factor_index(algebra, factor)
        index == 0 && continue
        1 <= index <= length(algebra.factors) || throw(BoundsError(algebra.factors, index))
        push!(get!(grouped, index, AbstractOperatorExpr[]), _unwrap_factor(factor))
    end
    order = 0
    for (index, local_factors) in grouped
        local_expression = length(local_factors) == 1 ? only(local_factors) : OperatorProduct(local_factors)
        order += body_order(algebra.factors[index], local_expression)
    end
    return order
end

"""
    body_order(algebra, expression)

Return the algebra-aware many-body rank of an operator expression. For fermionic, bosonic, and Majorana algebras, rank is the ceiling of the primitive field-leg count divided by two, so quadratic field monomials are one-body and quartic monomials are two-body. For spin and coordinate-momentum algebras, rank counts distinct local supports. Composite-algebra ranks add across tensor factors. The function is structural and does not perform algebraic rewriting; canonicalize reducible expressions before calling it when exact simplification matters.
"""
body_order(::AbstractOperatorAlgebra, ::IdentityOperator) = 0
body_order(::CompositeAlgebra, ::IdentityOperator) = 0
body_order(algebra::AbstractOperatorAlgebra, expression::ScaledOperator) = body_order(algebra, expression.operator)
body_order(algebra::AbstractOperatorAlgebra, expression::OperatorSum) = maximum(body_order(algebra, term) for term in expression.terms; init=0)
body_order(::SpinAlgebra, expression::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}) = 1
body_order(::SpinAlgebra, expression::OperatorProduct) = length(_spin_support(expression))
body_order(::CoordinateMomentumAlgebra, expression::Union{PositionOperator,MomentumOperator}) = 1
body_order(::CoordinateMomentumAlgebra, expression::OperatorProduct) = length(_coordinate_support(expression))
body_order(::BosonAlgebra, expression::Union{BosonAnnihilate,BosonCreate,NumberOperator}) = cld(_particle_legs(expression, :boson), 2)
body_order(::BosonAlgebra, expression::OperatorProduct) = cld(_particle_legs(expression, :boson), 2)
body_order(::FermionAlgebra, expression::Union{FermionAnnihilate,FermionCreate,NumberOperator}) = cld(_particle_legs(expression, :fermion), 2)
body_order(::FermionAlgebra, expression::OperatorProduct) = cld(_particle_legs(expression, :fermion), 2)
body_order(::MajoranaAlgebra, ::MajoranaOperator) = 1
body_order(::MajoranaAlgebra, expression::OperatorProduct) = cld(_majorana_legs(expression), 2)
body_order(algebra::CompositeAlgebra, expression::Union{AbstractPrimitiveOperator,OperatorProduct}) = _body_order_composite_term(algebra, expression)
