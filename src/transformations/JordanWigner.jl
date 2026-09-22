# Exact ordered spin-1/2 <-> spinless-fermion representation map.

struct JordanWigner <: AbstractTransformation
    ordering::Vector{Int}
    hbar::Float64
end

JordanWigner(ordering; hbar::Real=1.0) = JordanWigner(collect(Int, ordering), Float64(hbar))

transformation_name(::JordanWigner) = :jordan_wigner
transformation_hbar(T::JordanWigner) = T.hbar

certificate(::JordanWigner) = TransformationCertificate(
    true,
    true,
    AlgebraIsomorphism,
    :nonlocal;
    notes=("ordered spin-1/2 to spinless-fermion map",),
)

function applicable(T::JordanWigner, model::ManyBodyModel)
    R = model.representation
    R.algebra isa SpinAlgebra || return false
    R.state_space isa SpinHilbertSpace || return false
    all(S -> S == 1//2 || S == 0.5, R.state_space.spins) || return false
    return sort(T.ordering) == collect(1:R.algebra.nsites)
end

function target_representation(T::JordanWigner, model::ManyBodyModel)
    N = length(T.ordering)

    return Representation(
        :jordan_wigner_fermion,
        FermionFockSpace(N),
        FermionAlgebra(N),
        FermionOccupationBasis(copy(T.ordering));
        ordering=copy(T.ordering),
        reference_state=:fermion_vacuum,
    )
end


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------

@inline function _jw_position(T::JordanWigner, site::Int)
    @inbounds for position in eachindex(T.ordering)
        T.ordering[position] == site && return position
    end

    throw(ArgumentError("site $site is not in Jordan-Wigner ordering"))
end

function _jw_preceding(T::JordanWigner, site::Int)
    position = _jw_position(T, site)
    return @view T.ordering[1:position-1]
end

function _jw_string(T::JordanWigner, site::Int)
    preceding = _jw_preceding(T, site)
    isempty(preceding) && return IdentityOperator()

    factors = Vector{AbstractOperatorExpr}(undef, length(preceding))

    @inbounds for i in eachindex(preceding)
        factors[i] = IdentityOperator() - 2 * nf(preceding[i])
    end

    return OperatorProduct(factors)
end


# ---------------------------------------------------------------------------
# Primitive Jordan-Wigner map
# ---------------------------------------------------------------------------

transform_generator(T::JordanWigner, op::SpinZ) = T.hbar * (nf(op.site) - (1/2) * IdentityOperator())

transform_generator(T::JordanWigner, op::SpinPlus) = T.hbar * c(op.site)' * _jw_string(T, op.site)

transform_generator(T::JordanWigner, op::SpinMinus) = T.hbar * _jw_string(T, op.site) * c(op.site)

transform_generator(T::JordanWigner, op::SpinX) = (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))

transform_generator(T::JordanWigner, op::SpinY) = (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))


# ---------------------------------------------------------------------------
# Adjacent transverse-spin fast path
#
# The generic primitive-by-primitive map constructs
#
#     P_i = prod_{m<i} (1 - 2n_m)
#     P_j = prod_{m<j} (1 - 2n_m)
#
# separately and lets canonicalization rediscover that their common prefix
# squares to the identity.
#
# For adjacent sites in the Jordan-Wigner ordering,
#
#     P_j = P_i (1 - 2n_i)
#
# and therefore
#
#     P_i P_j = 1 - 2n_i.
#
# The cancellation can be performed exactly before either parity string is
# constructed.
# ---------------------------------------------------------------------------

const _JWTransversePrimitive = Union{SpinX, SpinY, SpinPlus, SpinMinus}

@inline function _jw_unwrap_scalar(expr::AbstractOperatorExpr)
    coefficient::Number = 1
    core = expr

    while core isa ScaledOperator
        coefficient *= core.coefficient
        core = core.operator
    end

    return coefficient, core
end

# Write a transverse spin operator in the form
#
#     S_i = hbar * alpha * P_i * (u*c_i^dagger + v*c_i).
#
@inline _jw_transverse_data(::SpinX) = (1/2, 1, 1)
@inline _jw_transverse_data(::SpinY) = (1/(2im), 1, -1)
@inline _jw_transverse_data(::SpinPlus) = (1, 1, 0)
@inline _jw_transverse_data(::SpinMinus) = (1, 0, 1)

@inline function _jw_linear_fermion(site::Int, u::Number, v::Number)
    if iszero(u)
        return v * c(site)
    elseif iszero(v)
        return u * c(site)'
    else
        return u * c(site)' + v * c(site)
    end
end

function _jw_adjacent_transverse_product(
    T::JordanWigner,
    left::_JWTransversePrimitive,
    right::_JWTransversePrimitive,
)
    alpha_left, u_left, v_left = _jw_transverse_data(left)
    alpha_right, u_right, v_right = _jw_transverse_data(right)

    # For the earlier site,
    #
    #     (u*c† + v*c)(1 - 2n) = u*c† - v*c.
    #
    left_fermion = _jw_linear_fermion(left.site, u_left, -v_left)
    right_fermion = _jw_linear_fermion(right.site, u_right, v_right)

    coefficient = T.hbar^2 * alpha_left * alpha_right

    return coefficient * left_fermion * right_fermion
end

function _jw_try_adjacent_transverse_product(T::JordanWigner, expr::OperatorProduct)
    length(expr.factors) == 2 || return nothing

    coefficient_left, left = _jw_unwrap_scalar(expr.factors[1])
    coefficient_right, right = _jw_unwrap_scalar(expr.factors[2])

    left isa _JWTransversePrimitive || return nothing
    right isa _JWTransversePrimitive || return nothing
    left.site == right.site && return nothing

    position_left = _jw_position(T, left.site)
    position_right = _jw_position(T, right.site)

    abs(position_left - position_right) == 1 || return nothing

    mapped = if position_left < position_right
        _jw_adjacent_transverse_product(T, left, right)
    else
        # Spin operators on different sites commute, so put the sites into
        # Jordan-Wigner ordering before applying the exact parity cancellation.
        _jw_adjacent_transverse_product(T, right, left)
    end

    return (coefficient_left * coefficient_right) * mapped
end


# ---------------------------------------------------------------------------
# Product-level Jordan-Wigner transformation
#
# This method is more specific than the generic
#
#     transform_operator(::AbstractTransformation, ::OperatorProduct)
#
# method. Adjacent transverse-spin products therefore bypass construction of
# the individual Jordan-Wigner strings entirely.
# ---------------------------------------------------------------------------

function transform_operator(
    T::JordanWigner,
    expr::OperatorProduct,
)
    fast = _jw_try_adjacent_transverse_product(T, expr)
    fast === nothing || return fast

    return OperatorProduct(AbstractOperatorExpr[transform_operator(T, factor) for factor in expr.factors])
end


# ---------------------------------------------------------------------------
# Term-local Hamiltonian canonicalization
#
# Transform and canonicalize each Hamiltonian term independently before
# assembling the final sum. This prevents intermediate expressions from
# unrelated Hamiltonian terms from participating in the same fixed-point
# canonicalization pass.
# ---------------------------------------------------------------------------

function _jw_transform_sum_canonical(
    T::JordanWigner,
    target::Representation,
    expr::OperatorSum,
)
    terms = AbstractOperatorExpr[]
    sizehint!(terms, length(expr.terms))

    for term in expr.terms
        mapped = transform_operator(T, term)
        canonical = canonicalize_transformed(T, target, mapped)

        if canonical isa OperatorSum
            append!(terms, canonical.terms)
        else
            push!(terms, canonical)
        end
    end

    return canonicalize_transformed(T, target, OperatorSum(terms))
end

transform_hamiltonian(T::JordanWigner, target::Representation, H::OperatorSum) = _jw_transform_sum_canonical(T, target, H)

transform_observable(T::JordanWigner, target::Representation, O::OperatorSum) = _jw_transform_sum_canonical(T, target, O)
