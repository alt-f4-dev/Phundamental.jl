# Linear bosonic/fermionic canonical transformation.

struct BogoliubovTransformation{T<:Number,M1<:AbstractMatrix{T},M2<:AbstractMatrix{T}} <: AbstractTransformation
    U::M1
    V::M2
    statistics::Symbol
    atol::Float64

    function BogoliubovTransformation(
        U::AbstractMatrix{T},
        V::AbstractMatrix{T};
        statistics::Symbol=:boson,
        atol::Real=1e-10,
    ) where {T<:Number}
        size(U) == size(V) || throw(ArgumentError("U and V must have equal size"))
        size(U, 1) == size(U, 2) || throw(ArgumentError("U and V must be square"))
        statistics in (:boson, :fermion) || throw(ArgumentError("statistics must be :boson or :fermion"))
        Uc, Vc = Matrix{T}(U), Matrix{T}(V)
        Iu = Matrix{T}(I, size(Uc, 1), size(Uc, 1))
        if statistics == :boson
            isapprox(Uc * adjoint(Uc) - Vc * adjoint(Vc), Iu; atol=atol, rtol=0) ||
                throw(ArgumentError("bosonic Bogoliubov condition U*U†-V*V†=I failed"))
            isapprox(Uc * transpose(Vc) - Vc * transpose(Uc), zero(Iu); atol=atol, rtol=0) ||
                throw(ArgumentError("bosonic Bogoliubov condition U*Vᵀ-V*Uᵀ=0 failed"))
        else
            isapprox(Uc * adjoint(Uc) + Vc * adjoint(Vc), Iu; atol=atol, rtol=0) ||
                throw(ArgumentError("fermionic Bogoliubov condition U*U†+V*V†=I failed"))
            isapprox(Uc * transpose(Vc) + Vc * transpose(Uc), zero(Iu); atol=atol, rtol=0) ||
                throw(ArgumentError("fermionic Bogoliubov condition U*Vᵀ+V*Uᵀ=0 failed"))
        end
        new{T,typeof(Uc),typeof(Vc)}(Uc, Vc, statistics, Float64(atol))
    end
end

transformation_name(::BogoliubovTransformation) = :bogoliubov
certificate(::BogoliubovTransformation) = TransformationCertificate(
    true,
    true,
    CanonicalMap,
    :nonlocal;
    notes=("linear canonical quasiparticle transformation",),
)

function applicable(T::BogoliubovTransformation, model::ManyBodyModel)
    A = model.representation.algebra
    M = size(T.U, 1)
    return (T.statistics == :boson && A isa BosonAlgebra && A.nmodes == M) ||
           (T.statistics == :fermion && A isa FermionAlgebra && A.nmodes == M)
end

function target_representation(T::BogoliubovTransformation, model::ManyBodyModel)
    M = size(T.U, 1)
    if T.statistics == :boson
        space = BosonFockSpace(M)
        algebra = BosonAlgebra(M)
        physical, constraints = _transported_physical_subspace(T, model.representation, space)
        gauge = _transported_gauge(T, model.representation, algebra)
        return Representation(
            :bosonic_quasiparticle,
            space,
            algebra,
            BosonOccupationBasis(collect(1:M));
            physical_subspace=physical,
            constraints=constraints,
            gauge_structure=gauge,
            reference_state=:bogoliubov_vacuum,
            ordering=collect(1:M),
            truncation=_transported_truncation(T, model.representation),
        )
    else
        space = FermionFockSpace(M)
        algebra = FermionAlgebra(M)
        physical, constraints = _transported_physical_subspace(T, model.representation, space)
        gauge = _transported_gauge(T, model.representation, algebra)
        return Representation(
            :fermionic_quasiparticle,
            space,
            algebra,
            FermionOccupationBasis(collect(1:M));
            physical_subspace=physical,
            constraints=constraints,
            gauge_structure=gauge,
            reference_state=:bogoliubov_vacuum,
            ordering=collect(1:M),
            truncation=_transported_truncation(T, model.representation),
        )
    end
end

# Source generators are expressed in target quasiparticle generators using the
# inverse canonical map.
function transform_generator(T::BogoliubovTransformation, op::BosonAnnihilate)
    T.statistics == :boson || throw(ArgumentError("fermionic Bogoliubov transform cannot map bosons"))
    j = op.mode
    M = size(T.U, 1)
    return OperatorSum(AbstractOperatorExpr[
        [conj(T.U[k, j]) * b(k) for k in 1:M]...,
        [-T.V[k, j] * b(k)' for k in 1:M]...,
    ])
end

function transform_generator(T::BogoliubovTransformation, op::BosonCreate)
    return adjoint(transform_generator(T, BosonAnnihilate(op.mode)))
end

function transform_generator(T::BogoliubovTransformation, op::FermionAnnihilate)
    T.statistics == :fermion || throw(ArgumentError("bosonic Bogoliubov transform cannot map fermions"))
    j = op.mode
    M = size(T.U, 1)
    return OperatorSum(AbstractOperatorExpr[
        [conj(T.U[k, j]) * c(k) for k in 1:M]...,
        [T.V[k, j] * c(k)' for k in 1:M]...,
    ])
end

function transform_generator(T::BogoliubovTransformation, op::FermionCreate)
    return adjoint(transform_generator(T, FermionAnnihilate(op.mode)))
end

transform_generator(T::BogoliubovTransformation, op::NumberOperator) =
    op.kind == :boson && T.statistics == :boson ?
        transform_operator(T, b(op.mode)' * b(op.mode)) :
    op.kind == :fermion && T.statistics == :fermion ?
        transform_operator(T, c(op.mode)' * c(op.mode)) :
        throw(ArgumentError("number operator is outside Bogoliubov domain"))

# ---------------------------------------------------------------------------
# Quadratic fast path
# ---------------------------------------------------------------------------

# The generic operator map expands every source generator into a sum of M
# quasiparticle generators. A quadratic monomial therefore creates O(M^2)
# symbolic terms before canonicalization. For a quadratic Hamiltonian this is
# unnecessary: the same exact substitution can be performed on its Nambu
# quadratic form using dense matrix multiplication, followed by direct
# reconstruction of a canonical symbolic quadratic Hamiltonian.

function _bogoliubov_monomial(expr::AbstractOperatorExpr)
    if expr isa IdentityOperator
        return 1, AbstractPrimitiveOperator[]
    elseif expr isa AbstractPrimitiveOperator
        return 1, AbstractPrimitiveOperator[expr]
    elseif expr isa ScaledOperator
        expr.coefficient isa Number || return nothing
        inner = _bogoliubov_monomial(expr.operator)
        inner === nothing && return nothing
        coefficient, primitives = inner
        return expr.coefficient * coefficient, primitives
    elseif expr isa OperatorProduct
        coefficient::Number = 1
        primitives = AbstractPrimitiveOperator[]
        sizehint!(primitives, 2)

        for factor in expr.factors
            inner = _bogoliubov_monomial(factor)
            inner === nothing && return nothing
            factor_coefficient, factor_primitives = inner
            coefficient *= factor_coefficient
            append!(primitives, factor_primitives)
            length(primitives) <= 2 || return nothing
        end

        return coefficient, primitives
    end

    # A sum inside a product has not been expanded and is deliberately sent to
    # the generic symbolic fallback.
    return nothing
end

function _bogoliubov_monomials(expr::AbstractOperatorExpr)
    source_terms = expr isa OperatorSum ? expr.terms : AbstractOperatorExpr[expr]
    monomials = Tuple{Number,Vector{AbstractPrimitiveOperator}}[]
    sizehint!(monomials, length(source_terms))

    for term in source_terms
        monomial = _bogoliubov_monomial(term)
        monomial === nothing && return nothing
        push!(monomials, monomial)
    end

    return monomials
end

_bogoliubov_fast_scalar_type(::Type{T}) where {T} =
    T <: Integer || T <: Rational || T <: AbstractFloat || T <: Complex

function _bogoliubov_coefficient_type(T::BogoliubovTransformation, monomials)
    CT = promote_type(eltype(T.U), eltype(T.V))
    for (coefficient, _) in monomials
        CT = promote_type(CT, typeof(coefficient))
    end

    # Ensure division by two used by Nambu normal ordering is representable.
    CT = promote_type(CT, typeof(one(CT) / 2))
    return CT
end

function _bogoliubov_extract_fermion_quadratic(T::BogoliubovTransformation, expr::AbstractOperatorExpr)
    monomials = _bogoliubov_monomials(expr)
    monomials === nothing && return nothing

    CT = _bogoliubov_coefficient_type(T, monomials)
    _bogoliubov_fast_scalar_type(CT) || return nothing

    M = size(T.U, 1)
    A = zeros(CT, M, M)
    Delta = zeros(CT, M, M)
    ann = zeros(CT, M, M)
    constant = zero(CT)

    for (raw_coefficient, primitives) in monomials
        coefficient = convert(CT, raw_coefficient)

        if isempty(primitives)
            constant += coefficient
        elseif length(primitives) == 1
            p = primitives[1]
            if p isa NumberOperator && p.kind == :fermion
                A[p.mode, p.mode] += coefficient
            else
                return nothing
            end
        else
            p, q = primitives

            if p isa FermionCreate && q isa FermionAnnihilate
                A[p.mode, q.mode] += coefficient
            elseif p isa FermionAnnihilate && q isa FermionCreate
                if p.mode == q.mode
                    constant += coefficient
                end
                A[q.mode, p.mode] -= coefficient
            elseif p isa FermionCreate && q isa FermionCreate
                p.mode == q.mode && continue
                i, j, value = p.mode, q.mode, coefficient
                if i > j
                    i, j, value = j, i, -value
                end
                Delta[i, j] += value
                Delta[j, i] -= value
            elseif p isa FermionAnnihilate && q isa FermionAnnihilate
                p.mode == q.mode && continue
                i, j, value = p.mode, q.mode, coefficient
                if i > j
                    i, j, value = j, i, -value
                end
                ann[i, j] += value
                ann[j, i] -= value
            else
                return nothing
            end
        end
    end

    return A, Delta, ann, constant
end

function _bogoliubov_fermion_matrix_map(
    T::BogoliubovTransformation,
    A::AbstractMatrix{CT},
    Delta::AbstractMatrix{CT},
    ann::AbstractMatrix{CT},
    constant::CT,
) where {CT<:Number}
    M = size(A, 1)
    half = one(CT) / 2
    U = Matrix{CT}(T.U)
    V = Matrix{CT}(T.V)

    # For Psi_a = (a, a†)^T and Psi_g = (gamma, gamma†)^T,
    #
    #     Psi_a = W Psi_g,
    #
    # with a = U† gamma + V^T gamma† for the fermionic convention used by
    # transform_generator above.
    W = Matrix{CT}(undef, 2 * M, 2 * M)
    @views begin
        W[1:M, 1:M] .= adjoint(U)
        W[1:M, M + 1:2 * M] .= transpose(V)
        W[M + 1:2 * M, 1:M] .= adjoint(V)
        W[M + 1:2 * M, M + 1:2 * M] .= transpose(U)
    end

    # A general quadratic fermion operator can be represented as
    #
    #     H = C + 1/2 Psi_a† Q Psi_a + 1/2 tr(A),
    #
    # with Q = [A Delta; ann -A^T]. The creation and annihilation pairing
    # blocks are retained independently so that the fast path reproduces the
    # generic symbolic substitution even before Hermiticity is assumed.
    Qsource = zeros(CT, 2 * M, 2 * M)
    @views begin
        Qsource[1:M, 1:M] .= A
        Qsource[1:M, M + 1:2 * M] .= Delta
        Qsource[M + 1:2 * M, 1:M] .= ann
        Qsource[M + 1:2 * M, M + 1:2 * M] .= -transpose(A)
    end

    workspace = Qsource * W
    Qtarget = adjoint(W) * workspace

    Q11 = @view Qtarget[1:M, 1:M]
    Q12 = @view Qtarget[1:M, M + 1:2 * M]
    Q21 = @view Qtarget[M + 1:2 * M, 1:M]
    Q22 = @view Qtarget[M + 1:2 * M, M + 1:2 * M]

    # Normal order the target Nambu form. Antisymmetrization removes the
    # algebraically null symmetric component of fermionic pair operators.
    Anew = half .* (Q11 .- transpose(Q22))
    Deltanew = half .* (Q12 .- transpose(Q12))
    annnew = half .* (Q21 .- transpose(Q21))
    constantnew = constant + half * tr(A) + half * tr(Q22)

    return Anew, Deltanew, annnew, constantnew
end

function _bogoliubov_build_fermion_quadratic(
    A::AbstractMatrix,
    Delta::AbstractMatrix,
    ann::AbstractMatrix,
    constant::Number,
)
    M = size(A, 1)
    terms = AbstractOperatorExpr[]
    sizehint!(terms, 1 + M^2 + M * (M - 1))

    iszero(constant) || push!(terms, constant * IdentityOperator())

    for i in 1:M, j in 1:M
        coefficient = A[i, j]
        iszero(coefficient) && continue
        operator = i == j ? nf(i) : c(i)' * c(j)
        push!(terms, coefficient * operator)
    end

    for i in 1:M-1, j in i+1:M
        creation_coefficient = Delta[i, j]
        iszero(creation_coefficient) || push!(terms, creation_coefficient * c(i)' * c(j)')

        annihilation_coefficient = ann[i, j]
        iszero(annihilation_coefficient) || push!(terms, annihilation_coefficient * c(i) * c(j))
    end

    isempty(terms) && return 0 * IdentityOperator()
    length(terms) == 1 && return terms[1]
    return OperatorSum(terms)
end

function _bogoliubov_fermion_quadratic_fastpath(
    T::BogoliubovTransformation,
    expr::AbstractOperatorExpr,
)
    extracted = _bogoliubov_extract_fermion_quadratic(T, expr)
    extracted === nothing && return nothing

    A, Delta, ann, constant = extracted
    Anew, Deltanew, annnew, constantnew = _bogoliubov_fermion_matrix_map(T, A, Delta, ann, constant)
    return _bogoliubov_build_fermion_quadratic(Anew, Deltanew, annnew, constantnew)
end

function _bogoliubov_generic_transform(
    T::BogoliubovTransformation,
    target::Representation,
    expr::AbstractOperatorExpr,
)
    return canonicalize_transformed(T, target, transform_operator(T, expr))
end

function transform_hamiltonian(
    T::BogoliubovTransformation,
    target::Representation,
    H::AbstractOperatorExpr,
)
    if T.statistics == :fermion && target.algebra isa FermionAlgebra
        mapped = _bogoliubov_fermion_quadratic_fastpath(T, H)
        mapped === nothing || return mapped
    end

    return _bogoliubov_generic_transform(T, target, H)
end

function transform_observable(
    T::BogoliubovTransformation,
    target::Representation,
    O::AbstractOperatorExpr,
)
    if T.statistics == :fermion && target.algebra isa FermionAlgebra
        mapped = _bogoliubov_fermion_quadratic_fastpath(T, O)
        mapped === nothing || return mapped
    end

    return _bogoliubov_generic_transform(T, target, O)
end
