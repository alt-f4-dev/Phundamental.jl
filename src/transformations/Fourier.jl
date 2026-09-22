# Unitary mode transformation between two mode bases.

struct FourierTransformation{T<:Number,M<:AbstractMatrix{T}} <: AbstractTransformation
    matrix::M
    statistics::Symbol
    atol::Float64

    function FourierTransformation(
        matrix::AbstractMatrix{T};
        statistics::Symbol=:boson,
        atol::Real=1e-10,
    ) where {T<:Number}
        size(matrix, 1) == size(matrix, 2) ||
            throw(ArgumentError("Fourier matrix must be square"))
        statistics in (:boson, :fermion) ||
            throw(ArgumentError("statistics must be :boson or :fermion"))
        F = Matrix{T}(matrix)
        I_F = Matrix{T}(I, size(F, 1), size(F, 1))
        isapprox(F * adjoint(F), I_F; atol=atol, rtol=0) ||
            throw(ArgumentError("Fourier matrix must be unitary"))
        new{T,typeof(F)}(F, statistics, Float64(atol))
    end
end

transformation_name(::FourierTransformation) = :fourier
certificate(::FourierTransformation) = TransformationCertificate(
    true,
    true,
    UnitaryEquivalence,
    :nonlocal;
    notes=("unitary mode/basis transformation",),
)
has_inverse(::FourierTransformation) = true
inverse(T::FourierTransformation) = FourierTransformation(
    adjoint(T.matrix);
    statistics=T.statistics,
    atol=T.atol,
)

function applicable(T::FourierTransformation, model::ManyBodyModel)
    A = model.representation.algebra
    return (T.statistics == :boson && A isa BosonAlgebra && A.nmodes == size(T.matrix, 1)) ||
           (T.statistics == :fermion && A isa FermionAlgebra && A.nmodes == size(T.matrix, 1))
end

function target_representation(T::FourierTransformation, model::ManyBodyModel)
    M = size(T.matrix, 1)
    if T.statistics == :boson
        space = BosonFockSpace(M)
        algebra = BosonAlgebra(M)
        physical, constraints = _transported_physical_subspace(T, model.representation, space)
        gauge = _transported_gauge(T, model.representation, algebra)
        return Representation(
            :boson_fourier,
            space,
            algebra,
            BosonOccupationBasis(collect(1:M));
            physical_subspace=physical,
            constraints=constraints,
            gauge_structure=gauge,
            ordering=collect(1:M),
            truncation=_transported_truncation(T, model.representation),
        )
    else
        space = FermionFockSpace(M)
        algebra = FermionAlgebra(M)
        physical, constraints = _transported_physical_subspace(T, model.representation, space)
        gauge = _transported_gauge(T, model.representation, algebra)
        return Representation(
            :fermion_fourier,
            space,
            algebra,
            FermionOccupationBasis(collect(1:M));
            physical_subspace=physical,
            constraints=constraints,
            gauge_structure=gauge,
            ordering=collect(1:M),
            truncation=_transported_truncation(T, model.representation),
        )
    end
end

function transform_generator(T::FourierTransformation, op::BosonAnnihilate)
    T.statistics == :boson || throw(ArgumentError("fermionic Fourier transform cannot map boson operators"))
    terms = AbstractOperatorExpr[
        conj(T.matrix[k, op.mode]) * b(k) for k in axes(T.matrix, 1)
    ]
    return OperatorSum(terms)
end

function transform_generator(T::FourierTransformation, op::BosonCreate)
    T.statistics == :boson || throw(ArgumentError("fermionic Fourier transform cannot map boson operators"))
    terms = AbstractOperatorExpr[
        T.matrix[k, op.mode] * b(k)' for k in axes(T.matrix, 1)
    ]
    return OperatorSum(terms)
end

function transform_generator(T::FourierTransformation, op::FermionAnnihilate)
    T.statistics == :fermion || throw(ArgumentError("bosonic Fourier transform cannot map fermion operators"))
    return OperatorSum(AbstractOperatorExpr[
        conj(T.matrix[k, op.mode]) * c(k) for k in axes(T.matrix, 1)
    ])
end

function transform_generator(T::FourierTransformation, op::FermionCreate)
    T.statistics == :fermion || throw(ArgumentError("bosonic Fourier transform cannot map fermion operators"))
    return OperatorSum(AbstractOperatorExpr[
        T.matrix[k, op.mode] * c(k)' for k in axes(T.matrix, 1)
    ])
end

transform_generator(T::FourierTransformation, op::NumberOperator) =
    op.kind == :boson && T.statistics == :boson ?
        transform_operator(T, b(op.mode)' * b(op.mode)) :
    op.kind == :fermion && T.statistics == :fermion ?
        transform_operator(T, c(op.mode)' * c(op.mode)) :
        throw(ArgumentError("number operator $(op.kind) is outside Fourier domain"))

# ---------------------------------------------------------------------------
# Fermionic quadratic fast path
# ---------------------------------------------------------------------------

# The generic map substitutes every c_j or c_j^dagger by an M-term sum and
# distributes those sums symbolically. For a quadratic fermion expression this
# is unnecessary. Extract
#
#     H = C + c^dagger A c + pairing terms,
#
# apply the unitary mode map directly to the coefficient matrices, and rebuild
# an already canonical quadratic operator expression.

function _fourier_monomial(expr::AbstractOperatorExpr)
    if expr isa IdentityOperator
        return 1, AbstractPrimitiveOperator[]
    elseif expr isa AbstractPrimitiveOperator
        return 1, AbstractPrimitiveOperator[expr]
    elseif expr isa ScaledOperator
        expr.coefficient isa Number || return nothing
        inner = _fourier_monomial(expr.operator)
        inner === nothing && return nothing
        coefficient, primitives = inner
        return expr.coefficient * coefficient, primitives
    elseif expr isa OperatorProduct
        coefficient::Number = 1
        primitives = AbstractPrimitiveOperator[]
        sizehint!(primitives, 2)

        for factor in expr.factors
            inner = _fourier_monomial(factor)
            inner === nothing && return nothing
            factor_coefficient, factor_primitives = inner
            coefficient *= factor_coefficient
            append!(primitives, factor_primitives)
            length(primitives) <= 2 || return nothing
        end

        return coefficient, primitives
    end

    return nothing
end

function _fourier_monomials(expr::AbstractOperatorExpr)
    source_terms = expr isa OperatorSum ? expr.terms : AbstractOperatorExpr[expr]
    monomials = Tuple{Number,Vector{AbstractPrimitiveOperator}}[]
    sizehint!(monomials, length(source_terms))

    for term in source_terms
        monomial = _fourier_monomial(term)
        monomial === nothing && return nothing
        push!(monomials, monomial)
    end

    return monomials
end

_fourier_fast_scalar_type(::Type{T}) where {T} =
    T <: Integer || T <: Rational || T <: AbstractFloat || T <: Complex

function _fourier_coefficient_type(T::FourierTransformation, monomials)
    CT = eltype(T.matrix)
    for (coefficient, _) in monomials
        CT = promote_type(CT, typeof(coefficient))
    end
    return CT
end

function _fourier_extract_fermion_quadratic(
    T::FourierTransformation,
    expr::AbstractOperatorExpr,
)
    monomials = _fourier_monomials(expr)
    monomials === nothing && return nothing

    CT = _fourier_coefficient_type(T, monomials)
    _fourier_fast_scalar_type(CT) || return nothing

    M = size(T.matrix, 1)
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

function _fourier_fermion_matrix_map(
    T::FourierTransformation,
    A::AbstractMatrix{CT},
    Delta::AbstractMatrix{CT},
    ann::AbstractMatrix{CT},
    constant::CT,
) where {CT<:Number}
    M = size(A, 1)
    F = Matrix{CT}(T.matrix)
    Fconj = conj.(F)

    workspace = Matrix{CT}(undef, M, M)
    Anew = Matrix{CT}(undef, M, M)
    Deltanew = Matrix{CT}(undef, M, M)
    annnew = Matrix{CT}(undef, M, M)

    # c_j = sum_k conj(F[k,j]) d_k and
    # c_j^dagger = sum_k F[k,j] d_k^dagger imply
    #
    #     A'     = F A F^dagger,
    #     Delta' = F Delta F^T,
    #     ann'   = conj(F) ann F^dagger.
    mul!(workspace, F, A)
    mul!(Anew, workspace, adjoint(F))

    mul!(workspace, F, Delta)
    mul!(Deltanew, workspace, transpose(F))

    mul!(workspace, Fconj, ann)
    mul!(annnew, workspace, adjoint(F))

    return Anew, Deltanew, annnew, constant
end

function _fourier_build_fermion_quadratic(
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

function _fourier_fermion_quadratic_fastpath(
    T::FourierTransformation,
    expr::AbstractOperatorExpr,
)
    extracted = _fourier_extract_fermion_quadratic(T, expr)
    extracted === nothing && return nothing

    A, Delta, ann, constant = extracted
    Anew, Deltanew, annnew, constantnew = _fourier_fermion_matrix_map(T, A, Delta, ann, constant)
    return _fourier_build_fermion_quadratic(Anew, Deltanew, annnew, constantnew)
end

function _fourier_generic_transform(
    T::FourierTransformation,
    target::Representation,
    expr::AbstractOperatorExpr,
)
    return canonicalize_transformed(T, target, transform_operator(T, expr))
end

function transform_hamiltonian(
    T::FourierTransformation,
    target::Representation,
    H::AbstractOperatorExpr,
)
    if T.statistics == :fermion && target.algebra isa FermionAlgebra
        mapped = _fourier_fermion_quadratic_fastpath(T, H)
        mapped === nothing || return mapped
    end

    return _fourier_generic_transform(T, target, H)
end

function transform_observable(
    T::FourierTransformation,
    target::Representation,
    O::AbstractOperatorExpr,
)
    if T.statistics == :fermion && target.algebra isa FermionAlgebra
        mapped = _fourier_fermion_quadratic_fastpath(T, O)
        mapped === nothing || return mapped
    end

    return _fourier_generic_transform(T, target, O)
end
