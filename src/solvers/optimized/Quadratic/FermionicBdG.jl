# Extraction and diagonalization of quadratic fermionic Hamiltonians.
#
# The optimized path reads canonical quadratic operator ASTs directly into
# coefficient matrices. It deliberately bypasses the generic operator compiler,
# whose materialized term expansion is inappropriate for dense O(M^2)
# quadratic Hamiltonians produced by exact mode transformations.

# ---------------------------------------------------------------------------
# Direct canonical-AST extraction
# ---------------------------------------------------------------------------

@inline function _fermion_factor_signature(expr::AbstractOperatorExpr)
    coefficient = 1.0 + 0.0im
    core = expr

    while core isa ScaledOperator
        core.coefficient isa Number || return false, coefficient, _OP_IDENTITY, 0
        coefficient *= ComplexF64(core.coefficient)
        core = core.operator
    end

    if core isa IdentityOperator
        return true, coefficient, _OP_IDENTITY, 0
    elseif core isa FermionAnnihilate
        return true, coefficient, _OP_FERMION_A, core.mode
    elseif core isa FermionCreate
        return true, coefficient, _OP_FERMION_C, core.mode
    elseif core isa NumberOperator && core.kind == :fermion
        return true, coefficient, _OP_FERMION_N, core.mode
    end

    return false, coefficient, _OP_IDENTITY, 0
end

@inline function _fermion_check_mode(index::Int, M::Int)
    1 <= index <= M || throw(ArgumentError(
        "fermionic mode index $index lies outside the representation with $M modes"
    ))
    return nothing
end

@inline function _fermion_accumulate_signature!(
    A::Matrix{ComplexF64},
    Delta::Matrix{ComplexF64},
    ann::Matrix{ComplexF64},
    constant::Base.RefValue{ComplexF64},
    coefficient::ComplexF64,
    nprimitives::Int,
    code1::UInt8,
    index1::Int,
    code2::UInt8,
    index2::Int,
)
    iszero(coefficient) && return true
    M = size(A, 1)

    if nprimitives == 0
        constant[] += coefficient
        return true
    elseif nprimitives == 1
        code1 == _OP_FERMION_N || return false
        _fermion_check_mode(index1, M)
        A[index1, index1] += coefficient
        return true
    elseif nprimitives != 2
        return false
    end

    _fermion_check_mode(index1, M)
    _fermion_check_mode(index2, M)

    if code1 == _OP_FERMION_C && code2 == _OP_FERMION_A
        A[index1, index2] += coefficient
        return true
    elseif code1 == _OP_FERMION_A && code2 == _OP_FERMION_C
        index1 == index2 && (constant[] += coefficient)
        A[index2, index1] -= coefficient
        return true
    elseif code1 == _OP_FERMION_C && code2 == _OP_FERMION_C
        index1 == index2 && return true
        i, j, value = index1, index2, coefficient
        if i > j
            i, j, value = j, i, -value
        end
        Delta[i, j] += value
        Delta[j, i] -= value
        return true
    elseif code1 == _OP_FERMION_A && code2 == _OP_FERMION_A
        index1 == index2 && return true
        i, j, value = index1, index2, coefficient
        if i > j
            i, j, value = j, i, -value
        end
        ann[i, j] += value
        ann[j, i] -= value
        return true
    end

    return false
end

function _fermion_accumulate_monomial!(
    A::Matrix{ComplexF64},
    Delta::Matrix{ComplexF64},
    ann::Matrix{ComplexF64},
    constant::Base.RefValue{ComplexF64},
    expr::AbstractOperatorExpr,
    coefficient::ComplexF64,
)
    if expr isa IdentityOperator
        constant[] += coefficient
        return true
    elseif expr isa FermionAnnihilate || expr isa FermionCreate
        return false
    elseif expr isa NumberOperator
        expr.kind == :fermion || return false
        return _fermion_accumulate_signature!(
            A, Delta, ann, constant, coefficient, 1, _OP_FERMION_N, expr.mode, _OP_IDENTITY, 0,
        )
    elseif !(expr isa OperatorProduct)
        return false
    end

    nprimitives = 0
    code1 = _OP_IDENTITY
    code2 = _OP_IDENTITY
    index1 = 0
    index2 = 0
    value = coefficient

    for factor in expr.factors
        supported, factor_coefficient, code, index = _fermion_factor_signature(factor)
        supported || return false
        value *= factor_coefficient
        iszero(value) && return true
        code == _OP_IDENTITY && continue

        nprimitives += 1
        nprimitives <= 2 || return false

        if nprimitives == 1
            code1 = code
            index1 = index
        else
            code2 = code
            index2 = index
        end
    end

    return _fermion_accumulate_signature!(
        A, Delta, ann, constant, value, nprimitives, code1, index1, code2, index2,
    )
end

function _fermion_accumulate_expr!(
    A::Matrix{ComplexF64},
    Delta::Matrix{ComplexF64},
    ann::Matrix{ComplexF64},
    constant::Base.RefValue{ComplexF64},
    expr::AbstractOperatorExpr,
    coefficient::ComplexF64=1.0 + 0.0im,
)
    if expr isa OperatorSum
        for term in expr.terms
            _fermion_accumulate_expr!(A, Delta, ann, constant, term, coefficient) || return false
        end
        return true
    elseif expr isa ScaledOperator
        expr.coefficient isa Number || return false
        value = coefficient * ComplexF64(expr.coefficient)
        iszero(value) && return true
        return _fermion_accumulate_expr!(A, Delta, ann, constant, expr.operator, value)
    end

    return _fermion_accumulate_monomial!(A, Delta, ann, constant, expr, coefficient)
end

# ---------------------------------------------------------------------------
# Generic compiler fallback
# ---------------------------------------------------------------------------

function _fermion_accumulate_compiled!(
    A::Matrix{ComplexF64},
    Delta::Matrix{ComplexF64},
    ann::Matrix{ComplexF64},
    constant::Base.RefValue{ComplexF64},
    model::ManyBodyModel,
)
    terms = _expanded_operator_terms(model.hamiltonian)

    for term in terms
        code1 = _OP_IDENTITY
        code2 = _OP_IDENTITY
        index1 = 0
        index2 = 0
        nprimitives = 0

        for primitive in term.primitives
            primitive.code == _OP_IDENTITY && continue
            nprimitives += 1
            nprimitives <= 2 || throw(ArgumentError(
                "Hamiltonian contains interaction terms beyond quadratic fermion form"
            ))

            if nprimitives == 1
                code1 = primitive.code
                index1 = primitive.index
            else
                code2 = primitive.code
                index2 = primitive.index
            end
        end

        _fermion_accumulate_signature!(
            A, Delta, ann, constant, term.coefficient, nprimitives, code1, index1, code2, index2,
        ) || throw(ArgumentError("Hamiltonian is not quadratic in fermionic operators"))
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Validation without temporary dense matrices
# ---------------------------------------------------------------------------

function _fermion_quadratic_diagnostics(
    A::Matrix{ComplexF64},
    Delta::Matrix{ComplexF64},
    ann::Matrix{ComplexF64};
    tol::Real,
)
    normA2 = 0.0
    normDelta2 = 0.0
    normAnn2 = 0.0
    hermiticity2 = 0.0
    pair2 = 0.0

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        aij = A[i, j]
        dij = Delta[i, j]
        bij = ann[i, j]

        normA2 += abs2(aij)
        normDelta2 += abs2(dij)
        normAnn2 += abs2(bij)
        hermiticity2 += abs2(aij - conj(A[j, i]))
        pair2 += abs2(bij + conj(dij))
    end

    normA = sqrt(normA2)
    normDelta = sqrt(normDelta2)
    normAnn = sqrt(normAnn2)
    hermiticity_error = sqrt(hermiticity2)
    ascale = max(normA, 1.0)

    hermiticity_error <= tol * ascale || throw(ArgumentError(
        "quadratic normal matrix A is not Hermitian"
    ))

    pair_error = sqrt(pair2) / max(normDelta, normAnn, 1.0)
    pair_error <= max(tol, 1e-12) || throw(ArgumentError(
        "fermionic pairing creation/annihilation terms are not Hermitian conjugates; error=$pair_error"
    ))

    return pair_error, normA, normDelta
end

function _fermion_quadratic_data(model::ManyBodyModel; tol::Real=1e-10)
    rep = model.representation
    rep.algebra isa FermionAlgebra ||
        throw(ArgumentError("QuadraticFermionSolver requires FermionAlgebra"))

    M = rep.algebra.nmodes
    A = zeros(ComplexF64, M, M)
    Delta = zeros(ComplexF64, M, M)
    ann = zeros(ComplexF64, M, M)
    constant = Ref(0.0 + 0.0im)

    # Canonical Hamiltonians produced by the algebra and optimized exact
    # transformations take this direct path. No _CompiledTerm array is built,
    # so dense O(M^2) quadratic Hamiltonians are not subject to max_terms.
    direct = _fermion_accumulate_expr!(A, Delta, ann, constant, model.hamiltonian)

    if !direct
        # Preserve support for noncanonical but expandable quadratic ASTs.
        # The generic compiler remains a fallback rather than the hot path.
        fill!(A, 0)
        fill!(Delta, 0)
        fill!(ann, 0)
        constant[] = 0.0 + 0.0im
        _fermion_accumulate_compiled!(A, Delta, ann, constant, model)
    end

    pair_error, normA, normDelta = _fermion_quadratic_diagnostics(A, Delta, ann; tol=tol)
    return A, Delta, constant[], pair_error, normA, normDelta
end

function _fermion_quadratic_matrices(model::ManyBodyModel; tol::Real=1e-10)
    A, Delta, constant, pair_error, _, _ = _fermion_quadratic_data(model; tol=tol)
    return A, Delta, constant, pair_error
end

# ---------------------------------------------------------------------------
# BdG construction and diagonalization
# ---------------------------------------------------------------------------

function _fermion_bdg_matrix(A::Matrix{ComplexF64}, Delta::Matrix{ComplexF64})
    M = size(A, 1)
    Hbdg = Matrix{ComplexF64}(undef, 2 * M, 2 * M)

    @views begin
        Hbdg[1:M, 1:M] .= A
        Hbdg[1:M, M + 1:2 * M] .= Delta
        Hbdg[M + 1:2 * M, 1:M] .= adjoint(Delta)
        Hbdg[M + 1:2 * M, M + 1:2 * M] .= -transpose(A)
    end

    return Hbdg
end

function _relative_hermiticity_error(A::AbstractMatrix{ComplexF64})
    normA2 = 0.0
    error2 = 0.0

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        aij = A[i, j]
        normA2 += abs2(aij)
        error2 += abs2(aij - conj(A[j, i]))
    end

    return sqrt(error2) / max(sqrt(normA2), 1.0)
end

function solve(solver::QuadraticFermionSolver, model::ManyBodyModel)
    A, Delta, constant, pair_error, normA, normDelta =
        _fermion_quadratic_data(model; tol=solver.tol)
    M = size(A, 1)
    pairing = normDelta > solver.tol * max(normA, 1.0)

    if !pairing
        F = eigen(Hermitian(A))
        energies = F.values
        modes = F.vectors
        metadata = Dict{Symbol,Any}(
            :solver => :quadratic_fermion,
            :backend => :quadratic_fermion,
            :number_conserving => true,
            :pairing => false,
            :constant => constant,
            :dimension => M,
            :approximation => :none_if_input_is_quadratic,
            :mode_space => :single_particle,
        )
        return QuadraticModeResult(solver, model, energies, modes, metadata)
    end

    Hbdg = _fermion_bdg_matrix(A, Delta)
    herr = _relative_hermiticity_error(Hbdg)
    herr <= max(solver.tol, 1e-12) || throw(ArgumentError(
        "fermionic BdG matrix is not Hermitian; error=$herr"
    ))

    # Particle-hole symmetry orders the M positive quasiparticle eigenvalues in
    # the upper half of the 2M-dimensional Hermitian BdG spectrum. Request only
    # those eigenpairs instead of computing and storing the full eigensystem.
    F = eigen(Hermitian(Hbdg), (M + 1):(2 * M))
    scale = max(maximum(abs, F.values), 1.0)
    all(e -> e > solver.tol * scale, F.values) || throw(ArgumentError(
        "fermionic BdG spectrum does not contain exactly $M positive quasiparticle energies"
    ))

    energies = F.values
    modes = F.vectors
    metadata = Dict{Symbol,Any}(
        :solver => :quadratic_fermion,
        :backend => :fermionic_bdg,
        :number_conserving => false,
        :pairing => true,
        :constant => constant,
        :dimension => M,
        :bdg_matrix => Hbdg,
        :hermiticity_error => herr,
        :pair_conjugacy_error => pair_error,
        :approximation => :none_if_input_is_quadratic,
        :mode_space => :nambu,
    )
    return QuadraticModeResult(solver, model, energies, modes, metadata)
end
