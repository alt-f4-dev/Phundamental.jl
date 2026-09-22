struct PackedBosonOperator
    terms::Vector{_CompiledTerm}
    data::PackedBosonBasisData
end

Base.size(A::PackedBosonOperator) = (length(A.data.codes), length(A.data.codes))
Base.size(A::PackedBosonOperator, d::Integer) = size(A)[d]
Base.eltype(::PackedBosonOperator) = ComplexF64
Base.eltype(::Type{PackedBosonOperator}) = ComplexF64

@inline function _boson_occupation(data::PackedBosonBasisData, code::UInt64, pos::Int)
    return Int((code ÷ data.strides[pos]) % UInt64(data.radices[pos]))
end

@inline function _apply_boson_primitive(
    primitive::_CompiledPrimitive,
    code::UInt64,
    amp::ComplexF64,
    data::PackedBosonBasisData,
)
    primitive.code == _OP_IDENTITY && return code, amp, true
    mode = primitive.index
    1 <= mode <= length(data.mode_positions) ||
        throw(ArgumentError("boson operator refers to unknown mode $mode"))
    pos = data.mode_positions[mode]
    n = _boson_occupation(data, code, pos)
    stride = data.strides[pos]

    if primitive.code == _OP_BOSON_N
        n == 0 && return code, 0.0 + 0.0im, false
        return code, amp * n, true
    elseif primitive.code == _OP_BOSON_A
        n == 0 && return code, 0.0 + 0.0im, false
        return code - stride, amp * sqrt(n), true
    elseif primitive.code == _OP_BOSON_C
        n + 1 >= data.radices[pos] && return code, 0.0 + 0.0im, false
        return code + stride, amp * sqrt(n + 1), true
    end
    throw(ArgumentError("non-bosonic primitive encountered in packed boson kernel"))
end

@inline function _decode_boson_code(data::PackedBosonBasisData, code::UInt64)
    return [
        Int((code ÷ data.strides[pos]) % UInt64(data.radices[pos]))
        for pos in eachindex(data.radices)
    ]
end

@inline function _encode_boson_occupations(data::PackedBosonBasisData, occ::Vector{Int})
    code = UInt64(0)
    @inbounds for pos in eachindex(occ)
        0 <= occ[pos] < data.radices[pos] || return UInt64(0), false
        code += UInt64(occ[pos]) * data.strides[pos]
    end
    return code, true
end

@inline function _apply_boson_term(
    term::_CompiledTerm,
    input::UInt64,
    data::PackedBosonBasisData,
)
    # Numerical truncation is a projection on the FINAL state only. Intermediate
    # occupations are allowed to exceed the cutoff so that, for example,
    # P b b† P is not incorrectly replaced by P b P b† P.
    occ = _decode_boson_code(data, input)
    amp = term.coefficient
    @inbounds for j in length(term.primitives):-1:1
        p = term.primitives[j]
        p.code == _OP_IDENTITY && continue
        mode = p.index
        1 <= mode <= length(data.mode_positions) ||
            throw(ArgumentError("boson operator refers to unknown mode $mode"))
        pos = data.mode_positions[mode]
        n = occ[pos]
        if p.code == _OP_BOSON_N
            n == 0 && return input, 0.0 + 0.0im, false
            amp *= n
        elseif p.code == _OP_BOSON_A
            n == 0 && return input, 0.0 + 0.0im, false
            amp *= sqrt(n)
            occ[pos] = n - 1
        elseif p.code == _OP_BOSON_C
            amp *= sqrt(n + 1)
            occ[pos] = n + 1
        else
            throw(ArgumentError("non-bosonic primitive encountered in packed boson kernel"))
        end
    end
    output, inside = _encode_boson_occupations(data, occ)
    inside || return output, 0.0 + 0.0im, false
    return output, amp, true
end

function LinearAlgebra.mul!(y::AbstractVector, A::PackedBosonOperator, x::AbstractVector)
    dim = length(A.data.codes)
    length(x) == dim == length(y) ||
        throw(DimensionMismatch("packed boson multiply dimension mismatch"))
    fill!(y, zero(eltype(y)))

    @inbounds for col in 1:dim
        xcol = x[col]
        iszero(xcol) && continue
        input = A.data.codes[col]
        for term in A.terms
            output, amp, ok = _apply_boson_term(term, input, A.data)
            ok || continue
            row = Int(output) + 1
            y[row] += amp * xcol
        end
    end
    return y
end

function Base.:*(A::PackedBosonOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, x)
    return y
end

function _packed_boson_operator(model::ManyBodyModel; max_dimension::Int)
    data = _packed_boson_basis(model; max_dimension=max_dimension)
    terms = _expanded_operator_terms(model.hamiltonian)
    allowed = Set((_OP_IDENTITY, _OP_BOSON_A, _OP_BOSON_C, _OP_BOSON_N))
    for term in terms, primitive in term.primitives
        primitive.code in allowed || throw(ArgumentError(
            "packed boson backend encountered a non-bosonic operator"
        ))
    end
    return PackedBosonOperator(terms, data)
end

function _solve_packed_boson(solver::OptimizedLanczos, model::ManyBodyModel)
    isnothing(solver.sector) || throw(ArgumentError(
        "PackedBosonBackend does not currently accept an additional symmetry sector"
    ))
    A = _packed_boson_operator(model; max_dimension=solver.max_basis_dimension)
    dim = size(A, 1)
    solver.nev <= dim || throw(ArgumentError("nev=$(solver.nev) exceeds basis dimension $dim"))

    ok, herr = _optimized_hermitian_check(
        A, dim; seed=solver.seed + 23, tol=max(10 * solver.tol, 1e-9),
    )
    ok || throw(ArgumentError(
        "optimized packed-boson Lanczos requires Hermitian projected Hamiltonian; error=$herr"
    ))

    dec = _optimized_lanczos_decomposition(
        A, dim;
        krylov_dim=min(dim, max(solver.krylov_dim, solver.nev + 2)),
        tol=solver.tol,
        seed=solver.seed,
        reorthogonalize=solver.reorthogonalize,
    )
    energies, right, residuals = _optimized_ritz(dec, solver.nev; tol=solver.tol)

    metadata = Dict{Symbol,Any}(
        :solver => :optimized_lanczos,
        :backend => :packed_boson,
        :dimension => dim,
        :sector => nothing,
        :hermitian => true,
        :hermiticity_error => herr,
        :biorthogonal => false,
        :iterations => dec.iterations,
        :ritz_residuals => residuals,
        :converged => residuals .<= solver.tol,
        :matrix_free => true,
        :approximation => :numerical_boson_truncation,
        :physical_representation_preserved => true,
    )
    return SolverResult(
        solver, model, energies, right, right, A.data.basis, metadata,
    )
end
