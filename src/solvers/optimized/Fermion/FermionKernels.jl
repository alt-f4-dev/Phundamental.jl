struct PackedFermionOperator
    terms::Vector{_CompiledTerm}
    data::PackedFermionBasisData
end

Base.size(A::PackedFermionOperator) = (length(A.data.masks), length(A.data.masks))
Base.size(A::PackedFermionOperator, d::Integer) = size(A)[d]
Base.eltype(::PackedFermionOperator) = ComplexF64
Base.eltype(::Type{PackedFermionOperator}) = ComplexF64

@inline function _fermion_parity(mask::UInt64, pos::Int)
    pos <= 1 && return 1.0
    lower = (UInt64(1) << (pos - 1)) - UInt64(1)
    return isodd(count_ones(mask & lower)) ? -1.0 : 1.0
end

@inline function _apply_fermion_primitive(
    primitive::_CompiledPrimitive,
    mask::UInt64,
    amp::ComplexF64,
    data::PackedFermionBasisData,
)
    code = primitive.code
    code == _OP_IDENTITY && return mask, amp, true
    mode = primitive.index
    1 <= mode <= length(data.mode_positions) ||
        throw(ArgumentError("fermion operator refers to unknown mode $mode"))
    pos = data.mode_positions[mode]
    bit = _bit_at(mask, pos)
    flag = UInt64(1) << (pos - 1)

    if code == _OP_FERMION_N
        bit == UInt64(0) && return mask, 0.0 + 0.0im, false
        return mask, amp, true
    elseif code == _OP_FERMION_A
        bit == UInt64(0) && return mask, 0.0 + 0.0im, false
        return mask & ~flag, amp * _fermion_parity(mask, pos), true
    elseif code == _OP_FERMION_C
        bit == UInt64(1) && return mask, 0.0 + 0.0im, false
        return mask | flag, amp * _fermion_parity(mask, pos), true
    end
    throw(ArgumentError("non-fermionic primitive encountered in packed fermion kernel"))
end

@inline function _apply_fermion_term(
    term::_CompiledTerm,
    input::UInt64,
    data::PackedFermionBasisData,
)
    state = input
    amp = term.coefficient
    @inbounds for j in length(term.primitives):-1:1
        state, amp, ok = _apply_fermion_primitive(
            term.primitives[j], state, amp, data,
        )
        ok || return state, 0.0 + 0.0im, false
    end
    return state, amp, true
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    A::PackedFermionOperator,
    x::AbstractVector,
)
    dim = length(A.data.masks)
    length(x) == dim == length(y) ||
        throw(DimensionMismatch("packed fermion multiply dimension mismatch"))
    fill!(y, zero(eltype(y)))

    @inbounds for col in 1:dim
        xcol = x[col]
        iszero(xcol) && continue
        input = A.data.masks[col]
        for term in A.terms
            output, amp, ok = _apply_fermion_term(term, input, A.data)
            ok || continue
            row = get(A.data.index, output, 0)
            row == 0 && continue
            y[row] += amp * xcol
        end
    end
    return y
end

function Base.:*(A::PackedFermionOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, x)
    return y
end

function _packed_fermion_operator(
    model::ManyBodyModel,
    sector::Union{Nothing,FermionSector};
    max_dimension::Int,
)
    data = _packed_fermion_basis(model, sector; max_dimension=max_dimension)
    terms = _expanded_operator_terms(model.hamiltonian)
    allowed = Set((_OP_IDENTITY, _OP_FERMION_A, _OP_FERMION_C, _OP_FERMION_N))
    for term in terms, primitive in term.primitives
        primitive.code in allowed || throw(ArgumentError(
            "packed fermion backend encountered a non-fermionic operator"
        ))
    end
    return PackedFermionOperator(terms, data)
end
