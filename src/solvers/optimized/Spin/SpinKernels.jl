# Deterministic packed spin-1/2 action.
#
# Common one-site terms and same-axis two-site terms are fused once during
# operator construction. Coefficients, bit masks, and hbar factors are also
# precompiled so the Lanczos/TPQ hot loop performs only bit tests, complex
# multiply-adds, and direct output-index updates. For the nearest-neighbor
# Heisenberg model, Sx_i*Sx_j + Sy_i*Sy_j + Sz_i*Sz_j becomes one bond kernel.

struct _PackedSpinSingleKernel
    flip_mask::UInt64
    diagonal_up::ComplexF64
    flip_up::ComplexF64
    flip_down::ComplexF64
end

struct _PackedSpinPairKernel
    bit_i::UInt64
    bit_j::UInt64
    flip_mask::UInt64
    diagonal_same::ComplexF64
    diagonal_opposite::ComplexF64
    flip_same::ComplexF64
    flip_opposite::ComplexF64
    gather_opposite::ComplexF64
end

struct PackedSpinOperator
    constant::ComplexF64
    singles::Vector{_PackedSpinSingleKernel}
    pairs::Vector{_PackedSpinPairKernel}
    residual_terms::Vector{_CompiledTerm}
    data::PackedSpinBasisData
    hbar::Float64
    norm_upper_bound::Float64
    hermitian_certified::Bool
    threaded::Bool
end

Base.size(A::PackedSpinOperator) = (A.data.dimension, A.data.dimension)
Base.size(A::PackedSpinOperator, d::Integer) = size(A)[d]
Base.eltype(::PackedSpinOperator) = ComplexF64
Base.eltype(::Type{PackedSpinOperator}) = ComplexF64

@inline function _apply_spin_primitive(primitive::_CompiledPrimitive, mask::UInt64, amp::ComplexF64, data::PackedSpinBasisData, hbar::Float64)
    code = primitive.code
    code == _OP_IDENTITY && return mask, amp, true

    site = primitive.index
    1 <= site <= length(data.site_positions) || throw(ArgumentError("spin operator refers to unknown site $site"))
    pos = data.site_positions[site]
    bit = _bit_at(mask, pos)
    flip = UInt64(1) << (pos - 1)

    if code == _OP_SPIN_Z
        m = bit == UInt64(1) ? 0.5 : -0.5
        return mask, amp * (hbar * m), true
    elseif code == _OP_SPIN_X
        return mask ⊻ flip, amp * (hbar / 2), true
    elseif code == _OP_SPIN_Y
        phase = bit == UInt64(1) ? 1im : -1im
        return mask ⊻ flip, amp * (phase * hbar / 2), true
    elseif code == _OP_SPIN_P
        bit == UInt64(1) && return mask, 0.0 + 0.0im, false
        return mask | flip, amp * hbar, true
    elseif code == _OP_SPIN_M
        bit == UInt64(0) && return mask, 0.0 + 0.0im, false
        return mask & ~flip, amp * hbar, true
    end
    throw(ArgumentError("non-spin primitive encountered in packed spin kernel"))
end

@inline function _apply_spin_term(term::_CompiledTerm, input::UInt64, data::PackedSpinBasisData, hbar::Float64)
    state = input
    amp = term.coefficient
    @inbounds for j in length(term.primitives):-1:1
        state, amp, ok = _apply_spin_primitive(term.primitives[j], state, amp, data, hbar)
        ok || return state, 0.0 + 0.0im, false
    end
    return state, amp, true
end

function _collect_fused_spin_coefficients(terms::Vector{_CompiledTerm})
    constant = 0.0 + 0.0im
    single_acc = Dict{Int,Vector{ComplexF64}}()
    pair_acc = Dict{Tuple{Int,Int},Vector{ComplexF64}}()
    residual = _CompiledTerm[]

    for term in terms
        primitives = term.primitives
        if length(primitives) == 1 && primitives[1].code == _OP_IDENTITY
            constant += term.coefficient
            continue
        end

        if length(primitives) == 1 && primitives[1].code in (_OP_SPIN_X, _OP_SPIN_Y, _OP_SPIN_Z)
            p = primitives[1]
            coeffs = get!(single_acc, p.index) do
                zeros(ComplexF64, 3)
            end
            idx = p.code == _OP_SPIN_X ? 1 : p.code == _OP_SPIN_Y ? 2 : 3
            coeffs[idx] += term.coefficient
            continue
        end

        if length(primitives) == 2
            p1, p2 = primitives
            same_axis = p1.code == p2.code && p1.code in (_OP_SPIN_X, _OP_SPIN_Y, _OP_SPIN_Z)
            if same_axis && p1.index != p2.index
                i, j = minmax(p1.index, p2.index)
                coeffs = get!(pair_acc, (i, j)) do
                    zeros(ComplexF64, 3)
                end
                idx = p1.code == _OP_SPIN_X ? 1 : p1.code == _OP_SPIN_Y ? 2 : 3
                coeffs[idx] += term.coefficient
                continue
            end
        end

        push!(residual, term)
    end

    return constant, single_acc, pair_acc, residual
end

function _residual_spin_norm_bound(term::_CompiledTerm, hbar::Float64)
    bound = abs(term.coefficient)
    for primitive in term.primitives
        primitive.code == _OP_IDENTITY && continue
        if primitive.code in (_OP_SPIN_X, _OP_SPIN_Y, _OP_SPIN_Z)
            bound *= hbar / 2
        elseif primitive.code in (_OP_SPIN_P, _OP_SPIN_M)
            bound *= hbar
        else
            throw(ArgumentError("packed spin norm bound encountered a non-spin primitive"))
        end
    end
    return Float64(bound)
end

function _compile_fused_spin_kernels(terms::Vector{_CompiledTerm}, data::PackedSpinBasisData, hbar::Float64)
    constant, single_acc, pair_acc, residual = _collect_fused_spin_coefficients(terms)
    singles = _PackedSpinSingleKernel[]
    pairs = _PackedSpinPairKernel[]
    norm_bound = abs(constant)
    hermitian = abs(imag(constant)) <= 1e-13

    sizehint!(singles, length(single_acc))
    for site in sort!(collect(keys(single_acc)))
        c = single_acc[site]
        pos = data.site_positions[site]
        flip = UInt64(1) << (pos - 1)
        prefactor = hbar / 2
        diagonal_up = prefactor * c[3]
        flip_up = prefactor * (c[1] + 1im * c[2])
        flip_down = prefactor * (c[1] - 1im * c[2])
        push!(singles, _PackedSpinSingleKernel(flip, diagonal_up, flip_up, flip_down))
        norm_bound += prefactor * (abs(c[1]) + abs(c[2]) + abs(c[3]))
        hermitian &= maximum(abs.(imag.(c))) <= 1e-13
    end

    pair_keys = sort!(collect(keys(pair_acc)))
    sizehint!(pairs, length(pair_keys))
    pair_prefactor = hbar^2 / 4
    for (i, j) in pair_keys
        c = pair_acc[(i, j)]
        pos_i = data.site_positions[i]
        pos_j = data.site_positions[j]
        bit_i = UInt64(1) << (pos_i - 1)
        bit_j = UInt64(1) << (pos_j - 1)
        diagonal_same = pair_prefactor * c[3]
        diagonal_opposite = -diagonal_same
        flip_same = pair_prefactor * (c[1] - c[2])
        flip_opposite = pair_prefactor * (c[1] + c[2])
        kernel = _PackedSpinPairKernel(bit_i, bit_j, bit_i | bit_j, diagonal_same, diagonal_opposite, flip_same, flip_opposite, conj(flip_opposite))
        push!(pairs, kernel)
        norm_bound += pair_prefactor * (abs(c[1]) + abs(c[2]) + abs(c[3]))
        hermitian &= maximum(abs.(imag.(c))) <= 1e-13
    end

    for term in residual
        norm_bound += _residual_spin_norm_bound(term, hbar)
    end
    hermitian &= isempty(residual)
    return constant, singles, pairs, residual, Float64(norm_bound), hermitian
end

@inline function _packed_spin_row(data::PackedSpinBasisData, output::UInt64)
    if isnothing(data.nup)
        output < UInt64(data.dimension) || return 0
        return Int(output) + 1
    end
    return _fixed_weight_rank(output, data)
end

@inline function _packed_spin_diagonal(A::PackedSpinOperator, input::UInt64)
    diagonal = A.constant
    @inbounds for kernel in A.singles
        up = !iszero(input & kernel.flip_mask)
        diagonal += up ? kernel.diagonal_up : -kernel.diagonal_up
    end
    @inbounds for kernel in A.pairs
        same = iszero(input & kernel.bit_i) == iszero(input & kernel.bit_j)
        diagonal += same ? kernel.diagonal_same : kernel.diagonal_opposite
    end
    return diagonal
end

@inline function _packed_spin_sector_gather_supported(A::PackedSpinOperator)
    isnothing(A.data.nup) && return false
    isempty(A.residual_terms) || return false
    A.hermitian_certified || return false
    all(kernel -> iszero(kernel.flip_up) && iszero(kernel.flip_down), A.singles) || return false
    all(kernel -> iszero(kernel.flip_same), A.pairs) || return false
    return true
end

@inline function _packed_spin_gather_value_combinatorial(A::PackedSpinOperator, output::UInt64, col::Int, x::AbstractVector)
    xcol = x[col]
    value = A.constant * xcol
    @inbounds for kernel in A.singles
        up = !iszero(output & kernel.flip_mask)
        value += (up ? kernel.diagonal_up : -kernel.diagonal_up) * xcol
    end
    @inbounds for kernel in A.pairs
        same = iszero(output & kernel.bit_i) == iszero(output & kernel.bit_j)
        if same
            value += kernel.diagonal_same * xcol
        else
            value += kernel.diagonal_opposite * xcol
            if !iszero(kernel.gather_opposite)
                source = output ⊻ kernel.flip_mask
                source_col = _fixed_weight_rank(source, A.data)
                source_col != 0 && (value += kernel.gather_opposite * x[source_col])
            end
        end
    end
    return value
end

@inline function _packed_spin_gather_value_dense(A::PackedSpinOperator, lookup::Vector{UInt32}, output::UInt64, col::Int, x::AbstractVector)
    xcol = x[col]
    value = A.constant * xcol
    @inbounds for kernel in A.singles
        up = !iszero(output & kernel.flip_mask)
        value += (up ? kernel.diagonal_up : -kernel.diagonal_up) * xcol
    end
    @inbounds for kernel in A.pairs
        same = iszero(output & kernel.bit_i) == iszero(output & kernel.bit_j)
        if same
            value += kernel.diagonal_same * xcol
        else
            value += kernel.diagonal_opposite * xcol
            if !iszero(kernel.gather_opposite)
                source = output ⊻ kernel.flip_mask
                source_col = Int(lookup[Int(source) + 1])
                value += kernel.gather_opposite * x[source_col]
            end
        end
    end
    return value
end

function _packed_spin_gather_chunk_combinatorial!(y::AbstractVector, A::PackedSpinOperator, x::AbstractVector, first_col::Int, last_col::Int)
    first_col > last_col && return nothing
    mask = _fixed_weight_mask_at(first_col, A.data)
    @inbounds for col in first_col:last_col
        y[col] = _packed_spin_gather_value_combinatorial(A, mask, col, x)
        col < last_col && (mask = _next_lex_fixed_weight_mask(mask, A.data))
    end
    return nothing
end

function _packed_spin_gather_chunk_dense!(y::AbstractVector, A::PackedSpinOperator, x::AbstractVector, lookup::Vector{UInt32}, first_col::Int, last_col::Int)
    first_col > last_col && return nothing
    mask = _fixed_weight_mask_at(first_col, A.data)
    @inbounds for col in first_col:last_col
        y[col] = _packed_spin_gather_value_dense(A, lookup, mask, col, x)
        col < last_col && (mask = _next_lex_fixed_weight_mask(mask, A.data))
    end
    return nothing
end

@inline function _apply_packed_spin_column!(y::AbstractVector, A::PackedSpinOperator, input::UInt64, col::Int, xcol)
    diagonal = A.constant

    @inbounds for kernel in A.singles
        up = !iszero(input & kernel.flip_mask)
        diagonal += up ? kernel.diagonal_up : -kernel.diagonal_up
        amp = up ? kernel.flip_up : kernel.flip_down
        if !iszero(amp)
            row = _packed_spin_row(A.data, input ⊻ kernel.flip_mask)
            row != 0 && (y[row] += amp * xcol)
        end
    end

    @inbounds for kernel in A.pairs
        same = iszero(input & kernel.bit_i) == iszero(input & kernel.bit_j)
        diagonal += same ? kernel.diagonal_same : kernel.diagonal_opposite
        amp = same ? kernel.flip_same : kernel.flip_opposite
        if !iszero(amp)
            row = _packed_spin_row(A.data, input ⊻ kernel.flip_mask)
            row != 0 && (y[row] += amp * xcol)
        end
    end

    @inbounds for term in A.residual_terms
        output, amp, ok = _apply_spin_term(term, input, A.data, A.hbar)
        ok || continue
        row = _packed_spin_row(A.data, output)
        row != 0 && (y[row] += amp * xcol)
    end

    !iszero(diagonal) && (y[col] += diagonal * xcol)
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector, A::PackedSpinOperator, x::AbstractVector)
    dim = A.data.dimension
    length(x) == dim == length(y) || throw(DimensionMismatch("packed spin multiply dimension mismatch"))

    if _packed_spin_sector_gather_supported(A)
        lookup = A.data.rank_lookup
        if A.threaded && Threads.nthreads() > 1 && dim >= 100_000
            nchunks = min(Threads.nthreads(), dim)
            chunk_size = cld(dim, nchunks)
            if isnothing(lookup)
                Threads.@threads :static for chunk in 1:nchunks
                    first_col = (chunk - 1) * chunk_size + 1
                    last_col = min(chunk * chunk_size, dim)
                    _packed_spin_gather_chunk_combinatorial!(y, A, x, first_col, last_col)
                end
            else
                Threads.@threads :static for chunk in 1:nchunks
                    first_col = (chunk - 1) * chunk_size + 1
                    last_col = min(chunk * chunk_size, dim)
                    _packed_spin_gather_chunk_dense!(y, A, x, lookup, first_col, last_col)
                end
            end
        elseif isnothing(lookup)
            _packed_spin_gather_chunk_combinatorial!(y, A, x, 1, dim)
        else
            _packed_spin_gather_chunk_dense!(y, A, x, lookup, 1, dim)
        end
        return y
    end

    fill!(y, zero(eltype(y)))
    if isnothing(A.data.nup)
        @inbounds for col in 1:dim
            xcol = x[col]
            iszero(xcol) && continue
            _apply_packed_spin_column!(y, A, UInt64(col - 1), col, xcol)
        end
    else
        mask = _fixed_weight_mask_at(1, A.data)
        @inbounds for col in 1:dim
            xcol = x[col]
            !iszero(xcol) && _apply_packed_spin_column!(y, A, mask, col, xcol)
            col < dim && (mask = _next_lex_fixed_weight_mask(mask, A.data))
        end
    end
    return y
end

function Base.:*(A::PackedSpinOperator, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, x)
    return y
end

_packed_spin_hermitian_certified(A::PackedSpinOperator; tol::Real=1e-13) = A.hermitian_certified

function _packed_spin_operator(model::ManyBodyModel, sector::Union{Nothing,SpinSector}; hbar::Real, max_dimension::Int, materialize_basis::Bool=true,
                               rank_lookup::Union{Nothing,Vector{UInt32}}=nothing, populate_rank_lookup::Bool=false, threaded::Bool=false)
    data = _packed_spin_basis(model, sector; max_dimension=max_dimension, materialize_basis=materialize_basis,
                              rank_lookup=rank_lookup, populate_rank_lookup=populate_rank_lookup)
    terms = _expanded_operator_terms(model.hamiltonian)
    allowed = Set((_OP_IDENTITY, _OP_SPIN_X, _OP_SPIN_Y, _OP_SPIN_Z, _OP_SPIN_P, _OP_SPIN_M))
    for term in terms, primitive in term.primitives
        primitive.code in allowed || throw(ArgumentError("packed spin backend encountered a non-spin operator"))
    end
    constant, singles, pairs, residual, norm_bound, hermitian = _compile_fused_spin_kernels(terms, data, Float64(hbar))
    return PackedSpinOperator(constant, singles, pairs, residual, data, Float64(hbar), norm_bound, hermitian, threaded)
end
