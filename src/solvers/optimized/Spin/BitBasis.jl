# Implicit packed spin-1/2 computational basis.
#
# Full Hilbert spaces and fixed-nup sectors are represented combinatorially.
# Product-state objects, mask vectors, and hash tables are materialized only when an API explicitly requires a ComputationalBasis, for example when returned eigenvectors are consumed by generic observable machinery.
#
# A fixed-weight sector may additionally share a dense UInt32 mask->sector-rank lookup. For N <= 30 this can trade at most 4 GiB at N=30 for O(1) off-diagonal sector indexing, which is substantially faster than evaluating a combinatorial rank for every bond transition. The lookup is optional and never changes the basis ordering.

struct PackedSpinBasisData
    nsites::Int
    dimension::Int
    nup::Union{Nothing,Int}
    site_positions::Vector{Int}
    binomial::Matrix{Int}
    rank_lookup::Union{Nothing,Vector{UInt32}}
    basis::Union{Nothing,ComputationalBasis}
    sector::Union{Nothing,SpinSector}
end

Base.length(data::PackedSpinBasisData) = data.dimension

function _spin_nup(sector::SpinSector, nsites::Int)
    if !isnothing(sector.nup)
        0 <= sector.nup <= nsites || throw(ArgumentError("spin-sector nup must lie in 0:$nsites"))
        return sector.nup
    end
    value = sector.total_sz + nsites / 2
    abs(value - round(value)) <= 1e-10 || throw(ArgumentError("requested total_sz=$(sector.total_sz) is incompatible with $nsites spin-1/2 sites"))
    nup = Int(round(value))
    0 <= nup <= nsites || throw(ArgumentError("requested total_sz lies outside the spin spectrum"))
    return nup
end

function _spin_binomial_table(n::Int)
    table = zeros(Int, n + 1, n + 1)
    table[1, 1] = 1
    for nn in 1:n
        table[nn + 1, 1] = 1
        table[nn + 1, nn + 1] = 1
        for kk in 1:(nn - 1)
            table[nn + 1, kk + 1] = table[nn, kk] + table[nn, kk + 1]
        end
    end
    return table
end

@inline _spin_binomial(table::Matrix{Int}, n::Int, k::Int) = (k < 0 || k > n) ? 0 : table[n + 1, k + 1]

@inline function _first_fixed_weight_mask(n::Int, k::Int)
    0 <= k <= n <= 64 || throw(ArgumentError("invalid fixed-weight mask parameters n=$n, k=$k"))
    k == 0 && return UInt64(0)
    k == 64 && return typemax(UInt64)
    return (UInt64(1) << k) - UInt64(1)
end

@inline function _last_fixed_weight_mask(n::Int, k::Int)
    0 <= k <= n <= 64 || throw(ArgumentError("invalid fixed-weight mask parameters n=$n, k=$k"))
    k == 0 && return UInt64(0)
    k == 64 && return typemax(UInt64)
    return ((UInt64(1) << k) - UInt64(1)) << (n - k)
end

@inline function _next_fixed_weight_mask(mask::UInt64)
    c = mask & (~mask + UInt64(1))
    r = mask + c
    return r | (((r ⊻ mask) >> 2) ÷ c)
end

@inline function _low_bit_mask(n::Int)
    n == 64 && return typemax(UInt64)
    return (UInt64(1) << n) - UInt64(1)
end

@inline function _previous_fixed_weight_mask(mask::UInt64, n::Int)
    domain = _low_bit_mask(n)
    complement = (~mask) & domain
    next_complement = _next_fixed_weight_mask(complement)
    return (~next_complement) & domain
end

@inline function _reverse_low_bits(mask::UInt64, n::Int)
    x = mask
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1)
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2)
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0f) | ((x & 0x0f0f0f0f0f0f0f0f) << 4)
    x = ((x >> 8) & 0x00ff00ff00ff00ff) | ((x & 0x00ff00ff00ff00ff) << 8)
    x = ((x >> 16) & 0x0000ffff0000ffff) | ((x & 0x0000ffff0000ffff) << 16)
    x = (x >> 32) | (x << 32)
    return n == 64 ? x : x >> (64 - n)
end

@inline function _colex_fixed_weight_rank(mask::UInt64, k::Int, table::Matrix{Int})
    count_ones(mask) == k || return 0
    rank0 = 0
    remaining = mask
    j = 1
    while remaining != 0
        p = trailing_zeros(remaining)
        rank0 += _spin_binomial(table, p, j)
        remaining &= remaining - UInt64(1)
        j += 1
    end
    return rank0 + 1
end

# `_fixed_weight_masks` in Sectors/SymmetryBasis.jl enumerates site combinations lexicographically. Random-access ranking preserves that order through bit-reversed colex ranking, while sequential traversal uses a direct lexicographic successor without repeated bit reversal.
@inline function _fixed_weight_rank(mask::UInt64, data::PackedSpinBasisData)
    k = data.nup
    isnothing(k) && return Int(mask) + 1
    if !isnothing(data.rank_lookup)
        idx = Int(mask) + 1
        1 <= idx <= length(data.rank_lookup) || return 0
        return Int(@inbounds data.rank_lookup[idx])
    end
    count_ones(mask) == k || return 0
    reversed = _reverse_low_bits(mask, data.nsites)
    colex = _colex_fixed_weight_rank(reversed, k, data.binomial)
    return data.dimension - colex + 1
end

"""Return the mask at one-based lexicographic sector index without materializing the sector basis."""
function _fixed_weight_mask_at(index::Int, data::PackedSpinBasisData)
    1 <= index <= data.dimension || throw(BoundsError(1:data.dimension, index))
    k = data.nup
    isnothing(k) && return UInt64(index - 1)
    k == 0 && return UInt64(0)

    rank0 = index - 1
    mask = UInt64(0)
    start = 1
    remaining = k
    while remaining > 0
        last = data.nsites - remaining + 1
        chosen = 0
        for pos in start:last
            block = _spin_binomial(data.binomial, data.nsites - pos, remaining - 1)
            if rank0 < block
                chosen = pos
                break
            end
            rank0 -= block
        end
        chosen != 0 || throw(ArgumentError("failed to unrank fixed-weight spin basis index $index"))
        mask |= UInt64(1) << (chosen - 1)
        start = chosen + 1
        remaining -= 1
    end
    return mask
end

@inline function _next_lex_fixed_weight_mask(mask::UInt64, data::PackedSpinBasisData)
    k = data.nup
    isnothing(k) && return mask + UInt64(1)
    k == 0 && return mask

    remaining = mask
    suffix = 0
    while remaining != 0
        position = 64 - leading_zeros(remaining)
        maximum_position = data.nsites - suffix
        if position < maximum_position
            prefix = position == 1 ? UInt64(0) : mask & ((UInt64(1) << (position - 1)) - UInt64(1))
            moved_count = suffix + 1
            moved = _low_bit_mask(moved_count) << position
            return prefix | moved
        end
        remaining &= ~(UInt64(1) << (position - 1))
        suffix += 1
    end

    return mask
end

function _populate_spin_rank_lookup!(lookup::Vector{UInt32}, data::PackedSpinBasisData)
    isnothing(data.nup) && return lookup
    data.dimension <= Int(typemax(UInt32)) || throw(ArgumentError("UInt32 sector-rank lookup cannot represent dimension $(data.dimension)"))
    data.nsites < Sys.WORD_SIZE - 1 || throw(ArgumentError("dense sector-rank lookup requires a directly indexable 2^N mask space"))
    required = Int(1) << data.nsites
    length(lookup) == required || throw(DimensionMismatch("sector-rank lookup has length $(length(lookup)); expected $required"))

    mask = _fixed_weight_mask_at(1, data)
    @inbounds for col in 1:data.dimension
        lookup[Int(mask) + 1] = UInt32(col)
        col < data.dimension && (mask = _next_lex_fixed_weight_mask(mask, data))
    end
    return lookup
end

function _materialized_spin_masks(data::PackedSpinBasisData)
    if isnothing(data.nup)
        masks = Vector{UInt64}(undef, data.dimension)
        @inbounds for i in eachindex(masks)
            masks[i] = UInt64(i - 1)
        end
        return masks
    end
    return _fixed_weight_masks(data.nsites, data.nup; max_dimension=data.dimension)
end

function _materialize_spin_computational_basis(model::ManyBodyModel, factors, data::PackedSpinBasisData)
    rep = model.representation
    masks = _materialized_spin_masks(data)
    states = Vector{SpinBasisState}(undef, length(masks))
    for (i, mask) in enumerate(masks)
        levels = ntuple(pos -> Int(_bit_at(mask, pos)), data.nsites)
        states[i] = SpinBasisState(levels)
    end
    state_index = Dict{SpinBasisState,Int}(state => i for (i, state) in enumerate(states))
    layout = BasisLayout(
        factors,
        _family_map(factors),
        _all_physical_constraints(rep),
        rep.truncation,
        !isempty(_all_physical_constraints(rep)) || !isnothing(rep.physical_subspace),
        !isnothing(rep.truncation),
    )
    return ComputationalBasis(rep, states, state_index, layout)
end

function _packed_spin_basis(model::ManyBodyModel, sector::Union{Nothing,SpinSector}; max_dimension::Int, materialize_basis::Bool=true,
                            rank_lookup::Union{Nothing,Vector{UInt32}}=nothing, populate_rank_lookup::Bool=false)
    rep = model.representation
    rep.algebra isa SpinAlgebra || throw(ArgumentError("packed spin backend requires SpinAlgebra"))
    factors = _representation_factors(rep)
    length(factors) == 1 || throw(ArgumentError("packed spin backend requires a non-composite spin representation"))
    factor = factors[1]
    factor.family === :spin || throw(ArgumentError("invalid spin factor"))
    spins = factor.state_space.spins
    all(S -> abs(S - 0.5) <= 1e-12, spins) || throw(ArgumentError("packed spin backend currently supports spin-1/2 only"))
    N = length(spins)
    N <= 64 || throw(ArgumentError("packed spin backend supports at most 64 spin-1/2 sites"))
    isempty(_all_physical_constraints(rep)) || throw(ArgumentError("packed spin backend does not combine representation-level spin constraints with SpinSector"))

    table = _spin_binomial_table(N)
    nup = isnothing(sector) ? nothing : _spin_nup(sector, N)
    dimension = if isnothing(nup)
        N < Sys.WORD_SIZE - 1 || throw(ArgumentError("full packed spin space for N >= $(Sys.WORD_SIZE - 1) cannot be indexed by Int; use SpinSector"))
        Int(1) << N
    else
        _spin_binomial(table, N, nup)
    end
    _checked_dimension(dimension, max_dimension, "spin")
    dimension > 0 || throw(ArgumentError("spin sector has an empty basis"))

    if !isnothing(rank_lookup)
        isnothing(nup) && throw(ArgumentError("dense sector-rank lookup is only meaningful for a fixed-nup SpinSector"))
        N < Sys.WORD_SIZE - 1 || throw(ArgumentError("dense sector-rank lookup requires a directly indexable 2^N mask space"))
        length(rank_lookup) == (Int(1) << N) || throw(DimensionMismatch("sector-rank lookup length does not match 2^N"))
        dimension <= Int(typemax(UInt32)) || throw(ArgumentError("sector dimension exceeds UInt32 rank capacity"))
    end

    site_positions = [factor.positions[i] for i in 1:N]
    data = PackedSpinBasisData(N, dimension, nup, site_positions, table, rank_lookup, nothing, sector)
    populate_rank_lookup && !isnothing(rank_lookup) && _populate_spin_rank_lookup!(rank_lookup, data)
    if materialize_basis
        basis = _materialize_spin_computational_basis(model, factors, data)
        data = PackedSpinBasisData(N, dimension, nup, site_positions, table, rank_lookup, basis, sector)
    end
    return data
end
