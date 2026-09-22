# Compact combination generators shared by packed spin and fermion sectors.

function _checked_dimension(n::Integer, limit::Integer, what::AbstractString)
    n >= 0 || throw(ArgumentError("$what dimension must be nonnegative"))
    n <= limit || throw(ArgumentError(
        "$what basis dimension $n exceeds configured limit $limit"
    ))
    return Int(n)
end

function _binomial_checked(n::Int, k::Int)
    0 <= k <= n || return 0
    k = min(k, n - k)
    value = big(1)
    for j in 1:k
        value = div(value * (n - k + j), j)
        value <= typemax(Int) || throw(OverflowError("basis dimension exceeds Int"))
    end
    return Int(value)
end

function _fixed_weight_masks(n::Int, k::Int; max_dimension::Int=50_000_000)
    0 <= n <= 64 || throw(ArgumentError("packed basis supports at most 64 bits"))
    0 <= k <= n || return UInt64[]
    dim = _checked_dimension(_binomial_checked(n, k), max_dimension, "sector")
    out = Vector{UInt64}()
    sizehint!(out, dim)

    # Recursive combination generation avoids UInt64 overflow at n=64 while
    # still doing O(number of retained basis states) work.
    function rec(start::Int, remaining::Int, mask::UInt64)
        if remaining == 0
            push!(out, mask)
            return
        end
        last = n - remaining + 1
        for pos in start:last
            rec(pos + 1, remaining - 1, mask | (UInt64(1) << (pos - 1)))
        end
    end
    rec(1, k, UInt64(0))
    return out
end

function _fixed_subset_masks(
    positions::Vector{Int},
    k::Int;
    max_dimension::Int=50_000_000,
)
    n = length(positions)
    local_masks = _fixed_weight_masks(n, k; max_dimension=max_dimension)
    out = Vector{UInt64}(undef, length(local_masks))
    for (idx, lm) in enumerate(local_masks)
        gm = UInt64(0)
        for j in 1:n
            ((lm >> (j - 1)) & UInt64(1)) == UInt64(0) && continue
            p = positions[j]
            1 <= p <= 64 || throw(ArgumentError("bit position must lie in 1:64"))
            gm |= UInt64(1) << (p - 1)
        end
        out[idx] = gm
    end
    return out
end

function _product_sector_masks(
    first_positions::Vector{Int},
    first_count::Int,
    second_positions::Vector{Int},
    second_count::Int;
    max_dimension::Int=50_000_000,
)
    a = _fixed_subset_masks(first_positions, first_count; max_dimension=max_dimension)
    b = _fixed_subset_masks(second_positions, second_count; max_dimension=max_dimension)
    dim_big = big(length(a)) * big(length(b))
    dim_big <= max_dimension || throw(ArgumentError(
        "sector basis dimension $dim_big exceeds configured limit $max_dimension"
    ))
    out = Vector{UInt64}()
    sizehint!(out, Int(dim_big))
    for x in a, y in b
        (x & y) == 0 || throw(ArgumentError("sector mode groups overlap"))
        push!(out, x | y)
    end
    sort!(out)
    return out
end
