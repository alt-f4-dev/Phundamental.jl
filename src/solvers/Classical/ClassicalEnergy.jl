# ClassicalEnergy.jl

function ising_energy(problem::IsingProblem, state::AbstractVector)
    length(state) == length(problem) ||
        throw(DimensionMismatch("state length does not match IsingProblem"))
    return problem.constant +
           0.5 * dot(state, problem.J * state) +
           dot(problem.field, state)
end

"""
    ising_flip_delta(problem, state, local_product, i)

Exact O(1) energy change for flipping site `i`, assuming

    local_product = J * state.

The diagonal `J[i,i]` is removed because `s_i^2` is unchanged by an Ising flip.
"""
@inline function ising_flip_delta(
    problem::IsingProblem,
    state::AbstractVector,
    local_product::AbstractVector,
    i::Int,
)
    si = state[i]
    effective = local_product[i] - problem.J[i, i] * si + problem.field[i]
    return -2 * si * effective
end

function _update_local_product!(
    local_product::AbstractVector,
    J::Matrix{Float64},
    i::Int,
    delta_spin::Real,
)
    @inbounds @simd for j in eachindex(local_product)
        local_product[j] += delta_spin * J[j, i]
    end
    return local_product
end

function _update_local_product!(
    local_product::AbstractVector,
    J::SparseMatrixCSC{Float64,Int},
    i::Int,
    delta_spin::Real,
)
    @inbounds for p in nzrange(J, i)
        row = rowvals(J)[p]
        local_product[row] += delta_spin * nonzeros(J)[p]
    end
    return local_product
end

function _initial_ising_state(problem::IsingProblem, initial, rng)
    N = length(problem)
    if initial === :random
        return Int8[rand(rng, Bool) ? 1 : -1 for _ in 1:N]
    elseif initial === :up
        return fill(Int8(1), N)
    elseif initial === :down
        return fill(Int8(-1), N)
    elseif initial isa AbstractVector
        length(initial) == N ||
            throw(DimensionMismatch("initial state must contain one value per site"))
        all(x -> x == -1 || x == 1, initial) ||
            throw(ArgumentError("initial Ising entries must be ±1"))
        return Int8.(initial)
    end
    throw(ArgumentError("initial must be :random, :up, :down, or a ±1 vector"))
end
