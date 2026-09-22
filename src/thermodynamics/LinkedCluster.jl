# Finite-reference linked-cluster inclusion-exclusion.

const _THERMO_EXTENSIVE_PROPERTIES = (
    :free_energy,
    :internal_energy,
    :entropy,
    :heat_capacity,
)

function _curve_property(curve::ThermodynamicCurve, property::Symbol)
    property === :free_energy && return curve.free_energy
    property === :internal_energy && return curve.internal_energy
    property === :entropy && return curve.entropy
    property === :heat_capacity && return curve.heat_capacity
    throw(ArgumentError("unsupported linked-cluster property $property"))
end

function _proper_subset_keys(key::Tuple{Vararg{Int}})
    n = length(key)
    n <= 1 && return Tuple{Vararg{Int}}[]

    out = Tuple{Vararg{Int}}[]
    chosen = Int[]

    function rec(i::Int)
        if i > n
            isempty(chosen) && return
            length(chosen) == n && return
            push!(out, Tuple(chosen))
            return
        end
        rec(i + 1)
        push!(chosen, key[i])
        rec(i + 1)
        pop!(chosen)
    end

    rec(1)
    return out
end

"""
    linked_cluster_weights(raw_curves, clusters, temperatures, reference_sites)

For each embedded connected cluster C, recursively construct

    W_P(C) = P(C) - Σ_{S proper connected subset of C} W_P(S)

for extensive thermal quantities.  Embedded clusters are retained separately;
no graph-isomorphism assumption is made, so inequivalent material sites and
bond environments remain distinct.
"""
function linked_cluster_weights(
    raw_curves::Dict{Tuple{Vararg{Int}},ThermodynamicCurve},
    clusters,
    temperatures,
    reference_sites::Integer,
)
    Ts = Float64.(collect(temperatures))
    nt = length(Ts)
    reference_sites > 0 || throw(ArgumentError("reference_sites must be positive"))

    sorted_clusters = sort(collect(clusters), by=c -> (c.order, c.key))
    weights = Dict{Symbol,Dict{Tuple{Vararg{Int}},Vector{Float64}}}()

    for property in _THERMO_EXTENSIVE_PROPERTIES
        wp = Dict{Tuple{Vararg{Int}},Vector{Float64}}()
        for cluster in sorted_clusters
            haskey(raw_curves, cluster.key) ||
                throw(KeyError("missing raw thermodynamic curve for cluster $(cluster.key)"))
            raw = copy(_curve_property(raw_curves[cluster.key], property))
            length(raw) == nt || throw(DimensionMismatch(
                "cluster thermodynamic curve has inconsistent temperature grid"
            ))

            for subkey in _proper_subset_keys(cluster.key)
                haskey(wp, subkey) || continue  # disconnected subsets have zero linked weight
                raw .-= wp[subkey]
            end
            wp[cluster.key] = raw
        end
        weights[property] = wp
    end
    return weights
end

function _linked_partial_sums(
    weights,
    clusters,
    max_order::Int,
    nt::Int,
    reference_sites::Int,
)
    partial = Dict{Symbol,Matrix{Float64}}()
    convergence = Dict{Symbol,Matrix{Float64}}()

    for property in _THERMO_EXTENSIVE_PROPERTIES
        sums = zeros(Float64, max_order, nt)
        wp = weights[property]
        for order in 1:max_order
            order > 1 && (@views sums[order, :] .= sums[order - 1, :])
            for cluster in clusters
                cluster.order == order || continue
                @views sums[order, :] .+= wp[cluster.key] ./ reference_sites
            end
        end

        delta = fill(NaN, max_order, nt)
        if max_order >= 2
            for order in 2:max_order
                @views delta[order, :] .= abs.(sums[order, :] .- sums[order - 1, :])
            end
        end
        partial[property] = sums
        convergence[property] = delta
    end
    return partial, convergence
end
