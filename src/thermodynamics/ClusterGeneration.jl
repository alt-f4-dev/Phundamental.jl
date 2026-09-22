# Connected induced-cluster generation from a finite reference supercell.

function _cluster_adjacency(
    reference::Supercell,
    bonds::AbstractVector{<:Bond},
)
    N = nsites(reference)
    adj = [BitSet() for _ in 1:N]
    for bond in bonds
        1 <= bond.i <= N || throw(BoundsError(reference.sites, bond.i))
        1 <= bond.j <= N || throw(BoundsError(reference.sites, bond.j))
        bond.i == bond.j && continue
        push!(adj[bond.i], bond.j)
        push!(adj[bond.j], bond.i)
    end
    return adj
end

function _induced_thermal_cluster(
    reference::Supercell,
    bonds::AbstractVector{<:Bond},
    key::Tuple{Vararg{Int}},
)
    selected = collect(key)
    index_map = Dict(old => new for (new, old) in enumerate(selected))

    T = eltype(reference.crystal.lattice.direct)
    sites = Vector{CrystalSite{T}}()
    sizehint!(sites, length(selected))
    for (newidx, oldidx) in enumerate(selected)
        s = reference.sites[oldidx]
        push!(sites, CrystalSite{T}(
            newidx,
            s.basis_index,
            copy(s.cell),
            s.label,
            s.species,
            copy(s.fractional),
            copy(s.cartesian),
        ))
    end

    subbonds = Bond{T}[]
    for bond in bonds
        haskey(index_map, bond.i) || continue
        haskey(index_map, bond.j) || continue
        push!(subbonds, Bond{T}(
            index_map[bond.i],
            index_map[bond.j],
            copy(bond.displacement),
            T(bond.distance),
            bond.shell,
        ))
    end

    # The induced graph is an open cluster even when it was extracted from a
    # larger reference object. Interactions are supplied explicitly in
    # `subbonds`, so neighbor reconstruction is not needed by build_model.
    subcell = typeof(reference)(
        reference.crystal,
        copy(reference.repetitions),
        falses(length(reference.periodic)),
        sites,
    )

    return ThermalCluster(
        key,
        length(selected),
        selected,
        subcell,
        subbonds,
    )
end

"""
    connected_clusters(reference, bonds; max_order, min_order=1,
                       max_clusters=100_000)

Generate every connected induced site cluster through `max_order` without
enumerating all `2^N` site subsets.  Clusters are expanded only across the
reference bond graph and deduplicated by their sorted parent-site tuple.

For linked-cluster work the reference should normally be open.  Periodic
self-image bonds can encode interactions to images that are not independent
cluster sites and are therefore rejected by default.
"""
function connected_clusters(
    reference::Supercell,
    bonds::AbstractVector{<:Bond};
    max_order::Integer,
    min_order::Integer=1,
    max_clusters::Integer=100_000,
    allow_periodic_reference::Bool=false,
)
    N = nsites(reference)
    1 <= min_order <= max_order <= N ||
        throw(ArgumentError("require 1 <= min_order <= max_order <= number of reference sites"))
    max_clusters > 0 || throw(ArgumentError("max_clusters must be positive"))

    if any(reference.periodic) && !allow_periodic_reference
        throw(ArgumentError(
            "connected linked-cluster generation expects an open reference Supercell; " *
            "set allow_periodic_reference=true only if periodic-image semantics are intended"
        ))
    end

    adj = _cluster_adjacency(reference, bonds)
    by_order = Dict{Int,Set{Tuple{Vararg{Int}}}}()
    by_order[1] = Set((i,) for i in 1:N)
    total = N

    for order in 2:max_order
        previous = by_order[order - 1]
        current = Set{Tuple{Vararg{Int}}}()
        for key in previous
            boundary = BitSet()
            members = BitSet(key)
            for i in key
                union!(boundary, adj[i])
            end
            setdiff!(boundary, members)

            for j in boundary
                candidate = sort!(vcat(collect(key), j))
                push!(current, Tuple(candidate))
            end
        end

        total += length(current)
        total <= max_clusters || throw(ArgumentError(
            "connected cluster count exceeded max_clusters=$max_clusters at order $order; " *
            "reduce max_order, shrink the reference, or raise the limit deliberately"
        ))
        by_order[order] = current
    end

    clusters = ThermalCluster[]
    for order in min_order:max_order
        keys_order = sort!(collect(by_order[order]))
        for key in keys_order
            push!(clusters, _induced_thermal_cluster(reference, bonds, key))
        end
    end
    return clusters
end
