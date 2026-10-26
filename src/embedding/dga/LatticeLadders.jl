# Multiorbital particle-hole Bethe-Salpeter calculations on finite Matsubara boxes.
# The scalar ladder implementation and the established orbital lattice bubble remain unchanged.

import ..Greens: bethe_salpeter, OrbitalLatticeBubble

"""Momentum-resolved orbital ladder result. `susceptibilities` is `nothing` for contracted-only execution.

`contracted[iq,iΩ]` is `w† χ(q,Ω) w / β²`, summed over both fermionic indices.
The contraction is only defined when `projection_weights` is provided.
"""
struct OrbitalLadderResult
    qpoints::Vector{Any}
    susceptibilities::Union{Nothing,Vector{OrbitalPairSusceptibility}}
    contracted::Union{Nothing,Matrix{ComplexF64}}
    diagnostics::Vector{Vector{NamedTuple}}
    metadata::Dict{Symbol,Any}
end

function _orbital_bse_compatible(bubble::OrbitalPairSusceptibility, vertex::OrbitalPairIrreducibleVertex)
    bubble.pairs == vertex.pairs || throw(ArgumentError("bubble and vertex orbital-pair ordering differ"))
    bubble.labels == vertex.labels || throw(ArgumentError("bubble and vertex orbital labels differ"))
    bubble.fermionic_axis.indices == vertex.fermionic_axis.indices || throw(ArgumentError("fermionic frequency indices differ"))
    bubble.bosonic_axis.indices == vertex.bosonic_axis.indices || throw(ArgumentError("bosonic frequency indices differ"))
    isapprox(bubble.fermionic_axis.beta, vertex.fermionic_axis.beta; rtol=0, atol=16eps(Float64) * max(bubble.fermionic_axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic inverse temperatures differ"))
    isapprox(bubble.bosonic_axis.beta, vertex.bosonic_axis.beta; rtol=0, atol=16eps(Float64) * max(bubble.bosonic_axis.beta, 1.0)) ||
        throw(ArgumentError("bosonic inverse temperatures differ"))
    size(vertex.values) == size(bubble.values) || throw(DimensionMismatch("vertex has incorrect orbital-pair frequency dimensions"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), vertex.values) || throw(ArgumentError("vertex contains nonfinite entries"))
    return nothing
end

function _orbital_bse_selection(pairs, dimension::Int; allow_projected_pairs::Bool=false)
    selected = _orbital_pair_selection(dimension, pairs)
    if !allow_projected_pairs && length(selected) != dimension^2
        throw(ArgumentError("an incomplete orbital-pair space is a projected approximation; use allow_projected_pairs=true to acknowledge it"))
    end
    return selected
end

function _orbital_bse_limits(np::Int, nf::Int, nb::Int, nq::Int, retain::Bool, max_workspace_bytes::Integer, max_output_bytes::Integer)
    max_workspace_bytes > 0 && max_output_bytes > 0 || throw(ArgumentError("memory budgets must be positive"))
    D = big(np) * nf
    # Bubble matrix, kernel, solution and residual workspace; current q's input/output tensors.
    work = big(16) * (4 * D^2 + 2 * big(np)^2 * nf^2 * nb)
    output = retain ? big(16) * big(nq) * big(np)^2 * nf^2 * nb : big(16) * big(nq) * nb
    work <= max_workspace_bytes || throw(ArgumentError("estimated live ladder workspace $work exceeds max_workspace_bytes=$max_workspace_bytes"))
    output <= max_output_bytes || throw(ArgumentError("estimated retained ladder output $output exceeds max_output_bytes=$max_output_bytes"))
    return (estimated_workspace_bytes=Int(work), estimated_output_bytes=Int(output), matrix_dimension=Int(D))
end

function _orbital_bse_matrix!(destination::Matrix{ComplexF64}, data::Array{ComplexF64,5}, iΩ::Int)
    np = size(data, 1)
    nf = size(data, 3)
    @inbounds for j in 1:np, i in 1:np, nprime in 1:nf, n in 1:nf
        destination[(i-1)*nf+n, (j-1)*nf+nprime] = data[i, j, n, nprime, iΩ]
    end
    return destination
end

function _orbital_bse_unflatten!(data::Array{ComplexF64,5}, matrix::Matrix{ComplexF64}, iΩ::Int)
    np = size(data, 1)
    nf = size(data, 3)
    @inbounds for j in 1:np, i in 1:np, nprime in 1:nf, n in 1:nf
        data[i, j, n, nprime, iΩ] = matrix[(i-1)*nf+n, (j-1)*nf+nprime]
    end
    return data
end

function _orbital_bse_solve(bubble::OrbitalPairSusceptibility, vertex::OrbitalPairIrreducibleVertex;
                            max_workspace_bytes::Integer=536870912, max_output_bytes::Integer=536870912,
                            min_relative_pivot::Real=1e-14, allow_projected_pairs::Bool=false)
    _orbital_bse_compatible(bubble, vertex)
    isfinite(min_relative_pivot) && 0 <= min_relative_pivot < 1 || throw(ArgumentError("min_relative_pivot must lie in [0,1)"))
    np, _, nf, _, nb = size(bubble.values)
    _orbital_bse_selection(bubble.pairs, length(bubble.labels); allow_projected_pairs=allow_projected_pairs)
    limits = _orbital_bse_limits(np, nf, nb, 1, true, max_workspace_bytes, max_output_bytes)
    D = limits.matrix_dimension
    B = Matrix{ComplexF64}(undef, D, D)
    V = Matrix{ComplexF64}(undef, D, D)
    K = Matrix{ComplexF64}(undef, D, D)
    X = Matrix{ComplexF64}(undef, D, D)
    values = Array{ComplexF64}(undef, size(bubble.values))
    diagnostics = Vector{NamedTuple}(undef, nb)
    for iΩ in 1:nb
        _orbital_bse_matrix!(B, bubble.values, iΩ)
        _orbital_bse_matrix!(V, vertex.values, iΩ)
        # (I - χ₀ Γ) χ = χ₀. LU factorization is never replaced by an explicit inverse.
        mul!(K, B, V)
        @inbounds for j in 1:D, i in 1:D
            K[i, j] = (i == j ? 1.0 : 0.0) - K[i, j]
        end
        factor = lu!(K; check=true)
        pivots = abs.(diag(factor.U))
        pivot_ratio = minimum(pivots) / max(maximum(pivots), eps(Float64))
        pivot_ratio >= min_relative_pivot || throw(LinearAlgebra.SingularException(argmin(pivots)))
        copyto!(X, B)
        ldiv!(factor, X)
        # Reuse the factored kernel's storage for an independently computed equation residual.
        mul!(K, V, X)
        mul!(V, B, K)
        error_sq = 0.0
        @inbounds for index in eachindex(X)
            diff = X[index] - B[index] - V[index]
            error_sq += abs2(diff)
        end
        residual = sqrt(error_sq) / max(norm(B), norm(X), eps(Float64))
        _orbital_bse_unflatten!(values, X, iΩ)
        diagnostics[iΩ] = (relative_residual=residual, pivot_ratio=pivot_ratio, matrix_dimension=D,
            estimated_workspace_bytes=limits.estimated_workspace_bytes)
    end
    metadata = Dict{Symbol,Any}(:construction => :orbital_pair_bethe_salpeter,
        :equation => :identity_minus_bubble_vertex, :matrix_order => :pair_major_frequency_minor,
        :normalization => :minus_beta_bubble, :inverse_formed => false,
        :allow_projected_pairs => allow_projected_pairs,
        :diagnostics => diagnostics, :estimated_workspace_bytes => limits.estimated_workspace_bytes)
    return OrbitalPairSusceptibility(bubble.fermionic_axis, bubble.bosonic_axis, bubble.pairs, bubble.labels, values;
        metadata=metadata, copy_values=false), diagnostics
end

"""Solve the orbital-pair particle-hole ladder equation for a lattice or synthetic bubble.

At each bosonic transfer, the solver calculates `(I - χ₀ Γ) χ = χ₀` with LU factors.
The matrix index is `(pair_index-1)*Nν + frequency_index`; this is identical to
`multiorbital_irreducible_vertex`. An incomplete pair selection requires an explicit
`allow_projected_pairs=true` acknowledgment and is not a closed exact channel.
"""
function bethe_salpeter(bubble::OrbitalPairSusceptibility, vertex::OrbitalPairIrreducibleVertex; kwargs...)
    susceptibility, _ = _orbital_bse_solve(bubble, vertex; kwargs...)
    return susceptibility
end

"""Convert an existing `OrbitalLatticeBubble` to the impurity's `(a,b),(c,d)` convention.

The legacy tensor has indices `[b,a,c,d,ν,Ω]` for the desired impurity bubble
`-β G[d,a](k,ν) G[b,c](k+q,ν+Ω)`. A selected frequency box is accepted only
when *all* its shifted frequencies are present in the original lattice grid.
"""
function orbital_pair_lattice_bubble(bubble::OrbitalLatticeBubble, selected_axis::FermionicMatsubaraAxis=bubble.fermionic_axis;
                                      pairs=nothing, max_output_bytes::Integer=536870912)
    isapprox(selected_axis.beta, bubble.bosonic_axis.beta; rtol=0, atol=16eps(Float64) * max(selected_axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic axes have different beta"))
    npairs = _orbital_pair_selection(length(bubble.labels), pairs)
    nf, nb = length(selected_axis), length(bubble.bosonic_axis)
    big(16) * big(length(npairs))^2 * nf^2 * nb <= max_output_bytes || throw(ArgumentError("converted bubble exceeds max_output_bytes"))
    lookup = Dict(n => i for (i, n) in enumerate(bubble.fermionic_axis.indices))
    for m in bubble.bosonic_axis.indices, n in selected_axis.indices
        haskey(lookup, n) && haskey(lookup, n+m) || throw(ArgumentError("incomplete lattice frequency data for ν index $n and Ω index $m"))
    end
    result = zeros(ComplexF64, length(npairs), length(npairs), nf, nf, nb)
    @inbounds for iΩ in 1:nb, (iν, n) in enumerate(selected_axis.indices)
        oldν = lookup[n]
        for (j, (c, d)) in enumerate(npairs), (i, (a, b)) in enumerate(npairs)
            result[i, j, iν, iν, iΩ] = bubble.values[b, a, c, d, oldν, iΩ]
        end
    end
    return OrbitalPairSusceptibility(selected_axis, bubble.bosonic_axis, npairs, bubble.labels, result;
        metadata=Dict{Symbol,Any}(:construction => :legacy_orbital_bubble_adapter,
            :frequency_box_complete => true, :diagonal_in_fermionic_indices => true,
            :orbital_index_adapter => :transpose_first_pair), copy_values=false)
end

function _orbital_ladder_maps(green::MatrixMomentumResolvedGreenFunction, qmaps, selected_axis::FermionicMatsubaraAxis,
                              baxis::BosonicMatsubaraAxis)
    get(green.metadata, :bloch_gauge, :periodic) === :periodic ||
        throw(ArgumentError("orbital bubbles require a periodic Bloch gauge"))
    isapprox(green.axis.beta, selected_axis.beta; rtol=0, atol=16eps(Float64) * max(selected_axis.beta, 1.0)) ||
        throw(ArgumentError("single-particle and selected fermionic axes have different beta"))
    isapprox(green.axis.beta, baxis.beta; rtol=0, atol=16eps(Float64) * max(baxis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic axes have different beta"))
    nk = size(green.values, 3)
    maps = [Int.(collect(mapping)) for mapping in qmaps]
    !isempty(maps) || throw(ArgumentError("at least one momentum-shift map is required"))
    for mapping in maps
        length(mapping) == nk && sort(mapping) == collect(1:nk) ||
            throw(ArgumentError("each momentum shift must permute the k-point indices"))
    end
    lookup = Dict(n => i for (i, n) in enumerate(green.axis.indices))
    indices = Vector{Vector{Tuple{Int,Int}}}(undef, length(baxis))
    for (iΩ, m) in enumerate(baxis.indices)
        indices[iΩ] = Tuple{Int,Int}[]
        for n in selected_axis.indices
            haskey(lookup, n) && haskey(lookup, n+m) ||
                throw(ArgumentError("incomplete single-particle frequency grid: requires indices $n and $(n+m) for Ω=$m"))
            push!(indices[iΩ], (lookup[n], lookup[n+m]))
        end
    end
    return maps, indices
end

function _orbital_ladder_bubble_at(green::MatrixMomentumResolvedGreenFunction, mapping::Vector{Int},
                                    selected_axis::FermionicMatsubaraAxis, baxis::BosonicMatsubaraAxis,
                                    pairs::Vector{Tuple{Int,Int}}, indices)
    np, nf, nb = length(pairs), length(selected_axis), length(baxis)
    values = zeros(ComplexF64, np, np, nf, nf, nb)
    nk = size(green.values, 3)
    @inbounds for iΩ in 1:nb, iν in 1:nf
        n, shifted = indices[iΩ][iν]
        for ik in 1:nk
            wk = -selected_axis.beta * green.weights[ik]
            for (j, (c, d)) in enumerate(pairs), (i, (a, b)) in enumerate(pairs)
                values[i, j, iν, iν, iΩ] += wk * green.values[d, a, ik, n] * green.values[b, c, mapping[ik], shifted]
            end
        end
    end
    return OrbitalPairSusceptibility(selected_axis, baxis, pairs, green.labels, values;
        metadata=Dict{Symbol,Any}(:construction => :strict_momentum_bubble,
            :frequency_box_complete => true, :diagonal_in_fermionic_indices => true,
            :orbital_index_adapter => :transpose_first_pair), copy_values=false)
end

"""Construct orbital-pair lattice bubbles for selected two-particle frequencies from a potentially expanded single-particle grid.

The distinct `selected_axis` makes all shifts `ν+Ω` explicit. Missing values raise
`ArgumentError` instead of being silently replaced by zero. The output is ordered
by momentum-transfer map and uses the same pair convention as impurity vertices.
"""
function orbital_pair_lattice_bubble(green::MatrixMomentumResolvedGreenFunction, qmaps,
                                      selected_axis::FermionicMatsubaraAxis, baxis::BosonicMatsubaraAxis;
                                      pairs=nothing, max_output_bytes::Integer=536870912)
    maps, indices = _orbital_ladder_maps(green, qmaps, selected_axis, baxis)
    chosen = _orbital_pair_selection(length(green.labels), pairs)
    bytes = big(16) * length(maps) * big(length(chosen))^2 * length(selected_axis)^2 * length(baxis)
    max_output_bytes > 0 && bytes <= max_output_bytes || throw(ArgumentError("orbital-pair bubble outputs exceed max_output_bytes"))
    return [_orbital_ladder_bubble_at(green, mapping, selected_axis, baxis, chosen, indices) for mapping in maps]
end

function _orbital_bse_contraction(susceptibility::OrbitalPairSusceptibility, weights::Vector{ComplexF64})
    np = length(weights)
    nf, nb = length(susceptibility.fermionic_axis), length(susceptibility.bosonic_axis)
    result = zeros(ComplexF64, nb)
    for iΩ in 1:nb, j in 1:np, i in 1:np
        coefficient = conj(weights[i]) * weights[j] / susceptibility.fermionic_axis.beta^2
        @views result[iΩ] += coefficient * sum(susceptibility.values[i, j, :, :, iΩ])
    end
    return result
end

"""Compute multiorbital momentum-dependent ladder responses with bounded per-q workspace.

`return_full=false` discards each q-resolved four-point tensor after projecting it;
this mode requires `projection_weights`. No approximate orbital symmetry reduction is
inferred; incomplete pair selections require `allow_projected_pairs=true`.
"""
function multiorbital_ladder_susceptibility(green::MatrixMomentumResolvedGreenFunction, qmaps,
                                             vertex::OrbitalPairIrreducibleVertex;
                                             return_full::Bool=true, projection_weights=nothing,
                                             allow_projected_pairs::Bool=false,
                                             max_workspace_bytes::Integer=536870912,
                                             max_output_bytes::Integer=536870912,
                                             min_relative_pivot::Real=1e-14, qpoints=nothing)
    green.labels == vertex.labels || throw(ArgumentError("lattice and vertex orbital labels differ"))
    !return_full && isnothing(projection_weights) && throw(ArgumentError("contracted-only execution requires projection_weights"))
    chosen = _orbital_bse_selection(vertex.pairs, length(vertex.labels); allow_projected_pairs=allow_projected_pairs)
    maps, indices = _orbital_ladder_maps(green, qmaps, vertex.fermionic_axis, vertex.bosonic_axis)
    qlabels = isnothing(qpoints) ? Any[iq for iq in eachindex(maps)] : Any[collect(qpoints)...]
    length(qlabels) == length(maps) || throw(DimensionMismatch("qpoint labels must match the number of momentum-shift maps"))
    weights = if isnothing(projection_weights)
        nothing
    else
        w = ComplexF64.(collect(projection_weights))
        length(w) == length(chosen) && all(z -> isfinite(real(z)) && isfinite(imag(z)), w) ||
            throw(ArgumentError("projection weights must be finite and match the orbital-pair count"))
        w
    end
    np, nf, nb = length(chosen), length(vertex.fermionic_axis), length(vertex.bosonic_axis)
    limits = _orbital_bse_limits(np, nf, nb, length(maps), return_full, max_workspace_bytes, max_output_bytes)
    storage = return_full ? Vector{OrbitalPairSusceptibility}(undef, length(maps)) : nothing
    contractions = isnothing(weights) ? nothing : Matrix{ComplexF64}(undef, length(maps), nb)
    diagnostics = Vector{Vector{NamedTuple}}(undef, length(maps))
    for iq in eachindex(maps)
        bubble = _orbital_ladder_bubble_at(green, maps[iq], vertex.fermionic_axis, vertex.bosonic_axis, chosen, indices)
        susceptibility, info = _orbital_bse_solve(bubble, vertex; allow_projected_pairs=allow_projected_pairs,
            max_workspace_bytes=max_workspace_bytes, max_output_bytes=max_output_bytes,
            min_relative_pivot=min_relative_pivot)
        diagnostics[iq] = info
        if return_full
            storage[iq] = susceptibility
        end
        if !isnothing(contractions)
            contractions[iq, :] .= _orbital_bse_contraction(susceptibility, weights)
        end
    end
    metadata = Dict{Symbol,Any}(:construction => :streamed_orbital_ladder, :frequency_box_complete => true,
        :pair_order => copy(chosen), :allow_projected_pairs => allow_projected_pairs,
        :estimated_workspace_bytes => limits.estimated_workspace_bytes,
        :estimated_output_bytes => limits.estimated_output_bytes,
        :contracted_normalization => :beta_squared, :inverse_formed => false)
    return OrbitalLadderResult(qlabels, storage, contractions, diagnostics, metadata)
end
