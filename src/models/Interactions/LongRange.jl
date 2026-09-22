# LongRange.jl
#
# Shared periodic-cell geometry and direct lattice sums.

function _require_3d(cluster::Supercell)
    lattice_dimension(cluster.crystal.lattice) == 3 ||
        throw(ArgumentError("long-range Coulomb/dipolar implementation currently requires 3D geometry"))
    return true
end

function _require_full_3d_periodicity(cluster::Supercell)
    _require_3d(cluster)
    all(cluster.periodic) ||
        throw(ArgumentError("Ewald summation requires periodic=true in all three dimensions"))
    return true
end

function _supercell_direct_matrix(cluster::Supercell)
    A = Matrix{Float64}(cluster.crystal.lattice.direct)
    return A * Diagonal(Float64.(cluster.repetitions))
end

function _supercell_reciprocal_matrix(cluster::Supercell)
    H = _supercell_direct_matrix(cluster)
    return 2pi .* inv(transpose(H))
end

_supercell_volume(cluster::Supercell) = abs(det(_supercell_direct_matrix(cluster)))

function _position_matrix(cluster::Supercell)
    N = nsites(cluster)
    R = zeros(Float64, 3, N)
    for i in 1:N
        R[:, i] .= cluster.sites[i].cartesian
    end
    return R
end

function _cell_pair_diameter(positions::AbstractMatrix)
    N = size(positions, 2)
    diameter = 0.0
    for i in 1:N, j in (i + 1):N
        diameter = max(diameter, norm(view(positions, :, j) .- view(positions, :, i)))
    end
    return diameter
end

function _enumerate_integer_vectors(matrix::AbstractMatrix, cutoff::Real)
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    sigma = minimum(svdvals(Matrix{Float64}(matrix)))
    sigma > 0 || throw(ArgumentError("periodic lattice matrix is singular"))
    nmax = ceil(Int, cutoff / sigma) + 1

    vectors = Tuple{NTuple{3,Int},Vector{Float64}}[]
    cutoff2 = Float64(cutoff)^2
    for n1 in -nmax:nmax, n2 in -nmax:nmax, n3 in -nmax:nmax
        n = (n1, n2, n3)
        v = matrix * Float64[n1, n2, n3]
        dot(v, v) <= cutoff2 * (1 + 64eps(Float64)) || continue
        push!(vectors, (n, Vector{Float64}(v)))
    end
    sort!(vectors, by=x -> (dot(x[2], x[2]), x[1]))
    return vectors
end

@inline _zero_shift(n::NTuple{3,Int}) = n == (0, 0, 0)

function _real_image_vectors(cluster::Supercell, cutoff::Real, positions)
    H = _supercell_direct_matrix(cluster)
    # A translation R with |R| > cutoff + max|r_j-r_i| cannot bring any pair
    # separation inside the requested sphere. This bound is conservative and
    # prevents missed cancellation in skewed cells.
    bound = Float64(cutoff) + _cell_pair_diameter(positions)
    return _enumerate_integer_vectors(H, bound)
end

function _reciprocal_vectors(cluster::Supercell, cutoff::Real)
    B = _supercell_reciprocal_matrix(cluster)
    return [
        item for item in _enumerate_integer_vectors(B, cutoff)
        if !_zero_shift(item[1])
    ]
end

function _pair_distance(dr0, image)
    r = dr0 .+ image
    r2 = dot(r, r)
    return r, r2
end

function _direct_scalar_sum(
    interaction::CoulombInteraction,
    cluster::Supercell,
    cutoff::Real,
)
    _require_3d(cluster)
    positions = _position_matrix(cluster)
    images = if any(cluster.periodic)
        _real_image_vectors(cluster, cutoff, positions)
    else
        [((0, 0, 0), zeros(Float64, 3))]
    end

    N = nsites(cluster)
    K = zeros(Float64, N, N)
    cutoff2 = Float64(cutoff)^2

    for i in 1:N
        ri = view(positions, :, i)
        for j in i:N
            dr0 = view(positions, :, j) .- ri
            value = 0.0
            for (n, R) in images
                i == j && _zero_shift(n) && continue
                r, r2 = _pair_distance(dr0, R)
                (r2 <= 0 || r2 > cutoff2 * (1 + 64eps(Float64))) && continue
                value += interaction.strength / sqrt(r2)
            end
            K[i, j] = value
            K[j, i] = value
        end
    end

    metadata = Dict{Symbol,Any}(
        :interaction => :coulomb,
        :summation => :real_space_cutoff,
        :cutoff => Float64(cutoff),
        :periodic => collect(cluster.periodic),
        :conditionally_convergent_if_periodic => any(cluster.periodic),
        :approximation => :finite_real_space_cutoff,
    )
    Z = zeros(Float64, N, N)
    return ScalarInteractionResult(K, K, Z, Z, Z, metadata)
end

@inline function _dipolar_tensor(r::AbstractVector, strength::Real)
    r2 = dot(r, r)
    r2 > 0 || throw(ArgumentError("dipolar tensor is singular at r=0"))
    rinv = inv(sqrt(r2))
    invr3 = rinv^3
    invr5 = rinv^5
    return strength .* (
        invr3 .* Matrix{Float64}(I, 3, 3) .-
        (3 * invr5) .* (r * transpose(r))
    )
end

function _direct_tensor_sum(
    interaction::DipolarInteraction,
    cluster::Supercell,
    cutoff::Real,
)
    _require_3d(cluster)
    positions = _position_matrix(cluster)
    images = if any(cluster.periodic)
        _real_image_vectors(cluster, cutoff, positions)
    else
        [((0, 0, 0), zeros(Float64, 3))]
    end

    N = nsites(cluster)
    K = zeros(Float64, N, N, 3, 3)
    cutoff2 = Float64(cutoff)^2

    for i in 1:N
        ri = view(positions, :, i)
        for j in i:N
            dr0 = view(positions, :, j) .- ri
            Tij = zeros(Float64, 3, 3)
            for (n, R) in images
                i == j && _zero_shift(n) && continue
                r, r2 = _pair_distance(dr0, R)
                (r2 <= 0 || r2 > cutoff2 * (1 + 64eps(Float64))) && continue
                Tij .+= _dipolar_tensor(r, interaction.strength)
            end
            K[i, j, :, :] .= Tij
            K[j, i, :, :] .= transpose(Tij)
        end
    end

    metadata = Dict{Symbol,Any}(
        :interaction => :dipolar,
        :summation => :real_space_cutoff,
        :cutoff => Float64(cutoff),
        :periodic => collect(cluster.periodic),
        :conditionally_convergent_if_periodic => any(cluster.periodic),
        :approximation => :finite_real_space_cutoff,
    )
    Z = zeros(Float64, size(K))
    return TensorInteractionResult(K, K, Z, Z, Z, metadata)
end

_interaction_matrix(interaction::CoulombInteraction, cluster::Supercell, method::RealSpaceCutoff) = _direct_scalar_sum(interaction, cluster, method.cutoff)

_interaction_tensor(interaction::DipolarInteraction, cluster::Supercell, method::RealSpaceCutoff) = _direct_tensor_sum(interaction, cluster, method.cutoff)
