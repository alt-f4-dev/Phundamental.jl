# Dipolar.jl
#
# Energy evaluation, local-frame projection, and symbolic spin Hamiltonian
# construction from long-range interaction results.

function interaction_energy(
    result::ScalarInteractionResult,
    charges::AbstractVector;
    neutrality_tolerance::Real=1e-10,
)
    N = size(result.total, 1)
    length(charges) == N ||
        throw(DimensionMismatch("charge vector length must match interaction matrix"))
    q = Float64.(charges)

    if get(result.metadata, :requires_neutrality, false)
        scale = max(sum(abs, q), 1.0)
        abs(sum(q)) <= neutrality_tolerance * scale ||
            throw(ArgumentError(
                "periodic Coulomb Ewald energy requires a neutral cell; " *
                "use CoulombInteraction(...; neutralizing_background=true) " *
                "only when a uniform compensating background is physically intended"
            ))
    end

    return 0.5 * dot(q, result.total * q)
end

function interaction_energy(
    result::TensorInteractionResult,
    moments::AbstractMatrix,
)
    size(moments, 1) == 3 ||
        throw(DimensionMismatch("moments must be a 3×N matrix"))
    N = size(result.total, 1)
    size(moments, 2) == N ||
        throw(DimensionMismatch("moment count must match interaction tensor"))

    value = 0.0
    for i in 1:N
        mi = view(moments, :, i)
        for j in 1:N
            mj = view(moments, :, j)
            value += dot(mi, view(result.total, i, j, :, :) * mj)
        end
    end
    return 0.5 * value
end

"""
    project_to_local_frames(result, field, cluster)

Transform every pair tensor according to

    K_local(i,j) = R_i' K_global(i,j) R_j.

All decomposition pieces are transformed independently.
"""
function project_to_local_frames(
    result::TensorInteractionResult,
    field::LocalFrameField,
    cluster::Supercell,
)
    N = nsites(cluster)
    size(result.total, 1) == N ||
        throw(DimensionMismatch("interaction tensor does not match cluster"))

    function project(K)
        out = zeros(Float64, size(K))
        for i in 1:N
            Ri = local_frame(field, cluster, i).rotation
            for j in 1:N
                Rj = local_frame(field, cluster, j).rotation
                @views out[i, j, :, :] .=
                    transpose(Ri) * K[i, j, :, :] * Rj
            end
        end
        return out
    end

    metadata = copy(result.metadata)
    metadata[:coordinate_frame] = :local_site_frames
    return TensorInteractionResult(
        project(result.total),
        project(result.real_space),
        project(result.reciprocal_space),
        project(result.self),
        project(result.surface),
        metadata,
    )
end



"""
    ising_coupling_matrix(result, moment_vectors)

Contract a Cartesian tensor interaction into scalar local-Ising couplings:

    J_ij = m_i' K_ij m_j,

where column `i` of `moment_vectors` is the physical moment vector for
`σ_i=+1`. The returned matrix is directly compatible with the convention

    E(σ) = 1/2 σ' J σ + h' σ + constant

used by the classical sampling extension.
"""
function ising_coupling_matrix(
    result::TensorInteractionResult,
    moment_vectors::AbstractMatrix,
)
    N = size(result.total, 1)
    size(moment_vectors) == (3, N) ||
        throw(DimensionMismatch("moment_vectors must be a 3×N matrix"))
    moments = Matrix{Float64}(moment_vectors)
    J = zeros(Float64, N, N)

    for i in 1:N
        mi = view(moments, :, i)
        for j in i:N
            mj = view(moments, :, j)
            value = dot(mi, view(result.total, i, j, :, :) * mj)
            J[i, j] = value
            J[j, i] = value
        end
    end
    return J
end

"""
    ising_zeeman_field(moment_vectors, field; coupling=1)

Return the linear coefficients `h_i` for the classical convention

    E = ... + sum_i h_i σ_i

when the physical Zeeman energy is

    -coupling * sum_i (m_i · B) σ_i.
"""
function ising_zeeman_field(
    moment_vectors::AbstractMatrix,
    field::AbstractVector;
    coupling::Real=1.0,
)
    size(moment_vectors, 1) == 3 ||
        throw(DimensionMismatch("moment_vectors must be 3×N"))
    length(field) == 3 || throw(DimensionMismatch("field must have three components"))
    B = Float64.(field)
    return [
        -Float64(coupling) * dot(view(moment_vectors, :, i), B)
        for i in 1:size(moment_vectors, 2)
    ]
end

@inline function _spin_component_operator(component::Int, site::Int)
    component == 1 && return Sx(site)
    component == 2 && return Sy(site)
    component == 3 && return Sz(site)
    throw(BoundsError(1:3, component))
end

"""
    dipolar_hamiltonian(result; frames=nothing, cluster=nothing,
                         hbar=1, coefficient_tolerance=0)

Construct the symbolic quantum-spin Hamiltonian corresponding to the periodic
pair tensor. The tensor convention is

    H = 1/2 sum_ij sum_ab K_ij^{ab}
        (S_i^a/hbar) (S_j^b/hbar).

When `frames` is supplied, the global tensor is first projected to each site's
local frame, so the symbolic `Sx/Sy/Sz` operators are interpreted as local
components.

Diagonal `i==j` pair blocks are retained because they represent interactions
with periodic images and Ewald self corrections. Their explicit factor 1/2 is
required by the pair-energy convention.
"""
function dipolar_hamiltonian(
    result::TensorInteractionResult;
    frames=nothing,
    cluster=nothing,
    hbar::Real=1.0,
    coefficient_tolerance::Real=0.0,
)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    coefficient_tolerance >= 0 ||
        throw(ArgumentError("coefficient_tolerance must be nonnegative"))

    effective = if isnothing(frames)
        result
    else
        isnothing(cluster) && throw(ArgumentError("cluster is required with local frames"))
        project_to_local_frames(result, frames, cluster)
    end

    K = effective.total
    N = size(K, 1)
    terms = AbstractOperatorExpr[]
    invh2 = inv(hbar^2)

    for i in 1:N
        # Periodic self-image block. Summing all a,b with 1/2 automatically
        # symmetrizes cross components when Kii is symmetric.
        for a in 1:3, b in 1:3
            coeff = 0.5 * K[i, i, a, b] * invh2
            abs(coeff) <= coefficient_tolerance && continue
            push!(terms, coeff * (
                _spin_component_operator(a, i) *
                _spin_component_operator(b, i)
            ))
        end

        for j in (i + 1):N
            for a in 1:3, b in 1:3
                coeff = K[i, j, a, b] * invh2
                abs(coeff) <= coefficient_tolerance && continue
                push!(terms, coeff * (
                    _spin_component_operator(a, i) *
                    _spin_component_operator(b, j)
                ))
            end
        end
    end

    return _operator_sum(terms)
end
