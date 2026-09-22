# LocalFrames.jl
#
# Site-dependent right-handed orthonormal coordinate frames.
#
# A LocalFrame stores the proper rotation R whose columns are the local basis
# vectors expressed in global Cartesian coordinates:
#
#     v_global = R * v_local
#     v_local  = R' * v_global
#
# Frames are geometry. They do not change the model's operator algebra or
# constitute a representation transformation.

"""
    LocalFrame(rotation; atol=1e-12, rtol=1e-10)

A right-handed orthonormal 3D local coordinate frame. `rotation[:,a]` is the
local basis vector `e_a` expressed in global Cartesian coordinates.
"""
struct LocalFrame{T<:AbstractFloat}
    rotation::Matrix{T}
end

function LocalFrame(
    rotation::AbstractMatrix{<:Real};
    atol::Real=1e-12,
    rtol::Real=1e-10,
)
    size(rotation) == (3, 3) ||
        throw(DimensionMismatch("a LocalFrame rotation must be 3×3"))
    R = Matrix{Float64}(rotation)
    all(isfinite, R) || throw(ArgumentError("LocalFrame entries must be finite"))

    gram = transpose(R) * R
    isapprox(gram, Matrix{Float64}(I, 3, 3); atol=atol, rtol=rtol) ||
        throw(ArgumentError("LocalFrame basis vectors must be orthonormal"))

    d = det(R)
    isapprox(d, 1.0; atol=max(atol, 10eps(Float64)), rtol=rtol) ||
        throw(ArgumentError(
            "LocalFrame must be right-handed with det(R)=+1; got det(R)=$d"
        ))
    return LocalFrame{Float64}(R)
end

"""
    LocalFrame(ex, ey, ez; kwargs...)

Construct a frame from three global Cartesian unit vectors.
"""
function LocalFrame(ex, ey, ez; kwargs...)
    length(ex) == length(ey) == length(ez) == 3 ||
        throw(DimensionMismatch("LocalFrame axes must be three-dimensional"))
    return LocalFrame(hcat(Float64.(ex), Float64.(ey), Float64.(ez)); kwargs...)
end

"""
    LocalFrame(zaxis; xhint=nothing, atol=1e-12)

Construct a stable right-handed frame with the requested local `z` direction.
`xhint`, when supplied, is projected perpendicular to `zaxis`; otherwise the
global Cartesian direction least parallel to `zaxis` is selected automatically.
"""
function LocalFrame(
    zaxis::AbstractVector{<:Real};
    xhint=nothing,
    atol::Real=1e-12,
    rtol::Real=1e-10,
)
    length(zaxis) == 3 || throw(DimensionMismatch("zaxis must have length three"))
    z = Float64.(zaxis)
    nz = norm(z)
    nz > atol || throw(ArgumentError("zaxis must have nonzero norm"))
    z ./= nz

    h = if isnothing(xhint)
        # Choose the Cartesian axis least aligned with z. This avoids the loss
        # of precision that occurs when Gram-Schmidt starts from a nearly
        # parallel reference direction.
        candidates = (
            Float64[1, 0, 0],
            Float64[0, 1, 0],
            Float64[0, 0, 1],
        )
        candidates[argmin([abs(dot(c, z)) for c in candidates])]
    else
        length(xhint) == 3 || throw(DimensionMismatch("xhint must have length three"))
        Float64.(xhint)
    end

    x = h .- dot(h, z) .* z
    nx = norm(x)
    nx > atol || throw(ArgumentError("xhint is parallel to zaxis"))
    x ./= nx

    y = cross(z, x)
    ny = norm(y)
    ny > atol || throw(ArgumentError("failed to construct local y axis"))
    y ./= ny

    # Recompute x from y×z to suppress accumulated Gram-Schmidt roundoff.
    x = cross(y, z)
    x ./= norm(x)

    return LocalFrame(x, y, z; atol=atol, rtol=rtol)
end

local_x(frame::LocalFrame) = copy(view(frame.rotation, :, 1))
local_y(frame::LocalFrame) = copy(view(frame.rotation, :, 2))
local_z(frame::LocalFrame) = copy(view(frame.rotation, :, 3))

global_vector(frame::LocalFrame, v::AbstractVector) = frame.rotation * v
local_vector(frame::LocalFrame, v::AbstractVector) = transpose(frame.rotation) * v

"""
    global_tensor(frame, Tlocal)
    local_tensor(frame, Tglobal)

Transform a rank-2 Cartesian tensor between local and global coordinates.
"""
function global_tensor(frame::LocalFrame, tensor::AbstractMatrix)
    size(tensor) == (3, 3) || throw(DimensionMismatch("tensor must be 3×3"))
    return frame.rotation * tensor * transpose(frame.rotation)
end

function local_tensor(frame::LocalFrame, tensor::AbstractMatrix)
    size(tensor) == (3, 3) || throw(DimensionMismatch("tensor must be 3×3"))
    return transpose(frame.rotation) * tensor * frame.rotation
end

"""
    global_pair_tensor(frame_i, Klocal, frame_j)
    local_pair_tensor(frame_i, Kglobal, frame_j)

Transform a two-site bilinear coupling

    v_i' K v_j

between independently oriented local frames.
"""
function global_pair_tensor(
    frame_i::LocalFrame,
    tensor::AbstractMatrix,
    frame_j::LocalFrame,
)
    size(tensor) == (3, 3) || throw(DimensionMismatch("pair tensor must be 3×3"))
    return frame_i.rotation * tensor * transpose(frame_j.rotation)
end

function local_pair_tensor(
    frame_i::LocalFrame,
    tensor::AbstractMatrix,
    frame_j::LocalFrame,
)
    size(tensor) == (3, 3) || throw(DimensionMismatch("pair tensor must be 3×3"))
    return transpose(frame_i.rotation) * tensor * frame_j.rotation
end

"""
    LocalFrameField

Associates local frames either with crystallographic basis sites (`scope=:basis`)
or with explicit finite-supercell sites (`scope=:site`). The existing
`BasisSite`, `SiteProperties`, and `Supercell` layouts are intentionally left
unchanged.
"""
struct LocalFrameField{F<:AbstractVector}
    frames::F
    scope::Symbol

    function LocalFrameField(frames::AbstractVector{<:LocalFrame}, scope::Symbol)
        isempty(frames) && throw(ArgumentError("LocalFrameField cannot be empty"))
        scope in (:basis, :site) ||
            throw(ArgumentError("LocalFrameField scope must be :basis or :site"))
        new{typeof(collect(frames))}(collect(frames), scope)
    end
end

LocalFrameField(crystal::CrystalStructure, frames::AbstractVector{<:LocalFrame}) =
    length(frames) == length(crystal.basis) ?
        LocalFrameField(frames, :basis) :
        throw(DimensionMismatch("basis LocalFrameField requires one frame per basis site"))

LocalFrameField(cluster::Supercell, frames::AbstractVector{<:LocalFrame}) =
    length(frames) == nsites(cluster) ?
        LocalFrameField(frames, :site) :
        throw(DimensionMismatch("site LocalFrameField requires one frame per supercell site"))

function local_frame(field::LocalFrameField, cluster::Supercell, i::Integer)
    1 <= i <= nsites(cluster) || throw(BoundsError(cluster.sites, i))
    if field.scope === :site
        length(field.frames) == nsites(cluster) ||
            throw(DimensionMismatch("site LocalFrameField does not match this supercell"))
        return field.frames[Int(i)]
    end

    length(field.frames) == length(cluster.crystal.basis) ||
        throw(DimensionMismatch("basis LocalFrameField does not match this crystal basis"))
    return field.frames[cluster.sites[Int(i)].basis_index]
end

local_frames(field::LocalFrameField, cluster::Supercell) =
    [local_frame(field, cluster, i) for i in 1:nsites(cluster)]

local_z_axes(field::LocalFrameField, cluster::Supercell) =
    hcat((local_z(local_frame(field, cluster, i)) for i in 1:nsites(cluster))...)

"""
    frame_moment_vectors(field, cluster; magnitudes=1)

Return a 3×N matrix whose column `i` is the positive local-z moment vector for
site `i`. This is convenient for local-Ising classical ensembles and neutron
structure factors.
"""
function frame_moment_vectors(
    field::LocalFrameField,
    cluster::Supercell;
    magnitudes=1.0,
)
    N = nsites(cluster)
    mags = if magnitudes isa Number
        fill(Float64(magnitudes), N)
    else
        length(magnitudes) == N ||
            throw(DimensionMismatch("magnitudes must contain one value per site"))
        Float64.(collect(magnitudes))
    end

    out = zeros(Float64, 3, N)
    for i in 1:N
        out[:, i] .= mags[i] .* local_z(local_frame(field, cluster, i))
    end
    return out
end

"""
    validate_local_frames(field, cluster; atol=1e-11)

Return numerical orthonormality, handedness, and round-trip diagnostics.
"""
function validate_local_frames(
    field::LocalFrameField,
    cluster::Supercell;
    atol::Real=1e-11,
)
    worst_orthogonality = 0.0
    worst_determinant = 0.0
    worst_vector_roundtrip = 0.0
    worst_tensor_roundtrip = 0.0

    testv = Float64[0.371, -0.217, 0.901]
    testT = Float64[
        1.2   0.1  -0.3
        0.1   2.1   0.4
       -0.3   0.4   0.8
    ]

    for f in local_frames(field, cluster)
        R = f.rotation
        worst_orthogonality = max(
            worst_orthogonality,
            norm(transpose(R) * R - Matrix{Float64}(I, 3, 3)),
        )
        worst_determinant = max(worst_determinant, abs(det(R) - 1))
        worst_vector_roundtrip = max(
            worst_vector_roundtrip,
            norm(local_vector(f, global_vector(f, testv)) - testv),
        )
        worst_tensor_roundtrip = max(
            worst_tensor_roundtrip,
            norm(local_tensor(f, global_tensor(f, testT)) - testT),
        )
    end

    passed = maximum((
        worst_orthogonality,
        worst_determinant,
        worst_vector_roundtrip,
        worst_tensor_roundtrip,
    )) <= atol

    return Dict{Symbol,Any}(
        :passed => passed,
        :orthogonality_error => worst_orthogonality,
        :determinant_error => worst_determinant,
        :vector_roundtrip_error => worst_vector_roundtrip,
        :tensor_roundtrip_error => worst_tensor_roundtrip,
    )
end
