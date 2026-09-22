# Reciprocal-space decomposition utilities shared by phonon solvers and scattering APIs.

"""
    ReciprocalDecomposition

Canonical decomposition of a full physical scattering vector `Q` into a reduced crystal momentum `q` and reciprocal-lattice translation `G` satisfying `Q = q + G`.

Both Cartesian and reciprocal-lattice coordinates are retained so downstream solvers can use the reduced momentum while scattering amplitudes continue to use the full physical vector.
"""
struct ReciprocalDecomposition
    Q_cartesian::CartesianWaveVector{Float64}
    Q_reciprocal::ReciprocalWaveVector{Float64}
    q_cartesian::CartesianWaveVector{Float64}
    q_reciprocal::ReciprocalWaveVector{Float64}
    G_cartesian::CartesianWaveVector{Float64}
    G_reciprocal::ReciprocalWaveVector{Float64}
    G_indices::Vector{Int}
    convention::Symbol
    reconstruction_error::Float64
end

function _validate_reciprocal_matrix(reciprocal_matrix::AbstractMatrix{<:Real})
    B = Matrix{Float64}(reciprocal_matrix)
    size(B, 1) == size(B, 2) || throw(DimensionMismatch("reciprocal-lattice matrix must be square"))
    isempty(B) && throw(ArgumentError("reciprocal-lattice matrix cannot be empty"))
    all(isfinite, B) || throw(ArgumentError("reciprocal-lattice matrix must contain only finite values"))
    sigma_min = minimum(svdvals(B))
    sigma_min > eps(Float64) * max(opnorm(B), 1.0) || throw(ArgumentError("reciprocal-lattice matrix must be nonsingular"))
    return B, sigma_min
end

function _lexicographically_less(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    for index in eachindex(a, b)
        a[index] < b[index] && return true
        a[index] > b[index] && return false
    end
    return false
end

function _nearest_reciprocal_translation(B::Matrix{Float64}, h::Vector{Float64}, sigma_min::Float64; atol::Real=1e-12)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    initial = round.(Int, h)
    initial_delta = B * (h .- initial)
    best_distance_squared = dot(initial_delta, initial_delta)
    best = copy(initial)

    radius = sqrt(max(best_distance_squared, 0.0)) / sigma_min + max(Float64(atol), 8 * eps(Float64))
    ranges = [floor(Int, h[index] - radius):ceil(Int, h[index] + radius) for index in eachindex(h)]
    comparison_scale = max(opnorm(B)^2, best_distance_squared, 1.0)
    comparison_tolerance = Float64(atol) * comparison_scale

    for candidate_tuple in Iterators.product(ranges...)
        candidate = Int[candidate_tuple...]
        delta = B * (h .- candidate)
        distance_squared = dot(delta, delta)
        if distance_squared < best_distance_squared - comparison_tolerance ||
           (abs(distance_squared - best_distance_squared) <= comparison_tolerance && _lexicographically_less(candidate, best))
            best .= candidate
            best_distance_squared = distance_squared
        end
    end
    return best
end

function _reciprocal_input_coordinates(B::Matrix{Float64}, Q::ReciprocalWaveVector)
    h = Float64.(Q.coordinates)
    length(h) == size(B, 2) || throw(DimensionMismatch("reciprocal wavevector dimension does not match the reciprocal-lattice matrix"))
    return h, B * h
end

function _reciprocal_input_coordinates(B::Matrix{Float64}, Q::CartesianWaveVector)
    Qcart = Float64.(Q.coordinates)
    length(Qcart) == size(B, 1) || throw(DimensionMismatch("Cartesian wavevector dimension does not match the reciprocal-lattice matrix"))
    return B \ Qcart, Qcart
end

"""
    decompose_reciprocal_vector(reciprocal_matrix, Q; convention=:first_bz, atol=1e-12)

Decompose a full wavevector as `Q = q + G`, where `G` is a reciprocal-lattice vector and `q` is the representative nearest the origin in the reciprocal-lattice metric. This is the Wigner-Seitz first-Brillouin-zone reduction for a nondegenerate reciprocal basis, with deterministic lexicographic tie breaking on exact zone boundaries.

`Q` must be an explicit `ReciprocalWaveVector` or `CartesianWaveVector`; raw vectors are intentionally rejected because their coordinate basis is ambiguous.
"""
function decompose_reciprocal_vector(reciprocal_matrix::AbstractMatrix{<:Real}, Q::AbstractWaveVector; convention::Symbol=:first_bz, atol::Real=1e-12)
    convention === :first_bz || throw(ArgumentError("unsupported reciprocal reduction convention $convention; expected :first_bz"))
    B, sigma_min = _validate_reciprocal_matrix(reciprocal_matrix)
    h, Qcart = _reciprocal_input_coordinates(B, Q)
    all(isfinite, h) || throw(ArgumentError("wavevector coordinates must be finite"))

    G_indices = _nearest_reciprocal_translation(B, h, sigma_min; atol=atol)
    q_reciprocal_coordinates = h .- G_indices
    G_reciprocal_coordinates = Float64.(G_indices)
    q_cartesian_coordinates = B * q_reciprocal_coordinates
    G_cartesian_coordinates = B * G_reciprocal_coordinates
    reconstruction_error = norm(Qcart - (q_cartesian_coordinates + G_cartesian_coordinates))
    reconstruction_tolerance = max(Float64(atol), 32 * eps(Float64)) * max(norm(Qcart), opnorm(B), 1.0)
    reconstruction_error <= reconstruction_tolerance || throw(ArgumentError("reciprocal decomposition failed reconstruction tolerance; residual=$reconstruction_error"))

    return ReciprocalDecomposition(
        CartesianWaveVector(Qcart),
        ReciprocalWaveVector(h),
        CartesianWaveVector(q_cartesian_coordinates),
        ReciprocalWaveVector(q_reciprocal_coordinates),
        CartesianWaveVector(G_cartesian_coordinates),
        ReciprocalWaveVector(G_reciprocal_coordinates),
        G_indices,
        convention,
        reconstruction_error,
    )
end

"""
    decompose_reciprocal_vector(model, Q; convention=:first_bz, atol=1e-12)

Use reciprocal-lattice metadata retained by `model` to decompose a full physical wavevector into reduced crystal momentum and reciprocal translation.
"""
function decompose_reciprocal_vector(model::ManyBodyModel, Q::AbstractWaveVector; convention::Symbol=:first_bz, atol::Real=1e-12)
    reciprocal_matrix = get(model.parameters, :phonon_reciprocal_matrix, nothing)
    isnothing(reciprocal_matrix) && throw(ArgumentError("model does not retain reciprocal-lattice metadata under :phonon_reciprocal_matrix"))
    return decompose_reciprocal_vector(reciprocal_matrix, Q; convention=convention, atol=atol)
end
