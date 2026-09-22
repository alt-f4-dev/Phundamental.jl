# Primitive Bravais-lattice geometry.
# Direct lattice vectors are stored as columns of `direct`; reciprocal vectors
# are stored as columns of `reciprocal`, with A'B = 2*pi*I.

struct BravaisLattice{T<:AbstractFloat}
    direct::Matrix{T}
    reciprocal::Matrix{T}
    measure::T
end

function BravaisLattice(vectors::AbstractMatrix{<:Real}; vectors_as::Symbol=:columns)
    size(vectors, 1) == size(vectors, 2) ||
        throw(DimensionMismatch("Bravais lattice requires a square D×D vector matrix"))
    1 <= size(vectors, 1) <= 3 ||
        throw(ArgumentError("this implementation supports spatial dimensions D=1,2,3"))
    vectors_as in (:columns, :rows) ||
        throw(ArgumentError("vectors_as must be :columns or :rows"))

    T = float(eltype(vectors))
    A = Matrix{T}(vectors_as === :columns ? vectors : transpose(vectors))
    abs(det(A)) > sqrt(eps(T)) || throw(ArgumentError("direct lattice vectors are linearly dependent"))
    B = Matrix{T}(2 * T(pi) .* inv(transpose(A)))
    return BravaisLattice{T}(A, B, T(abs(det(A))))
end

function BravaisLattice(vectors::AbstractVector{<:AbstractVector{<:Real}})
    D = length(vectors)
    D > 0 || throw(ArgumentError("at least one lattice vector is required"))
    all(length(v) == D for v in vectors) ||
        throw(DimensionMismatch("each lattice vector must have D components"))
    A = hcat(vectors...)
    return BravaisLattice(A; vectors_as=:columns)
end

lattice_dimension(lattice::BravaisLattice) = size(lattice.direct, 1)
direct_matrix(lattice::BravaisLattice) = lattice.direct
reciprocal_matrix(lattice::BravaisLattice) = lattice.reciprocal
cell_measure(lattice::BravaisLattice) = lattice.measure

function _coordinate_vector(v, D::Int, name::AbstractString)
    length(v) == D || throw(DimensionMismatch("$name must have $D components"))
    return v
end

function fractional_to_cartesian(lattice::BravaisLattice, fractional)
    D = lattice_dimension(lattice)
    f = _coordinate_vector(fractional, D, "fractional coordinate")
    return lattice.direct * collect(f)
end

function cartesian_to_fractional(lattice::BravaisLattice, cartesian)
    D = lattice_dimension(lattice)
    r = _coordinate_vector(cartesian, D, "Cartesian coordinate")
    return lattice.direct \ collect(r)
end
