"""Fractional-coordinate crystallographic structure used by the Phundamental symmetry layer."""
struct CrystalStructure
    lattice::Matrix{Float64}
    positions::Matrix{Float64}
    species::Vector{Int}
    labels::Vector{Symbol}

    function CrystalStructure(lattice::AbstractMatrix{<:Real}, positions, species::AbstractVector{<:Integer}; labels=nothing)
        size(lattice) == (3, 3) || throw(DimensionMismatch("lattice must be a 3×3 matrix with lattice vectors stored as columns"))
        pos = _position_matrix(positions)
        length(species) == size(pos, 2) || throw(DimensionMismatch("species count must equal the number of fractional positions"))
        names = isnothing(labels) ? [Symbol("site_", i) for i in eachindex(species)] : Symbol.(collect(labels))
        length(names) == length(species) || throw(DimensionMismatch("labels count must equal the number of sites"))
        length(unique(names)) == length(names) || throw(ArgumentError("crystal-site labels must be unique"))
        new(Matrix{Float64}(lattice), pos, Int.(collect(species)), names)
    end
end

function _position_matrix(positions::AbstractMatrix{<:Real})
    if size(positions, 1) == 3
        return Matrix{Float64}(positions)
    elseif size(positions, 2) == 3
        return Matrix{Float64}(permutedims(positions))
    end
    throw(DimensionMismatch("positions must be a 3×N or N×3 array of fractional coordinates"))
end

function _position_matrix(positions)
    vectors = collect(positions)
    isempty(vectors) && throw(ArgumentError("at least one crystal site is required"))
    all(v -> length(v) == 3, vectors) || throw(DimensionMismatch("each fractional position must contain three coordinates"))
    return hcat((Float64.(collect(v)) for v in vectors)...)
end

Base.length(structure::CrystalStructure) = length(structure.species)

"""Finite crystallographic operation `x -> W*x + w` in fractional coordinates."""
struct SpaceGroupOperation
    rotation::Matrix{Int}
    translation::Vector{Float64}
    time_reversal::Bool

    function SpaceGroupOperation(rotation::AbstractMatrix{<:Integer}, translation::AbstractVector{<:Real}; time_reversal::Bool=false)
        size(rotation) == (3, 3) || throw(DimensionMismatch("space-group rotation must be 3×3"))
        length(translation) == 3 || throw(DimensionMismatch("space-group translation must contain three components"))
        new(Matrix{Int}(rotation), Float64.(collect(translation)), time_reversal)
    end
end

"""Phundamental-owned crystallographic dataset detached from backend-specific types."""
struct CrystallographicDataset
    spacegroup_number::Int
    hall_number::Int
    international_symbol::String
    hall_symbol::String
    pointgroup_symbol::String
    operations::Vector{SpaceGroupOperation}
    wyckoffs::Vector{String}
    site_symmetry_symbols::Vector{String}
    equivalent_atoms::Vector{Int}
    crystallographic_orbits::Vector{Int}
    metadata::Dict{Symbol,Any}
end

function CrystallographicDataset(; spacegroup_number::Integer, hall_number::Integer=0, international_symbol::AbstractString="", hall_symbol::AbstractString="", pointgroup_symbol::AbstractString="", operations, wyckoffs=String[], site_symmetry_symbols=String[], equivalent_atoms=Int[], crystallographic_orbits=Int[], metadata=Dict{Symbol,Any}())
    ops = SpaceGroupOperation[operations...]
    isempty(ops) && throw(ArgumentError("a crystallographic dataset requires at least one symmetry operation"))
    return CrystallographicDataset(Int(spacegroup_number), Int(hall_number), String(international_symbol), String(hall_symbol), String(pointgroup_symbol), ops, String.(collect(wyckoffs)), String.(collect(site_symmetry_symbols)), Int.(collect(equivalent_atoms)), Int.(collect(crystallographic_orbits)), Dict{Symbol,Any}(metadata))
end

"""Backend-neutral configuration for symmetry discovery and invariant generation."""
struct CrystallographyOptions
    symprec::Float64
    site_tolerance::Float64
    linear_tolerance::Float64

    function CrystallographyOptions(; symprec::Real=1e-5, site_tolerance::Real=1e-7, linear_tolerance::Real=1e-10)
        symprec > 0 || throw(ArgumentError("symprec must be positive"))
        site_tolerance > 0 || throw(ArgumentError("site_tolerance must be positive"))
        linear_tolerance > 0 || throw(ArgumentError("linear_tolerance must be positive"))
        new(Float64(symprec), Float64(site_tolerance), Float64(linear_tolerance))
    end
end

wrap_fractional(x::Real) = mod(Float64(x), 1.0)
wrap_fractional(x::AbstractVector{<:Real}) = wrap_fractional.(x)

function fractional_displacement(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == 3 && length(b) == 3 || throw(DimensionMismatch("fractional displacements require three-component vectors"))
    delta = Float64.(collect(a .- b))
    delta .-= round.(delta)
    return delta
end

function cartesian_rotation(structure::CrystalStructure, operation::SpaceGroupOperation)
    A = structure.lattice
    return A * operation.rotation * inv(A)
end

function axial_rotation(structure::CrystalStructure, operation::SpaceGroupOperation)
    R = cartesian_rotation(structure, operation)
    return det(R) * R
end

function site_permutation(structure::CrystalStructure, operation::SpaceGroupOperation; atol::Real=1e-7)
    permutation = Vector{Int}(undef, length(structure))
    for i in 1:length(structure)
        transformed = wrap_fractional(operation.rotation * structure.positions[:, i] + operation.translation)
        matches = Int[]
        for j in 1:length(structure)
            structure.species[j] == structure.species[i] || continue
            norm(fractional_displacement(transformed, structure.positions[:, j])) <= atol && push!(matches, j)
        end
        length(matches) == 1 || throw(ArgumentError("symmetry operation does not map site $(i) uniquely onto the supplied crystal structure"))
        permutation[i] = only(matches)
    end
    length(unique(permutation)) == length(permutation) || throw(ArgumentError("symmetry operation does not induce a bijective site permutation"))
    return permutation
end
