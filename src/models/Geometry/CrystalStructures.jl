# Atomic basis and finite supercell realization of a crystal structure.

struct BasisSite{T<:AbstractFloat,P}
    label::Symbol
    species::Symbol
    fractional::Vector{T}
    properties::P
end

function BasisSite(
    label::Symbol,
    species::Symbol,
    fractional::AbstractVector{<:Real};
    properties=NamedTuple(),
)
    T = float(eltype(fractional))
    return BasisSite{T,typeof(properties)}(label, species, T.(fractional), properties)
end

struct CrystalStructure{L<:BravaisLattice,B<:AbstractVector}
    lattice::L
    basis::B
end

function CrystalStructure(lattice::BravaisLattice, basis::AbstractVector{<:BasisSite})
    D = lattice_dimension(lattice)
    isempty(basis) && throw(ArgumentError("crystal basis cannot be empty"))
    all(length(site.fractional) == D for site in basis) ||
        throw(DimensionMismatch("every basis coordinate must match the lattice dimension"))
    labels = Symbol[site.label for site in basis]
    length(unique(labels)) == length(labels) ||
        throw(ArgumentError("basis-site labels must be unique"))
    return CrystalStructure{typeof(lattice),typeof(collect(basis))}(lattice, collect(basis))
end

struct CrystalSite{T<:AbstractFloat}
    index::Int
    basis_index::Int
    cell::Vector{Int}
    label::Symbol
    species::Symbol
    fractional::Vector{T}
    cartesian::Vector{T}
end

struct Supercell{C<:CrystalStructure,T<:AbstractFloat}
    crystal::C
    repetitions::Vector{Int}
    periodic::BitVector
    sites::Vector{CrystalSite{T}}
end

function _linear_cell_index(cell::AbstractVector{<:Integer}, repetitions::AbstractVector{<:Integer})
    length(cell) == length(repetitions) || throw(DimensionMismatch("cell/repetition dimensions differ"))
    idx = 1
    stride = 1
    @inbounds for d in eachindex(repetitions)
        0 <= cell[d] < repetitions[d] || throw(BoundsError(cell, d))
        idx += Int(cell[d]) * stride
        stride *= Int(repetitions[d])
    end
    return idx
end

function site_index(supercell::Supercell, cell, basis_index::Integer)
    nb = length(supercell.crystal.basis)
    1 <= basis_index <= nb || throw(BoundsError(supercell.crystal.basis, basis_index))
    cidx = _linear_cell_index(Int.(collect(cell)), supercell.repetitions)
    return (cidx - 1) * nb + Int(basis_index)
end

function Supercell(
    crystal::CrystalStructure,
    repetitions::AbstractVector{<:Integer};
    periodic=false,
)
    D = lattice_dimension(crystal.lattice)
    length(repetitions) == D || throw(DimensionMismatch("repetitions must contain one entry per lattice dimension"))
    reps = Int.(collect(repetitions))
    all(>(0), reps) || throw(ArgumentError("all supercell repetitions must be positive"))

    pbc = if periodic isa Bool
        BitVector(fill(periodic, D))
    else
        length(periodic) == D || throw(DimensionMismatch("periodic flags must contain one entry per dimension"))
        BitVector(periodic)
    end

    T = eltype(crystal.lattice.direct)
    nb = length(crystal.basis)
    total = nb * prod(reps)
    records = Vector{CrystalSite{T}}()
    sizehint!(records, total)

    # `_linear_cell_index` uses first coordinate as the fastest-running cell
    # dimension. Iterators.product follows the same nesting when collected in
    # this explicit recursion.
    cell = zeros(Int, D)
    function emit_cells!(d::Int)
        if d > D
            for (mu, basis_site) in enumerate(crystal.basis)
                frac = T.(cell) .+ T.(basis_site.fractional)
                cart = crystal.lattice.direct * frac
                idx = site_index_placeholder(cell, mu)
                push!(records, CrystalSite{T}(
                    idx, mu, copy(cell), basis_site.label, basis_site.species,
                    frac, cart,
                ))
            end
            return
        end
        for n in 0:(reps[d] - 1)
            cell[d] = n
            emit_cells!(d + 1)
        end
    end

    # Avoid constructing a partially initialized Supercell just to call
    # site_index while records are emitted.
    site_index_placeholder(c, mu) = (_linear_cell_index(c, reps) - 1) * nb + mu
    emit_cells!(1)
    sort!(records, by=s -> s.index)

    return Supercell{typeof(crystal),T}(crystal, reps, pbc, records)
end

Supercell(crystal::CrystalStructure, repetitions::NTuple{N,<:Integer}; kwargs...) where {N} =
    Supercell(crystal, collect(repetitions); kwargs...)

nsites(supercell::Supercell) = length(supercell.sites)
site_position(supercell::Supercell, i::Integer) = supercell.sites[Int(i)].cartesian
site_positions(supercell::Supercell) = [site.cartesian for site in supercell.sites]
