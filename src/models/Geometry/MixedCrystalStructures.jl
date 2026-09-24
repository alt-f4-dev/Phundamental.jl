# Mixed chemical occupancy and explicit disorder realization.

using Random: AbstractRNG, default_rng, rand, shuffle!

"""Abstract chemical-occupancy description for one crystallographic site."""
abstract type AbstractOccupancy end

"""Definite occupation of a crystallographic site by one chemical species."""
struct SpeciesOccupancy <: AbstractOccupancy
    species::Symbol
end

SpeciesOccupancy(species::AbstractString) = SpeciesOccupancy(Symbol(species))

"""Probability distribution over two or more chemical species occupying one crystallographic site."""
struct MixedOccupancy <: AbstractOccupancy
    species::Vector{Symbol}
    probabilities::Vector{Float64}

    function MixedOccupancy(species, probabilities; atol::Real=1e-12)
        atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
        names = Symbol[s isa Symbol ? s : Symbol(string(s)) for s in species]
        weights = Float64.(collect(probabilities))
        length(names) == length(weights) || throw(DimensionMismatch("mixed-occupancy species and probabilities must have equal lengths"))
        length(names) >= 2 || throw(ArgumentError("MixedOccupancy requires at least two species"))
        length(unique(names)) == length(names) || throw(ArgumentError("mixed-occupancy species must be unique"))
        all(isfinite, weights) || throw(ArgumentError("mixed-occupancy probabilities must be finite"))
        all(weight -> weight > 0, weights) || throw(ArgumentError("mixed-occupancy probabilities must be strictly positive"))
        total = sum(weights)
        isapprox(total, 1.0; atol=atol, rtol=atol) || throw(ArgumentError("mixed-occupancy probabilities must sum to one"))
        new(names, weights ./ total)
    end
end

MixedOccupancy(entries::Pair...; kwargs...) = MixedOccupancy(first.(entries), last.(entries); kwargs...)
Base.length(occupancy::MixedOccupancy) = length(occupancy.species)

"""A crystallographic basis site whose chemical species is represented by a `MixedOccupancy`."""
struct MixedBasisSite{T<:AbstractFloat,P}
    label::Symbol
    occupancy::MixedOccupancy
    fractional::Vector{T}
    properties::P
end

function MixedBasisSite(label::Symbol, occupancy::MixedOccupancy, fractional::AbstractVector{<:Real}; properties=NamedTuple())
    T = float(eltype(fractional))
    return MixedBasisSite{T,typeof(properties)}(label, occupancy, T.(fractional), properties)
end

"""Crystal geometry containing at least one mixed-occupancy basis site while preserving ordinary `BasisSite` entries for chemically definite sites."""
struct MixedCrystalStructure{L<:BravaisLattice,B<:AbstractVector}
    lattice::L
    basis::B

    function MixedCrystalStructure(lattice::L, basis::B) where {L<:BravaisLattice,B<:AbstractVector}
        D = lattice_dimension(lattice)
        isempty(basis) && throw(ArgumentError("mixed crystal basis cannot be empty"))
        all(site -> site isa BasisSite || site isa MixedBasisSite, basis) || throw(ArgumentError("MixedCrystalStructure basis entries must be BasisSite or MixedBasisSite objects"))
        any(site -> site isa MixedBasisSite, basis) || throw(ArgumentError("MixedCrystalStructure requires at least one MixedBasisSite"))
        all(length(site.fractional) == D for site in basis) || throw(DimensionMismatch("every basis coordinate must match the lattice dimension"))
        labels = Symbol[site.label for site in basis]
        length(unique(labels)) == length(labels) || throw(ArgumentError("basis-site labels must be unique"))
        sites = collect(basis)
        return new{L,typeof(sites)}(lattice, sites)
    end
end

Base.length(crystal::MixedCrystalStructure) = length(crystal.basis)

occupancy(site::BasisSite) = SpeciesOccupancy(site.species)
occupancy(site::MixedBasisSite) = site.occupancy

function _occupancy_key(occupancy::MixedOccupancy)
    order = sortperm(eachindex(occupancy.species); by=i -> string(occupancy.species[i]))
    return Tuple((occupancy.species[i], occupancy.probabilities[i]) for i in order)
end

function _cell_from_linear(cell_index::Integer, repetitions::AbstractVector{<:Integer})
    value = Int(cell_index) - 1
    cell = Vector{Int}(undef, length(repetitions))
    @inbounds for d in eachindex(repetitions)
        cell[d] = value % repetitions[d]
        value ÷= repetitions[d]
    end
    return cell
end

function _exact_species_pool(occupancy::MixedOccupancy, count::Integer, rng::AbstractRNG)
    n = Int(count)
    n >= 0 || throw(ArgumentError("occupation count must be nonnegative"))
    targets = occupancy.probabilities .* n
    counts = floor.(Int, targets)
    remaining = n - sum(counts)
    residuals = targets .- counts
    priority = sortperm(collect(eachindex(counts)); by=i -> (-residuals[i], string(occupancy.species[i])))
    @inbounds for k in 1:remaining
        counts[priority[k]] += 1
    end
    pool = Symbol[]
    sizehint!(pool, n)
    for (species, species_count) in zip(occupancy.species, counts)
        append!(pool, fill(species, species_count))
    end
    shuffle!(rng, pool)
    return pool
end

function _sample_species(occupancy::MixedOccupancy, rng::AbstractRNG)
    draw = rand(rng)
    cumulative = 0.0
    @inbounds for i in eachindex(occupancy.species)
        cumulative += occupancy.probabilities[i]
        draw <= cumulative && return occupancy.species[i]
    end
    return last(occupancy.species)
end

function _realized_label(label::Symbol, cell::AbstractVector{<:Integer}, ncells::Integer)
    ncells == 1 && return label
    return Symbol(string(label), "__", join(cell, "_"))
end

"""
    disordered_supercell(crystal, repetitions; composition=:stochastic, rng=default_rng(), periodic=false)

Resolve every `MixedBasisSite` in `crystal` into a definite chemical species and return an ordinary downstream-compatible `Supercell`. The requested replication is encoded in an enlarged realized lattice and an explicit ordinary `BasisSite` basis, so the returned `Supercell` has unit internal repetitions.

`composition=:stochastic` samples every mixed site independently. `composition=:exact` groups mixed sites with identical occupancy distributions, applies finite-size largest-remainder apportionment to each group, and randomly assigns the resulting integer species counts across the group's realized sites. Exact finite supercells therefore reproduce the closest integer composition allowed by their size rather than inventing a virtual species.
"""
function disordered_supercell(crystal::MixedCrystalStructure, repetitions::AbstractVector{<:Integer}; composition::Symbol=:stochastic, rng::AbstractRNG=default_rng(), periodic=false)
    D = lattice_dimension(crystal.lattice)
    length(repetitions) == D || throw(DimensionMismatch("repetitions must contain one entry per lattice dimension"))
    reps = Int.(collect(repetitions))
    all(rep -> rep > 0, reps) || throw(ArgumentError("all supercell repetitions must be positive"))
    composition in (:stochastic, :exact) || throw(ArgumentError("composition must be :stochastic or :exact"))
    ncells = prod(reps)

    pools = Dict{Any,Vector{Symbol}}()
    pool_offsets = Dict{Any,Int}()
    if composition === :exact
        group_counts = Dict{Any,Int}()
        group_occupancies = Dict{Any,MixedOccupancy}()
        for site in crystal.basis
            site isa MixedBasisSite || continue
            key = _occupancy_key(site.occupancy)
            group_counts[key] = get(group_counts, key, 0) + ncells
            group_occupancies[key] = site.occupancy
        end
        for key in sort!(collect(keys(group_counts)); by=string)
            pools[key] = _exact_species_pool(group_occupancies[key], group_counts[key], rng)
            pool_offsets[key] = 0
        end
    end

    direct = crystal.lattice.direct * Diagonal(Float64.(reps))
    realized_lattice = BravaisLattice(direct)
    realized_basis = BasisSite[]
    sizehint!(realized_basis, length(crystal.basis) * ncells)

    for linear_cell in 1:ncells
        cell = _cell_from_linear(linear_cell, reps)
        for site in crystal.basis
            resolved_species = if site isa BasisSite
                site.species
            elseif composition === :stochastic
                _sample_species(site.occupancy, rng)
            else
                key = _occupancy_key(site.occupancy)
                offset = pool_offsets[key] + 1
                pool_offsets[key] = offset
                pools[key][offset]
            end
            fractional = (Float64.(cell) .+ Float64.(site.fractional)) ./ Float64.(reps)
            label = _realized_label(site.label, cell, ncells)
            push!(realized_basis, BasisSite(label, resolved_species, fractional; properties=site.properties))
        end
    end

    realized_crystal = CrystalStructure(realized_lattice, realized_basis)
    return Supercell(realized_crystal, ones(Int, D); periodic=periodic)
end

disordered_supercell(crystal::MixedCrystalStructure, repetitions::NTuple{N,<:Integer}; kwargs...) where {N} = disordered_supercell(crystal, collect(repetitions); kwargs...)
