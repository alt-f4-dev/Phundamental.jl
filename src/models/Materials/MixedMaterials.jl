# Property resolution for mixed occupancies and realized disordered cells.

function _occupancy_species(database, symbol::Symbol)
    haskey(database, symbol) || throw(KeyError(symbol))
    species = database[symbol]
    species isa AtomicSpecies || throw(ArgumentError("species entries must be AtomicSpecies"))
    return species
end

mean_mass(occupancy::SpeciesOccupancy, database) = _occupancy_species(database, occupancy.species).mass
mean_scattering_length(occupancy::SpeciesOccupancy, database) = _occupancy_species(database, occupancy.species).nuclear_scattering_length

function mean_mass(occupancy::MixedOccupancy, database)
    return sum(probability * _occupancy_species(database, species).mass for (species, probability) in zip(occupancy.species, occupancy.probabilities))
end

function mean_scattering_length(occupancy::MixedOccupancy, database)
    return sum(probability * _occupancy_species(database, species).nuclear_scattering_length for (species, probability) in zip(occupancy.species, occupancy.probabilities))
end

mean_mass(site::Union{BasisSite,MixedBasisSite}, database) = mean_mass(occupancy(site), database)
mean_scattering_length(site::Union{BasisSite,MixedBasisSite}, database) = mean_scattering_length(occupancy(site), database)

"""Construct a `Material` directly from the explicit realized crystal stored in a `Supercell`."""
Material(name::AbstractString, supercell::Supercell; kwargs...) = Material(name, supercell.crystal; kwargs...)
