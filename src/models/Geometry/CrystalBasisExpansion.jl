# Symmetry-driven construction of ordered and mixed crystallographic bases.

_basis_container_keys(data::NamedTuple) = collect(keys(data))

function _basis_container_keys(data::AbstractDict)
    labels = collect(keys(data))
    all(label -> label isa Symbol, labels) || throw(ArgumentError("crystal-basis dictionaries must use Symbol keys"))
    return sort!(Symbol.(labels); by=string)
end

_basis_container_keys(data) = throw(ArgumentError("crystal-basis data must be supplied as a NamedTuple or AbstractDict"))
_basis_container_haskey(data::NamedTuple, label::Symbol) = label in keys(data)
_basis_container_haskey(data::AbstractDict, label::Symbol) = haskey(data, label)
_basis_container_get(data::NamedTuple, label::Symbol) = getproperty(data, label)
_basis_container_get(data::AbstractDict, label::Symbol) = data[label]

function _validate_basis_data(irreps, occupancies, site_properties)
    irrep_labels = _basis_container_keys(irreps)
    occupancy_labels = _basis_container_keys(occupancies)
    property_labels = _basis_container_keys(site_properties)

    irrep_set = Set(irrep_labels)
    irrep_set == Set(occupancy_labels) || throw(ArgumentError("irreps and occupancies must define the same site labels"))
    all(label -> label in irrep_set, property_labels) || throw(ArgumentError("site_properties contains labels that are absent from irreps"))
    return irrep_labels
end

function _expanded_basis_site(label::Symbol, occupation::MixedOccupancy, position, properties)
    return MixedBasisSite(label, occupation, position; properties=properties)
end

function _expanded_basis_site(label::Symbol, occupation::SpeciesOccupancy, position, properties)
    return BasisSite(label, occupation.species, position; properties=properties)
end

function _expanded_basis_site(label::Symbol, species::Symbol, position, properties)
    return BasisSite(label, species, position; properties=properties)
end

function _expanded_basis_site(label::Symbol, species::AbstractString, position, properties)
    return BasisSite(label, Symbol(species), position; properties=properties)
end

function _expanded_basis_site(label::Symbol, occupation, position, properties)
    throw(ArgumentError("unsupported occupancy specification $(repr(occupation)) for basis site $label; expected Symbol, AbstractString, SpeciesOccupancy, or MixedOccupancy"))
end

"""
    expand_crystal_basis(operations, irreps, occupancies, site_properties=NamedTuple(); atol=1e-8)

Expand one asymmetric-unit representative per site label into a complete crystallographic basis using explicit space-group `operations`. `irreps` and `occupancies` must be `NamedTuple` or `AbstractDict` containers with identical `Symbol` keys. `site_properties` may omit labels, in which case the generated sites receive an empty `NamedTuple` as their properties.

Definite occupations supplied as a species `Symbol`, species string, or `SpeciesOccupancy` generate ordinary `BasisSite` objects. `MixedOccupancy` values generate `MixedBasisSite` objects. Generated labels are formed as `<label>_<orbit index>`.
"""
function expand_crystal_basis(operations::AbstractVector{<:SpaceGroupOperation}, irreps, occupancies, site_properties=NamedTuple(); atol::Real=1e-8)
    labels = _validate_basis_data(irreps, occupancies, site_properties)
    basis = Union{BasisSite,MixedBasisSite}[]

    for label in labels
        representative = _basis_container_get(irreps, label)
        occupation = _basis_container_get(occupancies, label)
        properties = _basis_container_haskey(site_properties, label) ? _basis_container_get(site_properties, label) : NamedTuple()
        orbit = expand_symmetry_orbit(representative, operations; atol=atol)

        for (orbit_index, position) in enumerate(orbit)
            site_label = Symbol(string(label), "_", string(orbit_index))
            push!(basis, _expanded_basis_site(site_label, occupation, position, properties))
        end
    end

    return basis
end

"""
    expand_crystal_basis(spacegroup_number, irreps, occupancies, site_properties=NamedTuple(); hall_number=nothing, choice=nothing, atol=1e-8)

Construct a complete crystallographic basis directly from an international space-group number and asymmetric-unit data. The requested Hall setting is resolved through `space_group_operations`; `hall_number` and `choice` are mutually exclusive.
"""
function expand_crystal_basis(spacegroup_number::Integer, irreps, occupancies, site_properties=NamedTuple(); hall_number::Union{Nothing,Integer}=nothing, choice=nothing, atol::Real=1e-8)
    operations = space_group_operations(spacegroup_number; hall_number=hall_number, choice=choice)
    return expand_crystal_basis(operations, irreps, occupancies, site_properties; atol=atol)
end
