# Symmetry-driven construction of ordered and mixed crystallographic bases.

const _SiteLabelSequence = Union{Tuple{Vararg{Symbol}},AbstractVector{Symbol}}

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

function _basis_sequence_values(values, expected::Int, name::AbstractString)
    (values isa Tuple || values isa AbstractVector) || throw(ArgumentError("$name must be a tuple or vector with one entry per site label"))
    length(values) == expected || throw(DimensionMismatch("$name must contain one entry per site label"))
    return collect(values)
end

function _basis_sequence_containers(site_labels::_SiteLabelSequence, irreps; species=nothing, properties=nothing)
    labels = collect(site_labels)
    isempty(labels) && throw(ArgumentError("site_labels cannot be empty"))
    length(unique(labels)) == length(labels) || throw(ArgumentError("site_labels must be unique"))

    representatives = _basis_sequence_values(irreps, length(labels), "irreps")
    all(irrep -> irrep isa AbstractVector{<:Real}, representatives) || throw(ArgumentError("irreps must contain one real coordinate vector per site label"))
    occupations = isnothing(species) ? copy(labels) : _basis_sequence_values(species, length(labels), "species")
    property_values = isnothing(properties) ? nothing : _basis_sequence_values(properties, length(labels), "properties")

    names = Tuple(labels)
    irrep_data = NamedTuple{names}(Tuple(representatives))
    occupancy_data = NamedTuple{names}(Tuple(occupations))
    property_data = isnothing(property_values) ? NamedTuple() : NamedTuple{names}(Tuple(property_values))
    return irrep_data, occupancy_data, property_data
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
    expand_crystal_basis(operations, site_label, irrep; species=nothing, properties=NamedTuple(), atol=1e-8)

Expand one asymmetric-unit representative into a complete crystallographic basis. `site_label` is the crystallographic site identity, while `species` is the chemical identity or occupancy specification. When `species=nothing`, the site label is also used as the species label.
"""
function expand_crystal_basis(operations::AbstractVector{<:SpaceGroupOperation}, site_label::Symbol, irrep::AbstractVector{<:Real}; species=nothing, properties=NamedTuple(), atol::Real=1e-8)
    occupation = isnothing(species) ? site_label : species
    return expand_crystal_basis(operations, (site_label,), (irrep,); species=(occupation,), properties=(properties,), atol=atol)
end

"""
    expand_crystal_basis(operations, site_labels, irreps; species=nothing, properties=nothing, atol=1e-8)

Expand parallel collections of crystallographic site labels and asymmetric-unit representatives into a complete basis. `site_labels` must be unique. `species`, when supplied, must contain one chemical identity or occupancy specification per site label; otherwise the site labels are used as species labels. `properties`, when supplied, must contain one property object per site label.
"""
function expand_crystal_basis(operations::AbstractVector{<:SpaceGroupOperation}, site_labels::_SiteLabelSequence, irreps; species=nothing, properties=nothing, atol::Real=1e-8)
    irrep_data, occupancy_data, property_data = _basis_sequence_containers(site_labels, irreps; species=species, properties=properties)
    return expand_crystal_basis(operations, irrep_data, occupancy_data, property_data; atol=atol)
end

"""
    expand_crystal_basis(spacegroup_number, irreps, occupancies, site_properties=NamedTuple(); hall_number=nothing, choice=nothing, atol=1e-8)

Construct a complete crystallographic basis directly from an international space-group number and asymmetric-unit data. The requested Hall setting is resolved through `space_group_operations`; `hall_number` and `choice` are mutually exclusive.
"""
function expand_crystal_basis(spacegroup_number::Integer, irreps, occupancies, site_properties=NamedTuple(); hall_number::Union{Nothing,Integer}=nothing, choice=nothing, atol::Real=1e-8)
    operations = space_group_operations(spacegroup_number; hall_number=hall_number, choice=choice)
    return expand_crystal_basis(operations, irreps, occupancies, site_properties; atol=atol)
end

"""
    expand_crystal_basis(spacegroup_number, site_label, irrep; species=nothing, properties=NamedTuple(), hall_number=nothing, choice=nothing, atol=1e-8)

Construct a single-site crystallographic basis directly from an international space-group number. The requested Hall setting is resolved through `space_group_operations`; `hall_number` and `choice` are mutually exclusive.
"""
function expand_crystal_basis(spacegroup_number::Integer, site_label::Symbol, irrep::AbstractVector{<:Real}; species=nothing, properties=NamedTuple(), hall_number::Union{Nothing,Integer}=nothing, choice=nothing, atol::Real=1e-8)
    operations = space_group_operations(spacegroup_number; hall_number=hall_number, choice=choice)
    return expand_crystal_basis(operations, site_label, irrep; species=species, properties=properties, atol=atol)
end

"""
    expand_crystal_basis(spacegroup_number, site_labels, irreps; species=nothing, properties=nothing, hall_number=nothing, choice=nothing, atol=1e-8)

Construct a multi-site crystallographic basis directly from an international space-group number using parallel site-label, representative-coordinate, species, and property collections. The requested Hall setting is resolved through `space_group_operations`; `hall_number` and `choice` are mutually exclusive.
"""
function expand_crystal_basis(spacegroup_number::Integer, site_labels::_SiteLabelSequence, irreps; species=nothing, properties=nothing, hall_number::Union{Nothing,Integer}=nothing, choice=nothing, atol::Real=1e-8)
    operations = space_group_operations(spacegroup_number; hall_number=hall_number, choice=choice)
    return expand_crystal_basis(operations, site_labels, irreps; species=species, properties=properties, atol=atol)
end
