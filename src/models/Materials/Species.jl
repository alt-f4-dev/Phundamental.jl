# Chemical/isotopic species metadata used by material definitions.

struct AtomicSpecies
    symbol::Symbol
    mass::Float64
    nuclear_scattering_length::ComplexF64
    metadata::Dict{Symbol,Any}
end

function AtomicSpecies(
    symbol::Symbol;
    mass::Real,
    nuclear_scattering_length::Number=0.0,
    metadata=Dict{Symbol,Any}(),
)
    mass > 0 || throw(ArgumentError("atomic mass must be positive"))
    return AtomicSpecies(
        symbol,
        Float64(mass),
        ComplexF64(nuclear_scattering_length),
        Dict{Symbol,Any}(metadata),
    )
end

@inline function _lookup_element_symbol(symbol::Symbol)
    symbol === :D && return (:H, 2)
    symbol === :T && return (:H, 3)
    return (symbol, nothing)
end

"""
    lookup_species(symbol::Symbol)
    lookup_species(symbol::Symbol, isotope::Integer)

Construct an `AtomicSpecies` from Phundamental's bundled, version-stable
tabulated atomic data.

`lookup_species(:O)` uses the natural-abundance mass and coherent neutron
scattering length. `lookup_species(:O, 18)` uses isotope-specific values and
fails if either quantity is absent from the bundled isotope tables.

The returned `AtomicSpecies` remains an ordinary explicit data object: callers
may always construct `AtomicSpecies(...)` directly when measured, enriched, or
refined values should override the bundled tables.

Masses are in atomic mass units (amu). Coherent bound neutron scattering
lengths are in femtometres (fm).
"""
function lookup_species(symbol::Symbol)
    element, tagged_isotope = _lookup_element_symbol(symbol)
    !isnothing(tagged_isotope) && return lookup_species(element, tagged_isotope)

    haskey(_MASS_U, element) ||
        throw(KeyError("no natural-abundance mass is tabulated for element $element"))
    haskey(_BCOH_FM, element) ||
        throw(KeyError("no natural-abundance coherent neutron scattering length is tabulated for element $element"))

    metadata = Dict{Symbol,Any}(
        :composition => :natural_abundance,
        :isotope_mass_number => nothing,
        :mass_unit => :amu,
        :nuclear_scattering_length_unit => :fm,
        :mass_source => :phunny_pubchem_table,
        :coherent_scattering_length_source => :phunny_nist_table,
        :tabulated_lookup => true,
    )
    return AtomicSpecies(
        element;
        mass=_MASS_U[element],
        nuclear_scattering_length=_BCOH_FM[element],
        metadata=metadata,
    )
end

function lookup_species(symbol::Symbol, isotope::Integer)
    isotope > 0 || throw(ArgumentError("isotope mass number must be positive"))
    element, tagged_isotope = _lookup_element_symbol(symbol)
    if !isnothing(tagged_isotope) && tagged_isotope != isotope
        throw(ArgumentError("symbol $symbol already denotes isotope $tagged_isotope"))
    end

    key = (element, Int(isotope))
    haskey(_MASS_ISO_U, key) ||
        throw(KeyError("no isotope-specific mass is tabulated for $(element)-$(isotope)"))
    haskey(_BCOH_ISO_FM, key) ||
        throw(KeyError("no isotope-specific coherent neutron scattering length is tabulated for $(element)-$(isotope)"))

    metadata = Dict{Symbol,Any}(
        :composition => :isotope,
        :isotope_mass_number => Int(isotope),
        :mass_unit => :amu,
        :nuclear_scattering_length_unit => :fm,
        :mass_source => :phunny_isotope_mass_table,
        :coherent_scattering_length_source => :phunny_nist_isotope_table,
        :tabulated_lookup => true,
    )
    return AtomicSpecies(
        element;
        mass=_MASS_ISO_U[key],
        nuclear_scattering_length=_BCOH_ISO_FM[key],
        metadata=metadata,
    )
end
