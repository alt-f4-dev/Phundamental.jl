# Material = crystal geometry + species database + basis-site physical data.

struct SiteProperties
    spin::Union{Nothing,Float64}
    g_tensor::Union{Nothing,Matrix{Float64}}
    magnetic_form_factor::AbstractMagneticFormFactor
    debye_waller::Union{Nothing,DebyeWallerTensor}
    metadata::Dict{Symbol,Any}
end

function SiteProperties(
    ; spin=nothing,
      g_tensor=nothing,
      magnetic_form_factor::AbstractMagneticFormFactor=ConstantFormFactor(),
      debye_waller=nothing,
      metadata=Dict{Symbol,Any}(),
)
    s = isnothing(spin) ? nothing : Float64(spin)
    !isnothing(s) && s < 0 && throw(ArgumentError("spin quantum number must be nonnegative"))
    if !isnothing(s)
        isinteger(2s) || throw(ArgumentError("spin quantum number must be integer or half-integer"))
    end

    g = if isnothing(g_tensor)
        nothing
    elseif g_tensor isa Number
        Matrix{Float64}(I, 3, 3) .* Float64(g_tensor)
    else
        G = Float64.(Matrix(g_tensor))
        size(G) == (3, 3) || throw(DimensionMismatch("g tensor must be 3×3"))
        G
    end

    !isnothing(debye_waller) && !(debye_waller isa DebyeWallerTensor) &&
        throw(ArgumentError("debye_waller must be DebyeWallerTensor or nothing"))

    return SiteProperties(
        s, g, magnetic_form_factor, debye_waller, Dict{Symbol,Any}(metadata),
    )
end

struct Material{C<:CrystalStructure}
    name::String
    crystal::C
    species::Dict{Symbol,AtomicSpecies}
    basis_properties::Vector{SiteProperties}
    metadata::Dict{Symbol,Any}
end

function Material(
    name::AbstractString,
    crystal::CrystalStructure;
    species,
    basis_properties=nothing,
    metadata=Dict{Symbol,Any}(),
)
    species_db = Dict{Symbol,AtomicSpecies}()
    for (key, value) in pairs(species)
        value isa AtomicSpecies || throw(ArgumentError("species entries must be AtomicSpecies"))
        species_db[key isa Symbol ? key : Symbol(string(key))] = value
    end
    for site in crystal.basis
        haskey(species_db, site.species) ||
            throw(ArgumentError("basis species $(site.species) has no AtomicSpecies definition"))
    end

    props = if isnothing(basis_properties)
        [SiteProperties() for _ in crystal.basis]
    else
        length(basis_properties) == length(crystal.basis) ||
            throw(DimensionMismatch("basis_properties must contain one entry per basis site"))
        [p isa SiteProperties ? p : throw(ArgumentError("basis properties must be SiteProperties"))
         for p in basis_properties]
    end

    return Material{typeof(crystal)}(
        String(name), crystal, species_db, props, Dict{Symbol,Any}(metadata),
    )
end

atomic_species(material::Material, symbol::Symbol) = material.species[symbol]
basis_properties(material::Material, basis_index::Integer) = material.basis_properties[Int(basis_index)]
site_properties(material::Material, site::CrystalSite) = basis_properties(material, site.basis_index)
site_properties(material::Material, supercell::Supercell, i::Integer) =
    site_properties(material, supercell.sites[Int(i)])

site_mass(material::Material, site::CrystalSite) = atomic_species(material, site.species).mass
site_mass(material::Material, supercell::Supercell, i::Integer) = site_mass(material, supercell.sites[Int(i)])
site_spin(material::Material, supercell::Supercell, i::Integer) = site_properties(material, supercell, i).spin
site_g_tensor(material::Material, supercell::Supercell, i::Integer) = site_properties(material, supercell, i).g_tensor
site_form_factor(material::Material, supercell::Supercell, i::Integer) = site_properties(material, supercell, i).magnetic_form_factor
site_scattering_length(material::Material, supercell::Supercell, i::Integer) =
    atomic_species(material, supercell.sites[Int(i)].species).nuclear_scattering_length
