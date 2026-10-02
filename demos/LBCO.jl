#--------------------------------------------------------------------------------#
# Reference: https://www.sciencedirect.com/science/article/pii/0921453493901083
#
# Material: La₂₋ₓBaₓCuO4
#
#--------------------------------------------------------------------------------#

#--------------#
# Dependencies #
#--------------#
using Phundamental
using LinearAlgebra
using Random


#----------------#
# Material State #
#----------------#

# Katano et al., Physica C 214 (1993): La₂₋ₓBaₓCuO₄ at x = 1/8 and T = 15 K.
#
x = 0.125; temperature = 15.0; phase = :LTT

# Low-temperature tetragonal structure P4₂/ncm, space group 138, origin choice 2.
#
sg = "P4_2/ncm"; sg_number = 138; sg_choice = 2


#-----------------#
# Crystal Lattice #
#-----------------#

# Experimental tetragonal lattice parameters in Å.
#
a = 5.34837; c = 13.2352

lattice = BravaisLattice([a 0.0 0.0
                          0.0 a 0.0
                          0.0 0.0 c])


#--------------------#
# Chemical Occupancy #
#--------------------#

# La₂₋ₓBaₓCuO₄ contains two crystallographic A sites per formula unit, so the Ba probability on each A site is x/2.
#
p_Ba = x / 2; p_La = 1 - p_Ba
mixed_occ = MixedOccupancy(:La => p_La, :Ba => p_Ba)

# Retain the physical atomic species explicitly; no virtual La/Ba species is required.
species = Dict(symbol => lookup_species(symbol) for symbol in (:La, :Ba, :Cu, :O))


#-----------------#
# Site Properties #
#-----------------#

# Convert the crystallographic isotropic B factor to the displacement covariance U using B = 8π²U.
function isotropic_debye_waller(B::Real)::DebyeWallerTensor
    U = Float64(B) / (8π^2)
    return DebyeWallerTensor(Matrix{Float64}(I, 3, 3) .* U)
end

# Experimental isotropic displacement parameters for the five crystallographic orbits.
site_props = (LaBa=SiteProperties(debye_waller=isotropic_debye_waller(0.25), metadata=Dict{Symbol,Any}(:B_iso_A2 => 0.25, :orbit => :LaBa)),
              Cu=SiteProperties(debye_waller=isotropic_debye_waller(0.18), metadata=Dict{Symbol,Any}(:B_iso_A2 => 0.18, :orbit => :Cu)),
              O1=SiteProperties(debye_waller=isotropic_debye_waller(0.28), metadata=Dict{Symbol,Any}(:B_iso_A2 => 0.28, :orbit => :O1)),
              O1p=SiteProperties(debye_waller=isotropic_debye_waller(0.50), metadata=Dict{Symbol,Any}(:B_iso_A2 => 0.50, :orbit => :O1p)),
              O2=SiteProperties(debye_waller=isotropic_debye_waller(0.99), metadata=Dict{Symbol,Any}(:B_iso_A2 => 0.99, :orbit => :O2)))


#-------------------#
# Crystal Basis Set #
#-------------------#

# Asymmetric-unit representatives reported for the P4₂/ncm refinement.
#
irreps = (LaBa= [+0.0053, +0.0053, +0.36059],
          Cu  = [+0.0000, +0.0000, +0.00000],
          O1  = [+0.2500, +0.2500, +0.00740],
          O1p = [+0.7500, +0.2500, +0.00000],
          O2  = [-0.0163, -0.0163, +0.18210] )

# Chemical occupation is specified independently of crystallographic position.
#
occs = (LaBa=mixed_occ, Cu=:Cu, O1=:O, O1p=:O, O2=:O)

# Expand every asymmetric-unit representative through the selected space-group setting.
#
basis = expand_crystal_basis(sg_number, irreps, occs, site_props; choice=sg_choice)

mixed_crystal = MixedCrystalStructure(lattice, basis)

# The conventional P4₂/ncm cell contains 28 sites, including eight mixed A sites.
@assert length(mixed_crystal.basis) == 28
@assert count(site -> site isa MixedBasisSite, mixed_crystal.basis) == 8


#----------------------------#
# Parent Symmetry Validation #
#----------------------------#

# Treat the mixed A sublattice as one crystallographic species when validating the average parent structure.
parent_species_ids = [site isa MixedBasisSite ? 1 : site.species === :Cu ? 2 : 3 for site in mixed_crystal.basis]

symmetry_lattice = direct_matrix(lattice)
symmetry_fpos    = [site.fractional for site in mixed_crystal.basis]
site_labels      = [site.label for site in mixed_crystal.basis]

symmetry_crystal = CrystalStructure(symmetry_lattice, symmetry_fpos, parent_species_ids; labels=site_labels)

dataset = discover_symmetry(symmetry_crystal; options=CrystallographyOptions(symprec=1e-5))

@assert dataset.spacegroup_number == sg_number


#----------------------#
# Disordered Supercell #
#----------------------#

# Two conventional cells provide 16 A sites, allowing the x = 1/8 composition to be represented exactly by 15 La and 1 Ba.

disorder_repetitions = (2, 1, 1); disorder_seed = 20260924


supercell = disordered_supercell(mixed_crystal, disorder_repetitions; composition=:exact, rng=MersenneTwister(disorder_seed), periodic=true)


realized_species = [site.species for site in supercell.sites]

# Validate the explicit chemical realization and resulting Ba concentration.
@assert nsites(supercell) == 56
@assert count(==(:La), realized_species) == 15
@assert count(==(:Ba), realized_species) == 1
@assert count(==(:Cu), realized_species) == 8
@assert count(==(:O), realized_species) == 32

realized_x = count(==(:Ba), realized_species) / count(==(:Cu), realized_species)

@assert isapprox(realized_x, x; atol=1e-12)


#-----------------#
# Material System #
#-----------------#

# The disorder realization preserves the site properties generated from the parent crystallographic basis.
basis_props = SiteProperties[site.properties for site in supercell.crystal.basis]

# Record the experimental state and the explicit-disorder realization used by downstream models.
material_meta = Dict{Symbol,Any}(:formula => "La1.875Ba0.125CuO4",
                                 :doping => x,
                                 :temperature_K => temperature,
                                 :phase => phase,
                                 :space_group => sg,
                                 :space_group_number => sg_number,
                                 :space_group_choice => sg_choice,
                                 :length_unit => :angstrom,
                                 :structural_model => :average_Rietveld,
                                 :A_site_model => :explicit_disorder,
                                 :disorder_composition => :exact,
                                 :disorder_repetitions => disorder_repetitions,
                                 :disorder_seed => disorder_seed,
                                 :reference => "Katano et al., Physica C 214 (1993) 64-72")

material = Material("LBCO x=1/8 LTT T=15K", supercell; species=species, basis_properties=basis_props, metadata=material_meta)


println("LBCO Demo Finished!")
