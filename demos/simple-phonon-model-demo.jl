include(joinpath(@__DIR__,"Phundamental.jl"))
using .Phundamental


#---------------------------#
# Bravais & Crystal Lattice #
#---------------------------#
lattice = BravaisLattice(reshape([1.0], 1, 1))

sites = [BasisSite(:A,:A,[0.0]), BasisSite(:B,:B, [0.5])]

crystal = LatticeCrystalStructure(lattice,sites)

#----------------#
# Material Model #
#----------------#
species = Dict(:A => AtomicSpecies(:A; mass=1.0, nuclear_scattering_length=1.0),
               :B => AtomicSpecies(:B, mass=2.0, nuclear_scattering_length=1.5))

material = Material("Diatomic chain", crystal; species=species)

#-------------------#
# Hamiltonian Model #
#-------------------#
interactions = [HarmonicBondInteraction(1, 2, [0]; longitudinal=4.0),
                HarmonicBondInteraction(1, 2, [-1]; longitudinal=4.0)]

model = harmonic_model(material, interactions)


#-------------#
# Observables #
#-------------#
E0 = groundenergy(model; qmesh=(256,))
bands = phonon_dispersion(model, range(-0.5, 0.5; length=101))
Sqω = one_phonon_neutron_intensity(model, range(0.0, 2.0; length=161), range(0.0, 5.0; length=401); temperature=0.0, broadening=GaussianBroadening(0.03))
