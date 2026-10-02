using Phundamental
using LinearAlgebra

# SrCu₂(BO₃)₂ single-dimer neutron spectrum through the high-level ForwardProblem API.

const K_B_MEV_PER_K = 8.617333262e-2

#-------------------#
# Crystal Structure #
#-------------------#

a = 8.982 # Å
c = 6.643 # Å
lattice = BravaisLattice([a 0.0 0.0
                          0.0 a 0.0
                          0.0 0.0 c])

space_group_number = 121 # I-42m
cu_irrep = [0.11456, 0.11456, 0.2884]

#--------------------------#
# Cu²⁺ Material Properties #
#--------------------------#

g_perp = 2.057 
g_para = 2.269

g_cu = [g_perp 0.0 0.0
        0.0 g_perp 0.0
        0.0 0.0 g_para]


cu2_form_factor = GaussianFormFactor([0.0232, 0.4023, 0.5882], [34.969, 11.564, 3.843]; c=-0.0137, scale=4π)


cu_properties = SiteProperties(spin=0.5, g_tensor=g_cu, magnetic_form_factor=cu2_form_factor, metadata=Dict{Symbol,Any}(:ion => :Cu²⁺))


cu_basis = expand_crystal_basis(space_group_number, :Cu, cu_irrep; properties=cu_properties)



parent_crystal = LatticeCrystalStructure(lattice, cu_basis)
parent_cluster = Supercell(parent_crystal, (1, 1, 1); periodic=true)

material = Material("SrCu2(BO3)2",
                    parent_crystal;
                    species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)),
                    basis_properties=fill(cu_properties, length(cu_basis)),
                    metadata=Dict{Symbol,Any}(:formula => "SrCu₂(BO₃)₂",
                                              :space_group => "I-42m",
                                              :space_group_number => space_group_number,
                                              :reference => "Sparta et al., Eur. Phys. J. B 19, 507-516 (2001)"
                                             )
                   )

#----------------------#
# Physical Cu-Cu Dimer #
#----------------------#

parent_bonds = neighbor_bonds(parent_cluster; cutoff=3.1, max_shell=1)

dimer_bond = first(bonds_in_shell(parent_bonds, 1))

cluster = bond_cluster(parent_cluster, dimer_bond)

#-------------------#
# Hamiltonian Model #
#-------------------#

J_meV = 84.0 * K_B_MEV_PER_K
model = build_model(material, cluster, HeisenbergModel(J_meV; shell=1))

#------------------------#
# Measurement Definition #
#------------------------#

positions = model.parameters[:positions]

dimer_vector = positions[2] - positions[1]
dimer_length = norm(dimer_vector)

Q = (π / dimer_length^2) .* dimer_vector # Q·d = π


energy_axis = collect(range(J_meV - 5.0, J_meV + 5.0; length=5001))

broadening  = LorentzianBroadening(0.05)

convention  = SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=K_B_MEV_PER_K)

observable  = SpinTensorStructureFactorRequest(Q, energy_axis; temperature=0.0, broadening=broadening, convention=convention)

probe       = MagneticNeutronProbe(material, cluster; site=1)

resolution  = GaussianResolution(0.10)


problem = ForwardProblem(model; observable=observable, probe=probe, resolution=resolution)
calculation = forward(problem)

#---------------------#
# Measurable Spectrum #
#---------------------#

spectrum = calculation.result; energy_meV = spectrum.axis

intensity = vec(spectrum.intensity); peak_index = argmax(intensity)

println()
println(" -------------------------------- ")
println("| SrCu₂(BO₃)₂ Dimer Forward Run |")
println(" -------------------------------- ")
println("Material                  : ", material.metadata[:formula])
println("Cu-Cu distance [Å]        : ", round(dimer_length; digits=6))
println("Exchange J [meV]          : ", round(J_meV; digits=6))
println("|Q| [Å⁻¹]                 : ", round(norm(Q); digits=6))
println("Q·d                       : ", round(dot(Q, dimer_vector); digits=12))
println("Measured peak [meV]       : ", round(energy_meV[peak_index]; digits=6))
println("Peak intensity            : ", round(intensity[peak_index]; digits=6))
println("Intensity convention      : relative units (probe prefactor = 1)")
println()
println("Final spectrum arrays are available as `energy_meV` and `intensity`.")
