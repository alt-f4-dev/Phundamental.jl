using Phundamental
using Phundamental.Observables: eigenbasis_matrix, MomentumField, SpectrumConvention, static_structure_factor, 
                                momentum_operator, lehmann_lines, LorentzianBroadening, dynamic_structure_factor,
                                SpinTensorField, spin_tensor_structure_factor, MagneticNeutronProbe, scattering_intensity,
                                GaussianResolution, convolve_resolution, resolve_observable, validate_observable_equivalence

using Phundamental.Models: GaussianFormFactor, site_g_tensor, site_form_factor, form_factor

using Phundamental.Thermodynamics: thermal_curve, thermal_state, partition_function,
                                  log_partition_function, free_energy, internal_energy,
                                  entropy, heat_capacity, expectation

using Phundamental.Transformations: JordanWigner, AlgebraIsomorphism, transform, transform_observable

using LinearAlgebra

#--------------------#
# Material Reference #
#--------------------#

# SrCu₂(BO₃)₂ low-temperature tetragonal structure
formula = "SrCu₂(BO₃)₂"
temperature = 100.0

sg = "I-42m"
sg_number = 121

a = 8.982; c = 6.643 #Å

#Cu ~ Wyckoff 8i
cu_irrep = [0.11456, 0.11456, 0.2884]

#-----------------#
# Crystal Lattice #
#-----------------#

# Conventional body-centered tetragonal cell in Cartesian
lattice = BravaisLattice([ a 0.0 0.0
                           0.0 a 0.0
                           0.0 0.0 c ])

#--------------------------#
# Magnetic Site Properties #
#--------------------------#

#Cu²⁺ ~ effective S=1/2 moment
cu = lookup_species(:Cu)
species = Dict{Symbol, Any}(:Cu => cu)

#g⟂ ∈ ab-plane & g∥ ∈ c-axis
g_perp = 2.057; g_para = 2.269

g_cu = [g_perp 0.0 0.0
        0.0 g_perp 0.0
        0.0 0.0 g_para]

# Ionic Cu²⁺ magnetic form factor <j₀(Q)> from Brown, International Tables for Crystallography.
cu2_form_factor = GaussianFormFactor([0.0232, 0.4023, 0.5882], [34.969, 11.564, 3.843]; c=-0.0137, scale=4π)

cu_props = SiteProperties(spin=0.5, g_tensor=g_cu, magnetic_form_factor=cu2_form_factor, metadata=Dict{Symbol,Any}(:ion => :Cu²⁺, :magnetic_orbital => :effective_spin_half))

#A + B + C + D ≈ 1 
@assert isapprox(form_factor(cu2_form_factor, 0.0), 1.0; atol=1e-4)
@assert form_factor(cu2_form_factor, 1.0) < 1.0


#----------------------#
# Parent Cu Sublattice #
#----------------------#
operations = space_group_operations(sg_number)

cu_orbit = expand_symmetry_orbit(cu_irrep, operations)
@assert length(cu_orbit) == 8

cu_basis = BasisSite[BasisSite(Symbol("Cu_",i), :Cu, position; properties=cu_props) for (i, position) in enumerate(cu_orbit)]

parent_crystal = LatticeCrystalStructure(lattice, cu_basis)
parent_cluster = Supercell(parent_crystal, (1,1,1); periodic=true)

#--------------------------#
# Physical Intradimer Bond #
#--------------------------#

# Nearest-neighbor Cu-Cu distance shell ~ 2.91Å
cu_bonds = neighbor_bonds(parent_cluster; cutoff=3.1, max_shell=1)

dimer_bonds = bonds_in_shell(cu_bonds, 1)
@assert !isempty(dimer_bonds)

dimer_bond = first(dimer_bonds)
@assert isapprox(dimer_bond.distance, 2.9104; atol=1e-3)

#----------------------#
# Two-site Dimer Model #
#----------------------#

# Convert the periodic nearest-neighbor image into an explicit two-site geometry
site1 = parent_cluster.sites[dimer_bond.i]

r1_fractional = copy(site1.fractional)
r2_cartesian  = site1.cartesian + dimer_bond.displacement
r2_fractional = direct_matrix(lattice) \ r2_cartesian

dimer_basis = BasisSite[BasisSite(:Cu1, :Cu, r1_fractional; properties=cu_props),
                        BasisSite(:Cu2, :Cu, r2_fractional; properties=cu_props)]

dimer_crystal = LatticeCrystalStructure(lattice, dimer_basis)

# Open boundaries ensure numerical problem contains exactly one Cu-Cu dimer
dimer_cluster = Supercell(dimer_crystal, (1,1,1); periodic=false)

@assert nsites(dimer_cluster) == 2
@assert norm(site_position(dimer_cluster, 2) - site_position(dimer_cluster, 1)) ≈ dimer_bond.distance


#-----------------#
# Material System #
#-----------------#

# This represents the magnetic two-Cu reduction of the real crystal
material_meta = Dict{Symbol,Any}(:formula => formula, 
                                 :temperature => temperature,
                                 :space_group => sg,
                                 :space_group_number => sg_number,
                                 :model_reduction => :magnetic_sublattice,
                                 :J_intradimer => 84.0,
                                 :J_interdimer => 54.0,
                                 :J_interlayer => 8.0,
                                 :reference => "Sparta et al., Eur. Phys. J. B 19, 507-516 (2001)")

material = Material("SrCu2(BO3)2 ~ Single Cu dimer", dimer_crystal; 
                    species=species, basis_properties = [cu_props, cu_props], metadata=material_meta)


#----------------#
# Stage 1 Checks #
#----------------#
@assert material.crystal === dimer_crystal
@assert length(material.crystal.basis) == 2
@assert all(site -> site.species === :Cu, material.crystal.basis)
@assert all(props -> props.spin == 0.5, material.basis_properties)

dimer_fractional = [site.fractional for site in dimer_crystal.basis]
dimer_cartesian = [site.cartesian for site in dimer_cluster.sites]



#-----------------#
# Stage 1 Summary #
#-----------------#

println()
println(" ------------------------ ")
println("| Cuprate Dimer Material |")
println(" ------------------------ ")
println("Reference      : ", material.metadata[:reference])
println("Material       : ", material.metadata[:formula])
println("Space Group    : ", material.metadata[:space_group], " (", material.metadata[:space_group_number],")")
println("Spin Carrier   : ", cu_props.metadata[:ion])
println("Cu-Cu Distance : ", dimer_bond.distance)



#---------------------------#
# Hamiltonian Specification #
#---------------------------#

# Convert experimental exchange from Kelvin to meV
Kelvin2meV = 8.617333262e-2
J = material.metadata[:J_intradimer]*Kelvin2meV

@assert J > 0 

dimer_distance = norm(site_position(dimer_cluster, 2) - site_position(dimer_cluster, 1))
neighbor_cutoff = dimer_distance + 1e-6

# Reduced magnetic model containing one isotropic AFM S=1/2 Heisenberg bond with no applied field
spec = HeisenbergModel(J; neighbor_cutoff=neighbor_cutoff, max_shell=1, hbar=1.0)


#----------------#
# Symbolic Model #
#----------------#
model = build_model(material, dimer_cluster, spec)
@assert model isa ManyBodyModel


#----------------------#
# Model Representation #
#----------------------#

# The physical Hilbert space contains two S=1/2 spins, so it has exact dimensions 2² = 4.
representation = model.representation

@assert representation.name === :spin
@assert representation.algebra isa SpinAlgebra
@assert representation.state_space isa SpinHilbertSpace
@assert representation.basis isa SpinProductBasis
@assert representationdimension(representation) == 4
@assert representation.ordering == [1,2]


#----------------------#
# Symbolic Hamiltonian #
#----------------------#

# The generated model should contain three Cartesian components of one isotropic exchange bond.
H = model.hamiltonian

@assert H isa OperatorSum
@assert length(H.terms) == 3
@assert operator_degree(H) == 2
@assert support_size(H) == 2


#----------------------#
# Analytic Hamiltonian #
#----------------------#

# Canonicalization requires the algebra for rewrite rules
spin_algebra = model.representation.algebra
hbar = model.parameters[:hbar]

# This is the expected symbolic dimer Hamiltonian
H′ = J*(Sx(1)*Sx(2) + Sy(1)*Sy(2) + Sz(1)*Sz(2))

H_canonical = Phundamental.Algebra.canonicalize(spin_algebra, H; hbar=hbar)
H_expected = Phundamental.Algebra.canonicalize(spin_algebra, H′; hbar=hbar)


#-------------------#
# Exchange Geometry #
#-------------------#

# build_model stores the geometric bonds used to generate the Hamiltonian
model_bonds = model.parameters[:bonds]

@assert length(model_bonds) == 1

exchange_bond = only(model_bonds)

@assert exchange_bond.i == 1
@assert exchange_bond.j == 2
@assert isapprox(exchange_bond.distance, dimer_distance; atol=1e-12)


#------------------------#
# Registered Observables #
#------------------------#

# Spin-model construction automatically registers three local spin components on every magnetic site
observables = sort!(collect(keys(model.observables)); by=string)
@assert length(observables) == 6


#-------------------#
# Model Information #
#-------------------#
#println()
#println()
# Retain the connection between the symbolic Hamiltonian and the physical material from which it was constructed.
#@show model.parameters[:material].name
#@show model.parameters[:hamiltonian_spec]
#@show model.parameters[:positions]
#@show model.parameters[:bonds]
#@show model.parameters[:hbar]
#@show model.parameters[:coupling_convention]
#@show model.provenance

#-----------------#
# Stage 2 Summary #
#-----------------#

println()
println(" ----------------------------- ")
println("| Symbolic Hamiltonian Checks |")
println(" ----------------------------- ")
println("State Space        : ", representation.state_space)
println("Algebra            : ", representation.algebra)
println("Basis              : ", representation.basis)
println("Expected Symbolics : ", H_expected)
println("Obtained Symbolics : ", H_canonical)
println()
println("Observables")
println("-----------")
for name in observables
    println(name, " → ", model.observables[name])
end


#-------------------#
# Algebra Utilities #
#-------------------#
is_symbolic_zero(expr) = expr isa ScaledOperator && iszero(expr.coefficient)
canonical(expr) = Phundamental.Algebra.canonicalize(spin_algebra, expr; hbar=hbar)
commute(A,B) = Phundamental.Algebra.commutator(spin_algebra, A, B; hbar=hbar)

#--------------------#
# Local Spin Algebra #
#--------------------#

# Verify the Cartesian SU(2) relations
comm_xy = commute(Sx(1), Sy(1))
comm_yz = commute(Sy(1), Sz(1))
comm_zx = commute(Sz(1), Sx(1))

@assert is_symbolic_zero(canonical(comm_xy - (im*hbar)*Sz(1)))
@assert is_symbolic_zero(canonical(comm_yz - (im*hbar)*Sx(1)))
@assert is_symbolic_zero(canonical(comm_zx - (im*hbar)*Sy(1)))

SU2_check = is_symbolic_zero(canonical(comm_xy - (im*hbar)*Sz(1))) && is_symbolic_zero(canonical(comm_yz - (im*hbar)*Sx(1))) && is_symbolic_zero(canonical(comm_zx - (im*hbar)*Sy(1)))

# Spin operators on distinct sites must commute
comm_cross = commute(Sx(1),Sy(2))
@assert is_symbolic_zero(comm_cross)
exchange_commute_check = is_symbolic_zero(comm_cross)

#---------------------#
# Ladder Spin Algebra #
#---------------------#

# Total spin is the sum of the two local Cu moments
Sx_tot = Sx(1) + Sx(2)
Sy_tot = Sy(1) + Sy(2)
Sz_tot = Sz(1) + Sz(2)

S²_tot = Sx_tot*Sx_tot + Sy_tot*Sy_tot + Sz_tot*Sz_tot

S₁² = Sx(1)*Sx(1) + Sy(1)*Sy(1) + Sz(1)*Sz(1)
S₂² = Sx(2)*Sx(2) + Sy(2)*Sy(2) + Sz(2)*Sz(2)

S_squared = canonical(S²_tot)
S1_squared = canonical(S₁²)
S2_squared = canonical(S₂²)

#----------------------#
# Hamiltonian Symmetry #
#----------------------#

# An isotropic Heisenberg dimer must commute with every component of total spin.
comm_H_Sx = commute(H, Sx_tot)
comm_H_Sy = commute(H, Sy_tot)
comm_H_Sz = commute(H, Sz_tot)

@assert is_symbolic_zero(comm_H_Sx)
@assert is_symbolic_zero(comm_H_Sy)
@assert is_symbolic_zero(comm_H_Sz)

# Consequently, total spin squared is also conserved.
comm_H_S2 = commute(H, S_squared)

@assert is_symbolic_zero(comm_H_S2)


println()
println(" ----------------------- ")
println("| Commutation Relations |")
println(" ----------------------- ")
println("[Sᵢᵅ, Sᵢᵝ] = iħSᵢᵞ  : ", SU2_check)
println("[Sᵢᵅ, Sⱼᵝ] =  0     : ", exchange_commute_check)
println("[ H , S² ] =  0     : ", is_symbolic_zero(comm_H_S2))
println()
@info "Not all commutation relations are shown here, only a representative sample." 

#----------------------#
# Total-Spin Identity  #
#----------------------#

# The model convention is H = (J / ħ²) S₁·S₂.
H_total_spin = (J / (2 * hbar^2)) * (S_squared - S1_squared - S2_squared)

identity_residual = canonical(H - H_total_spin)
@assert is_symbolic_zero(identity_residual)


#-----------------------#
# Analytic Spin Sectors #
#-----------------------#

# Each Cu site carries S = 1/2, so angular-momentum addition gives S_tot = 0 or 1.
S1 = model.representation.state_space.spins[1]
S2 = model.representation.state_space.spins[2]

@assert S1 == 0.5
@assert S2 == 0.5

S_singlet = 0.0
S_triplet = 1.0

# For H = (J / ħ²) S₁·S₂, the ħ² factors cancel in the energy eigenvalues.
exchange_energy(Stot) = (J / 2) * (Stot * (Stot + 1) - S1 * (S1 + 1) - S2 * (S2 + 1))

E_singlet = exchange_energy(S_singlet)
E_triplet = exchange_energy(S_triplet)
spin_gap = E_triplet - E_singlet

@assert isapprox(E_singlet, -3J / 4; atol=1e-12)
@assert isapprox(E_triplet, J / 4; atol=1e-12)
@assert isapprox(spin_gap, J; atol=1e-12)

#-----------------#
# Stage 3 Summary #
#-----------------#

println()
println(" -------------------------")
println("| Symbolic dimer analysis |")
println(" -------------------------")
println("Hilbert-space dimension : ", representationdimension(model.representation))
println("Local spins             : ", (S1, S2))
println("Ground-state sector     : S_tot = 0")
println("Excited sector          : S_tot = 1")
println("Singlet energy [meV]    : ", E_singlet)
println("Triplet energy [meV]    : ", E_triplet)
println("Spin gap [meV]          : ", spin_gap)
println("SU(2) conserved         : true")
println()
@info "Congratulations! All information derived so far has been directly from symbolics without a single Hamiltonian matrix involved!"



#---------------------#
# Computational Basis #
#---------------------#
basis = computational_basis(model)
@assert basis_dimension(basis) == 4
@assert all(state -> state isa SpinBasisState, basis.states)

#S=1/2 ⟹ level 0 is m=-1/2 and level 1 is m=+1/2
spin_arrow(level::Integer) = level == 0 ? "↓" : "↑"
basis_labels = ["|"*join(spin_arrow.(state.levels))*"⟩" for state in basis.states]


#--------------------#
# Matrix Realization #
#--------------------#

# Lower noncommutative symbolic Hamiltonian onto the finite spin-product basis
realized_H = realize(H, basis; hbar=hbar, sparse=false, check_physical_closure=true)

H_matrix = realized_H.matrix

@assert size(H_matrix) == (4, 4)
@assert isapprox(H_matrix, adjoint(H_matrix); atol=1e-12)
@assert realized_H.diagnostics[:physical_leakage] ≤ 1e-12
@assert realized_H.diagnostics[:truncation_leakage] ≤ 1e-12

#-----------------------#
# Analytic Matrix Check #
#-----------------------#

# Locate the four product states in the actual computational-basis ordering.
i_dd = basis.index[SpinBasisState((0, 0))]
i_du = basis.index[SpinBasisState((0, 1))]
i_ud = basis.index[SpinBasisState((1, 0))]
i_uu = basis.index[SpinBasisState((1, 1))]

# Analytic matrix for H = J S₁·S₂ with ħ=1.
H_matrix_expected = zeros(ComplexF64, 4, 4)

H_matrix_expected[i_dd, i_dd] = J / 4
H_matrix_expected[i_uu, i_uu] = J / 4

H_matrix_expected[i_du, i_du] = -J / 4
H_matrix_expected[i_ud, i_ud] = -J / 4

H_matrix_expected[i_du, i_ud] = J / 2
H_matrix_expected[i_ud, i_du] = J / 2

matrix_agreement = isapprox(H_matrix, H_matrix_expected; rtol=1e-12, atol=1e-12)
@assert matrix_agreement


#-----------------------#
# Exact Diagonalization #
#-----------------------#

# Solve the complete four-dimensional problem while retaining all eigenvectors.
solver = ExactDiagonalization(hermitian=:auto, check_physical_closure=true, hbar=hbar)

solution = solve(solver, model)

energies = Float64.(real.(solution.energies))
expected_energies = [E_singlet, E_triplet, E_triplet, E_triplet]

@assert nlevels(solution) == 4
@assert solution.metadata[:hermitian]
@assert !isbiorthogonal(solution)

spectrum_agreement = isapprox(energies, expected_energies; rtol=1e-12, atol=1e-12)
@assert spectrum_agreement

numerical_gap = energies[2] - energies[1]
@assert isapprox(numerical_gap, spin_gap; rtol=1e-12, atol=1e-12)


#--------------------#
# Singlet Validation #
#--------------------#

# Construct |s⟩ = (|↑↓⟩ - |↓↑⟩)/√2 in the actual computational basis.
ψ_singlet = zeros(ComplexF64, basis_dimension(basis))

ψ_singlet[i_ud] = 1 / sqrt(2)
ψ_singlet[i_du] = -1 / sqrt(2)

ψ_ground = collect(groundstate(solution))

singlet_fidelity = abs2(dot(ψ_singlet, ψ_ground))
@assert isapprox(singlet_fidelity, 1.0; atol=1e-12)


#-----------------------#
# Spin-Sector Validation #
#-----------------------#

# Realize S_tot² independently and evaluate it in every numerical eigenstate.
S2_matrix = matrix(S_squared, basis; hbar=hbar, sparse=false)

S2_expectations = [real(dot(eigenstate(solution, n), S2_matrix * eigenstate(solution, n))) / hbar^2 for n in 1:nlevels(solution)]

expected_S2 = [0.0, 2.0, 2.0, 2.0]

@assert isapprox(S2_expectations, expected_S2; rtol=1e-12, atol=1e-12)

triplet_degeneracy = count(energy -> isapprox(energy, E_triplet; rtol=1e-12, atol=1e-12), energies)
@assert triplet_degeneracy == 3


#-----------------#
# Stage 4 Summary #
#-----------------#

println()
println(" --------------------------------- ")
println("| Numerical Dimer Diagonalization |")
println(" --------------------------------- ")
println("Basis order              : ", join(basis_labels, ", "))
println("Matrix dimension         : ", size(H_matrix))
println("Matrix nonzero elements  : ", realized_H.diagnostics[:nnz])
println("Hermiticity error        : ", solution.metadata[:hermiticity_error])
println("Physical leakage         : ", realized_H.diagnostics[:physical_leakage])
println("Truncation leakage       : ", realized_H.diagnostics[:truncation_leakage])
println()
println("Spectrum")
println("--------")
println("Energies [meV]           : ", round.(energies; digits=6))
println("Ground-state energy [meV]: ", round(groundenergy(solution); digits=6))
println("Spin gap [meV]           : ", round(numerical_gap; digits=6))
println("Triplet degeneracy       : ", triplet_degeneracy)
println("Singlet fidelity         : ", round(singlet_fidelity; digits=12))
println("⟨S²⟩ / ħ²               : ", round.(S2_expectations; digits=12))
println()
println("Validation")
println("----------")
println("Analytic matrix agreement: ", matrix_agreement)
println("Analytic spectrum match  : ", spectrum_agreement)
println()
println("Hamiltonian matrix [meV]")
println("------------------------")
show(stdout, "text/plain", round.(real.(H_matrix); digits=6))
println()


#--------------------------#
# Ground-State Observables # WARNING: `eigenbasis_matrix` not defined in `Main` !!!
#--------------------------#

# Evaluate an operator directly in the ED eigenbasis and extract the ground-state expectation value.
ground_expectation(op) = eigenbasis_matrix(model, solution, op; hbar=hbar)[1, 1]

# The singlet has zero local magnetization on both Cu sites.
Sz1_ground = real(ground_expectation(:spin_z_1))
Sz2_ground = real(ground_expectation(:spin_z_2))

@assert isapprox(Sz1_ground, 0.0; atol=1e-12)
@assert isapprox(Sz2_ground, 0.0; atol=1e-12)

# The isotropic pair correlation is an intrinsic observable of the dimer ground state.
S1_dot_S2 = Sx(1)*Sx(2) + Sy(1)*Sy(2) + Sz(1)*Sz(2)

pair_correlation = real(ground_expectation(S1_dot_S2))
pair_correlation_expected = -3hbar^2 / 4

@assert isapprox(pair_correlation, pair_correlation_expected; atol=1e-12)


#-------------------------#
# Momentum-Space Geometry #
#-------------------------#

# Use the actual crystallographic Cu positions retained by the ManyBodyModel.
positions = model.parameters[:positions]

dimer_vector = positions[2] - positions[1]
dimer_length = norm(dimer_vector)
dimer_direction = dimer_vector / dimer_length

# q = 0 probes the uniform channel, while q·d = π probes the staggered dimer channel.
q_uniform = zeros(length(dimer_vector))
q_staggered = (π / dimer_length) .* dimer_direction

@assert isapprox(dot(q_staggered, dimer_vector), π; atol=1e-12)

qgrid = [q_uniform, q_staggered]



#------------------------#
# Momentum-Space Spin Sz #
#------------------------#

# Construct Sᶻ(q) = N^(-1/2) Σⱼ exp(-iq·rⱼ) Sⱼᶻ from registered local observables.
Sz_field = MomentumField([:spin_z_1, :spin_z_2], positions; normalization=:sqrtN)

spectrum_convention = SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=hbar, kB=Kelvin2meV)


#-------------------------#
# Static Structure Factor #
#-------------------------#

# Calculate the equal-time Sᶻᶻ(q) directly from the ED solution.
static_Sz = static_structure_factor(model, solution, Sz_field, qgrid; temperature=0.0, convention=spectrum_convention)

S_uniform = real(static_Sz.intensity[1])
S_staggered = real(static_Sz.intensity[2])

@assert isapprox(S_uniform, 0.0; atol=1e-12)
@assert isapprox(S_staggered, hbar^2 / 2; atol=1e-12)


#--------------------------#
# Exact Dynamic Spin Lines #
#--------------------------#

# Construct the uniform and staggered momentum operators explicitly.
Sz_q0 = momentum_operator(model, Sz_field, q_uniform)
Sz_qπ = momentum_operator(model, Sz_field, q_staggered)

Sz_minus_q0 = momentum_operator(model, Sz_field, -q_uniform)
Sz_minus_qπ = momentum_operator(model, Sz_field, -q_staggered)

# Obtain the exact delta-function transitions before line broadening.
lines_uniform = lehmann_lines(model, solution, Sz_q0, Sz_minus_q0; temperature=0.0, convention=spectrum_convention, weight_tol=1e-12)


lines_staggered = lehmann_lines(model, solution, Sz_qπ, Sz_minus_qπ; temperature=0.0, convention=spectrum_convention, weight_tol=1e-12)


uniform_weight = isempty(lines_uniform.weights) ? 0.0 : sum(abs.(lines_uniform.weights))


gap_weight = sum(real(lines_staggered.weights[i]) for i in eachindex(lines_staggered.centers) if isapprox(lines_staggered.centers[i], spin_gap; atol=1e-10))


total_staggered_weight = isempty(lines_staggered.weights) ? 0.0 : real(sum(lines_staggered.weights))

@assert isapprox(uniform_weight, 0.0; atol=1e-12)
@assert isapprox(gap_weight, hbar^2 / 2; atol=1e-12)
@assert isapprox(total_staggered_weight, S_staggered; atol=1e-12)


#--------------------------#
# Dynamic Structure Factor #
#--------------------------#

# Represent the exact excitation using a narrow numerical Lorentzian line shape.
line_width = 0.05 # meV
broadening = LorentzianBroadening(line_width)
energy_axis = collect(range(0.0, 1.5spin_gap; length=2001))


dynamic_Sz = dynamic_structure_factor(model, solution, Sz_field, qgrid, energy_axis; temperature=0.0, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)


uniform_spectrum = real.(dynamic_Sz.intensity[1, :])
staggered_spectrum = real.(dynamic_Sz.intensity[2, :])


peak_index = argmax(staggered_spectrum)
peak_energy = energy_axis[peak_index]
energy_step = energy_axis[2] - energy_axis[1]

@assert maximum(abs.(uniform_spectrum)) ≤ 1e-12
@assert isapprox(peak_energy, spin_gap; atol=energy_step)


#-----------------#
# Stage 5 Summary #
#-----------------#

println()
println(" ----------------------------- ")
println("| Intrinsic Dimer Observables |")
println(" ----------------------------- ")
println("⟨S₁ᶻ⟩                  : ", round(Sz1_ground; digits=12))
println("⟨S₂ᶻ⟩                  : ", round(Sz2_ground; digits=12))
println("⟨S₁·S₂⟩                : ", round(pair_correlation; digits=12))
println()
println("Momentum Transfer")
println("-----------------")
println("|q_uniform| [Å⁻¹]       : ", round(norm(q_uniform); digits=6))
println("|q_staggered| [Å⁻¹]     : ", round(norm(q_staggered); digits=6))
println("q_staggered·d           : ", round(dot(q_staggered, dimer_vector); digits=12))
println()
println("Static Sᶻᶻ(q)")
println("-------------")
println("Sᶻᶻ(q=0)                : ", round(S_uniform; digits=12))
println("Sᶻᶻ(q·d=π)              : ", round(S_staggered; digits=12))
println()
println("Dynamic Sᶻᶻ(q,E)")
println("----------------")
println("Exact transition [meV]  : ", round(spin_gap; digits=6))
println("Lehmann weight          : ", round(gap_weight; digits=12))
println("Broadened peak [meV]    : ", round(peak_energy; digits=6))
println("Lorentzian width [meV]  : ", line_width)
println()
println("Validation")
println("----------")
println("Uniform channel extinct : ", isapprox(uniform_weight, 0.0; atol=1e-12))
println("Static/dynamic sum rule : ", isapprox(total_staggered_weight, S_staggered; atol=1e-12))


#-------------------#
# Full Spin Tensor  #
#-------------------#

# Construct the three Cartesian momentum fields from the local spin observables already registered by build_model.
spin_field = SpinTensorField(xkeys=[:spin_x_1, :spin_x_2], ykeys=[:spin_y_1, :spin_y_2], zkeys=[:spin_z_1, :spin_z_2], positions=positions, normalization=:sqrtN)

# Use the staggered dimer wavevector where the singlet-triplet response is maximal.
Q = q_staggered

# Center the spectral grid exactly on the singlet-triplet gap so that the line maximum lies on a grid point.
tensor_window = 5.0 # meV
tensor_axis = collect(range(spin_gap - tensor_window, spin_gap + tensor_window; length=5001))

spin_tensor = spin_tensor_structure_factor(model, solution, spin_field, [Q], tensor_axis; temperature=0.0, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)

@assert size(spin_tensor.intensity) == (1, length(tensor_axis), 3, 3)


#--------------------------#
# Tensor Symmetry Check    #
#--------------------------#

tensor_peak_index = argmin(abs.(tensor_axis .- spin_gap))
tensor_peak_energy = tensor_axis[tensor_peak_index]

S_peak = Matrix(@view spin_tensor.intensity[1, tensor_peak_index, :, :])

tensor_diagonal = real.([S_peak[i, i] for i in 1:3])
tensor_offdiagonal = [S_peak[i, j] for i in 1:3 for j in 1:3 if i != j]

tensor_isotropic = all(value -> isapprox(value, tensor_diagonal[1]; rtol=1e-10, atol=1e-12), tensor_diagonal)

tensor_offdiagonal_zero = maximum(abs.(tensor_offdiagonal)) ≤ 1e-12

@assert tensor_isotropic
@assert tensor_offdiagonal_zero

# At q·d=π each Cartesian component carries total Lehmann weight ħ²/2.
# A Lorentzian of half-width Γ has peak height weight/(πΓ).
expected_tensor_peak = (hbar^2 / 2) / (π * line_width)

@assert all(value -> isapprox(value, expected_tensor_peak; rtol=1e-10, atol=1e-10), tensor_diagonal)


#----------------------------#
# Magnetic Probe Parameters  #
#----------------------------#

# Both Cu sites are crystallographically equivalent in the reduced dimer model.
g1 = site_g_tensor(material, dimer_cluster, 1)
g2 = site_g_tensor(material, dimer_cluster, 2)

ff1 = site_form_factor(material, dimer_cluster, 1)
ff2 = site_form_factor(material, dimer_cluster, 2)

@assert g1 == g2
@assert typeof(ff1) == typeof(ff2)

g_tensor_neutron = g1
magnetic_form_factor = ff1

F_Q = form_factor(magnetic_form_factor, Q)

# The Stage 1 material currently uses the default ConstantFormFactor(), so F(Q)=1.
# The API nevertheless applies it explicitly here; a tabulated Cu²⁺ form factor can replace it later without changing this stage.
probe = MagneticNeutronProbe(form_factor=Qvec -> form_factor(magnetic_form_factor, Qvec), g_tensor=g_tensor_neutron, prefactor=1.0)


#---------------------------#
# Magnetic Neutron Response #
#---------------------------#

neutron_response = scattering_intensity(probe, spin_tensor)

@assert neutron_response.metadata[:probe] === :magnetic_neutron
@assert neutron_response.metadata[:polarization_projection] === :transverse
@assert neutron_response.metadata[:g_tensor_applied]
@assert neutron_response.metadata[:form_factor_applied]
@assert !neutron_response.metadata[:resolution_applied]

neutron_peak = neutron_response.intensity[1, tensor_peak_index]

#----------------------------#
# Probe Contraction Check    #
#----------------------------#

Qhat = Q / norm(Q)
polarization = Matrix{Float64}(I, 3, 3) - Qhat * transpose(Qhat)

g_complex = ComplexF64.(g_tensor_neutron)
S_effective = g_complex * S_peak * adjoint(g_complex)

neutron_peak_expected = abs2(F_Q) * real(sum(polarization .* S_effective))

probe_contraction_match = isapprox(neutron_peak, neutron_peak_expected; rtol=1e-10, atol=1e-10)

@assert probe_contraction_match


#-----------------------#
# Instrument Resolution #
#-----------------------#

resolution_sigma = 0.10 # meV

resolution = GaussianResolution(resolution_sigma; nsigma=5.0)

resolved_neutron = convolve_resolution(resolution, neutron_response)

@assert resolved_neutron.metadata[:resolution_applied]
@assert resolved_neutron.metadata[:resolution_role] === :instrument_response

resolved_peak_index = argmax(resolved_neutron.intensity[1, :])
resolved_peak_energy = tensor_axis[resolved_peak_index]
resolved_peak = resolved_neutron.intensity[1, resolved_peak_index]

@assert isapprox(resolved_peak_energy, spin_gap; atol=tensor_axis[2] - tensor_axis[1])


#-----------------------#
# Line-Shape Diagnostics #
#-----------------------#

function discrete_fwhm(axis, intensity)
    ymax = maximum(intensity)
    indices = findall(value -> value >= ymax / 2, intensity)
    isempty(indices) && return NaN
    return axis[last(indices)] - axis[first(indices)]
end

function trapezoid_integral(axis, intensity)
    return sum(0.5 * (intensity[i] + intensity[i + 1]) * (axis[i + 1] - axis[i]) for i in 1:(length(axis) - 1))
end

intrinsic_neutron = neutron_response.intensity[1, :]
instrument_neutron = resolved_neutron.intensity[1, :]

intrinsic_fwhm = discrete_fwhm(tensor_axis, intrinsic_neutron)
resolved_fwhm = discrete_fwhm(tensor_axis, instrument_neutron)

intrinsic_weight = trapezoid_integral(tensor_axis, intrinsic_neutron)
resolved_weight = trapezoid_integral(tensor_axis, instrument_neutron)

#--------------------------------#
# Integrated Neutron Sum Rule    #
#--------------------------------#

intrinsic_spin_weight = hbar^2 / 2

probe_weight_expected = intrinsic_spin_weight * abs2(F_Q) * real(sum(polarization .* (g_complex * adjoint(g_complex))))

# The finite spectral window omits only the far Lorentzian tails.
@assert isapprox(intrinsic_weight, probe_weight_expected; rtol=1e-2, atol=1e-12)


#-----------------#
# Stage 6 Summary #
#-----------------#

println()
println(" ----------------------------------- ")
println("| Magnetic Neutron Dimer Scattering |")
println(" ----------------------------------- ")
println(" Q  [Å⁻¹]                : ", round.(Q; digits=6))
println("|Q| [Å⁻¹]                : ", round(norm(Q); digits=6))
println("Q·d                      : ", round(dot(Q, dimer_vector); digits=12))
println()
println("Intrinsic Spin Tensor")
println("---------------------")
println("Peak energy [meV]        : ", round(tensor_peak_energy; digits=6))
println("Diagonal Sᵅᵅ peak        : ", round.(tensor_diagonal; digits=6))
println("Tensor isotropic         : ", tensor_isotropic)
println("Off-diagonal tensor zero : ", tensor_offdiagonal_zero)
println()
println("Magnetic Probe")
println("--------------")
println("g tensor diagonal        : ", diag(g_tensor_neutron))
println("Form-factor model        : ", nameof(typeof(magnetic_form_factor)))
println("F(Q)                     : ", round(F_Q; digits=8))
println("Probe peak intensity     : ", round(neutron_peak; digits=6))
println("Probe contraction match  : ", probe_contraction_match)
println("Integrated intensity     : ", round(intrinsic_weight; digits=6))
println("Expected weight          : ", round(probe_weight_expected; digits=6))
println()
println("Instrument Resolution")
println("---------------------")
println("Gaussian σ [meV]         : ", resolution_sigma)
println("Intrinsic FWHM [meV]     : ", round(intrinsic_fwhm; digits=6))
println("Resolved FWHM [meV]      : ", round(resolved_fwhm; digits=6))
println("Resolved peak [meV]      : ", round(resolved_peak_energy; digits=6))
println("Resolved peak intensity  : ", round(resolved_peak; digits=6))
println("Resolved spectral weight : ", round(resolved_weight; digits=6))


#-------------------------#
# Thermal Energy Scale    #
#-------------------------#

# The natural thermal scale of the isolated dimer is T_J = J/kB.
T_exchange = spin_gap / Kelvin2meV

@assert isapprox(T_exchange, 84.0; atol=1e-12)

# Sample from T=0 through four times the exchange scale.
thermal_temperatures = collect(range(0.0, 4 * T_exchange; length=4001))


#-------------------------#
# Exact Thermodynamics    #
#-------------------------#

thermodynamics = thermal_curve(solution, thermal_temperatures; kB=Kelvin2meV)

@assert thermodynamics.metadata[:spectrum_reused]
@assert thermodynamics.metadata[:dimension] == 4


#-------------------------#
# Analytic Thermodynamics #
#-------------------------#

function analytic_dimer_thermodynamics(T, Es, Et, gap, kB)
    if iszero(T)
        return (logZ=NaN, free_energy=Es, internal_energy=Es, entropy=0.0, heat_capacity=0.0)
    end

    β = inv(kB * T)
    x = exp(-β * gap)
    Zshift = 1 + 3x

    logZ = -β * Es + log(Zshift)
    U = (Es + 3x * Et) / Zshift
    F = -kB * T * logZ
    S = (U - F) / T
    Cv = 3 * kB * (β * gap)^2 * x / Zshift^2

    return (logZ=logZ, free_energy=F, internal_energy=U, entropy=S, heat_capacity=Cv)
end

analytic_thermodynamics = [analytic_dimer_thermodynamics(T, E_singlet, E_triplet, spin_gap, Kelvin2meV) for T in thermal_temperatures]

F_expected = [point.free_energy for point in analytic_thermodynamics]
U_expected = [point.internal_energy for point in analytic_thermodynamics]
S_expected = [point.entropy for point in analytic_thermodynamics]
Cv_expected = [point.heat_capacity for point in analytic_thermodynamics]
logZ_expected = [point.logZ for point in analytic_thermodynamics]

positive_T = findall(>(0.0), thermal_temperatures)

thermodynamics_match = isapprox(thermodynamics.free_energy, F_expected; rtol=1e-11, atol=1e-12) && isapprox(thermodynamics.internal_energy, U_expected; rtol=1e-11, atol=1e-12) && isapprox(thermodynamics.entropy, S_expected; rtol=1e-11, atol=1e-12) && isapprox(thermodynamics.heat_capacity, Cv_expected; rtol=1e-11, atol=1e-12) && isapprox(thermodynamics.logZ[positive_T], logZ_expected[positive_T]; rtol=1e-11, atol=1e-12)

@assert thermodynamics_match


#-------------------------#
# Schottky Anomaly        #
#-------------------------#

Cv_peak_index = argmax(thermodynamics.heat_capacity)

Cv_peak_temperature = thermodynamics.temperatures[Cv_peak_index]
Cv_peak = thermodynamics.heat_capacity[Cv_peak_index]

@assert Cv_peak_temperature > 0
@assert Cv_peak > 0


#---------------------------#
# Thermal State at T = J/kB #
#---------------------------#

thermal_state_exchange = thermal_state(solution; temperature=T_exchange, kB=Kelvin2meV)

β_exchange = inv(Kelvin2meV * T_exchange)

Z_numeric = partition_function(thermal_state_exchange)
logZ_numeric = log_partition_function(thermal_state_exchange)

Z_expected_exchange = exp(-β_exchange * E_singlet) + 3exp(-β_exchange * E_triplet)

@assert isapprox(Z_numeric, Z_expected_exchange; rtol=1e-12, atol=1e-12)
@assert isapprox(logZ_numeric, log(Z_expected_exchange); rtol=1e-12, atol=1e-12)


#--------------------------#
# Thermal Spin Correlation #
#--------------------------#

pair_correlation_T = real(expectation(thermal_state_exchange, S1_dot_S2; hbar=hbar).value)

x_exchange = exp(-β_exchange * spin_gap)
Zshift_exchange = 1 + 3x_exchange

p_singlet = inv(Zshift_exchange)
p_triplet_each = x_exchange / Zshift_exchange

pair_correlation_T_expected = hbar^2 * ((-3 / 4) * p_singlet + 3 * (1 / 4) * p_triplet_each)

@assert isapprox(pair_correlation_T, pair_correlation_T_expected; rtol=1e-12, atol=1e-12)


#----------------------------------#
# Finite-T Static Structure Factor #
#----------------------------------#

static_Sz_T = static_structure_factor(model, solution, Sz_field, [q_staggered]; temperature=T_exchange, convention=spectrum_convention)

S_staggered_T = real(only(static_Sz_T.intensity))

S_staggered_T_expected = (hbar^2 / 2) * (p_singlet + p_triplet_each)

@assert isapprox(S_staggered_T, S_staggered_T_expected; rtol=1e-12, atol=1e-12)


#-------------------------#
# Detailed Balance        #
#-------------------------#

thermal_lines = lehmann_lines(model, solution, Sz_qπ, Sz_minus_qπ; temperature=T_exchange, convention=spectrum_convention, weight_tol=1e-12)

function spectral_weight_at(lines, energy; atol=1e-10)
    return sum(real(lines.weights[i]) for i in eachindex(lines.centers) if isapprox(lines.centers[i], energy; atol=atol, rtol=1e-10))
end

positive_weight = spectral_weight_at(thermal_lines, spin_gap)

negative_weight = spectral_weight_at(thermal_lines, -spin_gap)

positive_weight_expected = (hbar^2 / 2) * p_singlet
negative_weight_expected = (hbar^2 / 2) * p_triplet_each

@assert isapprox(positive_weight, positive_weight_expected; rtol=1e-12, atol=1e-12)
@assert isapprox(negative_weight, negative_weight_expected; rtol=1e-12, atol=1e-12)

detailed_balance_ratio = negative_weight / positive_weight
detailed_balance_expected = exp(-β_exchange * spin_gap)

@assert isapprox(detailed_balance_ratio, detailed_balance_expected; rtol=1e-12, atol=1e-12)

thermal_sumrule_match = isapprox(positive_weight + negative_weight, S_staggered_T; rtol=1e-12, atol=1e-12)

@assert thermal_sumrule_match


#-----------------------------------#
# Finite-T Dynamic Structure Factor #
#-----------------------------------#

thermal_energy_axis = collect(range(-1.5 * spin_gap, 1.5 * spin_gap; length=4001))

dynamic_Sz_T = dynamic_structure_factor(model, solution, Sz_field, [q_staggered], thermal_energy_axis; temperature=T_exchange, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)

positive_index = argmin(abs.(thermal_energy_axis .- spin_gap))
negative_index = argmin(abs.(thermal_energy_axis .+ spin_gap))

positive_peak_intensity = real(dynamic_Sz_T.intensity[1, positive_index])
negative_peak_intensity = real(dynamic_Sz_T.intensity[1, negative_index])


#-----------------#
# Stage 7 Summary #
#-----------------#

println()
println(" ------------------------------- ")
println("| Finite-T Dimer Thermodynamics |")
println(" ------------------------------- ")
println("Exchange temperature [K]    : ", round(T_exchange; digits=6))
println("Temperature range [K]       : ", (first(thermal_temperatures), last(thermal_temperatures)))
println("Analytic thermodynamics     : ", thermodynamics_match)
println()
println("Schottky Anomaly")
println("----------------")
println("Peak temperature [K]        : ", round(Cv_peak_temperature; digits=6))
println("Maximum Cᵥ/kB               : ", round(Cv_peak / Kelvin2meV; digits=6))
println()
println("T = J/kB")
println("--------")
println("Partition function          : ", round(Z_numeric; digits=8))
println("Internal energy [meV]       : ", round(internal_energy(thermal_state_exchange); digits=6))
println("Free energy [meV]           : ", round(free_energy(thermal_state_exchange); digits=6))
println("Entropy / kB                : ", round(entropy(thermal_state_exchange) / Kelvin2meV; digits=6))
println("Heat capacity / kB          : ", round(heat_capacity(thermal_state_exchange) / Kelvin2meV; digits=6))
println("⟨S₁·S₂⟩ / ħ²                : ", round(pair_correlation_T / hbar^2; digits=6))
println()
println("Thermal Scattering")
println("------------------")
println("Sᶻᶻ(q·d=π)                  : ", round(S_staggered_T; digits=6))
println("+J Lehmann weight           : ", round(positive_weight; digits=6))
println("-J Lehmann weight           : ", round(negative_weight; digits=6))
println("Detailed-balance ratio      : ", round(detailed_balance_ratio; digits=8))
println("Expected exp(-βJ)           : ", round(detailed_balance_expected; digits=8))
println("Thermal sum rule            : ", thermal_sumrule_match)
println()
println("Broadened Sᶻᶻ(q,E)")
println("-------------------")
println("Intensity at +J             : ", round(positive_peak_intensity; digits=6))
println("Intensity at -J             : ", round(negative_peak_intensity; digits=6))



#---------------------------#
# Jordan-Wigner Transform   #
#---------------------------#

# Use the same explicit site ordering carried by the source spin representation.
jw = JordanWigner(model.representation.ordering; hbar=hbar)

jw_result = transform(jw, model)
jw_model = jw_result.model
jw_certificate = jw_result.certificate

@assert all(values(jw_result.validation))
@assert jw_certificate.exact
@assert jw_certificate.invertible
@assert jw_certificate.relation === AlgebraIsomorphism
@assert jw_certificate.locality === :nonlocal

@assert jw_model.representation.name === :jordan_wigner_fermion
@assert jw_model.representation.algebra isa FermionAlgebra
@assert jw_model.representation.state_space isa FermionFockSpace
@assert representationdimension(jw_model.representation) == 4

#-----------------------------#
# Observable Registry Mapping #
#-----------------------------#

source_observable_names = sort!(collect(keys(model.observables)); by=string)
jw_observable_names = sort!(collect(keys(jw_model.observables)); by=string)

@assert jw_observable_names == source_observable_names


#-----------------------------#
# Exact Observable Mapping    #
#-----------------------------#

jw_Sz1 = resolve_observable(jw_model, :spin_z_1)
jw_Sz2 = resolve_observable(jw_model, :spin_z_2)

jw_Sz1_expected = hbar * (nf(1) - (1 / 2) * IdentityOperator())

jw_Sz2_expected = hbar * (nf(2) - (1 / 2) * IdentityOperator())

jw_Sz1_residual = Phundamental.Algebra.canonicalize(jw_model.representation.algebra, jw_Sz1 - jw_Sz1_expected; hbar=hbar)

jw_Sz2_residual = Phundamental.Algebra.canonicalize(jw_model.representation.algebra, jw_Sz2 - jw_Sz2_expected; hbar=hbar)

@assert is_symbolic_zero(jw_Sz1_residual)
@assert is_symbolic_zero(jw_Sz2_residual)


#-----------------------------#
# Fermionic Diagonalization   #
#-----------------------------#

jw_basis = computational_basis(jw_model)

@assert basis_dimension(jw_basis) == 4

jw_solution = solve(solver, jw_model)

jw_energies = Float64.(real.(jw_solution.energies))

spectrum_equivalence = isapprox(jw_energies, energies; rtol=1e-12, atol=1e-12)

@assert spectrum_equivalence

maximum_spectrum_difference = maximum(abs.(jw_energies .- energies))



#----------------------------------#
# Transformed Correlation Operator #
#----------------------------------#

jw_S1_dot_S2 = transform_observable(jw, jw_model.representation, S1_dot_S2)

jw_ground_pair_correlation = real(eigenbasis_matrix(jw_model, jw_solution, jw_S1_dot_S2; hbar=hbar)[1, 1])

@assert isapprox(jw_ground_pair_correlation, pair_correlation; rtol=1e-12, atol=1e-12)


#-----------------------------#
# Thermal Equivalence         #
#-----------------------------#

jw_thermodynamics = thermal_curve(jw_solution, thermal_temperatures; kB=Kelvin2meV)

thermal_free_energy_match = isapprox(jw_thermodynamics.free_energy, thermodynamics.free_energy; rtol=1e-12, atol=1e-12)

thermal_internal_energy_match = isapprox(jw_thermodynamics.internal_energy, thermodynamics.internal_energy; rtol=1e-12, atol=1e-12)

thermal_entropy_match = isapprox(jw_thermodynamics.entropy, thermodynamics.entropy; rtol=1e-12, atol=1e-12)

thermal_heat_capacity_match = isapprox(jw_thermodynamics.heat_capacity, thermodynamics.heat_capacity; rtol=1e-12, atol=1e-12)

thermal_logZ_match = isapprox(jw_thermodynamics.logZ[positive_T], thermodynamics.logZ[positive_T]; rtol=1e-12, atol=1e-12)

thermodynamic_equivalence = thermal_free_energy_match && thermal_internal_energy_match && thermal_entropy_match && thermal_heat_capacity_match && thermal_logZ_match

@assert thermodynamic_equivalence



#----------------------------------#
# Thermal Observable Equivalence   #
#----------------------------------#

jw_thermal_state_exchange = thermal_state(jw_solution; temperature=T_exchange, kB=Kelvin2meV)

jw_pair_correlation_T = real(expectation(jw_thermal_state_exchange, jw_S1_dot_S2; hbar=hbar).value)

@assert isapprox(jw_pair_correlation_T, pair_correlation_T; rtol=1e-12, atol=1e-12)


#--------------------------------#
# Static Structure Equivalence   #
#--------------------------------#

jw_static_Sz_T = static_structure_factor(jw_model, jw_solution, Sz_field, [q_staggered]; temperature=T_exchange, convention=spectrum_convention)

static_structure_equivalence = isapprox(jw_static_Sz_T.intensity, static_Sz_T.intensity; rtol=1e-11, atol=1e-12)

@assert static_structure_equivalence



#--------------------------------#
# Dynamic Structure Equivalence  #
#--------------------------------#

jw_dynamic_Sz_T = dynamic_structure_factor(jw_model, jw_solution, Sz_field, [q_staggered], thermal_energy_axis; temperature=T_exchange, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)

dynamic_structure_equivalence = validate_observable_equivalence(dynamic_Sz_T, jw_dynamic_Sz_T; rtol=1e-10, atol=1e-12)

@assert dynamic_structure_equivalence



#-----------------------------#
# Spin-Tensor Equivalence     #
#-----------------------------#

jw_spin_tensor = spin_tensor_structure_factor(jw_model, jw_solution, spin_field, [Q], tensor_axis; temperature=0.0, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)

spin_tensor_equivalence = validate_observable_equivalence(spin_tensor, jw_spin_tensor; rtol=1e-10, atol=1e-12)

@assert spin_tensor_equivalence



#--------------------------------#
# Neutron-Scattering Equivalence #
#--------------------------------#

jw_neutron_response = scattering_intensity(probe, jw_spin_tensor)

neutron_equivalence = validate_observable_equivalence(neutron_response, jw_neutron_response; rtol=1e-10, atol=1e-12)

@assert neutron_equivalence

jw_resolved_neutron = convolve_resolution(resolution, jw_neutron_response)

resolved_neutron_equivalence = validate_observable_equivalence(resolved_neutron, jw_resolved_neutron; rtol=1e-10, atol=1e-12)

@assert resolved_neutron_equivalence




#-----------------#
# Stage 8 Summary #
#-----------------#

maximum_free_energy_difference = maximum(
    abs.(jw_thermodynamics.free_energy .- thermodynamics.free_energy)
)

maximum_internal_energy_difference = maximum(
    abs.(jw_thermodynamics.internal_energy .- thermodynamics.internal_energy)
)

maximum_heat_capacity_difference = maximum(
    abs.(jw_thermodynamics.heat_capacity .- thermodynamics.heat_capacity)
)

println()
println(" ------------------------------------ ")
println("| Jordan-Wigner Representation Check |")
println(" ------------------------------------ ")
println("Source representation        : ", model.representation.name)
println("Target representation        : ", jw_model.representation.name)
println("Transformation relation      : ", jw_certificate.relation)
println("Exact                        : ", jw_certificate.exact)
println("Invertible                   : ", jw_certificate.invertible)
println("Locality                     : ", jw_certificate.locality)
println("Source dimension             : ", representationdimension(model.representation))
println("Target dimension             : ", representationdimension(jw_model.representation))
println()
println("Symbolic Mapping")
println("----------------")
println("S₁ᶻ → ħ(n₁ - 1/2)            : ", is_symbolic_zero(jw_Sz1_residual))
println("S₂ᶻ → ħ(n₂ - 1/2)            : ", is_symbolic_zero(jw_Sz2_residual))
println("Observable registry preserved: ", jw_observable_names == source_observable_names)
println()
println("Spectrum")
println("--------")
println("Equivalent                   : ", spectrum_equivalence)
println("Maximum |ΔE| [meV]           : ", maximum_spectrum_difference)
println("Ground correlation preserved : ", isapprox(jw_ground_pair_correlation, pair_correlation; atol=1e-12))
println()
println("Thermodynamics")
println("--------------")
println("Equivalent                   : ", thermodynamic_equivalence)
println("Maximum |ΔF| [meV]           : ", maximum_free_energy_difference)
println("Maximum |ΔU| [meV]           : ", maximum_internal_energy_difference)
println("Maximum |ΔCᵥ| [meV/K]        : ", maximum_heat_capacity_difference)
println("Thermal correlation preserved: ", isapprox(jw_pair_correlation_T, pair_correlation_T; atol=1e-12))
println()
println("Observables")
println("-----------")
println("Static Sᶻᶻ equivalent        : ", static_structure_equivalence)
println("Dynamic Sᶻᶻ equivalent       : ", dynamic_structure_equivalence)
println("Spin tensor equivalent       : ", spin_tensor_equivalence)
println("Neutron response equivalent  : ", neutron_equivalence)
println("Resolved response equivalent : ", resolved_neutron_equivalence)




#-----------------------------#
# Forward Observable Request  #
#-----------------------------#

# Define the intrinsic observable calculation independently of the representation.
# The callback receives whichever model is actually supplied to the solver.
spin_tensor_request = ObservableRequest(
    :dimer_spin_tensor,
    function (working_model, working_solution)
        working_positions = working_model.parameters[:positions]

        working_spin_field = SpinTensorField(xkeys=[:spin_x_1, :spin_x_2], ykeys=[:spin_y_1, :spin_y_2], zkeys=[:spin_z_1, :spin_z_2], positions=working_positions, normalization=:sqrtN)

        return spin_tensor_structure_factor(working_model, working_solution, working_spin_field, [Q], tensor_axis; 
                                            temperature=0.0, convention=spectrum_convention, broadening=broadening, weight_tol=1e-12)
    end;
    metadata=Dict{Symbol,Any}(:observable => :spin_tensor_dynamic_structure_factor, :temperature_K => 0.0, :momentum_transfer => copy(Q), :spectral_axis => :energy)
   )

#-----------------------------#
# Spin Forward Problem        #
#-----------------------------#

spin_forward_problem = ForwardProblem(model; solver=solver, observable=spin_tensor_request, probe=probe, resolution=resolution, 
                                      metadata=Dict{Symbol,Any}(:example => :SrCu2BO3_2_dimer, :representation_path => :spin))

spin_forward = forward(spin_forward_problem)

@assert spin_forward isa ForwardResult
@assert spin_forward.source_model === model
@assert spin_forward.model === model
@assert isnothing(spin_forward.transformation_result)


#--------------------------------#
# Spin-Pipeline Closure          #
#--------------------------------#

forward_spin_spectrum_match = isapprox(real.(spin_forward.solution.energies), energies; rtol=1e-12, atol=1e-12)

forward_spin_tensor_match = validate_observable_equivalence(spin_forward.observable_result, spin_tensor; rtol=1e-10, atol=1e-12)

forward_spin_probe_match = validate_observable_equivalence(spin_forward.scattering_result, neutron_response; rtol=1e-10, atol=1e-12)

forward_spin_resolution_match = validate_observable_equivalence(spin_forward.result, resolved_neutron; rtol=1e-10, atol=1e-12)

@assert forward_spin_spectrum_match
@assert forward_spin_tensor_match
@assert forward_spin_probe_match
@assert forward_spin_resolution_match


#-----------------------------#
# Jordan-Wigner Forward Path  #
#-----------------------------#

jw_forward_problem = ForwardProblem(model; transformation=jw, solver=solver, observable=spin_tensor_request, probe=probe, resolution=resolution, 
                                    metadata=Dict{Symbol,Any}(:example => :SrCu2BO3_2_dimer, :representation_path => :jordan_wigner))

jw_forward = forward(jw_forward_problem)

@assert jw_forward isa ForwardResult
@assert jw_forward.source_model === model
@assert !isnothing(jw_forward.transformation_result)
@assert jw_forward.model.representation.name === :jordan_wigner_fermion
@assert jw_forward.transformation_result.certificate.exact


#--------------------------------#
# JW-Pipeline Closure            #
#--------------------------------#

forward_jw_spectrum_match = isapprox(real.(jw_forward.solution.energies), jw_energies; rtol=1e-12, atol=1e-12)

forward_jw_tensor_match = validate_observable_equivalence(jw_forward.observable_result, jw_spin_tensor; rtol=1e-10, atol=1e-12)

forward_jw_probe_match = validate_observable_equivalence(jw_forward.scattering_result, jw_neutron_response; rtol=1e-10, atol=1e-12)

forward_jw_resolution_match = validate_observable_equivalence(jw_forward.result, jw_resolved_neutron; rtol=1e-10, atol=1e-12)

@assert forward_jw_spectrum_match
@assert forward_jw_tensor_match
@assert forward_jw_probe_match
@assert forward_jw_resolution_match

#----------------------------------#
# Representation-Independent Result #
#----------------------------------#

forward_representation_spectrum_match = isapprox(real.(spin_forward.solution.energies), real.(jw_forward.solution.energies); rtol=1e-12, atol=1e-12)

forward_representation_tensor_match = validate_observable_equivalence(spin_forward.observable_result, jw_forward.observable_result; rtol=1e-10, atol=1e-12)

forward_representation_probe_match = validate_observable_equivalence(spin_forward.scattering_result, jw_forward.scattering_result; rtol=1e-10, atol=1e-12)

forward_representation_final_match = validate_observable_equivalence(spin_forward.result, jw_forward.result; rtol=1e-10, atol=1e-12)

@assert forward_representation_spectrum_match
@assert forward_representation_tensor_match
@assert forward_representation_probe_match
@assert forward_representation_final_match



#----------------------------------#
# Material-to-Measurement Closure  #
#----------------------------------#

material_forward = forward(material, dimer_cluster, spec; transformation=jw, solver=solver, observable=spin_tensor_request, probe=probe, resolution=resolution)

@assert material_forward.source_model isa ManyBodyModel
@assert material_forward.model.representation.name === :jordan_wigner_fermion

material_forward_final_match = validate_observable_equivalence(material_forward.result, jw_forward.result; rtol=1e-10, atol=1e-12)

@assert material_forward_final_match


#-----------------#
# Stage 9 Summary #
#-----------------#

manual_pipeline_complete = forward_spin_spectrum_match && forward_spin_tensor_match && forward_spin_probe_match && forward_spin_resolution_match

jw_pipeline_complete = forward_jw_spectrum_match && forward_jw_tensor_match && forward_jw_probe_match && forward_jw_resolution_match

representation_independent = forward_representation_spectrum_match && forward_representation_tensor_match && forward_representation_probe_match && forward_representation_final_match

println()
println(" ----------------------------------- ")
println("| End-to-End Forward Pipeline Check |")
println(" ----------------------------------- ")
println("Source model                : ", spin_forward.source_model.representation.name)
println("Spin working representation : ", spin_forward.model.representation.name)
println("JW working representation   : ", jw_forward.model.representation.name)
println()
println("Manual → Forward Closure")
println("------------------------")
println("Spin spectrum               : ", forward_spin_spectrum_match)
println("Spin tensor                 : ", forward_spin_tensor_match)
println("Neutron response            : ", forward_spin_probe_match)
println("Resolved response           : ", forward_spin_resolution_match)
println("Complete spin path          : ", manual_pipeline_complete)
println()
println("Jordan-Wigner Closure")
println("---------------------")
println("JW spectrum                 : ", forward_jw_spectrum_match)
println("JW spin tensor              : ", forward_jw_tensor_match)
println("JW neutron response         : ", forward_jw_probe_match)
println("JW resolved response        : ", forward_jw_resolution_match)
println("Complete JW path            : ", jw_pipeline_complete)
println()
println("Representation Invariance")
println("-------------------------")
println("Spectrum                    : ", forward_representation_spectrum_match)
println("Intrinsic tensor            : ", forward_representation_tensor_match)
println("Probe response              : ", forward_representation_probe_match)
println("Final measured response     : ", forward_representation_final_match)
println("Representation independent  : ", representation_independent)
println()
println("Full API Closure")
println("----------------")
println("Material → measurement      : ", material_forward_final_match)
