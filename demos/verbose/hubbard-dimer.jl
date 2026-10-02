using Phundamental
using LinearAlgebra
using Printf

# =============================================================================
# Stage E1 — Symmetric two-site Hubbard dimer
#
# Reference:
# D. J. Carrascal, J. Ferrer, J. C. Smith, and K. Burke,
# J. Phys.: Condens. Matter 27, 393001 (2015).
# DOI: 10.1088/0953-8984/27/39/393001
#
# Reference parameters: U = 1 and 2t = 1.
# =============================================================================

const t = 0.5
const U = 1.0

# The geometry realizes the two abstract Hubbard sites as a one-angstrom open dimer. The H labels follow the paper's connection to a minimal H2 basis; the chosen distance is bookkeeping and does not parameterize t or U.
lattice = BravaisLattice(reshape([2.0], 1, 1))

basis = BasisSite[BasisSite(:H1, :H, [0.0]),
                  BasisSite(:H2, :H, [0.5])]

crystal = LatticeCrystalStructure(lattice, basis)
cluster = Supercell(crystal, (1,); periodic=false)

material = Material("Carrascal symmetric Hubbard dimer", crystal;
                    species=Dict(:H => lookup_species(:H)),
                    metadata=Dict(:reference => "Carrascal, J. Phys.: Condens. Matter 27, 393001 (2015)",
                                  :doi => "10.1088/0953-8984/27/39/393001",
                                  :model_role => :analytic_electronic_benchmark))

bond_distance = norm(site_position(cluster, 2) - site_position(cluster, 1))

hubbard = HubbardModel(t, U; chemical_potential=0.0, neighbor_cutoff=bond_distance + 1e-8, max_shell=1, hbar=1.0)

model = build_model(material, cluster, hubbard)

@assert model.representation.name === :spinful_fermion
@assert spacedimension(model.representation.state_space) == 16
@assert length(model.parameters[:bonds]) == 1
@assert isapprox(model.parameters[:bonds][1].distance, 1.0; atol=1e-12, rtol=0.0)

mode_map = model.parameters[:fermion_mode_map]

@assert mode_map[(1, :up)] == 1
@assert mode_map[(1, :down)] == 2
@assert mode_map[(2, :up)] == 3
@assert mode_map[(2, :down)] == 4

solver = ExactDiagonalization()
solution = solve(solver, model)

realization = realize(model.hamiltonian, solution.basis; hbar=1.0, sparse=false, check_physical_closure=true)

H_full = realization.matrix

# Carrascal Eq. (28), using the package fermion ordering
#
#   1 = site 1 up
#   2 = site 1 down
#   3 = site 2 up
#   4 = site 2 down
#
# The corresponding Fock bit patterns are:
#
#   |1↑1↓> = 0011
#   |1↑2↓> = 1001
#   |1↓2↑> = 0110
#   |2↑2↓> = 1100

sector_bits = UInt64[0x03, 0x09, 0x06, 0x0c]
sector_states = FermionBasisState.(sector_bits)
sector_indices = [solution.basis.index[state] for state in sector_states]

H_sector = H_full[sector_indices, sector_indices]

H_reference = ComplexF64[+U  -t  +t  0.0;
                         -t  0.0 0.0 -t ;
                         +t  0.0 0.0 +t ;
                         0.0 -t  +t  +U ]

matrix_error = maximum(abs.(H_sector - H_reference))

@assert isapprox(H_sector, H_reference; atol=1e-12, rtol=0.0)

eigensystem = eigen(Hermitian(H_sector))
sector_energies = eigensystem.values
ground_state = eigensystem.vectors[:, 1]

reference_ground_energy = U / 2 - sqrt((2t)^2 + (U / 2)^2)
ground_energy_error = abs(sector_energies[1] - reference_ground_energy)

@assert isapprox(sector_energies[1], reference_ground_energy; atol=1e-12, rtol=0.0)

# Hellmann-Feynman gives the total double occupancy dE0/dU.
double_occupancy = abs2(ground_state[1]) + abs2(ground_state[4])
reference_double_occupancy = 0.5 * (1 - U / sqrt(U^2 + 16t^2))

@assert isapprox(double_occupancy, reference_double_occupancy; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E1       |")
println(" --------------------------------")
@printf("t                         : %.12f\n", t)
@printf("U                         : %.12f\n", U)
@printf("site separation           : %.12f\n", bond_distance)
@printf("full Fock dimension       : %d\n", length(solution.basis))
@printf("N=2, Sz=0 block dimension : %d\n", size(H_sector, 1))
@printf("max |ΔH| vs Eq. (28)      : %.3e\n", matrix_error)
@printf("E0 numerical              : %.12f\n", sector_energies[1])
@printf("E0 Eq. (35)               : %.12f\n", reference_ground_energy)
@printf("|ΔE0|                     : %.3e\n", ground_energy_error)
@printf("double occupancy          : %.12f\n", double_occupancy)
@printf("double occupancy analytic : %.12f\n", reference_double_occupancy)
println("sector energies           : ", round.(sector_energies; digits=12))


# =============================================================================
# Stage E2 — Quantum numbers, singlet character, and exact limits
# =============================================================================

ψ0 = ComplexF64.(groundstate(solution))

function ground_expectation(operator)
    O = realize(operator, solution.basis; hbar=1.0, sparse=false, check_physical_closure=true).matrix
    return real(dot(ψ0, O * ψ0))
end

n1 = model.observables[:density_1]
n2 = model.observables[:density_2]
d1 = model.observables[:double_occupancy_1]
d2 = model.observables[:double_occupancy_2]

Ntot = n1 + n2

Sx_tot = model.observables[:spin_x_1] + model.observables[:spin_x_2]
Sy_tot = model.observables[:spin_y_1] + model.observables[:spin_y_2]
Sz_tot = model.observables[:spin_z_1] + model.observables[:spin_z_2]
S2_tot = Sx_tot * Sx_tot + Sy_tot * Sy_tot + Sz_tot * Sz_tot

N_expectation = ground_expectation(Ntot)
N2_expectation = ground_expectation(Ntot * Ntot)
N_variance = N2_expectation - N_expectation^2

n1_expectation = ground_expectation(n1)
n2_expectation = ground_expectation(n2)

d1_expectation = ground_expectation(d1)
d2_expectation = ground_expectation(d2)
double_occupancy_total = d1_expectation + d2_expectation

Sx_expectation = ground_expectation(Sx_tot)
Sy_expectation = ground_expectation(Sy_tot)
Sz_expectation = ground_expectation(Sz_tot)
S2_expectation = ground_expectation(S2_tot)

@assert isapprox(N_expectation, 2.0; atol=1e-12, rtol=0.0)
@assert isapprox(N_variance, 0.0; atol=1e-12, rtol=0.0)
@assert isapprox(n1_expectation, 1.0; atol=1e-12, rtol=0.0)
@assert isapprox(n2_expectation, 1.0; atol=1e-12, rtol=0.0)
@assert isapprox(d1_expectation, reference_double_occupancy / 2; atol=1e-12, rtol=0.0)
@assert isapprox(d2_expectation, reference_double_occupancy / 2; atol=1e-12, rtol=0.0)
@assert isapprox(double_occupancy_total, reference_double_occupancy; atol=1e-12, rtol=0.0)

@assert isapprox(Sx_expectation, 0.0; atol=1e-12, rtol=0.0)
@assert isapprox(Sy_expectation, 0.0; atol=1e-12, rtol=0.0)
@assert isapprox(Sz_expectation, 0.0; atol=1e-12, rtol=0.0)
@assert isapprox(S2_expectation, 0.0; atol=1e-12, rtol=0.0)

n2_sector_indices = [i for (i, state) in enumerate(solution.basis.states) if state isa FermionBasisState && count_ones(state.bits) == 2]

H_n2 = H_full[n2_sector_indices, n2_sector_indices]
n2_sector_energies = eigvals(Hermitian(H_n2))

reference_excited_singlet = U / 2 + sqrt((2t)^2 + (U / 2)^2)
reference_n2_spectrum = [reference_ground_energy, 
                         0.0, 0.0, 0.0, U,
                         reference_excited_singlet]

@assert length(n2_sector_energies) == 6
@assert isapprox(n2_sector_energies, reference_n2_spectrum; atol=1e-12, rtol=0.0)

function n2_sector_spectrum(t_value::Real, U_value::Real)
    spec = HubbardModel(t_value, U_value; chemical_potential=0.0, 
                        neighbor_cutoff=bond_distance + 1e-8,
                        max_shell=1, hbar=1.0)

    limit_model = build_model(material, cluster, spec)
    limit_solution = solve(ExactDiagonalization(), limit_model)
    limit_realization = realize(limit_model.hamiltonian, limit_solution.basis;
                                hbar=1.0, sparse=false, check_physical_closure=true)

    indices = [i for (i, state) in enumerate(limit_solution.basis.states) if state isa FermionBasisState && count_ones(state.bits) == 2]

    return eigvals(Hermitian(limit_realization.matrix[indices, indices]))
end

noninteracting_spectrum = n2_sector_spectrum(t, 0.0)
atomic_spectrum = n2_sector_spectrum(0.0, U)

reference_noninteracting = [-2t, 0.0, 0.0, 0.0, 0.0, 2t]
reference_atomic = [0.0, 0.0, 0.0, 0.0, U, U]

@assert isapprox(noninteracting_spectrum, reference_noninteracting; atol=1e-12, rtol=0.0)
@assert isapprox(atomic_spectrum, reference_atomic; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E2       |")
println(" --------------------------------")
@printf("⟨N⟩                       : %.12f\n", N_expectation)
@printf("Var(N)                    : %.3e\n", N_variance)
@printf("⟨n₁⟩                      : %.12f\n", n1_expectation)
@printf("⟨n₂⟩                      : %.12f\n", n2_expectation)
@printf("⟨d₁⟩                      : %.12f\n", d1_expectation)
@printf("⟨d₂⟩                      : %.12f\n", d2_expectation)
@printf("⟨d₁+d₂⟩                   : %.12f\n", double_occupancy_total)
@printf("⟨Sx_tot⟩                  : %.3e\n", Sx_expectation)
@printf("⟨Sy_tot⟩                  : %.3e\n", Sy_expectation)
@printf("⟨Sz_tot⟩                  : %.3e\n", Sz_expectation)
@printf("⟨S²_tot⟩                  : %.3e\n", S2_expectation)
println("full N=2 spectrum         : ", round.(n2_sector_energies; digits=12))
println("U = 0 spectrum            : ", round.(noninteracting_spectrum; digits=12))
println("t = 0 spectrum            : ", round.(atomic_spectrum; digits=12))


# =============================================================================
# Stage E3 — Particle-addition and particle-removal sectors
# =============================================================================

ground_sector = FermionSector(nup=1, ndown=1)

removal_up_sector = FermionSector(nup=0, ndown=1)
addition_up_sector = FermionSector(nup=2, ndown=1)

removal_down_sector = FermionSector(nup=1, ndown=0)
addition_down_sector = FermionSector(nup=1, ndown=2)

function solve_complete_fermion_sector(model, sector::FermionSector)
    backend = optimized_matrix_free_operator(model; sector=sector, backend=:fermion)
    dimension = backend.dimension

    solver = OptimizedLanczos(nev=dimension, krylov_dim=max(dimension, 4), tol=1e-12, 
                              sector=sector, backend=:fermion, reorthogonalize=true, seed=17)

    result = solve(solver, model)
    return result, dimension
end

removal_up, removal_up_dimension = solve_complete_fermion_sector(model, removal_up_sector)
addition_up, addition_up_dimension = solve_complete_fermion_sector(model, addition_up_sector)

removal_down, removal_down_dimension = solve_complete_fermion_sector(model, removal_down_sector)
addition_down, addition_down_dimension = solve_complete_fermion_sector(model, addition_down_sector)

@assert removal_up_dimension == 2
@assert addition_up_dimension == 2
@assert removal_down_dimension == 2
@assert addition_down_dimension == 2

removal_up_energies = Float64.(real.(removal_up.energies))
addition_up_energies = Float64.(real.(addition_up.energies))

removal_down_energies = Float64.(real.(removal_down.energies))
addition_down_energies = Float64.(real.(addition_down.energies))

reference_removal_energies = [-t, t]
reference_addition_energies = [U - t, U + t]

@assert isapprox(removal_up_energies, reference_removal_energies; atol=1e-12, rtol=0.0)
@assert isapprox(addition_up_energies, reference_addition_energies; atol=1e-12, rtol=0.0)

@assert isapprox(removal_down_energies, reference_removal_energies; atol=1e-12, rtol=0.0)
@assert isapprox(addition_down_energies, reference_addition_energies; atol=1e-12, rtol=0.0)

@assert isapprox(removal_up_energies, removal_down_energies; atol=1e-12, rtol=0.0)
@assert isapprox(addition_up_energies, addition_down_energies; atol=1e-12, rtol=0.0)

E0 = reference_ground_energy

removal_poles = sort(E0 .- removal_up_energies)
addition_poles = sort(addition_up_energies .- E0)

reference_removal_poles = [E0 - t, E0 + t]

reference_addition_poles = [U - t - E0, U + t - E0]

sort!(reference_removal_poles)
sort!(reference_addition_poles)

@assert isapprox(removal_poles, reference_removal_poles; atol=1e-12, rtol=0.0)
@assert isapprox(addition_poles, reference_addition_poles; atol=1e-12, rtol=0.0)

particle_hole_center = U / 2

@assert isapprox(removal_poles[1] + addition_poles[2], U; atol=1e-12, rtol=0.0)
@assert isapprox(removal_poles[2] + addition_poles[1], U; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E3       |")
println(" --------------------------------")
println("ground sector              : (N↑, N↓) = (1, 1)")
println("spin-up removal sector     : (N↑, N↓) = (0, 1)")
println("spin-up addition sector    : (N↑, N↓) = (2, 1)")
@printf("removal-sector dimension   : %d\n", removal_up_dimension)
@printf("addition-sector dimension  : %d\n", addition_up_dimension)
println("N - 1 energies             : ", round.(removal_up_energies; digits=12))
println("N + 1 energies             : ", round.(addition_up_energies; digits=12))
println("removal poles              : ", round.(removal_poles; digits=12))
println("addition poles             : ", round.(addition_poles; digits=12))
@printf("particle-hole center       : %.12f\n", particle_hole_center)
println("spin-sector equivalence    : true")



# =============================================================================
# Stage E4 — Fermion Green-function workspace closure
# =============================================================================

green_workspace = fermion_green_workspace(model; ground_sector=ground_sector, spin=:up,
                                          ground_krylov_dim=4, green_krylov_dim=2, tol=1e-12,
                                          ground_reorthogonalize=true,
                                          green_reorthogonalize=true,
                                          parallel=false, seed=17)

@assert green_workspace.spin === :up
@assert green_workspace.sites == [1, 2]
@assert green_workspace.modes == [1, 3]

@assert green_workspace.ground_sector.nup == 1
@assert green_workspace.ground_sector.ndown == 1
@assert green_workspace.hole_sector.nup == 0
@assert green_workspace.hole_sector.ndown == 1
@assert green_workspace.particle_sector.nup == 2
@assert green_workspace.particle_sector.ndown == 1

ground_dimension = length(green_workspace.ground_basis)
hole_dimension = length(green_workspace.hole_basis)
particle_dimension = length(green_workspace.particle_basis)

@assert ground_dimension == 4
@assert hole_dimension == 2
@assert particle_dimension == 2

workspace_ground_energy = green_workspace.ground_energy
workspace_ground_error = abs(workspace_ground_energy - reference_ground_energy)

@assert isapprox(workspace_ground_energy, reference_ground_energy; atol=1e-12, rtol=0.0)
@assert green_workspace.metadata[:ground_residual] <= 1e-10

workspace_hole_energies = sort(Float64.(eigvals(Hermitian(Matrix(green_workspace.hole_hamiltonian)))))
workspace_particle_energies = sort(Float64.(eigvals(Hermitian(Matrix(green_workspace.particle_hamiltonian)))))

@assert isapprox(workspace_hole_energies, removal_up_energies; atol=1e-12, rtol=0.0)
@assert isapprox(workspace_particle_energies, addition_up_energies; atol=1e-12, rtol=0.0)

workspace_removal_poles = sort(workspace_ground_energy .- workspace_hole_energies)
workspace_addition_poles = sort(workspace_particle_energies .- workspace_ground_energy)

@assert isapprox(workspace_removal_poles, removal_poles; atol=1e-12, rtol=0.0)
@assert isapprox(workspace_addition_poles, addition_poles; atol=1e-12, rtol=0.0)

particle_chain_lengths = [chain.iterations for chain in green_workspace.particle_resolvents]
hole_chain_lengths = [chain.iterations for chain in green_workspace.hole_resolvents]

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E4       |")
println(" --------------------------------")
println("spin channel                : ", green_workspace.spin)
println("sites                       : ", green_workspace.sites)
println("fermion modes               : ", green_workspace.modes)
println("ground sector               : (N↑, N↓) = (1, 1)")
println("hole sector                 : (N↑, N↓) = (0, 1)")
println("particle sector             : (N↑, N↓) = (2, 1)")
@printf("ground-sector dimension     : %d\n", ground_dimension)
@printf("hole-sector dimension       : %d\n", hole_dimension)
@printf("particle-sector dimension   : %d\n", particle_dimension)
@printf("workspace E0                : %.12f\n", workspace_ground_energy)
@printf("|ΔE0| vs analytic           : %.3e\n", workspace_ground_error)
@printf("ground-state residual       : %.3e\n", green_workspace.metadata[:ground_residual])
println("hole-sector energies        : ", round.(workspace_hole_energies; digits=12))
println("particle-sector energies    : ", round.(workspace_particle_energies; digits=12))
println("workspace removal poles     : ", round.(workspace_removal_poles; digits=12))
println("workspace addition poles    : ", round.(workspace_addition_poles; digits=12))
println("hole Krylov lengths         : ", hole_chain_lengths)
println("particle Krylov lengths     : ", particle_chain_lengths)
println("E1–E3 sector closure        : true")


# =============================================================================
# Stage E5 — Lehmann residues and local Green function
# =============================================================================

hole_eigensystem = eigen(Hermitian(Matrix(green_workspace.hole_hamiltonian)))
particle_eigensystem = eigen(Hermitian(Matrix(green_workspace.particle_hamiltonian)))

hole_seed = fermion_transition_vector(model, 
                                      green_workspace.ground_state, 
                                      green_workspace.ground_basis,
                                      green_workspace.hole_basis, 1;
                                      spin=:up, create=false)

particle_seed = fermion_transition_vector(model,
                                          green_workspace.ground_state,
                                          green_workspace.ground_basis,
                                          green_workspace.particle_basis, 1;
                                          spin=:up, create=true)

hole_weights_by_energy = abs2.(adjoint(hole_eigensystem.vectors) * hole_seed)
particle_weights_by_energy = abs2.(adjoint(particle_eigensystem.vectors) * particle_seed)

hole_poles_by_energy = green_workspace.ground_energy .- hole_eigensystem.values
particle_poles_by_energy = particle_eigensystem.values .- green_workspace.ground_energy

hole_order = sortperm(hole_poles_by_energy)
particle_order = sortperm(particle_poles_by_energy)

hole_poles_e5 = Float64.(hole_poles_by_energy[hole_order])
particle_poles_e5 = Float64.(particle_poles_by_energy[particle_order])

hole_weights = Float64.(hole_weights_by_energy[hole_order])
particle_weights = Float64.(particle_weights_by_energy[particle_order])

weight_split = t / sqrt(U^2 + 16t^2)
reference_outer_weight = 1 / 4 - weight_split
reference_inner_weight = 1 / 4 + weight_split

reference_hole_weights = [reference_outer_weight, reference_inner_weight]

reference_particle_weights = [reference_inner_weight, reference_outer_weight]

@assert isapprox(hole_poles_e5, removal_poles; atol=1e-12, rtol=0.0)
@assert isapprox(particle_poles_e5, addition_poles; atol=1e-12, rtol=0.0)

@assert isapprox(hole_weights, reference_hole_weights; atol=1e-12, rtol=0.0)
@assert isapprox(particle_weights, reference_particle_weights; atol=1e-12, rtol=0.0)

hole_total_weight = sum(hole_weights)
particle_total_weight = sum(particle_weights)
total_spectral_weight = hole_total_weight + particle_total_weight

@assert isapprox(hole_total_weight, 0.5; atol=1e-12, rtol=0.0)
@assert isapprox(particle_total_weight, 0.5; atol=1e-12, rtol=0.0)
@assert isapprox(total_spectral_weight, 1.0; atol=1e-12, rtol=0.0)

reference_poles = [removal_poles[1], removal_poles[2],
                   addition_poles[1], addition_poles[2]]

reference_weights = [reference_outer_weight, reference_inner_weight,
                     reference_inner_weight, reference_outer_weight ]

function reference_local_green(z)
    value = 0.0 + 0.0im
    for (pole, weight) in zip(reference_poles, reference_weights)
        value += weight / (z - pole)
    end
    return value
end

sample_z = ComplexF64[-0.75 + 0.137im,
                      +0.25 + 0.137im,
                      +0.90 + 0.137im,
                      +1.75 + 0.137im]

numerical_green = ComplexF64[]
reference_green = ComplexF64[]
site_symmetry_errors = Float64[]

for z in sample_z
    G = cluster_green_matrix(green_workspace, z)
    push!(numerical_green, G[1, 1])
    push!(reference_green, reference_local_green(z))
    push!(site_symmetry_errors, abs(G[1, 1] - G[2, 2]))
end

green_error = maximum(abs.(numerical_green .- reference_green))
site_symmetry_error = maximum(site_symmetry_errors)

@assert green_error <= 1e-11
@assert site_symmetry_error <= 1e-11

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E5       |")
println(" --------------------------------")
println("removal poles              : ", round.(hole_poles_e5; digits=12))
println("removal weights            : ", round.(hole_weights; digits=12))
println("addition poles             : ", round.(particle_poles_e5; digits=12))
println("addition weights           : ", round.(particle_weights; digits=12))
@printf("removal spectral weight    : %.12f\n", hole_total_weight)
@printf("addition spectral weight   : %.12f\n", particle_total_weight)
@printf("total spectral weight      : %.12f\n", total_spectral_weight)
@printf("max |ΔG₁₁(z)|              : %.3e\n", green_error)
@printf("max |G₁₁-G₂₂|              : %.3e\n", site_symmetry_error)
println("Lehmann residue closure    : true")
println("projected-resolvent closure: true")


# =============================================================================
# Stage E6 — Broadened local spectral function
# =============================================================================

eta = 0.05
spectral_axis = collect(-10.0:0.001:11.0)

green_axis = cluster_green_matrix(green_workspace, spectral_axis; eta=eta, parallel=false)

A11 = -imag.(vec(green_axis[1, 1, :])) ./ π
A22 = -imag.(vec(green_axis[2, 2, :])) ./ π

function reference_local_spectral_density(omega::Real)
    value = 0.0
    for (pole, weight) in zip(reference_poles, reference_weights)
        value += weight * eta / (π * ((omega - pole)^2 + eta^2))
    end
    return value
end

reference_spectrum = reference_local_spectral_density.(spectral_axis)

spectral_error = maximum(abs.(A11 .- reference_spectrum))
site_spectrum_error = maximum(abs.(A11 .- A22))

@assert spectral_error <= 5e-11
@assert site_spectrum_error <= 5e-11

function trapezoid_integral(x, y)
    length(x) == length(y) || throw(DimensionMismatch("integration axes must have equal lengths"))
    length(x) >= 2 || throw(ArgumentError("integration requires at least two points"))
    return sum(diff(x) .* (y[1:(end - 1)] .+ y[2:end])) / 2
end

function lorentzian_window_weight(pole, weight, eta, lower, upper)
    return weight * (atan((upper - pole) / eta) - atan((lower - pole) / eta)) / π
end

numerical_window_weight = trapezoid_integral(spectral_axis, A11)

reference_window_weight = sum(lorentzian_window_weight(pole, weight, eta, first(spectral_axis), last(spectral_axis)) for (pole, weight) in zip(reference_poles, reference_weights))

window_integration_error = abs(numerical_window_weight - reference_window_weight)
spectral_tail_weight = 1.0 - reference_window_weight

@assert window_integration_error <= 1e-6
@assert spectral_tail_weight <= 5e-3

reflection_error = maximum(abs.(A11 .- reverse(A11)))

@assert reflection_error <= 5e-11

function local_peak(axis, intensity, pole; halfwidth=0.30)
    indices = findall(abs.(axis .- pole) .<= halfwidth)
    isempty(indices) && throw(ArgumentError("peak window does not intersect the spectral axis"))
    peak_index = indices[argmax(intensity[indices])]
    return axis[peak_index], intensity[peak_index]
end

peak_results = [local_peak(spectral_axis, A11, pole) for pole in reference_poles]

peak_positions = first.(peak_results)
peak_intensities = last.(peak_results)
peak_position_errors = abs.(peak_positions .- reference_poles)

@assert maximum(peak_position_errors) <= 2e-3

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E6       |")
println(" --------------------------------")
@printf("Lorentzian η               : %.6f\n", eta)
@printf("frequency range            : [%.3f, %.3f]\n", first(spectral_axis), last(spectral_axis))
@printf("frequency step             : %.6f\n", spectral_axis[2] - spectral_axis[1])
println("reference poles            : ", round.(reference_poles; digits=12))
println("broadened peak positions   : ", round.(peak_positions; digits=12))
println("broadened peak intensities : ", round.(peak_intensities; digits=12))
@printf("max peak-position error    : %.3e\n", maximum(peak_position_errors))
@printf("max |ΔA₁₁(ω)|              : %.3e\n", spectral_error)
@printf("max |A₁₁-A₂₂|              : %.3e\n", site_spectrum_error)
@printf("particle-hole reflection   : %.3e\n", reflection_error)
@printf("finite-window weight       : %.12f\n", numerical_window_weight)
@printf("analytic window weight     : %.12f\n", reference_window_weight)
@printf("missing Lorentzian tails   : %.3e\n", spectral_tail_weight)
@printf("window integration error   : %.3e\n", window_integration_error)
println("broadened spectrum closure : true")



# =============================================================================
# Stage E7 — Strong-coupling Carrascal Fig. 14 regression
# =============================================================================

strong_U = 5.0

strong_spec = HubbardModel(t, strong_U; 
                           chemical_potential=0.0,
                           neighbor_cutoff=bond_distance + 1e-8,
                           max_shell=1, hbar=1.0)

strong_model = build_model(material, cluster, strong_spec)

strong_workspace = fermion_green_workspace(strong_model; ground_sector=ground_sector, spin=:up, 
                                           ground_krylov_dim=4, green_krylov_dim=2, tol=1e-12, 
                                           ground_reorthogonalize=true, 
                                           green_reorthogonalize=true, 
                                           parallel=false, seed=17)

strong_reference_E0 = strong_U / 2 - sqrt((2t)^2 + (strong_U / 2)^2)

@assert isapprox(strong_workspace.ground_energy, strong_reference_E0; atol=1e-12, rtol=0.0)

strong_hole_eigensystem = eigen(Hermitian(Matrix(strong_workspace.hole_hamiltonian)))
strong_particle_eigensystem = eigen(Hermitian(Matrix(strong_workspace.particle_hamiltonian)))

strong_hole_seed = fermion_transition_vector(strong_model, 
                                             strong_workspace.ground_state, 
                                             strong_workspace.ground_basis,
                                             strong_workspace.hole_basis, 1;
                                             spin=:up, create=false)

strong_particle_seed = fermion_transition_vector(strong_model,
                                                 strong_workspace.ground_state,
                                                 strong_workspace.ground_basis,
                                                 strong_workspace.particle_basis, 1;
                                                 spin=:up, create=true)

strong_hole_poles_raw = strong_workspace.ground_energy .- strong_hole_eigensystem.values
strong_particle_poles_raw = strong_particle_eigensystem.values .- strong_workspace.ground_energy

strong_hole_weights_raw = abs2.(adjoint(strong_hole_eigensystem.vectors) * strong_hole_seed)
strong_particle_weights_raw = abs2.(adjoint(strong_particle_eigensystem.vectors) * strong_particle_seed)

strong_hole_order = sortperm(strong_hole_poles_raw)
strong_particle_order = sortperm(strong_particle_poles_raw)

strong_hole_poles = Float64.(strong_hole_poles_raw[strong_hole_order])
strong_particle_poles = Float64.(strong_particle_poles_raw[strong_particle_order])

strong_hole_weights = Float64.(strong_hole_weights_raw[strong_hole_order])
strong_particle_weights = Float64.(strong_particle_weights_raw[strong_particle_order])

strong_weight_split = t / sqrt(strong_U^2 + 16t^2)
strong_outer_weight = 1 / 4 - strong_weight_split
strong_inner_weight = 1 / 4 + strong_weight_split

strong_reference_hole_poles = sort([strong_reference_E0 - t, strong_reference_E0 + t])

strong_reference_particle_poles = sort([strong_U - t - strong_reference_E0,
                                        strong_U + t - strong_reference_E0])

strong_reference_hole_weights = [strong_outer_weight, strong_inner_weight]

strong_reference_particle_weights = [strong_inner_weight, strong_outer_weight]

@assert isapprox(strong_hole_poles, strong_reference_hole_poles; atol=1e-12, rtol=0.0)
@assert isapprox(strong_particle_poles, strong_reference_particle_poles; atol=1e-12, rtol=0.0)
@assert isapprox(strong_hole_weights, strong_reference_hole_weights; atol=1e-12, rtol=0.0)
@assert isapprox(strong_particle_weights, strong_reference_particle_weights; atol=1e-12, rtol=0.0)

strong_poles = [strong_hole_poles..., strong_particle_poles...]

strong_weights = [strong_hole_weights..., strong_particle_weights...]

@assert isapprox(sum(strong_weights), 1.0; atol=1e-12, rtol=0.0)

strong_reflection_errors = abs.([strong_poles[1] + strong_poles[4] - strong_U,
                                 strong_poles[2] + strong_poles[3] - strong_U ])

@assert maximum(strong_reflection_errors) <= 1e-12

strong_reference_green(z) = sum(weight / (z - pole) for (pole, weight) in zip(strong_poles, strong_weights))

strong_sample_z = ComplexF64[-0.25 + 0.137im,
                             +0.75 + 0.137im,
                             +4.25 + 0.137im,
                             +5.25 + 0.137im]

strong_green_errors = Float64[]

for z in strong_sample_z
    G = cluster_green_matrix(strong_workspace, z)
    push!(strong_green_errors, abs(G[1, 1] - strong_reference_green(z)))
end

strong_green_error = maximum(strong_green_errors)

@assert strong_green_error <= 1e-11

weak_weight_imbalance = reference_inner_weight - reference_outer_weight
strong_weight_imbalance = strong_inner_weight - strong_outer_weight

@assert strong_weight_imbalance < weak_weight_imbalance
@assert round(strong_poles[2]; digits=1) == 0.3
@assert round(strong_poles[3]; digits=1) == 4.7

println()
println(" --------------------------------")
println("| Hubbard Dimer — Stage E7       |")
println(" --------------------------------")
@printf("t                          : %.12f\n", t)
@printf("U                          : %.12f\n", strong_U)
@printf("U / (2t)                   : %.6f\n", strong_U / (2t))
@printf("E0                         : %.12f\n", strong_workspace.ground_energy)
println("many-body poles            : ", round.(strong_poles; digits=12))
println("many-body weights          : ", round.(strong_weights; digits=12))
@printf("total spectral weight      : %.12f\n", sum(strong_weights))
@printf("particle-hole center       : %.12f\n", strong_U / 2)
@printf("reflection error           : %.3e\n", maximum(strong_reflection_errors))
@printf("max |ΔG₁₁(z)|              : %.3e\n", strong_green_error)
@printf("U=1 weight imbalance       : %.12f\n", weak_weight_imbalance)
@printf("U=5 weight imbalance       : %.12f\n", strong_weight_imbalance)
@printf("Fig. 14 inner hole peak    : %.12f\n", strong_poles[2])
@printf("Fig. 14 inner particle peak: %.12f\n", strong_poles[3])
println("strong-coupling closure    : true")
