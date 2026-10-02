using Phundamental
using LinearAlgebra
using Printf

# =============================================================================
# Holstein dimer benchmark
#
# Reference:
# N. Säkkinen, Y. Peng, H. Appel, and R. van Leeuwen,
# "Many-body Green's function theory for electron-phonon interactions:
# ground state properties of the Holstein dimer,"
# J. Chem. Phys. 143, 234101 (2015).
# DOI: 10.1063/1.4936142
# =============================================================================

# =============================================================================
# Stage EP1 — Coupled model construction and conserved-sector closure
# =============================================================================

A = 1.0
t = 1.0
U = 0.0
omega = 1.0
g = 0.5
phonon_cutoff = 6

lattice = BravaisLattice(reshape([A], 1, 1))
crystal = LatticeCrystalStructure(lattice, [BasisSite(:X, :X, [0.0])])
material = Material("Holstein dimer", crystal; species=Dict(:X => AtomicSpecies(:X; mass=1.0)))
cluster = Supercell(crystal, [2]; periodic=false)

spec = HolsteinModel(t, U, omega, g; phonon_cutoff=phonon_cutoff, shell=1)
model = build_model(material, cluster, spec)

@assert nsites(cluster) == 2
@assert length(model.parameters[:bonds]) == 1
@assert model.parameters[:bonds][1].shell == 1
@assert model.representation.state_space isa CompositeStateSpace
@assert model.representation.algebra isa CompositeAlgebra

realization = hamiltonian_matrix(model; sparse=false)
H = realization.matrix
basis = realization.basis

expected_full_dimension = 16 * (phonon_cutoff + 1)^2
@assert size(H) == (expected_full_dimension, expected_full_dimension)
@assert length(basis.states) == expected_full_dimension

hermiticity_error = norm(H - H') / max(norm(H), 1.0)
@assert hermiticity_error <= 1e-12

Ne_operator = nf(1) + nf(2) + nf(3) + nf(4)
Nup_operator = nf(1) + nf(3)
Ndown_operator = nf(2) + nf(4)

Ne = realize(Ne_operator, basis; sparse=false).matrix
Nup = realize(Nup_operator, basis; sparse=false).matrix
Ndown = realize(Ndown_operator, basis; sparse=false).matrix

electron_number_error = norm(H * Ne - Ne * H) / max(norm(H), 1.0)
spin_up_number_error = norm(H * Nup - Nup * H) / max(norm(H), 1.0)
spin_down_number_error = norm(H * Ndown - Ndown * H) / max(norm(H), 1.0)

@assert electron_number_error <= 1e-12
@assert spin_up_number_error <= 1e-12
@assert spin_down_number_error <= 1e-12

up_mask = (UInt64(1) << 0) | (UInt64(1) << 2)
down_mask = (UInt64(1) << 1) | (UInt64(1) << 3)

function electron_counts(state::CompositeBasisState)
    fermion_state = state.parts[1]
    bits = fermion_state.bits
    nup = count_ones(bits & up_mask)
    ndown = count_ones(bits & down_mask)
    return nup, ndown
end

two_electron_sector = findall(state -> sum(electron_counts(state)) == 2, basis.states)
singlet_sz0_sector = findall(state -> electron_counts(state) == (1, 1), basis.states)

expected_two_electron_dimension = 6 * (phonon_cutoff + 1)^2
expected_sz0_dimension = 4 * (phonon_cutoff + 1)^2

@assert length(two_electron_sector) == expected_two_electron_dimension
@assert length(singlet_sz0_sector) == expected_sz0_dimension

complement = setdiff(eachindex(basis.states), singlet_sz0_sector)
Hsector = H[singlet_sz0_sector, singlet_sz0_sector]
sector_leakage = norm(H[complement, singlet_sz0_sector]) / max(norm(Hsector), 1.0)

@assert sector_leakage <= 1e-12

sector_eigenvalues = eigvals(Hermitian(Hsector))
sector_ground_energy = first(sector_eigenvalues)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP1     |")
println(" --------------------------------")
@printf("site count                    : %d\n", nsites(cluster))
@printf("bond count                    : %d\n", length(model.parameters[:bonds]))
@printf("fermion mode count            : %d\n", 4)
@printf("phonon mode count             : %d\n", 2)
@printf("phonon cutoff                 : %d\n", phonon_cutoff)
@printf("full Hilbert dimension        : %d\n", size(H, 1))
@printf("N_e = 2 sector dimension      : %d\n", length(two_electron_sector))
@printf("N_up = N_down = 1 dimension   : %d\n", length(singlet_sz0_sector))
@printf("Hermiticity error             : %.3e\n", hermiticity_error)
@printf("[H, N_e] relative norm        : %.3e\n", electron_number_error)
@printf("[H, N_up] relative norm       : %.3e\n", spin_up_number_error)
@printf("[H, N_down] relative norm     : %.3e\n", spin_down_number_error)
@printf("sector leakage                : %.3e\n", sector_leakage)
@printf("sector ground energy          : %.12f\n", sector_ground_energy)
println("coupled-sector closure        : true")


# =============================================================================
# Stage EP2 — Decoupled electron-plus-phonon spectrum
# =============================================================================

decoupled_spec = HolsteinModel(t, U, omega, 0.0; phonon_cutoff=phonon_cutoff, shell=1)
decoupled_model = build_model(material, cluster, decoupled_spec)

decoupled_realization = hamiltonian_matrix(decoupled_model; sparse=false)
H0 = decoupled_realization.matrix
basis0 = decoupled_realization.basis

decoupled_sector = findall(state -> electron_counts(state) == (1, 1), basis0.states)
H0sector = H0[decoupled_sector, decoupled_sector]
package_spectrum = eigvals(Hermitian(H0sector))

electronic_energies = [-2t, 0.0, 0.0, 2t]
reference_spectrum = sort([
    electronic_energy + omega * (n1 + n2)
    for electronic_energy in electronic_energies
    for n1 in 0:phonon_cutoff
    for n2 in 0:phonon_cutoff
])

@assert length(package_spectrum) == length(reference_spectrum)

spectrum_error = maximum(abs.(package_spectrum .- reference_spectrum))
ground_energy_error = abs(first(package_spectrum) + 2t)

@assert spectrum_error <= 1e-12
@assert ground_energy_error <= 1e-12

function energy_degeneracy(spectrum::AbstractVector{<:Real}, energy::Real; atol::Real=1e-12)
    return count(value -> isapprox(value, energy; atol=atol, rtol=0.0), spectrum)
end

reference_low_energies = [-2.0, -1.0, 0.0, 1.0, 2.0]
expected_degeneracies = [1, 2, 5, 8, 12]
package_degeneracies = [energy_degeneracy(package_spectrum, energy) for energy in reference_low_energies]

@assert package_degeneracies == expected_degeneracies

unique_spectrum = unique(round.(package_spectrum; digits=12))
first_gap = unique_spectrum[2] - unique_spectrum[1]

@assert isapprox(first_gap, omega; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP2     |")
println(" --------------------------------")
@printf("sector dimension               : %d\n", length(package_spectrum))
@printf("maximum spectrum error         : %.3e\n", spectrum_error)
@printf("ground energy                  : %.12f\n", first(package_spectrum))
@printf("ground-energy error            : %.3e\n", ground_energy_error)
@printf("first excitation gap           : %.12f\n", first_gap)
println("low-energy levels              : ", reference_low_energies)
println("low-energy degeneracies        : ", package_degeneracies)
println("decoupled-spectrum closure     : true")


# =============================================================================
# Stage EP3 — Atomic Holstein / Lang-Firsov limit
# =============================================================================

function truncated_displaced_oscillator_spectrum(omega::Real, coupling::Real, cutoff::Integer)
    dimension = cutoff + 1
    a = zeros(Float64, dimension, dimension)
    for n in 1:cutoff
        a[n, n + 1] = sqrt(n)
    end
    number = Diagonal(Float64.(0:cutoff))
    H = omega .* number .+ coupling .* (a .+ a')
    return eigvals(Hermitian(H))
end

function atomic_reference_spectrum(g::Real, omega::Real, cutoff::Integer)
    occupations = [(2, 0), (1, 1), (1, 1), (0, 2)]
    local_spectra = Dict(n => truncated_displaced_oscillator_spectrum(omega, g * n, cutoff) for n in 0:2)

    reference = Float64[]
    sizehint!(reference, 4 * (cutoff + 1)^2)
    for (n1, n2) in occupations
        for E1 in local_spectra[n1], E2 in local_spectra[n2]
            push!(reference, E1 + E2)
        end
    end
    sort!(reference)
    return reference
end

atomic_cutoffs = [4, 6, 8, 10, 12, 14, 16]
atomic_ground_energies = Float64[]
atomic_reference_errors = Float64[]

for cutoff in atomic_cutoffs
    atomic_spec = HolsteinModel(0.0, U, omega, g; phonon_cutoff=cutoff, shell=1)
    atomic_model = build_model(material, cluster, atomic_spec)

    atomic_realization = hamiltonian_matrix(atomic_model; sparse=false)
    Hatomic = atomic_realization.matrix
    atomic_basis = atomic_realization.basis
    atomic_sector = findall(state -> electron_counts(state) == (1, 1), atomic_basis.states)
    Hatomic_sector = Hatomic[atomic_sector, atomic_sector]

    package_atomic_spectrum = eigvals(Hermitian(Hatomic_sector))
    reference_atomic_spectrum = atomic_reference_spectrum(g, omega, cutoff)

    @assert length(package_atomic_spectrum) == length(reference_atomic_spectrum)

    reference_error = maximum(abs.(package_atomic_spectrum .- reference_atomic_spectrum))
    push!(atomic_reference_errors, reference_error)
    push!(atomic_ground_energies, first(package_atomic_spectrum))

    @assert reference_error <= 1e-11
end

analytic_atomic_ground = -4g^2 / omega
atomic_ground_errors = abs.(atomic_ground_energies .- analytic_atomic_ground)

@assert all(atomic_ground_errors[i + 1] <= atomic_ground_errors[i] + 1e-14 for i in 1:length(atomic_ground_errors)-1)
@assert last(atomic_ground_errors) <= 1e-10

expected_bipolaron_offset = -4g^2 / omega
expected_separated_offset = -2g^2 / omega
atomic_binding_energy = expected_separated_offset - expected_bipolaron_offset

@assert isapprox(expected_bipolaron_offset, -1.0; atol=1e-12, rtol=0.0)
@assert isapprox(expected_separated_offset, -0.5; atol=1e-12, rtol=0.0)
@assert isapprox(atomic_binding_energy, 0.5; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP3     |")
println(" --------------------------------")
println("phonon cutoffs                  : ", atomic_cutoffs)
println("ground energies                 : ", atomic_ground_energies)
println("analytic ground energy          : ", analytic_atomic_ground)
println("ground-energy errors            : ", atomic_ground_errors)
println("finite-cutoff spectrum errors   : ", atomic_reference_errors)
@printf("bipolaron atomic offset         : %.12f\n", expected_bipolaron_offset)
@printf("separated-electron offset       : %.12f\n", expected_separated_offset)
@printf("atomic binding energy           : %.12f\n", atomic_binding_energy)
println("atomic-limit closure            : true")


# =============================================================================
# Stage EP4 — Finite-hopping electron-phonon cutoff convergence
# =============================================================================

coupled_cutoffs = [4, 6, 8, 10, 12, 14, 16]
coupled_low_spectra = Vector{Vector{Float64}}()
coupled_sector_dimensions = Int[]

for cutoff in coupled_cutoffs
    coupled_spec = HolsteinModel(t, U, omega, g; phonon_cutoff=cutoff, shell=1)
    coupled_model = build_model(material, cluster, coupled_spec)

    coupled_realization = hamiltonian_matrix(coupled_model; sparse=true)
    Hcoupled = coupled_realization.matrix
    coupled_basis = coupled_realization.basis
    coupled_sector = findall(state -> electron_counts(state) == (1, 1), coupled_basis.states)

    expected_dimension = 4 * (cutoff + 1)^2
    @assert length(coupled_sector) == expected_dimension
    push!(coupled_sector_dimensions, expected_dimension)

    Hcoupled_sector = Matrix(Hcoupled[coupled_sector, coupled_sector])
    low_spectrum = Float64.(eigvals(Hermitian(Hcoupled_sector), 1:4))
    push!(coupled_low_spectra, low_spectrum)
end

coupled_ground_energies = first.(coupled_low_spectra)
reference_low_spectrum = last(coupled_low_spectra)
reference_ground_energy = first(reference_low_spectrum)

ground_energy_errors = abs.(coupled_ground_energies .- reference_ground_energy)
low_spectrum_errors = [maximum(abs.(spectrum .- reference_low_spectrum)) for spectrum in coupled_low_spectra]

@assert all(coupled_ground_energies[i + 1] <= coupled_ground_energies[i] + 1e-12 for i in 1:length(coupled_ground_energies)-1)
@assert all(ground_energy_errors[i + 1] <= ground_energy_errors[i] + 1e-12 for i in 1:length(ground_energy_errors)-1)
@assert ground_energy_errors[end - 1] <= 1e-10
@assert low_spectrum_errors[end - 1] <= 1e-9

coupled_gaps = [spectrum[2:4] .- spectrum[1] for spectrum in coupled_low_spectra]
reference_gaps = last(coupled_gaps)
gap_errors = [maximum(abs.(gaps .- reference_gaps)) for gaps in coupled_gaps]

@assert gap_errors[end - 1] <= 1e-9

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP4     |")
println(" --------------------------------")
println("phonon cutoffs                  : ", coupled_cutoffs)
println("sector dimensions               : ", coupled_sector_dimensions)
println("ground energies                 : ", coupled_ground_energies)
@printf("reference ground energy         : %.12f\n", reference_ground_energy)
println("ground-energy errors            : ", ground_energy_errors)
println("low-spectrum errors             : ", low_spectrum_errors)
println("lowest three excitation gaps    :")
for (cutoff, gaps) in zip(coupled_cutoffs, coupled_gaps)
    @printf("  Nph = %2d : [%.12f, %.12f, %.12f]\n", cutoff, gaps...)
end
println("gap errors                      : ", gap_errors)
println("finite-coupling closure         : true")


# =============================================================================
# Stage EP5 — Coupled ground-state observables
# =============================================================================

observable_cutoff = 16

observable_spec = HolsteinModel(t, U, omega, g; phonon_cutoff=observable_cutoff, shell=1)
observable_model = build_model(material, cluster, observable_spec)

observable_realization = hamiltonian_matrix(observable_model; sparse=true)
Hobservable = observable_realization.matrix
observable_basis = observable_realization.basis
observable_sector = findall(state -> electron_counts(state) == (1, 1), observable_basis.states)
Hobservable_sector = Matrix(Hobservable[observable_sector, observable_sector])

ground_system = eigen(Hermitian(Hobservable_sector))
E0 = ground_system.values[1]
psi0 = ground_system.vectors[:, 1]

function sector_operator(operator, basis, sector)
    matrix = realize(operator, basis; sparse=true).matrix
    return matrix[sector, sector]
end

function ground_expectation(operator, basis, sector, state)
    O = sector_operator(operator, basis, sector)
    value = dot(state, O * state)
    @assert abs(imag(value)) <= 1e-11
    return real(value)
end

density_1 = observable_model.observables[:density_1]
density_2 = observable_model.observables[:density_2]
double_1 = observable_model.observables[:double_occupancy_1]
double_2 = observable_model.observables[:double_occupancy_2]
phonon_number_1 = observable_model.observables[:phonon_number_1]
phonon_number_2 = observable_model.observables[:phonon_number_2]
displacement_1 = observable_model.observables[:phonon_displacement_1]
displacement_2 = observable_model.observables[:phonon_displacement_2]

n1 = ground_expectation(density_1, observable_basis, observable_sector, psi0)
n2 = ground_expectation(density_2, observable_basis, observable_sector, psi0)
D1 = ground_expectation(double_1, observable_basis, observable_sector, psi0)
D2 = ground_expectation(double_2, observable_basis, observable_sector, psi0)
Nph1 = ground_expectation(phonon_number_1, observable_basis, observable_sector, psi0)
Nph2 = ground_expectation(phonon_number_2, observable_basis, observable_sector, psi0)
X1 = ground_expectation(displacement_1, observable_basis, observable_sector, psi0)
X2 = ground_expectation(displacement_2, observable_basis, observable_sector, psi0)

density_difference = density_1 - density_2
displacement_difference = displacement_1 - displacement_2
charge_fluctuation_operator = density_difference * density_difference
electron_phonon_correlation_operator = density_difference * displacement_difference
density_correlation_operator = density_1 * density_2
coupling_operator = g * (density_1 * displacement_1 + density_2 * displacement_2)

charge_fluctuation = ground_expectation(charge_fluctuation_operator, observable_basis, observable_sector, psi0)
electron_phonon_correlation = ground_expectation(electron_phonon_correlation_operator, observable_basis, observable_sector, psi0)
density_correlation = ground_expectation(density_correlation_operator, observable_basis, observable_sector, psi0)
coupling_energy = ground_expectation(coupling_operator, observable_basis, observable_sector, psi0)

phonon_energy = omega * (Nph1 + Nph2)
hopping_energy = E0 - phonon_energy - coupling_energy

mean_double_occupancy = 0.5 * (D1 + D2)
mean_phonon_number = 0.5 * (Nph1 + Nph2)
mean_displacement = 0.5 * (X1 + X2)

@assert isapprox(E0, reference_ground_energy; atol=1e-11, rtol=0.0)

@assert isapprox(n1, 1.0; atol=1e-11, rtol=0.0)
@assert isapprox(n2, 1.0; atol=1e-11, rtol=0.0)
@assert isapprox(X1, -1.0; atol=1e-11, rtol=0.0)
@assert isapprox(X2, -1.0; atol=1e-11, rtol=0.0)

@assert isapprox(charge_fluctuation, 4 * (D1 + D2); atol=1e-11, rtol=0.0)
@assert isapprox(density_correlation + D1 + D2, 1.0; atol=1e-11, rtol=0.0)

uniform_difference_decomposition = 0.5 * ((n1 + n2) * (X1 + X2) + electron_phonon_correlation)
@assert isapprox(coupling_energy / g, uniform_difference_decomposition; atol=1e-10, rtol=0.0)

reconstructed_energy = hopping_energy + phonon_energy + coupling_energy
@assert isapprox(reconstructed_energy, E0; atol=1e-11, rtol=0.0)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP5     |")
println(" --------------------------------")
@printf("phonon cutoff                  : %d\n", observable_cutoff)
@printf("ground energy                  : %.12f\n", E0)
@printf("<n1>                           : %.12f\n", n1)
@printf("<n2>                           : %.12f\n", n2)
@printf("<D1>                           : %.12f\n", D1)
@printf("<D2>                           : %.12f\n", D2)
@printf("mean double occupancy          : %.12f\n", mean_double_occupancy)
@printf("<Nph1>                         : %.12f\n", Nph1)
@printf("<Nph2>                         : %.12f\n", Nph2)
@printf("mean phonon number             : %.12f\n", mean_phonon_number)
@printf("<X1>                           : %.12f\n", X1)
@printf("<X2>                           : %.12f\n", X2)
@printf("mean displacement              : %.12f\n", mean_displacement)
@printf("<(n1-n2)^2>                    : %.12f\n", charge_fluctuation)
@printf("<n1 n2>                        : %.12f\n", density_correlation)
@printf("<(n1-n2)(X1-X2)>               : %.12f\n", electron_phonon_correlation)
@printf("hopping energy                 : %.12f\n", hopping_energy)
@printf("phonon energy                  : %.12f\n", phonon_energy)
@printf("electron-phonon energy         : %.12f\n", coupling_energy)
@printf("reconstructed total energy     : %.12f\n", reconstructed_energy)
println("ground-state observable closure: true")



# =============================================================================
# Stage EP6 — Exact electronic Lehmann spectrum
# =============================================================================

function sector_indices(basis, nup::Integer, ndown::Integer)
    return findall(state -> electron_counts(state) == (nup, ndown), basis.states)
end

function local_electron_lehmann_data(H, basis, ground_sector, psi0, E0, mode::Integer)
    nup0, ndown0 = electron_counts(basis.states[first(ground_sector)])

    spin = isodd(mode) ? :up : :down
    particle_counts = spin === :up ? (nup0 + 1, ndown0) : (nup0, ndown0 + 1)
    hole_counts = spin === :up ? (nup0 - 1, ndown0) : (nup0, ndown0 - 1)

    particle_sector = sector_indices(basis, particle_counts...)
    hole_sector = sector_indices(basis, hole_counts...)

    Hparticle = Matrix(H[particle_sector, particle_sector])
    Hhole = Matrix(H[hole_sector, hole_sector])

    particle_system = eigen(Hermitian(Hparticle))
    hole_system = eigen(Hermitian(Hhole))

    annihilation = realize(c(mode), basis; sparse=true).matrix
    creation = adjoint(annihilation)

    particle_seed = creation[particle_sector, ground_sector] * psi0
    hole_seed = annihilation[hole_sector, ground_sector] * psi0

    particle_amplitudes = adjoint(particle_system.vectors) * particle_seed
    hole_amplitudes = adjoint(hole_system.vectors) * hole_seed

    particle_weights = abs2.(particle_amplitudes)
    hole_weights = abs2.(hole_amplitudes)

    particle_centers = Float64.(particle_system.values .- E0)
    hole_centers = Float64.(-(hole_system.values .- E0))

    centers = vcat(hole_centers, particle_centers)
    weights = vcat(hole_weights, particle_weights)

    order = sortperm(centers)
    return (
        centers=centers[order],
        weights=weights[order],
        particle_centers=particle_centers,
        particle_weights=particle_weights,
        hole_centers=hole_centers,
        hole_weights=hole_weights,
        particle_sector=particle_sector,
        hole_sector=hole_sector,
    )
end

function lorentzian_spectrum(centers, weights, axis, eta::Real)
    spectrum = zeros(Float64, length(axis))
    for (center, weight) in zip(centers, weights)
        spectrum .+= weight .* (eta / π) ./ ((axis .- center).^2 .+ eta^2)
    end
    return spectrum
end

# -----------------------------------------------------------------------------
# EP6a — Exactly soluble g = 0 spectral function
# -----------------------------------------------------------------------------

spectral_cutoff = 10

spectral_free_spec = HolsteinModel(t, U, omega, 0.0; phonon_cutoff=spectral_cutoff, shell=1)
spectral_free_model = build_model(material, cluster, spectral_free_spec)

spectral_free_realization = hamiltonian_matrix(spectral_free_model; sparse=true)
Hfree = spectral_free_realization.matrix
basis_free = spectral_free_realization.basis

ground_free_sector = sector_indices(basis_free, 1, 1)
Hfree_ground = Matrix(Hfree[ground_free_sector, ground_free_sector])
free_ground_system = eigen(Hermitian(Hfree_ground))

E0_free = free_ground_system.values[1]
psi0_free = free_ground_system.vectors[:, 1]

free_site_1 = local_electron_lehmann_data(Hfree, basis_free, ground_free_sector, psi0_free, E0_free, 1)

free_weight_tol = 1e-10
free_active = findall(>(free_weight_tol), free_site_1.weights)
free_centers = free_site_1.centers[free_active]
free_weights = free_site_1.weights[free_active]

@assert length(free_centers) == 2
@assert free_centers ≈ [-1.0, 1.0] atol=1e-12 rtol=0.0
@assert free_weights ≈ [0.5, 0.5] atol=1e-12 rtol=0.0
@assert isapprox(sum(free_site_1.weights), 1.0; atol=1e-12, rtol=0.0)
@assert isapprox(sum(free_site_1.hole_weights), 0.5; atol=1e-12, rtol=0.0)
@assert isapprox(sum(free_site_1.particle_weights), 0.5; atol=1e-12, rtol=0.0)

# -----------------------------------------------------------------------------
# EP6b — Coupled g = 0.5 spectral function
# -----------------------------------------------------------------------------

coupled_site_1 = local_electron_lehmann_data(Hobservable, observable_basis, observable_sector, psi0, E0, 1)
coupled_site_2 = local_electron_lehmann_data(Hobservable, observable_basis, observable_sector, psi0, E0, 3)

total_weight_1 = sum(coupled_site_1.weights)
hole_weight_1 = sum(coupled_site_1.hole_weights)
particle_weight_1 = sum(coupled_site_1.particle_weights)

total_weight_2 = sum(coupled_site_2.weights)
hole_weight_2 = sum(coupled_site_2.hole_weights)
particle_weight_2 = sum(coupled_site_2.particle_weights)

@assert isapprox(total_weight_1, 1.0; atol=1e-11, rtol=0.0)
@assert isapprox(hole_weight_1, 0.5; atol=1e-11, rtol=0.0)
@assert isapprox(particle_weight_1, 0.5; atol=1e-11, rtol=0.0)

@assert isapprox(total_weight_2, 1.0; atol=1e-11, rtol=0.0)
@assert isapprox(hole_weight_2, 0.5; atol=1e-11, rtol=0.0)
@assert isapprox(particle_weight_2, 0.5; atol=1e-11, rtol=0.0)

spectral_axis = collect(range(-6.0, 6.0; length=2401))
spectral_eta = 0.03

A1 = lorentzian_spectrum(coupled_site_1.centers, coupled_site_1.weights, spectral_axis, spectral_eta)
A2 = lorentzian_spectrum(coupled_site_2.centers, coupled_site_2.weights, spectral_axis, spectral_eta)

site_symmetry_error = maximum(abs.(A1 .- A2))
@assert site_symmetry_error <= 1e-10

spectral_first_moment = sum(coupled_site_1.centers .* coupled_site_1.weights)

annihilation_1 = realize(c(1), observable_basis; sparse=true).matrix
creation_1 = adjoint(annihilation_1)
Hfull = Hobservable

psi_full = zeros(ComplexF64, length(observable_basis.states))
psi_full[observable_sector] .= psi0

commutator = Hfull * creation_1 - creation_1 * Hfull
first_moment_operator = annihilation_1 * commutator + commutator * annihilation_1
first_moment_reference = real(dot(psi_full, first_moment_operator * psi_full))

@assert isapprox(spectral_first_moment, first_moment_reference; atol=1e-10, rtol=0.0)

significant_weight = 1e-6
significant_lines = findall(>(significant_weight), coupled_site_1.weights)
significant_centers = coupled_site_1.centers[significant_lines]
significant_weights = coupled_site_1.weights[significant_lines]

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP6     |")
println(" --------------------------------")
@printf("free ground energy              : %.12f\n", E0_free)
println("free active pole centers        : ", free_centers)
println("free active pole weights        : ", free_weights)
@printf("coupled total spectral weight   : %.12f\n", total_weight_1)
@printf("coupled removal weight          : %.12f\n", hole_weight_1)
@printf("coupled addition weight         : %.12f\n", particle_weight_1)
@printf("site-spectrum symmetry error    : %.3e\n", site_symmetry_error)
@printf("spectral first moment           : %.12f\n", spectral_first_moment)
@printf("commutator first moment         : %.12f\n", first_moment_reference)
@printf("significant Lehmann lines       : %d\n", length(significant_lines))
println("significant pole centers        : ", significant_centers)
println("significant pole weights        : ", significant_weights)
println("electronic-spectrum closure     : true")


# =============================================================================
# Stage EP7 — Neutral charge and lattice-displacement response
# =============================================================================

function neutral_lehmann_data(operator, basis, sector, eigensystem)
    O = sector_operator(operator, basis, sector)
    ground_energy = eigensystem.values[1]
    ground_state = eigensystem.vectors[:, 1]
    seed = O * ground_state
    amplitudes = adjoint(eigensystem.vectors) * seed
    centers = Float64.(eigensystem.values .- ground_energy)
    weights = Float64.(abs2.(amplitudes))
    return (centers=centers, weights=weights, operator=O, seed=seed)
end

function cross_lehmann_data(operator_a, operator_b, basis, sector, eigensystem)
    A = sector_operator(operator_a, basis, sector)
    B = sector_operator(operator_b, basis, sector)
    ground_energy = eigensystem.values[1]
    ground_state = eigensystem.vectors[:, 1]
    amplitudes_a = adjoint(eigensystem.vectors) * (A * ground_state)
    amplitudes_b = adjoint(eigensystem.vectors) * (B * ground_state)
    centers = Float64.(eigensystem.values .- ground_energy)
    weights = conj.(amplitudes_a) .* amplitudes_b
    return (centers=centers, weights=weights)
end

function coalesce_inelastic_lines(centers, weights; center_tol::Real=1e-10, weight_tol::Real=1e-12)
    selected = findall(i -> centers[i] > center_tol && abs(weights[i]) > weight_tol, eachindex(centers))
    isempty(selected) && return (centers=Float64[], weights=eltype(weights)[])

    order = sort(selected; by=i -> centers[i])
    merged_centers = Float64[]
    merged_weights = eltype(weights)[]

    current_center = centers[first(order)]
    current_weight = weights[first(order)]

    for index in Iterators.drop(order, 1)
        if abs(centers[index] - current_center) <= center_tol
            current_weight += weights[index]
        else
            push!(merged_centers, current_center)
            push!(merged_weights, current_weight)
            current_center = centers[index]
            current_weight = weights[index]
        end
    end

    push!(merged_centers, current_center)
    push!(merged_weights, current_weight)

    retained = findall(weight -> abs(weight) > weight_tol, merged_weights)
    return (centers=merged_centers[retained], weights=merged_weights[retained])
end

function response_first_moment(H, response)
    seed = response.seed
    ground_state = ground_system.vectors[:, 1]
    commutator_seed = H * seed - response.operator * (H * ground_state)
    return real(dot(seed, commutator_seed))
end

function strongest_response_lines(lines; count::Integer=8)
    isempty(lines.weights) && return (centers=Float64[], weights=eltype(lines.weights)[])
    order = sortperm(abs.(lines.weights); rev=true)
    selected = order[1:min(count, length(order))]
    selected = sort(selected; by=i -> lines.centers[i])
    return (centers=lines.centers[selected], weights=lines.weights[selected])
end

# -----------------------------------------------------------------------------
# EP7a — Exactly soluble g = 0 neutral responses
# -----------------------------------------------------------------------------

free_density_difference = spectral_free_model.observables[:density_1] - spectral_free_model.observables[:density_2]
free_displacement_difference = spectral_free_model.observables[:phonon_displacement_1] - spectral_free_model.observables[:phonon_displacement_2]

free_charge_response = neutral_lehmann_data(free_density_difference, basis_free, ground_free_sector, free_ground_system)
free_phonon_response = neutral_lehmann_data(free_displacement_difference, basis_free, ground_free_sector, free_ground_system)
free_cross_response = cross_lehmann_data(free_density_difference, free_displacement_difference, basis_free, ground_free_sector, free_ground_system)

free_charge_lines = coalesce_inelastic_lines(free_charge_response.centers, free_charge_response.weights)
free_phonon_lines = coalesce_inelastic_lines(free_phonon_response.centers, free_phonon_response.weights)

@assert length(free_charge_lines.centers) == 1
@assert isapprox(free_charge_lines.centers[1], 2t; atol=1e-12, rtol=0.0)
@assert isapprox(free_charge_lines.weights[1], 2.0; atol=1e-12, rtol=0.0)

@assert length(free_phonon_lines.centers) == 1
@assert isapprox(free_phonon_lines.centers[1], omega; atol=1e-12, rtol=0.0)
@assert isapprox(free_phonon_lines.weights[1], 2.0; atol=1e-12, rtol=0.0)

@assert maximum(abs.(free_cross_response.weights)) <= 1e-12

# -----------------------------------------------------------------------------
# EP7b — Coupled neutral responses
# -----------------------------------------------------------------------------

charge_response = neutral_lehmann_data(density_difference, observable_basis, observable_sector, ground_system)
phonon_response = neutral_lehmann_data(displacement_difference, observable_basis, observable_sector, ground_system)
cross_response = cross_lehmann_data(density_difference, displacement_difference, observable_basis, observable_sector, ground_system)

charge_lines = coalesce_inelastic_lines(charge_response.centers, charge_response.weights; weight_tol=1e-10)
phonon_lines = coalesce_inelastic_lines(phonon_response.centers, phonon_response.weights; weight_tol=1e-10)
cross_lines = coalesce_inelastic_lines(cross_response.centers, cross_response.weights; weight_tol=1e-10)

charge_matrix = charge_response.operator
phonon_matrix = phonon_response.operator

charge_equal_time = real(dot(psi0, charge_matrix * charge_matrix * psi0))
phonon_equal_time = real(dot(psi0, phonon_matrix * phonon_matrix * psi0))
cross_equal_time = real(dot(psi0, charge_matrix * phonon_matrix * psi0))

charge_zeroth_moment = sum(charge_response.weights)
phonon_zeroth_moment = sum(phonon_response.weights)
cross_zeroth_moment = sum(cross_response.weights)

@assert isapprox(charge_equal_time, charge_fluctuation; atol=1e-11, rtol=0.0)
@assert isapprox(cross_equal_time, electron_phonon_correlation; atol=1e-11, rtol=0.0)

@assert isapprox(charge_zeroth_moment, charge_equal_time; atol=1e-10, rtol=0.0)
@assert isapprox(phonon_zeroth_moment, phonon_equal_time; atol=1e-10, rtol=0.0)
@assert abs(imag(cross_zeroth_moment)) <= 1e-11
@assert isapprox(real(cross_zeroth_moment), cross_equal_time; atol=1e-10, rtol=0.0)

charge_elastic_weight = charge_response.weights[1]
phonon_elastic_weight = phonon_response.weights[1]

@assert charge_elastic_weight <= 1e-20
@assert phonon_elastic_weight <= 1e-20

charge_first_moment = sum(charge_response.centers .* charge_response.weights)
phonon_first_moment = sum(phonon_response.centers .* phonon_response.weights)

charge_first_moment_commutator = response_first_moment(Hobservable_sector, charge_response)
phonon_first_moment_commutator = response_first_moment(Hobservable_sector, phonon_response)

@assert isapprox(charge_first_moment, charge_first_moment_commutator; atol=1e-10, rtol=0.0)
@assert isapprox(phonon_first_moment, phonon_first_moment_commutator; atol=1e-10, rtol=0.0)

@assert isapprox(charge_first_moment, -2hopping_energy; atol=1e-10, rtol=0.0)
@assert isapprox(phonon_first_moment, 2omega; atol=1e-10, rtol=0.0)

strong_charge_lines = strongest_response_lines(charge_lines)
strong_phonon_lines = strongest_response_lines(phonon_lines)
strong_cross_lines = strongest_response_lines(cross_lines)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP7     |")
println(" --------------------------------")
println("free charge pole centers        : ", free_charge_lines.centers)
println("free charge pole weights        : ", free_charge_lines.weights)
println("free phonon pole centers        : ", free_phonon_lines.centers)
println("free phonon pole weights        : ", free_phonon_lines.weights)
@printf("charge zeroth moment            : %.12f\n", charge_zeroth_moment)
@printf("charge equal-time reference     : %.12f\n", charge_equal_time)
@printf("phonon zeroth moment            : %.12f\n", phonon_zeroth_moment)
@printf("phonon equal-time reference     : %.12f\n", phonon_equal_time)
@printf("cross zeroth moment             : %.12f\n", real(cross_zeroth_moment))
@printf("cross equal-time reference      : %.12f\n", cross_equal_time)
@printf("charge first moment             : %.12f\n", charge_first_moment)
@printf("charge commutator moment        : %.12f\n", charge_first_moment_commutator)
@printf("-2 <H_t>                        : %.12f\n", -2hopping_energy)
@printf("phonon first moment             : %.12f\n", phonon_first_moment)
@printf("phonon commutator moment        : %.12f\n", phonon_first_moment_commutator)
@printf("2 omega                         : %.12f\n", 2omega)
println("strong charge centers           : ", strong_charge_lines.centers)
println("strong charge weights           : ", strong_charge_lines.weights)
println("strong phonon centers           : ", strong_phonon_lines.centers)
println("strong phonon weights           : ", strong_phonon_lines.weights)
println("strong cross centers            : ", strong_cross_lines.centers)
println("strong cross weights            : ", strong_cross_lines.weights)
println("neutral-response closure        : true")



# =============================================================================
# Stage EP8 — Published reduced-Holstein-dimer reference
#
# Säkkinen et al., J. Chem. Phys. 143, 234101 (2015), Eq. (11).
#
# The paper uses
#
#     H_ep = -g sum_i n_i (a_i + a_i†),
#
# whereas Phundamental currently uses the opposite sign convention. The two
# conventions are unitarily related by a_i -> -a_i and therefore have identical
# energies, electronic observables, phonon occupations, and response weights.
# =============================================================================

reference_phonon_cutoff = 100
reference_dimension = reference_phonon_cutoff + 1

a_reference = zeros(Float64, reference_dimension, reference_dimension)
for n in 1:reference_phonon_cutoff
    a_reference[n, n + 1] = sqrt(n)
end

identity_reference = Matrix{Float64}(I, reference_dimension, reference_dimension)
zero_reference = zeros(Float64, reference_dimension, reference_dimension)
number_reference = Matrix(Diagonal(Float64.(0:reference_phonon_cutoff)))
u_reference = (a_reference + a_reference') / sqrt(2)

# Electronic singlet basis of Säkkinen et al.:
#
# |1> = singly occupied spin singlet,
# |2> = symmetric double occupancy,
# |3> = antisymmetric double occupancy.
#
# Their relative Hamiltonian is
#
#     [ omega a†a    -2T          0      ]
#     [ -2T          omega a†a   -2g u   ]
#     [ 0            -2g u        omega a†a ].

Hrel_reference = vcat(
    hcat(omega .* number_reference, (-2t) .* identity_reference, zero_reference),
    hcat((-2t) .* identity_reference, omega .* number_reference, (-2g) .* u_reference),
    hcat(zero_reference, (-2g) .* u_reference, omega .* number_reference),
)

reference_relative_system = eigen(Hermitian(Hrel_reference))
reference_relative_energies = Float64.(reference_relative_system.values)
reference_relative_ground = reference_relative_system.vectors[:, 1]

# The total phonon mode is an exactly displaced oscillator. For N_e = 2,
#
#     E_tot(n) = n * omega - g^2 N_e^2 / (2 omega).

reference_electron_number = 2
reference_total_shift = -g^2 * reference_electron_number^2 / (2omega)

reference_candidates = Float64[]
for relative_energy in reference_relative_energies[1:12], total_phonon_number in 0:8
    push!(reference_candidates, relative_energy + reference_total_shift + total_phonon_number * omega)
end
sort!(reference_candidates)

reference_low_spectrum = reference_candidates[1:4]
package_low_spectrum = Float64.(eigvals(Hermitian(Hobservable_sector), 1:4))
low_spectrum_reference_error = maximum(abs.(package_low_spectrum .- reference_low_spectrum))

@assert low_spectrum_reference_error <= 1e-10
@assert isapprox(reference_low_spectrum[1], E0; atol=1e-10, rtol=0.0)

# -----------------------------------------------------------------------------
# Published-reference ground-state observables
# -----------------------------------------------------------------------------

identity_electronic = Matrix{Float64}(I, 3, 3)
relative_number_operator = kron(identity_electronic, number_reference)

electron_reference = vcat(
    hcat(zero_reference, (-2t) .* identity_reference, zero_reference),
    hcat((-2t) .* identity_reference, zero_reference, zero_reference),
    hcat(zero_reference, zero_reference, zero_reference),
)

double_occupancy_total_reference = vcat(
    hcat(zero_reference, zero_reference, zero_reference),
    hcat(zero_reference, identity_reference, zero_reference),
    hcat(zero_reference, zero_reference, identity_reference),
)

relative_ep_reference = Hrel_reference - electron_reference - omega .* relative_number_operator

reference_electron_energy = real(dot(reference_relative_ground, electron_reference * reference_relative_ground))
reference_relative_phonon_energy = omega * real(dot(reference_relative_ground, relative_number_operator * reference_relative_ground))
reference_relative_ep_energy = real(dot(reference_relative_ground, relative_ep_reference * reference_relative_ground))

reference_uniform_phonon_energy = g^2 * reference_electron_number^2 / (2omega)
reference_uniform_ep_energy = -g^2 * reference_electron_number^2 / omega

reference_phonon_energy = reference_relative_phonon_energy + reference_uniform_phonon_energy
reference_ep_energy = reference_relative_ep_energy + reference_uniform_ep_energy
reference_total_energy = reference_electron_energy + reference_phonon_energy + reference_ep_energy

reference_double_occupancy = 0.5 * real(dot(reference_relative_ground, double_occupancy_total_reference * reference_relative_ground))

@assert isapprox(reference_total_energy, E0; atol=1e-10, rtol=0.0)
@assert isapprox(reference_electron_energy, hopping_energy; atol=1e-10, rtol=0.0)
@assert isapprox(reference_phonon_energy, phonon_energy; atol=1e-10, rtol=0.0)
@assert isapprox(reference_ep_energy, coupling_energy; atol=1e-10, rtol=0.0)
@assert isapprox(reference_double_occupancy, mean_double_occupancy; atol=1e-10, rtol=0.0)

# -----------------------------------------------------------------------------
# Natural occupation numbers
#
# Säkkinen et al. use the eigenvalues eta_1 and eta_2 of the spin-summed
# one-body reduced density matrix and relate them to the electronic energy.
# -----------------------------------------------------------------------------

gamma12_operator = c(3)' * c(1) + c(4)' * c(2)
gamma21_operator = c(1)' * c(3) + c(2)' * c(4)

gamma12 = ground_expectation(gamma12_operator, observable_basis, observable_sector, psi0)
gamma21 = ground_expectation(gamma21_operator, observable_basis, observable_sector, psi0)

one_body_density_matrix = Hermitian([
    n1 gamma12
    gamma21 n2
])

package_natural_occupations = sort(Float64.(eigvals(one_body_density_matrix)); rev=true)

reference_bond_order = -reference_electron_energy / (2t)
reference_natural_occupations = [1 + reference_bond_order, 1 - reference_bond_order]

@assert package_natural_occupations ≈ reference_natural_occupations atol=1e-10 rtol=0.0
@assert isapprox(sum(package_natural_occupations), 2.0; atol=1e-11, rtol=0.0)

gamma_parameter = omega / t
lambda_parameter = 2g^2 / (t * omega)

println()
println(" --------------------------------")
println("| Holstein Dimer — Stage EP8     |")
println(" --------------------------------")
@printf("adiabatic ratio gamma           : %.12f\n", gamma_parameter)
@printf("effective interaction lambda    : %.12f\n", lambda_parameter)
@printf("reference phonon cutoff         : %d\n", reference_phonon_cutoff)
println("package low spectrum            : ", package_low_spectrum)
println("published-form low spectrum     : ", reference_low_spectrum)
@printf("low-spectrum reference error    : %.3e\n", low_spectrum_reference_error)
@printf("package ground energy           : %.12f\n", E0)
@printf("published-form ground energy    : %.12f\n", reference_total_energy)
@printf("package electron energy         : %.12f\n", hopping_energy)
@printf("published-form electron energy  : %.12f\n", reference_electron_energy)
@printf("package phonon energy           : %.12f\n", phonon_energy)
@printf("published-form phonon energy    : %.12f\n", reference_phonon_energy)
@printf("package e-ph energy             : %.12f\n", coupling_energy)
@printf("published-form e-ph energy      : %.12f\n", reference_ep_energy)
@printf("package double occupancy        : %.12f\n", mean_double_occupancy)
@printf("published-form double occupancy : %.12f\n", reference_double_occupancy)
println("package natural occupations     : ", package_natural_occupations)
println("published-form occupations      : ", reference_natural_occupations)
println("published-reference closure     : true")
