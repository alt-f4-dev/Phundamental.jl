using Phundamental
using LinearAlgebra
using Printf

# =============================================================================
# Reference
# =============================================================================
#
# E. Ghavanloo, S. A. Fazelzadeh, and H. Rafii-Tabar,
# "Formulation of an efficient continuum mechanics-based model to study wave
# propagation in one-dimensional diatomic lattices,"
# Mechanics Research Communications 103, 103467 (2020).
# DOI: 10.1016/j.mechrescom.2019.103467
#
# Discrete reference model: Eqs. (1), (2), and (6)-(8).
#
# The paper denotes the nearest-neighbor spacing by a. The primitive
# translation of the two-atom cell is therefore A = 2a.

# =============================================================================
# Stage P1 — Geometry, material, interactions, and real-space IFCs
# =============================================================================

A = 1.0
a = A / 2

M = 2.0
m = 1.0
k = 4.0

alpha = m / M
omega0 = sqrt(k / m)

lattice = BravaisLattice(reshape([A], 1, 1))

basis = BasisSite[BasisSite(:M, :M, [0.0]),
                  BasisSite(:m, :m, [0.5])]

crystal = LatticeCrystalStructure(lattice, basis)

species = Dict(:M => AtomicSpecies(:M; mass=M, nuclear_scattering_length=1.0),
               :m => AtomicSpecies(:m; mass=m, nuclear_scattering_length=1.0) )

material = Material("Ghavanloo diatomic chain", crystal; species=species)

interactions = HarmonicBondInteraction[HarmonicBondInteraction(1, 2, [0]; longitudinal=k),
                                       HarmonicBondInteraction(1, 2, [-1]; longitudinal=k)]

ifcs = force_constants(crystal, interactions)

model = harmonic_model(material, interactions)

function scalar_ifc(ifcs, from_site::Int, to_site::Int, translation::Int)
    matches = [block for block in ifcs.blocks if block.from_site == from_site && block.to_site == to_site && block.translation == [translation]]

    length(matches) == 1 || throw(ArgumentError("expected one IFC block for ($from_site, $to_site, $translation), found $(length(matches))"))

    block = only(matches)
    size(block.tensor) == (1, 1) || throw(DimensionMismatch("one-dimensional benchmark requires 1×1 IFC blocks"))

    return block.tensor[1, 1]
end

expected_ifcs = Dict((1, 1, 0)  => 2k,
                     (2, 2, 0)  => 2k,
                     (1, 2, 0)  => -k,
                     (1, 2, -1) => -k,
                     (2, 1, 0)  => -k,
                     (2, 1, 1)  => -k,)

@assert length(ifcs.blocks) == length(expected_ifcs)

for ((i, j, R), reference) in expected_ifcs
    @assert isapprox(scalar_ifc(ifcs, i, j, R), reference; atol=1e-12, rtol=0.0)
end

site1_sum = sum(block.tensor[1, 1] for block in ifcs.blocks if block.from_site == 1)

site2_sum = sum(block.tensor[1, 1] for block in ifcs.blocks if block.from_site == 2)

@assert isapprox(site1_sum, 0.0; atol=1e-12, rtol=0.0)
@assert isapprox(site2_sum, 0.0; atol=1e-12, rtol=0.0)

asr_residual = acoustic_sum_rule_residual(model)

@assert asr_residual <= 1e-12

model_ifcs = model.parameters[:periodic_force_constants]

@assert model_ifcs.lattice_dimension == 1
@assert model_ifcs.displacement_dimension == 1
@assert model.parameters[:phonon_basis_count] == 2
@assert model.parameters[:displacement_dimension] == 1
@assert isapprox(model.parameters[:masses], [M, m]; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P1     |")
println(" --------------------------------")
@printf("primitive-cell length A    : %.12f\n", A)
@printf("nearest-neighbor spacing a : %.12f\n", a)
@printf("heavy mass M               : %.12f\n", M)
@printf("light mass m               : %.12f\n", m)
@printf("spring constant k          : %.12f\n", k)
@printf("mass ratio alpha = m/M     : %.12f\n", alpha)
@printf("reference omega0           : %.12f\n", omega0)
println("basis fractional positions : ", [site.fractional for site in crystal.basis])
println("compiled IFC block count   : ", length(ifcs.blocks))
@printf("site-1 IFC sum             : %.3e\n", site1_sum)
@printf("site-2 IFC sum             : %.3e\n", site2_sum)
@printf("acoustic-sum-rule residual : %.3e\n", asr_residual)
println("real-space IFC closure     : true")


# =============================================================================
# Stage P2 — Bloch dynamical matrix
# =============================================================================

function reference_dynamical_matrix(h::Real)
    kappa = 2π * h / A
    coupling = -2k * cos(kappa * a) / sqrt(M * m)

    return [2k / M coupling
            coupling 2k / m]
end

reference_trace = 2k * (1 / M + 1 / m)

sample_h = [0.0, 1 / 8, 1 / 4, 3 / 8, 1 / 2]

matrix_errors = Float64[]
hermiticity_errors = Float64[]
trace_errors = Float64[]
determinant_errors = Float64[]

for h in sample_h
    kappa = 2π * h / A

    D_numeric = dynamical_matrix(model, [kappa])
    D_reference = ComplexF64.(reference_dynamical_matrix(h))

    push!(matrix_errors, maximum(abs.(D_numeric .- D_reference)))

    push!(hermiticity_errors, norm(D_numeric - adjoint(D_numeric)) / max(norm(D_numeric), 1.0))

    push!(trace_errors, abs(real(tr(D_numeric)) - reference_trace))

    reference_determinant = 4k^2 * sin(kappa * a)^2 / (M * m)

    push!(determinant_errors, abs(det(D_numeric) - reference_determinant))
end

matrix_error = maximum(matrix_errors)
hermiticity_error = maximum(hermiticity_errors)
trace_error = maximum(trace_errors)
determinant_error = maximum(determinant_errors)

@assert matrix_error <= 1e-12
@assert hermiticity_error <= 1e-12
@assert trace_error <= 1e-12
@assert determinant_error <= 1e-12

D_gamma = dynamical_matrix(model, [0.0])
D_boundary = dynamical_matrix(model, [π / A])

reference_gamma = ComplexF64[2k / M -2k / sqrt(M * m)
                             -2k / sqrt(M * m) 2k / m]

reference_boundary = ComplexF64[2k / M 0.0
                                0.0 2k / m]

@assert isapprox(D_gamma, reference_gamma; atol=1e-12, rtol=0.0)
@assert isapprox(D_boundary, reference_boundary; atol=1e-12, rtol=0.0)

gamma_eigenvalues = eigvals(Hermitian(D_gamma))
boundary_eigenvalues = eigvals(Hermitian(D_boundary))

@assert isapprox(gamma_eigenvalues, [0.0, 2k * (1 / M + 1 / m)]; atol=1e-12, rtol=0.0)
@assert isapprox(boundary_eigenvalues, sort([2k / M, 2k / m]); atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P2     |")
println(" --------------------------------")
println("sample reciprocal h        : ", sample_h)
@printf("max |ΔD(h)|                : %.3e\n", matrix_error)
@printf("max Hermiticity error      : %.3e\n", hermiticity_error)
@printf("max trace error            : %.3e\n", trace_error)
@printf("max determinant error      : %.3e\n", determinant_error)
println("Gamma dynamical matrix     :")
display(real.(D_gamma))
println("zone-boundary matrix       :")
display(real.(D_boundary))
println("Gamma omega^2              : ", round.(gamma_eigenvalues; digits=12))
println("boundary omega^2           : ", round.(boundary_eigenvalues; digits=12))
println("Bloch dynamical closure    : true")


# =============================================================================
# Stage P3 — Exact acoustic and optical phonon dispersion
# =============================================================================

function reference_phonon_frequencies(h::Real)
    kappa = 2π * h / A
    discriminant = sqrt((M + m)^2 - 4M * m * sin(kappa * a)^2)

    omega2_acoustic = k * ((M + m) - discriminant) / (M * m)
    omega2_optical = k * ((M + m) + discriminant) / (M * m)

    omega2_acoustic = max(omega2_acoustic, 0.0)
    omega2_optical = max(omega2_optical, 0.0)

    return sqrt(omega2_acoustic), sqrt(omega2_optical)
end

solver = HarmonicPhononSolver()

h_path = collect(range(0.0, 0.5; length=201))

q_path = [ReciprocalWaveVector([h]) for h in h_path]

dispersion = phonon_dispersion(solver, model, q_path)

reference_frequencies = Matrix{Float64}(undef, length(h_path), 2)

for (iq, h) in enumerate(h_path)
    reference_frequencies[iq, :] .= reference_phonon_frequencies(h)
end

frequency_error = maximum(abs.(dispersion.frequencies .- reference_frequencies))

@assert size(dispersion.frequencies) == (length(h_path), 2)
@assert frequency_error <= 1e-12

gamma_frequencies = vec(dispersion.frequencies[1, :])
boundary_frequencies = vec(dispersion.frequencies[end, :])

reference_gamma_frequencies = [0.0, sqrt(2k * (1 / M + 1 / m))]

reference_boundary_frequencies = sort([sqrt(2k / M), sqrt(2k / m)])

@assert isapprox(gamma_frequencies, reference_gamma_frequencies; atol=1e-12, rtol=0.0)

@assert isapprox(boundary_frequencies, reference_boundary_frequencies; atol=1e-12, rtol=0.0)

quarter_index = findfirst(h -> isapprox(h, 0.25; atol=1e-14, rtol=0.0), h_path)

isnothing(quarter_index) && error("h = 0.25 is absent from the Stage P3 q path")

quarter_frequencies = vec(dispersion.frequencies[quarter_index, :])

reference_quarter_frequencies = collect(reference_phonon_frequencies(0.25))

@assert isapprox(quarter_frequencies, reference_quarter_frequencies; atol=1e-12, rtol=0.0)

zero_mode_mask = dispersion.metadata[:zero_mode_mask]

@assert zero_mode_mask[1, 1]
@assert !zero_mode_mask[1, 2]
@assert count(zero_mode_mask) == 1

reciprocal_cartesian_errors = Float64[]

for h in sample_h
    reciprocal_result = solve(solver, model, ReciprocalWaveVector([h]))

    cartesian_result = solve(solver, model, CartesianWaveVector([2π * h / A]))

    push!(reciprocal_cartesian_errors, maximum(abs.(reciprocal_result.frequencies .- cartesian_result.frequencies)))
end

reciprocal_cartesian_error = maximum(reciprocal_cartesian_errors)

@assert reciprocal_cartesian_error <= 1e-12

acoustic_frequencies = dispersion.frequencies[:, 1]
optical_frequencies = dispersion.frequencies[:, 2]

@assert all(acoustic_frequencies .>= 0.0)
@assert all(optical_frequencies .>= acoustic_frequencies)

@assert issorted(acoustic_frequencies)
@assert issorted(reverse(optical_frequencies))

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P3     |")
println(" --------------------------------")
println("q-point count              : ", length(h_path))
println("branch count               : ", size(dispersion.frequencies, 2))
@printf("max |Δω(h)|                : %.3e\n", frequency_error)
println("Gamma frequencies          : ", round.(gamma_frequencies; digits=12))
println("h = 0.25 frequencies       : ", round.(quarter_frequencies; digits=12))
println("boundary frequencies       : ", round.(boundary_frequencies; digits=12))
@printf("reciprocal/Cartesian error : %.3e\n", reciprocal_cartesian_error)
println("zero-mode count            : ", count(zero_mode_mask))
println("acoustic monotonicity      : true")
println("optical monotonicity       : true")
println("analytic dispersion closure: true")


# =============================================================================
# Stage P4 — Polarization vectors and physical displacements
# =============================================================================

function normalized_overlap(v, reference)
    return abs(dot(reference, v)) / (norm(reference) * norm(v))
end

mass_matrix = Diagonal([M, m])
inverse_sqrt_mass = Diagonal(1 ./ sqrt.([M, m]))

gamma_result = solve(solver, model, ReciprocalWaveVector([0.0]))

boundary_result = solve(solver, model, ReciprocalWaveVector([0.5]))

gamma_modes = gamma_result.modes
boundary_modes = boundary_result.modes

gamma_physical_modes = gamma_result.metadata[:physical_displacement_modes]
boundary_physical_modes = boundary_result.metadata[:physical_displacement_modes]

reference_gamma_acoustic = [sqrt(M), sqrt(m)] ./ sqrt(M + m)

reference_gamma_optical = [sqrt(m), -sqrt(M)] ./ sqrt(M + m)

gamma_acoustic_overlap = normalized_overlap(gamma_modes[:, 1], reference_gamma_acoustic)

gamma_optical_overlap = normalized_overlap(gamma_modes[:, 2], reference_gamma_optical)

@assert isapprox(gamma_acoustic_overlap, 1.0; atol=1e-12, rtol=0.0)

@assert isapprox(gamma_optical_overlap, 1.0; atol=1e-12, rtol=0.0)

mass_weighted_orthogonality_error = maximum([norm(adjoint(dispersion.modes[:, :, iq]) * dispersion.modes[:, :, iq] - I) for iq in eachindex(h_path)])

physical_metric_errors = Float64[]
eigenvector_residuals = Float64[]

for (iq, q) in enumerate(q_path)
    modes_q = dispersion.modes[:, :, iq]
    physical_modes_q = inverse_sqrt_mass * modes_q
    Dq = dynamical_matrix(model, q)

    push!(physical_metric_errors, norm(adjoint(physical_modes_q) * mass_matrix * physical_modes_q - I))

    for branch in 1:2
        omega2 = dispersion.frequencies[iq, branch]^2
        mode = modes_q[:, branch]

        push!(eigenvector_residuals, norm(Dq * mode - omega2 * mode))
    end
end

physical_metric_error = maximum(physical_metric_errors)
eigenvector_residual = maximum(eigenvector_residuals)

@assert mass_weighted_orthogonality_error <= 1e-12
@assert physical_metric_error <= 1e-12
@assert eigenvector_residual <= 1e-12

gamma_acoustic_relative_displacement = abs(gamma_physical_modes[1, 1] - gamma_physical_modes[2, 1])

gamma_optical_center_of_mass = abs(M * gamma_physical_modes[1, 2] + m * gamma_physical_modes[2, 2])

gamma_optical_ratio = gamma_physical_modes[2, 2] / gamma_physical_modes[1, 2]

@assert gamma_acoustic_relative_displacement <= 1e-12
@assert gamma_optical_center_of_mass <= 1e-12

@assert isapprox(gamma_optical_ratio, -M / m; atol=1e-12, rtol=0.0)

boundary_acoustic_heavy_weight = abs2(boundary_modes[1, 1])

boundary_acoustic_light_weight = abs2(boundary_modes[2, 1])

boundary_optical_heavy_weight = abs2(boundary_modes[1, 2])

boundary_optical_light_weight = abs2(boundary_modes[2, 2])

@assert isapprox(boundary_acoustic_heavy_weight, 1.0; atol=1e-12, rtol=0.0)

@assert boundary_acoustic_light_weight <= 1e-12
@assert boundary_optical_heavy_weight <= 1e-12

@assert isapprox(boundary_optical_light_weight, 1.0; atol=1e-12, rtol=0.0)

@assert isapprox(gamma_physical_modes, inverse_sqrt_mass * gamma_modes; atol=1e-12, rtol=0.0)

@assert isapprox(boundary_physical_modes, inverse_sqrt_mass * boundary_modes; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P4     |")
println(" --------------------------------")
@printf("Gamma acoustic overlap     : %.12f\n", gamma_acoustic_overlap)
@printf("Gamma optical overlap      : %.12f\n", gamma_optical_overlap)
@printf("max e†e - I error          : %.3e\n", mass_weighted_orthogonality_error)
@printf("max u†Mu - I error         : %.3e\n", physical_metric_error)
@printf("max eigenvector residual   : %.3e\n", eigenvector_residual)
@printf("Gamma acoustic Δu          : %.3e\n", gamma_acoustic_relative_displacement)
@printf("Gamma optical COM residual : %.3e\n", gamma_optical_center_of_mass)
println("Gamma optical u_m/u_M      : ", round(gamma_optical_ratio; digits=12))
@printf("boundary acoustic M weight : %.12f\n", boundary_acoustic_heavy_weight)
@printf("boundary acoustic m weight : %.3e\n", boundary_acoustic_light_weight)
@printf("boundary optical M weight  : %.3e\n", boundary_optical_heavy_weight)
@printf("boundary optical m weight  : %.12f\n", boundary_optical_light_weight)
println("polarization closure       : true")



# =============================================================================
# Stage P5 — Group velocities and acoustic sound velocity
# =============================================================================

function reference_group_velocities(h::Real)
    kappa = 2π * h / A
    theta = kappa * a
    S = M + m
    discriminant = sqrt(S^2 - 4M * m * sin(theta)^2)

    # Stable acoustic expression avoids cancellation as h -> 0.
    omega_acoustic = 2sqrt(k) * abs(sin(theta)) / sqrt(S + discriminant)
    omega_optical = sqrt(k * (S + discriminant) / (M * m))

    prefactor = k * a * sin(2theta) / discriminant

    return (prefactor / omega_acoustic, -prefactor / omega_optical)
end

velocity_h = collect(range(0.025, 0.475; length=19))

direct_velocity_errors = Float64[]

for h in velocity_h
    result = solve(solver, model, ReciprocalWaveVector([h]))

    numerical = vec(group_velocities(result))
    reference = collect(reference_group_velocities(h))

    push!(direct_velocity_errors, maximum(abs.(numerical .- reference)))
end

direct_velocity_error = maximum(direct_velocity_errors)

@assert direct_velocity_error <= 1e-11

velocity_q_path = [ReciprocalWaveVector([h]) for h in velocity_h]

velocity_dispersion = phonon_dispersion(solver, model, velocity_q_path; derivatives=true)

stored_velocities = velocity_dispersion.metadata[:group_velocities][:, :, 1]

reference_velocities = Matrix{Float64}(undef, length(velocity_h), 2)

for (iq, h) in enumerate(velocity_h)
    reference_velocities[iq, :] .= reference_group_velocities(h)
end

stored_velocity_error = maximum(abs.(stored_velocities .- reference_velocities))

@assert stored_velocity_error <= 1e-11

@assert all(stored_velocities[:, 1] .> 0.0)
@assert all(stored_velocities[:, 2] .< 0.0)

gamma_velocities = group_velocities(gamma_result)
boundary_velocities = group_velocities(boundary_result)

@assert isnan(gamma_velocities[1, 1])
@assert abs(gamma_velocities[2, 1]) <= 1e-12
@assert maximum(abs.(boundary_velocities)) <= 1e-12

sound_velocity_reference = a * sqrt(2k / (M + m))

h_sound = 1e-4

sound_result = solve(solver, model, ReciprocalWaveVector([h_sound]))

sound_velocity_numeric = group_velocities(sound_result)[1, 1]

sound_velocity_error = abs(sound_velocity_numeric - sound_velocity_reference)

@assert sound_velocity_error <= 1e-6

quarter_result = solve(solver, model, ReciprocalWaveVector([0.25]))

quarter_velocities = vec(group_velocities(quarter_result))

quarter_reference_velocities = collect(reference_group_velocities(0.25))

@assert isapprox(quarter_velocities, quarter_reference_velocities; atol=1e-12, rtol=0.0)

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P5     |")
println(" --------------------------------")
println("velocity q-point count      : ", length(velocity_h))
@printf("max direct velocity error  : %.3e\n", direct_velocity_error)
@printf("max stored velocity error  : %.3e\n", stored_velocity_error)
println("h = 0.25 velocities        : ", round.(quarter_velocities; digits=12))
println("Gamma acoustic velocity    : ", gamma_velocities[1, 1])
@printf("Gamma optical velocity     : %.3e\n", gamma_velocities[2, 1])
println("boundary velocities        : ", round.(vec(boundary_velocities); digits=12))
@printf("reference sound velocity   : %.12f\n", sound_velocity_reference)
@printf("numeric sound velocity     : %.12f\n", sound_velocity_numeric)
@printf("sound-velocity error       : %.3e\n", sound_velocity_error)
println("group-velocity closure     : true")



# =============================================================================
# Stage P6 — Brillouin-zone integration and zero-point energy
# =============================================================================

function reference_zero_point_energy(nq::Integer)
    nq > 0 || throw(ArgumentError("nq must be positive"))

    total = 0.0

    for index in 0:(nq - 1)
        h = index / nq
        omega_acoustic, omega_optical = reference_phonon_frequencies(h)
        total += 0.5 * (omega_acoustic + omega_optical)
    end

    return total / nq
end

mesh_sizes = [32, 64, 128, 256, 512, 1024]

package_zero_point = Float64[]
reference_zero_point = Float64[]
mesh_errors = Float64[]

for nq in mesh_sizes
    package_value = phonon_zero_point_energy(model; qmesh=(nq,), solver=solver, hbar=1.0)

    reference_value = reference_zero_point_energy(nq)

    push!(package_zero_point, package_value)
    push!(reference_zero_point, reference_value)
    push!(mesh_errors, abs(package_value - reference_value))
end

maximum_mesh_error = maximum(mesh_errors)

@assert maximum_mesh_error <= 1e-11

for index in eachindex(mesh_sizes)
    @assert isapprox(package_zero_point[index], reference_zero_point[index]; atol=1e-11, rtol=0.0)
end

nq_check = 256

zero_point_256 = phonon_zero_point_energy(model; qmesh=(nq_check,), solver=solver, hbar=1.0)

ground_energy_256 = groundenergy(model; qmesh=(nq_check,))

@assert isapprox(zero_point_256, ground_energy_256; atol=1e-12, rtol=0.0)

dense_reference_nq = 262_144

dense_reference = reference_zero_point_energy(dense_reference_nq)

convergence_errors = abs.(reference_zero_point .- dense_reference)

@assert all(convergence_errors[index + 1] < convergence_errors[index] for index in 1:(length(convergence_errors) - 1))

zero_point_256_reference = reference_zero_point_energy(nq_check)

@assert isapprox(zero_point_256, zero_point_256_reference; atol=1e-11, rtol=0.0)

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P6     |")
println(" --------------------------------")
println("mesh sizes                  : ", mesh_sizes)
@printf("max package/reference error : %.3e\n", maximum_mesh_error)
@printf("E_ZP, Nq = 256              : %.12f\n", zero_point_256)
@printf("groundenergy, Nq = 256      : %.12f\n", ground_energy_256)
@printf("dense analytic reference    : %.12f\n", dense_reference)
@printf("Nq = 256 integration error  : %.3e\n", abs(zero_point_256 - dense_reference))
println("monotonic mesh convergence  : true")
println("zero-point energy closure   : true")


# =============================================================================
# Stage P7 — Intrinsic coherent one-phonon neutron scattering
# =============================================================================

probe = NuclearNeutronProbe(prefactor=1.0)

scattering_convention = SpectrumConvention(spectral_axis=:omega, hbar=1.0, kB=1.0)

b_heavy = ComplexF64(material.species[:M].nuclear_scattering_length)

b_light = ComplexF64(material.species[:m].nuclear_scattering_length)

function reference_reciprocal_decomposition(H::Real)
    G = round(Int, H)
    q = Float64(H - G)

    abs(q) < 0.5 || error("reference reduction requires a non-boundary point inside the first Brillouin zone")

    return q, G, -q
end

function reference_one_phonon_lines(H::Real)
    q, G, qprime = reference_reciprocal_decomposition(H)

    eigensystem = eigen(Hermitian(reference_dynamical_matrix(qprime)))

    omega = sqrt.(max.(Float64.(eigensystem.values),0.0))

    physical_modes = inverse_sqrt_mass * eigensystem.vectors

    Qcart = 2π * H / A
    Gcart = 2π * G / A

    phase_heavy = 1.0 + 0.0im
    phase_light = cis(Gcart * a)

    weights = Vector{Float64}(undef,length(omega))

    for branch in eachindex(omega)
        amplitude = b_heavy * phase_heavy * Qcart * physical_modes[1, branch] + b_light * phase_light * Qcart * physical_modes[2, branch]

        oscillator = sqrt(1 / (2omega[branch]))

        weights[branch] = abs2(oscillator * amplitude)
    end

    return (centers=omega, weights=weights, q=q, G=G, qprime=qprime)
end

full_Q_values = [0.25, 0.75, 1.25, 1.75]

package_weights = Matrix{Float64}(undef, length(full_Q_values), 2)

reference_weights = similar(package_weights)

line_center_errors = Float64[]
line_weight_errors = Float64[]
imaginary_weight_errors = Float64[]

for (index, H) in enumerate(full_Q_values)
    Q = ReciprocalWaveVector([H])

    lines = one_phonon_neutron_lines(solver, model, probe, Q; temperature=0.0, convention=scattering_convention)

    reference = reference_one_phonon_lines(H)

    @assert length(lines.centers) == 2
    @assert length(lines.weights) == 2
    @assert all(lines.centers .> 0.0)

    push!(line_center_errors, maximum(abs.(Float64.(lines.centers) .- reference.centers)))

    push!(line_weight_errors,maximum(abs.(real.(lines.weights) .- reference.weights)))

    push!(imaginary_weight_errors,maximum(abs.(imag.(lines.weights))))

    package_weights[index, :] .= real.(lines.weights)

    reference_weights[index, :] .= reference.weights

    @assert isapprox(only(lines.metadata[:full_Q_reciprocal]),H;atol=1e-12,rtol=0.0)

    @assert isapprox(only(lines.metadata[:reduced_q_reciprocal]),reference.q;atol=1e-12,rtol=0.0)

    @assert only(lines.metadata[:reciprocal_G_indices]) == reference.G

    @assert isapprox(only(lines.metadata[:eigenvector_qprime_reciprocal]),reference.qprime;atol=1e-12,rtol=0.0)

    @assert lines.metadata[:momentum_policy] === :full_Q_to_q_plus_G_with_qprime_equals_negative_q
end

line_center_error = maximum(line_center_errors)

line_weight_error = maximum(line_weight_errors)

imaginary_weight_error = maximum(imaginary_weight_errors)

@assert line_center_error <= 1e-12
@assert line_weight_error <= 1e-11
@assert imaginary_weight_error <= 1e-12

Qcart_squared = [(2π * H / A)^2 for H in full_Q_values]

normalized_weights = package_weights ./ Qcart_squared

@assert isapprox(normalized_weights[1, :], normalized_weights[4, :]; atol=1e-12, rtol=0.0)

@assert isapprox(normalized_weights[2, :], normalized_weights[3, :]; atol=1e-12, rtol=0.0)

basis_phase_contrast = maximum(abs.(normalized_weights[1, :] .- normalized_weights[2, :]))

@assert basis_phase_contrast > 0.1

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P7     |")
println(" --------------------------------")
@printf("max line-center error       : %.3e\n", line_center_error)
@printf("max line-weight error       : %.3e\n", line_weight_error)
@printf("max imaginary weight        : %.3e\n", imaginary_weight_error)

for (index, H) in enumerate(full_Q_values)
    reference = reference_one_phonon_lines(H)

    @printf(
        "Q = %.2f  q = %+.2f  G = %d  weights = [%.12f, %.12f]\n",
        H,
        reference.q,
        reference.G,
        package_weights[index, 1],
        package_weights[index, 2],
    )
end

println("Q²-normalized even-G weights: ", round.(normalized_weights[1, :]; digits=12))
println("Q²-normalized odd-G weights : ", round.(normalized_weights[2, :]; digits=12))
@printf("basis-phase contrast        : %.3e\n", basis_phase_contrast)
println("one-phonon line closure     : true")



# =============================================================================
# Stage P8 — Finite temperature, detailed balance, and spectral broadening
# =============================================================================

temperature = 1.0
Qthermal = ReciprocalWaveVector([0.25])

zero_temperature_lines = one_phonon_neutron_lines(solver, model, probe, Qthermal; temperature=0.0, convention=scattering_convention)
finite_temperature_lines = one_phonon_neutron_lines(solver, model, probe, Qthermal; temperature=temperature, convention=scattering_convention)

@assert length(zero_temperature_lines.centers) == 2
@assert length(finite_temperature_lines.centers) == 4
@assert all(zero_temperature_lines.centers .> 0.0)
@assert count(>(0.0), finite_temperature_lines.centers) == 2
@assert count(<(0.0), finite_temperature_lines.centers) == 2

function line_weight_at(lines, center::Real; atol::Real=1e-12)::Float64
    matches = findall(value -> isapprox(value, center; atol=atol, rtol=0.0), lines.centers)
    length(matches) == 1 || throw(ArgumentError("expected exactly one spectral line at center $center"))
    return Float64(real(lines.weights[only(matches)]))
end

thermal_weight_errors = Float64[]
detailed_balance_errors = Float64[]
occupations = Float64[]

for branch in eachindex(zero_temperature_lines.centers)
    omega = Float64(zero_temperature_lines.centers[branch])
    bare_weight = Float64(real(zero_temperature_lines.weights[branch]))
    occupation = inv(expm1(scattering_convention.hbar * omega / (scattering_convention.kB * temperature)))
    positive_weight = line_weight_at(finite_temperature_lines, omega)
    negative_weight = line_weight_at(finite_temperature_lines, -omega)
    expected_positive = (occupation + 1) * bare_weight
    expected_negative = occupation * bare_weight
    expected_ratio = exp(-scattering_convention.hbar * omega / (scattering_convention.kB * temperature))

    push!(occupations, occupation)
    push!(thermal_weight_errors, max(abs(positive_weight - expected_positive), abs(negative_weight - expected_negative)))
    push!(detailed_balance_errors, abs(negative_weight / positive_weight - expected_ratio))
end

thermal_weight_error = maximum(thermal_weight_errors)
detailed_balance_error = maximum(detailed_balance_errors)

@assert thermal_weight_error <= 1e-12
@assert detailed_balance_error <= 1e-12

broadening = GaussianBroadening(0.03)
axis = collect(range(-4.5, 4.5; length=18_001))

spectrum = one_phonon_neutron_intensity(solver, model, probe, Qthermal, axis;
                                        broadening=broadening, temperature=temperature, convention=scattering_convention)

manual_broadened = Phundamental.Observables.broaden_lines(finite_temperature_lines, axis, broadening)
broadening_error = maximum(abs.(vec(spectrum.intensity) .- real.(manual_broadened)))

@assert broadening_error <= 1e-12
@assert minimum(spectrum.intensity) >= -1e-14
@assert spectrum.metadata[:broadening] === typeof(broadening)
@assert !spectrum.metadata[:resolution_applied]

function trapezoid_integral(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Float64
    length(x) == length(y) || throw(DimensionMismatch("integration vectors must have equal length"))
    return sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
end

integrated_weight = trapezoid_integral(axis, vec(spectrum.intensity))
intrinsic_weight = sum(real, finite_temperature_lines.weights)
integrated_weight_error = abs(integrated_weight - intrinsic_weight)

@assert integrated_weight_error <= 1e-10

println()
println(" --------------------------------")
println("| Diatomic Phonon — Stage P8     |")
println(" --------------------------------")
@printf("temperature                  : %.6f\n", temperature)
println("Bose occupations             : ", round.(occupations; digits=12))
@printf("max thermal-weight error     : %.3e\n", thermal_weight_error)
@printf("max detailed-balance error   : %.3e\n", detailed_balance_error)
@printf("max broadening error         : %.3e\n", broadening_error)
@printf("intrinsic total weight       : %.12f\n", intrinsic_weight)
@printf("broadened integrated weight  : %.12f\n", integrated_weight)
@printf("integrated-weight error      : %.3e\n", integrated_weight_error)
println("finite-temperature closure   : true")
println("detailed-balance closure     : true")
println("spectral-broadening closure  : true")
