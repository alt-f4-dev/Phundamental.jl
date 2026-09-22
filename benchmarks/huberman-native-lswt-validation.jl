# -------------------------------------------------------------------------
# Native Phundamental LSWT neutron-tensor validation.
#
# This compares the new SpinWaveResult-native S^{αβ}(Q,E) implementation
# against Huberman et al. Eqs. (4)-(5) on the identical finite periodic
# momentum grid. The comparison is deliberately finite-L so that any error is
# an implementation error rather than a thermodynamic-limit finite-size error.
# -------------------------------------------------------------------------

native_L = 12
native_L % 4 == 0 || error("native Huberman validation requires L divisible by 4")
native_points_per_segment = native_L ÷ 4
native_h, native_k, native_x = symmetry_path(native_points_per_segment)

native_cluster, native_model, native_signs = build_rb2mnf4_model(native_L)
native_result = solve(
    LinearSpinWaveSolver(
        reference_signs=native_signs,
        tol=1e-11,
    ),
    native_model,
)

native_positions = M.site_positions(native_cluster)
native_sites = collect(1:M.nsites(native_cluster))
native_field = SpinTensorField(
    xkeys=[Sx(i) for i in native_sites],
    ykeys=[Sy(i) for i in native_sites],
    zkeys=[Sz(i) for i in native_sites],
    positions=native_positions,
    normalization=:sqrtN,
)
native_q = [M.reciprocal_vector(crystal, [h, k]) for (h, k) in zip(native_h, native_k)]

native_energy_axis = collect(range(0.0, 13.5; length=541))
native_broadening = GaussianBroadening(σ_two)

native_one = nothing
native_two = nothing

GC.gc()
native_one_time = @elapsed begin
    native_one = spin_tensor_structure_factor(
        native_model,
        native_result,
        native_field,
        native_q,
        native_energy_axis;
        temperature=0.0,
        broadening=native_broadening,
        channels=(:one_magnon,),
        renormalize_transverse=true,
        spectral_accumulation=:direct,
        weight_tol=1e-14,
    )
end

GC.gc()
native_two_time = @elapsed begin
    native_two = spin_tensor_structure_factor(
        native_model,
        native_result,
        native_field,
        native_q,
        native_energy_axis;
        temperature=0.0,
        broadening=native_broadening,
        channels=(:two_magnon,),
        spectral_accumulation=:direct,
        weight_tol=1e-14,
    )
end

native_ΔS = discrete_spin_reduction(native_L)
reference_one = zeros(Float64, length(native_h), length(native_energy_axis), 3, 3)
reference_two = zeros(Float64, length(native_h), length(native_energy_axis))

for iq in eachindex(native_h)
    h = native_h[iq]
    k = native_k[iq]
    u, v = bogoliubov_uv(h, k)
    center = magnon_energy(h, k)
    weight = 0.5 * (S - native_ΔS) * (u + v)^2

    @inbounds for ie in eachindex(native_energy_axis)
        kernel = exp(-0.5 * ((native_energy_axis[ie] - center) / σ_two)^2) / (sqrt(2π) * σ_two)
        reference_one[iq, ie, 1, 1] = weight * kernel
        reference_one[iq, ie, 2, 2] = weight * kernel
    end
end

q1grid = collect(range(-0.5, 0.5; length=native_L + 1))[1:end-1]
for iq in eachindex(native_h)
    Qh = native_h[iq]
    Qk = native_k[iq]

    for h1 in q1grid, k1 in q1grid
        h2 = Qh - h1
        k2 = Qk - k1
        u1, v1 = bogoliubov_uv(h1, k1)
        u2, v2 = bogoliubov_uv(h2, k2)
        center = magnon_energy(h1, k1) + magnon_energy(h2, k2)
        weight = 0.5 * (u1 * v2 - u2 * v1)^2 / native_L^2

        @inbounds for ie in eachindex(native_energy_axis)
            kernel = exp(-0.5 * ((native_energy_axis[ie] - center) / σ_two)^2) / (sqrt(2π) * σ_two)
            reference_two[iq, ie] += weight * kernel
        end
    end
end

native_one_values = native_one.intensity
native_two_values = @view native_two.intensity[:, :, 3, 3]

one_error = norm(native_one_values - reference_one) / max(norm(reference_one), eps(Float64))
two_error = norm(native_two_values - reference_two) / max(norm(reference_two), eps(Float64))
xy_longitudinal_leakage = maximum(abs, native_two.intensity[:, :, 1:2, :])
transverse_z_leakage = maximum(abs, native_one.intensity[:, :, 3, :])

native_one_pass = one_error <= 1e-8 && transverse_z_leakage <= 1e-10
native_two_pass = two_error <= 1e-8 && xy_longitudinal_leakage <= 1e-10

println()
println("NATIVE LSWT NEUTRON-TENSOR VALIDATION")
@printf("Finite validation lattice          = %d × %d (%d spins)\n", native_L, native_L, native_L^2)
@printf("Native one-magnon tensor time      = %.6f s\n", native_one_time)
@printf("Native two-magnon tensor time      = %.6f s\n", native_two_time)
@printf("One-magnon tensor relative error   = %.3e\n", one_error)
@printf("Two-magnon Szz relative error      = %.3e\n", two_error)
@printf("One-magnon longitudinal leakage    = %.3e\n", transverse_z_leakage)
@printf("Two-magnon transverse leakage      = %.3e\n", xy_longitudinal_leakage)
println("  Native one-magnon Sαβ vs Eq. (4): ", native_one_pass ? "PASS" : "FAIL")
println("  Native two-magnon Szz vs Eq. (5): ", native_two_pass ? "PASS" : "FAIL")
