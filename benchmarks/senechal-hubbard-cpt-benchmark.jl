using SparseArrays
using Statistics

# Senechal, Perez, and Pioro-Ladriere, Phys. Rev. Lett. 84, 522 (2000), DOI: 10.1103/PhysRevLett.84.522.
# This benchmark validates spin-resolved sector-changing Green functions and the one-dimensional CPT reconstruction of Eqs. (6), (7), and (9).

function _senechal_chain_model(nsites::Int, U::Real; t::Real=1.0, periodic::Bool=false)
    crystal = M.CrystalStructure(M.BravaisLattice(reshape([1.0], 1, 1)), [M.BasisSite(:A, :X, [0.0])])
    material = Material("Senechal 1D Hubbard benchmark", crystal; species=Dict(:X => M.AtomicSpecies(:X; mass=1.0)))
    cluster = Supercell(crystal, [nsites]; periodic=periodic)
    spec = HubbardModel(t, U; chemical_potential=U / 2, neighbor_cutoff=1.01)
    return build_model(material, cluster, spec)
end

function _senechal_trapezoid(axis::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    length(axis) == length(values) || throw(DimensionMismatch("integration axis and values must have the same length"))
    total = 0.0
    for index in 1:(length(axis) - 1)
        total += 0.5 * (values[index] + values[index + 1]) * (axis[index + 1] - axis[index])
    end
    return total
end

function _senechal_exact_cluster_green(workspace::FermionGreenWorkspace, z::ComplexF64)
    ground_eigensystem = eigen(Hermitian(Matrix(workspace.ground_hamiltonian)))
    ground_energy = Float64(ground_eigensystem.values[1])
    ground_state = ComplexF64.(ground_eigensystem.vectors[:, 1])
    particle_eigensystem = eigen(Hermitian(Matrix(workspace.particle_hamiltonian)))
    hole_eigensystem = eigen(Hermitian(Matrix(workspace.hole_hamiltonian)))

    particle_seeds = hcat([
        fermion_transition_vector(
            workspace.model, ground_state, workspace.ground_basis, workspace.particle_basis, site; spin=workspace.spin, create=true,
        ) for site in workspace.sites
    ]...)
    hole_seeds = hcat([
        fermion_transition_vector(
            workspace.model, ground_state, workspace.ground_basis, workspace.hole_basis, site; spin=workspace.spin, create=false,
        ) for site in workspace.sites
    ]...)

    particle_denominator = 1.0 ./ (z .+ ground_energy .- particle_eigensystem.values)
    hole_denominator = 1.0 ./ (z .- ground_energy .+ hole_eigensystem.values)
    particle_vectors = adjoint(particle_eigensystem.vectors) * particle_seeds
    hole_vectors = adjoint(hole_eigensystem.vectors) * hole_seeds
    particle_green = adjoint(particle_vectors) * Diagonal(particle_denominator) * particle_vectors
    hole_overlap = adjoint(hole_vectors) * Diagonal(hole_denominator) * hole_vectors
    return particle_green + transpose(hole_overlap), ground_energy, ground_state
end

function _senechal_particle_hole_residual(spectrum)
    reference = reverse(reverse(spectrum.intensity; dims=1); dims=2)
    return norm(spectrum.intensity - reference) / max(norm(spectrum.intensity), eps(Float64))
end

function _senechal_parse_int_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Int, strip.(split(get(ENV, name, default), ',')))))
    isempty(values) && error("$name must contain at least one integer")
    return values
end

function _senechal_parse_float_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Float64, strip.(split(get(ENV, name, default), ',')))))
    isempty(values) && error("$name must contain at least one number")
    return values
end

function _senechal_log_fit(x, y)
    xv = Float64.(collect(x))
    yv = Float64.(collect(y))
    valid = findall(index -> isfinite(xv[index]) && isfinite(yv[index]) && xv[index] > 0 && yv[index] > 0, eachindex(xv))
    length(valid) >= 2 || return (exponent=NaN, intercept=NaN, r2=NaN, n=length(valid))
    lx = log.(xv[valid])
    ly = log.(yv[valid])
    design = hcat(ones(length(lx)), lx)
    coefficients = design \ ly
    predicted = design * coefficients
    denominator = sum(abs2, ly .- mean(ly))
    r2 = denominator > 0 ? 1 - sum(abs2, ly .- predicted) / denominator : 1.0
    return (exponent=coefficients[2], intercept=coefficients[1], r2=r2, n=length(valid))
end

function _senechal_map_residual(test, reference)
    size(test) == size(reference) || throw(DimensionMismatch("spectral maps must have equal dimensions"))
    return norm(test - reference) / max(norm(reference), eps(Float64))
end

function _senechal_smoothed(values::AbstractVector{<:Real}, radius::Int)
    radius <= 0 && return Float64.(values)
    output = zeros(Float64, length(values))
    prefix = cumsum(vcat(0.0, Float64.(values)))
    for index in eachindex(values)
        lo = max(firstindex(values), index - radius)
        hi = min(lastindex(values), index + radius)
        output[index] = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1)
    end
    return output
end

function _senechal_separated_peaks(axis, values; relative_threshold::Float64=0.08, min_separation::Float64=0.25, max_peaks::Int=4)
    isempty(values) && return Int[]
    threshold = relative_threshold * maximum(values)
    candidates = Int[]
    for index in 2:(length(values) - 1)
        values[index] >= threshold || continue
        values[index] > values[index - 1] && values[index] >= values[index + 1] && push!(candidates, index)
    end
    sort!(candidates; by=index -> values[index], rev=true)
    selected = Int[]
    for index in candidates
        all(abs(axis[index] - axis[other]) >= min_separation for other in selected) || continue
        push!(selected, index)
        length(selected) >= max_peaks && break
    end
    return selected
end

function _senechal_two_ridge_diagnostic(k_values, omega_axis, intensity; eta::Float64)
    length(omega_axis) >= 3 || return (coverage=0.0, median_separation=NaN, resolved=0, total=0)
    spacing = abs(omega_axis[2] - omega_axis[1])
    smoothing_radius = max(1, round(Int, max(2 * eta, 0.08) / spacing))
    resolved = 0
    total = 0
    separations = Float64[]
    sectors = ((0.15pi, 0.45pi, :removal), (0.55pi, 0.85pi, :addition))
    for (kmin, kmax, channel) in sectors
        for kindex in eachindex(k_values)
            kmin <= k_values[kindex] <= kmax || continue
            mask = channel === :removal ? findall(omega -> omega < -0.35, omega_axis) : findall(omega -> omega > 0.35, omega_axis)
            isempty(mask) && continue
            total += 1
            local_axis = omega_axis[mask]
            local_values = _senechal_smoothed(vec(intensity[kindex, mask]), smoothing_radius)
            peaks = _senechal_separated_peaks(local_axis, local_values)
            length(peaks) >= 2 || continue
            resolved += 1
            push!(separations, abs(local_axis[peaks[1]] - local_axis[peaks[2]]))
        end
    end
    coverage = total > 0 ? resolved / total : 0.0
    separation = isempty(separations) ? NaN : median(separations)
    return (coverage=coverage, median_separation=separation, resolved=resolved, total=total)
end

function _senechal_workspace_components(workspace::FermionGreenWorkspace)
    metadata = workspace.metadata
    ground = metadata[:ground_build_time] + metadata[:ground_solve_time]
    sector_build = metadata[:particle_build_time] + metadata[:hole_build_time]
    seeds = metadata[:particle_seed_time] + metadata[:hole_seed_time]
    chains = metadata[:particle_chain_time] + metadata[:hole_chain_time]
    return (ground=ground, sector_build=sector_build, seeds=seeds, chains=chains)
end

let
    eta = parse(Float64, get(ENV, "PHUNDAMENTAL_CPT_ETA", "0.03"))
    source_n = parse(Int, get(ENV, "PHUNDAMENTAL_CPT_SOURCE_N", "12"))
    source_n == 12 || @warn "The published Senechal Fig. 3 source benchmark uses N=12; the requested override is N=$source_n"
    iseven(source_n) || error("PHUNDAMENTAL_CPT_SOURCE_N must be even for the half-filled benchmark")
    source_krylov = parse(Int, get(ENV, "PHUNDAMENTAL_CPT_KRYLOV_DIM", "160"))
    source_ground_krylov = parse(Int, get(ENV, "PHUNDAMENTAL_CPT_GROUND_KRYLOV_DIM", string(source_krylov)))
    source_k_points = parse(Int, get(ENV, "PHUNDAMENTAL_CPT_K_POINTS", "161"))
    source_omega_points = parse(Int, get(ENV, "PHUNDAMENTAL_CPT_OMEGA_POINTS", "481"))
    source_parallel = lowercase(strip(get(ENV, "PHUNDAMENTAL_CPT_PARALLEL", "true"))) in ("1", "true", "yes", "on")
    source_enabled = lowercase(strip(get(ENV, "PHUNDAMENTAL_CPT_SOURCE", "true"))) in ("1", "true", "yes", "on")
    display_quantile = parse(Float64, get(ENV, "PHUNDAMENTAL_CPT_DISPLAY_QUANTILE", "0.995"))
    scaling_sizes = _senechal_parse_int_list("PHUNDAMENTAL_CPT_SCALING_SIZES", "6,8,10")
    krylov_convergence_dims = _senechal_parse_int_list("PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS", "96,128,160")
    krylov_convergence_tol = parse(Float64, get(ENV, "PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_TOL", "0.05"))
    omega_halfwidths = _senechal_parse_float_list("PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS", "6,8,10")
    omega_sum_rule_tol = parse(Float64, get(ENV, "PHUNDAMENTAL_CPT_OMEGA_SUM_RULE_TOL", "0.01"))
    ridge_coverage_tol = parse(Float64, get(ENV, "PHUNDAMENTAL_CPT_RIDGE_COVERAGE_TOL", "0.50"))

    eta > 0 || error("PHUNDAMENTAL_CPT_ETA must be positive")
    source_krylov > 1 || error("PHUNDAMENTAL_CPT_KRYLOV_DIM must exceed one")
    source_ground_krylov > 1 || error("PHUNDAMENTAL_CPT_GROUND_KRYLOV_DIM must exceed one")
    source_k_points >= 3 || error("PHUNDAMENTAL_CPT_K_POINTS must be at least 3")
    source_omega_points >= 101 || error("PHUNDAMENTAL_CPT_OMEGA_POINTS must be at least 101")
    0 < display_quantile <= 1 || error("PHUNDAMENTAL_CPT_DISPLAY_QUANTILE must lie in (0, 1]")
    all(iseven, scaling_sizes) || error("PHUNDAMENTAL_CPT_SCALING_SIZES must contain even half-filled cluster sizes")
    all(>(1), krylov_convergence_dims) || error("PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS must contain integers greater than one")
    all(>(0), omega_halfwidths) || error("PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS must contain positive numbers")

    println()
    println("SENECHAL 1D HUBBARD CLUSTER-PERTURBATION BENCHMARK")
    println("Fermionic Lanczos -> cluster Green matrix -> CPT -> A(k,omega)")

    # Exact small-cluster reference: the projected Lanczos matrix Green function must reproduce the complete Lehmann representation.
    validation_model = _senechal_chain_model(4, 4.0)
    validation_workspace = fermion_green_workspace(
        validation_model;
        ground_sector=FermionSector(nup=2, ndown=2),
        spin=:up,
        ground_krylov_dim=36,
        green_krylov_dim=24,
        tol=1e-12,
        ground_reorthogonalize=true,
        green_reorthogonalize=true,
        parallel=false,
        seed=41,
    )
    validation_points = ComplexF64[-2.3 + 0.07im, -0.4 + 0.07im, 0.8 + 0.07im, 2.7 + 0.07im]
    validation_references = Matrix{ComplexF64}[]
    lehmann_error = 0.0
    exact_ground_energy = 0.0
    exact_ground_state = ComplexF64[]
    for z in validation_points
        reference_green, exact_ground_energy, exact_ground_state = _senechal_exact_cluster_green(validation_workspace, z)
        push!(validation_references, reference_green)
        native_green = cluster_green_matrix(validation_workspace, z)
        lehmann_error = max(lehmann_error, norm(native_green - reference_green) / max(norm(reference_green), 1.0))
    end
    ground_energy_error = abs(validation_workspace.ground_energy - exact_ground_energy)
    ground_state_overlap = abs(dot(exact_ground_state, validation_workspace.ground_state))

    streaming_validation_timing = @timed fermion_green_workspace(validation_model;
                                                                    ground_sector=FermionSector(nup=2, ndown=2),
                                                                    spin=:up, ground_krylov_dim=36, green_krylov_dim=24,
                                                                    tol=1e-12, ground_reorthogonalize=true, green_reorthogonalize=false,
                                                                    parallel=source_parallel, seed=41)
    streaming_validation_workspace = streaming_validation_timing.value
    streaming_lehmann_error = 0.0
    for (z, reference_green) in zip(validation_points, validation_references)
        native_green = cluster_green_matrix(streaming_validation_workspace, z)
        streaming_lehmann_error = max(streaming_lehmann_error, norm(native_green - reference_green) / max(norm(reference_green), 1.0))
    end
    transition_norm_error = 0.0
    for site in validation_workspace.sites
        particle = fermion_transition_vector(
            validation_workspace.model, exact_ground_state, validation_workspace.ground_basis, validation_workspace.particle_basis, site;
            spin=:up, create=true,
        )
        hole = fermion_transition_vector(
            validation_workspace.model, exact_ground_state, validation_workspace.ground_basis, validation_workspace.hole_basis, site;
            spin=:up, create=false,
        )
        transition_norm_error = max(transition_norm_error, abs(norm(particle)^2 + norm(hole)^2 - 1.0))
    end
    small_cluster_pass = ground_energy_error <= 1e-10 && 1 - ground_state_overlap <= 1e-10 && lehmann_error <= 1e-9 &&
                         streaming_lehmann_error <= 1e-9 && transition_norm_error <= 1e-10

    println()
    println("SMALL-CLUSTER GREEN-FUNCTION / LEHMANN VALIDATION")
    @printf("Ground-state energy error           = %.3e t\n", ground_energy_error)
    @printf("Ground-state overlap                = %.12f\n", ground_state_overlap)
    @printf("Maximum matrix Green-function error = %.3e\n", lehmann_error)
    @printf("Streaming Green-function error      = %.3e\n", streaming_lehmann_error)
    println("Streaming projected-chain backend   = ", streaming_validation_workspace.metadata[:green_chain_backend])
    @printf("Streaming validation workspace      = %.6f s\n", streaming_validation_timing.time)
    @printf("Maximum c/cdag completeness error   = %.3e\n", transition_norm_error)
    println("  Sector-changing fermion operators : ", transition_norm_error <= 1e-10 ? "PASS" : "FAIL")
    println("  Reorthogonalized Lanczos vs Lehmann: ", lehmann_error <= 1e-9 ? "PASS" : "FAIL")
    println("  Streaming Lanczos vs Lehmann G_ab : ", streaming_lehmann_error <= 1e-9 ? "PASS" : "FAIL")

    # CPT is exact for U=0 when the intra- and inter-cluster hopping amplitudes are equal.
    free_model = _senechal_chain_model(4, 0.0)
    free_workspace = fermion_green_workspace(
        free_model;
        ground_sector=FermionSector(nup=2, ndown=2),
        spin=:up,
        ground_krylov_dim=36,
        green_krylov_dim=24,
        tol=1e-12,
        ground_reorthogonalize=true,
        green_reorthogonalize=true,
        parallel=false,
        seed=43,
    )
    free_error = 0.0
    for k in (0.13pi, 0.31pi, 0.57pi, 0.83pi), omega in (-2.4, -0.7, 0.6, 2.1)
        z = ComplexF64(omega, 0.08)
        native = cluster_perturbation_green(cluster_green_matrix(free_workspace, z), k; t_intercluster=1.0)
        reference = inv(z + 2 * cos(k))
        free_error = max(free_error, abs(native - reference) / max(abs(reference), 1.0))
    end
    free_limit_pass = free_error <= 1e-9

    println()
    println("NONINTERACTING CPT EXACTNESS")
    @printf("Maximum U=0 infinite-chain G error = %.3e\n", free_error)
    println("  Senechal CPT weak-coupling limit  : ", free_limit_pass ? "PASS" : "FAIL")

    # N=4 validation above provides a global compilation warm-up. Timed scaling therefore begins at N=6 and excludes the compilation-dominated N=4 point.
    scaling_rows = NamedTuple[]
    for nsites in scaling_sizes
        nsites < source_n || continue
        GC.gc()
        model = _senechal_chain_model(nsites, 4.0)
        nup = nsites ÷ 2
        timing = @timed fermion_green_workspace(
            model;
            ground_sector=FermionSector(nup=nup, ndown=nup),
            spin=:up,
            ground_krylov_dim=source_ground_krylov,
            green_krylov_dim=source_krylov,
            tol=1e-10,
            ground_reorthogonalize=false,
            green_reorthogonalize=false,
            parallel=source_parallel,
            seed=47 + nsites,
        )
        workspace = timing.value
        components = _senechal_workspace_components(workspace)
        push!(scaling_rows, (
            nsites=nsites,
            ground_dimension=workspace.metadata[:ground_dimension],
            particle_dimension=workspace.metadata[:particle_dimension],
            time=timing.time,
            allocated=timing.bytes / 2.0^30,
            ground_time=components.ground,
            sector_build_time=components.sector_build,
            seed_time=components.seeds,
            chain_time=components.chains,
        ))
    end

    source_pass = true
    source_spectrum = nothing
    source_workspace = nothing
    source_workspace_time = NaN
    source_workspace_bytes = NaN
    source_evaluation_time = NaN
    source_sum_rule_error = NaN
    particle_hole_error = NaN
    gap_ratio = NaN
    ridge_diagnostic = (coverage=NaN, median_separation=NaN, resolved=0, total=0)
    source_ground_residual = NaN
    krylov_convergence_pass = true
    omega_window_pass = true
    krylov_rows = NamedTuple[]
    omega_rows = NamedTuple[]

    if source_enabled
        GC.gc()
        source_model = _senechal_chain_model(source_n, 4.0)
        half_filling = source_n ÷ 2
        workspace_timing = @timed fermion_green_workspace(
            source_model;
            ground_sector=FermionSector(nup=half_filling, ndown=half_filling),
            spin=:up,
            ground_krylov_dim=source_ground_krylov,
            green_krylov_dim=source_krylov,
            tol=1e-10,
            ground_reorthogonalize=false,
            green_reorthogonalize=false,
            parallel=source_parallel,
            seed=59,
        )
        source_workspace = workspace_timing.value
        source_workspace_time = workspace_timing.time
        source_workspace_bytes = workspace_timing.bytes / 2.0^30
        source_ground_residual = source_workspace.metadata[:ground_residual]
        source_components = _senechal_workspace_components(source_workspace)
        push!(scaling_rows, (
            nsites=source_n,
            ground_dimension=source_workspace.metadata[:ground_dimension],
            particle_dimension=source_workspace.metadata[:particle_dimension],
            time=source_workspace_time,
            allocated=source_workspace_bytes,
            ground_time=source_components.ground,
            sector_build_time=source_components.sector_build,
            seed_time=source_components.seeds,
            chain_time=source_components.chains,
        ))

        k_values = collect(range(0.0, pi; length=source_k_points))
        omega_axis = collect(range(-6.0, 6.0; length=source_omega_points))
        spectrum_timing = @timed cluster_perturbation_spectral_function(
            source_workspace, k_values, omega_axis; eta=eta, t_intercluster=1.0, normalization=:spectral_density, parallel=source_parallel,
        )
        source_spectrum = spectrum_timing.value
        source_evaluation_time = spectrum_timing.time
        source_sum_rule_error = maximum(
            abs(_senechal_trapezoid(omega_axis, vec(source_spectrum.intensity[index, :])) - 1.0) for index in eachindex(k_values)
        )
        particle_hole_error = _senechal_particle_hole_residual(source_spectrum)

        kpi2 = argmin(abs.(k_values .- pi / 2))
        central = findall(abs.(omega_axis) .<= 0.25)
        side = findall((abs.(omega_axis) .>= 0.5) .& (abs.(omega_axis) .<= 2.5))
        gap_ratio = maximum(source_spectrum.intensity[kpi2, central]) / max(maximum(source_spectrum.intensity[kpi2, side]), eps(Float64))
        ridge_diagnostic = _senechal_two_ridge_diagnostic(k_values, omega_axis, source_spectrum.intensity; eta=eta)

        convergence_dims = sort!(unique(vcat(filter(<=(source_krylov), krylov_convergence_dims), source_krylov)))
        for dimension in convergence_dims
            if dimension == source_krylov
                residual = 0.0
                evaluation_time = source_evaluation_time
                sum_rule_error = source_sum_rule_error
            else
                timing = @timed cluster_perturbation_spectral_function(
                    source_workspace, k_values, omega_axis; eta=eta, t_intercluster=1.0, normalization=:spectral_density,
                    parallel=source_parallel, krylov_dim=dimension,
                )
                spectrum = timing.value
                residual = _senechal_map_residual(spectrum.intensity, source_spectrum.intensity)
                evaluation_time = timing.time
                sum_rule_error = maximum(
                    abs(_senechal_trapezoid(omega_axis, vec(spectrum.intensity[index, :])) - 1.0) for index in eachindex(k_values)
                )
            end
            push!(krylov_rows, (dimension=dimension, residual=residual, sum_rule_error=sum_rule_error, time=evaluation_time))
        end
        if length(krylov_rows) >= 2
            krylov_convergence_pass = krylov_rows[end - 1].residual <= krylov_convergence_tol
        end

        spacing = abs(omega_axis[2] - omega_axis[1])
        probe_k = [0.0, pi / 4, pi / 2, 3pi / 4, pi]
        for halfwidth in omega_halfwidths
            points = round(Int, 2 * halfwidth / spacing) + 1
            axis = collect(range(-halfwidth, halfwidth; length=points))
            timing = @timed cluster_perturbation_spectral_function(
                source_workspace, probe_k, axis; eta=eta, t_intercluster=1.0, normalization=:spectral_density, parallel=source_parallel,
            )
            spectrum = timing.value
            sum_rule_error = maximum(
                abs(_senechal_trapezoid(axis, vec(spectrum.intensity[index, :])) - 1.0) for index in eachindex(probe_k)
            )
            push!(omega_rows, (halfwidth=halfwidth, points=points, sum_rule_error=sum_rule_error, time=timing.time))
        end
        omega_window_pass = omega_rows[end].sum_rule_error <= omega_sum_rule_tol

        source_pass = source_ground_residual <= 1e-7 && source_sum_rule_error <= 0.05 && particle_hole_error <= 0.05 &&
                      gap_ratio <= 0.20 && ridge_diagnostic.coverage >= ridge_coverage_tol && krylov_convergence_pass &&
                      omega_window_pass && all(isfinite, source_spectrum.intensity)

        println()
        println("SENECHAL FIG. 3 SOURCE-REGIME RECONSTRUCTION")
        @printf("Cluster size / half-filled sector   = N=%d, (Nup,Ndown)=(%d,%d)\n", source_n, half_filling, half_filling)
        @printf("U/t / eta                           = 4.0 / %.4f\n", eta)
        @printf(
            "Ground / particle Hilbert dimension = %d / %d\n",
            source_workspace.metadata[:ground_dimension], source_workspace.metadata[:particle_dimension],
        )
        @printf("Ground-state Ritz residual          = %.3e\n", source_ground_residual)
        println("Projected-chain backend             = ", source_workspace.metadata[:green_chain_backend])
        @printf("Green-workspace construction        = %.3f s, %.3f GiB allocated\n", source_workspace_time, source_workspace_bytes)
        @printf("CPT spectral-map evaluation         = %.3f s\n", source_evaluation_time)
        @printf("Display clipping quantile           = %.4f\n", display_quantile)
        @printf("Maximum spin-resolved sum-rule error = %.3e\n", source_sum_rule_error)
        @printf("Particle-hole symmetry residual     = %.3e\n", particle_hole_error)
        @printf("k=pi/2 central-gap intensity ratio  = %.3e\n", gap_ratio)
        @printf(
            "Two-ridge momentum coverage         = %.3f [%d/%d]\n",
            ridge_diagnostic.coverage, ridge_diagnostic.resolved, ridge_diagnostic.total,
        )
        @printf("Median resolved ridge separation    = %.3f t\n", ridge_diagnostic.median_separation)
        println("  Half-filled Mott-gap diagnostic    : ", gap_ratio <= 0.20 ? "PASS" : "FAIL")
        println("  Particle-hole symmetry             : ", particle_hole_error <= 0.05 ? "PASS" : "FAIL")
        println("  Spectral normalization             : ", source_sum_rule_error <= 0.05 ? "PASS" : "FAIL")
        println("  Spinon/holon two-ridge structure   : ", ridge_diagnostic.coverage >= ridge_coverage_tol ? "PASS" : "FAIL")

        println()
        println("PROJECTED-LANCZOS GREEN-FUNCTION CONVERGENCE")
        @printf("%10s %14s %16s %12s\n", "Krylov", "map residual", "sum-rule error", "eval [s]")
        for row in krylov_rows
            @printf("%10d %14.3e %16.3e %12.3f\n", row.dimension, row.residual, row.sum_rule_error, row.time)
        end
        if length(krylov_rows) >= 2
            @printf("Final pre-reference map residual    = %.3e [tol %.3e]\n", krylov_rows[end - 1].residual, krylov_convergence_tol)
        end
        println("  Projected-Lanczos convergence      : ", krylov_convergence_pass ? "PASS" : "FAIL")

        println()
        println("FINITE FREQUENCY-WINDOW SUM-RULE CONVERGENCE")
        @printf("%12s %10s %16s %12s\n", "halfwidth", "Nomega", "sum-rule error", "eval [s]")
        for row in omega_rows
            @printf("%12.2f %10d %16.3e %12.3f\n", row.halfwidth, row.points, row.sum_rule_error, row.time)
        end
        println("  Wide-window spectral normalization : ", omega_window_pass ? "PASS" : "FAIL")

        positive = [value for value in vec(source_spectrum.intensity) if isfinite(value) && value > 0]
        display_scale = isempty(positive) ? 1.0 : quantile(positive, display_quantile)
        display_map = clamp.(source_spectrum.intensity ./ max(display_scale, eps(Float64)), 0.0, 1.0)

        fig = Figure(size=(1450, 820))
        ax_map = Axis(fig[1:2, 1]; xlabel="k/pi", ylabel="omega/t", title="Senechal CPT: 1D Hubbard, U=4t, N=$source_n")
        heatmap!(ax_map, k_values ./ pi, omega_axis, display_map; colorrange=(0.0, 1.0))
        Colorbar(fig[1:2, 2]; limits=(0.0, 1.0), label="display-normalized A(k,omega)")

        ax_cuts = Axis(fig[1, 3]; xlabel="omega/t", ylabel="normalized A", title="Representative momentum cuts")
        for (label, target_k) in (("k=0", 0.0), ("k=pi/4", pi / 4), ("k=pi/2", pi / 2))
            index = argmin(abs.(k_values .- target_k))
            values = vec(source_spectrum.intensity[index, :])
            lines!(ax_cuts, omega_axis, values ./ max(maximum(values), eps(Float64)); label=label)
        end
        axislegend(ax_cuts; position=:rt)

        ax_scale = Axis(
            fig[2, 3]; xscale=log10, yscale=log10, xlabel="half-filled Hilbert dimension", ylabel="workspace time [s]",
            title="Electron-sector scaling",
        )
        sorted_rows = sort(scaling_rows; by=row -> row.ground_dimension)
        scatter!(ax_scale, [row.ground_dimension for row in sorted_rows], [row.time for row in sorted_rows])
        lines!(ax_scale, [row.ground_dimension for row in sorted_rows], [row.time for row in sorted_rows])
        save(joinpath(BENCHMARK_FIGURE_DIR, "Senechal-Hubbard-CPT-benchmark.png"), fig)
    else
        println()
        println("SENECHAL FIG. 3 SOURCE-REGIME RECONSTRUCTION: NOT RUN [PHUNDAMENTAL_CPT_SOURCE=0]")
    end

    sort!(scaling_rows; by=row -> row.ground_dimension)
    runtime_fit = _senechal_log_fit([row.ground_dimension for row in scaling_rows], [row.time for row in scaling_rows])
    allocation_fit = _senechal_log_fit([row.ground_dimension for row in scaling_rows], [row.allocated for row in scaling_rows])
    chain_fit = _senechal_log_fit([row.ground_dimension for row in scaling_rows], [row.chain_time for row in scaling_rows])

    println()
    println("FERMION GREEN-WORKSPACE SCALING")
    println("N=4 is excluded from scaling because it serves as the compilation/Lehmann warm-up point.")
    @printf(
        "%5s %12s %12s %10s %10s %10s %10s %10s %11s\n",
        "N", "dim N", "dim N+1", "ground", "build +/-", "seeds", "chains", "total", "alloc GiB",
    )
    for row in scaling_rows
        @printf(
            "%5d %12d %12d %10.3f %10.3f %10.3f %10.3f %10.3f %11.3f\n",
            row.nsites, row.ground_dimension, row.particle_dimension, row.ground_time, row.sector_build_time, row.seed_time,
            row.chain_time, row.time, row.allocated,
        )
    end
    @printf("Workspace runtime exponent vs dim  = %.3f (R²=%.3f; %d points)\n", runtime_fit.exponent, runtime_fit.r2, runtime_fit.n)
    @printf("Projected-chain exponent vs dim     = %.3f (R²=%.3f; %d points)\n", chain_fit.exponent, chain_fit.r2, chain_fit.n)
    @printf("Workspace allocation exponent       = %.3f (R²=%.3f; %d points)\n", allocation_fit.exponent, allocation_fit.r2, allocation_fit.n)

    core_pass = small_cluster_pass && free_limit_pass
    overall_pass = core_pass && source_pass
    println()
    println("SENECHAL HUBBARD CPT BENCHMARK VALIDATION")
    println("  Sector-changing fermion API         : ", transition_norm_error <= 1e-10 ? "PASS" : "FAIL")
    println("  Cluster G_ab vs exact Lehmann       : ", lehmann_error <= 1e-9 ? "PASS" : "FAIL")
    println("  Streaming G_ab vs exact Lehmann     : ", streaming_lehmann_error <= 1e-9 ? "PASS" : "FAIL")
    println("  U=0 CPT exactness                   : ", free_limit_pass ? "PASS" : "FAIL")
    println("  Warmed Green-workspace scaling      : GENERATED")
    println("  Projected-Lanczos map convergence   : ", source_enabled ? (krylov_convergence_pass ? "PASS" : "FAIL") : "NOT RUN")
    println("  Frequency-window normalization      : ", source_enabled ? (omega_window_pass ? "PASS" : "FAIL") : "NOT RUN")
    println("  N=12, U=4t source-regime spectrum   : ", source_enabled ? (source_pass ? "PASS" : "FAIL") : "NOT RUN")
    println()
    println("Senechal Hubbard CPT electron-sector benchmark: ", overall_pass ? "PASS" : "FAIL")
    overall_pass || error("Senechal Hubbard CPT benchmark failed; inspect subsection diagnostics")
end
