# Exact symmetry-sector canonical TPQ thermodynamics.
#
# The canonical trace is reconstructed from unnormalized stochastic trace
# estimators in every conserved sector. Sector observables are never averaged
# after normalization. For a representative sector q with dimension Dq and
# exact symmetry multiplicity mq, each normalized random-state trace estimate
# is weighted by mq*Dq before sectors are combined.

@inline function _bounded_int_bytes(value::Integer)
    value <= typemax(Int) || return typemax(Int)
    return Int(value)
end

function _sector_memory_budget(decomposition::SectorDecomposition)
    decomposition.memory_budget_bytes > 0 && return decomposition.memory_budget_bytes
    total = _bounded_int_bytes(big(Sys.total_memory()))
    return floor(Int, decomposition.memory_fraction * total)
end

function _sector_lanczos_workspace_bytes(dim::Int, method::CanonicalTPQ)
    m = min(dim, _thermal_krylov_dim(method))
    bytes = big(3) * dim * sizeof(ComplexF64)
    if _thermal_reorthogonalize(method)
        bytes += big(dim) * m * sizeof(ComplexF64)
        bytes += big(m) * sizeof(ComplexF64)
    end
    return _bounded_int_bytes(bytes)
end

function _sector_rank_lookup_requirements(plan)
    N = plan.nsites
    N < Sys.WORD_SIZE - 1 || return (possible=false, entries=0, bytes=typemax(Int), reason=:mask_space_not_int_indexable)
    maximum(plan.dimensions) <= Int(typemax(UInt32)) || return (possible=false, entries=0, bytes=typemax(Int), reason=:sector_rank_exceeds_uint32)
    entries_big = big(1) << N
    bytes_big = entries_big * sizeof(UInt32)
    entries_big <= typemax(Int) || return (possible=false, entries=0, bytes=typemax(Int), reason=:lookup_length_exceeds_int)
    bytes_big <= typemax(Int) || return (possible=false, entries=0, bytes=typemax(Int), reason=:lookup_bytes_exceed_int)
    return (possible=true, entries=Int(entries_big), bytes=Int(bytes_big), reason=:ok)
end

function _allocate_sector_rank_lookup(plan, method::CanonicalTPQ, decomposition::SectorDecomposition, budget::Int)
    req = _sector_rank_lookup_requirements(plan)
    largest_workspace = _sector_lanczos_workspace_bytes(maximum(plan.dimensions), method)

    if decomposition.rank_lookup === :none
        return nothing, :combinatorial, 0, largest_workspace
    end

    if !req.possible
        decomposition.rank_lookup === :force && throw(ArgumentError("forced dense sector-rank lookup is unavailable ($(req.reason))"))
        return nothing, :combinatorial, 0, largest_workspace
    end

    projected = big(largest_workspace) + req.bytes
    if decomposition.rank_lookup === :auto && projected > budget
        return nothing, :combinatorial, 0, largest_workspace
    elseif decomposition.rank_lookup === :force && projected > budget
        projected_gib = Float64(projected) / 2.0^30
        budget_gib = budget / 2.0^30
        throw(ArgumentError("forced sector-rank lookup would require an estimated $projected_gib GiB working set, exceeding the configured $budget_gib GiB budget"))
    end

    lookup = try
        zeros(UInt32, req.entries)
    catch err
        if err isa OutOfMemoryError && decomposition.rank_lookup === :auto
            return nothing, :combinatorial_allocation_fallback, 0, largest_workspace
        elseif err isa OutOfMemoryError
            throw(ErrorException("failed to allocate $(req.bytes / 2.0^30) GiB sector-rank lookup; use rank_lookup=:none or a smaller decomposition"))
        end
        rethrow()
    end
    return lookup, :dense_uint32, req.bytes, _bounded_int_bytes(projected)
end

function _validate_sector_workspace_method(method::CanonicalTPQ)
    decomposition = method.decomposition
    isnothing(decomposition) && throw(ArgumentError("sector thermal workspace requires CanonicalTPQ(decomposition=SectorDecomposition(...))"))
    method.matrix_free || throw(ArgumentError("sector-decomposed CanonicalTPQ requires matrix_free=true"))
    method.store_vectors && throw(ArgumentError("sector-decomposed CanonicalTPQ currently supports scalar thermal curves only; set store_vectors=false"))
    return decomposition
end

function _sector_thermal_workspace(model::ManyBodyModel, method::CanonicalTPQ)
    decomposition = _validate_sector_workspace_method(method)
    decomposition.symmetry === :total_sz || throw(ArgumentError("unsupported thermal-sector symmetry $(decomposition.symmetry)"))

    plan = Solvers.spin_sector_plan(model; exploit_equivalent_sectors=decomposition.exploit_equivalent_sectors, hbar=method.hbar)
    largest_dimension = maximum(plan.dimensions)
    largest_dimension <= method.max_basis_dimension ||
        throw(ArgumentError("largest spin sector dimension $largest_dimension exceeds max_basis_dimension=$(method.max_basis_dimension)"))

    budget = _sector_memory_budget(decomposition)
    base_peak = _sector_lanczos_workspace_bytes(largest_dimension, method)
    if base_peak > budget
        peak_gib = base_peak / 2.0^30
        budget_gib = budget / 2.0^30
        message = "largest sector Krylov workspace is estimated at $peak_gib GiB, exceeding the configured $budget_gib GiB memory budget; " *
                  "disable reorthogonalization or increase the budget"
        throw(ArgumentError(message))
    end
    rank_lookup, rank_strategy, rank_bytes, estimated_peak = _allocate_sector_rank_lookup(plan, method, decomposition, budget)

    operators = Vector{Solvers.PackedSpinOperator}(undef, length(plan.sectors))
    for i in eachindex(plan.sectors)
        if method.progress
            @printf("Preparing S^z sector nup=%d/%d, dimension=%d, multiplicity=%d\n", plan.nup[i], plan.nsites, plan.dimensions[i], plan.multiplicities[i])
            flush(stdout)
        end
        packed = Solvers.optimized_matrix_free_operator(
            model;
            sector=plan.sectors[i],
            hbar=method.hbar,
            max_basis_dimension=method.max_basis_dimension,
            materialize_basis=false,
            threaded=decomposition.threaded_matvec,
            rank_lookup=rank_lookup,
            populate_rank_lookup=!isnothing(rank_lookup),
        )
        packed.dimension == plan.dimensions[i] || throw(DimensionMismatch("sector plan and packed operator dimensions disagree for nup=$(plan.nup[i])"))
        packed.total_sz_conserved || throw(ArgumentError("packed sector operator lost total-Sz conservation certification"))
        operators[i] = packed.operator
    end

    metadata = Dict{Symbol,Any}(
        :symmetry => decomposition.symmetry,
        :nsites => plan.nsites,
        :equivalence_requested => plan.equivalence_requested,
        :equivalence_used => plan.equivalence_used,
        :global_spin_flip_symmetric => plan.profile.global_spin_flip_symmetric,
        :total_sz_conserved => plan.profile.conserves_total_sz,
        :sector_nup => copy(plan.nup),
        :sector_dimensions => copy(plan.dimensions),
        :sector_multiplicities => copy(plan.multiplicities),
        :rank_lookup_strategy => rank_strategy,
        :rank_lookup_bytes => rank_bytes,
        :largest_sector_workspace_bytes => base_peak,
        :estimated_peak_bytes => estimated_peak,
        :memory_budget_bytes => budget,
        :threaded_matvec => decomposition.threaded_matvec,
        :thread_count => Threads.nthreads(),
    )

    return SectorThermalWorkspace(model, decomposition, plan.sectors, plan.nup, plan.multiplicities, plan.dimensions, operators, rank_lookup,
                                  plan.total_dimension, :packed_spin_sectorized, method.hbar, budget, estimated_peak, metadata)
end

@inline function _canonical_tpq_sector_sample_seed(seed::Int, nup::Int, sample::Int)
    modulus = Int128(typemax(Int))
    raw = Int128(seed) + Int128(130363) * Int128(nup + 1) + Int128(104729) * Int128(sample - 1)
    return Int(mod(raw, modulus))
end

function _trace_sector_quadratures(ws::SectorThermalWorkspace, method::CanonicalTPQ; temperatures=nothing, kB::Real=1.0)
    R = _thermal_samples(method)
    convergence_betas = _thermal_convergence_betas(method, temperatures, kB)
    nsectors = length(ws.sectors)
    quadratures = [Vector{_ThermalQuadrature}(undef, R) for _ in 1:nsectors]
    threaded_matvec = ws.decomposition.threaded_matvec && Threads.nthreads() > 1
    requested_sample_parallel = method.parallel && !threaded_matvec && Threads.nthreads() > 1 && R > 1
    rank_bytes = get(ws.metadata, :rank_lookup_bytes, 0)
    largest_sample_bytes = _sector_lanczos_workspace_bytes(maximum(ws.dimensions), method)
    parallel_bytes = big(rank_bytes) + big(min(Threads.nthreads(), R)) * largest_sample_bytes
    sample_parallel = requested_sample_parallel && parallel_bytes <= ws.memory_budget_bytes
    runtime_peak_bytes = sample_parallel ? _bounded_int_bytes(parallel_bytes) : ws.estimated_peak_bytes
    progress_lock = ReentrantLock()

    for s in 1:nsectors
        A = ws.operators[s]
        dim = ws.dimensions[s]
        nup = ws.nup[s]
        if method.progress
            @printf("CanonicalTPQ sector nup=%d: %d realization(s), dimension=%d\n", nup, R, dim)
            flush(stdout)
        end

        run_sample = function (r)
            rng = MersenneTwister(_canonical_tpq_sector_sample_seed(method.seed, nup, r))
            qd, _, _ = _sample_quadrature(A, dim, method, rng; store_basis=false, convergence_betas=convergence_betas)
            quadratures[s][r] = qd
            if method.progress
                lock(progress_lock) do
                    @printf("CanonicalTPQ sector nup=%d realization %d/%d complete\n", nup, r, R)
                    flush(stdout)
                end
            end
            return nothing
        end

        if sample_parallel
            Threads.@threads :static for r in 1:R
                run_sample(r)
            end
        else
            for r in 1:R
                run_sample(r)
            end
        end
    end

    return quadratures, sample_parallel, threaded_matvec, runtime_peak_bytes
end

function _trace_thermodynamic_from_sums(zsum::Float64, usum::Float64, h2sum::Float64, nsamples::Int, beta::Float64, T::Float64, kb::Float64, eref::Float64)
    zbar = zsum / nsamples
    zbar > 0 && isfinite(zbar) || throw(ArgumentError("sector-combined partition estimator vanished or became non-finite"))
    logZ = -beta * eref + log(zbar)
    U = (usum / nsamples) / zbar
    E2 = (h2sum / nsamples) / zbar
    varE = max(E2 - U^2, 0.0)
    F = -kb * T * logZ
    S = (U - F) / T
    Cv = varE / (kb * T^2)
    return (logZ=logZ, free_energy=F, internal_energy=U, entropy=S, heat_capacity=Cv, energy_variance=varE)
end

function _sector_jackknife_thermodynamic_errors(z::Vector{Float64}, u::Vector{Float64}, h2::Vector{Float64}, beta::Float64, T::Float64, kb::Float64, eref::Float64)
    n = length(z)
    if n <= 1
        return Dict{Symbol,Float64}(:logZ => NaN, :free_energy => NaN, :internal_energy => NaN, :entropy => NaN, :heat_capacity => NaN)
    end

    zsum = sum(z)
    usum = sum(u)
    h2sum = sum(h2)
    logZ = Vector{Float64}(undef, n)
    F = similar(logZ)
    U = similar(logZ)
    S = similar(logZ)
    Cv = similar(logZ)

    @inbounds for r in 1:n
        theta = _trace_thermodynamic_from_sums(zsum - z[r], usum - u[r], h2sum - h2[r], n - 1, beta, T, kb, eref)
        logZ[r] = theta.logZ
        F[r] = theta.free_energy
        U[r] = theta.internal_energy
        S[r] = theta.entropy
        Cv[r] = theta.heat_capacity
    end

    return Dict{Symbol,Float64}(
        :logZ => _jackknife_stderr(logZ),
        :free_energy => _jackknife_stderr(F),
        :internal_energy => _jackknife_stderr(U),
        :entropy => _jackknife_stderr(S),
        :heat_capacity => _jackknife_stderr(Cv),
    )
end

@inline function _quadrature_shifted_moments(q::_ThermalQuadrature, beta::Float64, eref::Float64)
    z = 0.0
    u = 0.0
    h2 = 0.0
    @inbounds for j in eachindex(q.energies)
        energy = q.energies[j]
        weight = q.overlaps[j] * exp(-beta * (energy - eref))
        z += weight
        u += weight * energy
        h2 += weight * energy^2
    end
    return z, u, h2
end

function _sector_sampled_point(quadratures::Vector{Vector{_ThermalQuadrature}}, ws::SectorThermalWorkspace, temperature::Real, kB::Real)
    temperature > 0 || throw(ArgumentError("sectorized stochastic thermal sampling requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    isempty(quadratures) && throw(ArgumentError("sector quadrature collection is empty"))

    T = Float64(temperature)
    kb = Float64(kB)
    beta = inv(kb * T)
    R = length(quadratures[1])
    all(length(qs) == R for qs in quadratures) || throw(DimensionMismatch("all sectors must use the same number of TPQ realizations for stratified jackknife recombination"))
    eref = minimum(minimum(q.energies) for qs in quadratures for q in qs)

    zglobal = zeros(Float64, R)
    uglobal = zeros(Float64, R)
    h2global = zeros(Float64, R)
    sector_shifted_trace = zeros(Float64, length(quadratures))

    for s in eachindex(quadratures)
        factor = Float64(ws.multiplicities[s]) * Float64(ws.dimensions[s])
        for r in 1:R
            zr, ur, h2r = _quadrature_shifted_moments(quadratures[s][r], beta, eref)
            weighted_z = factor * zr
            zglobal[r] += weighted_z
            uglobal[r] += factor * ur
            h2global[r] += factor * h2r
            sector_shifted_trace[s] += weighted_z
        end
    end

    theta = _trace_thermodynamic_from_sums(sum(zglobal), sum(uglobal), sum(h2global), R, beta, T, kb, eref)
    stderr = _sector_jackknife_thermodynamic_errors(zglobal, uglobal, h2global, beta, T, kb, eref)
    total_shifted_trace = sum(sector_shifted_trace)
    total_shifted_trace > 0 || throw(ArgumentError("sector-combined partition estimator is zero"))
    sector_weights = sector_shifted_trace ./ total_shifted_trace

    point = ThermodynamicPoint(
        T,
        kb,
        theta.logZ,
        theta.free_energy,
        theta.internal_energy,
        theta.entropy,
        theta.heat_capacity,
        stderr,
        Dict{Symbol,Any}(
            :method => :canonical_tpq_sector_trace,
            :samples => R,
            :dimension => ws.total_dimension,
            :energy_reference => eref,
            :energy_variance => theta.energy_variance,
            :symmetry => ws.decomposition.symmetry,
            :sector_nup => copy(ws.nup),
            :sector_dimensions => copy(ws.dimensions),
            :sector_multiplicities => copy(ws.multiplicities),
            :sector_partition_weights => sector_weights,
            :uncertainty_method => R > 1 ? :stratified_delete_one_jackknife : :unavailable_single_realization,
        ),
    )
    return point, sector_weights
end

function _sector_thermal_curve(model::ManyBodyModel, method::CanonicalTPQ, temperatures; kB::Real=1.0, workspace=nothing)
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperature grid cannot be empty"))
    all(>(0), Ts) || throw(ArgumentError("stochastic thermal curves currently require all temperatures > 0"))

    ws = if isnothing(workspace)
        _sector_thermal_workspace(model, method)
    else
        workspace isa SectorThermalWorkspace || throw(ArgumentError("sector-decomposed CanonicalTPQ requires a SectorThermalWorkspace"))
        workspace.model === model || throw(ArgumentError("thermal workspace belongs to a different model"))
        workspace.hbar == method.hbar || throw(ArgumentError("thermal workspace hbar does not match the requested thermal method"))
        workspace.decomposition == method.decomposition || throw(ArgumentError("thermal workspace decomposition does not match the requested thermal method"))
        workspace
    end

    quadratures, sample_parallel, threaded_matvec, runtime_peak_bytes = _trace_sector_quadratures(ws, method; temperatures=Ts, kB=kB)
    points = ThermodynamicPoint[]
    sizehint!(points, length(Ts))
    partition_weights = zeros(Float64, length(ws.sectors), length(Ts))
    for (iT, T) in enumerate(Ts)
        point, weights = _sector_sampled_point(quadratures, ws, T, kB)
        push!(points, point)
        @views partition_weights[:, iT] .= weights
    end

    return _curve_from_points(points; metadata=Dict{Symbol,Any}(
        :method => :canonical_tpq,
        :physical_method => :canonical_tpq,
        :propagator => method.propagator,
        :decomposition => ws.decomposition.symmetry,
        :samples => method.samples,
        :samples_per_sector => fill(method.samples, length(ws.sectors)),
        :krylov_dim => method.krylov_dim,
        :min_krylov_dim => method.min_krylov_dim,
        :breakdown_tol => method.breakdown_tol,
        :adaptive_krylov => method.adaptive_krylov,
        :convergence_tol => method.convergence_tol,
        :convergence_check_interval => method.convergence_check_interval,
        :convergence_consecutive => method.convergence_consecutive,
        :matrix_free => true,
        :dimension => ws.total_dimension,
        :operator_backend => ws.backend,
        :basis_materialized => false,
        :sectors_evaluated => length(ws.sectors),
        :sector_nup => copy(ws.nup),
        :sector_dimensions => copy(ws.dimensions),
        :sector_multiplicities => copy(ws.multiplicities),
        :sector_partition_weights => partition_weights,
        :equivalent_sector_pairing => get(ws.metadata, :equivalence_used, false),
        :global_spin_flip_symmetric => get(ws.metadata, :global_spin_flip_symmetric, false),
        :rank_lookup_strategy => get(ws.metadata, :rank_lookup_strategy, :unknown),
        :rank_lookup_bytes => get(ws.metadata, :rank_lookup_bytes, 0),
        :memory_budget_bytes => ws.memory_budget_bytes,
        :estimated_peak_bytes => runtime_peak_bytes,
        :threaded_matvec => threaded_matvec,
        :parallel_realizations_requested => method.parallel,
        :parallel_realizations => sample_parallel,
        :parallel_realizations_suppressed_by_threaded_matvec => method.parallel && threaded_matvec,
        :parallel_realizations_memory_limited => method.parallel && !threaded_matvec && !sample_parallel,
        :thread_count => Threads.nthreads(),
        :quadratures_reused_across_temperatures => true,
        :uncertainty_method => method.samples > 1 ? :stratified_delete_one_jackknife : :unavailable_single_realization,
        :quadrature_iterations => [[q.iterations for q in qs] for qs in quadratures],
        :quadrature_tail_beta => [[q.residual_beta for q in qs] for qs in quadratures],
        :quadrature_convergence_assessed => [[q.convergence_assessed for q in qs] for qs in quadratures],
        :quadrature_converged => [[q.converged for q in qs] for qs in quadratures],
        :quadrature_convergence_error => [[q.convergence_error for q in qs] for qs in quadratures],
        :quadrature_convergence_checks => [[q.convergence_checks for q in qs] for qs in quadratures],
        :adaptive_krylov_converged => method.adaptive_krylov ? all(q.converged for qs in quadratures for q in qs) : nothing,
    ))
end
