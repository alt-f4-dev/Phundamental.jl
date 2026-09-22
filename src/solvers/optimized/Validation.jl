"""
    validate_optimized_against_generic(model; sector=nothing, nev=6, ...)

For a small model, compare the packed matrix-free action and Ritz energies against the generic matrix realization on the identical computational basis.
"""
function validate_optimized_against_generic(model::ManyBodyModel; sector=nothing, nev::Integer=6, krylov_dim::Integer=64, tol::Real=1e-10, seed::Integer=1234, hbar::Real=get(model.parameters, :hbar, 1.0), max_basis_dimension::Integer=100_000)
    solver = OptimizedLanczos(nev=nev, krylov_dim=krylov_dim, tol=tol, sector=sector, seed=seed, hbar=hbar, fallback=false, max_basis_dimension=max_basis_dimension)
    packed = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=true)
    A = packed.operator
    basis = packed.basis
    realized = realize(model.hamiltonian, basis; hbar=hbar, sparse=false, check_physical_closure=false)
    H = realized.matrix

    rng = MersenneTwister(seed + 99)
    x = randn(rng, ComplexF64, length(basis))
    yop = A * x
    ymat = H * x
    action_error = norm(yop - ymat) / max(norm(ymat), 1.0)

    F = eigen(Hermitian(0.5 .* (H .+ adjoint(H))))
    take = min(Int(nev), length(F.values))
    reference_energies = Float64.(F.values[1:take])
    optimized = solve(solver, model)
    energy_error = maximum(abs.(optimized.energies[1:take] .- reference_energies))

    return Dict{Symbol,Any}(:passed => action_error <= max(100 * tol, 1e-9) && energy_error <= max(100 * tol, 1e-8), :backend => get(optimized.metadata, :backend, :unknown), :dimension => length(basis), :action_relative_error => action_error, :max_energy_error => energy_error, :reference_energies => reference_energies, :optimized_energies => optimized.energies[1:take], :fused_spin_pairs => A isa PackedSpinOperator ? length(A.pairs) : 0, :residual_spin_terms => A isa PackedSpinOperator ? length(A.residual_terms) : 0)
end

"""
    validate_packed_spin_implicit_basis(model; sector=nothing, tol=1e-12, seed=1234)

Check that basis-implicit and basis-materialized packed spin operators produce the same vector action. This directly validates the low-memory TPQ representation.
"""
function validate_packed_spin_implicit_basis(model::ManyBodyModel; sector=nothing, tol::Real=1e-12, seed::Integer=1234, hbar::Real=get(model.parameters, :hbar, 1.0), max_basis_dimension::Integer=100_000)
    implicit = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=false)
    explicit = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=true)
    implicit.backend === :packed_spin || throw(ArgumentError("validation requires the packed-spin backend"))
    implicit.dimension == explicit.dimension || throw(DimensionMismatch("implicit and explicit packed-spin dimensions differ"))
    rng = MersenneTwister(seed)
    x = randn(rng, ComplexF64, implicit.dimension)
    yi = implicit.operator * x
    ye = explicit.operator * x
    error = norm(yi - ye) / max(norm(ye), 1.0)
    return Dict{Symbol,Any}(:passed => error <= tol, :dimension => implicit.dimension, :relative_action_error => error, :implicit_basis_materialized => !isnothing(implicit.basis), :explicit_basis_materialized => !isnothing(explicit.basis), :hermitian_certified => implicit.hermitian_certified)
end

"""
    validate_packed_spin_rank_lookup(model; sector, ...)

For a small fixed-`Sᶻ` sector, compare the combinatorial sector-index path with the optional dense UInt32 mask-to-rank lookup. The vector actions must agree to `tol` without materializing product-state basis objects.
"""
function validate_packed_spin_rank_lookup(model::ManyBodyModel; sector::SpinSector, tol::Real=1e-12, seed::Integer=1234, hbar::Real=get(model.parameters, :hbar, 1.0), max_basis_dimension::Integer=100_000)
    plain = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=false, threaded=false)
    plain.backend === :packed_spin || throw(ArgumentError("validation requires the packed-spin backend"))
    N = plain.operator.data.nsites
    N < Sys.WORD_SIZE - 1 || throw(ArgumentError("validation rank lookup requires an Int-indexable full mask space"))
    plain.dimension <= Int(typemax(UInt32)) || throw(ArgumentError("validation sector dimension exceeds UInt32 rank capacity"))
    lookup = zeros(UInt32, Int(1) << N)
    accelerated = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=false, threaded=false, rank_lookup=lookup, populate_rank_lookup=true)
    threaded = optimized_matrix_free_operator(model; sector=sector, hbar=hbar, max_basis_dimension=max_basis_dimension, materialize_basis=false, threaded=true, rank_lookup=lookup, populate_rank_lookup=false)

    rng = MersenneTwister(seed)
    x = randn(rng, ComplexF64, plain.dimension)
    yplain = plain.operator * x
    ylookup = accelerated.operator * x
    ythreaded = threaded.operator * x
    lookup_error = norm(yplain - ylookup) / max(norm(yplain), 1.0)
    threaded_error = norm(yplain - ythreaded) / max(norm(yplain), 1.0)
    populated = count(x -> !iszero(x), lookup)
    return Dict{Symbol,Any}(
        :passed => lookup_error <= tol && threaded_error <= tol && populated == plain.dimension,
        :dimension => plain.dimension,
        :lookup_entries_populated => populated,
        :lookup_relative_action_error => lookup_error,
        :threaded_relative_action_error => threaded_error,
        :threads => Threads.nthreads(),
    )
end

"""
    validate_spin_sector_decomposition(model; ...)

For a small spin-1/2 model, verify that the exact fixed-total-`Sᶻ` decomposition reconstructs the complete many-body spectrum. When global-spin-inversion pairing is certified, the representative sector spectrum is repeated according to its exact multiplicity before comparison with the full dense spectrum.
"""
function validate_spin_sector_decomposition(model::ManyBodyModel; exploit_equivalent_sectors::Bool=true, tol::Real=1e-10, hbar::Real=get(model.parameters, :hbar, 1.0), max_total_dimension::Integer=100_000)
    plan = spin_sector_plan(model; exploit_equivalent_sectors=exploit_equivalent_sectors, hbar=hbar)
    plan.total_dimension <= max_total_dimension || throw(ArgumentError("full validation dimension $(plan.total_dimension) exceeds max_total_dimension=$max_total_dimension"))

    full_basis = computational_basis(model)
    length(full_basis) == plan.total_dimension || throw(DimensionMismatch("full computational basis dimension disagrees with sector plan"))
    full_realized = realize(model.hamiltonian, full_basis; hbar=hbar, sparse=false, check_physical_closure=false)
    full_energies = sort!(Float64.(eigvals(Hermitian(0.5 .* (full_realized.matrix .+ adjoint(full_realized.matrix))))))

    sector_energies = Float64[]
    sizehint!(sector_energies, plan.total_dimension)
    for i in eachindex(plan.sectors)
        packed = optimized_matrix_free_operator(model; sector=plan.sectors[i], hbar=hbar, max_basis_dimension=max_total_dimension, materialize_basis=true, threaded=false)
        basis = packed.basis
        realized = realize(model.hamiltonian, basis; hbar=hbar, sparse=false, check_physical_closure=false)
        energies = sort!(Float64.(eigvals(Hermitian(0.5 .* (realized.matrix .+ adjoint(realized.matrix))))))
        for _ in 1:plan.multiplicities[i]
            append!(sector_energies, energies)
        end
    end
    sort!(sector_energies)

    length(sector_energies) == length(full_energies) || throw(DimensionMismatch("sector spectrum length does not reconstruct full spectrum"))
    max_error = maximum(abs.(sector_energies .- full_energies))
    return Dict{Symbol,Any}(
        :passed => max_error <= tol,
        :dimension => plan.total_dimension,
        :sectors_evaluated => length(plan.sectors),
        :equivalent_sector_pairing => plan.equivalence_used,
        :max_spectrum_error => max_error,
    )
end

function validate_phonon_modes(result::PhononModeResult; tol::Real=1e-9)
    D = result.metadata[:dynamical_matrix]
    omega2 = result.metadata[:omega_squared]
    V = result.modes
    residuals = Float64[]
    for j in eachindex(omega2)
        v = view(V, :, j)
        push!(residuals, norm(D * v - omega2[j] * v) / max(norm(v), 1.0))
    end
    worst = maximum(residuals)
    return Dict{Symbol,Any}(:passed => worst <= tol, :max_residual => worst, :residuals => residuals)
end

function validate_phonon_dispersion(result::PhononDispersionResult; tol::Real=1e-9)
    nq, nbranch = size(result.frequencies)
    size(result.modes) == (nbranch, nbranch, nq) || throw(DimensionMismatch("phonon dispersion mode array has inconsistent dimensions"))
    residuals = Float64[]
    hermiticity_errors = Float64[]
    stable = true

    for iq in 1:nq
        Dq = dynamical_matrix(result.model, result.qpoints[iq])
        Dnorm = max(norm(Dq), 1.0)
        hermitian_error = norm(Dq - adjoint(Dq)) / Dnorm
        push!(hermiticity_errors, hermitian_error)
        for branch in 1:nbranch
            omega = result.frequencies[iq, branch]
            stable &= omega >= -tol && isfinite(omega)
            v = view(result.modes, :, branch, iq)
            vnorm = norm(v)
            scale = max(Dnorm * vnorm, abs2(omega) * vnorm, 1.0)
            push!(residuals, norm(Dq * v - omega^2 * v) / scale)
        end
    end

    max_residual = isempty(residuals) ? 0.0 : maximum(residuals)
    max_hermiticity_error = isempty(hermiticity_errors) ? 0.0 : maximum(hermiticity_errors)
    passed = stable && max_residual <= tol && max_hermiticity_error <= tol
    return Dict{Symbol,Any}(
        :passed => passed,
        :stable => stable,
        :max_residual => max_residual,
        :max_hermiticity_error => max_hermiticity_error,
        :residuals => residuals,
        :hermiticity_errors => hermiticity_errors,
    )
end

function validate_mode_result(result::QuadraticModeResult; tol::Real=1e-9)
    finite = all(isfinite, result.energies)
    ordered = issorted(real.(result.energies))
    positive_if_bdg = get(result.metadata, :pairing, false) ? all(>(-tol), real.(result.energies)) : true
    return Dict{Symbol,Any}(:passed => finite && ordered && positive_if_bdg, :finite => finite, :ordered => ordered, :stable_positive_modes => positive_if_bdg)
end

function _spinwave_mode_metric_error(result::SpinWaveResult)
    N = size(result.A, 1)
    modes = result.modes
    M = size(modes, 2)
    if size(modes, 1) == N
        gram = adjoint(modes) * modes
    elseif size(modes, 1) == 2N
        weighted = copy(modes)
        @views weighted[N+1:2N, :] .*= -1
        gram = adjoint(modes) * weighted
    else
        return Inf
    end
    error = 0.0
    @inbounds for mu in 1:M, nu in 1:M
        target = mu == nu ? 1.0 : 0.0
        error = max(error, abs(gram[mu, nu] - target))
    end
    return error
end

function validate_spinwave_result(result::SpinWaveResult; tol::Real=1e-9)
    finite = all(isfinite, result.frequencies)
    stable = all(omega -> omega >= -tol, result.frequencies)
    hermitian_A = norm(result.A - adjoint(result.A)) <= tol * max(norm(result.A), 1.0)
    symmetric_B = norm(result.B - transpose(result.B)) <= tol * max(norm(result.B), 1.0)
    mode_metric_error = _spinwave_mode_metric_error(result)
    canonical_modes = mode_metric_error <= max(10 * tol, 1e-10)
    return Dict{Symbol,Any}(:passed => finite && stable && hermitian_A && symmetric_B && canonical_modes, :finite => finite, :stable => stable, :A_hermitian => hermitian_A, :B_symmetric => symmetric_B, :canonical_modes => canonical_modes, :mode_metric_error => mode_metric_error)
end
