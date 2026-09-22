# Canonical TPQ / stochastic Lanczos trace sampling.

struct _ThermalQuadrature
    energies::Vector{Float64}
    overlaps::Vector{Float64}
    iterations::Int
    residual_beta::Float64
    convergence_assessed::Bool
    converged::Bool
    convergence_error::Float64
    convergence_checks::Int
end

@inline _thermal_samples(method::ThermalTypicality) = method.samples
@inline _thermal_samples(method::ThermalLanczos) = method.samples
@inline _thermal_samples(method::CanonicalTPQ) = method.samples

@inline _thermal_krylov_dim(method::ThermalTypicality) = method.krylov_dim
@inline _thermal_krylov_dim(method::ThermalLanczos) = method.krylov_dim
@inline _thermal_krylov_dim(method::CanonicalTPQ) = method.krylov_dim

@inline _thermal_breakdown_tol(method::ThermalTypicality) = method.tol
@inline _thermal_breakdown_tol(method::ThermalLanczos) = method.tol
@inline _thermal_breakdown_tol(method::CanonicalTPQ) = method.breakdown_tol

@inline _thermal_matrix_free(method::ThermalTypicality) = method.matrix_free
@inline _thermal_matrix_free(method::ThermalLanczos) = method.matrix_free
@inline _thermal_matrix_free(method::CanonicalTPQ) = method.matrix_free

@inline _thermal_store_vectors(method::ThermalTypicality) = method.store_vectors
@inline _thermal_store_vectors(::ThermalLanczos) = false
@inline _thermal_store_vectors(method::CanonicalTPQ) = method.store_vectors

@inline _thermal_reorthogonalize(method::ThermalTypicality) = method.reorthogonalize
@inline _thermal_reorthogonalize(method::ThermalLanczos) = method.reorthogonalize
@inline _thermal_reorthogonalize(method::CanonicalTPQ) = method.reorthogonalize

@inline _thermal_hbar(method::ThermalTypicality) = method.hbar
@inline _thermal_hbar(method::ThermalLanczos) = method.hbar
@inline _thermal_hbar(method::CanonicalTPQ) = method.hbar

@inline _thermal_parallel(::Union{ThermalTypicality,ThermalLanczos}) = false
@inline _thermal_parallel(method::CanonicalTPQ) = method.parallel

@inline _thermal_progress(::Union{ThermalTypicality,ThermalLanczos}) = false
@inline _thermal_progress(method::CanonicalTPQ) = method.progress

@inline _thermal_max_basis_dimension(::Union{ThermalTypicality,ThermalLanczos}) = typemax(Int)
@inline _thermal_max_basis_dimension(method::CanonicalTPQ) = method.max_basis_dimension

@inline _thermal_decomposition(::Union{ThermalTypicality,ThermalLanczos}) = nothing
@inline _thermal_decomposition(method::CanonicalTPQ) = method.decomposition

@inline _thermal_adaptive_krylov(::Union{ThermalTypicality,ThermalLanczos}) = false
@inline _thermal_adaptive_krylov(method::CanonicalTPQ) = method.adaptive_krylov

@inline _thermal_min_krylov_dim(method::Union{ThermalTypicality,ThermalLanczos}) = _thermal_krylov_dim(method)
@inline _thermal_min_krylov_dim(method::CanonicalTPQ) = method.min_krylov_dim

@inline _thermal_convergence_tol(::Union{ThermalTypicality,ThermalLanczos}) = NaN
@inline _thermal_convergence_tol(method::CanonicalTPQ) = method.convergence_tol

@inline _thermal_convergence_check_interval(::Union{ThermalTypicality,ThermalLanczos}) = 0
@inline _thermal_convergence_check_interval(method::CanonicalTPQ) = method.convergence_check_interval

@inline _thermal_convergence_consecutive(::Union{ThermalTypicality,ThermalLanczos}) = 0
@inline _thermal_convergence_consecutive(method::CanonicalTPQ) = method.convergence_consecutive

@inline _thermal_method_symbol(::ThermalTypicality) = :thermal_typicality
@inline _thermal_method_symbol(::ThermalLanczos) = :thermal_lanczos
@inline _thermal_method_symbol(::CanonicalTPQ) = :canonical_tpq

@inline function _thermal_mul!(y, A, x)
    mul!(y, A, x)
    return y
end

function _sample_method_seed(method::ThermalTypicality, seed::Integer)
    return ThermalTypicality(samples=method.samples, krylov_dim=method.krylov_dim, tol=method.tol, seed=seed, matrix_free=method.matrix_free,
                             store_vectors=method.store_vectors, reorthogonalize=method.reorthogonalize, hbar=method.hbar)
end

function _sample_method_seed(method::ThermalLanczos, seed::Integer)
    return ThermalLanczos(samples=method.samples, krylov_dim=method.krylov_dim, tol=method.tol, seed=seed, matrix_free=method.matrix_free,
                          reorthogonalize=method.reorthogonalize, hbar=method.hbar)
end

function _sample_method_seed(method::CanonicalTPQ, seed::Integer)
    return CanonicalTPQ(samples=method.samples, krylov_dim=method.krylov_dim, min_krylov_dim=method.min_krylov_dim,
                        breakdown_tol=method.breakdown_tol, adaptive_krylov=method.adaptive_krylov, convergence_tol=method.convergence_tol,
                        convergence_check_interval=method.convergence_check_interval, convergence_consecutive=method.convergence_consecutive,
                        seed=seed, matrix_free=method.matrix_free, store_vectors=method.store_vectors, reorthogonalize=method.reorthogonalize,
                        hbar=method.hbar, propagator=method.propagator, parallel=method.parallel, progress=method.progress,
                        max_basis_dimension=method.max_basis_dimension, decomposition=method.decomposition)
end

_sample_method_seed(method::ExactGibbs, seed::Integer) = method

function _optimized_thermal_operator(model::ManyBodyModel, method; materialize_basis::Bool)
    isdefined(Solvers, :optimized_matrix_free_operator) || return nothing
    try
        return Solvers.optimized_matrix_free_operator(model; hbar=_thermal_hbar(method),
                                                      max_basis_dimension=_thermal_max_basis_dimension(method), materialize_basis=materialize_basis)
    catch err
        if err isa ArgumentError || err isa DimensionMismatch
            return nothing
        end
        rethrow()
    end
end

function _randomized_hermiticity_error(A, dim::Int, seed::Int)
    rng = MersenneTwister(seed)
    Ax = zeros(ComplexF64, dim)
    Ay = zeros(ComplexF64, dim)
    worst = 0.0
    for _ in 1:min(3, dim)
        x = randn(rng, ComplexF64, dim)
        y = randn(rng, ComplexF64, dim)
        _thermal_mul!(Ax, A, x)
        _thermal_mul!(Ay, A, y)
        lhs = dot(x, Ay)
        rhs = dot(Ax, y)
        err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
        worst = max(worst, err)
    end
    return worst
end

"""
    thermal_workspace(model, method; require_basis=false)

Construct a reusable Hamiltonian workspace for stochastic thermodynamics. For supported packed spin-1/2 models and `require_basis=false`, the Hilbert space remains basis-implicit: only the dimension and packed operator are retained.
"""
function thermal_workspace(model::ManyBodyModel, method::Union{ThermalTypicality,ThermalLanczos,CanonicalTPQ}; require_basis::Bool=false)
    if method isa CanonicalTPQ && !isnothing(method.decomposition)
        require_basis && throw(ArgumentError("sector-decomposed CanonicalTPQ does not materialize a global ComputationalBasis"))
        return _sector_thermal_workspace(model, method)
    end

    breakdown_tol = _thermal_breakdown_tol(method)

    if _thermal_matrix_free(method)
        packed = _optimized_thermal_operator(model, method; materialize_basis=require_basis)
        if isnothing(packed)
            model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("matrix-free thermal sampling requires a symbolic Hamiltonian"))
            basis = computational_basis(model)
            A = SymbolicLinearOperator(model.hamiltonian, basis; hbar=_thermal_hbar(method))
            dim = length(basis)
            backend = :generic_symbolic
            certified = false
        else
            A = packed.operator
            basis = packed.basis
            dim = packed.dimension
            backend = packed.backend
            certified = packed.hermitian_certified
        end
    else
        basis = computational_basis(model)
        realized = realize(model.hamiltonian, basis; hbar=_thermal_hbar(method), sparse=true, check_physical_closure=true, closure_tol=max(breakdown_tol, 1e-10))
        A = realized.matrix
        dim = length(basis)
        backend = :sparse_matrix
        certified = false
    end

    herr = certified ? 0.0 : _randomized_hermiticity_error(A, dim, method.seed + 104729)
    herr <= max(10 * breakdown_tol, 1e-9) || throw(ArgumentError("thermal Lanczos sampling requires a Hermitian Hamiltonian; estimated relative error is $herr"))
    return ThermalWorkspace(model, A, basis, dim, backend, herr, certified, _thermal_hbar(method))
end

function _thermal_quadrature_signature(alpha::Vector{Float64}, beta::Vector{Float64}, m::Int, convergence_betas::Vector{Float64})
    diagonal = view(alpha, 1:m)
    offdiagonal = view(beta, 1:(m - 1))
    T = SymTridiagonal(diagonal, offdiagonal)

    F = eigen(T)
    energies = Float64.(F.values)
    overlaps = abs2.(F.vectors[1, :])
    emin = minimum(energies)
    logz = Vector{Float64}(undef, length(convergence_betas))
    u = similar(logz)
    h2 = similar(logz)
    cv_over_kb = similar(logz)

    @inbounds for i in eachindex(convergence_betas)
        beta_value = convergence_betas[i]
        z = 0.0
        uvalue = 0.0
        h2value = 0.0

        for j in eachindex(energies)
            energy = energies[j]
            weight = overlaps[j] * exp(-beta_value * (energy - emin))
            z += weight
            uvalue += weight * energy
            h2value += weight * energy^2
        end

        z > 0 && isfinite(z) || throw(ArgumentError("adaptive thermal Krylov checkpoint produced a non-finite partition estimate"))

        logz[i] = -beta_value * emin + log(z)
        u[i] = uvalue / z
        h2[i] = h2value / z
        cv_over_kb[i] = beta_value^2 * max(h2[i] - u[i]^2, 0.0)
    end

    return (logz=logz, internal_energy=u, second_moment=h2, heat_capacity_over_kb=cv_over_kb)
end

function _thermal_quadrature_signature_error(current, previous)
    error = 0.0
    @inbounds for i in eachindex(current.logz)
        error = max(error, abs(current.logz[i] - previous.logz[i]) / max(1.0, abs(current.logz[i])))
        error = max(error, abs(current.internal_energy[i] - previous.internal_energy[i]) / max(1.0, abs(current.internal_energy[i])))
        error = max(error, abs(current.second_moment[i] - previous.second_moment[i]) / max(1.0, abs(current.second_moment[i])))
        error = max(error, abs(current.heat_capacity_over_kb[i] - previous.heat_capacity_over_kb[i]) / max(1.0, abs(current.heat_capacity_over_kb[i])))
    end
    return error
end

function _thermal_lanczos(A, q1::Vector{ComplexF64}, krylov_dim::Int, breakdown_tol::Float64; reorthogonalize::Bool, store_basis::Bool,
                          adaptive::Bool=false, min_krylov_dim::Int=krylov_dim, convergence_tol::Float64=1e-8,
                          convergence_check_interval::Int=8, convergence_consecutive::Int=2,
                          convergence_betas::Union{Nothing,Vector{Float64}}=nothing)
    dim = length(q1)
    mmax = min(dim, krylov_dim)
    q = q1
    qnorm = norm(q)
    qnorm > 0 || throw(ArgumentError("initial thermal Krylov vector had zero norm"))
    q ./= qnorm
    qprev = zeros(ComplexF64, dim)
    z = zeros(ComplexF64, dim)
    alpha = zeros(Float64, mmax)
    beta = zeros(Float64, mmax)

    keepQ = store_basis || reorthogonalize
    Q = keepQ ? zeros(ComplexF64, dim, mmax) : nothing
    projection_coefficients = reorthogonalize ? zeros(ComplexF64, mmax) : nothing

    adaptive && isnothing(convergence_betas) && throw(ArgumentError("adaptive thermal Krylov requires the requested temperature grid"))
    effective_min = min(mmax, max(2, min_krylov_dim))
    previous_signature = nothing
    convergence_error = NaN
    convergence_checks = 0
    consecutive_passes = 0
    converged = false
    actual = 0

    for j in 1:mmax
        actual = j
        keepQ && (@views Q[:, j] .= q)

        _thermal_mul!(z, A, q)
        j > 1 && (@. z -= beta[j - 1] * qprev)

        alpha_j = real(dot(q, z))
        alpha[j] = alpha_j
        @. z -= alpha_j * q

        if reorthogonalize
            Qj = view(Q, :, 1:j)
            cj = view(projection_coefficients, 1:j)
            for _ in 1:2
                mul!(cj, adjoint(Qj), z)
                mul!(z, Qj, cj, -1.0 + 0.0im, 1.0 + 0.0im)
            end
        end

        beta_j = norm(z)
        beta[j] = beta_j
        breakdown = beta_j <= breakdown_tol || (j == mmax && mmax == dim)

        if adaptive && j >= effective_min && ((j - effective_min) % convergence_check_interval == 0 || breakdown || j == mmax)
            signature = _thermal_quadrature_signature(alpha, beta, j, convergence_betas)
            convergence_checks += 1
            if isnothing(previous_signature)
                previous_signature = signature
            else
                convergence_error = _thermal_quadrature_signature_error(signature, previous_signature)
                if convergence_error <= convergence_tol
                    consecutive_passes += 1
                else
                    consecutive_passes = 0
                end
                previous_signature = signature
                if consecutive_passes >= convergence_consecutive
                    converged = true
                end
            end
        end

        if breakdown
            adaptive && (converged = true)
            break
        elseif adaptive && converged
            break
        elseif j == mmax
            break
        end

        qprev, q, z = q, z, qprev
        q ./= beta_j
    end

    T = SymTridiagonal(alpha[1:actual], actual > 1 ? beta[1:(actual - 1)] : Float64[])
    F = eigen(T)
    energies = Float64.(F.values)
    overlaps = abs2.(F.vectors[1, :])
    residual_beta = beta[actual]
    Qout = store_basis ? Q[:, 1:actual] : nothing
    return energies, overlaps, F.vectors, Qout, actual, residual_beta, adaptive, converged, convergence_error, convergence_checks
end

function _random_unit_vector(rng::AbstractRNG, dim::Int)
    q = randn(rng, ComplexF64, dim)
    nrm = norm(q)
    nrm > 0 || throw(ArgumentError("random thermal vector had zero norm"))
    q ./= nrm
    return q
end

@inline function _canonical_tpq_sample_seed(seed::Int, sample::Int)
    modulus = Int128(typemax(Int))
    return Int(mod(Int128(seed) + Int128(104729) * Int128(sample - 1), modulus))
end

function _thermal_convergence_betas(method, temperatures, kB::Real)
    _thermal_adaptive_krylov(method) || return nothing
    isnothing(temperatures) && throw(ArgumentError("adaptive CanonicalTPQ requires a temperature grid"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("adaptive CanonicalTPQ requires a nonempty temperature grid"))
    all(>(0), Ts) || throw(ArgumentError("adaptive CanonicalTPQ requires positive temperatures"))
    return inv.(Float64(kB) .* Ts)
end

function _sample_quadrature(A, dim::Int, method, rng::AbstractRNG; store_basis::Bool, convergence_betas=nothing)
    q = _random_unit_vector(rng, dim)
    energies, overlaps, V, Q, iterations, beta_tail, assessed, converged, convergence_error, convergence_checks = _thermal_lanczos(
        A,
        q,
        _thermal_krylov_dim(method),
        _thermal_breakdown_tol(method);
        reorthogonalize=_thermal_reorthogonalize(method),
        store_basis=store_basis,
        adaptive=_thermal_adaptive_krylov(method),
        min_krylov_dim=_thermal_min_krylov_dim(method),
        convergence_tol=_thermal_convergence_tol(method),
        convergence_check_interval=max(_thermal_convergence_check_interval(method), 1),
        convergence_consecutive=max(_thermal_convergence_consecutive(method), 1),
        convergence_betas=convergence_betas,
    )
    return _ThermalQuadrature(energies, overlaps, iterations, beta_tail, assessed, converged, convergence_error, convergence_checks), Q, store_basis ? Matrix(V) : nothing
end

function _trace_quadratures(model::ManyBodyModel, method::Union{ThermalTypicality,ThermalLanczos,CanonicalTPQ}; store_bases::Bool=false,
                            workspace=nothing, temperatures=nothing, kB::Real=1.0)
    workspace isa SectorThermalWorkspace && throw(ArgumentError("sector thermal workspaces are handled by the sector-decomposed CanonicalTPQ path"))
    require_basis = store_bases
    if !isnothing(workspace)
        workspace.model === model || throw(ArgumentError("thermal workspace belongs to a different model"))
        workspace.hbar == _thermal_hbar(method) || throw(ArgumentError("thermal workspace hbar does not match the requested thermal method"))
    end
    ws = if isnothing(workspace) || (require_basis && isnothing(workspace.basis))
        thermal_workspace(model, method; require_basis=require_basis)
    else
        workspace
    end
    dim = ws.dimension
    R = _thermal_samples(method)
    convergence_betas = _thermal_convergence_betas(method, temperatures, kB)

    quadratures = Vector{_ThermalQuadrature}(undef, R)
    bases = store_bases ? Vector{Matrix{ComplexF64}}(undef, R) : nothing
    eigvecs = store_bases ? Vector{Matrix{Float64}}(undef, R) : nothing

    if method isa CanonicalTPQ
        progress_lock = ReentrantLock()
        run_sample = function (r)
            rng = MersenneTwister(_canonical_tpq_sample_seed(method.seed, r))
            qd, Q, V = _sample_quadrature(ws.operator, dim, method, rng; store_basis=store_bases, convergence_betas=convergence_betas)
            quadratures[r] = qd
            if store_bases
                bases[r] = Q
                eigvecs[r] = V
            end
            if _thermal_progress(method)
                lock(progress_lock) do
                    println("CanonicalTPQ realization $r/$R complete")
                    flush(stdout)
                end
            end
            return nothing
        end

        if _thermal_parallel(method) && Threads.nthreads() > 1 && R > 1
            Threads.@threads for r in 1:R
                run_sample(r)
            end
        else
            for r in 1:R
                run_sample(r)
            end
        end
    else
        rng = MersenneTwister(method.seed)
        for r in 1:R
            qd, Q, V = _sample_quadrature(ws.operator, dim, method, rng; store_basis=store_bases, convergence_betas=convergence_betas)
            quadratures[r] = qd
            if store_bases
                bases[r] = Q
                eigvecs[r] = V
            end
        end
    end

    return ws, quadratures, bases, eigvecs
end

function _sem(x::AbstractVector{<:Real})
    n = length(x)
    n <= 1 && return NaN
    return std(x; corrected=true) / sqrt(n)
end

function _thermodynamic_from_sums(zsum::Float64, usum::Float64, h2sum::Float64, nsamples::Int, dim::Int, β::Float64, T::Float64, kb::Float64, eref::Float64)
    zbar = zsum / nsamples
    zbar > 0 && isfinite(zbar) || throw(ArgumentError("sampled partition estimator vanished or became non-finite"))
    logZ = log(Float64(dim)) - β * eref + log(zbar)
    U = (usum / nsamples) / zbar
    E2 = (h2sum / nsamples) / zbar
    varE = max(E2 - U^2, 0.0)
    F = -kb * T * logZ
    S = (U - F) / T
    Cv = varE / (kb * T^2)
    return (logZ=logZ, free_energy=F, internal_energy=U, entropy=S, heat_capacity=Cv, energy_variance=varE)
end

function _jackknife_stderr(values::Vector{Float64})
    n = length(values)
    n <= 1 && return NaN
    center = mean(values)
    return sqrt((n - 1) / n * sum(abs2, values .- center))
end

function _jackknife_thermodynamic_errors(z::Vector{Float64}, u::Vector{Float64}, h2::Vector{Float64}, dim::Int, β::Float64, T::Float64, kb::Float64, eref::Float64)
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
        θ = _thermodynamic_from_sums(zsum - z[r], usum - u[r], h2sum - h2[r], n - 1, dim, β, T, kb, eref)
        logZ[r] = θ.logZ
        F[r] = θ.free_energy
        U[r] = θ.internal_energy
        S[r] = θ.entropy
        Cv[r] = θ.heat_capacity
    end

    return Dict{Symbol,Float64}(
        :logZ => _jackknife_stderr(logZ),
        :free_energy => _jackknife_stderr(F),
        :internal_energy => _jackknife_stderr(U),
        :entropy => _jackknife_stderr(S),
        :heat_capacity => _jackknife_stderr(Cv),
    )
end

function _sampled_point(quadratures::Vector{_ThermalQuadrature}, dim::Int, temperature::Real, kB::Real)
    temperature > 0 || throw(ArgumentError("stochastic thermal sampling currently requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))

    T = Float64(temperature)
    kb = Float64(kB)
    β = inv(kb * T)
    eref = minimum(minimum(q.energies) for q in quadratures)
    z = zeros(Float64, length(quadratures))
    u = similar(z)
    h2 = similar(z)

    for (r, q) in enumerate(quadratures)
        w = q.overlaps .* exp.(-β .* (q.energies .- eref))
        z[r] = sum(w)
        u[r] = dot(w, q.energies)
        h2[r] = dot(w, q.energies .* q.energies)
    end

    θ = _thermodynamic_from_sums(sum(z), sum(u), sum(h2), length(z), dim, β, T, kb, eref)
    stderr = _jackknife_thermodynamic_errors(z, u, h2, dim, β, T, kb, eref)
    point = ThermodynamicPoint(
        T, kb, θ.logZ, θ.free_energy, θ.internal_energy, θ.entropy, θ.heat_capacity,
        stderr,
        Dict{Symbol,Any}(
            :method => :canonical_tpq_trace,
            :samples => length(quadratures),
            :dimension => dim,
            :energy_reference => eref,
            :energy_variance => θ.energy_variance,
            :uncertainty_method => length(quadratures) > 1 ? :delete_one_jackknife : :unavailable_single_realization,
        ),
    )
    return point, eref, z
end

function _filtered_sample_vectors(quadratures, bases, eigvecs, temperature::Float64, kB::Float64)
    β = inv(kB * temperature)
    R = length(quadratures)
    dim = size(bases[1], 1)
    Vthermal = zeros(ComplexF64, dim, R)
    norms = zeros(Float64, R)
    minima = [minimum(q.energies) for q in quadratures]
    eref = minimum(minima)
    scales = exp.(-β .* (minima .- eref))

    for r in 1:R
        q = quadratures[r]
        Q = bases[r]
        U = eigvecs[r]
        emin = minima[r]
        e1 = zeros(Float64, size(U, 1))
        e1[1] = 1.0
        coeff = U * (exp.(-0.5β .* (q.energies .- emin)) .* (transpose(U) * e1))
        ψ = Q * coeff
        @views Vthermal[:, r] .= ψ
        norms[r] = real(dot(ψ, ψ))
    end

    return Vthermal, norms, scales, eref
end

function _sampled_state_metadata(method, ws::ThermalWorkspace, quadratures; filtered_vectors_retained::Bool)
    return Dict{Symbol,Any}(
        :method => _thermal_method_symbol(method),
        :physical_method => :canonical_tpq,
        :propagator => method isa CanonicalTPQ ? method.propagator : :krylov,
        :decomposition => nothing,
        :samples => _thermal_samples(method),
        :krylov_dim => _thermal_krylov_dim(method),
        :min_krylov_dim => _thermal_min_krylov_dim(method),
        :breakdown_tol => _thermal_breakdown_tol(method),
        :adaptive_krylov => _thermal_adaptive_krylov(method),
        :convergence_tol => _thermal_convergence_tol(method),
        :convergence_check_interval => _thermal_convergence_check_interval(method),
        :convergence_consecutive => _thermal_convergence_consecutive(method),
        :matrix_free => _thermal_matrix_free(method),
        :hermiticity_error => ws.hermiticity_error,
        :hermitian_certified => ws.hermitian_certified,
        :operator_backend => ws.backend,
        :basis_materialized => !isnothing(ws.basis),
        :filtered_vectors_retained => filtered_vectors_retained,
        :quadrature_iterations => [q.iterations for q in quadratures],
        :quadrature_tail_beta => [q.residual_beta for q in quadratures],
        :quadrature_convergence_assessed => [q.convergence_assessed for q in quadratures],
        :quadrature_converged => [q.converged for q in quadratures],
        :quadrature_convergence_error => [q.convergence_error for q in quadratures],
        :quadrature_convergence_checks => [q.convergence_checks for q in quadratures],
        :adaptive_krylov_converged => _thermal_adaptive_krylov(method) ? all(q.converged for q in quadratures) : nothing,
        :parallel_realizations => _thermal_parallel(method),
    )
end

function thermal_state(model::ManyBodyModel, method::ThermalLanczos; temperature::Real, kB::Real=1.0, workspace=nothing)
    temperature > 0 || throw(ArgumentError("ThermalLanczos currently requires temperature > 0"))
    ws, quadratures, _, _ = _trace_quadratures(model, method; store_bases=false, workspace=workspace, temperatures=[temperature], kB=kB)
    point, _, _ = _sampled_point(quadratures, ws.dimension, temperature, kB)
    beta = inv(Float64(kB) * Float64(temperature))
    metadata = _sampled_state_metadata(method, ws, quadratures; filtered_vectors_retained=false)
    return SampledThermalState(model, ws.basis, method, Float64(temperature), Float64(kB), beta, nothing, Float64[], Float64[], point, metadata)
end

function thermal_state(model::ManyBodyModel, method::ThermalTypicality; temperature::Real, kB::Real=1.0, workspace=nothing)
    temperature > 0 || throw(ArgumentError("ThermalTypicality currently requires temperature > 0"))
    store = method.store_vectors
    ws, quadratures, bases, eigvecs = _trace_quadratures(model, method; store_bases=store, workspace=workspace, temperatures=[temperature], kB=kB)
    point, _, _ = _sampled_point(quadratures, ws.dimension, temperature, kB)

    if store
        Vthermal, norms, scales, _ = _filtered_sample_vectors(quadratures, bases, eigvecs, Float64(temperature), Float64(kB))
    else
        Vthermal = nothing
        norms = Float64[]
        scales = Float64[]
    end

    beta = inv(Float64(kB) * Float64(temperature))
    metadata = _sampled_state_metadata(method, ws, quadratures; filtered_vectors_retained=store)
    return SampledThermalState(model, ws.basis, method, Float64(temperature), Float64(kB), beta, Vthermal, norms, scales, point, metadata)
end

function thermal_state(model::ManyBodyModel, method::CanonicalTPQ; temperature::Real, kB::Real=1.0, workspace=nothing)
    temperature > 0 || throw(ArgumentError("CanonicalTPQ currently requires temperature > 0"))
    !isnothing(method.decomposition) && throw(ArgumentError("sector-decomposed CanonicalTPQ currently supports thermal_curve only because retained sampled states would require an explicit direct-sum state representation"))
    store = method.store_vectors
    ws, quadratures, bases, eigvecs = _trace_quadratures(model, method; store_bases=store, workspace=workspace, temperatures=[temperature], kB=kB)
    point, _, _ = _sampled_point(quadratures, ws.dimension, temperature, kB)

    if store
        Vthermal, norms, scales, _ = _filtered_sample_vectors(quadratures, bases, eigvecs, Float64(temperature), Float64(kB))
    else
        Vthermal = nothing
        norms = Float64[]
        scales = Float64[]
    end

    beta = inv(Float64(kB) * Float64(temperature))
    metadata = _sampled_state_metadata(method, ws, quadratures; filtered_vectors_retained=store)
    return SampledThermalState(model, ws.basis, method, Float64(temperature), Float64(kB), beta, Vthermal, norms, scales, point, metadata)
end

function thermal_curve(model::ManyBodyModel, method::Union{ThermalTypicality,ThermalLanczos,CanonicalTPQ}, temperatures; kB::Real=1.0, workspace=nothing)
    if method isa CanonicalTPQ && !isnothing(method.decomposition)
        return _sector_thermal_curve(model, method, temperatures; kB=kB, workspace=workspace)
    end

    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperature grid cannot be empty"))
    all(>(0), Ts) || throw(ArgumentError("stochastic thermal curves currently require all temperatures > 0"))

    ws, quadratures, _, _ = _trace_quadratures(model, method; store_bases=false, workspace=workspace, temperatures=Ts, kB=kB)
    points = ThermodynamicPoint[]
    sizehint!(points, length(Ts))
    for T in Ts
        point, _, _ = _sampled_point(quadratures, ws.dimension, T, kB)
        push!(points, point)
    end

    return _curve_from_points(points; metadata=Dict{Symbol,Any}(
        :method => _thermal_method_symbol(method),
        :physical_method => :canonical_tpq,
        :propagator => method isa CanonicalTPQ ? method.propagator : :krylov,
        :samples => _thermal_samples(method),
        :krylov_dim => _thermal_krylov_dim(method),
        :min_krylov_dim => _thermal_min_krylov_dim(method),
        :breakdown_tol => _thermal_breakdown_tol(method),
        :adaptive_krylov => _thermal_adaptive_krylov(method),
        :convergence_tol => _thermal_convergence_tol(method),
        :convergence_check_interval => _thermal_convergence_check_interval(method),
        :convergence_consecutive => _thermal_convergence_consecutive(method),
        :matrix_free => _thermal_matrix_free(method),
        :dimension => ws.dimension,
        :hermiticity_error => ws.hermiticity_error,
        :hermitian_certified => ws.hermitian_certified,
        :operator_backend => ws.backend,
        :basis_materialized => !isnothing(ws.basis),
        :quadratures_reused_across_temperatures => true,
        :uncertainty_method => _thermal_samples(method) > 1 ? :delete_one_jackknife : :unavailable_single_realization,
        :parallel_realizations => _thermal_parallel(method),
        :quadrature_iterations => [q.iterations for q in quadratures],
        :quadrature_tail_beta => [q.residual_beta for q in quadratures],
        :quadrature_convergence_assessed => [q.convergence_assessed for q in quadratures],
        :quadrature_converged => [q.converged for q in quadratures],
        :quadrature_convergence_error => [q.convergence_error for q in quadratures],
        :quadrature_convergence_checks => [q.convergence_checks for q in quadratures],
        :adaptive_krylov_converged => _thermal_adaptive_krylov(method) ? all(q.converged for q in quadratures) : nothing,
    ))
end
