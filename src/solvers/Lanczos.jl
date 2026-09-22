# Dependency-free Hermitian Lanczos solver with optional matrix-free action.
# This implementation targets low-energy eigenpairs and uses full
# reorthogonalization for numerical robustness.

struct LanczosSolver <: AbstractSolver
    nev::Int
    krylov_dim::Int
    tol::Float64
    matrix_free::Bool
    reorthogonalize::Bool
    seed::Int
    hbar::Float64
end

function LanczosSolver(
    ; nev::Integer=6,
      krylov_dim::Integer=64,
      tol::Real=1e-10,
      matrix_free::Bool=false,
      reorthogonalize::Bool=true,
      seed::Integer=0,
      hbar::Real=1.0,
)
    nev > 0 || throw(ArgumentError("nev must be positive"))
    krylov_dim > 1 || throw(ArgumentError("krylov_dim must exceed one"))
    return LanczosSolver(Int(nev), Int(krylov_dim), Float64(tol), matrix_free,
                         reorthogonalize, Int(seed), Float64(hbar))
end

solver_name(::LanczosSolver) = :lanczos

function _mul_operator!(y, A::AbstractMatrix, x)
    return mul!(y, A, x)
end
function _mul_operator!(y, A::SymbolicLinearOperator, x)
    return mul!(y, A, x)
end

function _random_hermitian_check(A, dim::Int; seed::Int=0, trials::Int=3, tol::Real=1e-9)
    rng = MersenneTwister(seed)
    Ax = zeros(ComplexF64, dim)
    Ay = zeros(ComplexF64, dim)
    worst = 0.0
    for _ in 1:trials
        x = randn(rng, ComplexF64, dim)
        y = randn(rng, ComplexF64, dim)
        _mul_operator!(Ax, A, x)
        _mul_operator!(Ay, A, y)
        lhs = dot(x, Ay)
        rhs = dot(Ax, y)
        err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
        worst = max(worst, err)
    end
    return worst <= tol, worst
end

function _lanczos(A, dim::Int, solver::LanczosSolver)
    mmax = min(dim, max(solver.krylov_dim, solver.nev + 2))
    rng = MersenneTwister(solver.seed)

    q = randn(rng, ComplexF64, dim)
    q ./= norm(q)
    qprev = zeros(ComplexF64, dim)
    z = zeros(ComplexF64, dim)
    Q = zeros(ComplexF64, dim, mmax)
    α = zeros(Float64, mmax)
    β = zeros(Float64, mmax)

    actual = 0
    for j in 1:mmax
        actual = j
        Q[:, j] .= q
        _mul_operator!(z, A, q)
        j > 1 && (z .-= β[j - 1] .* qprev)

        αj = real(dot(q, z))
        α[j] = αj
        z .-= αj .* q

        if solver.reorthogonalize
            # Two modified Gram-Schmidt passes strongly suppress ghost Ritz
            # values for finite precision Hermitian Lanczos.
            for _ in 1:2
                for k in 1:j
                    coeff = dot(view(Q, :, k), z)
                    z .-= coeff .* view(Q, :, k)
                end
            end
        end

        βj = norm(z)
        β[j] = βj
        if βj <= solver.tol || j == mmax
            break
        end
        qprev .= q
        q .= z ./ βj
    end

    T = SymTridiagonal(α[1:actual], β[1:max(actual - 1, 0)])
    F = eigen(T)
    order = sortperm(F.values)
    take = order[1:min(solver.nev, length(order))]

    energies = F.values[take]
    Y = F.vectors[:, take]
    right = Q[:, 1:actual] * Y

    # Ritz residual ||A v - theta v|| = |beta_m * e_m^T y|.
    residuals = if actual < dim && actual <= length(β)
        abs.(β[actual] .* Y[end, :])
    else
        zeros(Float64, length(take))
    end

    return energies, right, residuals, actual
end

function solve(solver::LanczosSolver, model::ManyBodyModel)
    capabilities = numerical_capabilities(model)
    capabilities.numerically_realizable || throw(ArgumentError(
        "LanczosSolver cannot construct the numerical target representation: $(join(capabilities.reasons, "; "))"
    ))
    capabilities.biorthogonal && throw(ArgumentError(
        "LanczosSolver is Hermitian and is not compatible with a biorthogonal/non-Hermitian representation; use ExactDiagonalization(hermitian=:no)"
    ))

    basis = computational_basis(model)
    dim = length(basis)
    solver.nev <= dim || throw(ArgumentError("nev=$(solver.nev) exceeds basis dimension $dim"))

    if dim <= solver.nev + 1
        # Dense ED is exact and cheaper than building a Krylov basis of nearly
        # the full space.
        ed = ExactDiagonalization(hbar=solver.hbar)
        exact = solve(ed, model)
        metadata = copy(exact.metadata)
        metadata[:solver] = :lanczos_ed_fallback
        return SolverResult(solver, model, exact.energies, exact.right_states,
                            exact.left_states, exact.basis, metadata)
    end

    if solver.matrix_free
        model.hamiltonian isa AbstractOperatorExpr ||
            throw(ArgumentError("matrix_free Lanczos requires a symbolic operator expression"))
        A = SymbolicLinearOperator(model.hamiltonian, basis; hbar=solver.hbar)
        hermitian_ok, hermitian_error = _random_hermitian_check(
            A, dim; seed=solver.seed + 17, tol=max(solver.tol * 10, 1e-9)
        )
        matrix_diagnostics = Dict{Symbol,Any}(:matrix_free => true)
    else
        realized = realize(
            model.hamiltonian,
            basis;
            hbar=solver.hbar,
            sparse=true,
            check_physical_closure=true,
            closure_tol=max(solver.tol, 1e-10),
        )
        A = realized.matrix
        herr = norm(A - adjoint(A)) / max(norm(A), eps(Float64))
        hermitian_error = herr
        hermitian_ok = herr <= max(solver.tol * 10, 1e-9)
        matrix_diagnostics = realized.diagnostics
    end

    hermitian_ok || throw(ArgumentError(
        "LanczosSolver requires a Hermitian Hamiltonian; estimated relative error is $hermitian_error"
    ))

    energies, right, residuals, iterations = _lanczos(A, dim, solver)
    metadata = Dict{Symbol,Any}(
        :solver => :lanczos,
        :dimension => dim,
        :hermitian => true,
        :hermiticity_error => hermitian_error,
        :biorthogonal => false,
        :iterations => iterations,
        :ritz_residuals => residuals,
        :converged => residuals .<= solver.tol,
        :matrix_free => solver.matrix_free,
        :matrix_diagnostics => matrix_diagnostics,
    )

    return SolverResult(solver, model, energies, right, right, basis, metadata)
end
