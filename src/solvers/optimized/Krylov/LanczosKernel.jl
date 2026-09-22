# Shared allocation-conscious Hermitian Lanczos kernel.

@inline function _optimized_mul!(y, A, x)
    mul!(y, A, x)
    return y
end

function _optimized_hermitian_check(
    A,
    dim::Int;
    seed::Int=0,
    trials::Int=3,
    tol::Real=1e-9,
)
    rng = MersenneTwister(seed)
    Ax = zeros(ComplexF64, dim)
    Ay = zeros(ComplexF64, dim)
    worst = 0.0
    for _ in 1:trials
        x = randn(rng, ComplexF64, dim)
        y = randn(rng, ComplexF64, dim)
        _optimized_mul!(Ax, A, x)
        _optimized_mul!(Ay, A, y)
        lhs = dot(x, Ay)
        rhs = dot(Ax, y)
        err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
        worst = max(worst, err)
    end
    return worst <= tol, worst
end

function _optimized_lanczos_decomposition(
    A,
    dim::Int;
    krylov_dim::Int,
    tol::Real,
    seed::Int=0,
    reorthogonalize::Bool=true,
    initial_vector=nothing,
)
    mmax = min(dim, krylov_dim)
    mmax > 0 || throw(ArgumentError("empty Krylov space"))

    q = if isnothing(initial_vector)
        rng = MersenneTwister(seed)
        randn(rng, ComplexF64, dim)
    else
        ComplexF64.(collect(initial_vector))
    end
    length(q) == dim || throw(DimensionMismatch("initial Krylov vector has wrong length"))
    qnorm = norm(q)
    qnorm > 0 || throw(ArgumentError("initial Krylov vector has zero norm"))
    q ./= qnorm

    qprev = zeros(ComplexF64, dim)
    z = zeros(ComplexF64, dim)
    Q = zeros(ComplexF64, dim, mmax)
    alpha = zeros(Float64, mmax)
    beta = zeros(Float64, max(mmax - 1, 0))

    actual = 0
    tail_beta = 0.0

    for j in 1:mmax
        actual = j
        @views Q[:, j] .= q
        _optimized_mul!(z, A, q)
        if j > 1
            z .-= beta[j - 1] .* qprev
        end

        aj = real(dot(q, z))
        alpha[j] = aj
        z .-= aj .* q

        if reorthogonalize
            # Two MGS passes are intentionally retained for robust small/medium
            # sector calculations and reproducible validation.
            for _ in 1:2
                for k in 1:j
                    qk = view(Q, :, k)
                    z .-= dot(qk, z) .* qk
                end
            end
        end

        bj = norm(z)
        tail_beta = bj
        if bj <= tol || j == mmax
            break
        end
        beta[j] = bj
        qprev .= q
        q .= z ./ bj
    end

    return KrylovDecomposition(
        alpha[1:actual],
        beta[1:max(actual - 1, 0)],
        Q[:, 1:actual],
        tail_beta,
        actual,
    )
end

function _optimized_ritz(
    dec::KrylovDecomposition,
    nev::Int;
    tol::Real,
)
    T = SymTridiagonal(dec.alpha, dec.beta)
    F = eigen(T)
    order = sortperm(F.values)
    take = order[1:min(nev, length(order))]
    energies = Float64.(F.values[take])
    Y = F.vectors[:, take]
    right = dec.basis * Y

    residuals = if dec.iterations == size(dec.basis, 2)
        abs.(dec.tail_beta .* Y[end, :])
    else
        zeros(Float64, length(take))
    end
    return energies, right, residuals
end
