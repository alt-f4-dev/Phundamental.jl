"""
    continued_fraction(A, seed; krylov_dim=128, tol=1e-12, ...)

Generate the Lanczos continued-fraction coefficients for
`<seed|(z-H)^(-1)|seed>`.
"""
function continued_fraction(
    A,
    seed;
    krylov_dim::Integer=128,
    tol::Real=1e-12,
    reorthogonalize::Bool=true,
)
    v = ComplexF64.(collect(seed))
    n2 = real(dot(v, v))
    n2 > 0 || throw(ArgumentError("continued-fraction seed has zero norm"))
    dec = _optimized_lanczos_decomposition(
        A,
        length(v);
        krylov_dim=Int(krylov_dim),
        tol=tol,
        initial_vector=v,
        reorthogonalize=reorthogonalize,
    )
    return ContinuedFractionResult(
        dec.alpha,
        dec.beta,
        n2,
        Dict{Symbol,Any}(
            :iterations => dec.iterations,
            :tail_beta => dec.tail_beta,
        ),
    )
end

function continued_fraction_value(cf::ContinuedFractionResult, z::Number)
    n = length(cf.alpha)
    n > 0 || return zero(ComplexF64)
    g = inv(ComplexF64(z) - cf.alpha[end])
    for j in (n - 1):-1:1
        bj = cf.beta[j]
        g = inv(ComplexF64(z) - cf.alpha[j] - bj^2 * g)
    end
    return cf.norm2 * g
end

"""
    continued_fraction_spectrum(cf, axis; eta)

Return `-Im G(ω+iη)/π`.
"""
function continued_fraction_spectrum(
    cf::ContinuedFractionResult,
    axis;
    eta::Real,
)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    return [
        -imag(continued_fraction_value(cf, ComplexF64(w, eta))) / pi
        for w in axis
    ]
end
