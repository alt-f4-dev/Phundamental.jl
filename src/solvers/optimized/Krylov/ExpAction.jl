"""
    exp_action(A, v, tau; krylov_dim=64, tol=1e-12)

Krylov approximation to `exp(tau*A) * v`. `tau` may be real or complex.
"""
function exp_action(
    A,
    v,
    tau::Number;
    krylov_dim::Integer=64,
    tol::Real=1e-12,
    reorthogonalize::Bool=true,
)
    x = ComplexF64.(collect(v))
    nrm = norm(x)
    nrm > 0 || return zeros(ComplexF64, length(x))

    dec = _optimized_lanczos_decomposition(
        A,
        length(x);
        krylov_dim=Int(krylov_dim),
        tol=tol,
        initial_vector=x,
        reorthogonalize=reorthogonalize,
    )
    T = SymTridiagonal(dec.alpha, dec.beta)
    F = eigen(T)
    e1 = zeros(ComplexF64, length(dec.alpha))
    e1[1] = nrm
    coeff = F.vectors * (exp.(tau .* F.values) .* (transpose(F.vectors) * e1))
    return dec.basis * coeff
end
