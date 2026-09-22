# High-frequency diagnostics and asymptotic moments.

"""
    high_frequency_moment(G; ntail=min(8, length(G)))

Estimate the zeroth fermionic spectral moment from the largest stored Matsubara frequencies using `M₀ ≈ mean(iωₙ G(iωₙ))`.
"""
function high_frequency_moment(G::GreenFunction; ntail::Integer=min(8, length(G)))
    1 <= ntail <= length(G) || throw(ArgumentError("ntail must lie in 1:length(G)"))
    order = sortperm(collect(1:length(G)); by=i -> abs(G.axis[i]), rev=true)
    selected = order[1:Int(ntail)]
    moment = zeros(ComplexF64, matrix_dimension(G), matrix_dimension(G))
    for i in selected
        moment .+= G.axis[i] .* G[i]
    end
    moment ./= length(selected)
    return moment
end

"""
    high_frequency_moments(G; order=2, ntail=max(4 * (order + 1), 16))

Fit the asymptotic expansion `G(z)=Σₖ Mₖ/z^(k+1)` on the largest stored Matsubara frequencies and return `[M₀, …, M_order]`. The fit is performed independently for every matrix element by complex linear least squares.
"""
function high_frequency_moments(G::GreenFunction; order::Integer=2, ntail::Integer=min(length(G), max(4 * (Int(order) + 1), 16)))
    order >= 0 || throw(ArgumentError("order must be nonnegative"))
    order_count = Int(order) + 1
    order_count <= ntail <= length(G) || throw(ArgumentError("ntail must satisfy order + 1 <= ntail <= length(G)"))
    indices = sortperm(collect(1:length(G)); by=i -> abs(G.axis[i]), rev=true)[1:Int(ntail)]
    design = Matrix{ComplexF64}(undef, length(indices), order_count)
    for (row, i) in enumerate(indices)
        inverse_frequency = inv(G.axis[i])
        power = inverse_frequency
        for column in 1:order_count
            design[row, column] = power
            power *= inverse_frequency
        end
    end

    dimension = matrix_dimension(G)
    moments = [zeros(ComplexF64, dimension, dimension) for _ in 1:order_count]
    response = Vector{ComplexF64}(undef, length(indices))
    for a in 1:dimension, b in 1:dimension
        @inbounds for (row, i) in enumerate(indices)
            response[row] = G[a, b, i]
        end
        coefficients = design \ response
        for k in 1:order_count
            moments[k][a, b] = coefficients[k]
        end
    end
    return moments
end

"""Relative deviation of the estimated fermionic zeroth moment from the identity matrix."""
function zeroth_moment_residual(G::GreenFunction; ntail::Integer=min(8, length(G)))
    moment = high_frequency_moment(G; ntail=ntail)
    identity_matrix = Matrix{ComplexF64}(I, size(moment, 1), size(moment, 2))
    return norm(moment - identity_matrix) / max(norm(identity_matrix), eps(Float64))
end

"""
    high_frequency_moment_residual(G, expected, moment_order; fit_order=max(moment_order, 2), ntail=...)

Return the relative residual between a fitted high-frequency moment and an expected matrix or scalar value.
"""
function high_frequency_moment_residual(G::GreenFunction, expected, moment_order::Integer;
                                        fit_order::Integer=max(Int(moment_order), 2),
                                        ntail::Integer=min(length(G), max(4 * (Int(fit_order) + 1), 16)))
    moment_order >= 0 || throw(ArgumentError("moment_order must be nonnegative"))
    fit_order >= moment_order || throw(ArgumentError("fit_order must be at least moment_order"))
    moments = high_frequency_moments(G; order=fit_order, ntail=ntail)
    target = expected isa Number ? fill(ComplexF64(expected), size(moments[Int(moment_order) + 1])) : ComplexF64.(expected)
    size(target) == size(moments[Int(moment_order) + 1]) || throw(DimensionMismatch("expected moment has incompatible dimensions"))
    denominator = max(norm(target), 1.0)
    return norm(moments[Int(moment_order) + 1] - target) / denominator
end
