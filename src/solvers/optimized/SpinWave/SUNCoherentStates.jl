# Pure site-resolved SU(2 * S+1) reference states; no intersite entanglement is presumed.

"""Normalized local pure states and their orthonormal tangent bases on CP^(2 * S).

Amplitudes are ordered by descending spin projection m=S,S-1,...,-S. `tangent_bases` may specify one complete local unitary matrix per site, with the normalized reference as its first column. This optional gauge is useful for verifying basis independence.
"""
struct SUNReferenceState
    amplitudes::Vector{Vector{ComplexF64}}
    bases::Vector{Matrix{ComplexF64}}
end

function _sun_complete_basis(z::Vector{ComplexF64}, tol::Real)
    d = length(z)
    basis = Matrix{ComplexF64}(undef, d, d)
    basis[:, 1] = z
    count = 1
    for j in 1:d
        v = zeros(ComplexF64, d)
        v[j] = 1
        for k in 1:count
            v .-= dot(view(basis, :, k), v) .* view(basis, :, k)
        end
        length_v = norm(v)
        length_v <= tol && continue
        count += 1
        basis[:, count] = v ./ length_v
        count == d && break
    end
    count == d || throw(ArgumentError("could not construct a complete local SU(N) basis"))
    return basis
end

function SUNReferenceState(amplitudes::AbstractVector; tangent_bases=nothing, tol::Real=1e-10)
    tol > 0 || throw(ArgumentError("SU(N) reference tolerance must be positive"))
    !isempty(amplitudes) || throw(ArgumentError("SU(N) reference needs at least one site"))
    states = Vector{ComplexF64}[]
    bases = Matrix{ComplexF64}[]
    !isnothing(tangent_bases) && length(tangent_bases) != length(amplitudes) && throw(DimensionMismatch("one tangent basis per site is required"))
    for (i, raw) in enumerate(amplitudes)
        z = ComplexF64.(collect(raw))
        length(z) >= 2 || throw(ArgumentError("site $i must have at least two local spin states"))
        all(isfinite, z) || throw(ArgumentError("site $i has nonfinite amplitudes"))
        abs(norm(z) - 1) <= tol || throw(ArgumentError("SU(N) reference amplitudes at site $i must be normalized"))
        z ./= norm(z)
        basis = if isnothing(tangent_bases)
            _sun_complete_basis(z, tol)
        else
            U = Matrix{ComplexF64}(tangent_bases[i])
            size(U) == (length(z), length(z)) || throw(DimensionMismatch("site $i tangent basis has the wrong dimensions"))
            norm(U[:, 1] - z) <= tol || throw(ArgumentError("site $i tangent basis first column must equal its reference"))
            norm(adjoint(U) * U - I) <= tol * max(length(z), 1) || throw(ArgumentError("site $i tangent basis must be unitary"))
            U
        end
        push!(states, z)
        push!(bases, basis)
    end
    return SUNReferenceState(states, bases)
end

function SUNReferenceState(amplitudes::AbstractMatrix; kwargs...)
    return SUNReferenceState([amplitudes[:, i] for i in axes(amplitudes, 2)]; kwargs...)
end

function _sun_spin_matrices(S::Real)
    twoS = round(Int, 2 * S)
    isapprox(twoS, 2 * S; atol=1e-10, rtol=0) && twoS >= 1 || throw(ArgumentError("SU(N) spin must be a positive integer or half-integer"))
    d = twoS + 1
    plus = zeros(ComplexF64, d, d)
    for j in 2:d
        m = S - (j - 1)
        plus[j-1, j] = sqrt(max(S * (S + 1) - m * (m + 1), 0.0))
    end
    minus = adjoint(plus)
    x = Matrix{ComplexF64}((plus + minus) / 2)
    y = Matrix{ComplexF64}((plus - minus) / (2im))
    z = Matrix{ComplexF64}(Diagonal([S - (j - 1) for j in 1:d]))
    return (x, y, z)
end

function _sun_resolved_reference(ref::SUNReferenceState, spins::AbstractVector)
    length(ref.amplitudes) == length(spins) || throw(DimensionMismatch("SU(N) site count differs from Hamiltonian"))
    for i in eachindex(spins)
        length(ref.amplitudes[i]) == round(Int, 2 * spins[i]) + 1 || throw(DimensionMismatch("SU(N) local dimension at site $i does not equal 2 * S+1"))
    end
    return ref
end

