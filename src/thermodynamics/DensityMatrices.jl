# Explicit density matrices are intentionally optional.

"""
    density_matrix(state::ExactGibbsState; max_dimension=4096)

Construct the exact Gibbs density matrix

    ρ = Σ_n p_n |R_n><L_n|

in the computational basis.  The dimension guard prevents accidental O(D^2)
storage for a thermal state that is already represented more efficiently in
the eigenbasis.
"""
function density_matrix(
    state::ExactGibbsState;
    max_dimension::Integer=4096,
)
    dim = length(state.solution.basis)
    dim <= max_dimension || throw(ArgumentError(
        "explicit density matrix would be $dim×$dim; increase max_dimension " *
        "deliberately or use expectation(state, O) without constructing ρ"
    ))
    R = state.solution.right_states
    L = state.solution.left_states
    return R * Diagonal(state.probabilities) * adjoint(L)
end

"""
    density_matrix(state::SampledThermalState; max_dimension=2048)

Construct the sampled density-matrix estimator

    ρ ≈ Σ_r s_r |ψ_r><ψ_r| / Σ_r s_r <ψ_r|ψ_r>

when thermally filtered vectors were retained.
"""
function density_matrix(
    state::SampledThermalState;
    max_dimension::Integer=2048,
)
    V = state.filtered_vectors
    isnothing(V) && throw(ArgumentError(
        "this sampled state does not retain filtered vectors; use " *
        "ThermalTypicality(store_vectors=true) or CanonicalTPQ(store_vectors=true)"
    ))
    isnothing(state.basis) && throw(ArgumentError("retained sampled vectors require a materialized computational basis"))
    dim = length(state.basis)
    dim <= max_dimension || throw(ArgumentError(
        "sampled density matrix would be $dim×$dim; use expectation(state, O) instead"
    ))

    ρ = zeros(ComplexF64, dim, dim)
    denom = 0.0
    for r in axes(V, 2)
        ψ = view(V, :, r)
        scale = state.sample_scales[r]
        ρ .+= scale .* (ψ * adjoint(ψ))
        denom += scale * state.sample_norms[r]
    end
    denom > 0 || throw(ArgumentError("sampled thermal normalization vanished"))
    ρ ./= denom
    return ρ
end

"""Trace normalization diagnostic for an explicitly constructed density matrix."""
density_matrix_trace_error(ρ::AbstractMatrix) = abs(tr(ρ) - 1)
