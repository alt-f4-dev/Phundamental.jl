# Common numerical result containers.

"""
    SolverResult

Spectral solver result. `right_states[:,i]` is the right eigenvector associated with `energies[i]`. For Hermitian problems `left_states` is the same matrix. For non-Hermitian/biorthogonal problems the columns of `left_states` are normalized so that

    left_states' * right_states ≈ I.
"""
struct SolverResult{S,M,E,R,L,B,D}
    solver::S
    model::M
    energies::E
    right_states::R
    left_states::L
    basis::B
    metadata::D
end

Base.length(result::SolverResult) = length(result.energies)

"""
    EnergySpectrumResult

Complete energy spectrum without retained eigenvectors or a retained computational basis. This is intended for exact thermodynamic calculations that require all eigenvalues but do not require eigenstates.
"""
struct EnergySpectrumResult{S,M,E,D}
    solver::S
    model::M
    energies::E
    dimension::Int
    metadata::D
end

Base.length(result::EnergySpectrumResult) = length(result.energies)

"""Number of returned energy levels or eigenpairs."""
nlevels(result::Union{SolverResult,EnergySpectrumResult}) = length(result)

"""Alias for the right-state matrix."""
states(result::SolverResult) = result.right_states

"""Return eigenstate `i`; choose `side=:left` for a left eigenvector."""
function eigenstate(result::SolverResult, i::Integer; side::Symbol=:right)
    1 <= i <= length(result) || throw(BoundsError(result.energies, i))
    side === :right && return view(result.right_states, :, i)
    side === :left && return view(result.left_states, :, i)
    throw(ArgumentError("side must be :right or :left"))
end

"""Lowest returned energy (results are sorted by the solver)."""
groundenergy(result::Union{SolverResult,EnergySpectrumResult}) = result.energies[1]

"""Lowest right eigenstate."""
groundstate(result::SolverResult) = eigenstate(result, 1)

"""True when distinct left/right eigenvectors were required."""
isbiorthogonal(result::SolverResult) = get(result.metadata, :biorthogonal, false)

"""
    TimeEvolutionResult

Columns of `states` are wavefunctions at the corresponding entries of `times`.
"""
struct TimeEvolutionResult{S,M,T,V,B,D}
    solver::S
    model::M
    times::T
    states::V
    basis::B
    metadata::D
end
