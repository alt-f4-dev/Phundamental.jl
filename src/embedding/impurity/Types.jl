# ============================================================================
# EmbeddingTypes.jl: preserved implementation from src/embedding/EmbeddingTypes.jl
# ============================================================================


# Shared embedding abstractions and result containers.

abstract type AbstractEmbeddingProblem end
abstract type AbstractLatticeEmbedding end

"""Result of one finite-temperature impurity solve."""
struct ImpuritySolverResult{M,G,G0,S,B,T}
    model::M
    green_function::G
    noninteracting_green_function::G0
    self_energy::S
    density::Float64
    double_occupancy::Float64
    bath::B
    thermal_state::T
    metadata::Dict{Symbol,Any}
end

# ============================================================================
# BathDiscretization.jl: preserved implementation from src/embedding/BathDiscretization.jl
# ============================================================================


# Finite bath representation of an impurity hybridization function.

"""
    DiscreteBath(energies, hybridizations)

Finite spin-degenerate ED bath with physical one-particle energies `εₚ` and impurity couplings `Vₚ`.
"""
struct DiscreteBath
    energies::Vector{Float64}
    hybridizations::Vector{ComplexF64}

    function DiscreteBath(energies, hybridizations)
        eps = Float64.(collect(energies))
        V = ComplexF64.(collect(hybridizations))
        length(eps) == length(V) || throw(DimensionMismatch("bath energies and hybridizations must have equal length"))
        all(isfinite, eps) || throw(ArgumentError("bath energies must be finite"))
        all(v -> isfinite(real(v)) && isfinite(imag(v)), V) || throw(ArgumentError("bath hybridizations must be finite"))
        return new(eps, V)
    end
end

Base.length(bath::DiscreteBath) = length(bath.energies)

"""Evaluate `Δ_ED(z)=Σₚ |Vₚ|²/(z+μ-εₚ)` for a finite bath."""
function hybridization_function(bath::DiscreteBath, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0)
    values = zeros(ComplexF64, length(axis))
    @inbounds for i in 1:length(axis)
        z = axis[i]
        for p in eachindex(bath.energies)
            values[i] += abs2(bath.hybridizations[p]) / (z + chemical_potential - bath.energies[p])
        end
    end
    return HybridizationFunction(axis, values; labels=[:impurity],
                                 metadata=Dict{Symbol,Any}(:kind => :finite_ed_bath, :chemical_potential => Float64(chemical_potential), :bath_sites => length(bath)))
end

# ============================================================================
# ImpuritySolver.jl: preserved implementation from src/embedding/ImpuritySolver.jl
# ============================================================================


# Narrow impurity-solver interface used by self-consistent embeddings.

abstract type AbstractImpuritySolver <: AbstractSolver end

# ============================================================================
# ImpurityProblems.jl: preserved implementation from src/embedding/ImpurityProblems.jl
# ============================================================================


# Typed impurity problems independent of a particular impurity solver.

"""
    ImpurityProblem(interaction, axis; chemical_potential=0, impurity_energy=0, target_hybridization=nothing, bath=nothing)

Single-orbital spin-degenerate impurity problem. Exactly one of `target_hybridization` or `bath` must be supplied: the former requests bath fitting by a finite-bath solver, while the latter fixes the bath explicitly.
"""
struct ImpurityProblem{A<:FermionicMatsubaraAxis,H,B} <: AbstractEmbeddingProblem
    interaction::Float64
    chemical_potential::Float64
    impurity_energy::Float64
    axis::A
    target_hybridization::H
    bath::B
end

function ImpurityProblem(interaction::Real, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0, impurity_energy::Real=0.0,
                         target_hybridization=nothing, bath=nothing)
    xor(isnothing(target_hybridization), isnothing(bath)) || throw(ArgumentError("supply exactly one of target_hybridization or bath"))
    if !isnothing(target_hybridization)
        target_hybridization isa HybridizationFunction || throw(ArgumentError("target_hybridization must be a HybridizationFunction"))
        _compatible_frequency_objects(target_hybridization, HybridizationFunction(axis, zeros(ComplexF64, length(axis)); labels=[:impurity]))
        matrix_dimension(target_hybridization) == 1 || throw(DimensionMismatch("the first ED-DMFT milestone supports a scalar impurity hybridization"))
    end
    !isnothing(bath) && !(bath isa DiscreteBath) && throw(ArgumentError("bath must be a DiscreteBath"))
    return ImpurityProblem(Float64(interaction), Float64(chemical_potential), Float64(impurity_energy), axis, target_hybridization, bath)
end

# ============================================================================
# MultiOrbitalImpurity.jl: preserved implementation from src/embedding/MultiOrbitalImpurity.jl
# ============================================================================


# Phase II.A.1: spin-orbital impurity data, antisymmetrized interactions, and finite matrix baths.

_isfinite_complex(z) = isfinite(real(z)) && isfinite(imag(z))

function _impurity_hermitian(values::AbstractMatrix, description::AbstractString; tolerance::Real=1e-10)
    n, m = size(values)
    n > 0 && n == m || throw(DimensionMismatch("$description must be a nonempty square matrix"))
    all(_isfinite_complex, values) || throw(ArgumentError("$description must contain finite entries"))
    isapprox(values, values'; atol=tolerance, rtol=tolerance) || throw(ArgumentError("$description must be Hermitian"))
    return Matrix{ComplexF64}(Hermitian(ComplexF64.(values)))
end

function _impurity_labels(labels, n::Int)
    names = isnothing(labels) ? [Symbol("spin_orbital_", a) for a in 1:n] : Symbol.(collect(labels))
    length(names) == n && length(unique(names)) == n || throw(DimensionMismatch("spin-orbital labels must be distinct and match the local dimension"))
    return names
end

"""
    SpinOrbitalInteraction(tensor; labels=nothing, tolerance=1e-10)

Antisymmetrized two-particle integral tensor `U[a,b,c,d]` with the convention
`H_int = (1/4) * sum(U[a,b,c,d] * d†[a] * d†[b] * d[d] * d[c])`.
Antisymmetry is required in `(a,b)` and `(c,d)`, with `U[a,b,c,d] = conj(U[c,d,a,b])`.
The constructor validates conventions rather than silently antisymmetrizing physical parameters.
"""
struct SpinOrbitalInteraction
    tensor::Array{ComplexF64,4}
    labels::Vector{Symbol}
end

function SpinOrbitalInteraction(tensor::AbstractArray{<:Number,4}; labels=nothing, tolerance::Real=1e-10)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("tolerance must be finite and nonnegative"))
    n = size(tensor, 1)
    n > 0 && size(tensor) == (n, n, n, n) || throw(DimensionMismatch("interaction tensor must have four equal nonzero dimensions"))
    values = ComplexF64.(tensor)
    all(_isfinite_complex, values) || throw(ArgumentError("interaction tensor must contain finite values"))
    @inbounds for d in 1:n, c in 1:n, b in 1:n, a in 1:n
        v = values[a, b, c, d]
        isapprox(v, -values[b, a, c, d]; atol=tolerance, rtol=tolerance) || throw(ArgumentError("interaction tensor must be antisymmetric in creation indices"))
        isapprox(v, -values[a, b, d, c]; atol=tolerance, rtol=tolerance) || throw(ArgumentError("interaction tensor must be antisymmetric in annihilation indices"))
        isapprox(v, conj(values[c, d, a, b]); atol=tolerance, rtol=tolerance) || throw(ArgumentError("interaction tensor must satisfy pair Hermiticity"))
    end
    return SpinOrbitalInteraction(values, _impurity_labels(labels, n))
end

# Add the antisymmetrized tensor corresponding to `coefficient * d†[a] d†[b] d[d] d[c]`.
function _add_impurity_term!(tensor::Array{ComplexF64,4}, a::Int, b::Int, c::Int, d::Int, coefficient::Real)
    a == b && throw(ArgumentError("creation indices must differ"))
    c == d && throw(ArgumentError("annihilation indices must differ"))
    tensor[a, b, c, d] += coefficient
    tensor[b, a, c, d] -= coefficient
    tensor[a, b, d, c] -= coefficient
    tensor[b, a, d, c] += coefficient
    return tensor
end

"""
    kanamori_interaction(norbitals; U, J=0, Uprime=U-2J)

Rotationally invariant spinful Kanamori interaction in interleaved order
`(orbital1_up, orbital1_down, orbital2_up, orbital2_down, ...)`, including
intraorbital `U`, opposite-spin interorbital `Uprime`, same-spin `Uprime-J`,
spin-flip, and pair-hopping terms. `Uprime` may be provided independently.
"""
function kanamori_interaction(norbitals::Integer; U::Real, J::Real=0.0, Uprime::Real=U - 2J)
    norbitals >= 1 || throw(ArgumentError("norbitals must be positive"))
    all(isfinite, (U, J, Uprime)) || throw(ArgumentError("Kanamori interaction parameters must be finite"))
    n = 2 * Int(norbitals)
    tensor = zeros(ComplexF64, n, n, n, n)
    up(m) = 2m - 1
    down(m) = 2m
    for m in 1:norbitals
        _add_impurity_term!(tensor, up(m), down(m), up(m), down(m), U)
    end
    for m in 1:norbitals-1, p in m+1:norbitals
        for (a, b, strength) in ((up(m), up(p), Uprime - J), (down(m), down(p), Uprime - J),
                                 (up(m), down(p), Uprime), (down(m), up(p), Uprime))
            _add_impurity_term!(tensor, a, b, a, b, strength)
        end
        # The following ordered operator pairs are Hermitian conjugates.
        _add_impurity_term!(tensor, up(m), down(p), up(p), down(m), J)
        _add_impurity_term!(tensor, up(p), down(m), up(m), down(p), J)
        _add_impurity_term!(tensor, up(m), down(m), up(p), down(p), J)
        _add_impurity_term!(tensor, up(p), down(p), up(m), down(m), J)
    end
    names = [Symbol("orbital_", m, spin) for m in 1:norbitals for spin in ("_up", "_down")]
    return SpinOrbitalInteraction(tensor; labels=names)
end

"""
    MatrixDiscreteBath(energies, hybridizations; labels=nothing)

Hermitian finite bath in its diagonal energy basis. `hybridizations[a,p]` couples
impurity spin-orbital `a` to bath orbital `p`; there is no spin-degeneracy assumption.
"""
struct MatrixDiscreteBath
    energies::Vector{Float64}
    hybridizations::Matrix{ComplexF64}
    labels::Vector{Symbol}
end

function MatrixDiscreteBath(energies::AbstractVector, hybridizations::AbstractMatrix; labels=nothing)
    eps = Float64.(collect(energies))
    V = ComplexF64.(hybridizations)
    nimp, nbath = size(V)
    nimp > 0 || throw(DimensionMismatch("a matrix bath must have at least one impurity component"))
    nbath == length(eps) || throw(DimensionMismatch("bath coupling columns must match energy count"))
    all(isfinite, eps) && all(_isfinite_complex, V) || throw(ArgumentError("bath parameters must be finite"))
    return MatrixDiscreteBath(eps, V, _impurity_labels(labels, nimp))
end

Base.length(bath::MatrixDiscreteBath) = length(bath.energies)

"""Return `sum_p V[:,p] * V[:,p]'`, the exact leading high-frequency hybridization moment."""
function hybridization_moment(bath::MatrixDiscreteBath)
    return bath.hybridizations * bath.hybridizations'
end

"""Evaluate the causal matrix hybridization `Δ(z) = V * diag(1/(z+μ-ε)) * V†`."""
function hybridization_function(bath::MatrixDiscreteBath, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0)
    mu = Float64(chemical_potential)
    isfinite(mu) || throw(ArgumentError("chemical_potential must be finite"))
    n = size(bath.hybridizations, 1)
    values = zeros(ComplexF64, n, n, length(axis))
    @inbounds for i in 1:length(axis), p in eachindex(bath.energies)
        factor = inv(axis[i] + mu - bath.energies[p])
        for b in 1:n, a in 1:n
            values[a, b, i] += factor * bath.hybridizations[a, p] * conj(bath.hybridizations[b, p])
        end
    end
    return HybridizationFunction(axis, values; labels=bath.labels,
                                 metadata=Dict{Symbol,Any}(:kind => :finite_matrix_ed_bath, :chemical_potential => mu, :bath_sites => length(bath)))
end

"""
    MultiOrbitalImpurityProblem(one_body, interaction, axis; chemical_potential=0, target_hybridization=nothing, bath=nothing, labels=nothing)

Spin-orbital impurity problem for Phase II.A.1. `one_body` contains physical
onsite energies (without `-μ`). Supply either a matrix Matsubara target or an
explicit `MatrixDiscreteBath`. This representation intentionally has no ED solver dispatch yet.
"""
struct MultiOrbitalImpurityProblem{A<:FermionicMatsubaraAxis,H,B} <: AbstractEmbeddingProblem
    one_body::Matrix{ComplexF64}
    interaction::SpinOrbitalInteraction
    chemical_potential::Float64
    axis::A
    target_hybridization::H
    bath::B
    labels::Vector{Symbol}
end

function MultiOrbitalImpurityProblem(one_body::AbstractMatrix, interaction::SpinOrbitalInteraction, axis::FermionicMatsubaraAxis;
                                    chemical_potential::Real=0.0, target_hybridization=nothing, bath=nothing, labels=nothing)
    h = _impurity_hermitian(one_body, "impurity one-body Hamiltonian")
    n = size(h, 1)
    n == size(interaction.tensor, 1) || throw(DimensionMismatch("interaction dimension must match the impurity Hamiltonian"))
    names = isnothing(labels) ? copy(interaction.labels) : _impurity_labels(labels, n)
    xor(isnothing(target_hybridization), isnothing(bath)) || throw(ArgumentError("supply exactly one of target_hybridization or bath"))
    mu = Float64(chemical_potential)
    isfinite(mu) || throw(ArgumentError("chemical_potential must be finite"))
    if !isnothing(target_hybridization)
        target_hybridization isa HybridizationFunction || throw(ArgumentError("target_hybridization must be a HybridizationFunction"))
        matrix_dimension(target_hybridization) == n || throw(DimensionMismatch("target matrix dimension must match the impurity"))
        target_hybridization.axis.indices == axis.indices && target_hybridization.axis.beta == axis.beta ||
            throw(ArgumentError("target hybridization and impurity Matsubara axes must match"))
        target_hybridization.labels == names || throw(ArgumentError("target hybridization labels must agree with impurity spin-orbital labels"))
    else
        bath isa MatrixDiscreteBath || throw(ArgumentError("bath must be a MatrixDiscreteBath"))
        size(bath.hybridizations, 1) == n || throw(DimensionMismatch("bath coupling rows must match the impurity"))
        bath.labels == names || throw(ArgumentError("bath labels must agree with impurity spin-orbital labels"))
    end
    return MultiOrbitalImpurityProblem(h, interaction, mu, axis, target_hybridization, bath, names)
end

function MultiOrbitalImpurityProblem(one_body::AbstractMatrix, tensor::AbstractArray{<:Number,4}, axis::FermionicMatsubaraAxis; labels=nothing, kwargs...)
    interaction = SpinOrbitalInteraction(tensor; labels=labels)
    return MultiOrbitalImpurityProblem(one_body, interaction, axis; labels=labels, kwargs...)
end

"""
    MultiOrbitalAndersonHamiltonian

Structured fermionic Anderson Hamiltonian, with the grand-canonical one-body
matrix on impurity plus bath modes and an interaction tensor acting only on the
first `impurity_dimension` spin-orbitals. II.A.2 will supply the many-body
realization and interacting eigensolver; this object does not claim to solve it.
"""
struct MultiOrbitalAndersonHamiltonian
    one_body::Matrix{ComplexF64}
    interaction::SpinOrbitalInteraction
    impurity_dimension::Int
    labels::Vector{Symbol}
end

"""Build the full structured Anderson Hamiltonian from an explicit finite bath."""
function anderson_hamiltonian(problem::MultiOrbitalImpurityProblem; bath=problem.bath)
    one_body = impurity_one_body_matrix(problem; bath=bath)
    n = size(problem.one_body, 1)
    all_labels = vcat(problem.labels, [Symbol("bath_", p) for p in 1:size(one_body, 1)-n])
    return MultiOrbitalAndersonHamiltonian(one_body, problem.interaction, n, all_labels)
end

"""Construct the complete grand-canonical impurity-plus-bath one-body matrix, without the interaction."""
function impurity_one_body_matrix(problem::MultiOrbitalImpurityProblem; bath=problem.bath)
    bath isa MatrixDiscreteBath || throw(ArgumentError("a finite MatrixDiscreteBath is required to assemble the Anderson one-body matrix"))
    n = size(problem.one_body, 1)
    size(bath.hybridizations, 1) == n || throw(DimensionMismatch("bath impurity component count differs"))
    bath.labels == problem.labels || throw(ArgumentError("bath spin-orbital labels do not match the impurity"))
    nbath = length(bath)
    h = zeros(ComplexF64, n + nbath, n + nbath)
    h[1:n, 1:n] .= problem.one_body
    h[1:n, n+1:end] .= bath.hybridizations
    h[n+1:end, 1:n] .= bath.hybridizations'
    @inbounds for p in 1:nbath
        h[n+p, n+p] = bath.energies[p]
    end
    @inbounds for a in 1:n+nbath
        h[a, a] -= problem.chemical_potential
    end
    return h
end

"""Noninteracting interacting-orbital block of the Anderson resolvent, with interactions explicitly omitted."""
function noninteracting_impurity_green(problem::MultiOrbitalImpurityProblem; bath=problem.bath)
    delta = isnothing(bath) ? problem.target_hybridization : hybridization_function(bath, problem.axis; chemical_potential=problem.chemical_potential)
    isnothing(delta) && throw(ArgumentError("a bath or target hybridization is required"))
    n = size(problem.one_body, 1)
    matrix_dimension(delta) == n || throw(DimensionMismatch("hybridization and impurity matrix dimensions differ"))
    delta.labels == problem.labels || throw(ArgumentError("hybridization spin-orbital labels do not match the impurity"))
    values = Array{ComplexF64}(undef, n, n, length(problem.axis))
    identity = Matrix{ComplexF64}(I, n, n)
    @inbounds for i in 1:length(problem.axis)
        values[:, :, i] = inv((problem.axis[i] + problem.chemical_potential) * identity - problem.one_body - view(delta.values, :, :, i))
    end
    return NonInteractingGreenFunction(problem.axis, values; labels=problem.labels,
                                      metadata=Dict{Symbol,Any}(:source => :multiorbital_anderson, :interaction_included => false))
end
