# Scalar-channel two-particle Matsubara objects for diagrammatic extensions of DMFT.

abstract type AbstractTwoParticleFrequencyObject end

"""Bosonic Matsubara transfer-frequency axis with `iΩₘ = i 2mπ/β`."""
struct BosonicMatsubaraAxis <: AbstractFrequencyAxis
    beta::Float64
    indices::Vector{Int}
    values::Vector{ComplexF64}
end

function BosonicMatsubaraAxis(beta::Real, indices::AbstractVector{<:Integer})
    beta > 0 || throw(ArgumentError("beta must be positive"))
    isempty(indices) && throw(ArgumentError("a bosonic Matsubara axis requires at least one frequency"))
    idx = Int.(collect(indices))
    length(unique(idx)) == length(idx) || throw(ArgumentError("Matsubara indices must be unique"))
    values = ComplexF64[im * 2m * π / Float64(beta) for m in idx]
    return BosonicMatsubaraAxis(Float64(beta), idx, values)
end

function BosonicMatsubaraAxis(beta::Real, npositive::Integer; include_negative::Bool=false, include_zero::Bool=true)
    npositive >= 0 || throw(ArgumentError("npositive must be nonnegative"))
    positive = collect(1:Int(npositive))
    indices = include_negative ? vcat(-reverse(positive), include_zero ? [0] : Int[], positive) : vcat(include_zero ? [0] : Int[], positive)
    isempty(indices) && throw(ArgumentError("bosonic Matsubara axis would be empty"))
    return BosonicMatsubaraAxis(beta, indices)
end

Base.length(axis::BosonicMatsubaraAxis) = length(axis.values)
Base.getindex(axis::BosonicMatsubaraAxis, i::Integer) = axis.values[Int(i)]
Base.iterate(axis::BosonicMatsubaraAxis, state...) = iterate(axis.values, state...)
Base.eltype(::Type{BosonicMatsubaraAxis}) = ComplexF64
matsubara_frequency(axis::BosonicMatsubaraAxis, i::Integer) = imag(axis[Int(i)])
matsubara_index(axis::BosonicMatsubaraAxis, i::Integer) = axis.indices[Int(i)]
inverse_temperature(axis::BosonicMatsubaraAxis) = axis.beta
axis_temperature(axis::BosonicMatsubaraAxis; kB::Real=1.0) = begin
    kB > 0 || throw(ArgumentError("kB must be positive"))
    inv(Float64(kB) * axis.beta)
end

function _two_particle_values(fermionic_axis::FermionicMatsubaraAxis, bosonic_axis::BosonicMatsubaraAxis, values; copy_values::Bool=true)
    size(values) == (length(fermionic_axis), length(fermionic_axis), length(bosonic_axis)) || throw(DimensionMismatch(
        "two-particle values must have shape (nfermionic, nfermionic, nbosonic)"
    ))
    isapprox(fermionic_axis.beta, bosonic_axis.beta; rtol=0.0, atol=16eps(Float64) * max(fermionic_axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic Matsubara axes must share the same beta"))
    if !copy_values && values isa Array{ComplexF64,3}
        return values
    end
    return ComplexF64.(values)
end

"""Generalized susceptibility `χ(ν,ν′,Ω)` in one explicitly labeled scattering channel."""
struct GeneralizedSusceptibility{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis,M} <: AbstractTwoParticleFrequencyObject
    fermionic_axis::F
    bosonic_axis::B
    values::Array{ComplexF64,3}
    channel::Symbol
    metadata::M
end

function GeneralizedSusceptibility(fermionic_axis::F, bosonic_axis::B, values; channel::Symbol=:density, metadata=Dict{Symbol,Any}(),
                                   copy_values::Bool=true) where {F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    cube = _two_particle_values(fermionic_axis, bosonic_axis, values; copy_values=copy_values)
    return GeneralizedSusceptibility{F,B,typeof(metadata)}(fermionic_axis, bosonic_axis, cube, channel, metadata)
end

"""Local irreducible two-particle vertex `Γ(ν,ν′,Ω)` in a specified channel."""
struct IrreducibleVertex{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis,M} <: AbstractTwoParticleFrequencyObject
    fermionic_axis::F
    bosonic_axis::B
    values::Array{ComplexF64,3}
    channel::Symbol
    metadata::M
end

function IrreducibleVertex(fermionic_axis::F, bosonic_axis::B, values; channel::Symbol=:density, metadata=Dict{Symbol,Any}(),
                           copy_values::Bool=true) where {F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    cube = _two_particle_values(fermionic_axis, bosonic_axis, values; copy_values=copy_values)
    return IrreducibleVertex{F,B,typeof(metadata)}(fermionic_axis, bosonic_axis, cube, channel, metadata)
end


function _active_two_particle_indices(object::AbstractTwoParticleFrequencyObject, iΩ::Integer)
    masks = get(object.metadata, :active_indices_by_bosonic, nothing)
    isnothing(masks) && return collect(1:length(object.fermionic_axis))
    1 <= iΩ <= length(masks) || throw(BoundsError(masks, iΩ))
    return Int.(collect(masks[Int(iΩ)]))
end

function _common_active_two_particle_indices(a::AbstractTwoParticleFrequencyObject, b::AbstractTwoParticleFrequencyObject, iΩ::Integer)
    left = Set(_active_two_particle_indices(a, iΩ))
    right = Set(_active_two_particle_indices(b, iΩ))
    active = sort!(collect(intersect(left, right)))
    isempty(active) && throw(ArgumentError("two-particle frequency box has no active fermionic frequencies at bosonic index $iΩ"))
    return active
end

function _compatible_two_particle(a::AbstractTwoParticleFrequencyObject, b::AbstractTwoParticleFrequencyObject)
    a.fermionic_axis.indices == b.fermionic_axis.indices || throw(ArgumentError("fermionic Matsubara grids differ"))
    a.bosonic_axis.indices == b.bosonic_axis.indices || throw(ArgumentError("bosonic Matsubara grids differ"))
    isapprox(a.fermionic_axis.beta, b.fermionic_axis.beta; rtol=0.0, atol=16eps(Float64) * max(a.fermionic_axis.beta, 1.0)) ||
        throw(ArgumentError("two-particle objects have incompatible inverse temperatures"))
    a.channel == b.channel || throw(ArgumentError("two-particle objects belong to different scattering channels"))
    return true
end

"""Numerical diagnostics for one matrix inversion in a two-particle equation."""
struct TwoParticleInverseDiagnostics
    condition_number::Float64
    numerical_rank::Int
    dimension::Int
    cutoff::Float64
    pseudoinverse::Bool
end

function _stable_inverse(matrix::AbstractMatrix; diagonal_shift::Real=0.0, rtol::Real=sqrt(eps(Float64)), atol::Real=0.0,
                         pseudoinverse::Bool=false, return_diagnostics::Bool=false)
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch("two-particle inversion requires a square matrix"))
    diagonal_shift >= 0 || throw(ArgumentError("diagonal_shift must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    M = Matrix{ComplexF64}(matrix)
    diagonal_shift > 0 && (M .+= diagonal_shift .* Matrix{ComplexF64}(I, size(M, 1), size(M, 2)))
    decomposition = svd(M)
    maximum_singular = isempty(decomposition.S) ? 0.0 : maximum(decomposition.S)
    cutoff = max(Float64(atol), Float64(rtol) * maximum_singular)
    rank = count(>(cutoff), decomposition.S)
    minimum_nonzero = rank == 0 ? 0.0 : minimum(decomposition.S[decomposition.S .> cutoff])
    condition = rank < size(M, 1) || iszero(minimum_nonzero) ? Inf : maximum_singular / minimum_nonzero
    if rank < size(M, 1) && !pseudoinverse
        throw(LinearAlgebra.SingularException(rank + 1))
    end
    inverse_singular = ComplexF64[s > cutoff ? inv(s) : 0.0 for s in decomposition.S]
    inverse_matrix = decomposition.V * Diagonal(inverse_singular) * adjoint(decomposition.U)
    diagnostics = TwoParticleInverseDiagnostics(Float64(condition), rank, size(M, 1), cutoff, pseudoinverse)
    return return_diagnostics ? (inverse_matrix, diagnostics) : inverse_matrix
end


function _diagonal_inverse(diagonal_values::AbstractVector; diagonal_shift::Real=0.0, rtol::Real=sqrt(eps(Float64)), atol::Real=0.0,
                           pseudoinverse::Bool=false, return_diagnostics::Bool=false)
    diagonal_shift >= 0 || throw(ArgumentError("diagonal_shift must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    shifted = ComplexF64[value + diagonal_shift for value in diagonal_values]
    magnitudes = abs.(shifted)
    maximum_singular = isempty(magnitudes) ? 0.0 : maximum(magnitudes)
    cutoff = max(Float64(atol), Float64(rtol) * maximum_singular)
    rank = count(>(cutoff), magnitudes)
    minimum_nonzero = rank == 0 ? 0.0 : minimum(magnitudes[magnitudes .> cutoff])
    condition = rank < length(shifted) || iszero(minimum_nonzero) ? Inf : maximum_singular / minimum_nonzero
    if rank < length(shifted) && !pseudoinverse
        throw(LinearAlgebra.SingularException(rank + 1))
    end
    inverse_diagonal = ComplexF64[magnitude > cutoff ? inv(value) : 0.0 for (value, magnitude) in zip(shifted, magnitudes)]
    inverse_matrix = Matrix(Diagonal(inverse_diagonal))
    diagnostics = TwoParticleInverseDiagnostics(Float64(condition), rank, length(shifted), cutoff, pseudoinverse)
    return return_diagnostics ? (inverse_matrix, diagnostics) : inverse_matrix
end

function _bubble_inverse(bubble::GeneralizedSusceptibility, active, iΩ::Integer; diagonal_shift::Real=0.0,
                         rtol::Real=sqrt(eps(Float64)), atol::Real=0.0, pseudoinverse::Bool=false, return_diagnostics::Bool=false)
    matrix = view(bubble.values, active, active, Int(iΩ))
    if get(bubble.metadata, :diagonal_in_fermionic_indices, false)
        diagonal_values = ComplexF64[matrix[index, index] for index in axes(matrix, 1)]
        return _diagonal_inverse(diagonal_values; diagonal_shift=diagonal_shift, rtol=rtol, atol=atol, pseudoinverse=pseudoinverse,
                                 return_diagnostics=return_diagnostics)
    end
    return _stable_inverse(matrix; diagonal_shift=diagonal_shift, rtol=rtol, atol=atol, pseudoinverse=pseudoinverse,
                           return_diagnostics=return_diagnostics)
end

# Backward-compatible helper used by the first API milestone.
_regularized_inverse(matrix::AbstractMatrix, regularization::Real) = _stable_inverse(matrix; diagonal_shift=regularization)

"""
    irreducible_vertex(bubble, susceptibility; regularization=0, inverse_rtol=sqrt(eps()), inverse_atol=0, pseudoinverse=false)

Extract `Γ = χ₀⁻¹ - χ⁻¹` independently at each bosonic transfer frequency. Singular-value diagnostics are retained in metadata so frequency-box and near-critical conditioning can be audited rather than hidden by an unconditional matrix inverse.
"""
function irreducible_vertex(bubble::GeneralizedSusceptibility, susceptibility::GeneralizedSusceptibility; regularization::Real=0.0,
                            inverse_rtol::Real=sqrt(eps(Float64)), inverse_atol::Real=0.0, pseudoinverse::Bool=false)
    _compatible_two_particle(bubble, susceptibility)
    values = zeros(ComplexF64, size(susceptibility.values))
    diagnostics = Vector{NamedTuple}(undef, size(values, 3))
    active_masks = Vector{Vector{Int}}(undef, size(values, 3))
    for iΩ in axes(values, 3)
        active = _common_active_two_particle_indices(bubble, susceptibility, iΩ)
        χ0inv, d0 = _bubble_inverse(bubble, active, iΩ; diagonal_shift=regularization, rtol=inverse_rtol, atol=inverse_atol,
                                    pseudoinverse=pseudoinverse, return_diagnostics=true)
        χinv, dχ = _stable_inverse(view(susceptibility.values, active, active, iΩ); diagonal_shift=regularization, rtol=inverse_rtol, atol=inverse_atol,
                                   pseudoinverse=pseudoinverse, return_diagnostics=true)
        values[active, active, iΩ] .= χ0inv .- χinv
        diagnostics[iΩ] = (bubble=d0, susceptibility=dχ, active_dimension=length(active))
        active_masks[iΩ] = active
    end
    metadata = Dict{Symbol,Any}(
        :construction => :inverse_bethe_salpeter,
        :regularization => Float64(regularization),
        :inverse_rtol => Float64(inverse_rtol),
        :inverse_atol => Float64(inverse_atol),
        :pseudoinverse => pseudoinverse,
        :inverse_diagnostics => diagnostics,
        :active_indices_by_bosonic => active_masks,
    )
    return IrreducibleVertex(susceptibility.fermionic_axis, susceptibility.bosonic_axis, values;
                             channel=susceptibility.channel, metadata=metadata, copy_values=false)
end
"""Solve `χ = (χ₀⁻¹ - Γ)⁻¹` independently at each bosonic transfer frequency with auditable singular-value diagnostics."""
function bethe_salpeter(bubble::GeneralizedSusceptibility, vertex::IrreducibleVertex; regularization::Real=0.0,
                        inverse_rtol::Real=sqrt(eps(Float64)), inverse_atol::Real=0.0, pseudoinverse::Bool=false)
    _compatible_two_particle(bubble, vertex)
    values = zeros(ComplexF64, size(bubble.values))
    diagnostics = Vector{NamedTuple}(undef, size(values, 3))
    active_masks = Vector{Vector{Int}}(undef, size(values, 3))
    for iΩ in axes(values, 3)
        active = _common_active_two_particle_indices(bubble, vertex, iΩ)
        χ0inv, d0 = _bubble_inverse(bubble, active, iΩ; diagonal_shift=regularization, rtol=inverse_rtol, atol=inverse_atol,
                                    pseudoinverse=pseudoinverse, return_diagnostics=true)
        kernel = χ0inv .- view(vertex.values, active, active, iΩ)
        inverse_kernel, dk = _stable_inverse(kernel; diagonal_shift=regularization, rtol=inverse_rtol, atol=inverse_atol,
                                             pseudoinverse=pseudoinverse, return_diagnostics=true)
        values[active, active, iΩ] .= inverse_kernel
        diagnostics[iΩ] = (bubble=d0, kernel=dk, active_dimension=length(active))
        active_masks[iΩ] = active
    end
    metadata = Dict{Symbol,Any}(
        :construction => :bethe_salpeter,
        :regularization => Float64(regularization),
        :inverse_rtol => Float64(inverse_rtol),
        :inverse_atol => Float64(inverse_atol),
        :pseudoinverse => pseudoinverse,
        :inverse_diagnostics => diagnostics,
        :active_indices_by_bosonic => active_masks,
    )
    return GeneralizedSusceptibility(bubble.fermionic_axis, bubble.bosonic_axis, values; channel=bubble.channel, metadata=metadata, copy_values=false)
end
"""Contract the two fermionic frequencies of a generalized susceptibility for each bosonic transfer frequency."""
function physical_susceptibility(susceptibility::GeneralizedSusceptibility; beta_normalization::Real=1.0)
    beta_normalization > 0 || throw(ArgumentError("beta_normalization must be positive"))
    values = ComplexF64[]
    for iΩ in axes(susceptibility.values, 3)
        active = _active_two_particle_indices(susceptibility, iΩ)
        push!(values, sum(view(susceptibility.values, active, active, iΩ)) / beta_normalization)
    end
    return values
end

include("LatticeBubbles.jl")
include("ImpurityFourPoint.jl")
