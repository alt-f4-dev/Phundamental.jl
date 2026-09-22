# Frequency-axis abstractions for imaginary-frequency propagators.

abstract type AbstractFrequencyAxis end

"""
    FermionicMatsubaraAxis(beta, npositive; include_negative=false)
    FermionicMatsubaraAxis(beta, indices)

Discrete fermionic Matsubara axis with frequencies `iωₙ = im * (2n + 1)π / beta`. The one-argument count constructor stores the positive-frequency indices `0:npositive-1`; set `include_negative=true` to include the matching negative frequencies.
"""
struct FermionicMatsubaraAxis <: AbstractFrequencyAxis
    beta::Float64
    indices::Vector{Int}
    values::Vector{ComplexF64}
end

function FermionicMatsubaraAxis(beta::Real, indices::AbstractVector{<:Integer})
    beta > 0 || throw(ArgumentError("beta must be positive"))
    isempty(indices) && throw(ArgumentError("a Matsubara axis requires at least one frequency"))
    idx = Int.(collect(indices))
    length(unique(idx)) == length(idx) || throw(ArgumentError("Matsubara indices must be unique"))
    values = ComplexF64[im * (2n + 1) * π / Float64(beta) for n in idx]
    return FermionicMatsubaraAxis(Float64(beta), idx, values)
end

function FermionicMatsubaraAxis(beta::Real, npositive::Integer; include_negative::Bool=false)
    npositive > 0 || throw(ArgumentError("npositive must be positive"))
    indices = include_negative ? vcat(collect(-Int(npositive):-1), collect(0:Int(npositive)-1)) : collect(0:Int(npositive)-1)
    return FermionicMatsubaraAxis(beta, indices)
end

Base.length(axis::FermionicMatsubaraAxis) = length(axis.values)
Base.getindex(axis::FermionicMatsubaraAxis, i::Integer) = axis.values[Int(i)]
Base.iterate(axis::FermionicMatsubaraAxis, state...) = iterate(axis.values, state...)
Base.eltype(::Type{FermionicMatsubaraAxis}) = ComplexF64

"""Return the real Matsubara frequency `ωₙ` associated with stored entry `i`."""
matsubara_frequency(axis::FermionicMatsubaraAxis, i::Integer) = imag(axis[Int(i)])

"""Return the integer Matsubara index associated with stored entry `i`."""
matsubara_index(axis::FermionicMatsubaraAxis, i::Integer) = axis.indices[Int(i)]

"""Return the inverse-temperature parameter `β` carried by a frequency axis."""
inverse_temperature(axis::FermionicMatsubaraAxis) = axis.beta

"""Return the temperature represented by a Matsubara axis for the supplied Boltzmann constant `kB`."""
axis_temperature(axis::FermionicMatsubaraAxis; kB::Real=1.0) = begin
    kB > 0 || throw(ArgumentError("kB must be positive"))
    inv(Float64(kB) * axis.beta)
end
