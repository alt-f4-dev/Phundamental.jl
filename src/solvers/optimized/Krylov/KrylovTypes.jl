"""Internal Lanczos decomposition used by optimized solvers and spectral tools."""
struct KrylovDecomposition
    alpha::Vector{Float64}
    beta::Vector{Float64}
    basis::Matrix{ComplexF64}
    tail_beta::Float64
    iterations::Int
end

"""Continued-fraction coefficients generated from a normalized seed vector."""
struct ContinuedFractionResult
    alpha::Vector{Float64}
    beta::Vector{Float64}
    norm2::Float64
    metadata::Dict{Symbol,Any}
end
