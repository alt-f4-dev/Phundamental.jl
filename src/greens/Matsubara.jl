# Constructors and elementary operations for Matsubara propagators.

function _solve_inverse_matrix(matrix::AbstractMatrix)
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch("matrix inversion requires a square matrix"))
    n = size(matrix, 1)
    identity_matrix = Matrix{ComplexF64}(I, n, n)
    return factorize(ComplexF64.(matrix)) \ identity_matrix
end

"""
    noninteracting_green(h0, axis; chemical_potential=0, labels=nothing)

Construct `G₀(z) = [(z + μ)I - h₀]⁻¹` for a scalar or finite single-particle Hamiltonian `h0` on `axis`.
"""
function noninteracting_green(h0::Number, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0, labels=nothing)
    values = ComplexF64[inv(z + chemical_potential - h0) for z in axis]
    return NonInteractingGreenFunction(axis, values; labels=isnothing(labels) ? [:orbital_1] : labels,
                                       metadata=Dict{Symbol,Any}(:kind => :noninteracting, :chemical_potential => Float64(chemical_potential)))
end

function noninteracting_green(h0::AbstractMatrix, axis::FermionicMatsubaraAxis; chemical_potential::Real=0.0, labels=nothing)
    size(h0, 1) == size(h0, 2) || throw(DimensionMismatch("h0 must be square"))
    n = size(h0, 1)
    identity_matrix = Matrix{ComplexF64}(I, n, n)
    values = Array{ComplexF64}(undef, n, n, length(axis))
    @inbounds for i in eachindex(axis.values)
        kernel = (axis[i] + chemical_potential) .* identity_matrix .- h0
        values[:, :, i] .= factorize(kernel) \ identity_matrix
    end
    return NonInteractingGreenFunction(axis, values; labels=labels,
                                       metadata=Dict{Symbol,Any}(:kind => :noninteracting, :chemical_potential => Float64(chemical_potential)))
end

"""Construct an identically zero self-energy with the requested matrix dimension."""
function zero_self_energy(axis::FermionicMatsubaraAxis, dimension::Integer=1; labels=nothing)
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    values = zeros(ComplexF64, Int(dimension), Int(dimension), length(axis))
    return SelfEnergy(axis, values; labels=labels, metadata=Dict{Symbol,Any}(:kind => :zero_self_energy))
end
