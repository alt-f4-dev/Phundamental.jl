# Dyson relations between propagators and self-energies.

function _inverse_frequency_values(object::FrequencyMatrix)
    n = matrix_dimension(object)
    values = Array{ComplexF64}(undef, n, n, length(object))
    @inbounds for i in 1:length(object)
        values[:, :, i] .= _solve_inverse_matrix(object[i])
    end
    return values
end

"""
    self_energy(G0, G)

Compute the Dyson self-energy `Σ(z) = G₀(z)⁻¹ - G(z)⁻¹` frequency by frequency. Matrix inverses are realized through factorizations and linear solves rather than elementwise inversion.
"""
function self_energy(G0::NonInteractingGreenFunction, G::GreenFunction)
    _compatible_frequency_objects(G0, G)
    n = matrix_dimension(G)
    values = Array{ComplexF64}(undef, n, n, length(G))
    @inbounds for i in 1:length(G)
        values[:, :, i] .= _solve_inverse_matrix(G0[i]) .- _solve_inverse_matrix(G[i])
    end
    metadata = Dict{Symbol,Any}(:relation => :dyson, :nambu => get(G.metadata, :nambu, false))
    haskey(G.metadata, :nambu_index) && (metadata[:nambu_index] = G.metadata[:nambu_index])
    return SelfEnergy(G.axis, values; labels=G.labels, metadata=metadata)
end

"""
    dyson_green(G0, Sigma)

Solve Dyson's equation `G(z) = [G₀(z)⁻¹ - Σ(z)]⁻¹` on a shared frequency axis.
"""
function dyson_green(G0::NonInteractingGreenFunction, Sigma::SelfEnergy)
    _compatible_frequency_objects(G0, Sigma)
    n = matrix_dimension(G0)
    values = Array{ComplexF64}(undef, n, n, length(G0))
    @inbounds for i in 1:length(G0)
        kernel = _solve_inverse_matrix(G0[i]) .- Sigma[i]
        values[:, :, i] .= _solve_inverse_matrix(kernel)
    end
    metadata = Dict{Symbol,Any}(:relation => :dyson, :nambu => get(Sigma.metadata, :nambu, false))
    haskey(Sigma.metadata, :nambu_index) && (metadata[:nambu_index] = Sigma.metadata[:nambu_index])
    return GreenFunction(G0.axis, values; labels=G0.labels, metadata=metadata)
end
