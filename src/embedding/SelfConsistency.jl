# Single-site Dyson self-consistency maps.

"""Construct the Weiss propagator from `𝒢₀⁻¹ = G_loc⁻¹ + Σ`."""
function weiss_green(local_green::GreenFunction, sigma::SelfEnergy)
    _compatible_frequency_objects(local_green, sigma)
    n = matrix_dimension(local_green)
    values = Array{ComplexF64}(undef, n, n, length(local_green))
    @inbounds for i in 1:length(local_green)
        inverse_weiss = _solve_inverse_matrix(local_green[i]) .+ sigma[i]
        values[:, :, i] .= _solve_inverse_matrix(inverse_weiss)
    end
    return NonInteractingGreenFunction(local_green.axis, values; labels=local_green.labels,
                                       metadata=Dict{Symbol,Any}(:kind => :weiss, :relation => :dmft_self_consistency))
end

"""Recover the scalar impurity hybridization from `Δ(z)=z+μ-ε_d-𝒢₀(z)⁻¹`."""
function hybridization_from_weiss(weiss::NonInteractingGreenFunction; chemical_potential::Real=0.0, impurity_energy::Real=0.0)
    matrix_dimension(weiss) == 1 || throw(DimensionMismatch("the first ED-DMFT milestone supports a scalar Weiss field"))
    values = ComplexF64[
        weiss.axis[i] + chemical_potential - impurity_energy - inv(weiss[1, 1, i])
        for i in 1:length(weiss)
    ]
    return HybridizationFunction(weiss.axis, values; labels=[:impurity],
                                 metadata=Dict{Symbol,Any}(:kind => :dmft_target, :chemical_potential => Float64(chemical_potential),
                                                           :impurity_energy => Float64(impurity_energy)))
end
