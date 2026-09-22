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
