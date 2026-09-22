# Validation diagnostics for Matsubara and Nambu Green functions.

"""Maximum violation of the positive-frequency Matsubara causality condition `-Im X(iωₙ) ⪰ 0`."""
function causality_residual(object::FrequencyMatrix)
    residual = 0.0
    found = false
    for i in 1:length(object)
        imag(object.axis[i]) > 0 || continue
        found = true
        imaginary_part = Hermitian((object[i] - adjoint(object[i])) / (2im))
        largest = maximum(real.(eigvals(imaginary_part)))
        residual = max(residual, largest)
    end
    found || throw(ArgumentError("causality validation requires at least one positive Matsubara frequency"))
    return max(residual, 0.0)
end

"""Residual of the Matsubara conjugation identity `X(iωₙ)=X(-iωₙ)†`; paired positive and negative frequencies are required."""
function matsubara_conjugation_residual(object::FrequencyMatrix)
    lookup = Dict(index => i for (i, index) in enumerate(object.axis.indices))
    residual = 0.0
    scale = 0.0
    pairs = 0
    for (i, index) in enumerate(object.axis.indices)
        partner_index = -index - 1
        haskey(lookup, partner_index) || continue
        j = lookup[partner_index]
        residual = max(residual, norm(object[i] - adjoint(object[j])))
        scale = max(scale, norm(object[i]), norm(object[j]))
        pairs += 1
    end
    pairs > 0 || throw(ArgumentError("Matsubara conjugation validation requires positive/negative frequency pairs"))
    return residual / max(scale, 1.0)
end

"""
    particle_hole_symmetry_residual(object; center=0)

Residual of the scalar particle-hole identity `[X(iω)-C]+[X(-iω)-C]=0`, where `C` is zero for a particle-hole-symmetric Green function and is the Hartree term for a self-energy.
"""
function particle_hole_symmetry_residual(object::FrequencyMatrix; center::Number=0.0)
    matrix_dimension(object) == 1 || throw(DimensionMismatch("particle-hole symmetry validation currently supports scalar frequency objects"))
    lookup = Dict(index => i for (i, index) in enumerate(object.axis.indices))
    residual = 0.0
    scale = 0.0
    pairs = 0
    offset = ComplexF64(center)
    for (i, index) in enumerate(object.axis.indices)
        partner_index = -index - 1
        haskey(lookup, partner_index) || continue
        j = lookup[partner_index]
        lhs = object[1, 1, i] - offset
        rhs = object[1, 1, j] - offset
        residual = max(residual, abs(lhs + rhs))
        scale = max(scale, abs(lhs), abs(rhs))
        pairs += 1
    end
    pairs > 0 || throw(ArgumentError("particle-hole symmetry validation requires positive/negative frequency pairs"))
    return residual / max(scale, 1.0)
end

"""Return the maximum norm of the anomalous Nambu block over the stored frequency axis."""
function anomalous_norm(G::GreenFunction)
    index = nambu_index(G)
    value = 0.0
    for i in 1:length(G)
        value = max(value, norm(view(G.values, index.particle, index.hole, i)))
    end
    return value
end

"""Residual of the full Nambu redundancy `𝒢(iω)=-τₓ𝒢(-iω)ᵀτₓ`; paired positive and negative Matsubara indices are required."""
function nambu_symmetry_residual(G::GreenFunction)
    index = nambu_index(G)
    n = length(index.modes)
    lookup = Dict(matsubara_index => i for (i, matsubara_index) in enumerate(G.axis.indices))
    permutation = vcat(collect(n+1:2n), collect(1:n))
    residual = 0.0
    scale = 0.0
    pairs = 0
    for (i, matsubara_index) in enumerate(G.axis.indices)
        partner_index = -matsubara_index - 1
        haskey(lookup, partner_index) || continue
        j = lookup[partner_index]
        partner = G.values[permutation, permutation, j]
        difference = G[i] + transpose(partner)
        residual = max(residual, norm(difference))
        scale = max(scale, norm(G[i]), norm(partner))
        pairs += 1
    end
    pairs > 0 || throw(ArgumentError("Nambu symmetry validation requires positive/negative Matsubara pairs"))
    return residual / max(scale, 1.0)
end

"""Collect core finite-temperature Green-function validation diagnostics in one dictionary."""
function validate_green_function(G::GreenFunction; ntail::Integer=min(8, length(G)))
    diagnostics = Dict{Symbol,Any}(
        :causality_residual => causality_residual(G),
        :zeroth_moment_residual => zeroth_moment_residual(G; ntail=ntail),
        :nambu => get(G.metadata, :nambu, false),
    )
    index_set = Set(G.axis.indices)
    paired_axis = any(index -> (-index - 1) in index_set, G.axis.indices)
    paired_axis && (diagnostics[:matsubara_conjugation_residual] = matsubara_conjugation_residual(G))
    if diagnostics[:nambu]
        diagnostics[:anomalous_norm] = anomalous_norm(G)
        diagnostics[:nambu_symmetry_residual] = paired_axis ? nambu_symmetry_residual(G) : NaN
    end
    return diagnostics
end
