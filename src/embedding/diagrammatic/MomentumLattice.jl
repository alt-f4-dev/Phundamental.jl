# Momentum-resolved lattice embedding used by ladder diagrammatic DMFT.

"""
    MomentumLattice(energies; kpoints=eachindex(energies), weights=nothing, qmaps=nothing, qpoints=nothing)

Scalar single-band momentum lattice with explicit dispersion values, normalized momentum weights, and optional exact `k -> k+q` index maps. When `qmaps` is supplied the object can be used directly by ladder diagrammatic DMFT; without it the object remains a valid single-site DMFT lattice embedding but does not define momentum-transfer topology.
"""
struct MomentumLattice{K,Q,M} <: AbstractLatticeEmbedding
    energies::Vector{Float64}
    weights::Vector{Float64}
    kpoints::K
    qpoints::Q
    qmaps::M
end

function MomentumLattice(energies; kpoints=collect(eachindex(energies)), weights=nothing, qmaps=nothing, qpoints=nothing)
    eps = Float64.(collect(energies))
    isempty(eps) && throw(ArgumentError("MomentumLattice requires at least one momentum point"))
    all(isfinite, eps) || throw(ArgumentError("momentum-lattice energies must be finite"))
    ks = collect(kpoints)
    length(ks) == length(eps) || throw(DimensionMismatch("kpoint and dispersion counts differ"))
    w = isnothing(weights) ? fill(inv(length(eps)), length(eps)) : Float64.(collect(weights))
    length(w) == length(eps) || throw(DimensionMismatch("momentum weights and dispersion counts differ"))
    all(>=(0.0), w) || throw(ArgumentError("momentum weights must be nonnegative"))
    sum(w) > 0 || throw(ArgumentError("momentum weights must have positive total weight"))
    w ./= sum(w)
    maps = isnothing(qmaps) ? nothing : [Int.(collect(mapping)) for mapping in qmaps]
    if !isnothing(maps)
        all(mapping -> length(mapping) == length(eps) && all(index -> 1 <= index <= length(eps), mapping), maps) ||
            throw(ArgumentError("each q map must contain one valid shifted momentum index per k point"))
    end
    qs = isnothing(qpoints) ? (isnothing(maps) ? nothing : collect(1:length(maps))) : collect(qpoints)
    !isnothing(maps) && length(qs) != length(maps) && throw(DimensionMismatch("qpoint and q-map counts differ"))
    return MomentumLattice{typeof(ks),typeof(qs),typeof(maps)}(eps, w, ks, qs, maps)
end

"""Construct a periodic Cartesian momentum lattice from a dispersion sampled on a mesh whose first index varies fastest."""
function periodic_momentum_lattice(shape::NTuple{D,<:Integer}, dispersion::Function; weights=nothing) where {D}
    shifted = periodic_momentum_shift_maps(shape)
    dims = Int.(collect(shape))
    coordinates = shifted.qpoints
    reduced = [ntuple(d -> 2π * coordinate[d] / dims[d], D) for coordinate in coordinates]
    energies = Float64[dispersion(k) for k in reduced]
    return MomentumLattice(energies; kpoints=reduced, weights=weights, qmaps=shifted.maps, qpoints=reduced)
end

function lattice_green(lattice::MomentumLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    matrix_dimension(sigma) == 1 || throw(DimensionMismatch("MomentumLattice currently supports a scalar local self-energy"))
    _compatible_frequency_objects(sigma, zero_self_energy(axis, 1; labels=[:impurity]))
    values = zeros(ComplexF64, length(axis))
    @inbounds for iν in eachindex(axis.values)
        zeta = axis[iν] + chemical_potential - sigma[1, 1, iν]
        for ik in eachindex(lattice.energies)
            values[iν] += lattice.weights[ik] / (zeta - lattice.energies[ik])
        end
    end
    return GreenFunction(axis, values; labels=[:local], metadata=Dict{Symbol,Any}(:lattice => :momentum, :nk => length(lattice.energies)))
end

function momentum_green(lattice::MomentumLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    return momentum_green(lattice.energies, axis, sigma; chemical_potential=chemical_potential, kpoints=lattice.kpoints)
end
