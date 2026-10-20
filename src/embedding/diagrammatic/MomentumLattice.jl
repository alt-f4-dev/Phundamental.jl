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


"""A Hermitian internal-orbital Hamiltonian on each momentum point, with explicitly normalized quadrature weights."""
struct MatrixMomentumLattice{K,Q,M} <: AbstractLatticeEmbedding
    hamiltonians::Array{ComplexF64,3}
    weights::Vector{Float64}
    kpoints::K
    qpoints::Q
    qmaps::M
    labels::Vector{Symbol}
    gauge::Symbol
end

function MatrixMomentumLattice(hamiltonians::AbstractArray{<:Number,3}; kpoints=collect(axes(hamiltonians, 3)),
                               weights=nothing, qmaps=nothing, qpoints=nothing, labels=nothing,
                               gauge::Symbol=:periodic, tolerance::Real=1e-10)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("Hermiticity tolerance must be finite and nonnegative"))
    gauge in (:periodic, :embedded) || throw(ArgumentError("matrix Bloch gauge must be :periodic or :embedded"))
    values = ComplexF64.(hamiltonians)
    n, m, nk = size(values)
    n > 0 && n == m && nk > 0 || throw(DimensionMismatch("matrix momentum lattice requires nonempty square Hamiltonians"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), values) || throw(ArgumentError("orbital Hamiltonians must be finite"))
    for ik in 1:nk
        block = view(values, :, :, ik)
        isapprox(block, adjoint(block); atol=tolerance, rtol=tolerance) ||
            throw(ArgumentError("Hamiltonian at k index $ik is not Hermitian"))
    end
    ks = collect(kpoints)
    length(ks) == nk || throw(DimensionMismatch("kpoint count differs from matrix dispersion"))
    w = isnothing(weights) ? fill(inv(nk), nk) : Float64.(collect(weights))
    length(w) == nk || throw(DimensionMismatch("momentum weight count differs from matrix dispersion"))
    all(x -> isfinite(x) && x >= 0, w) && sum(w) > 0 || throw(ArgumentError("momentum weights must be finite and nonnegative with positive sum"))
    w ./= sum(w)
    maps = isnothing(qmaps) ? nothing : [Int.(collect(mapping)) for mapping in qmaps]
    if !isnothing(maps)
        all(mapping -> length(mapping) == nk && sort(mapping) == collect(1:nk), maps) ||
            throw(ArgumentError("each exact momentum q map must be a permutation of the k grid"))
    end
    qs = isnothing(qpoints) ? (isnothing(maps) ? nothing : collect(eachindex(maps))) : collect(qpoints)
    !isnothing(maps) && length(qs) != length(maps) && throw(DimensionMismatch("qpoints and qmaps have different lengths"))
    names = isnothing(labels) ? [Symbol("orbital_", i) for i in 1:n] : Symbol.(collect(labels))
    length(names) == n && length(unique(names)) == n || throw(DimensionMismatch("matrix orbital labels must be distinct and match matrix dimension"))
    return MatrixMomentumLattice{typeof(ks),typeof(qs),typeof(maps)}(values, w, ks, qs, maps, names, gauge)
end

function MatrixMomentumLattice(hamiltonians::AbstractVector{<:AbstractMatrix}; kwargs...)
    isempty(hamiltonians) && throw(ArgumentError("matrix momentum lattice requires nonempty Hamiltonians"))
    n = size(first(hamiltonians), 1)
    all(h -> size(h) == (n, n), hamiltonians) || throw(DimensionMismatch("momentum Hamiltonians must share one square dimension"))
    return MatrixMomentumLattice(cat(hamiltonians...; dims=3); kwargs...)
end

"""Compute the full local orbital matrix propagator without invoking a multiorbital impurity solver."""
function lattice_green(lattice::MatrixMomentumLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    resolved = momentum_green(lattice, axis, sigma; chemical_potential=chemical_potential)
    n, _, nk, nfreq = size(resolved.values)
    values = zeros(ComplexF64, n, n, nfreq)
    @inbounds for iν in 1:nfreq, ik in 1:nk
        values[:, :, iν] .+= lattice.weights[ik] .* view(resolved.values, :, :, ik, iν)
    end
    return GreenFunction(axis, values; labels=lattice.labels,
                         metadata=Dict{Symbol,Any}(:lattice => :matrix_momentum, :nk => nk, :solver => :matrix_dyson_only))
end

"""Matrix-valued one-particle propagator on the given Hermitian momentum lattice; no impurity approximation is implied."""
function momentum_green(lattice::MatrixMomentumLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    resolved = momentum_green(lattice.hamiltonians, axis, sigma; chemical_potential=chemical_potential,
                              kpoints=lattice.kpoints, labels=lattice.labels, weights=lattice.weights)
    resolved.metadata[:bloch_gauge] = lattice.gauge
    return resolved
end


function DMFTProblem(lattice::MatrixMomentumLattice, interaction::Real; kwargs...)
    throw(ArgumentError("matrix momentum lattices do not have a certified multiorbital ED-DMFT impurity solver"))
end
