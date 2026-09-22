# Local lattice Green functions used by single-site embeddings.

"""Atomic lattice with one local orbital and no hopping."""
struct AtomicLattice <: AbstractLatticeEmbedding
    onsite_energy::Float64
end
AtomicLattice(; onsite_energy::Real=0.0) = AtomicLattice(Float64(onsite_energy))

"""Discrete scalar noninteracting lattice spectrum with normalized quadrature weights."""
struct DiscreteLattice <: AbstractLatticeEmbedding
    energies::Vector{Float64}
    weights::Vector{Float64}
end

function DiscreteLattice(energies; weights=nothing)
    eps = Float64.(collect(energies))
    isempty(eps) && throw(ArgumentError("DiscreteLattice requires at least one energy"))
    all(isfinite, eps) || throw(ArgumentError("lattice energies must be finite"))
    w = isnothing(weights) ? fill(inv(length(eps)), length(eps)) : Float64.(collect(weights))
    length(w) == length(eps) || throw(DimensionMismatch("lattice weights and energies must have equal length"))
    all(>=(0.0), w) || throw(ArgumentError("lattice weights must be nonnegative"))
    sum(w) > 0 || throw(ArgumentError("lattice weights must have positive total weight"))
    w ./= sum(w)
    return DiscreteLattice(eps, w)
end

"""Infinite-coordination Bethe lattice represented by a semicircular density of states with half-bandwidth `D`."""
struct BetheLattice <: AbstractLatticeEmbedding
    half_bandwidth::Float64
end

function BetheLattice(half_bandwidth::Real)
    half_bandwidth > 0 || throw(ArgumentError("half_bandwidth must be positive"))
    return BetheLattice(Float64(half_bandwidth))
end

function _require_scalar_self_energy(axis::FermionicMatsubaraAxis, sigma::SelfEnergy)
    matrix_dimension(sigma) == 1 || throw(DimensionMismatch("the first single-site ED-DMFT implementation supports a scalar self-energy"))
    _compatible_frequency_objects(sigma, zero_self_energy(axis, 1; labels=[:impurity]))
    return sigma
end

function lattice_green(lattice::AtomicLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    _require_scalar_self_energy(axis, sigma)
    values = ComplexF64[
        inv(axis[i] + chemical_potential - lattice.onsite_energy - sigma[1, 1, i])
        for i in 1:length(axis)
    ]
    return GreenFunction(axis, values; labels=[:local], metadata=Dict{Symbol,Any}(:lattice => :atomic))
end

function lattice_green(lattice::DiscreteLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    _require_scalar_self_energy(axis, sigma)
    values = zeros(ComplexF64, length(axis))
    @inbounds for i in 1:length(axis)
        zeta = axis[i] + chemical_potential - sigma[1, 1, i]
        for k in eachindex(lattice.energies)
            values[i] += lattice.weights[k] / (zeta - lattice.energies[k])
        end
    end
    return GreenFunction(axis, values; labels=[:local], metadata=Dict{Symbol,Any}(:lattice => :discrete, :nk => length(lattice.energies)))
end

function _bethe_local_green(zeta::ComplexF64, half_bandwidth::Float64)
    root = sqrt(zeta^2 - half_bandwidth^2)
    imag(root) * imag(zeta) < 0 && (root = -root)
    return 2 / (zeta + root)
end

function lattice_green(lattice::BetheLattice, axis::FermionicMatsubaraAxis, sigma::SelfEnergy; chemical_potential::Real=0.0)
    _require_scalar_self_energy(axis, sigma)
    values = ComplexF64[
        _bethe_local_green(ComplexF64(axis[i] + chemical_potential - sigma[1, 1, i]), lattice.half_bandwidth)
        for i in 1:length(axis)
    ]
    return GreenFunction(axis, values; labels=[:local],
                         metadata=Dict{Symbol,Any}(:lattice => :bethe, :half_bandwidth => lattice.half_bandwidth))
end
