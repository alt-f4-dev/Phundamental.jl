# Packed fermionic occupation bases, optionally restricted by exact number
# sectors. Bit positions follow FermionOccupationBasis.ordering.

struct PackedFermionBasisData
    masks::Vector{UInt64}
    index::Dict{UInt64,Int}
    mode_positions::Vector{Int}
    basis::ComputationalBasis
    sector::Union{Nothing,FermionSector}
end

function _spinful_mode_positions(model::ManyBodyModel, factor::FactorLayout)
    fmap = get(model.parameters, :fermion_mode_map, nothing)
    isnothing(fmap) && throw(ArgumentError(
        "nup/ndown sectors require model.parameters[:fermion_mode_map]"
    ))
    up = Int[]
    down = Int[]
    pairs = collect(keys(fmap))
    sites = sort!(unique(first(k) for k in pairs))
    for site in sites
        haskey(fmap, (site, :up)) && push!(up, factor.positions[Int(fmap[(site, :up)])])
        haskey(fmap, (site, :down)) && push!(down, factor.positions[Int(fmap[(site, :down)])])
    end
    (isempty(up) || isempty(down)) && throw(ArgumentError("invalid fermion_mode_map"))
    return up, down
end

function _packed_fermion_basis(
    model::ManyBodyModel,
    sector::Union{Nothing,FermionSector};
    max_dimension::Int,
)
    rep = model.representation
    rep.algebra isa FermionAlgebra || throw(ArgumentError("packed fermion backend requires FermionAlgebra"))
    factors = _representation_factors(rep)
    length(factors) == 1 || throw(ArgumentError("packed fermion backend requires a non-composite representation"))
    factor = factors[1]
    factor.family === :fermion || throw(ArgumentError("invalid fermion factor"))
    M = factor.state_space.nmodes
    M <= 64 || throw(ArgumentError("packed fermion backend supports at most 64 modes"))

    # Exact representation-level occupancy constraints (e.g. Abrikosov) are
    # left to the generic backend for now. Number sectors here are numerical
    # symmetry reductions of an otherwise ordinary fermionic Fock space.
    isempty(_all_physical_constraints(rep)) || throw(ArgumentError(
        "packed fermion backend currently does not combine FermionSector with representation-level constraints"
    ))

    masks = if isnothing(sector)
        M < 63 || throw(ArgumentError(
            "full fermionic Fock basis for M >= 63 is not representable as an in-memory Julia vector; use FermionSector"
        ))
        dim = Int(1) << M
        _checked_dimension(dim, max_dimension, "fermion")
        collect(UInt64(0):UInt64(dim - 1))
    elseif !isnothing(sector.nparticles)
        0 <= sector.nparticles <= M || throw(ArgumentError("nparticles must lie in 0:$M"))
        _fixed_weight_masks(M, sector.nparticles; max_dimension=max_dimension)
    else
        up, down = _spinful_mode_positions(model, factor)
        _product_sector_masks(
            up, sector.nup, down, sector.ndown;
            max_dimension=max_dimension,
        )
    end
    isempty(masks) && throw(ArgumentError("fermion sector has an empty basis"))

    states = FermionBasisState[FermionBasisState(mask) for mask in masks]
    state_index = Dict{FermionBasisState,Int}(s => i for (i, s) in enumerate(states))
    layout = BasisLayout(
        factors,
        _family_map(factors),
        _all_physical_constraints(rep),
        rep.truncation,
        !isempty(_all_physical_constraints(rep)) || !isnothing(rep.physical_subspace),
        !isnothing(rep.truncation),
    )
    basis = ComputationalBasis(rep, states, state_index, layout)
    mask_index = Dict{UInt64,Int}(mask => i for (i, mask) in enumerate(masks))
    mode_positions = [factor.positions[i] for i in 1:M]
    return PackedFermionBasisData(masks, mask_index, mode_positions, basis, sector)
end
