# Mixed-radix encoding for finite, unconstrained bosonic occupation bases.

struct PackedBosonBasisData
    codes::Vector{UInt64}
    strides::Vector{UInt64}
    radices::Vector{Int}
    mode_positions::Vector{Int}
    basis::ComputationalBasis
end

function _packed_boson_basis(
    model::ManyBodyModel;
    max_dimension::Int,
)
    rep = model.representation
    rep.algebra isa BosonAlgebra || throw(ArgumentError("packed boson backend requires BosonAlgebra"))
    factors = _representation_factors(rep)
    length(factors) == 1 || throw(ArgumentError("packed boson backend requires non-composite BosonAlgebra"))
    factor = factors[1]
    factor.family === :boson || throw(ArgumentError("invalid boson factor"))
    isempty(_all_physical_constraints(rep)) || throw(ArgumentError(
        "packed boson backend currently supports numerical truncation but not constrained bosonic embeddings"
    ))

    cutoffs = _cutoffs_from_truncation(rep.truncation, factor.algebra.nmodes)
    isnothing(cutoffs) && throw(ArgumentError(
        "packed boson backend requires a finite NumericalTruncation"
    ))
    ordered_cutoffs = [cutoffs[label] for label in factor.ordering]
    radices = ordered_cutoffs .+ 1

    dim_big = prod(big(r) for r in radices)
    dim_big <= max_dimension || throw(ArgumentError(
        "boson basis dimension $dim_big exceeds configured limit $max_dimension"
    ))
    dim_big <= typemax(UInt64) || throw(OverflowError("mixed-radix boson code exceeds UInt64"))
    dim = Int(dim_big)

    strides = Vector{UInt64}(undef, length(radices))
    stride = UInt64(1)
    # Last mode varies fastest, matching generic recursive basis construction.
    for pos in length(radices):-1:1
        strides[pos] = stride
        stride *= UInt64(radices[pos])
    end

    codes = collect(UInt64(0):UInt64(dim - 1))
    states = Vector{BosonBasisState}(undef, dim)
    for (idx, code) in enumerate(codes)
        occ = ntuple(pos -> Int((code ÷ strides[pos]) % UInt64(radices[pos])), length(radices))
        states[idx] = BosonBasisState(occ)
    end
    state_index = Dict{BosonBasisState,Int}(s => i for (i, s) in enumerate(states))

    factor.local_maximum .= ordered_cutoffs
    layout = BasisLayout(
        factors,
        _family_map(factors),
        _all_physical_constraints(rep),
        rep.truncation,
        false,
        true,
    )
    basis = ComputationalBasis(rep, states, state_index, layout)
    mode_positions = [factor.positions[i] for i in 1:factor.algebra.nmodes]
    return PackedBosonBasisData(codes, strides, radices, mode_positions, basis)
end
