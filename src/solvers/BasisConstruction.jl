# Computational basis construction.
#
# The exact ambient mathematical space, exact physical subspace, and numerical
# truncation are intentionally handled separately. The returned basis is the
# finite set on which numerical matrices are represented.

abstract type AbstractComputationalState end

"""Spin-product state; each integer is the local level k=0,...,2S."""
struct SpinBasisState <: AbstractComputationalState
    levels::Tuple{Vararg{Int}}
end

"""Fermion occupation state encoded in the ordering of the basis descriptor."""
struct FermionBasisState <: AbstractComputationalState
    bits::UInt64
end

"""Bosonic occupation state in the ordering of the basis descriptor."""
struct BosonBasisState <: AbstractComputationalState
    occupations::Tuple{Vararg{Int}}
end

"""Tensor-product computational state."""
struct CompositeBasisState <: AbstractComputationalState
    parts::Tuple
end

Base.:(==)(a::SpinBasisState, b::SpinBasisState) = a.levels == b.levels
Base.:(==)(a::FermionBasisState, b::FermionBasisState) = a.bits == b.bits
Base.:(==)(a::BosonBasisState, b::BosonBasisState) = a.occupations == b.occupations
Base.:(==)(a::CompositeBasisState, b::CompositeBasisState) = a.parts == b.parts
Base.hash(a::SpinBasisState, h::UInt) = hash(a.levels, h)
Base.hash(a::FermionBasisState, h::UInt) = hash(a.bits, h)
Base.hash(a::BosonBasisState, h::UInt) = hash(a.occupations, h)
Base.hash(a::CompositeBasisState, h::UInt) = hash(a.parts, h)

"""Numerical layout metadata for one tensor factor."""
struct FactorLayout
    family::Symbol
    state_space::AbstractStateSpace
    algebra::AbstractOperatorAlgebra
    basis::AbstractBasis
    ordering::Vector{Int}
    positions::Dict{Int,Int}
    local_maximum::Vector{Int}
end

"""Layout metadata used by matrix realization and matrix-free application."""
struct BasisLayout
    factors::Vector{FactorLayout}
    family_to_factor::Dict{Symbol,Int}
    physical_constraints::Tuple
    truncation::Any
    has_physical_constraints::Bool
    has_numerical_truncation::Bool
end

"""
    ComputationalBasis

Finite ordered basis with O(1) state-to-column lookup.
"""
struct ComputationalBasis{R,S<:AbstractComputationalState}
    representation::R
    states::Vector{S}
    index::Dict{S,Int}
    layout::BasisLayout
end

Base.length(basis::ComputationalBasis) = length(basis.states)
basis_dimension(basis::ComputationalBasis) = length(basis)
basis_state(basis::ComputationalBasis, i::Integer) = basis.states[i]

# ---------------------------------------------------------------------------
# Generic metadata helpers
# ---------------------------------------------------------------------------

_unwrap_space(space::BiorthogonalSpace) = _unwrap_space(space.ambient)
_unwrap_space(space::AbstractStateSpace) = space

_unwrap_basis(basis::BiorthogonalBasis) = _unwrap_basis(basis.right)
_unwrap_basis(basis::AbstractBasis) = basis

function _constraint_kind(c)
    c isa FactorConstraint && return _constraint_kind(c.constraint)
    c isa NamedTuple || return nothing
    haskey(c, :kind) || return nothing
    return c.kind
end

function _unwrap_constraint(c)
    c isa FactorConstraint && return FactorConstraint(c.factor, _unwrap_constraint(c.constraint))
    return c
end

function _all_physical_constraints(rep::Representation)
    raw = Any[rep.constraints...]
    if !isnothing(rep.physical_subspace)
        append!(raw, collect(rep.physical_subspace.constraints))
    end
    out = Any[]
    for c in raw
        u = _unwrap_constraint(c)
        any(x -> isequal(x, u), out) || push!(out, u)
    end
    return Tuple(out)
end

function _constraint_family(c)
    c isa FactorConstraint && return _constraint_family(c.constraint)
    kind = _constraint_kind(c)
    kind === :occupation_bounds && return :boson
    kind === :schwinger_occupancy && return :boson
    kind === :single_occupancy && return :fermion
    return :unknown
end

function _validate_ordering(ordering::Vector{Int}, labels::UnitRange{Int}, what::AbstractString)
    sort(ordering) == collect(labels) ||
        throw(ArgumentError("$what ordering must be a permutation of $(collect(labels))"))
    return ordering
end

function _factor_layout(space::AbstractStateSpace, algebra::AbstractOperatorAlgebra, basis::AbstractBasis)
    raw_space = _unwrap_space(space)
    raw_basis = _unwrap_basis(basis)

    if algebra isa SpinAlgebra
        raw_space isa SpinHilbertSpace || throw(ArgumentError("SpinAlgebra requires SpinHilbertSpace"))
        raw_basis isa SpinProductBasis || throw(ArgumentError("SpinAlgebra requires SpinProductBasis"))
        ordering = _validate_ordering(copy(raw_basis.ordering), 1:algebra.nsites, "spin")
        return FactorLayout(:spin, raw_space, algebra, raw_basis, ordering,
                            Dict(label => i for (i, label) in enumerate(ordering)),
                            Int.(round.(2 .* raw_space.spins)))
    elseif algebra isa FermionAlgebra
        raw_space isa FermionFockSpace || throw(ArgumentError("FermionAlgebra requires FermionFockSpace"))
        raw_basis isa FermionOccupationBasis || throw(ArgumentError("FermionAlgebra requires FermionOccupationBasis"))
        algebra.nmodes <= 64 || throw(ArgumentError("the optimized fermion basis supports at most 64 modes"))
        ordering = _validate_ordering(copy(raw_basis.ordering), 1:algebra.nmodes, "fermion")
        return FactorLayout(:fermion, raw_space, algebra, raw_basis, ordering,
                            Dict(label => i for (i, label) in enumerate(ordering)),
                            ones(Int, algebra.nmodes))
    elseif algebra isa MajoranaAlgebra
        raw_space isa FermionFockSpace || throw(ArgumentError("MajoranaAlgebra requires an underlying FermionFockSpace"))
        raw_basis isa FermionOccupationBasis || throw(ArgumentError("MajoranaAlgebra requires FermionOccupationBasis"))
        nmodes = algebra.ngenerators ÷ 2
        raw_space.nmodes == nmodes || throw(ArgumentError("Majorana generator count is inconsistent with the Fock space"))
        nmodes <= 64 || throw(ArgumentError("the optimized fermion basis supports at most 64 modes"))
        ordering = _validate_ordering(copy(raw_basis.ordering), 1:nmodes, "Majorana/fermion")
        return FactorLayout(:majorana, raw_space, algebra, raw_basis, ordering,
                            Dict(label => i for (i, label) in enumerate(ordering)),
                            ones(Int, nmodes))
    elseif algebra isa BosonAlgebra
        raw_space isa BosonFockSpace || throw(ArgumentError("BosonAlgebra requires BosonFockSpace"))
        raw_basis isa BosonOccupationBasis || throw(ArgumentError("BosonAlgebra requires BosonOccupationBasis"))
        ordering = _validate_ordering(copy(raw_basis.ordering), 1:algebra.nmodes, "boson")
        return FactorLayout(:boson, raw_space, algebra, raw_basis, ordering,
                            Dict(label => i for (i, label) in enumerate(ordering)),
                            fill(-1, algebra.nmodes))
    elseif algebra isa CoordinateMomentumAlgebra
        throw(ArgumentError(
            "CoordinateHilbertSpace has no finite default matrix realization. " *
            "Apply CoordinateLadder first or provide a future discretization backend."
        ))
    end

    throw(ArgumentError("unsupported state-space/algebra pair $(typeof(space)), $(typeof(algebra))"))
end

function _representation_factors(rep::Representation)
    if rep.algebra isa CompositeAlgebra
        rep.state_space isa CompositeStateSpace ||
            throw(ArgumentError("CompositeAlgebra requires CompositeStateSpace"))
        rep.basis isa CompositeBasis ||
            throw(ArgumentError("CompositeAlgebra requires CompositeBasis"))
        length(rep.algebra.factors) == length(rep.state_space.factors) == length(rep.basis.factors) ||
            throw(ArgumentError("composite algebra, state-space, and basis factor counts must match"))
        return [
            _factor_layout(rep.state_space.factors[i], rep.algebra.factors[i], rep.basis.factors[i])
            for i in eachindex(rep.algebra.factors)
        ]
    end
    return [_factor_layout(rep.state_space, rep.algebra, rep.basis)]
end

function _family_map(factors::Vector{FactorLayout})
    mapping = Dict{Symbol,Int}()
    for (i, factor) in enumerate(factors)
        if haskey(mapping, factor.family)
            mapping[factor.family] = 0
        else
            mapping[factor.family] = i
        end
    end
    return mapping
end

# ---------------------------------------------------------------------------
# Numerical truncation parsing
# ---------------------------------------------------------------------------

function _cutoffs_from_value(value, nmodes::Int)
    if value isa Integer
        value >= 0 || throw(ArgumentError("boson cutoff must be nonnegative"))
        return fill(Int(value), nmodes)
    elseif value isa AbstractVector || value isa Tuple
        length(value) == nmodes || throw(ArgumentError("boson cutoff vector must have one entry per mode"))
        vals = Int.(collect(value))
        all(>=(0), vals) || throw(ArgumentError("boson cutoffs must be nonnegative"))
        return vals
    elseif value isa AbstractDict
        vals = Int[]
        for i in 1:nmodes
            haskey(value, i) || throw(ArgumentError("boson cutoff dictionary is missing mode $i"))
            push!(vals, Int(value[i]))
        end
        all(>=(0), vals) || throw(ArgumentError("boson cutoffs must be nonnegative"))
        return vals
    elseif value isa NamedTuple
        if haskey(value, :kind) && value.kind === :transported_subspace
            via = haskey(value, :via) ? value.via : :unknown
            throw(ArgumentError(
                "a numerical truncation transported through $via is provenance only and cannot be reinterpreted as a target-basis occupation cutoff; " *
                "attach an explicit target truncation with with_truncation(model, NumericalTruncation(...))"
            ))
        elseif haskey(value, :cutoffs)
            return _cutoffs_from_value(value.cutoffs, nmodes)
        elseif haskey(value, :max_occupation)
            return _cutoffs_from_value(value.max_occupation, nmodes)
        elseif haskey(value, :maximum)
            return _cutoffs_from_value(value.maximum, nmodes)
        end
    end
    return nothing
end

function _cutoffs_from_truncation(truncation, nmodes::Int)
    isnothing(truncation) && return nothing
    truncation isa NumericalTruncation || throw(ArgumentError("invalid numerical truncation object"))
    return _cutoffs_from_value(truncation.cutoffs, nmodes)
end

# ---------------------------------------------------------------------------
# Constraint-aware local basis generation
# ---------------------------------------------------------------------------

function _constraints_for_factor(constraints::Tuple, family::Symbol, factor_index::Int, factors::Vector{FactorLayout})
    selected = Any[]
    normalized_family = family === :majorana ? :fermion : family
    family_count = count(f -> (f.family === :majorana ? :fermion : f.family) === normalized_family, factors)
    for c in constraints
        if c isa FactorConstraint
            c.factor == factor_index || continue
            cf = _constraint_family(c.constraint)
            cf === normalized_family || throw(ArgumentError("factor-local constraint family does not match tensor factor $factor_index"))
            push!(selected, c.constraint)
            continue
        end
        cf = _constraint_family(c)
        if cf === normalized_family
            family_count == 1 || throw(ArgumentError("constraint $(repr(c)) is ambiguous across multiple $normalized_family factors; wrap it in FactorConstraint"))
            push!(selected, c)
        elseif cf === :unknown
            kind = _constraint_kind(c)
            if kind === :transported_constraint || kind === :transported_projector || kind === :transported_physical_subspace
                via = c isa NamedTuple && haskey(c, :via) ? c.via : :unknown
                throw(ArgumentError(
                    "an exact physical constraint/projector was transported through $via but no exact target-basis projector is implemented; " *
                    "the transformed symbolic model is valid, but numerical basis construction requires an explicit target-sector implementation"
                ))
            end
            throw(ArgumentError("cannot construct an exact computational basis for unsupported physical constraint $(repr(c))"))
        end
    end
    return selected
end

function _bounded_compositions(total::Int, mins::Vector{Int}, maxs::Vector{Int})
    n = length(mins)
    out = Vector{Tuple{Vararg{Int}}}()
    current = zeros(Int, n)

    function rec(i::Int, remaining::Int)
        if i == n
            v = remaining
            mins[i] <= v <= maxs[i] || return
            current[i] = v
            push!(out, Tuple(current))
            return
        end
        lo = mins[i]
        hi = min(maxs[i], remaining - sum(mins[i+1:end]))
        hi < lo && return
        for v in lo:hi
            current[i] = v
            rec(i + 1, remaining - v)
        end
    end

    total >= 0 && rec(1, total)
    return out
end

function _combine_blocks(blocks, npositions::Int)
    states = Vector{Tuple{Vararg{Int}}}()
    current = zeros(Int, npositions)

    function rec(k::Int)
        if k > length(blocks)
            push!(states, Tuple(current))
            return
        end
        positions, configs = blocks[k]
        for config in configs
            for (j, pos) in enumerate(positions)
                current[pos] = config[j]
            end
            rec(k + 1)
        end
    end

    rec(1)
    return states
end

function _boson_states(
    factor::FactorLayout,
    constraints,
    truncation,
)
    nmodes = factor.algebra.nmodes
    mins = zeros(Int, nmodes)
    maxs = fill(typemax(Int), nmodes)

    numeric = _cutoffs_from_truncation(truncation, nmodes)
    if numeric !== nothing
        # Truncation vectors are indexed by physical mode label (1:nmodes),
        # while `maxs` follows the explicit basis ordering.
        for (pos, label) in enumerate(factor.ordering)
            maxs[pos] = numeric[label]
        end
    end

    groups = Vector{Tuple{Vector{Int},Int}}()
    grouped = falses(nmodes)

    for c in constraints
        kind = _constraint_kind(c)
        if kind === :occupation_bounds
            mode = Int(c.mode)
            pos = get(factor.positions, mode, 0)
            pos == 0 && throw(ArgumentError("occupation constraint refers to unknown boson mode $mode"))
            mins[pos] = max(mins[pos], Int(c.minimum))
            maxs[pos] = min(maxs[pos], Int(c.maximum))
        elseif kind === :schwinger_occupancy
            labels = Int.(collect(c.modes))
            positions = [get(factor.positions, mode, 0) for mode in labels]
            any(==(0), positions) && throw(ArgumentError("Schwinger constraint refers to an unknown mode"))
            any(grouped[p] for p in positions) &&
                throw(ArgumentError("overlapping fixed-occupation boson constraints are not supported"))
            total_real = c.total
            isinteger(total_real) || throw(ArgumentError("Schwinger occupancy total must be integral"))
            total = Int(round(total_real))
            for p in positions
                grouped[p] = true
                maxs[p] = min(maxs[p], total)
            end
            push!(groups, (positions, total))
        end
    end

    # Every boson mode must be finitely bounded either physically or
    # numerically before a matrix representation exists.
    for p in 1:nmodes
        maxs[p] == typemax(Int) && throw(ArgumentError(
            "boson mode $(factor.ordering[p]) is infinite-dimensional; attach NumericalTruncation " *
            "or use a finite physical constraint"
        ))
        maxs[p] >= mins[p] || throw(ArgumentError("inconsistent boson occupation bounds"))
    end

    blocks = Any[]
    for (positions, total) in groups
        configs = _bounded_compositions(total, mins[positions], maxs[positions])
        isempty(configs) && throw(ArgumentError("boson fixed-occupation constraint has no admissible states"))
        push!(blocks, (positions, configs))
    end
    for p in 1:nmodes
        grouped[p] && continue
        configs = Tuple{Vararg{Int}}[(n,) for n in mins[p]:maxs[p]]
        push!(blocks, ([p], configs))
    end
    sort!(blocks, by=x -> minimum(first(x)))

    occs = _combine_blocks(blocks, nmodes)
    factor.local_maximum .= maxs
    return BosonBasisState[BosonBasisState(x) for x in occs]
end

function _fermion_group_configs(n::Int, total::Int)
    0 <= total <= n || return UInt64[]
    out = UInt64[]
    function rec(pos::Int, remaining::Int, mask::UInt64)
        if pos > n
            remaining == 0 && push!(out, mask)
            return
        end
        remaining < 0 && return
        remaining > n - pos + 1 && return
        rec(pos + 1, remaining, mask)
        remaining > 0 && rec(pos + 1, remaining - 1, mask | (UInt64(1) << (pos - 1)))
    end
    rec(1, total, UInt64(0))
    return out
end

function _fermion_states(factor::FactorLayout, constraints)
    nmodes = factor.state_space.nmodes
    group_specs = Vector{Tuple{Vector{Int},Int}}()
    grouped = falses(nmodes)

    for c in constraints
        kind = _constraint_kind(c)
        kind === :single_occupancy || continue
        labels = Int.(collect(c.modes))
        positions = [get(factor.positions, mode, 0) for mode in labels]
        any(==(0), positions) && throw(ArgumentError("single-occupancy constraint refers to unknown fermion mode"))
        any(grouped[p] for p in positions) &&
            throw(ArgumentError("overlapping fermion occupancy constraints are not supported"))
        total = Int(c.total)
        for p in positions
            grouped[p] = true
        end
        push!(group_specs, (positions, total))
    end

    blocks = Vector{Tuple{Vector{Int},Vector{UInt64}}}()
    for (positions, total) in group_specs
        local_masks = _fermion_group_configs(length(positions), total)
        global_masks = UInt64[]
        for lm in local_masks
            gm = UInt64(0)
            for (j, p) in enumerate(positions)
                ((lm >> (j - 1)) & UInt64(1)) == UInt64(1) && (gm |= UInt64(1) << (p - 1))
            end
            push!(global_masks, gm)
        end
        push!(blocks, (positions, global_masks))
    end

    for p in 1:nmodes
        grouped[p] && continue
        push!(blocks, ([p], UInt64[0, UInt64(1) << (p - 1)]))
    end
    sort!(blocks, by=x -> minimum(first(x)))

    masks = UInt64[0]
    for (_, choices) in blocks
        next = UInt64[]
        sizehint!(next, length(masks) * length(choices))
        for base in masks, choice in choices
            push!(next, base | choice)
        end
        masks = next
    end
    return FermionBasisState[FermionBasisState(x) for x in masks]
end

function _spin_states(factor::FactorLayout)
    spins = factor.state_space.spins
    ranges = [0:Int(round(2 * spins[label])) for label in factor.ordering]
    states = SpinBasisState[]
    current = zeros(Int, length(ranges))
    function rec(i::Int)
        if i > length(ranges)
            push!(states, SpinBasisState(Tuple(current)))
            return
        end
        for level in ranges[i]
            current[i] = level
            rec(i + 1)
        end
    end
    rec(1)
    return states
end

function _factor_states(factor::FactorLayout, family_constraints, truncation)
    if factor.family === :spin
        isempty(family_constraints) || throw(ArgumentError("unsupported explicit spin-space physical constraint"))
        return _spin_states(factor)
    elseif factor.family === :fermion || factor.family === :majorana
        return _fermion_states(factor, family_constraints)
    elseif factor.family === :boson
        return _boson_states(factor, family_constraints, truncation)
    end
    throw(ArgumentError("unsupported computational-basis family $(factor.family)"))
end

function _composite_states(factor_states::Vector)
    states = CompositeBasisState[]
    parts = Vector{Any}(undef, length(factor_states))
    function rec(i::Int)
        if i > length(factor_states)
            push!(states, CompositeBasisState(Tuple(parts)))
            return
        end
        for state in factor_states[i]
            parts[i] = state
            rec(i + 1)
        end
    end
    rec(1)
    return states
end

# ---------------------------------------------------------------------------
# Constraint / truncation predicates used by matrix projection diagnostics
# ---------------------------------------------------------------------------

function _factor_state(state::CompositeBasisState, factor_index::Int)
    return state.parts[factor_index]
end
_factor_state(state::AbstractComputationalState, factor_index::Int) =
    factor_index == 1 ? state : throw(BoundsError())

function _occupation(state::BosonBasisState, factor::FactorLayout, mode::Int)
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("unknown boson mode $mode"))
    return state.occupations[pos]
end

function _occupation(state::FermionBasisState, factor::FactorLayout, mode::Int)
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("unknown fermion mode $mode"))
    return Int((state.bits >> (pos - 1)) & UInt64(1))
end

function _state_occupation(state::AbstractComputationalState, layout::BasisLayout, family::Symbol, mode::Int; factor_index=nothing)
    fidx = if isnothing(factor_index)
        idx = if family === :fermion && !haskey(layout.family_to_factor, :fermion) && haskey(layout.family_to_factor, :majorana)
            layout.family_to_factor[:majorana]
        else
            get(layout.family_to_factor, family, 0)
        end
        idx == 0 && throw(ArgumentError("operator/constraint family $family is absent or ambiguous; use a factor-qualified object"))
        idx
    else
        Int(factor_index)
    end
    1 <= fidx <= length(layout.factors) || throw(BoundsError(layout.factors, fidx))
    factor = layout.factors[fidx]
    expected = factor.family === :majorana ? :fermion : factor.family
    expected === family || throw(ArgumentError("factor $fidx has family $expected, not $family"))
    return _occupation(_factor_state(state, fidx), factor, mode)
end

function _satisfies_physical_constraints(state::AbstractComputationalState, basis::ComputationalBasis)
    for raw in basis.layout.physical_constraints
        factor_index = raw isa FactorConstraint ? raw.factor : nothing
        c = raw isa FactorConstraint ? raw.constraint : raw
        kind = _constraint_kind(c)
        if kind === :occupation_bounds
            n = _state_occupation(state, basis.layout, :boson, Int(c.mode); factor_index=factor_index)
            Int(c.minimum) <= n <= Int(c.maximum) || return false
        elseif kind === :schwinger_occupancy
            total = sum(_state_occupation(state, basis.layout, :boson, Int(m); factor_index=factor_index) for m in c.modes)
            total == Int(round(c.total)) || return false
        elseif kind === :single_occupancy
            total = sum(_state_occupation(state, basis.layout, :fermion, Int(m); factor_index=factor_index) for m in c.modes)
            total == Int(c.total) || return false
        else
            if kind === :transported_constraint || kind === :transported_projector || kind === :transported_physical_subspace
                via = c isa NamedTuple && haskey(c, :via) ? c.via : :unknown
                throw(ArgumentError("transported physical constraint through $via has no numerical target-basis projector"))
            end
            throw(ArgumentError("unsupported physical constraint encountered during projection"))
        end
    end
    return true
end

function _satisfies_numerical_truncation(state::AbstractComputationalState, basis::ComputationalBasis)
    basis.layout.has_numerical_truncation || return true
    for (fidx, factor) in enumerate(basis.layout.factors)
        factor.family === :boson || continue
        fs = _factor_state(state, fidx)
        for (pos, maxn) in enumerate(factor.local_maximum)
            maxn >= 0 || continue
            fs.occupations[pos] <= maxn || return false
        end
    end
    return true
end

function _truncation_for_factor(truncation, factor_index::Int)
    isnothing(truncation) && return nothing
    truncation isa NumericalTruncation && return truncation
    truncation isa FactorTruncation && return truncation.factor == factor_index ? truncation.truncation : nothing
    if truncation isa Tuple || truncation isa AbstractVector
        matches = [entry.truncation for entry in truncation if entry isa FactorTruncation && entry.factor == factor_index]
        length(matches) <= 1 || throw(ArgumentError("multiple numerical truncations target tensor factor $factor_index"))
        return isempty(matches) ? nothing : only(matches)
    end
    throw(ArgumentError("unsupported truncation specification"))
end

# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

"""
    computational_basis(rep)
    computational_basis(model)

Construct the finite numerical basis implied by the representation's exact
physical constraints and numerical truncation.

No infinite bosonic space is silently truncated. If a bosonic degree of
freedom is not finitely bounded, this function throws an informative error.
"""
function computational_basis(rep::Representation)
    constraints = _all_physical_constraints(rep)
    factors = _representation_factors(rep)
    family_map = _family_map(factors)

    factor_states = Any[]
    for (factor_index, factor) in enumerate(factors)
        family_constraints = _constraints_for_factor(constraints, factor.family, factor_index, factors)
        truncation = factor.family === :boson ? _truncation_for_factor(rep.truncation, factor_index) : nothing
        push!(factor_states, _factor_states(factor, family_constraints, truncation))
    end

    states = if length(factor_states) == 1
        factor_states[1]
    else
        _composite_states(factor_states)
    end

    if !isnothing(rep.physical_subspace) && !isnothing(rep.physical_subspace.projector)
        P = rep.physical_subspace.projector
        if P isa Function
            states = [s for s in states if P(s)]
        else
            throw(ArgumentError(
                "matrix-valued physical projectors are not yet convertible to a product-state basis; " *
                "provide a predicate projector or explicit constraints"
            ))
        end
    end

    isempty(states) && throw(ArgumentError("the representation has an empty computational basis"))

    S = eltype(states)
    index = Dict{S,Int}()
    sizehint!(index, length(states))
    for (i, state) in enumerate(states)
        index[state] = i
    end

    layout = BasisLayout(
        factors,
        family_map,
        constraints,
        rep.truncation,
        !isempty(constraints) || !isnothing(rep.physical_subspace),
        !isnothing(rep.truncation),
    )

    basis = ComputationalBasis(rep, states, index, layout)

    expected = representationdimension(rep)
    if expected !== nothing && !layout.has_numerical_truncation && length(basis) != expected
        throw(ArgumentError(
            "constructed basis dimension $(length(basis)) does not match exact physical dimension $expected"
        ))
    end

    return basis
end

computational_basis(model::ManyBodyModel) = computational_basis(model.representation)
