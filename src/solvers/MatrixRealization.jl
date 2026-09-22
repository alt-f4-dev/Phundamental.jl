# Sparse/dense numerical realization of the symbolic noncommutative operator
# tree. Products are applied right-to-left in the ambient algebra and are
# projected onto the computational basis only after the complete product has
# acted. This avoids the common error P A P B P != P A B P for constrained
# representations.

"""Rich result returned by `realize`."""
struct MatrixRealizationResult{M,B,D}
    matrix::M
    basis::B
    diagnostics::D
end

"""Internal numerical context, including cached local displacement matrices."""
mutable struct RealizationContext{T<:Number,B<:ComputationalBasis}
    basis::B
    hbar::T
    displacement_cache::Dict{Tuple{Int,ComplexF64,Int},Matrix{ComplexF64}}
end

function RealizationContext(basis::ComputationalBasis; hbar::Number=1)
    return RealizationContext(basis, hbar, Dict{Tuple{Int,ComplexF64,Int},Matrix{ComplexF64}}())
end

# ---------------------------------------------------------------------------
# State-factor routing
# ---------------------------------------------------------------------------

function _family_factor(layout::BasisLayout, family::Symbol)
    if family === :fermion && !haskey(layout.family_to_factor, :fermion) && haskey(layout.family_to_factor, :majorana)
        return layout.family_to_factor[:majorana]
    end
    idx = get(layout.family_to_factor, family, 0)
    idx == 0 && throw(ArgumentError("operator family $family is absent or ambiguous in this representation; use atfactor(factor, operator) when multiple factors share a family"))
    return idx
end

function _replace_factor_state(state::CompositeBasisState, idx::Int, replacement::AbstractComputationalState)
    parts = collect(state.parts)
    parts[idx] = replacement
    return CompositeBasisState(Tuple(parts))
end

_replace_factor_state(::AbstractComputationalState, idx::Int, replacement::AbstractComputationalState) =
    idx == 1 ? replacement : throw(BoundsError())

function _local_state(state::AbstractComputationalState, ctx::RealizationContext, family::Symbol)
    idx = _family_factor(ctx.basis.layout, family)
    return idx, _factor_state(state, idx), ctx.basis.layout.factors[idx]
end

# ---------------------------------------------------------------------------
# Primitive local actions
# ---------------------------------------------------------------------------

function _spin_transition(
    op::Union{SpinPlus,SpinMinus},
    state::SpinBasisState,
    factor::FactorLayout,
    hbar::Number,
)
    pos = get(factor.positions, op.site, 0)
    pos == 0 && throw(ArgumentError("spin operator refers to unknown site $(op.site)"))
    S = factor.state_space.spins[op.site]
    level = state.levels[pos]
    m = -S + level

    if op isa SpinPlus
        level >= Int(round(2S)) && return nothing
        amp = hbar * sqrt(S * (S + 1) - m * (m + 1))
        newlevel = level + 1
    else
        level <= 0 && return nothing
        amp = hbar * sqrt(S * (S + 1) - m * (m - 1))
        newlevel = level - 1
    end

    levels = collect(state.levels)
    levels[pos] = newlevel
    return SpinBasisState(Tuple(levels)), amp
end

function _fermion_transition(
    create::Bool,
    mode::Int,
    state::FermionBasisState,
    factor::FactorLayout,
)
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("fermion operator refers to unknown mode $mode"))
    bit = UInt64(1) << (pos - 1)
    occupied = (state.bits & bit) != 0
    create == occupied && return nothing

    preceding = bit - UInt64(1)
    sign = isodd(count_ones(state.bits & preceding)) ? -1 : 1
    newbits = create ? (state.bits | bit) : (state.bits & ~bit)
    return FermionBasisState(newbits), sign
end

function _boson_transition(
    create::Bool,
    mode::Int,
    state::BosonBasisState,
    factor::FactorLayout,
)
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("boson operator refers to unknown mode $mode"))
    n = state.occupations[pos]
    !create && n == 0 && return nothing
    newn = create ? n + 1 : n - 1
    amp = create ? sqrt(n + 1) : sqrt(n)
    occ = collect(state.occupations)
    occ[pos] = newn
    return BosonBasisState(Tuple(occ)), amp
end

function _boson_local_cutoff(ctx::RealizationContext, factor_index::Int, mode::Int)
    factor = ctx.basis.layout.factors[factor_index]
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("unknown boson mode $mode"))
    maxn = factor.local_maximum[pos]
    maxn >= 0 || throw(ArgumentError("bosonic displacement requires a finite local cutoff"))
    return maxn
end

function _displacement_matrix(ctx::RealizationContext, factor_index::Int, mode::Int, alpha::Number)
    maxn = _boson_local_cutoff(ctx, factor_index, mode)
    key = (mode, ComplexF64(alpha), maxn)
    return get!(ctx.displacement_cache, key) do
        d = maxn + 1
        a = zeros(ComplexF64, d, d)
        for n in 1:maxn
            # zero-based occupation n appears at Julia column n+1.
            a[n, n + 1] = sqrt(n)
        end
        adag = adjoint(a)
        generator = ComplexF64(alpha) * adag - conj(ComplexF64(alpha)) * a
        exp(generator)
    end
end

function _single_transition(state, amp)
    return Pair{AbstractComputationalState,ComplexF64}[
        state => ComplexF64(amp)
    ]
end

function _apply_primitive(op::IdentityOperator, state::AbstractComputationalState, ctx::RealizationContext)
    return _single_transition(state, 1)
end

function _apply_primitive(op::SpinZ, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :spin)
    local_state isa SpinBasisState || throw(ArgumentError("invalid spin basis state"))
    pos = factor.positions[op.site]
    S = factor.state_space.spins[op.site]
    m = -S + local_state.levels[pos]
    return _single_transition(state, ctx.hbar * m)
end

function _apply_primitive(op::Union{SpinPlus,SpinMinus}, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :spin)
    transition = _spin_transition(op, local_state, factor, ctx.hbar)
    transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
    newlocal, amp = transition
    return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
end

function _apply_primitive(op::SpinX, state::AbstractComputationalState, ctx::RealizationContext)
    return _combine_transitions(
        _scale_transitions(_apply_primitive(SpinPlus(op.site), state, ctx), 1/2),
        _scale_transitions(_apply_primitive(SpinMinus(op.site), state, ctx), 1/2),
    )
end

function _apply_primitive(op::SpinY, state::AbstractComputationalState, ctx::RealizationContext)
    return _combine_transitions(
        _scale_transitions(_apply_primitive(SpinPlus(op.site), state, ctx), 1/(2im)),
        _scale_transitions(_apply_primitive(SpinMinus(op.site), state, ctx), -1/(2im)),
    )
end

function _apply_primitive(op::FermionAnnihilate, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :fermion)
    local_state isa FermionBasisState || throw(ArgumentError("invalid fermion basis state"))
    transition = _fermion_transition(false, op.mode, local_state, factor)
    transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
    newlocal, amp = transition
    return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
end

function _apply_primitive(op::FermionCreate, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :fermion)
    local_state isa FermionBasisState || throw(ArgumentError("invalid fermion basis state"))
    transition = _fermion_transition(true, op.mode, local_state, factor)
    transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
    newlocal, amp = transition
    return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
end

function _apply_primitive(op::BosonAnnihilate, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :boson)
    local_state isa BosonBasisState || throw(ArgumentError("invalid boson basis state"))
    transition = _boson_transition(false, op.mode, local_state, factor)
    transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
    newlocal, amp = transition
    return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
end

function _apply_primitive(op::BosonCreate, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :boson)
    local_state isa BosonBasisState || throw(ArgumentError("invalid boson basis state"))
    transition = _boson_transition(true, op.mode, local_state, factor)
    newlocal, amp = transition
    return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
end

function _apply_primitive(op::NumberOperator, state::AbstractComputationalState, ctx::RealizationContext)
    if op.kind === :boson
        _, local_state, factor = _local_state(state, ctx, :boson)
        n = _occupation(local_state, factor, op.mode)
    elseif op.kind === :fermion
        _, local_state, factor = _local_state(state, ctx, :fermion)
        n = _occupation(local_state, factor, op.mode)
    else
        throw(ArgumentError("unsupported number-operator kind $(op.kind)"))
    end
    return _single_transition(state, n)
end

function _apply_primitive(op::MajoranaOperator, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :majorana)
    local_state isa FermionBasisState || throw(ArgumentError("Majorana representation requires fermion occupation states"))
    mode = (op.index + 1) ÷ 2
    annih = _fermion_transition(false, mode, local_state, factor)
    create = _fermion_transition(true, mode, local_state, factor)
    out = Pair{AbstractComputationalState,ComplexF64}[]

    # gamma_(2j-1) = c_j + c_j^dagger
    # gamma_(2j)   = -i(c_j - c_j^dagger)
    ca = isodd(op.index) ? 1.0 + 0im : -1im
    cc = isodd(op.index) ? 1.0 + 0im :  1im
    if annih !== nothing
        s, a = annih
        push!(out, _replace_factor_state(state, fidx, s) => ComplexF64(ca * a))
    end
    if create !== nothing
        s, a = create
        push!(out, _replace_factor_state(state, fidx, s) => ComplexF64(cc * a))
    end
    return out
end

function _apply_primitive(op::SqrtNumberFactor, state::AbstractComputationalState, ctx::RealizationContext)
    family = op.kind
    if family === :boson
        _, local_state, factor = _local_state(state, ctx, :boson)
        n = _occupation(local_state, factor, op.mode)
    elseif family === :fermion
        _, local_state, factor = _local_state(state, ctx, :fermion)
        n = _occupation(local_state, factor, op.mode)
    else
        throw(ArgumentError("unsupported square-root number family $family"))
    end
    radicand = op.offset + op.coefficient * n
    if real(radicand) < -100eps(Float64) && isreal(radicand)
        throw(DomainError(radicand, "negative square-root occupation factor"))
    end
    abs(radicand) < 100eps(Float64) && (radicand = zero(radicand))
    return _single_transition(state, sqrt(complex(radicand)))
end

function _apply_primitive(op::BosonicDisplacementFactor, state::AbstractComputationalState, ctx::RealizationContext)
    fidx, local_state, factor = _local_state(state, ctx, :boson)
    local_state isa BosonBasisState || throw(ArgumentError("invalid boson basis state"))
    pos = get(factor.positions, op.mode, 0)
    pos == 0 && throw(ArgumentError("displacement refers to unknown boson mode $(op.mode)"))
    n = local_state.occupations[pos]
    D = _displacement_matrix(ctx, fidx, op.mode, op.alpha)
    n + 1 <= size(D, 2) || return Pair{AbstractComputationalState,ComplexF64}[]

    out = Pair{AbstractComputationalState,ComplexF64}[]
    for m in 0:(size(D, 1) - 1)
        amp = D[m + 1, n + 1]
        iszero(amp) && continue
        occ = collect(local_state.occupations)
        occ[pos] = m
        newlocal = BosonBasisState(Tuple(occ))
        push!(out, _replace_factor_state(state, fidx, newlocal) => amp)
    end
    return out
end

function _apply_primitive(op::Union{PositionOperator,MomentumOperator}, state::AbstractComputationalState, ctx::RealizationContext)
    throw(ArgumentError(
        "coordinate/momentum primitives have no default finite realization; apply CoordinateLadder first"
    ))
end

function _apply_primitive(op::AbstractPrimitiveOperator, state::AbstractComputationalState, ctx::RealizationContext)
    throw(ArgumentError("no numerical realization is defined for primitive operator $(typeof(op))"))
end

# ---------------------------------------------------------------------------
# Expression action
# ---------------------------------------------------------------------------

function _accumulate!(dest::Dict{AbstractComputationalState,ComplexF64}, transitions, scale::Number=1)
    s = ComplexF64(scale)
    for pair in transitions
        state, amp = pair.first, pair.second
        value = s * amp
        iszero(value) && continue
        dest[state] = get(dest, state, 0.0 + 0im) + value
        iszero(dest[state]) && delete!(dest, state)
    end
    return dest
end

function _scale_transitions(transitions, scale::Number)
    s = ComplexF64(scale)
    return Pair{AbstractComputationalState,ComplexF64}[p.first => s * p.second for p in transitions]
end

function _combine_transitions(groups...)
    acc = Dict{AbstractComputationalState,ComplexF64}()
    for group in groups
        _accumulate!(acc, group)
    end
    return Pair{AbstractComputationalState,ComplexF64}[k => v for (k, v) in acc]
end

function _apply_expr(op::AbstractPrimitiveOperator, state::AbstractComputationalState, ctx::RealizationContext)
    return _apply_primitive(op, state, ctx)
end

function _apply_expr(op::ScaledOperator, state::AbstractComputationalState, ctx::RealizationContext)
    return _scale_transitions(_apply_expr(op.operator, state, ctx), op.coefficient)
end

function _apply_expr(op::OperatorSum, state::AbstractComputationalState, ctx::RealizationContext)
    acc = Dict{AbstractComputationalState,ComplexF64}()
    for term in op.terms
        _accumulate!(acc, _apply_expr(term, state, ctx))
    end
    return Pair{AbstractComputationalState,ComplexF64}[k => v for (k, v) in acc]
end

function _apply_expr(op::OperatorProduct, state::AbstractComputationalState, ctx::RealizationContext)
    current = Dict{AbstractComputationalState,ComplexF64}(state => (1.0 + 0im))

    # Right-most factor acts first. Intermediate states are NOT projected onto
    # the computational basis. Only the final result is projected by realize().
    for factor in reverse(op.factors)
        next = Dict{AbstractComputationalState,ComplexF64}()
        for (intermediate, amplitude) in current
            _accumulate!(next, _apply_expr(factor, intermediate, ctx), amplitude)
        end
        isempty(next) && return Pair{AbstractComputationalState,ComplexF64}[]
        current = next
    end

    return Pair{AbstractComputationalState,ComplexF64}[k => v for (k, v) in current]
end

# ---------------------------------------------------------------------------
# Matrix realization and matrix-free map
# ---------------------------------------------------------------------------

function _partition_leakage(
    state::AbstractComputationalState,
    amp::ComplexF64,
    basis::ComputationalBasis,
)
    if !_satisfies_physical_constraints(state, basis)
        return abs2(amp), 0.0
    elseif !_satisfies_numerical_truncation(state, basis)
        return 0.0, abs2(amp)
    end
    # A state can also be absent because of a predicate projector.
    return abs2(amp), 0.0
end

"""
    realize(operator, representation_or_model; ...)

Build `P O P`, where `P` projects onto the finite computational basis. For
polynomial operator products, projection occurs only after the complete
product acts, preserving the ambient operator product before projection.

`physical_leakage` and `truncation_leakage` report the largest column-wise
norm outside the exact physical sector and outside the chosen numerical
truncation, respectively.
"""
function realize(
    operator::AbstractOperatorExpr,
    basis::ComputationalBasis;
    hbar::Number=1,
    sparse::Bool=true,
    check_physical_closure::Bool=false,
    closure_tol::Real=1e-10,
)
    dim = length(basis)
    ctx = RealizationContext(basis; hbar=hbar)

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    sizehint!(rows, max(dim, 16))
    sizehint!(cols, max(dim, 16))
    sizehint!(vals, max(dim, 16))

    max_physical_leakage2 = 0.0
    max_truncation_leakage2 = 0.0

    for col in 1:dim
        input_state = basis.states[col]
        transitions = _apply_expr(operator, input_state, ctx)
        physical_leak2 = 0.0
        trunc_leak2 = 0.0

        for pair in transitions
            output_state, amp = pair.first, pair.second
            row = get(basis.index, output_state, 0)
            if row != 0
                push!(rows, row)
                push!(cols, col)
                push!(vals, amp)
            else
                p, t = _partition_leakage(output_state, amp, basis)
                physical_leak2 += p
                trunc_leak2 += t
            end
        end
        max_physical_leakage2 = max(max_physical_leakage2, physical_leak2)
        max_truncation_leakage2 = max(max_truncation_leakage2, trunc_leak2)
    end

    Msp = SparseArrays.sparse(rows, cols, vals, dim, dim)
    M = sparse ? Msp : Matrix(Msp)

    diagnostics = Dict{Symbol,Any}(
        :dimension => dim,
        :nnz => nnz(Msp),
        :density => nnz(Msp) / dim^2,
        :physical_leakage => sqrt(max_physical_leakage2),
        :truncation_leakage => sqrt(max_truncation_leakage2),
        :projection => :final_state_only,
        :hbar => hbar,
    )

    if check_physical_closure && diagnostics[:physical_leakage] > closure_tol
        throw(ArgumentError(
            "operator does not preserve the exact physical subspace: " *
            "maximum leakage $(diagnostics[:physical_leakage]) > $closure_tol"
        ))
    end

    return MatrixRealizationResult(M, basis, diagnostics)
end

function realize(
    operator::AbstractOperatorExpr,
    rep::Representation;
    kwargs...,
)
    return realize(operator, computational_basis(rep); kwargs...)
end

function realize(
    operator::AbstractOperatorExpr,
    model::ManyBodyModel;
    kwargs...,
)
    return realize(operator, computational_basis(model); kwargs...)
end

function realize(
    operator::AbstractMatrix,
    basis::ComputationalBasis;
    sparse::Bool=true,
    kwargs...,
)
    size(operator, 1) == size(operator, 2) == length(basis) ||
        throw(DimensionMismatch("matrix operator dimension does not match computational basis"))
    M = sparse ? SparseArrays.sparse(ComplexF64.(operator)) : Matrix{ComplexF64}(operator)
    diagnostics = Dict{Symbol,Any}(
        :dimension => length(basis),
        :nnz => count(x -> !iszero(x), operator),
        :density => count(x -> !iszero(x), operator) / length(basis)^2,
        :physical_leakage => 0.0,
        :truncation_leakage => 0.0,
        :projection => :already_realized,
    )
    return MatrixRealizationResult(M, basis, diagnostics)
end

"""Convenience function returning only the realized matrix."""
matrix(operator, basis_or_model; kwargs...) = realize(operator, basis_or_model; kwargs...).matrix

"""Realize the Hamiltonian carried by a model."""
function hamiltonian_matrix(model::ManyBodyModel; basis=computational_basis(model), kwargs...)
    return realize(model.hamiltonian, basis; kwargs...)
end

"""
    SymbolicLinearOperator

Matrix-free numerical action of an operator expression on a fixed
computational basis. The realization context is retained, so local expensive
objects such as displacement matrices are cached across Krylov iterations.
"""
mutable struct SymbolicLinearOperator{O,B,C}
    operator::O
    basis::B
    context::C
end

function SymbolicLinearOperator(operator::AbstractOperatorExpr, basis::ComputationalBasis; hbar::Number=1)
    return SymbolicLinearOperator(operator, basis, RealizationContext(basis; hbar=hbar))
end

Base.size(A::SymbolicLinearOperator) = (length(A.basis), length(A.basis))
Base.size(A::SymbolicLinearOperator, d::Integer) = size(A)[d]
Base.eltype(::Type{<:SymbolicLinearOperator}) = ComplexF64
Base.eltype(::SymbolicLinearOperator) = ComplexF64

function LinearAlgebra.mul!(y::AbstractVector, A::SymbolicLinearOperator, x::AbstractVector)
    dim = length(A.basis)
    length(x) == dim == length(y) || throw(DimensionMismatch("matrix-free multiply dimension mismatch"))
    fill!(y, zero(eltype(y)))

    for col in 1:dim
        xcol = x[col]
        iszero(xcol) && continue
        transitions = _apply_expr(A.operator, A.basis.states[col], A.context)
        for pair in transitions
            row = get(A.basis.index, pair.first, 0)
            row == 0 && continue
            y[row] += pair.second * xcol
        end
    end
    return y
end

function Base.:*(A::SymbolicLinearOperator, x::AbstractVector)
    y = zeros(promote_type(ComplexF64, eltype(x)), size(A, 1))
    return LinearAlgebra.mul!(y, A, x)
end

# ---------------------------------------------------------------------------
# Tensor-factor-qualified primitive realization
# ---------------------------------------------------------------------------

function _qualified_local_state(state::AbstractComputationalState, ctx::RealizationContext, factor_index::Int, family::Symbol)
    1 <= factor_index <= length(ctx.basis.layout.factors) || throw(BoundsError(ctx.basis.layout.factors, factor_index))
    factor = ctx.basis.layout.factors[factor_index]
    actual = factor.family === :majorana ? :fermion : factor.family
    expected = family === :majorana ? :fermion : family
    actual === expected || throw(ArgumentError("tensor factor $factor_index has operator family $actual, not $expected"))
    return _factor_state(state, factor_index), factor
end

function _preceding_fermion_parity(state::AbstractComputationalState, ctx::RealizationContext, factor_index::Int)
    parity = false
    for idx in 1:(factor_index - 1)
        factor = ctx.basis.layout.factors[idx]
        factor.family in (:fermion, :majorana) || continue
        factor_state = _factor_state(state, idx)
        factor_state isa FermionBasisState || continue
        parity ⊻= isodd(count_ones(factor_state.bits))
    end
    return parity ? -1.0 : 1.0
end

function _apply_primitive(op::FactorOperator, state::AbstractComputationalState, ctx::RealizationContext)
    inner = op.operator
    fidx = op.factor
    if inner isa IdentityOperator
        return _single_transition(state, 1)
    elseif inner isa SpinZ
        local_state, factor = _qualified_local_state(state, ctx, fidx, :spin)
        local_state isa SpinBasisState || throw(ArgumentError("invalid spin basis state"))
        pos = factor.positions[inner.site]
        S = factor.state_space.spins[inner.site]
        m = -S + local_state.levels[pos]
        return _single_transition(state, ctx.hbar * m)
    elseif inner isa SpinPlus || inner isa SpinMinus
        local_state, factor = _qualified_local_state(state, ctx, fidx, :spin)
        transition = _spin_transition(inner, local_state, factor, ctx.hbar)
        transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
        newlocal, amp = transition
        return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
    elseif inner isa SpinX
        return _combine_transitions(_scale_transitions(_apply_primitive(FactorOperator(fidx, SpinPlus(inner.site)), state, ctx), 1/2),
                                    _scale_transitions(_apply_primitive(FactorOperator(fidx, SpinMinus(inner.site)), state, ctx), 1/2))
    elseif inner isa SpinY
        return _combine_transitions(_scale_transitions(_apply_primitive(FactorOperator(fidx, SpinPlus(inner.site)), state, ctx), 1/(2im)),
                                    _scale_transitions(_apply_primitive(FactorOperator(fidx, SpinMinus(inner.site)), state, ctx), -1/(2im)))
    elseif inner isa FermionAnnihilate || inner isa FermionCreate
        local_state, factor = _qualified_local_state(state, ctx, fidx, :fermion)
        local_state isa FermionBasisState || throw(ArgumentError("invalid fermion basis state"))
        transition = _fermion_transition(inner isa FermionCreate, inner.mode, local_state, factor)
        transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
        newlocal, amp = transition
        amp *= _preceding_fermion_parity(state, ctx, fidx)
        return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
    elseif inner isa BosonAnnihilate || inner isa BosonCreate
        local_state, factor = _qualified_local_state(state, ctx, fidx, :boson)
        local_state isa BosonBasisState || throw(ArgumentError("invalid boson basis state"))
        transition = _boson_transition(inner isa BosonCreate, inner.mode, local_state, factor)
        transition === nothing && return Pair{AbstractComputationalState,ComplexF64}[]
        newlocal, amp = transition
        return _single_transition(_replace_factor_state(state, fidx, newlocal), amp)
    elseif inner isa NumberOperator
        family = inner.kind
        local_state, factor = _qualified_local_state(state, ctx, fidx, family)
        return _single_transition(state, _occupation(local_state, factor, inner.mode))
    elseif inner isa MajoranaOperator
        local_state, factor = _qualified_local_state(state, ctx, fidx, :fermion)
        local_state isa FermionBasisState || throw(ArgumentError("Majorana representation requires fermion occupation states"))
        mode = (inner.index + 1) ÷ 2
        sign = _preceding_fermion_parity(state, ctx, fidx)
        annih = _fermion_transition(false, mode, local_state, factor)
        create = _fermion_transition(true, mode, local_state, factor)
        out = Pair{AbstractComputationalState,ComplexF64}[]
        ca = isodd(inner.index) ? sign + 0im : -1im * sign
        cc = isodd(inner.index) ? sign + 0im : 1im * sign
        if annih !== nothing
            s, a = annih
            push!(out, _replace_factor_state(state, fidx, s) => ComplexF64(ca * a))
        end
        if create !== nothing
            s, a = create
            push!(out, _replace_factor_state(state, fidx, s) => ComplexF64(cc * a))
        end
        return out
    end
    throw(ArgumentError("no factor-qualified numerical realization is defined for $(typeof(inner))"))
end
