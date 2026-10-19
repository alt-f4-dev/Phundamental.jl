# Canonical structural intermediate representation for Hamiltonians.
# Numerical coupling values are intentionally excluded from structural identity.

"""Root type for coefficient-independent Hamiltonian structures."""
abstract type AbstractModelStructure end

"""One local support addressed by an operator term."""
struct StructuralSupport
    family::Symbol
    index::Int
    factor::Union{Nothing,Int}
end

StructuralSupport(family::Symbol, index::Integer; factor=nothing) = StructuralSupport(family, Int(index), isnothing(factor) ? nothing : Int(factor))
Base.:(==)(left::StructuralSupport, right::StructuralSupport) = left.family == right.family && left.index == right.index && left.factor == right.factor
Base.isequal(left::StructuralSupport, right::StructuralSupport) = left == right
Base.hash(support::StructuralSupport, seed::UInt) = hash(support.factor, hash(support.index, hash(support.family, seed)))

"""One coefficient-independent canonical operator term and its structural diagnostics."""
struct StructuralTerm
    operator::AbstractOperatorExpr
    operator_key::Any
    supports::Tuple
    body_order::Union{Nothing,Int}
    operator_degree::Int
end

"""Hyperedge connecting every local support acted on by one structural Hamiltonian term."""
struct InteractionEdge
    supports::Tuple
    operator_key::Any
end

"""Coefficient-independent interaction hypergraph associated with a `ModelStructure`."""
struct InteractionGraph
    vertices::Tuple
    edges::Tuple
end

"""Stable structural identity consisting of representation context and canonical coefficient-independent terms."""
struct StructuralSignature
    representation::Any
    terms::Tuple
end

Base.:(==)(left::StructuralSignature, right::StructuralSignature) = left.representation == right.representation && left.terms == right.terms
Base.isequal(left::StructuralSignature, right::StructuralSignature) = left == right
Base.hash(signature::StructuralSignature, seed::UInt) = hash(signature.terms, hash(signature.representation, seed))

"""
    ModelStructure

Canonical coefficient-independent Hamiltonian structure in one declared representation context. `signature` defines structural identity; `graph` records the corresponding support hypergraph. Exact transformations may relate distinct `ModelStructure`s without making their signatures equal.
"""
struct ModelStructure{C,T,G,S,M} <: AbstractModelStructure
    context::C
    terms::T
    graph::G
    signature::S
    metadata::M
end

"""
    canonical_model_expression(algebra, expression)

Protocol hook used by structural extraction. Foundation supplies an identity fallback because it cannot depend on the higher `Algebra` layer. `Phundamental.Algebra` extends this hook with exact algebraic canonicalization.
"""
canonical_model_expression(algebra, expression::AbstractOperatorExpr) = expression

_freeze_structural_value(value::Nothing) = nothing
_freeze_structural_value(value::Missing) = missing
_freeze_structural_value(value::Union{Bool,Number,Symbol,AbstractString,Char}) = value
_freeze_structural_value(value::Tuple) = Tuple(_freeze_structural_value(item) for item in value)
_freeze_structural_value(value::NamedTuple) = Tuple((Symbol(name), _freeze_structural_value(item)) for (name, item) in pairs(value))
_freeze_structural_value(value::AbstractArray) = (:array, size(value), Tuple(_freeze_structural_value(item) for item in value))
function _freeze_structural_value(value::AbstractDict)
    entries = [(_freeze_structural_value(key), _freeze_structural_value(item)) for (key, item) in pairs(value)]
    sort!(entries; by=repr)
    return Tuple(entries)
end
_freeze_structural_value(value::Function) = (:function, nameof(typeof(value)))
function _freeze_structural_value(value)
    T = typeof(value)
    fieldcount(T) == 0 && return (:type, nameof(T))
    return (:struct, nameof(T), Tuple((fieldname(T, i), _freeze_structural_value(getfield(value, i))) for i in 1:fieldcount(T)))
end

"""Return a coefficient-independent signature for a physical representation. Numerical truncation and the user-facing representation name are intentionally excluded."""
function representation_signature(representation::Representation)
    return (
        :representation,
        _freeze_structural_value(representation.state_space),
        _freeze_structural_value(representation.physical_subspace),
        _freeze_structural_value(representation.algebra),
        _freeze_structural_value(representation.basis),
        _freeze_structural_value(representation.constraints),
        _freeze_structural_value(representation.gauge_structure),
        _freeze_structural_value(representation.reference_state),
        _freeze_structural_value(representation.ordering),
    )
end

_structural_operator_key(operator::IdentityOperator) = (:identity, operator.label)
_structural_operator_key(operator::SpinX) = (:spin_x, operator.site)
_structural_operator_key(operator::SpinY) = (:spin_y, operator.site)
_structural_operator_key(operator::SpinZ) = (:spin_z, operator.site)
_structural_operator_key(operator::SpinPlus) = (:spin_plus, operator.site)
_structural_operator_key(operator::SpinMinus) = (:spin_minus, operator.site)
_structural_operator_key(operator::BosonAnnihilate) = (:boson_annihilate, operator.mode)
_structural_operator_key(operator::BosonCreate) = (:boson_create, operator.mode)
_structural_operator_key(operator::FermionAnnihilate) = (:fermion_annihilate, operator.mode)
_structural_operator_key(operator::FermionCreate) = (:fermion_create, operator.mode)
_structural_operator_key(operator::MajoranaOperator) = (:majorana, operator.index)
_structural_operator_key(operator::PositionOperator) = (:position, operator.mode)
_structural_operator_key(operator::MomentumOperator) = (:momentum, operator.mode)
_structural_operator_key(operator::NumberOperator) = (:number, operator.kind, operator.mode)
_structural_operator_key(operator::FactorOperator) = (:factor, operator.factor, _structural_operator_key(operator.operator))
_structural_operator_key(operator::ScaledOperator) = _structural_operator_key(operator.operator)
_structural_operator_key(operator::OperatorProduct) = (:product, Tuple(_structural_operator_key(factor) for factor in operator.factors))
function _structural_operator_key(operator::OperatorSum)
    keys = [_structural_operator_key(term) for term in operator.terms]
    sort!(keys; by=repr)
    return (:sum, Tuple(keys))
end
function _structural_operator_key(operator::AbstractPrimitiveOperator)
    T = typeof(operator)
    values = Tuple(_freeze_structural_value(getfield(operator, i)) for i in 1:fieldcount(T))
    return (:primitive, nameof(T), values)
end

function _outer_structural_scalar(operator::ScaledOperator)
    coefficient, core = _outer_structural_scalar(operator.operator)
    return operator.coefficient * coefficient, core
end
_outer_structural_scalar(operator::AbstractOperatorExpr) = (1, operator)

_structural_supports(::IdentityOperator; factor=nothing) = StructuralSupport[]
_structural_supports(operator::Union{SpinX,SpinY,SpinZ,SpinPlus,SpinMinus}; factor=nothing) = StructuralSupport[StructuralSupport(:spin, operator.site; factor=factor)]
_structural_supports(operator::Union{BosonAnnihilate,BosonCreate}; factor=nothing) = StructuralSupport[StructuralSupport(:boson, operator.mode; factor=factor)]
_structural_supports(operator::Union{FermionAnnihilate,FermionCreate}; factor=nothing) = StructuralSupport[StructuralSupport(:fermion, operator.mode; factor=factor)]
_structural_supports(operator::MajoranaOperator; factor=nothing) = StructuralSupport[StructuralSupport(:majorana, operator.index; factor=factor)]
_structural_supports(operator::Union{PositionOperator,MomentumOperator}; factor=nothing) = StructuralSupport[StructuralSupport(:coordinate_momentum, operator.mode; factor=factor)]
_structural_supports(operator::NumberOperator; factor=nothing) = StructuralSupport[StructuralSupport(operator.kind, operator.mode; factor=factor)]
_structural_supports(operator::FactorOperator; factor=nothing) = _structural_supports(operator.operator; factor=operator.factor)
_structural_supports(operator::ScaledOperator; factor=nothing) = _structural_supports(operator.operator; factor=factor)
function _structural_supports(operator::Union{OperatorProduct,OperatorSum}; factor=nothing)
    supports = StructuralSupport[]
    children = operator isa OperatorProduct ? operator.factors : operator.terms
    for child in children
        append!(supports, _structural_supports(child; factor=factor))
    end
    unique!(supports)
    sort!(supports; by=support -> (something(support.factor, 0), string(support.family), support.index))
    return supports
end
_structural_supports(::AbstractPrimitiveOperator; factor=nothing) = StructuralSupport[]

function _structural_term(algebra::AbstractOperatorAlgebra, operator::AbstractOperatorExpr)
    coefficient, core = _outer_structural_scalar(operator)
    coefficient isa Number && iszero(coefficient) && return nothing
    order = applicable(body_order, algebra, core) ? body_order(algebra, core) : nothing
    supports = Tuple(_structural_supports(core))
    return StructuralTerm(core, _structural_operator_key(core), supports, order, operator_degree(core))
end

function _canonical_structural_terms(algebra::AbstractOperatorAlgebra, expression::AbstractOperatorExpr)
    canonical = canonical_model_expression(algebra, expression)
    raw_terms = canonical isa OperatorSum ? canonical.terms : AbstractOperatorExpr[canonical]
    by_key = Dict{Any,StructuralTerm}()
    for operator in raw_terms
        term = _structural_term(algebra, operator)
        isnothing(term) && continue
        by_key[term.operator_key] = term
    end
    keys_sorted = sort!(collect(keys(by_key)); by=repr)
    return Tuple(by_key[key] for key in keys_sorted)
end

function _interaction_graph(terms::Tuple)
    vertices = StructuralSupport[]
    edges = InteractionEdge[]
    for term in terms
        append!(vertices, term.supports)
        push!(edges, InteractionEdge(term.supports, term.operator_key))
    end
    unique!(vertices)
    sort!(vertices; by=support -> (something(support.factor, 0), string(support.family), support.index))
    sort!(edges; by=edge -> repr(edge.operator_key))
    return InteractionGraph(Tuple(vertices), Tuple(edges))
end

function _model_structure(context, algebra::AbstractOperatorAlgebra, expressions; metadata=Dict{Symbol,Any}())
    by_key = Dict{Any,StructuralTerm}()
    for expression in expressions
        for term in _canonical_structural_terms(algebra, expression)
            by_key[term.operator_key] = term
        end
    end
    keys_sorted = sort!(collect(keys(by_key)); by=repr)
    terms = Tuple(by_key[key] for key in keys_sorted)
    signature = StructuralSignature(context, Tuple(term.operator_key for term in terms))
    graph = _interaction_graph(terms)
    stored_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    return ModelStructure(context, terms, graph, signature, stored_metadata)
end

"""Extract the canonical coefficient-independent structure of one Hamiltonian expression in an algebra context."""
function structure(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra)
    context = (:algebra, _freeze_structural_value(algebra))
    return _model_structure(context, algebra, (expression,); metadata=Dict(:scope => :operator_expression))
end

"""Extract the canonical coefficient-independent structure of one Hamiltonian expression in a physical representation context."""
function structure(expression::AbstractOperatorExpr, representation::Representation)
    return _model_structure(representation_signature(representation), representation.algebra, (expression,); metadata=Dict(:scope => :represented_operator_expression))
end

"""Canonicalize a Hamiltonian expression into its `ModelStructure`."""
canonicalize_structure(expression::AbstractOperatorExpr, representation::Representation) = structure(expression, representation)
canonicalize_structure(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra) = structure(expression, algebra)

"""
    structural_signature(model_structure)

Return the stable coefficient-independent `StructuralSignature` associated with `model_structure`. Structural identity is representation-local: equal signatures require both the same canonical operator terms and the same representation context.
"""
structural_signature(model_structure::ModelStructure) = model_structure.signature

"""
    interaction_graph(model_structure)

Return the coefficient-independent `InteractionGraph` associated with `model_structure`. Vertices are local structural supports and each hyperedge records the supports coupled by one canonical Hamiltonian term.
"""
interaction_graph(model_structure::ModelStructure) = model_structure.graph

"""Return true only when two structures have identical canonical terms in the same representation context."""
structurally_equivalent(left::AbstractModelStructure, right::AbstractModelStructure) = structural_signature(left) == structural_signature(right)

function _numeric_linear_terms(algebra::AbstractOperatorAlgebra, expression::AbstractOperatorExpr)
    canonical = canonical_model_expression(algebra, expression)
    terms = canonical isa OperatorSum ? canonical.terms : AbstractOperatorExpr[canonical]
    coefficients = Dict{Any,ComplexF64}()
    operators = Dict{Any,AbstractOperatorExpr}()
    for term in terms
        coefficient, core = _outer_structural_scalar(term)
        coefficient isa Number || return nothing
        value = try
            ComplexF64(coefficient)
        catch
            return nothing
        end
        iszero(value) && continue
        key = _structural_operator_key(core)
        coefficients[key] = get(coefficients, key, 0.0 + 0.0im) + value
        operators[key] = core
    end
    filter!(pair -> !iszero(last(pair)), coefficients)
    return coefficients, operators
end

"""
    structural_span_membership(basis, candidate, algebra; tolerance=1e-10)

Return `true` or `false` when canonical numeric coefficient vectors certify whether `candidate` belongs to the real span of `basis`, and `missing` when the current symbolic coefficients cannot be reduced to numeric linear coordinates.
"""
function structural_span_membership(basis::AbstractVector{<:AbstractOperatorExpr}, candidate::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra; tolerance::Real=1e-10)
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative"))
    basis_terms = [_numeric_linear_terms(algebra, operator) for operator in basis]
    candidate_terms = _numeric_linear_terms(algebra, candidate)
    any(isnothing, basis_terms) && return missing
    isnothing(candidate_terms) && return missing

    keys_all = Any[]
    for item in basis_terms
        append!(keys_all, keys(first(item)))
    end
    append!(keys_all, keys(first(candidate_terms)))
    unique!(keys_all)
    sort!(keys_all; by=repr)
    isempty(keys_all) && return true

    B = zeros(Float64, 2 * length(keys_all), length(basis))
    for (column, item) in enumerate(basis_terms)
        coefficients = first(item)
        for (row, key) in enumerate(keys_all)
            value = get(coefficients, key, 0.0 + 0.0im)
            B[row, column] = real(value)
            B[length(keys_all) + row, column] = imag(value)
        end
    end

    candidate_coefficients = first(candidate_terms)
    c = zeros(Float64, 2 * length(keys_all))
    for (row, key) in enumerate(keys_all)
        value = get(candidate_coefficients, key, 0.0 + 0.0im)
        c[row] = real(value)
        c[length(keys_all) + row] = imag(value)
    end

    rank_basis = rank(B; atol=tolerance, rtol=tolerance)
    rank_augmented = rank(hcat(B, c); atol=tolerance, rtol=tolerance)
    return rank_augmented == rank_basis
end

"""
    structural_span_coordinates(basis, targets, algebra; tolerance=1e-10)

Recover the real canonical coordinates of one or several physical operator expressions in a fixed Hermitian basis. Multiple targets share one coefficient matrix, rank check, and factorization. Numerical residuals use the joint real/imaginary canonical coordinates, and symbolic nonnumeric coefficients return `missing`.
"""
function structural_span_coordinates(basis::AbstractVector{<:AbstractOperatorExpr}, targets::AbstractVector{<:AbstractOperatorExpr},
                                     algebra::AbstractOperatorAlgebra; tolerance::Real=1e-10)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("tolerance must be finite and nonnegative"))
    isempty(basis) && throw(ArgumentError("coordinate recovery requires a nonempty Hermitian basis"))
    isempty(targets) && throw(ArgumentError("coordinate recovery requires at least one target operator"))
    maps = [_numeric_linear_terms(algebra, operator) for operator in basis]
    rhs_maps = [_numeric_linear_terms(algebra, operator) for operator in targets]
    (any(isnothing, maps) || any(isnothing, rhs_maps)) && return missing
    keys_all = Any[]
    for entry in (maps..., rhs_maps...)
        append!(keys_all, keys(first(entry)))
    end
    unique!(keys_all)
    sort!(keys_all; by=repr)
    isempty(keys_all) && throw(ArgumentError("basis and target operators are identically zero"))
    n = length(keys_all)
    B = zeros(Float64, 2n, length(basis))
    C = zeros(Float64, 2n, length(targets))
    for (j, entry) in enumerate(maps)
        for (i, key) in enumerate(keys_all)
            value = get(first(entry), key, 0.0 + 0.0im)
            B[i, j] = real(value)
            B[n + i, j] = imag(value)
        end
    end
    for (j, entry) in enumerate(rhs_maps)
        for (i, key) in enumerate(keys_all)
            value = get(first(entry), key, 0.0 + 0.0im)
            C[i, j] = real(value)
            C[n + i, j] = imag(value)
        end
    end
    rank_basis = rank(B; atol=tolerance, rtol=tolerance)
    rank_basis == length(basis) || throw(ArgumentError("the supplied invariant basis is numerically rank deficient at the requested tolerance"))
    coefficients = Matrix{Float64}(B \ C)
    all(isfinite, coefficients) || throw(ArgumentError("canonical basis coordinates are nonfinite"))
    residuals = Float64[norm(B * coefficients[:, j] - C[:, j]) for j in axes(C, 2)]
    norms = Float64[norm(C[:, j]) for j in axes(C, 2)]
    relative_residuals = residuals ./ max.(norms, eps(Float64))
    within_span = BitVector(residuals .<= tolerance .* max.(1.0, norms))
    return (; coefficients, residuals, relative_residuals, within_span, rank=rank_basis)
end

function structural_span_coordinates(basis::AbstractVector{<:AbstractOperatorExpr}, candidate::AbstractOperatorExpr,
                                     algebra::AbstractOperatorAlgebra; tolerance::Real=1e-10)
    result = structural_span_coordinates(basis, AbstractOperatorExpr[candidate], algebra; tolerance=tolerance)
    result === missing && return missing
    return (; coefficients=collect(result.coefficients[:, 1]), residual=result.residuals[1],
            relative_residual=result.relative_residuals[1], rank=result.rank, within_span=result.within_span[1])
end
