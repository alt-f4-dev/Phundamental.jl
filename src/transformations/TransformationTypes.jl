# Shared transformation types and generic operator-map machinery.

"""Root type for all representation transformations."""
abstract type AbstractTransformation end

"""Primary mathematical relation implemented by a transformation."""
@enum RelationType begin
    UnitaryEquivalence
    CanonicalMap
    AlgebraIsomorphism
    ConstrainedEmbedding
    SimilarityEquivalence
    GaugeRedundantEmbedding
    CompositeRelation
end

"""
    TransformationCertificate

Immutable mathematical metadata attached to a known transformation.  Users do
not construct this object for standard transformations; each implementation
provides its own `certificate(::Transformation)` method.
"""
struct TransformationCertificate
    exact::Bool
    invertible::Bool
    relation::RelationType
    locality::Symbol
    requires_constraints::Bool
    gauge_redundant::Bool
    approximate::Bool
    notes::Tuple
end

TransformationCertificate(
    exact::Bool,
    invertible::Bool,
    relation::RelationType,
    locality::Symbol;
    requires_constraints::Bool=false,
    gauge_redundant::Bool=false,
    approximate::Bool=false,
    notes=(),
) = TransformationCertificate(
    exact,
    invertible,
    relation,
    locality,
    requires_constraints,
    gauge_redundant,
    approximate,
    Tuple(notes),
)

# Required transformation interface.
transformation_name(T::AbstractTransformation) = nameof(typeof(T))
certificate(T::AbstractTransformation) =
    throw(ArgumentError("no certificate is defined for $(transformation_name(T))"))
applicable(::AbstractTransformation, ::ManyBodyModel) = false
target_representation(T::AbstractTransformation, ::ManyBodyModel) =
    throw(ArgumentError("no target representation is defined for $(transformation_name(T))"))
transform_generator(T::AbstractTransformation, op::AbstractPrimitiveOperator) =
    throw(ArgumentError("$(transformation_name(T)) has no generator rule for $(typeof(op))"))

"""Value of hbar used by algebra canonicalization for this transformation."""
transformation_hbar(::AbstractTransformation) = 1

"""Whether a concrete inverse implementation is available in software."""
has_inverse(::AbstractTransformation) = false
inverse(T::AbstractTransformation) =
    throw(ArgumentError("no implemented inverse for $(transformation_name(T))"))

# Identity always maps to identity.
transform_generator(::AbstractTransformation, op::IdentityOperator) = op

"""
    transform_operator(T, expr)

Recursively extend a primitive generator map to a general noncommutative
operator expression.  This implements linearity and multiplicativity of the
algebra map.
"""
transform_operator(T::AbstractTransformation, op::AbstractPrimitiveOperator) =
    transform_generator(T, op)

transform_operator(T::AbstractTransformation, op::ScaledOperator) =
    ScaledOperator(op.coefficient, transform_operator(T, op.operator))

transform_operator(T::AbstractTransformation, op::OperatorSum) =
    OperatorSum(AbstractOperatorExpr[transform_operator(T, x) for x in op.terms])

transform_operator(T::AbstractTransformation, op::OperatorProduct) =
    OperatorProduct(AbstractOperatorExpr[transform_operator(T, x) for x in op.factors])

"""Canonicalize a mapped expression in the target operator algebra."""
function canonicalize_transformed(
    T::AbstractTransformation,
    target::Representation,
    expr::AbstractOperatorExpr,
)
    return canonicalize(
        target.algebra,
        expr;
        hbar=transformation_hbar(T),
    )
end

"""Transform and canonicalize an arbitrary operator expression."""
function transform_operator_canonical(
    T::AbstractTransformation,
    target::Representation,
    expr::AbstractOperatorExpr,
)
    return canonicalize_transformed(T, target, transform_operator(T, expr))
end

"""Transform a Hamiltonian expression."""
transform_hamiltonian(
    T::AbstractTransformation,
    target::Representation,
    H::AbstractOperatorExpr,
) = transform_operator_canonical(T, target, H)

"""Transform a physical observable with the same representation map."""
transform_observable(
    T::AbstractTransformation,
    target::Representation,
    O::AbstractOperatorExpr,
) = transform_operator_canonical(T, target, O)

"""Transformation-specific validation hook."""
validate_parameters(::AbstractTransformation, ::ManyBodyModel) = true

"""Optional additional validation hook after construction of the target model."""
validate_result(::AbstractTransformation, ::ManyBodyModel, ::ManyBodyModel) = true

# ---------------------------------------------------------------------------
# Transformation-specific exact expression atoms
# ---------------------------------------------------------------------------

"""
    SqrtNumberFactor(kind, mode, offset, coefficient)

Symbolic exact factor

    sqrt(offset*I + coefficient*n_kind(mode)).

It is kept as an atomic noncommutative expression until a backend chooses an
explicit matrix or further analytic realization.
"""
struct SqrtNumberFactor{T<:Number} <: AbstractPrimitiveOperator
    kind::Symbol
    mode::Int
    offset::T
    coefficient::T
end

"""
    BosonicDisplacementFactor(mode, alpha)

Atomic displacement operator

    D(alpha) = exp(alpha*b^dagger - conj(alpha)*b).
"""
struct BosonicDisplacementFactor{T<:Number} <: AbstractPrimitiveOperator
    mode::Int
    alpha::T
end

Base.adjoint(op::SqrtNumberFactor) = op
Base.adjoint(op::BosonicDisplacementFactor) =
    BosonicDisplacementFactor(op.mode, -op.alpha)

function Base.show(io::IO, op::SqrtNumberFactor)
    print(io, "sqrt(", op.offset, "*I + ", op.coefficient,
          "*n_", op.kind, "(", op.mode, "))")
end

function Base.show(io::IO, op::BosonicDisplacementFactor)
    print(io, "D(", op.mode, ", ", op.alpha, ")")
end

# Teach Algebra enough about these atoms to retain them during exact
# canonicalization.  They deliberately remain opaque to local CCR rewrites.
Algebra.operator_family(op::SqrtNumberFactor) = op.kind
Algebra.operator_family(::BosonicDisplacementFactor) = :boson

Algebra._expr_equal(A::SqrtNumberFactor, B::SqrtNumberFactor) =
    A.kind == B.kind && A.mode == B.mode &&
    A.offset == B.offset && A.coefficient == B.coefficient

Algebra._expr_equal(A::BosonicDisplacementFactor, B::BosonicDisplacementFactor) =
    A.mode == B.mode && A.alpha == B.alpha

# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

struct CompositeTransformation{T<:Tuple} <: AbstractTransformation
    transformations::T
end

CompositeTransformation(ts::AbstractTransformation...) =
    CompositeTransformation(tuple(ts...))

function certificate(T::CompositeTransformation)
    certs = map(certificate, T.transformations)
    isempty(certs) && return TransformationCertificate(
        true, true, CompositeRelation, :local; notes=("identity composition",),
    )
    return TransformationCertificate(
        all(c -> c.exact, certs),
        all(c -> c.invertible, certs),
        CompositeRelation,
        any(c -> c.locality == :nonlocal, certs) ? :nonlocal : :local;
        requires_constraints=any(c -> c.requires_constraints, certs),
        gauge_redundant=any(c -> c.gauge_redundant, certs),
        approximate=any(c -> c.approximate, certs),
        notes=("composed transformation",),
    )
end

transformation_name(::CompositeTransformation) = :composite

# ---------------------------------------------------------------------------
# Metadata transport helpers for exact changes of representation
# ---------------------------------------------------------------------------

function _source_physical_constraints(source::Representation)
    constraints = Any[source.constraints...]
    if !isnothing(source.physical_subspace)
        for constraint in source.physical_subspace.constraints
            any(existing -> isequal(existing, constraint), constraints) || push!(constraints, constraint)
        end
    end
    return Tuple(constraints)
end

function _transported_constraints(T::AbstractTransformation, source::Representation)
    constraints = _source_physical_constraints(source)
    transported = Any[(
        kind=:transported_constraint,
        via=transformation_name(T),
        source_constraint=constraint,
        target_basis_projection=:required,
    ) for constraint in constraints]

    if !isnothing(source.physical_subspace) && !isnothing(source.physical_subspace.projector)
        push!(transported, (
            kind=:transported_projector,
            via=transformation_name(T),
            source_projector=source.physical_subspace.projector,
            target_basis_projection=:required,
        ))
    elseif !isnothing(source.physical_subspace) && isempty(constraints)
        push!(transported, (
            kind=:transported_physical_subspace,
            via=transformation_name(T),
            source_dimension=physicaldimension(source.physical_subspace),
            target_basis_projection=:required,
        ))
    end
    return Tuple(transported)
end

function _transported_physical_subspace(
    T::AbstractTransformation,
    source::Representation,
    target_ambient::AbstractStateSpace,
)
    source_constraints = _source_physical_constraints(source)
    if isnothing(source.physical_subspace) && isempty(source_constraints)
        return nothing, ()
    end

    constraints = _transported_constraints(T, source)
    dim = representationdimension(source)
    physical = PhysicalSubspace(target_ambient; constraints=constraints, dimension=dim)
    return physical, constraints
end

function _transported_gauge(
    T::AbstractTransformation,
    source::Representation,
    target_algebra::AbstractOperatorAlgebra,
)
    gauge = source.gauge_structure
    isnothing(gauge) && return nothing
    mapped = map(gauge.generators) do generator
        generator isa AbstractOperatorExpr || return generator
        canonicalize(
            target_algebra,
            transform_operator(T, generator);
            hbar=transformation_hbar(T),
        )
    end
    return GaugeStructure(
        gauge.group,
        Tuple(mapped);
        description="transported through $(transformation_name(T)): $(gauge.description)",
    )
end

function _transported_truncation(T::AbstractTransformation, source::Representation)
    truncation = source.truncation
    isnothing(truncation) && return nothing

    function transport_one(value::NumericalTruncation)
        metadata = (
            kind=:transported_subspace,
            via=transformation_name(T),
            source=value,
            source_representation=source.name,
            target_cutoff_required=true,
        )
        return NumericalTruncation(
            metadata;
            dimension=value.dimension,
            description="source-basis truncation transported as provenance through $(transformation_name(T)); select a target-basis cutoff before matrix realization",
        )
    end

    truncation isa NumericalTruncation && return transport_one(truncation)
    truncation isa FactorTruncation && return FactorTruncation(truncation.factor, transport_one(truncation.truncation))
    if truncation isa Tuple || truncation isa AbstractVector
        return tuple((FactorTruncation(entry.factor, transport_one(entry.truncation)) for entry in truncation)...)
    end
    throw(ArgumentError("unsupported numerical truncation metadata"))
end

# Preserve tensor-factor addressing through generator maps. If one primitive
# expands into a composite expression, every transformed primitive remains on
# the same addressed tensor factor while the global identity stays global.
function _qualify_transformed_expression(factor::Int, expr::AbstractOperatorExpr)
    expr isa IdentityOperator && return expr
    expr isa FactorOperator && return expr
    expr isa AbstractPrimitiveOperator && return FactorOperator(factor, expr)
    expr isa ScaledOperator && return ScaledOperator(expr.coefficient, _qualify_transformed_expression(factor, expr.operator))
    expr isa OperatorSum && return OperatorSum(AbstractOperatorExpr[_qualify_transformed_expression(factor, term) for term in expr.terms])
    expr isa OperatorProduct && return OperatorProduct(AbstractOperatorExpr[_qualify_transformed_expression(factor, term) for term in expr.factors])
    return expr
end

function transform_operator(T::AbstractTransformation, op::FactorOperator)
    mapped = transform_generator(T, op.operator)
    return _qualify_transformed_expression(op.factor, mapped)
end
