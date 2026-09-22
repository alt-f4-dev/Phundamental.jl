# Generic constrained/gauge-redundant slave-particle transformation hook.
#
# Concrete constructions such as Abrikosov fermions are preferable when a
# standard map exists.  This type supports user-defined exact parton maps while
# retaining the same transformation pipeline and provenance machinery.

struct SlaveParticleTransformation{R,F,P} <: AbstractTransformation
    target::R
    generator_map::F
    domain_predicate::P
    label::Symbol
end

function SlaveParticleTransformation(
    target::Representation,
    generator_map;
    domain_predicate=(model -> true),
    label::Symbol=:slave_particle,
)
    isconstrained(target) || throw(ArgumentError(
        "a slave-particle target representation must define a physical constraint"
    ))
    isnothing(target.gauge_structure) && throw(ArgumentError(
        "a slave-particle target representation must define gauge_structure"
    ))
    return SlaveParticleTransformation(target, generator_map, domain_predicate, label)
end

transformation_name(T::SlaveParticleTransformation) = T.label
certificate(::SlaveParticleTransformation) = TransformationCertificate(
    true, true, GaugeRedundantEmbedding, :local;
    requires_constraints=true,
    gauge_redundant=true,
    notes=("generic user-defined constrained slave-particle/parton map",),
)

applicable(T::SlaveParticleTransformation, model::ManyBodyModel) =
    Bool(T.domain_predicate(model))

target_representation(T::SlaveParticleTransformation, ::ManyBodyModel) = T.target

function transform_generator(T::SlaveParticleTransformation, op::AbstractPrimitiveOperator)
    op isa IdentityOperator && return op
    mapped = T.generator_map(op)
    mapped isa AbstractOperatorExpr || throw(ArgumentError(
        "slave-particle generator_map must return an AbstractOperatorExpr"
    ))
    return mapped
end
