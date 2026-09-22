# Fermionic particle-hole transformation.

struct ParticleHoleTransformation <: AbstractTransformation end

transformation_name(::ParticleHoleTransformation) = :particle_hole
certificate(::ParticleHoleTransformation) = TransformationCertificate(
    true, true, CanonicalMap, :local;
    notes=("fermionic particle-hole canonical transformation",),
)
has_inverse(::ParticleHoleTransformation) = true
inverse(T::ParticleHoleTransformation) = T

applicable(::ParticleHoleTransformation, model::ManyBodyModel) =
    model.representation.algebra isa FermionAlgebra

function target_representation(T::ParticleHoleTransformation, model::ManyBodyModel)
    M = model.representation.algebra.nmodes
    space = FermionFockSpace(M)
    algebra = FermionAlgebra(M)
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :hole,
        space,
        algebra,
        FermionOccupationBasis(collect(1:M));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        reference_state=:filled_reference,
        ordering=model.representation.ordering,
        truncation=_transported_truncation(T, model.representation),
    )
end

transform_generator(::ParticleHoleTransformation, op::FermionAnnihilate) = c(op.mode)'
transform_generator(::ParticleHoleTransformation, op::FermionCreate) = c(op.mode)
transform_generator(::ParticleHoleTransformation, op::NumberOperator) =
    op.kind == :fermion ? IdentityOperator() - nf(op.mode) :
    throw(ArgumentError("particle-hole map only acts on fermionic number operators"))
