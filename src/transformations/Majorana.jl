# Complex fermion -> Majorana Clifford representation.

struct MajoranaTransformation <: AbstractTransformation end

transformation_name(::MajoranaTransformation) = :majorana
certificate(::MajoranaTransformation) = TransformationCertificate(
    true, true, AlgebraIsomorphism, :local;
    notes=("complex fermion to paired Hermitian Majorana generators",),
)

applicable(::MajoranaTransformation, model::ManyBodyModel) =
    model.representation.algebra isa FermionAlgebra

function target_representation(T::MajoranaTransformation, model::ManyBodyModel)
    M = model.representation.algebra.nmodes
    space = FermionFockSpace(M)
    algebra = MajoranaAlgebra(2M)
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :majorana,
        space,
        algebra,
        FermionOccupationBasis(collect(1:M));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=model.representation.ordering,
        reference_state=model.representation.reference_state,
        truncation=_transported_truncation(T, model.representation),
    )
end

transform_generator(::MajoranaTransformation, op::FermionAnnihilate) =
    (1/2) * (γ(2 * op.mode - 1) + im * γ(2 * op.mode))

transform_generator(::MajoranaTransformation, op::FermionCreate) =
    (1/2) * (γ(2 * op.mode - 1) - im * γ(2 * op.mode))

transform_generator(::MajoranaTransformation, op::NumberOperator) =
    op.kind == :fermion ?
        (1/2) * (IdentityOperator() + im * γ(2 * op.mode - 1) * γ(2 * op.mode)) :
        throw(ArgumentError("Majorana map only acts on fermionic number operators"))
