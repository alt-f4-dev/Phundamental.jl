# Exact unitary bosonic displacement / coherent-state transformation.

struct BosonicDisplacement{T<:Number} <: AbstractTransformation
    amplitudes::Vector{T}
end
BosonicDisplacement(amplitudes::AbstractVector{T}) where {T<:Number} =
    BosonicDisplacement{T}(collect(amplitudes))

transformation_name(::BosonicDisplacement) = :bosonic_displacement
certificate(::BosonicDisplacement) = TransformationCertificate(
    true, true, UnitaryEquivalence, :local;
    notes=("active convention D(alpha) O D(alpha)^dagger",),
)
has_inverse(::BosonicDisplacement) = true
inverse(T::BosonicDisplacement) = BosonicDisplacement(-T.amplitudes)

function applicable(T::BosonicDisplacement, model::ManyBodyModel)
    A = model.representation.algebra
    return A isa BosonAlgebra && A.nmodes == length(T.amplitudes)
end

function target_representation(T::BosonicDisplacement, model::ManyBodyModel)
    M = length(T.amplitudes)
    space = BosonFockSpace(M)
    algebra = BosonAlgebra(M)
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :displaced_boson,
        space,
        algebra,
        BosonOccupationBasis(collect(1:M));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        reference_state=(kind=:coherent_vacuum, amplitudes=copy(T.amplitudes)),
        ordering=model.representation.ordering,
        truncation=_transported_truncation(T, model.representation),
    )
end

transform_generator(T::BosonicDisplacement, op::BosonAnnihilate) =
    b(op.mode) - T.amplitudes[op.mode] * IdentityOperator()
transform_generator(T::BosonicDisplacement, op::BosonCreate) =
    b(op.mode)' - conj(T.amplitudes[op.mode]) * IdentityOperator()
transform_generator(T::BosonicDisplacement, op::NumberOperator) =
    op.kind == :boson ? transform_operator(T, b(op.mode)' * b(op.mode)) :
    throw(ArgumentError("bosonic displacement only maps bosonic number operators"))
