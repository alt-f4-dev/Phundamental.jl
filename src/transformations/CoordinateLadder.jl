# Canonical coordinate-momentum -> bosonic ladder-operator representation.

struct CoordinateLadder{T<:Real} <: AbstractTransformation
    masses::Vector{T}
    frequencies::Vector{T}
    hbar::T
end

function CoordinateLadder(masses::AbstractVector{T}, frequencies::AbstractVector{T}; hbar::Real=1) where {T<:Real}
    length(masses) == length(frequencies) ||
        throw(ArgumentError("masses and frequencies must have the same length"))
    all(>(0), masses) || throw(ArgumentError("all masses must be positive"))
    all(>(0), frequencies) || throw(ArgumentError("all frequencies must be positive"))
    TT = promote_type(T, typeof(float(hbar)))
    return CoordinateLadder{TT}(TT.(masses), TT.(frequencies), TT(hbar))
end

transformation_name(::CoordinateLadder) = :coordinate_ladder
transformation_hbar(T::CoordinateLadder) = T.hbar
certificate(::CoordinateLadder) = TransformationCertificate(
    true, true, CanonicalMap, :local;
    notes=("coordinate-momentum to bosonic ladder-operator representation",),
)

function applicable(T::CoordinateLadder, model::ManyBodyModel)
    A = model.representation.algebra
    return A isa CoordinateMomentumAlgebra && A.ndof == length(T.masses)
end

function target_representation(T::CoordinateLadder, model::ManyBodyModel)
    M = length(T.masses)
    space = BosonFockSpace(M)
    algebra = BosonAlgebra(M)
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :bosonic_ladder,
        space,
        algebra,
        BosonOccupationBasis(collect(1:M));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=collect(1:M),
        reference_state=:harmonic_vacuum,
        truncation=_transported_truncation(T, model.representation),
    )
end

function transform_generator(T::CoordinateLadder, op::PositionOperator)
    i = op.mode
    a = sqrt(T.hbar / (2 * T.masses[i] * T.frequencies[i]))
    return a * (b(i) + b(i)')
end

function transform_generator(T::CoordinateLadder, op::MomentumOperator)
    i = op.mode
    a = -im * sqrt(T.hbar * T.masses[i] * T.frequencies[i] / 2)
    return a * (b(i) - b(i)')
end
