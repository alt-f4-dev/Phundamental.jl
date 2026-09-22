# Exact longitudinal spin-dependent phonon displacement transformation.

struct SpinPolaron{T<:Real} <: AbstractTransformation
    eta::Vector{T}
    hbar::T
end

function SpinPolaron(eta::AbstractVector{T}; hbar::Real=1) where {T<:Real}
    TT = promote_type(T, typeof(float(hbar)))
    return SpinPolaron{TT}(TT.(eta), TT(hbar))
end
SpinPolaron(eta::Real, nmodes::Integer; hbar::Real=1) =
    SpinPolaron(fill(eta, Int(nmodes)); hbar=hbar)

transformation_name(::SpinPolaron) = :spin_polaron
transformation_hbar(T::SpinPolaron) = T.hbar
certificate(::SpinPolaron) = TransformationCertificate(
    true, true, UnitaryEquivalence, :local;
    notes=("longitudinal spin-conditioned phonon displacement",),
)

function _spin_boson_factors(A::CompositeAlgebra)
    s = findfirst(x -> x isa SpinAlgebra, A.factors)
    bidx = findfirst(x -> x isa BosonAlgebra, A.factors)
    return s, bidx
end

function applicable(T::SpinPolaron, model::ManyBodyModel)
    A = model.representation.algebra
    A isa CompositeAlgebra || return false
    si, bi = _spin_boson_factors(A)
    (si === nothing || bi === nothing) && return false
    As, Ab = A.factors[si], A.factors[bi]
    return As.nsites == Ab.nmodes == length(T.eta)
end

function target_representation(T::SpinPolaron, model::ManyBodyModel)
    A = model.representation.algebra
    si, _ = _spin_boson_factors(A)
    spinA = A.factors[si]
    # Preserve spin quantum numbers when recoverable from the source composite space.
    spinspace = nothing
    if model.representation.state_space isa CompositeStateSpace
        spinspace = findfirst(x -> x isa SpinHilbertSpace, model.representation.state_space.factors)
    end
    Sspace = spinspace === nothing ? SpinHilbertSpace(fill(1//2, spinA.nsites)) :
        model.representation.state_space.factors[spinspace]
    space = CompositeStateSpace(Sspace, BosonFockSpace(length(T.eta)))
    algebra = CompositeAlgebra(SpinAlgebra(spinA.nsites), BosonAlgebra(length(T.eta)))
    basis = CompositeBasis(
        SpinProductBasis(collect(1:spinA.nsites)),
        BosonOccupationBasis(collect(1:length(T.eta))),
    )
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :spin_polaron,
        space,
        algebra,
        basis;
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=model.representation.ordering,
        reference_state=:spin_conditioned_displaced_phonons,
        truncation=_transported_truncation(T, model.representation),
    )
end

transform_generator(::SpinPolaron, op::SpinZ) = op
transform_generator(T::SpinPolaron, op::SpinPlus) =
    Sp(op.site) * BosonicDisplacementFactor(op.site, T.eta[op.site])
transform_generator(T::SpinPolaron, op::SpinMinus) =
    Sm(op.site) * BosonicDisplacementFactor(op.site, -T.eta[op.site])
transform_generator(T::SpinPolaron, op::SpinX) =
    (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))
transform_generator(T::SpinPolaron, op::SpinY) =
    (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))
transform_generator(T::SpinPolaron, op::BosonAnnihilate) =
    b(op.mode) - (T.eta[op.mode]/T.hbar) * Sz(op.mode)
transform_generator(T::SpinPolaron, op::BosonCreate) =
    b(op.mode)' - (T.eta[op.mode]/T.hbar) * Sz(op.mode)
transform_generator(T::SpinPolaron, op::NumberOperator) =
    op.kind == :boson ? transform_operator(T, b(op.mode)' * b(op.mode)) :
    throw(ArgumentError("spin-polaron transformation only maps bosonic number operators"))
