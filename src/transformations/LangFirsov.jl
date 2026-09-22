# Exact local Lang-Firsov electron-phonon transformation.
#
# `mode_to_phonon[m] = i` associates fermion mode `m` with local phonon mode
# `i`.  This supports both spinless models (one fermion mode per site) and
# spinful models (for example, two fermion modes coupled to one site phonon).

struct LangFirsov{T<:Real} <: AbstractTransformation
    lambda::Vector{T}
    mode_to_phonon::Vector{Int}
end

function LangFirsov(
    lambda::AbstractVector{T};
    mode_to_phonon=nothing,
) where {T<:Real}
    λ = collect(lambda)
    mapping = isnothing(mode_to_phonon) ? collect(1:length(λ)) : Int.(mode_to_phonon)
    isempty(mapping) && !isempty(λ) && throw(ArgumentError("at least one fermion mode is required"))
    all(i -> 1 <= i <= length(λ), mapping) ||
        throw(ArgumentError("mode_to_phonon entries must index lambda"))
    return LangFirsov{T}(λ, mapping)
end

function LangFirsov(
    lambda::Real,
    nphonons::Integer;
    fermion_modes_per_phonon::Integer=1,
)
    nphonons >= 0 || throw(ArgumentError("nphonons must be nonnegative"))
    fermion_modes_per_phonon > 0 ||
        throw(ArgumentError("fermion_modes_per_phonon must be positive"))
    λ = fill(lambda, Int(nphonons))
    mapping = Int[]
    for i in 1:Int(nphonons)
        append!(mapping, fill(i, Int(fermion_modes_per_phonon)))
    end
    return LangFirsov(λ; mode_to_phonon=mapping)
end

transformation_name(::LangFirsov) = :lang_firsov
certificate(::LangFirsov) = TransformationCertificate(
    true, true, UnitaryEquivalence, :local;
    notes=("occupation-conditioned phonon displacement", "electron-phonon polaron representation"),
)

function _fermion_boson_factors(A::CompositeAlgebra)
    f = findfirst(x -> x isa FermionAlgebra, A.factors)
    bidx = findfirst(x -> x isa BosonAlgebra, A.factors)
    return f, bidx
end

function applicable(T::LangFirsov, model::ManyBodyModel)
    A = model.representation.algebra
    A isa CompositeAlgebra || return false
    fi, bi = _fermion_boson_factors(A)
    (fi === nothing || bi === nothing) && return false
    Af, Ab = A.factors[fi], A.factors[bi]
    return Af.nmodes == length(T.mode_to_phonon) && Ab.nmodes == length(T.lambda)
end

function target_representation(T::LangFirsov, model::ManyBodyModel)
    nf_modes = length(T.mode_to_phonon)
    nb_modes = length(T.lambda)
    space = CompositeStateSpace(FermionFockSpace(nf_modes), BosonFockSpace(nb_modes))
    algebra = CompositeAlgebra(FermionAlgebra(nf_modes), BosonAlgebra(nb_modes))
    basis = CompositeBasis(
        FermionOccupationBasis(collect(1:nf_modes)),
        BosonOccupationBasis(collect(1:nb_modes)),
    )
    physical, constraints = _transported_physical_subspace(T, model.representation, space)
    gauge = _transported_gauge(T, model.representation, algebra)
    return Representation(
        :lang_firsov_polaron,
        space,
        algebra,
        basis;
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=model.representation.ordering,
        reference_state=:occupation_conditioned_displaced_phonons,
        truncation=_transported_truncation(T, model.representation),
    )
end

function _lf_density(T::LangFirsov, phonon_mode::Int)
    modes = findall(==(phonon_mode), T.mode_to_phonon)
    isempty(modes) && return 0 * IdentityOperator()
    terms = AbstractOperatorExpr[nf(m) for m in modes]
    return length(terms) == 1 ? terms[1] : OperatorSum(terms)
end

function transform_generator(T::LangFirsov, op::FermionAnnihilate)
    p = T.mode_to_phonon[op.mode]
    return c(op.mode) * BosonicDisplacementFactor(p, -T.lambda[p])
end

function transform_generator(T::LangFirsov, op::FermionCreate)
    p = T.mode_to_phonon[op.mode]
    return c(op.mode)' * BosonicDisplacementFactor(p, T.lambda[p])
end

function transform_generator(T::LangFirsov, op::BosonAnnihilate)
    p = op.mode
    return b(p) - T.lambda[p] * _lf_density(T, p)
end

function transform_generator(T::LangFirsov, op::BosonCreate)
    p = op.mode
    return b(p)' - T.lambda[p] * _lf_density(T, p)
end

function transform_generator(T::LangFirsov, op::NumberOperator)
    if op.kind == :fermion
        # Fermionic occupation commutes with the Lang-Firsov generator.
        return nf(op.mode)
    elseif op.kind == :boson
        return transform_operator(T, b(op.mode)' * b(op.mode))
    end
    throw(ArgumentError("unsupported number-operator kind $(op.kind)"))
end
