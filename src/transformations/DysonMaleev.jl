# Exact nonunitary Dyson-Maleev spin representation on the physical sector.

struct DysonMaleev <: AbstractTransformation
    hbar::Float64
end
DysonMaleev(; hbar::Real=1.0) = DysonMaleev(Float64(hbar))

transformation_name(::DysonMaleev) = :dyson_maleev
transformation_hbar(T::DysonMaleev) = T.hbar
certificate(::DysonMaleev) = TransformationCertificate(
    true, true, SimilarityEquivalence, :local;
    requires_constraints=true,
    notes=("nonunitary", "biorthogonal physical representation"),
)

applicable(::DysonMaleev, model::ManyBodyModel) =
    model.representation.algebra isa SpinAlgebra &&
    model.representation.state_space isa SpinHilbertSpace

struct _BoundDM{S} <: AbstractTransformation
    parent::DysonMaleev
    spins::S
end
certificate(T::_BoundDM) = certificate(T.parent)
transformation_hbar(T::_BoundDM) = T.parent.hbar
transformation_name(::_BoundDM) = :dyson_maleev

transform_generator(T::_BoundDM, op::SpinZ) =
    T.parent.hbar * (T.spins[op.site] * IdentityOperator() - nb(op.site))
transform_generator(T::_BoundDM, op::SpinPlus) =
    (T.parent.hbar * sqrt(2 * T.spins[op.site])) * b(op.site)
transform_generator(T::_BoundDM, op::SpinMinus) = begin
    S = T.spins[op.site]
    (T.parent.hbar * sqrt(2S)) * b(op.site)' *
        (IdentityOperator() - (1/(2S)) * nb(op.site))
end
transform_generator(T::_BoundDM, op::SpinX) =
    (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))
transform_generator(T::_BoundDM, op::SpinY) =
    (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))

function target_representation(::DysonMaleev, model::ManyBodyModel)
    spins = model.representation.state_space.spins
    N = length(spins)
    bosons = BosonFockSpace(N)
    ambient = BiorthogonalSpace(bosons)
    constraints = Tuple((kind=:occupation_bounds, mode=i, minimum=0,
                         maximum=Int(round(2 * spins[i]))) for i in 1:N)
    physical = PhysicalSubspace(
        ambient;
        constraints=constraints,
        dimension=prod(Int(round(2S + 1)) for S in spins),
    )
    basis0 = BosonOccupationBasis(collect(1:N))
    return Representation(
        :dyson_maleev,
        ambient,
        BosonAlgebra(N),
        BiorthogonalBasis(basis0, basis0);
        physical_subspace=physical,
        constraints=constraints,
        reference_state=:maximally_polarized_spin_vacuum,
        ordering=collect(1:N),
        truncation=model.representation.truncation,
    )
end

function transform(T::DysonMaleev, model::ManyBodyModel)
    applicable(T, model) || throw(ArgumentError("Dyson-Maleev requires a spin representation"))
    target = target_representation(T, model)
    bound = _BoundDM(T, model.representation.state_space.spins)
    Hnew = canonicalize_transformed(bound, target, transform_operator(bound, model.hamiltonian))
    Onew = _transform_observables(bound, target, model.observables)
    cert = certificate(T)
    history = Any[model.provenance...]
    push!(history, (transformation=:dyson_maleev, certificate=cert,
                    source_representation=model.representation.name,
                    target_representation=target.name))
    target_space = _transformed_model_space(T, target, model)
    target_specification = isnothing(target_space) ? _transformed_specification(target, model.specification) : target_space.specification
    out = ManyBodyModel(target, Hnew; parameters=model.parameters, observables=Onew, provenance=history,
                        specification=target_specification, model_space=target_space, parameterization=rebind_parameters(model.parameterization, target_space))
    return TransformationResult(out, T, cert, Dict(:domain=>true, :parameters=>true, :result=>true))
end
