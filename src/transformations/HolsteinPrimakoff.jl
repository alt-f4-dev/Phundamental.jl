# Exact constrained Holstein-Primakoff spin representation.

struct HolsteinPrimakoff <: AbstractTransformation
    hbar::Float64
end
HolsteinPrimakoff(; hbar::Real=1.0) = HolsteinPrimakoff(Float64(hbar))

transformation_name(::HolsteinPrimakoff) = :holstein_primakoff
transformation_hbar(T::HolsteinPrimakoff) = T.hbar
certificate(::HolsteinPrimakoff) = TransformationCertificate(
    true, true, ConstrainedEmbedding, :local;
    requires_constraints=true,
    notes=("exact when square-root factors and physical occupation bounds are retained",),
)

applicable(::HolsteinPrimakoff, model::ManyBodyModel) =
    model.representation.algebra isa SpinAlgebra &&
    model.representation.state_space isa SpinHilbertSpace

function _spin_values(model::ManyBodyModel)
    return model.representation.state_space.spins
end

function target_representation(::HolsteinPrimakoff, model::ManyBodyModel)
    spins = _spin_values(model)
    N = length(spins)
    ambient = BosonFockSpace(N)
    constraints = Tuple((
        kind=:occupation_bounds,
        mode=i,
        minimum=0,
        maximum=Int(round(2 * spins[i])),
    ) for i in 1:N)
    physical = PhysicalSubspace(
        ambient;
        constraints=constraints,
        dimension=prod(Int(round(2S + 1)) for S in spins),
    )
    return Representation(
        :holstein_primakoff,
        ambient,
        BosonAlgebra(N),
        BosonOccupationBasis(collect(1:N));
        physical_subspace=physical,
        constraints=constraints,
        reference_state=:maximally_polarized_spin_vacuum,
        ordering=collect(1:N),
        truncation=model.representation.truncation,
    )
end

_hp_root(S::Number, i::Int) = SqrtNumberFactor(:boson, i, 2S, -one(S))

function _hp_spin(T::HolsteinPrimakoff, model::ManyBodyModel, site::Int)
    return _spin_values(model)[site]
end

# These rules require source spin values, so transform(model) is specialized to
# bind the values before applying the generic recursive mapper.
struct _BoundHP{S} <: AbstractTransformation
    parent::HolsteinPrimakoff
    spins::S
end
certificate(T::_BoundHP) = certificate(T.parent)
transformation_hbar(T::_BoundHP) = T.parent.hbar
transformation_name(::_BoundHP) = :holstein_primakoff

transform_generator(T::_BoundHP, op::SpinZ) =
    T.parent.hbar * (T.spins[op.site] * IdentityOperator() - nb(op.site))
transform_generator(T::_BoundHP, op::SpinPlus) =
    T.parent.hbar * _hp_root(T.spins[op.site], op.site) * b(op.site)
transform_generator(T::_BoundHP, op::SpinMinus) =
    T.parent.hbar * b(op.site)' * _hp_root(T.spins[op.site], op.site)
transform_generator(T::_BoundHP, op::SpinX) =
    (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))
transform_generator(T::_BoundHP, op::SpinY) =
    (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))

function transform(T::HolsteinPrimakoff, model::ManyBodyModel)
    applicable(T, model) || throw(ArgumentError("Holstein-Primakoff requires a spin representation"))
    target = target_representation(T, model)
    bound = _BoundHP(T, _spin_values(model))
    Hnew = canonicalize_transformed(bound, target, transform_operator(bound, model.hamiltonian))
    Onew = _transform_observables(bound, target, model.observables)
    cert = certificate(T)
    history = Any[model.provenance...]
    push!(history, (transformation=:holstein_primakoff, certificate=cert,
                    source_representation=model.representation.name,
                    target_representation=target.name))
    target_space = _transformed_model_space(T, target, model)
    target_specification = isnothing(target_space) ? _transformed_specification(target, model.specification) : target_space.specification
    out = ManyBodyModel(target, Hnew; parameters=model.parameters, observables=Onew, provenance=history,
                        specification=target_specification, model_space=target_space, parameterization=rebind_parameters(model.parameterization, target_space))
    return TransformationResult(out, T, cert, Dict(:domain=>true, :parameters=>true, :result=>true))
end
