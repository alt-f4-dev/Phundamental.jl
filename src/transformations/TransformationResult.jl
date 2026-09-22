# Complete transformation result and public transformation pipeline.

struct TransformationResult{M,T,C,V}
    model::M
    transformation::T
    certificate::C
    validation::V
end

"""Validate source applicability and transformation-specific parameters."""
function validate_source(T::AbstractTransformation, model::ManyBodyModel)
    return Dict{Symbol,Bool}(
        :domain => applicable(T, model),
        :parameters => validate_parameters(T, model),
    )
end

function _transform_observables(
    T::AbstractTransformation,
    target::Representation,
    observables,
)
    result = Dict{Symbol,Any}()
    for (name, observable) in pairs(observables)
        if observable isa AbstractOperatorExpr
            result[Symbol(name)] = transform_observable(T, target, observable)
        else
            # Non-operator metadata is retained unchanged.
            result[Symbol(name)] = observable
        end
    end
    return result
end


function _transformed_specification(target::Representation, specification)
    isnothing(specification) && return nothing
    metadata = Dict{Symbol,Any}()
    try
        for (key, value) in pairs(specification.metadata)
            metadata[Symbol(key)] = value
        end
    catch
    end
    metadata[:representation_algebra_transport] = target.name
    return GenerativeSpecification(specification.space_group, specification.wyckoff_orbits, specification.local_spaces,
                                   target.algebra, specification.symmetry_action;
                                   admissibility_conditions=specification.admissibility_conditions,
                                   max_range=specification.max_range, max_body_order=specification.max_body_order,
                                   metadata=metadata)
end

function _transformed_model_space(T::AbstractTransformation, target::Representation, model::ManyBodyModel)
    isnothing(model.model_space) && return nothing
    source_space = model.model_space
    target_specification = _transformed_specification(target, source_space.specification)
    mapped = AbstractOperatorExpr[transform_hamiltonian(T, target, operator) for operator in source_space.basis.operators]
    metadata = Dict{Symbol,Any}(:transformed_from => model.representation.name, :transformation => transformation_name(T))
    basis = HamiltonianBasis(mapped; labels=source_space.basis.labels, metadata=source_space.basis.metadata)
    return HamiltonianSpace(target_specification, basis; metadata=metadata)
end

"""
    transform(T, model)

Apply a complete representation transformation to a `ManyBodyModel`:

1. validate the source representation,
2. construct the target representation,
3. map and canonicalize the Hamiltonian,
4. transform physical observables consistently,
5. retain parameters,
6. attach transformation provenance and certificate,
7. validate the resulting model.
"""
function transform(T::AbstractTransformation, model::ManyBodyModel)
    source_validation = validate_source(T, model)
    all(values(source_validation)) || throw(ArgumentError(
        "$(transformation_name(T)) is not valid for the supplied model: $(source_validation)"
    ))

    target = target_representation(T, model)

    Hnew = model.hamiltonian isa AbstractOperatorExpr ?
        transform_hamiltonian(T, target, model.hamiltonian) :
        throw(ArgumentError(
            "the initial Transformations layer expects symbolic AbstractOperatorExpr Hamiltonians"
        ))

    Onew = _transform_observables(T, target, model.observables)
    cert = certificate(T)

    provenance_record = (
        transformation=transformation_name(T),
        certificate=cert,
        source_representation=model.representation.name,
        target_representation=target.name,
        physical_constraint_transport=isconstrained(model.representation) ? :target_projection_required : :none,
        numerical_truncation_transport=isnothing(model.representation.truncation) ? :none : :target_cutoff_required,
    )

    history = Any[model.provenance...]
    push!(history, provenance_record)

    target_space = _transformed_model_space(T, target, model)
    target_specification = isnothing(target_space) ? _transformed_specification(target, model.specification) : target_space.specification
    transformed = ManyBodyModel(target, Hnew; parameters=model.parameters, observables=Onew,
                                provenance=history, specification=target_specification, model_space=target_space,
                                parameterization=rebind_parameters(model.parameterization, target_space))

    validation = copy(source_validation)
    validation[:result] = validate_result(T, model, transformed)

    all(values(validation)) || throw(ErrorException(
        "$(transformation_name(T)) produced a model that failed validation: $(validation)"
    ))

    return TransformationResult(transformed, T, cert, validation)
end

(T::AbstractTransformation)(model::ManyBodyModel) = transform(T, model)

"""Apply a sequence of transformations in order."""
function transform(T::CompositeTransformation, model::ManyBodyModel)
    current = model
    validations = Dict{Symbol,Bool}()

    for (i, step) in enumerate(T.transformations)
        result = transform(step, current)
        current = result.model
        validations[Symbol("step_", i)] = all(values(result.validation))
    end

    return TransformationResult(current, T, certificate(T), validations)
end

"""Convenience composition constructor: `compose(T1, T2, ...)`."""
compose(ts::AbstractTransformation...) = CompositeTransformation(ts...)
