"""
    ManyBodyModel

Complete Hamiltonian model carried through the framework. The legacy fields
`representation`, `hamiltonian`, `parameters`, `observables`, and `provenance`
remain unchanged. `specification`, `model_space`, and `parameterization`
optionally connect a selected Hamiltonian to the generalized construction
`G -> H(G) -> Theta(G) -> H(theta)` without overloading the legacy parameter/metadata map. An unbound typed parameterization is rebound to an attached `HamiltonianSpace`; an already bound parameterization must agree with the model's attached space.
"""
struct ManyBodyModel{R<:Representation,H,P,O,V,G,S,Q}
    representation::R
    hamiltonian::H
    parameters::P
    observables::O
    provenance::V
    specification::G
    model_space::S
    parameterization::Q
end

function ManyBodyModel(representation::Representation, hamiltonian; parameters=Dict{Symbol,Any}(), observables=Dict{Symbol,Any}(),
                       provenance=Any[], specification=nothing, model_space=nothing, parameterization=nothing)
    if !isnothing(model_space)
        model_space isa HamiltonianSpace || throw(ArgumentError("model_space must be HamiltonianSpace or nothing"))
        if isnothing(specification)
            specification = model_space.specification
        elseif specification !== model_space.specification && specification != model_space.specification
            throw(ArgumentError("specification must agree with model_space.specification"))
        end
    end
    !isnothing(specification) && !(specification isa GenerativeSpecification) &&
        throw(ArgumentError("specification must be GenerativeSpecification or nothing"))
    !isnothing(parameterization) && !(parameterization isa ParameterPoint || parameterization isa ParameterSpace) &&
        throw(ArgumentError("parameterization must be ParameterPoint, ParameterSpace, or nothing"))
    if !isnothing(parameterization) && !isnothing(model_space)
        schema = parameterization isa ParameterPoint ? parameterization.space : parameterization
        if isnothing(schema.hamiltonian_space)
            parameterization = rebind_parameters(parameterization, model_space)
        elseif schema.hamiltonian_space !== model_space
            throw(ArgumentError("parameterization is bound to a different HamiltonianSpace than model_space"))
        end
    end
    return ManyBodyModel(representation, hamiltonian, parameters, observables, collect(provenance), specification, model_space, parameterization)
end

# Backward-compatible positional constructor matching the pre-extension layout.
ManyBodyModel(representation::Representation, hamiltonian, parameters, observables, provenance) =
    ManyBodyModel(representation, hamiltonian; parameters=parameters, observables=observables, provenance=provenance)

_same_model_algebra(a::SpinAlgebra, b::SpinAlgebra) = a.nsites == b.nsites
_same_model_algebra(a::BosonAlgebra, b::BosonAlgebra) = a.nmodes == b.nmodes
_same_model_algebra(a::FermionAlgebra, b::FermionAlgebra) = a.nmodes == b.nmodes
_same_model_algebra(a::MajoranaAlgebra, b::MajoranaAlgebra) = a.ngenerators == b.ngenerators
_same_model_algebra(a::CoordinateMomentumAlgebra, b::CoordinateMomentumAlgebra) = a.ndof == b.ndof
_same_model_algebra(a::CompositeAlgebra, b::CompositeAlgebra) = length(a.factors) == length(b.factors) && all(_same_model_algebra(x, y) for (x, y) in zip(a.factors, b.factors))
_same_model_algebra(a::AbstractOperatorAlgebra, b::AbstractOperatorAlgebra) = typeof(a) === typeof(b) && isequal(a, b)

"""
    admissibility_report(model)

Evaluate the Hamiltonian carried by a `ManyBodyModel` against its attached `HamiltonianSpace` or `GenerativeSpecification`. The check is purely physical/model-space validation and never consults solver capability or execution policy. A model without attached model-space information cannot make a package-level admissibility claim and therefore raises `ArgumentError`.
"""
function admissibility_report(model::ManyBodyModel)
    specification = if !isnothing(model.model_space)
        model.model_space.specification
    elseif !isnothing(model.specification)
        model.specification
    else
        throw(ArgumentError("model carries no HamiltonianSpace or GenerativeSpecification for physical admissibility evaluation"))
    end
    operator_report = if isnothing(model.model_space)
        admissibility_report(specification, model.hamiltonian)
    else
        admissibility_report(model.model_space, model.hamiltonian)
    end
    algebra_matches = _same_model_algebra(model.representation.algebra, specification.algebra)
    algebra_reason = algebra_matches ? "" : "model representation algebra does not match the algebra declared by its physical specification"
    algebra_metadata = Dict{Symbol,Any}(
        :representation_algebra => model.representation.algebra,
        :specification_algebra => specification.algebra,
    )
    checks = AdmissibilityCheck[
        AdmissibilityCheck(:representation_algebra, algebra_matches; reason=algebra_reason, metadata=algebra_metadata),
    ]
    append!(checks, operator_report.checks)
    metadata = copy(operator_report.metadata)
    metadata[:scope] = :many_body_model
    metadata[:has_model_space] = !isnothing(model.model_space)
    return AdmissibilityReport(checks; metadata=metadata)
end

admissible(model::ManyBodyModel) = admissibility_report(model).admissible

"""Return a copy of a model with an additional provenance record."""
function with_provenance(model::ManyBodyModel, record)
    history = Any[model.provenance...]
    push!(history, record)
    return ManyBodyModel(model.representation, model.hamiltonian; parameters=model.parameters, observables=model.observables,
                         provenance=history, specification=model.specification, model_space=model.model_space,
                         parameterization=model.parameterization)
end

"""Return a copy of a model with a replacement representation and Hamiltonian."""
function remap_model(model::ManyBodyModel, representation::Representation, hamiltonian; observables=model.observables,
                     parameters=model.parameters, provenance=model.provenance, specification=model.specification,
                     model_space=model.model_space, parameterization=model.parameterization)
    rebound_parameterization = parameterization
    if parameterization === model.parameterization && model_space !== model.model_space
        rebound_parameterization = rebind_parameters(parameterization, model_space)
    end
    return ManyBodyModel(representation, hamiltonian; parameters=parameters, observables=observables, provenance=provenance,
                         specification=specification, model_space=model_space, parameterization=rebound_parameterization)
end

"""
    with_truncation(model, truncation; record=true)

Return a copy of `model` whose representation carries `truncation`. This is a numerical-basis choice only: the Hamiltonian, exact physical constraints, model-space metadata, parameters, and observables are unchanged. Set `truncation=nothing` to clear a numerical truncation.
"""
function with_truncation(model::ManyBodyModel, truncation; record::Bool=true)
    representation = with_truncation(model.representation, truncation)
    provenance = Any[model.provenance...]
    if record
        push!(provenance, (
            numerical_truncation=isnothing(truncation) ? :cleared : :set,
            representation=representation.name,
            truncation=truncation,
        ))
    end
    return ManyBodyModel(representation, model.hamiltonian; parameters=model.parameters, observables=model.observables,
                         provenance=provenance, specification=model.specification, model_space=model.model_space,
                         parameterization=model.parameterization)
end

"""
    instantiate_model(space, representation, parameters; observables=Dict(), metadata_parameters=Dict(), provenance=Any[])

Instantiate one selected Hamiltonian from a generated `HamiltonianSpace`.
`parameters` may be a `ParameterPoint` or a real coefficient vector. The
legacy `model.parameters` map remains available for solver metadata and is
populated with the selected named coefficients plus `metadata_parameters`.
"""
function instantiate_model(space::HamiltonianSpace, representation::Representation, parameters;
                           observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}(), provenance=Any[])
    point = if parameters isa ParameterPoint
        parameters
    elseif parameters isa AbstractVector{<:Real}
        ParameterPoint(ParameterSpace(space; values=parameters), parameters)
    else
        throw(ArgumentError("parameters must be ParameterPoint or a real coefficient vector"))
    end
    length(point.values) == length(space) || throw(DimensionMismatch("parameter dimension does not match HamiltonianSpace"))
    point = rebind_parameters(point, space)
    hamiltonian = hamiltonian_from_coefficients(space, point.values)
    legacy = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata_parameters))
    for (name, value) in zip(parameter_names(point), point.values)
        legacy[name] = value
    end
    model = ManyBodyModel(representation, hamiltonian; parameters=legacy, observables=observables, provenance=provenance,
                          specification=space.specification, model_space=space, parameterization=point)
    _require_complete_admissibility(admissibility_report(model), "instantiated model")
    return model
end

"""Extract the canonical coefficient-independent Hamiltonian structure of a `ManyBodyModel`."""
function structure(model::ManyBodyModel)
    model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("model Hamiltonian must subtype AbstractOperatorExpr for structural extraction"))
    metadata = Dict{Symbol,Any}(:scope => :many_body_model, :representation_name => model.representation.name)
    !isnothing(model.model_space) && (metadata[:model_space_dimension] = length(model.model_space))
    return _model_structure(representation_signature(model.representation), model.representation.algebra, (model.hamiltonian,); metadata=metadata)
end

canonicalize_structure(model::ManyBodyModel) = structure(model)
structural_signature(model::ManyBodyModel) = structural_signature(structure(model))
interaction_graph(model::ManyBodyModel) = interaction_graph(structure(model))
structurally_equivalent(left::ManyBodyModel, right::ManyBodyModel) = structurally_equivalent(structure(left), structure(right))
