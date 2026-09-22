# First-class approximation objects. These are deliberately distinct from
# exact representation transformations because an approximation may change
# the physical model rather than only its representation.

abstract type AbstractApproximation end

"""Metadata defining the assumptions and validity domain of an approximation."""
struct ApproximationCertificate{A,V,E,N}
    name::Symbol
    assumptions::A
    validity_conditions::V
    order::Union{Nothing,Int}
    error_model::E
    notes::N
end

function ApproximationCertificate(name::Symbol; assumptions=(), validity_conditions=(), order=nothing,
                                  error_model=nothing, notes=())
    !isnothing(order) && Int(order) < 0 && throw(ArgumentError("approximation order must be nonnegative"))
    return ApproximationCertificate(name, Tuple(assumptions), Tuple(validity_conditions),
                                    isnothing(order) ? nothing : Int(order), error_model, Tuple(notes))
end

approximation_name(A::AbstractApproximation) = nameof(typeof(A))
approximation_certificate(A::AbstractApproximation) = throw(ArgumentError("no approximation certificate is defined for $(approximation_name(A))"))
approximation_applicable(::AbstractApproximation, ::ManyBodyModel) = true

"""Result of an approximation, including optional retained/discarded operator content."""
struct ApproximationResult{M,A,C,R,D,E,V}
    model::M
    approximation::A
    certificate::C
    retained::R
    discarded::D
    error::E
    validation::V
end

apply_approximation(A::AbstractApproximation, model::ManyBodyModel) =
    throw(ArgumentError("no apply_approximation method is defined for $(approximation_name(A))"))

"""
    ModelSpaceProjection(target_space, projector; error_metric=nothing, certificate=...)

Project a Hamiltonian into a chosen target model space. `projector(H, space)`
must return either `H_parallel` or `(H_parallel, H_perp)`. This directly
implements the manuscript decomposition `T(H) = H_parallel + H_perp` without
prescribing how the projection is computed.
"""
struct ModelSpaceProjection{S<:HamiltonianSpace,P,E,C} <: AbstractApproximation
    target_space::S
    projector::P
    error_metric::E
    certificate::C
end

function ModelSpaceProjection(target_space::HamiltonianSpace, projector; error_metric=nothing,
                              certificate=ApproximationCertificate(:model_space_projection;
                                  assumptions=(:target_space_is_valid,), validity_conditions=(:projection_defined,)))
    return ModelSpaceProjection(target_space, projector, error_metric, certificate)
end

approximation_name(::ModelSpaceProjection) = :model_space_projection
approximation_certificate(A::ModelSpaceProjection) = A.certificate

function apply_approximation(A::ModelSpaceProjection, model::ManyBodyModel)
    approximation_applicable(A, model) || throw(ArgumentError("model-space projection is not applicable to the supplied model"))
    projected = A.projector(model.hamiltonian, A.target_space)
    retained, discarded = if projected isa Tuple && length(projected) == 2
        projected
    else
        projected, nothing
    end
    error = isnothing(A.error_metric) ? nothing : A.error_metric(model.hamiltonian, retained, discarded)
    history = Any[model.provenance...]
    push!(history, (approximation=approximation_name(A), certificate=A.certificate, error=error))
    approximated = ManyBodyModel(model.representation, retained; parameters=model.parameters, observables=model.observables,
                                 provenance=history, specification=A.target_space.specification, model_space=A.target_space,
                                 parameterization=rebind_parameters(model.parameterization, A.target_space))
    validation = Dict{Symbol,Bool}(:applicable => true, :admissible => admissible(A.target_space, retained))
    return ApproximationResult(approximated, A, A.certificate, retained, discarded, error, validation)
end

"""
    transformation_closure_defect(mapped_basis, target_basis, independent_rank)

Compute the manuscript closure defect when a caller supplies a rank function
for operator sets. This keeps linear-independence logic backend-independent.
"""
function transformation_closure_defect(mapped_basis, target_basis, independent_rank)
    mapped = collect(mapped_basis)
    target = collect(target_basis)
    return independent_rank(vcat(mapped, target)) - independent_rank(target)
end
