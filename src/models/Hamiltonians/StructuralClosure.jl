# Batch 12: authoritative structural reconstruction and interoperability.
# The physical operator expressions remain inside Julia; neutral records are descriptors, not executable model specifications.

const STRUCTURAL_SCHEMA_VERSION = "1.0"

function _structural_digest(kind::AbstractString, value)
    frozen = PhundamentalCore._freeze_structural_value(value)
    payload = string(STRUCTURAL_SCHEMA_VERSION, ":", kind, ":", repr(frozen))
    return string(kind, "-v1-", bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(payload)))))
end

"""A deterministic, versioned identifier derived from the canonical physical/structural description, never from Julia's salted `hash`."""
structural_id(direction::InvariantDirectionSignature) = _structural_digest("direction", direction.coordinates)
structural_id(family::StructuralFamilySignature) = _structural_digest("family", (family.specification, family.directions, family.partition))
structural_id(family::StructuralFamily) = structural_id(family.signature)
function structural_id(family::StructuralFamily, unit::StructuralUnit)
    directions = Tuple(sort!([signature.coordinates for signature in unit.direction_signatures]; by=repr))
    return _structural_digest("family_unit", (structural_id(family), directions))
end
structural_id(selection::StructuralSelectionSignature) = _structural_digest("selection", (structural_id(selection.family), selection.directions))
structural_id(selection::StructuralSelection) = structural_id(selection.signature)
structural_id(space::HamiltonianSpace) = _structural_digest("space", (_family_specification_key(space), Tuple(invariant_direction_signature(space, i).coordinates for i in 1:length(space))))
structural_id(specification::GenerativeSpecification) = _structural_digest("specification", (
    specification.space_group, specification.wyckoff_orbits, specification.local_spaces, specification.algebra,
    _physical_callback_key(specification.symmetry_action), Tuple(_physical_constraint_key(c) for c in specification.admissibility_conditions),
    specification.max_range, specification.max_body_order, _physical_callback_key(specification.range_evaluator),
    get(specification.metadata, :crystal_structure, nothing), get(specification.metadata, :generation_domain, nothing),
    get(specification.metadata, :operator_site_mapping, nothing)))

function _neutral_value(value::Union{Nothing,Bool,AbstractString,Integer})
    return value
end
_neutral_value(value::Symbol) = String(value)
_neutral_value(value::AbstractFloat) = isfinite(value) ? value : string(value)
_neutral_value(value::Rational) = Dict("numerator" => numerator(value), "denominator" => denominator(value))
_neutral_value(value::Complex) = Dict("real" => _neutral_value(real(value)), "imag" => _neutral_value(imag(value)))
_neutral_value(value::Tuple) = [_neutral_value(x) for x in value]
_neutral_value(value::AbstractVector) = [_neutral_value(x) for x in value]
function _neutral_value(value::AbstractDict)
    return Dict(string(k) => _neutral_value(v) for (k, v) in pairs(value))
end
function _neutral_value(value::NamedTuple)
    return Dict(string(k) => _neutral_value(v) for (k, v) in pairs(value))
end
_neutral_value(value) = throw(ArgumentError("cannot expose $(typeof(value)) as a neutral value; use an explicit descriptor"))

_record(kind::AbstractString; kwargs...) = Dict{String,Any}("schema_version" => STRUCTURAL_SCHEMA_VERSION, "kind" => String(kind),
                                                             (String(k) => v for (k, v) in pairs(kwargs))...)

"""Return a neutral, schema-versioned description; opaque physical callbacks and operator trees are never deserialized as executable Python inputs."""
function describe(spec::GenerativeSpecification)
    return _record("generative_specification"; id=structural_id(spec), algebra=string(nameof(typeof(spec.algebra))),
                   max_range=_neutral_value(spec.max_range), max_body_order=spec.max_body_order,
                   generation_domain=string(get(spec.metadata, :generation_domain, :unspecified)),
                   site_mapping=string(get(spec.metadata, :operator_site_mapping, :unspecified)),
                   symmetry_action_certification="implementation-bound; identity does not certify executable callback semantics")
end

function describe(space::HamiltonianSpace)
    audit = admissibility_report(space)
    return _record("hamiltonian_space"; id=structural_id(space), specification_id=structural_id(space.specification),
                   dimension=length(space), basis_labels=string.(space.basis.labels),
                   direction_ids=[structural_id(invariant_direction_signature(space, i)) for i in 1:length(space)],
                   completeness_scope=string(get(space.basis.metadata, :completeness_scope, :unknown)),
                   parent_completeness_scope=string(get(space.basis.metadata, :parent_completeness_scope, :not_applicable)),
                   certification=Dict("admissibility_complete" => audit.complete, "admissible" => audit.admissible,
                                      "failed_checks" => [Dict("name" => String(check.name), "reason" => check.reason) for check in audit.checks if check.passed !== true],
                                      "basis_generator" => string(get(space.basis.metadata, :generator, :unknown)),
                                      "independence_certification" => "inherited from invariant-basis generation; not independently audited here",
                                      "hermiticity_certification" => "inherited from invariant-basis generation; not independently audited here"))
end

function describe(unit::StructuralUnit)
    return _record("structural_unit"; id=_structural_digest("unit", Tuple(sort!(collect(getfield.(unit.direction_signatures, :coordinates)); by=repr))),
                   label=String(unit.id), parent_basis_indices=collect(unit.basis_indices),
                   basis_labels=string.(unit.basis_labels), direction_ids=[structural_id(d) for d in unit.direction_signatures],
                   symmetry_complete=get(unit.metadata, :symmetry_complete, false))
end

function describe(family::StructuralFamily)
    generation = get(family.metadata, :generation_result, nothing)
    return _record("structural_family"; id=structural_id(family), parent_space=describe(family.parent),
                   units=[merge(describe(unit), Dict("id" => structural_id(family, unit), "family_id" => structural_id(family))) for unit in family.units],
                   parameter_space=isnothing(family.parameter_space) ? nothing : describe(family.parameter_space),
                   material=isnothing(generation) || isnothing(generation.material) ? nothing : String(generation.material.name),
                   spacegroup_number=isnothing(generation) ? nothing : generation.dataset.spacegroup_number,
                   parent_completeness_scope=string(get(family.metadata, :parent_completeness_scope, :unknown)))
end

function describe(selection::StructuralSelection)
    return _record("structural_selection"; id=structural_id(selection), family_id=structural_id(selection.family),
                   active_units=string.(selection.unit_ids), parent_basis_indices=collect(selection.basis_indices),
                   selected_space=describe(selection.hamiltonian_space),
                   model_structure_id=_structural_digest("structure", structural_signature(selection.model_structure)))
end

function _describe_constraint(constraint::FixedParameterConstraint)
    return Dict("kind" => "fixed", "parameter" => String(constraint.parameter), "value" => _neutral_value(constraint.value),
                "atol" => constraint.atol, "rtol" => constraint.rtol)
end
function _describe_constraint(constraint::EqualParameterConstraint)
    return Dict("kind" => "equal", "left" => String(constraint.left), "right" => String(constraint.right),
                "atol" => constraint.atol, "rtol" => constraint.rtol)
end
function _describe_constraint(constraint::LinearParameterConstraint)
    return Dict("kind" => "linear", "name" => String(constraint.name), "coefficients" => Dict(String(k) => v for (k, v) in constraint.coefficients),
                "rhs" => constraint.rhs, "atol" => constraint.atol, "rtol" => constraint.rtol)
end
_describe_constraint(c::AbstractParameterConstraint) = throw(ArgumentError("no neutral constraint schema for $(typeof(c))"))

function describe(schema::ParameterSpace)
    return _record("parameter_space"; id=_structural_digest("parameters", (
                       isnothing(schema.hamiltonian_space) ? nothing : structural_id(schema.hamiltonian_space),
                       Tuple((p.name, p.lower, p.upper, p.fixed, p.fixed ? p.value : nothing, PhundamentalCore._freeze_structural_value(p.unit)) for p in schema.specifications),
                       Tuple(PhundamentalCore._freeze_structural_value(c) for c in parameter_constraints(schema)))),
                   names=string.(parameter_names(schema)), default_values=[_neutral_value(p.value) for p in schema.specifications],
                   bounds=[[_neutral_value(p.lower), _neutral_value(p.upper)] for p in schema.specifications],
                   units=[isnothing(p.unit) ? nothing : string(p.unit) for p in schema.specifications],
                   fixed=[p.fixed || p.lower == p.upper for p in schema.specifications],
                   constraints=[_describe_constraint(c) for c in parameter_constraints(schema)],
                   space_id=isnothing(schema.hamiltonian_space) ? nothing : structural_id(schema.hamiltonian_space))
end

function describe(result::HamiltonianGenerationResult)
    return _record("generation_result"; id=structural_id(result.hamiltonian_space),
                   material=isnothing(result.material) ? nothing : String(result.material.name),
                   spacegroup_number=result.dataset.spacegroup_number, international_symbol=String(result.dataset.international_symbol),
                   specification=describe(result.specification), space=describe(result.hamiltonian_space), parameters=describe(result.parameter_space))
end

function _structural_provenance(model::ManyBodyModel)
    index = findlast(record -> record isa NamedTuple && haskey(record, :generator) &&
                               record.generator === :batch12_structural_selection, model.provenance)
    return isnothing(index) ? nothing : model.provenance[index]
end

function describe(model::ManyBodyModel)
    selected = _structural_provenance(model)
    selection_id = isnothing(selected) ? nothing : selected.selection_id
    reference_spec = isnothing(model.model_space) ? model.specification : model.model_space.specification
    same_algebra = isnothing(reference_spec) || PhundamentalCore._same_model_algebra(model.representation.algebra, reference_spec.algebra)
    physical_status = if isnothing(reference_spec) || !same_algebra
        nothing
    else
        admissibility_report(model).admissible
    end
    model_id = _structural_digest("model", (selection_id,
        isnothing(model.model_space) ? nothing : structural_id(model.model_space),
        PhundamentalCore.representation_signature(model.representation),
        PhundamentalCore._freeze_structural_value(model.parameters),
        PhundamentalCore._freeze_structural_value(model.observables),
        isnothing(model.parameterization) ? nothing : (model.parameterization isa ParameterPoint ? Tuple(model.parameterization.values) : :unbound)))
    return _record("many_body_model"; id=model_id, space_id=isnothing(model.model_space) ? nothing : structural_id(model.model_space),
                   family_id=isnothing(selected) ? nothing : selected.family_id, selection_id=selection_id,
                   selected_units=isnothing(selected) ? nothing : string.(selected.selected_units),
                   representation=String(model.representation.name),
                   algebra=string(nameof(typeof(model.representation.algebra))),
                   parameter_values=model.parameterization isa ParameterPoint ? _neutral_value(model.parameterization.values) : nothing,
                   admissible=physical_status,
                   admissibility_scope=same_algebra ? "source_model" : "representation_transformed; consult source_model")
end

function _record_field(record, name::String; required::Bool=false)
    record isa AbstractDict || record isa NamedTuple || throw(ArgumentError("reconstruction request must be a dictionary or NamedTuple"))
    if record isa NamedTuple
        key = Symbol(name)
        required && !haskey(record, key) && throw(ArgumentError("missing reconstruction field $name"))
        return get(record, key, nothing)
    end
    if haskey(record, name)
        return record[name]
    elseif haskey(record, Symbol(name))
        return record[Symbol(name)]
    end
    required && throw(ArgumentError("missing reconstruction field $name"))
    return nothing
end

"""Reconstruct a selection only from a locally held certified family, validating its submitted identifiers. Without `active_units`, bounded enumeration is required."""
function reconstruct_selection(family::StructuralFamily, record; max_structures::Integer=4095)
    version = _record_field(record, "schema_version"; required=true)
    version == STRUCTURAL_SCHEMA_VERSION || throw(ArgumentError("unsupported structural schema version $version"))
    family_id = _record_field(record, "family_id"; required=true)
    family_id == structural_id(family) || throw(ArgumentError("reconstruction family identifier mismatch"))
    selected_id = _record_field(record, "selection_id")
    isnothing(selected_id) && (selected_id = _record_field(record, "id"))
    isnothing(selected_id) && throw(ArgumentError("reconstruction requires selection_id or id"))
    active = _record_field(record, "active_units")
    if !isnothing(active)
        active isa Union{Tuple,AbstractVector} || throw(ArgumentError("active_units must be a collection of structural-unit labels"))
        all(x -> x isa Union{Symbol,AbstractString}, active) || throw(ArgumentError("structural-unit labels must be strings or symbols"))
        choice = structural_selection(family, Symbol.(collect(active)))
        structural_id(choice) == selected_id || throw(ArgumentError("selection id does not match active structural units"))
        return choice
    end
    max_structures >= 1 || throw(ArgumentError("max_structures must be positive"))
    for candidate in enumerate_structures(family; max_structures=max_structures)
        structural_id(candidate) == selected_id && return candidate
    end
    throw(ArgumentError("selection identifier is not part of the supplied finite family"))
end

"""
    validate_structural_representation(selection, representation)

Check the algebra and unambiguous local Hilbert-space requirements of a generated selection without selecting a numerical solver. Custom, coordinate, and Majorana internal actions remain explicitly owned by their supplied representation.
"""
function validate_structural_representation(selection::StructuralSelection, representation::Representation)
    specification = selection.hamiltonian_space.specification
    PhundamentalCore._same_model_algebra(representation.algebra, specification.algebra) ||
        throw(ArgumentError("representation algebra does not match the generated specification"))
    algebra = specification.algebra
    spaces = specification.local_spaces
    if algebra isa SpinAlgebra
        representation.state_space isa SpinHilbertSpace || throw(ArgumentError("spin structural selections require an explicit SpinHilbertSpace"))
        if all(space -> space isa SpinHilbertSpace, spaces)
            spins = [spin for space in spaces for spin in space.spins]
            length(spins) == algebra.nsites || throw(DimensionMismatch("spin local spaces do not cover every generated spin index"))
            representation.state_space.spins == spins || throw(ArgumentError("representation spin quantum numbers disagree with generated local spaces"))
        end
    elseif algebra isa FermionAlgebra
        representation.state_space isa FermionFockSpace || throw(ArgumentError("fermion structural selections require a FermionFockSpace"))
        representation.state_space.nmodes == algebra.nmodes || throw(DimensionMismatch("fermion representation mode count disagrees with generated algebra"))
        if all(space -> space isa FermionFockSpace, spaces)
            sum(space.nmodes for space in spaces) == algebra.nmodes ||
                throw(DimensionMismatch("generated local fermion spaces do not cover the algebra's modes"))
        end
    elseif algebra isa BosonAlgebra
        representation.state_space isa BosonFockSpace || throw(ArgumentError("boson structural selections require a BosonFockSpace"))
        representation.state_space.nmodes == algebra.nmodes || throw(DimensionMismatch("boson representation mode count disagrees with generated algebra"))
        if all(space -> space isa BosonFockSpace, spaces)
            sum(space.nmodes for space in spaces) == algebra.nmodes ||
                throw(DimensionMismatch("generated local boson spaces do not cover the algebra's modes"))
        end
    end
    return true
end

"""Instantiate an explicitly selected child Hamiltonian with its complete parent constraints and provenance retained."""
function instantiate_model(selection::StructuralSelection, representation::Representation, parameters=nothing;
                           observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}(), provenance=Any[])
    validate_structural_representation(selection, representation)
    values = if isnothing(parameters)
        nothing
    elseif parameters isa ParameterPoint
        parameters.space.hamiltonian_space === selection.hamiltonian_space || throw(ArgumentError("point is not bound to the selected HamiltonianSpace"))
        parameter_names(parameters) == (isnothing(selection.family.parameter_space) ? selection.hamiltonian_space.basis.labels :
            parameter_names(selection.family.parameter_space)[[selection.basis_indices...]]) || throw(ArgumentError("point parameter ordering disagrees with selected family"))
        parameters.values
    else
        parameters
    end
    schema = parameter_subspace(selection, values)
    point = ParameterPoint(schema)
    history = Any[provenance...]
    spec = selection.family.parent.specification
    dataset = spec.space_group
    source = get(selection.family.metadata, :generation_result, nothing)
    push!(history, (generator=:batch12_structural_selection, family_id=structural_id(selection.family),
                    selection_id=structural_id(selection), selected_units=selection.unit_ids,
                    parent_basis_indices=selection.basis_indices,
                    generation_result=source,
                    material_name=get(spec.metadata, :material_name, nothing),
                    generation_domain=get(spec.metadata, :generation_domain, :unspecified),
                    spacegroup_number=hasproperty(dataset, :spacegroup_number) ? dataset.spacegroup_number : nothing,
                    parent_completeness_scope=get(selection.family.metadata, :parent_completeness_scope, :unknown)))
    _require_context_label_separation(selection.hamiltonian_space)
    context = default_observable_context(selection.family, representation;
        observables=observables, metadata_parameters=metadata_parameters)
    return instantiate_model(selection.hamiltonian_space, representation, point;
                             observables=context.observables, metadata_parameters=context.metadata_parameters, provenance=history)
end

"""Validate a versioned selection record and coefficient payload against a locally held family before constructing a model."""
function reconstruct_model(family::StructuralFamily, representation::Representation, record;
                           observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}(), provenance=Any[])
    selection = reconstruct_selection(family, record)
    values = _record_field(record, "parameters"; required=true)
    return instantiate_model(selection, representation, values; observables=observables,
                             metadata_parameters=metadata_parameters, provenance=provenance)
end
