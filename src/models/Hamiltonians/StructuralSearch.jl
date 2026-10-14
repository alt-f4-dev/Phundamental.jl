# Batch 11: finite, symmetry-complete structural search over an existing HamiltonianSpace.
# A structural unit contains complete Hermitian invariant basis directions, never individual monomials.

"""Coefficient-ratio-aware identity of one real direction in the canonical operator-coordinate space."""
struct InvariantDirectionSignature
    coordinates::Tuple
end

Base.:(==)(a::InvariantDirectionSignature, b::InvariantDirectionSignature) = a.coordinates == b.coordinates
Base.isequal(a::InvariantDirectionSignature, b::InvariantDirectionSignature) = isequal(a.coordinates, b.coordinates)
Base.hash(s::InvariantDirectionSignature, h::UInt) = hash(s.coordinates, h)

"""Identify one physically specified, finitely generated invariant basis and its structural-unit partition."""
struct StructuralFamilySignature
    specification::Any
    directions::Tuple
    partition::Tuple
end

Base.:(==)(a::StructuralFamilySignature, b::StructuralFamilySignature) = (a.specification, a.directions, a.partition) == (b.specification, b.directions, b.partition)
Base.isequal(a::StructuralFamilySignature, b::StructuralFamilySignature) = isequal((a.specification, a.directions, a.partition), (b.specification, b.directions, b.partition))
Base.hash(s::StructuralFamilySignature, h::UInt) = hash((s.specification, s.directions, s.partition), h)

"""One indivisible structural-search object containing one or more complete invariant directions."""
struct StructuralUnit
    id::Symbol
    basis_indices::Tuple
    basis_labels::Tuple
    direction_signatures::Tuple
    metadata::Dict{Symbol,Any}
end

"""Partition and physical parent space defining a finite structural-search domain."""
struct StructuralFamily
    parent::HamiltonianSpace
    parameter_space::Union{Nothing,ParameterSpace}
    units::Vector{StructuralUnit}
    signature::StructuralFamilySignature
    metadata::Dict{Symbol,Any}
end

"""Canonical order-independent identity of selected complete invariant directions within one family."""
struct StructuralSelectionSignature
    family::StructuralFamilySignature
    directions::Tuple
end

Base.:(==)(a::StructuralSelectionSignature, b::StructuralSelectionSignature) = a.family == b.family && a.directions == b.directions
Base.isequal(a::StructuralSelectionSignature, b::StructuralSelectionSignature) = isequal((a.family, a.directions), (b.family, b.directions))
Base.hash(s::StructuralSelectionSignature, h::UInt) = hash((s.family, s.directions), h)

"""A nonempty selected Hamiltonian subspace, with its exact parent-basis coordinates."""
struct StructuralSelection
    family::StructuralFamily
    unit_ids::Tuple
    basis_indices::Tuple
    hamiltonian_space::HamiltonianSpace
    model_structure::ModelStructure
    signature::StructuralSelectionSignature
    metadata::Dict{Symbol,Any}
end

"""Recorded reversible structural transition between two selections of one family."""
struct StructuralMove
    kind::Symbol
    added::Tuple
    removed::Tuple
    family_signature::StructuralFamilySignature
    source::StructuralSelectionSignature
    target::StructuralSelectionSignature
end

"""Center selection, valid one-move neighbors, and corresponding structural transitions."""
struct StructuralNeighborhood
    center::StructuralSelection
    neighbors::Vector{StructuralSelection}
    moves::Vector{StructuralMove}
    constraints::NamedTuple
end

"""Exact parent-coordinate correspondence for two nested selected subspaces."""
struct StructuralCoordinateMap
    small::StructuralSelection
    large::StructuralSelection
    positions::Tuple
end

"""Finite Hasse diagram under addition of precisely one complete structural unit."""
struct ModelSpaceHierarchy
    family::StructuralFamily
    selections::Vector{StructuralSelection}
    edges::Vector{Tuple{Int,Int}}
    metadata::Dict{Symbol,Any}
end

# Canonical numeric coordinates retain relative complex coefficients and quotient only real scalar rescaling.
# Quantization removes irrelevant roundoff from the canonical noncommutative coefficient collector.
function invariant_direction_signature(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra; digits::Integer=12)
    1 <= digits <= 15 || throw(ArgumentError("signature digits must lie in 1:15"))
    data = PhundamentalCore._numeric_linear_terms(algebra, expression)
    isnothing(data) && throw(ArgumentError("invariant directions require canonical numeric operator coefficients"))
    coefficients = first(data)
    isempty(coefficients) && throw(ArgumentError("a zero operator cannot define an invariant direction"))
    keys_sorted = sort!(collect(keys(coefficients)); by=repr)
    scale = maximum(abs(coefficients[key]) for key in keys_sorted)
    isfinite(scale) && scale > 0 || throw(ArgumentError("invariant direction must have finite, nonzero coefficient norm"))
    normalized = ComplexF64[coefficients[key] / scale for key in keys_sorted]
    # Fix only the REAL orientation; complex phase rotation would change a Hermitian real direction.
    leading = first(normalized)
    orientation = abs(real(leading)) >= abs(imag(leading)) ? real(leading) : imag(leading)
    orientation < 0 && (normalized .*= -1)
    coordinates = Tuple((PhundamentalCore._freeze_structural_value(key),
                         iszero(round(real(value); digits=digits)) ? 0.0 : round(real(value); digits=digits),
                         iszero(round(imag(value); digits=digits)) ? 0.0 : round(imag(value); digits=digits))
                        for (key, value) in zip(keys_sorted, normalized))
    return InvariantDirectionSignature(coordinates)
end

function invariant_direction_signature(space::HamiltonianSpace, index::Integer; kwargs...)
    return invariant_direction_signature(space.basis[Int(index)], space.specification.algebra; kwargs...)
end

# Physical callback captures and crystallographic geometry belong to the family identity.
# Machine-readable cross-process callback certification is deliberately reserved for Batch 12.
function _physical_callback_key(callback)
    isnothing(callback) && return nothing
    callback isa Function || return PhundamentalCore._freeze_structural_value(callback)
    T = typeof(callback)
    captures = Tuple((fieldname(T, i), PhundamentalCore._freeze_structural_value(getfield(callback, i))) for i in 1:fieldcount(T))
    return (:callback, string(parentmodule(T)), nameof(T), captures)
end

function _physical_constraint_key(constraint::PredicateAdmissibility)
    return (:predicate, constraint.name, constraint.description, _physical_callback_key(constraint.predicate))
end

_physical_constraint_key(constraint::AbstractAdmissibilityCondition) = PhundamentalCore._freeze_structural_value(constraint)

function _family_specification_key(space::HamiltonianSpace; include_domain::Bool=true)
    spec = space.specification
    geometry = get(spec.metadata, :crystal_structure, nothing)
    generation_domain = get(spec.metadata, :generation_domain, nothing)
    site_mapping = get(spec.metadata, :operator_site_mapping, nothing)
    options = get(spec.metadata, :crystallography_options, nothing)
    physical = (space_group=spec.space_group, wyckoff_orbits=spec.wyckoff_orbits,
                local_spaces=spec.local_spaces, algebra=spec.algebra,
                symmetry_action=_physical_callback_key(spec.symmetry_action),
                conditions=Tuple(_physical_constraint_key(c) for c in spec.admissibility_conditions), max_range=spec.max_range,
                max_body_order=spec.max_body_order, range_evaluator=_physical_callback_key(spec.range_evaluator),
                crystal=geometry, generation_domain=generation_domain, operator_site_mapping=site_mapping,
                crystallography_options=options,
                completeness_scope=include_domain ? get(space.basis.metadata, :completeness_scope, :unknown) : nothing,
                candidate_source=include_domain ? get(space.metadata, :candidate_source, :unknown) : nothing)
    return PhundamentalCore._freeze_structural_value(physical)
end

function _parameter_family_key(schema::Union{Nothing,ParameterSpace}, signatures)
    isnothing(schema) && return nothing
    # Attach scalar bounds and fixed-reference values to canonical direction identity,
    # not human-readable names or mutable initial values for free parameters.
    descriptions = [(signatures[i], spec.lower, spec.upper, spec.unit, spec.fixed,
                     spec.fixed || spec.lower == spec.upper ? spec.value : nothing,
                     PhundamentalCore._freeze_structural_value(spec.prior))
                    for (i, spec) in enumerate(schema.specifications)]
    sort!(descriptions; by=entry -> repr(first(entry)))
    A, b = parameter_constraint_matrix(schema)
    rows = Any[]
    for i in axes(A, 1)
        coefficients = Tuple(sort!([(signatures[j], A[i, j]) for j in axes(A, 2) if !iszero(A[i, j])]; by=repr))
        push!(rows, (coefficients, b[i]))
    end
    sort!(rows; by=repr)
    return PhundamentalCore._freeze_structural_value((Tuple(descriptions), Tuple(rows)))
end

function _basis_direction_signatures(space::HamiltonianSpace)
    signatures = [invariant_direction_signature(space, i) for i in 1:length(space)]
    length(unique(signatures)) == length(signatures) || throw(ArgumentError("structural search requires independent, nonproportional parent basis directions"))
    return signatures
end

function _group_indices(space::HamiltonianSpace, entries)
    selectors = entries isa AbstractVector || entries isa Tuple ? entries : (entries,)
    isempty(selectors) && throw(ArgumentError("structural units cannot be empty"))
    indices = Int[]
    for selector in selectors
        index = if selector isa Integer
            Int(selector)
        elseif selector isa Symbol || selector isa AbstractString
            position = findfirst(==(Symbol(selector)), space.basis.labels)
            isnothing(position) && throw(ArgumentError("unknown invariant basis label :$(selector)"))
            position
        else
            throw(ArgumentError("basis selectors must be integer indices or basis labels"))
        end
        1 <= index <= length(space) || throw(BoundsError(space.basis.operators, index))
        index in indices && throw(ArgumentError("repeated basis direction $(index) in one structural unit"))
        push!(indices, index)
    end
    return sort!(indices)
end

"""
    structural_family(space; parameter_space=nothing, groups=nothing, require_crystallographic=false)
    structural_family(generation_result; groups=nothing)

Partition an already generated real Hermitian basis into complete invariant-direction units.
Explicit groups must partition all parent directions exactly once.
"""
function structural_family(space::HamiltonianSpace; parameter_space=nothing, groups=nothing,
                           require_crystallographic::Bool=false, metadata=Dict{Symbol,Any}())
    if require_crystallographic
        get(space.basis.metadata, :generator, nothing) == :crystallographic_fixed_space ||
            throw(ArgumentError("the parent basis is not certified as a crystallographic fixed space"))
    end
    PhundamentalCore._require_complete_admissibility(admissibility_report(space), "parent structural family")
    if !isnothing(parameter_space)
        parameter_space isa ParameterSpace || throw(ArgumentError("parameter_space must be a ParameterSpace"))
        length(parameter_space) == length(space) || throw(DimensionMismatch("parent parameter schema dimension does not match"))
        isnothing(parameter_space.hamiltonian_space) && throw(ArgumentError("parent parameter schema must be bound to its HamiltonianSpace"))
        parameter_space.hamiltonian_space === space || throw(ArgumentError("parent parameter schema must belong to this exact HamiltonianSpace"))
    end
    signatures = _basis_direction_signatures(space)
    groups_iter = isnothing(groups) ? [label => (index,) for (index, label) in enumerate(space.basis.labels)] : collect(groups)
    isempty(groups_iter) && throw(ArgumentError("structural family needs a nonempty partition"))
    all(group -> group isa Pair, groups_iter) || throw(ArgumentError("groups must contain unit_id => basis_selector pairs"))
    units = StructuralUnit[]
    covered = Int[]
    used_ids = Set{Symbol}()
    for group in groups_iter
        name = Symbol(first(group))
        name in used_ids && throw(ArgumentError("duplicate structural unit id :$(name)"))
        push!(used_ids, name)
        indices = _group_indices(space, last(group))
        append!(covered, indices)
        orders = Tuple(applicable(body_order, space.specification.algebra, space.basis[i]) ? body_order(space.specification.algebra, space.basis[i]) : nothing for i in indices)
        ranges = Tuple(isnothing(space.specification.range_evaluator) ? nothing : space.specification.range_evaluator(space.basis[i]) for i in indices)
        unit_metadata = Dict{Symbol,Any}(:symmetry_complete => require_crystallographic || get(space.basis.metadata, :generator, nothing) == :crystallographic_fixed_space,
                                         :body_orders => orders, :physical_ranges => ranges)
        push!(units, StructuralUnit(name, Tuple(indices), Tuple(space.basis.labels[indices]), Tuple(signatures[indices]), unit_metadata))
    end
    sort(covered) == collect(1:length(space)) || throw(ArgumentError("structural units must partition the entire parent basis without overlaps or omissions"))
    unit_keys = [Tuple(sort!(collect(unit.direction_signatures); by=repr)) for unit in units]
    family_signature = StructuralFamilySignature((_family_specification_key(space), _parameter_family_key(parameter_space, signatures)),
        Tuple(sort!(copy(signatures); by=repr)), Tuple(sort!(unit_keys; by=repr)))
    stored = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    stored[:parent_completeness_scope] = get(space.basis.metadata, :completeness_scope, :unknown)
    stored[:parent_generator] = get(space.basis.metadata, :generator, :unknown)
    return StructuralFamily(space, parameter_space, units, family_signature, stored)
end

function structural_family(result::HamiltonianGenerationResult; metadata=Dict{Symbol,Any}(), kwargs...)
    stored = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    # Retain the authoritative Batch 10 material, crystal, symmetry dataset and parameter family for downstream provenance.
    stored[:generation_result] = result
    return structural_family(result.hamiltonian_space; parameter_space=result.parameter_space,
                             require_crystallographic=true, metadata=stored, kwargs...)
end

function _unit_ids(family::StructuralFamily, selector)
    entries = selector isa Symbol || selector isa AbstractString ? (selector,) : selector
    ids = Symbol[]
    for entry in entries
        id = Symbol(entry)
        any(unit -> unit.id == id, family.units) || throw(ArgumentError("unknown structural unit :$(id)"))
        id in ids && throw(ArgumentError("duplicate structural unit selector :$(id)"))
        push!(ids, id)
    end
    return Tuple(unit.id for unit in family.units if unit.id in ids)
end

"""Select a nonempty symmetry-complete Hamiltonian subspace from one fixed structural family."""
function structural_selection(family::StructuralFamily, selector)
    selected_ids = _unit_ids(family, selector)
    isempty(selected_ids) && throw(ArgumentError("empty HamiltonianBasis selections are not supported"))
    indices = sort!(Int[index for unit in family.units if unit.id in selected_ids for index in unit.basis_indices])
    parent = family.parent
    basis_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(parent.basis.metadata))
    basis_metadata[:completeness_scope] = :selected_structural_subspace
    basis_metadata[:parent_completeness_scope] = get(parent.basis.metadata, :completeness_scope, :unknown)
    basis_metadata[:selected_parent_basis_indices] = Tuple(indices)
    child_basis = HamiltonianBasis(AbstractOperatorExpr[parent.basis.operators[i] for i in indices];
                                  labels=parent.basis.labels[indices], metadata=basis_metadata)
    child_metadata = Dict{Symbol,Any}(:structural_scope => :selected_structural_subspace,
                                      :parent_completeness_scope => basis_metadata[:parent_completeness_scope],
                                      :selected_units => selected_ids, :parent_basis_indices => Tuple(indices))
    child = HamiltonianSpace(parent.specification, child_basis; metadata=child_metadata)
    PhundamentalCore._require_complete_admissibility(admissibility_report(child), "selected structural subspace")
    signature = StructuralSelectionSignature(family.signature,
        Tuple(sort!([invariant_direction_signature(parent, i) for i in indices]; by=repr)))
    model_structure = structure(child)
    return StructuralSelection(family, selected_ids, Tuple(indices), child, model_structure, signature, child_metadata)
end

structural_signature(selection::StructuralSelection) = selection.signature
structural_signature(family::StructuralFamily) = family.signature
structure(selection::StructuralSelection) = selection.model_structure
interaction_graph(selection::StructuralSelection) = interaction_graph(selection.model_structure)

function _same_family(a::StructuralSelection, b::StructuralSelection)
    a.family === b.family || throw(ArgumentError("structural operations require selections of the same operational family"))
    return true
end

"""Build a validated nontrivial structural transition by adding and/or removing entire units."""
function structural_transition(selection::StructuralSelection; add_units=(), remove_units=())
    family = selection.family
    added = _unit_ids(family, add_units)
    removed = _unit_ids(family, remove_units)
    current = Set(selection.unit_ids)
    any(id -> id in current, added) && throw(ArgumentError("cannot add an already selected structural unit"))
    any(id -> !(id in current), removed) && throw(ArgumentError("cannot remove a nonselected structural unit"))
    any(id -> id in removed, added) && throw(ArgumentError("cannot add and remove the same structural unit"))
    isempty(added) && isempty(removed) && throw(ArgumentError("a structural transition cannot be a no-op"))
    target = union(setdiff(current, Set(removed)), Set(added))
    isempty(target) && throw(ArgumentError("a structural transition cannot select the empty basis"))
    return structural_selection(family, Tuple(target))
end

"""Record and classify a transition between selections of the same structural family."""
function structural_move(source::StructuralSelection, target::StructuralSelection)
    _same_family(source, target)
    added = Tuple(id for id in target.unit_ids if !(id in source.unit_ids))
    removed = Tuple(id for id in source.unit_ids if !(id in target.unit_ids))
    isempty(added) && isempty(removed) && throw(ArgumentError("a structural move cannot be a no-op"))
    kind = if isempty(removed)
        :add
    elseif isempty(added)
        :remove
    elseif length(added) == 1 && length(removed) == 1
        :replace
    else
        :compound
    end
    return StructuralMove(kind, added, removed, source.family.signature, source.signature, target.signature)
end

"""Replay a recorded structural move with exact family, source, and target signature checks."""
function apply_structural_move(selection::StructuralSelection, move::StructuralMove)
    selection.family.signature == move.family_signature || throw(ArgumentError("structural move belongs to another family"))
    selection.signature == move.source || throw(ArgumentError("structural move source signature mismatch"))
    result = structural_transition(selection; add_units=move.added, remove_units=move.removed)
    result.signature == move.target || throw(ArgumentError("structural move target signature mismatch"))
    structural_move(selection, result).kind == move.kind || throw(ArgumentError("structural move kind mismatch"))
    return result
end

function _selection_constraints(family::StructuralFamily, required_units, forbidden_units, min_units::Integer, max_units::Integer)
    required = _unit_ids(family, required_units)
    forbidden = _unit_ids(family, forbidden_units)
    isempty(intersect(Set(required), Set(forbidden))) || throw(ArgumentError("required and forbidden structural units overlap"))
    1 <= min_units <= max_units <= length(family.units) || throw(ArgumentError("invalid min_units/max_units for a nonempty finite family"))
    length(required) <= max_units || throw(ArgumentError("required units exceed max_units"))
    return (required_units=required, forbidden_units=forbidden, min_units=Int(min_units), max_units=Int(max_units))
end

function _allowed_selection(ids, restrictions)
    all(id -> id in ids, restrictions.required_units) || return false
    any(id -> id in ids, restrictions.forbidden_units) && return false
    return restrictions.min_units <= length(ids) <= restrictions.max_units
end

"""
    structural_neighbors(selection; allow_add=true, allow_remove=true, allow_replace=true, ...)

Enumerate and deduplicate legal one-unit additions, removals and replacements.
"""
function structural_neighbors(selection::StructuralSelection; allow_add::Bool=true, allow_remove::Bool=true,
                              allow_replace::Bool=true, required_units=(), forbidden_units=(), min_units::Integer=1,
                              max_units::Integer=length(selection.family.units), max_neighbors::Integer=4096)
    max_neighbors >= 0 || throw(ArgumentError("max_neighbors must be nonnegative"))
    family = selection.family
    restrictions = _selection_constraints(family, required_units, forbidden_units, min_units, max_units)
    _allowed_selection(selection.unit_ids, restrictions) || throw(ArgumentError("center selection violates neighborhood constraints"))
    active = Set(selection.unit_ids)
    missing_ids = [unit.id for unit in family.units if !(unit.id in active)]
    present_ids = [unit.id for unit in family.units if unit.id in active]
    neighbors = StructuralSelection[]
    moves = StructuralMove[]
    seen = Set{StructuralSelectionSignature}()
    function append_neighbor(added, removed)
        ids = union(setdiff(active, Set(removed)), Set(added))
        _allowed_selection(ids, restrictions) || return
        neighbor = structural_transition(selection; add_units=added, remove_units=removed)
        neighbor.signature in seen && return
        length(neighbors) < max_neighbors || throw(ArgumentError("structural neighborhood exceeds max_neighbors=$(max_neighbors)"))
        push!(seen, neighbor.signature)
        push!(neighbors, neighbor)
        push!(moves, structural_move(selection, neighbor))
        return
    end
    allow_add && foreach(id -> append_neighbor((id,), ()), missing_ids)
    allow_remove && foreach(id -> append_neighbor((), (id,)), present_ids)
    if allow_replace
        for removed in present_ids, added in missing_ids
            append_neighbor((added,), (removed,))
        end
    end
    return StructuralNeighborhood(selection, neighbors, moves, merge(restrictions,
        (allow_add=allow_add, allow_remove=allow_remove, allow_replace=allow_replace, max_neighbors=Int(max_neighbors))))
end

function _admissible_count(family::StructuralFamily, restrictions)
    nfree = length(family.units) - length(restrictions.required_units) - length(restrictions.forbidden_units)
    nrequired = length(restrictions.required_units)
    return sum(binomial(big(nfree), k - nrequired) for k in max(1, restrictions.min_units):restrictions.max_units
               if 0 <= k - nrequired <= nfree; init=big(0))
end

"""Exhaustively enumerate constrained nonempty selections, refusing to exceed `max_structures` before allocating them."""
function enumerate_structures(family::StructuralFamily; required_units=(), forbidden_units=(), min_units::Integer=1,
                              max_units::Integer=length(family.units), max_structures::Integer=4096)
    max_structures >= 0 || throw(ArgumentError("max_structures must be nonnegative"))
    restrictions = _selection_constraints(family, required_units, forbidden_units, min_units, max_units)
    total = _admissible_count(family, restrictions)
    total <= max_structures || throw(ArgumentError("enumeration requires $(total) structures, above max_structures=$(max_structures)"))
    selected = Symbol[restrictions.required_units...]
    available = Symbol[unit.id for unit in family.units if !(unit.id in restrictions.required_units) && !(unit.id in restrictions.forbidden_units)]
    outputs = StructuralSelection[]
    function visit(index::Int)
        if index > length(available)
            _allowed_selection(selected, restrictions) && push!(outputs, structural_selection(family, selected))
            return
        end
        visit(index + 1)
        push!(selected, available[index])
        visit(index + 1)
        pop!(selected)
        return
    end
    visit(1)
    sort!(outputs; by=selection -> (length(selection.unit_ids), repr(selection.signature.directions)))
    return outputs
end

"""Test inclusion of whole invariant-direction units within one structural family."""
function is_subspace(a::StructuralSelection, b::StructuralSelection)
    _same_family(a, b)
    return issubset(Set(a.unit_ids), Set(b.unit_ids))
end

"""Test real canonical-span inclusion for two HamiltonianSpace objects with the same physical specification."""
function is_subspace(a::HamiltonianSpace, b::HamiltonianSpace)
    _family_specification_key(a; include_domain=false) == _family_specification_key(b; include_domain=false) || return false
    tolerance = max(get(a.basis.metadata, :linear_tolerance, 1e-10), get(b.basis.metadata, :linear_tolerance, 1e-10))
    return all(op -> structural_span_membership(b.basis.operators, op, b.specification.algebra; tolerance=tolerance) === true, a.basis.operators)
end

"""Prepare exact selected-coefficient insertion and restriction for two nested structural spaces."""
function structural_coordinate_map(small::StructuralSelection, large::StructuralSelection)
    is_subspace(small, large) || throw(ArgumentError("first selection must be a subspace of the second"))
    positions = Tuple(findfirst(==(index), large.basis_indices) for index in small.basis_indices)
    any(isnothing, positions) && throw(ArgumentError("nested structural coordinates are inconsistent"))
    return StructuralCoordinateMap(small, large, positions)
end

function embed_coordinates(map::StructuralCoordinateMap, values::AbstractVector{<:Real})
    length(values) == length(map.positions) || throw(DimensionMismatch("small-space coefficient length mismatch"))
    result = zeros(promote_type(eltype(values), Float64), length(map.large.basis_indices))
    for (index, value) in zip(map.positions, values)
        result[index] = value
    end
    return result
end

function project_coordinates(map::StructuralCoordinateMap, values::AbstractVector{<:Real}; strict::Bool=false, atol::Real=1e-10)
    length(values) == length(map.large.basis_indices) || throw(DimensionMismatch("large-space coefficient length mismatch"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    if strict
        positions = Set(map.positions)
        any(i -> !(i in positions) && !(abs(values[i]) <= atol), eachindex(values)) &&
            throw(ArgumentError("cannot strictly project nonzero omitted structural coefficients"))
    end
    return collect(values[[map.positions...]])
end

function _selected_values(selection::StructuralSelection, values)
    names = isnothing(selection.family.parameter_space) ? selection.hamiltonian_space.basis.labels :
        parameter_names(selection.family.parameter_space)[[selection.basis_indices...]]
    if values isa AbstractVector{<:Real}
        length(values) == length(names) || throw(DimensionMismatch("selected coefficient vector length mismatch"))
        return collect(values)
    elseif values isa AbstractDict || values isa NamedTuple
        provided = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(values))
        Set(keys(provided)) == Set(names) || throw(ArgumentError("selected parameter names must match exactly"))
        all(v -> v isa Real, Base.values(provided)) || throw(ArgumentError("parameter values must be real"))
        return Float64[provided[name] for name in names]
    end
    throw(ArgumentError("selected coefficients must be a real vector, dictionary, or named tuple"))
end

"""Embed selected coefficients into the full parent basis, with every omitted coefficient exactly zero."""
function embed_parameters(selection::StructuralSelection, values)
    selected_values = _selected_values(selection, values)
    parent = zeros(promote_type(eltype(selected_values), Float64), length(selection.family.parent))
    parent[[selection.basis_indices...]] .= selected_values
    schema = selection.family.parameter_space
    if !isnothing(schema)
        report = parameter_constraint_report(schema, parent)
        report.valid || throw(ArgumentError("embedded coefficients violate the parent schema: $(join(parameter_constraint_reasons(report), "; "))"))
    end
    return parent
end

"""Restrict a parent coefficient point; strict mode refuses to discard nonzero omitted coefficients."""
function project_parameters(selection::StructuralSelection, parent_values::AbstractVector{<:Real}; strict::Bool=true, atol::Real=1e-10)
    length(parent_values) == length(selection.family.parent) || throw(DimensionMismatch("parent coefficient vector length mismatch"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    if strict
        retained = Set(selection.basis_indices)
        any(i -> !(i in retained) && !(abs(parent_values[i]) <= atol), eachindex(parent_values)) &&
            throw(ArgumentError("strict structural projection would discard a nonzero parent coefficient"))
    end
    return collect(parent_values[[selection.basis_indices...]])
end

project_parameters(selection::StructuralSelection, point::ParameterPoint; kwargs...) =
    (point.space === selection.family.parameter_space || throw(ArgumentError("parameter point does not belong to this parent family"));
     project_parameters(selection, point.values; kwargs...))

function _project_parent_constraint(constraint::FixedParameterConstraint, retained::Set{Symbol})
    if constraint.parameter in retained
        return AbstractParameterConstraint[constraint]
    end
    isapprox(constraint.value, 0; atol=constraint.atol, rtol=constraint.rtol) ||
        throw(ArgumentError("omitted parameter :$(constraint.parameter) is constrained to nonzero value $(constraint.value)"))
    return AbstractParameterConstraint[]
end

function _project_parent_constraint(constraint::EqualParameterConstraint, retained::Set{Symbol})
    left = constraint.left in retained
    right = constraint.right in retained
    if left && right
        return AbstractParameterConstraint[constraint]
    elseif left || right
        name = left ? constraint.left : constraint.right
        return AbstractParameterConstraint[FixedParameterConstraint(name, 0.0; atol=constraint.atol, rtol=constraint.rtol,
                                            metadata=Dict{Symbol,Any}(:origin => :omitted_equal_parameter))]
    end
    return AbstractParameterConstraint[]
end

function _project_parent_constraint(constraint::LinearParameterConstraint, retained::Set{Symbol})
    coefficients = Dict(name => value for (name, value) in constraint.coefficients if name in retained)
    if isempty(coefficients)
        isapprox(constraint.rhs, 0; atol=constraint.atol, rtol=constraint.rtol) ||
            throw(ArgumentError("omitted linear parameter constraint :$(constraint.name) is inconsistent with zero coefficients"))
        return AbstractParameterConstraint[]
    end
    return AbstractParameterConstraint[LinearParameterConstraint(coefficients, constraint.rhs; name=constraint.name,
                                            atol=constraint.atol, rtol=constraint.rtol, metadata=constraint.metadata)]
end

_project_parent_constraint(constraint::AbstractParameterConstraint, ::Set{Symbol}) =
    throw(ArgumentError("no valid zero-coefficient projection is defined for $(typeof(constraint))"))

"""
    parameter_subspace(selection, values=nothing)

Construct the selected ParameterSpace, inheriting bounds, units, priors and every supported parent equality.
Omitted coefficients are fixed to zero in the parent schema; unknown constraints are rejected.
"""
function parameter_subspace(selection::StructuralSelection, values=nothing)
    parent_schema = selection.family.parameter_space
    parent_schema isa ParameterSpace || throw(ArgumentError("selected family has no attached parent ParameterSpace"))
    indices = [selection.basis_indices...]
    names = Set(parameter_names(parent_schema)[indices])
    defaults = isnothing(values) ? [parent_schema.specifications[i].value for i in indices] : _selected_values(selection, values)
    embed_parameters(selection, defaults)
    specs = ParameterSpec[]
    for (i, value) in zip(indices, defaults)
        source = parent_schema.specifications[i]
        push!(specs, ParameterSpec(source.name, value; lower=source.lower, upper=source.upper,
                                   unit=source.unit, fixed=source.fixed, prior=source.prior, metadata=source.metadata))
    end
    constraints = AbstractParameterConstraint[]
    for constraint in parent_schema.constraints
        append!(constraints, _project_parent_constraint(constraint, names))
    end
    # Omitted implicit fixed flags and degenerate bounds are checked against zero explicitly.
    for source in parent_schema.specifications
        if !(source.name in names) && (source.fixed || source.lower == source.upper)
            isapprox(source.value, 0; atol=1e-10, rtol=1e-10) ||
                throw(ArgumentError("omitted fixed parent parameter :$(source.name) cannot be set to zero"))
        end
    end
    stored_metadata = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(parent_schema.metadata))
    stored_metadata[:structural_selection_signature] = selection.signature
    stored_metadata[:parent_parameter_names] = Tuple(parameter_names(parent_schema))
    return ParameterSpace(specs; hamiltonian_space=selection.hamiltonian_space, constraints=Tuple(constraints), metadata=stored_metadata)
end

"""Construct the finite within-family Hasse diagram of allowed nonempty structural selections."""
function model_space_hierarchy(family::StructuralFamily; kwargs...)
    selections = enumerate_structures(family; kwargs...)
    positions = Dict(selection.unit_ids => i for (i, selection) in enumerate(selections))
    edges = Tuple{Int,Int}[]
    for (i, selection) in enumerate(selections)
        active = Set(selection.unit_ids)
        for unit in family.units
            unit.id in active && continue
            successor = Tuple(member.id for member in family.units if member.id in active || member.id == unit.id)
            j = get(positions, successor, nothing)
            isnothing(j) || push!(edges, (i, j))
        end
    end
    sort!(edges)
    return ModelSpaceHierarchy(family, selections, edges,
        Dict{Symbol,Any}(:edge_semantics => :add_exactly_one_structural_unit, :node_count => length(selections), :edge_count => length(edges)))
end

function _hierarchy_index(hierarchy::ModelSpaceHierarchy, selection::StructuralSelection)
    selection.family === hierarchy.family || throw(ArgumentError("selection belongs to a different structural hierarchy family"))
    index = findfirst(item -> item.signature == selection.signature, hierarchy.selections)
    isnothing(index) && throw(ArgumentError("selection does not belong to the enumerated hierarchy"))
    return index
end

"""Return the immediate one-unit-smaller structural parents of a selected hierarchy node."""
function parent_spaces(hierarchy::ModelSpaceHierarchy, selection::StructuralSelection)
    node = _hierarchy_index(hierarchy, selection)
    return [hierarchy.selections[i] for (i, j) in hierarchy.edges if j == node]
end

"""Return the immediate one-unit-larger structural children of a selected hierarchy node."""
function child_spaces(hierarchy::ModelSpaceHierarchy, selection::StructuralSelection)
    node = _hierarchy_index(hierarchy, selection)
    return [hierarchy.selections[j] for (i, j) in hierarchy.edges if i == node]
end
