"""Crystallographic fixed-space generator compatible with `generated_basis(specification, generator)`."""
struct CrystallographicGenerator{S,C,O}
    structure::S
    candidates::C
    options::O
    labels::Union{Nothing,Vector{Symbol}}
end

function CrystallographicGenerator(structure::CrystalStructure, candidates::AbstractVector{<:AbstractOperatorExpr}; options=nothing, labels=nothing)
    isnothing(options) || options isa CrystallographyOptions || throw(ArgumentError("options must be a CrystallographyOptions instance or nothing"))
    names = isnothing(labels) ? nothing : Symbol.(collect(labels))
    return CrystallographicGenerator(structure, collect(candidates), options, names)
end

function _primitive_generators(algebra::SpinAlgebra)
    generators = AbstractOperatorExpr[]
    for site in 1:algebra.nsites
        append!(generators, AbstractOperatorExpr[Sx(site), Sy(site), Sz(site)])
    end
    return generators
end

function _primitive_generators(algebra::BosonAlgebra)
    generators = AbstractOperatorExpr[]
    for mode in 1:algebra.nmodes
        append!(generators, AbstractOperatorExpr[b(mode), b(mode)'])
    end
    return generators
end

function _primitive_generators(algebra::FermionAlgebra)
    generators = AbstractOperatorExpr[]
    for mode in 1:algebra.nmodes
        append!(generators, AbstractOperatorExpr[c(mode), c(mode)'])
    end
    return generators
end

_primitive_generators(algebra::MajoranaAlgebra) = AbstractOperatorExpr[γ(i) for i in 1:algebra.ngenerators]

function _primitive_generators(algebra::CoordinateMomentumAlgebra)
    generators = AbstractOperatorExpr[]
    for mode in 1:algebra.ndof
        append!(generators, AbstractOperatorExpr[xop(mode), pop(mode)])
    end
    return generators
end

function _primitive_generators(algebra::CompositeAlgebra)
    generators = AbstractOperatorExpr[]
    for (factor, subalgebra) in enumerate(algebra.factors)
        for operator in _primitive_generators(subalgebra)
            operator isa AbstractPrimitiveOperator || throw(ArgumentError("composite primitive enumeration requires primitive factor generators"))
            push!(generators, atfactor(factor, operator))
        end
    end
    return generators
end

function _expression_key(expression::IdentityOperator)
    return (:identity, expression.label)
end

_expression_key(expression::SpinX) = (:spin_x, expression.site)
_expression_key(expression::SpinY) = (:spin_y, expression.site)
_expression_key(expression::SpinZ) = (:spin_z, expression.site)
_expression_key(expression::SpinPlus) = (:spin_plus, expression.site)
_expression_key(expression::SpinMinus) = (:spin_minus, expression.site)
_expression_key(expression::BosonAnnihilate) = (:boson_annihilate, expression.mode)
_expression_key(expression::BosonCreate) = (:boson_create, expression.mode)
_expression_key(expression::FermionAnnihilate) = (:fermion_annihilate, expression.mode)
_expression_key(expression::FermionCreate) = (:fermion_create, expression.mode)
_expression_key(expression::MajoranaOperator) = (:majorana, expression.index)
_expression_key(expression::PositionOperator) = (:position, expression.mode)
_expression_key(expression::MomentumOperator) = (:momentum, expression.mode)
_expression_key(expression::NumberOperator) = (:number, expression.kind, expression.mode)
_expression_key(expression::FactorOperator) = (:factor, expression.factor, _expression_key(expression.operator))
_expression_key(expression::OperatorProduct) = (:product, Tuple(_expression_key(factor) for factor in expression.factors))
_expression_key(expression::OperatorSum) = (:sum, Tuple(_expression_key(term) for term in expression.terms))
_expression_key(expression::ScaledOperator) = (:scaled, expression.coefficient, _expression_key(expression.operator))

function _canonical_core(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra)
    canonical = canonicalize(algebra, expression)
    canonical isa ScaledOperator && return canonical.coefficient, canonical.operator
    return 1, canonical
end

function _deduplicate_expressions(expressions::AbstractVector{<:AbstractOperatorExpr}, algebra::AbstractOperatorAlgebra)
    output = AbstractOperatorExpr[]
    seen = Set{Any}()
    for expression in expressions
        canonical = canonicalize(algebra, expression)
        key = _expression_key(canonical)
        key in seen && continue
        push!(seen, key)
        push!(output, canonical)
    end
    return output
end

_default_operator_degree(::Union{FermionAlgebra,BosonAlgebra,MajoranaAlgebra,CompositeAlgebra}, max_body_order::Integer) = 2 * max_body_order
_default_operator_degree(::Union{SpinAlgebra,CoordinateMomentumAlgebra}, max_body_order::Integer) = max_body_order

function enumerate_operator_candidates(algebra::AbstractOperatorAlgebra; max_body_order::Integer=2, max_operator_degree=nothing, include_identity::Bool=false)
    max_body_order > 0 || throw(ArgumentError("max_body_order must be positive"))
    degree_cutoff = isnothing(max_operator_degree) ? _default_operator_degree(algebra, max_body_order) : Int(max_operator_degree)
    degree_cutoff > 0 || throw(ArgumentError("max_operator_degree must be positive"))
    primitives = _primitive_generators(algebra)
    isempty(primitives) && throw(ArgumentError("operator algebra has no enumerable primitive generators"))
    candidates = AbstractOperatorExpr[]
    include_identity && push!(candidates, IdentityOperator())
    frontier = copy(primitives)
    for degree in 1:degree_cutoff
        if degree > 1
            next_frontier = AbstractOperatorExpr[]
            for prefix in frontier, primitive in primitives
                push!(next_frontier, prefix * primitive)
            end
            frontier = next_frontier
        end
        for expression in frontier
            canonical = canonicalize(algebra, expression)
            body_order(algebra, canonical) <= max_body_order || continue
            push!(candidates, canonical)
        end
    end
    return _deduplicate_expressions(candidates, algebra)
end

function _linear_terms(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra)
    canonical = canonicalize(algebra, expression)
    terms = canonical isa OperatorSum ? canonical.terms : AbstractOperatorExpr[canonical]
    coefficients = Dict{Any,ComplexF64}()
    for term in terms
        coefficient, core = _canonical_core(term, algebra)
        value = ComplexF64(coefficient)
        iszero(value) && continue
        key = _expression_key(core)
        coefficients[key] = get(coefficients, key, 0.0 + 0.0im) + value
    end
    filter!(pair -> !iszero(last(pair)), coefficients)
    return coefficients
end

function _independent_hermitian_basis(expressions::AbstractVector{<:AbstractOperatorExpr}, algebra::AbstractOperatorAlgebra; tolerance::Real=1e-10)
    coefficient_maps = [_linear_terms(expression, algebra) for expression in expressions]
    term_keys = unique(vcat((collect(Base.keys(coefficient_map)) for coefficient_map in coefficient_maps)...))
    isempty(term_keys) && return AbstractOperatorExpr[]
    matrix = zeros(ComplexF64, length(term_keys), 0)
    selected = AbstractOperatorExpr[]
    rank_now = 0
    for (expression, coefficients) in zip(expressions, coefficient_maps)
        column = ComplexF64[get(coefficients, key, 0.0 + 0.0im) for key in term_keys]
        norm(column) <= tolerance && continue
        trial = hcat(matrix, column)
        rank_trial = rank(trial; atol=tolerance, rtol=tolerance)
        rank_trial > rank_now || continue
        matrix = trial
        rank_now = rank_trial
        push!(selected, canonicalize(algebra, expression))
    end
    return selected
end

function _hermitian_components(expression::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra; tolerance::Real=1e-10)
    plus = canonicalize(algebra, 0.5 * (expression + adjoint(expression)))
    minus = canonicalize(algebra, (-0.5im) * (expression - adjoint(expression)))
    output = AbstractOperatorExpr[]
    norm(collect(values(_linear_terms(plus, algebra)))) > tolerance && push!(output, plus)
    norm(collect(values(_linear_terms(minus, algebra)))) > tolerance && push!(output, minus)
    return output
end


function _resolved_crystallography_options(specification::GenerativeSpecification, options)
    if !isnothing(options)
        options isa CrystallographyOptions || throw(ArgumentError("options must be a CrystallographyOptions instance or nothing"))
        return options
    end
    stored = get(specification.metadata, :crystallography_options, nothing)
    return stored isa CrystallographyOptions ? stored : CrystallographyOptions()
end

function reynolds_project(expression::AbstractOperatorExpr, specification::GenerativeSpecification, structure::CrystalStructure, dataset::CrystallographicDataset; options=nothing)
    isempty(dataset.operations) && throw(ArgumentError("crystallographic dataset contains no symmetry operations"))
    resolved_options = _resolved_crystallography_options(specification, options)
    transformed = AbstractOperatorExpr[]
    sizehint!(transformed, length(dataset.operations))
    for operation in dataset.operations
        if specification.symmetry_action === standard_symmetry_action
            push!(transformed, standard_symmetry_action(expression, operation, structure, specification.algebra; site_tolerance=resolved_options.site_tolerance))
        else
            push!(transformed, specification.symmetry_action(expression, operation, structure, specification.algebra))
        end
    end
    return canonicalize(specification.algebra, (1 / length(transformed)) * OperatorSum(transformed))
end

function generate_invariant_basis(specification::GenerativeSpecification, structure::CrystalStructure, candidates::AbstractVector{<:AbstractOperatorExpr}; options=nothing, labels=nothing)
    length(structure) > 0 || throw(ArgumentError("crystal structure must contain at least one site"))
    resolved_options = _resolved_crystallography_options(specification, options)
    dataset = specification.space_group isa CrystallographicDataset ? specification.space_group : discover_symmetry(structure; options=resolved_options)
    projected = AbstractOperatorExpr[]
    rejected_body_order = 0
    rejected_admissibility = 0
    for candidate in candidates
        canonical_candidate = canonicalize(specification.algebra, candidate)
        if !isnothing(specification.max_body_order) && body_order(specification.algebra, canonical_candidate) > specification.max_body_order
            rejected_body_order += 1
            continue
        end
        if !admissible(specification, canonical_candidate)
            rejected_admissibility += 1
            continue
        end
        invariant = reynolds_project(canonical_candidate, specification, structure, dataset; options=resolved_options)
        append!(projected, _hermitian_components(invariant, specification.algebra; tolerance=resolved_options.linear_tolerance))
    end
    independent = _independent_hermitian_basis(projected, specification.algebra; tolerance=resolved_options.linear_tolerance)
    isempty(independent) && throw(ArgumentError("no nonzero Hermitian invariant operators remain after symmetry/admissibility projection"))
    names = isnothing(labels) ? [Symbol("Phi_", i) for i in eachindex(independent)] : Symbol.(collect(labels))
    length(names) == length(independent) || throw(DimensionMismatch("basis labels must match the generated invariant-space dimension"))
    metadata = Dict{Symbol,Any}(:generator => :crystallographic_fixed_space, :backend => get(dataset.metadata, :backend, :provided), :spacegroup_number => dataset.spacegroup_number, :international_symbol => dataset.international_symbol, :candidate_count => length(candidates), :rejected_body_order => rejected_body_order, :rejected_admissibility => rejected_admissibility, :invariant_dimension => length(independent), :completeness_scope => :supplied_candidate_space, :max_body_order => specification.max_body_order, :symprec => resolved_options.symprec, :site_tolerance => resolved_options.site_tolerance, :linear_tolerance => resolved_options.linear_tolerance)
    return HamiltonianBasis(independent; labels=names, metadata=metadata)
end

function (generator::CrystallographicGenerator)(specification::GenerativeSpecification)
    return generate_invariant_basis(specification, generator.structure, generator.candidates; options=generator.options, labels=generator.labels)
end

function crystallographic_specification(structure::CrystalStructure, local_spaces, algebra::AbstractOperatorAlgebra; symmetry_action=standard_symmetry_action, admissibility_conditions=(), max_range=nothing, max_body_order=nothing, options::CrystallographyOptions=CrystallographyOptions(), metadata=Dict{Symbol,Any}())
    dataset = discover_symmetry(structure; options=options)
    orbits = isempty(dataset.crystallographic_orbits) ? Tuple(structure.labels) : Tuple(dataset.crystallographic_orbits)
    merged_metadata = Dict{Symbol,Any}(metadata)
    merged_metadata[:crystal_structure] = structure
    merged_metadata[:crystallographic_backend] = :spglib
    merged_metadata[:crystallography_options] = options
    return GenerativeSpecification(dataset, orbits, local_spaces, algebra, symmetry_action; admissibility_conditions=admissibility_conditions, max_range=max_range, max_body_order=max_body_order, metadata=merged_metadata)
end

function generate_crystallographic_hamiltonian_space(specification::GenerativeSpecification, structure::CrystalStructure; candidates=nothing, max_operator_degree=nothing, options=nothing, metadata=Dict{Symbol,Any}())
    order = isnothing(specification.max_body_order) ? 2 : specification.max_body_order
    resolved_options = _resolved_crystallography_options(specification, options)
    resolved_degree = isnothing(candidates) ? (isnothing(max_operator_degree) ? _default_operator_degree(specification.algebra, order) : Int(max_operator_degree)) : nothing
    candidate_basis = isnothing(candidates) ? enumerate_operator_candidates(specification.algebra; max_body_order=order, max_operator_degree=resolved_degree) : collect(candidates)
    generator = CrystallographicGenerator(structure, candidate_basis; options=resolved_options)
    space_metadata = Dict{Symbol,Any}(metadata)
    space_metadata[:generation_backend] = :spglib_fixed_space
    space_metadata[:candidate_source] = isnothing(candidates) ? :automatic_enumeration : :supplied
    space_metadata[:max_body_order] = specification.max_body_order
    space_metadata[:max_operator_degree] = resolved_degree
    return generate_hamiltonian_space(specification, generator; metadata=space_metadata)
end
