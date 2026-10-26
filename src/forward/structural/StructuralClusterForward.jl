# Bond-resolved open-cluster reconstruction for structurally selected spin Hamiltonians.
# Periodic-image coefficients must not be reused as open-cluster couplings.

using LinearAlgebra: norm

"""Internal NLCE descriptor used only with an explicitly certified structural cluster builder."""
struct InducedStructuralSpinSpec <: AbstractHamiltonianSpec
    family_id::String
    selection_id::String
end

"""An NLCE method context retaining the physical structural selection and the finite-reference cluster policy."""
struct StructuralNLCEForwardContext{S,C,B} <: AbstractForwardMethodContext
    selection::S
    context::C
    bond_terms::B
end

function StructuralNLCEForwardContext(selection::StructuralSelection, max_order::Integer; bond_terms=nothing, kwargs...)
    context = NLCEForwardContext(max_order; kwargs...)
    isnothing(context.bonds) && throw(ArgumentError("structural NLCE requires explicit reference bonds for certified induced-cluster enumeration"))
    !isnothing(context.model_builder) && throw(ArgumentError("structural NLCE provides its own induced-cluster model builder; omit model_builder"))
    !isnothing(context.model_transform) && throw(ArgumentError("structural NLCE does not support an additional cluster model_transform"))
    entries = isnothing(bond_terms) ? nothing : collect(bond_terms)
    return StructuralNLCEForwardContext(selection, context, entries)
end

StructuralNLCEForwardContext(selection::StructuralSelection, context::NLCEForwardContext) =
    StructuralNLCEForwardContext(selection, context, nothing)

# Reject unmatched operator representations rather than silently changing their algebraic meaning.
_induced_spin_primitive(op::SpinX, indices) = haskey(indices, op.site) ? Sx(indices[op.site]) : nothing
_induced_spin_primitive(op::SpinY, indices) = haskey(indices, op.site) ? Sy(indices[op.site]) : nothing
_induced_spin_primitive(op::SpinZ, indices) = haskey(indices, op.site) ? Sz(indices[op.site]) : nothing
_induced_spin_primitive(op::SpinPlus, indices) = haskey(indices, op.site) ? Sp(indices[op.site]) : nothing
_induced_spin_primitive(op::SpinMinus, indices) = haskey(indices, op.site) ? Sm(indices[op.site]) : nothing
_induced_spin_primitive(op::AbstractPrimitiveOperator, indices) = throw(ArgumentError("induced spin clusters support only primitive spin operators, not $(typeof(op))"))

function _induced_spin_expression(expression::AbstractPrimitiveOperator, indices)
    expression isa IdentityOperator && throw(ArgumentError("an identity energy offset has no certified extensive cluster assignment"))
    return _induced_spin_primitive(expression, indices)
end

function _induced_spin_expression(expression::ScaledOperator, indices)
    mapped = _induced_spin_expression(expression.operator, indices)
    return isnothing(mapped) ? nothing : expression.coefficient * mapped
end

function _induced_spin_expression(expression::OperatorProduct, indices)
    terms = AbstractOperatorExpr[]
    for factor in expression.factors
        mapped = _induced_spin_expression(factor, indices)
        isnothing(mapped) && return nothing
        push!(terms, mapped)
    end
    return reduce(*, terms)
end

function _induced_spin_expression(expression::OperatorSum, indices)
    terms = AbstractOperatorExpr[]
    for term in expression.terms
        mapped = _induced_spin_expression(term, indices)
        isnothing(mapped) || push!(terms, mapped)
    end
    return isempty(terms) ? 0 * IdentityOperator() : reduce(+, terms)
end

function _certified_induced_spin_source(model::ManyBodyModel, selection::StructuralSelection, reference::Supercell)
    model.representation.algebra isa SpinAlgebra || throw(ArgumentError("induced structural NLCE currently supports SpinAlgebra only"))
    model.representation.algebra.nsites == nsites(reference) || throw(DimensionMismatch("source spin count and reference site count differ"))
    model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("induced cluster regeneration requires an explicit symbolic Hamiltonian"))
    source = Models._structural_provenance(model)
    isnothing(source) && throw(ArgumentError("source model lacks certified structural-generation provenance"))
    get(selection.family.parent.specification.metadata, :operator_site_mapping, :custom) === :one_index_per_crystallographic_site ||
        throw(ArgumentError("induced structural spin clusters require a certified one-index-per-site mapping"))
    source.selection_id == structural_id(selection) || throw(ArgumentError("source model does not match the chosen structural selection"))
    source.family_id == structural_id(selection.family) || throw(ArgumentError("source model does not match the chosen structural family"))
    reference === get(model.parameters, :cluster, nothing) || throw(ArgumentError("NLCE reference must be the same Supercell used to instantiate the structural model"))
    material = get(model.parameters, :material, nothing)
    material isa Material || throw(ArgumentError("structural NLCE requires certified source material provenance"))
    location = locate_in_family(selection.family, model)
    location.in_span && location.valid_parameters && !isnothing(location.selection) ||
        throw(ArgumentError("source model is not a valid numerical member of the declared structural family"))
    structural_id(location.selection) == structural_id(selection) || throw(ArgumentError("source model's active structural selection differs"))
    return material
end

# A periodic Bond identifies a physical image, whereas a symbolic spin operator only identifies a finite-reference site.
# Require a declared image-resolved decomposition and certify its sum before constructing any open cluster.
function _structural_bond_image(reference::Supercell, bond::Bond; tolerance::Real=1e-8)
    1 <= bond.i <= nsites(reference) && 1 <= bond.j <= nsites(reference) ||
        throw(ArgumentError("structural bond indices must refer to sites in the source reference"))
    length(bond.displacement) == lattice_dimension(reference.crystal.lattice) || throw(DimensionMismatch("bond displacement dimension disagrees with the lattice"))
    isfinite(bond.distance) && bond.distance > 0 && all(isfinite, bond.displacement) ||
        throw(ArgumentError("structural bonds require finite, nonzero physical displacements"))
    isapprox(norm(bond.displacement), bond.distance; atol=tolerance, rtol=tolerance) ||
        throw(ArgumentError("structural bond distance disagrees with its displacement"))
    difference = bond.displacement .- (reference.sites[bond.j].cartesian .- reference.sites[bond.i].cartesian)
    fractional = reference.crystal.lattice.direct \ difference
    winding = fractional ./ reference.repetitions
    all(isfinite, winding) || throw(ArgumentError("nonfinite periodic-image winding"))
    shifts = Tuple(round(Int, value) for value in winding)
    all(d -> isapprox(winding[d], shifts[d]; atol=tolerance, rtol=tolerance), eachindex(winding)) ||
        throw(ArgumentError("bond displacement is not a lattice image of the stated site pair"))
    all(d -> reference.periodic[d] || iszero(shifts[d]), eachindex(shifts)) ||
        throw(ArgumentError("bond displacement crosses a nonperiodic reference direction"))
    return shifts
end

function _structural_bond_decomposition(model::ManyBodyModel, reference::Supercell, bonds, bond_terms; tolerance::Real=1e-9)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("structural bond tolerance must be positive and finite"))
    algebra = model.representation.algebra
    declared = collect(bonds)
    images = Tuple[]
    for bond in declared
        bond isa Bond || throw(ArgumentError("structural NLCE bonds must be physical Bond objects"))
        bond.i < bond.j || throw(ArgumentError("periodic self-image or reversed bonds cannot certify induced open-cluster couplings"))
        push!(images, (bond.i, bond.j, _structural_bond_image(reference, bond)))
    end
    length(unique(images)) == length(images) || throw(ArgumentError("reference contains duplicate physical bond images"))
    entries = isnothing(bond_terms) ? Pair[] : collect(bond_terms)
    contributions = NamedTuple[]
    seen = Set{Any}()
    for entry in entries
        entry isa Pair && first(entry) isa Bond && last(entry) isa AbstractOperatorExpr ||
            throw(ArgumentError("bond_terms must contain Bond => spin-operator-expression pairs"))
        bond, expression = first(entry), last(entry)
        image = (bond.i, bond.j, _structural_bond_image(reference, bond))
        image in images || throw(ArgumentError("a bond term does not match the declared physical reference bonds"))
        image in seen && throw(ArgumentError("combine multiple channels on one physical bond into a single bond term"))
        push!(seen, image)
        coefficients = PhundamentalCore._numeric_linear_terms(algebra, expression)
        isnothing(coefficients) && throw(ArgumentError("bond contributions require numeric canonical spin coefficients"))
        isempty(first(coefficients)) && throw(ArgumentError("a declared bond contribution must be nonzero"))
        for (key, coefficient) in first(coefficients)
            supports = PhundamentalCore._structural_supports(last(coefficients)[key])
            all(support -> support.family === :spin, supports) || throw(ArgumentError("bond terms must contain only spin operators"))
            Set(support.index for support in supports) == Set((bond.i, bond.j)) ||
                throw(ArgumentError("each bond contribution must act on exactly its two declared endpoints"))
            isfinite(real(coefficient)) && isfinite(imag(coefficient)) || throw(ArgumentError("bond coefficients must be finite"))
        end
        identity_mapping = Dict(i => i for i in 1:nsites(reference))
        try
            _induced_spin_expression(expression, identity_mapping)
        catch error
            throw(ArgumentError("bond expression cannot be mapped by the supported spin-cluster algebra: $(sprint(showerror, error))"))
        end
        push!(contributions, (image=image, operator=expression))
    end
    bond_sum = isempty(contributions) ? 0 * IdentityOperator() : reduce(+, (item.operator for item in contributions))
    difference = PhundamentalCore._numeric_linear_terms(algebra, model.hamiltonian - bond_sum)
    source = PhundamentalCore._numeric_linear_terms(algebra, model.hamiltonian)
    isnothing(difference) && throw(ArgumentError("structural residual contains nonnumeric coefficients"))
    isnothing(source) && throw(ArgumentError("source Hamiltonian has nonnumeric coefficients"))
    scale = maximum(abs, values(first(source)); init=1.0)
    onsite = AbstractOperatorExpr[]
    for (key, coefficient) in first(difference)
        supports = PhundamentalCore._structural_supports(last(difference)[key])
        isfinite(real(coefficient)) && isfinite(imag(coefficient)) || throw(ArgumentError("structural residual contains nonfinite coefficients"))
        isempty(supports) && abs(coefficient) > tolerance * scale &&
            throw(ArgumentError("an identity energy offset has no certified extensive cluster assignment"))
        if length(supports) > 1 && abs(coefficient) > tolerance * scale
            throw(ArgumentError("the declared physical bond contributions do not reconstruct the source Hamiltonian; periodic-image couplings cannot be inferred from folded operator indices"))
        end
        length(supports) == 1 && push!(onsite, coefficient * last(difference)[key])
    end
    open_bonds = Bond[]
    for (bond, image) in zip(declared, images)
        all(iszero, last(image)) && push!(open_bonds, bond)
    end
    return (; onsite, contributions, open_bonds)
end

function _induced_spin_model(model::ManyBodyModel, reference::Supercell, cluster::Supercell, decomposition)
    all(periodic -> !periodic, cluster.periodic) || throw(ArgumentError("an induced structural cluster must have open boundaries"))
    cluster.crystal === reference.crystal || throw(ArgumentError("the induced cluster must retain its source reference crystal"))
    original = Dict((Tuple(site.cell), site.basis_index) => i for (i, site) in enumerate(reference.sites))
    length(original) == nsites(reference) || throw(ArgumentError("reference contains nonunique cell/basis site identities"))
    mapping = Dict{Int,Int}()
    for (new, site) in enumerate(cluster.sites)
        old = get(original, (Tuple(site.cell), site.basis_index), nothing)
        isnothing(old) && throw(ArgumentError("subcluster contains a site absent from its reference"))
        reference.sites[old].species == site.species || throw(ArgumentError("subcluster site species differ from reference"))
        isapprox(reference.sites[old].cartesian, site.cartesian; atol=1e-10, rtol=1e-10) ||
            throw(ArgumentError("subcluster site positions differ from reference"))
        mapping[old] = new
    end
    length(mapping) == nsites(cluster) || throw(ArgumentError("subcluster contains duplicate parent sites"))
    terms = AbstractOperatorExpr[]
    for term in decomposition.onsite
        mapped = _induced_spin_expression(term, mapping)
        isnothing(mapped) || push!(terms, mapped)
    end
    for item in decomposition.contributions
        i, j, winding = item.image
        all(iszero, winding) || continue
        haskey(mapping, i) && haskey(mapping, j) || continue
        push!(terms, _induced_spin_expression(item.operator, mapping))
    end
    expression = isempty(terms) ? 0 * IdentityOperator() : reduce(+, terms)
    material = get(model.parameters, :material, nothing)
    material isa Material || throw(ArgumentError("subcluster generation requires source material"))
    spins = Models._spin_values(material, cluster)
    params = Dict{Symbol,Any}(:material => material, :cluster => cluster, :positions => site_positions(cluster),
                              :structural_parent_model => describe(model)["id"], :induced_reference_sites => sort!(collect(keys(mapping))))
    for key in (:hbar, :coupling_convention)
        haskey(model.parameters, key) && (params[key] = model.parameters[key])
    end
    return ManyBodyModel(Models._spin_representation(spins), expression; parameters=params,
                         observables=Models._spin_observables(nsites(cluster)),
                         provenance=[(generator=:induced_structural_spin_cluster, source_selection=Models._structural_provenance(model).selection_id,
                                      reference_sites=params[:induced_reference_sites], bond_resolution=:physical_image_resolved)])
end

"""
    regenerate_induced_spin_cluster(model, reference, cluster; bond_terms=nothing, bonds=nothing)

Regenerate an open structural spin cluster using explicit `Bond => operator` contributions for each physical periodic image. The supplied contributions must reconstruct every nonlocal source term; residual onsite terms are determined canonically. Periodic-image bonds with endpoints outside the open reference are omitted. No coupling is inferred by dividing a periodic supercell coefficient among geometrically distinct bonds. An onsite-only source needs no bond terms.
"""
function regenerate_induced_spin_cluster(model::ManyBodyModel, reference::Supercell, cluster::Supercell; bond_terms=nothing, bonds=nothing)
    provenance = Models._structural_provenance(model)
    isnothing(provenance) && throw(ArgumentError("induced spin-cluster regeneration requires structural-selection provenance"))
    get(model.model_space.specification.metadata, :operator_site_mapping, :custom) === :one_index_per_crystallographic_site ||
        throw(ArgumentError("induced spin clusters require a certified one-index-per-site mapping"))
    reference === get(model.parameters, :cluster, nothing) ||
        throw(ArgumentError("induced spin clusters must use the source model's exact reference Supercell"))
    model.representation.algebra isa SpinAlgebra || throw(ArgumentError("only spin Hamiltonians admit automatic induced-cluster regeneration"))
    !isnothing(bond_terms) && !all(entry -> entry isa Pair && first(entry) isa Bond, bond_terms) &&
        throw(ArgumentError("bond_terms must be Bond => spin-operator-expression pairs"))
    declared = isnothing(bonds) ? (isnothing(bond_terms) ? Bond[] : [first(entry) for entry in bond_terms]) : bonds
    decomposition = _structural_bond_decomposition(model, reference, declared, bond_terms)
    return _induced_spin_model(model, reference, cluster, decomposition)
end

# Periodic source links are physical translations, not edges joining the same two wrapped sites in an open cluster.
# Require each translated coupling to occur on an interior link with the same geometry and operator coefficients.
function _structural_equal_operator(left, right, algebra; tolerance::Real=1e-9)
    difference = PhundamentalCore._numeric_linear_terms(algebra, left - right)
    isnothing(difference) && return false
    return all(abs(value) <= tolerance for value in values(first(difference)))
end

function _structural_periodic_embedding_support(model::ManyBodyModel, reference::Supercell, decomposition, max_order::Int)
    any(reference.periodic) || return nothing
    nbasis = length(reference.crystal.basis)
    algebra = model.representation.algebra
    onsite_per_site = Dict{Int,Vector{AbstractOperatorExpr}}()
    for term in decomposition.onsite
        coefficients = PhundamentalCore._numeric_linear_terms(algebra, term)
        for (key, coefficient) in first(coefficients)
            supports = PhundamentalCore._structural_supports(last(coefficients)[key])
            length(supports) == 1 || throw(ArgumentError("periodic NLCE requires site-local residuals"))
            site = only(supports).index
            push!(get!(onsite_per_site, site, AbstractOperatorExpr[]), coefficient * last(coefficients)[key])
        end
    end
    for basis in 1:nbasis
        sites = [i for (i, site) in enumerate(reference.sites) if site.basis_index == basis]
        first_site = first(sites)
        baseline = get(onsite_per_site, first_site, AbstractOperatorExpr[])
        normalized = isempty(baseline) ? 0 * IdentityOperator() : reduce(+, (_induced_spin_expression(term, Dict(first_site => 1)) for term in baseline))
        for site in sites[2:end]
            terms = get(onsite_per_site, site, AbstractOperatorExpr[])
            candidate = isempty(terms) ? 0 * IdentityOperator() : reduce(+, (_induced_spin_expression(term, Dict(site => 1)) for term in terms))
            _structural_equal_operator(normalized, candidate, algebra) ||
                throw(ArgumentError("periodic structural NLCE requires translation-invariant onsite terms for each crystallographic basis site"))
        end
    end
    declared = decomposition.contributions
    open = Dict{Tuple{Int,Int},Any}()
    for item in declared
        i, j, winding = item.image
        all(iszero, winding) || continue
        key = (i, j)
        haskey(open, key) && throw(ArgumentError("periodic reference has multiple open physical links between the same site pair"))
        open[key] = item
    end
    for item in declared
        i, j, winding = item.image
        source_i = reference.sites[i]
        source_j = reference.sites[j]
        step = [source_j.cell[d] - source_i.cell[d] + winding[d] * reference.repetitions[d] for d in eachindex(reference.repetitions)]
        for d in eachindex(step)
            reference.periodic[d] || continue
            reference.repetitions[d] > (max_order - 1) * abs(step[d]) ||
                throw(ArgumentError("periodic reference is too short to represent every translated cluster through NLCE order $max_order; enlarge the supercell"))
        end
        normalized = _induced_spin_expression(item.operator, Dict(i => 1, j => 2))
        matches = 0
        for point in CartesianIndices(Tuple(reference.repetitions))
            cell_i = [point[d] - 1 for d in eachindex(reference.repetitions)]
            cell_j = cell_i .+ step
            all(d -> 0 <= cell_j[d] < reference.repetitions[d], eachindex(step)) || continue
            target_i = site_index(reference, cell_i, source_i.basis_index)
            target_j = site_index(reference, cell_j, source_j.basis_index)
            key = (min(target_i, target_j), max(target_i, target_j))
            haskey(open, key) ||
                throw(ArgumentError("missing physical interior bond for a translated periodic-image coupling; correct bond_terms or enlarge the reference"))
            candidate = open[key]
            u, v, _ = candidate.image
            mapped = Dict(u => (u == target_i ? 1 : 2), v => (v == target_i ? 1 : 2))
            comparison = _induced_spin_expression(candidate.operator, mapped)
            _structural_equal_operator(normalized, comparison, algebra) ||
                throw(ArgumentError("periodic bond coefficients or operator structure vary between physical translations"))
            matches += 1
        end
        matches > 0 || throw(ArgumentError("physical-image bond has no interior translated realization in this reference"))
    end
    return function(cluster::ThermalCluster)
        factor = 1.0
        for d in eachindex(reference.repetitions)
            reference.periodic[d] || continue
            coordinates = [site.cell[d] for site in cluster.supercell.sites]
            span = maximum(coordinates) - minimum(coordinates)
            span < reference.repetitions[d] || throw(ArgumentError("cluster exceeds periodic embedding reference"))
            factor *= reference.repetitions[d] / (reference.repetitions[d] - span)
        end
        return factor
    end
end

function _structural_nlce_support(model::ManyBodyModel, context::StructuralNLCEForwardContext)
    reference = get(model.parameters, :cluster, nothing)
    reference isa Supercell || throw(ArgumentError("structural NLCE requires a finite reference Supercell"))
    material = _certified_induced_spin_source(model, context.selection, reference)
    any(reference.periodic) && !context.context.allow_periodic_reference &&
        throw(ArgumentError("a periodic structural reference requires allow_periodic_reference=true"))
    bonds = collect(context.context.bonds)
    all(bond -> bond isa Bond && 1 <= bond.i <= nsites(reference) && 1 <= bond.j <= nsites(reference), bonds) ||
        throw(ArgumentError("structural NLCE reference bonds must index the source finite Supercell"))
    decomposition = _structural_bond_decomposition(model, reference, bonds, context.bond_terms)
    factor = _structural_periodic_embedding_support(model, reference, decomposition, context.context.max_order)
    return (material=material, reference=reference, bonds=decomposition.open_bonds, decomposition=decomposition, embedding_factor=factor)
end

function _forward_context_support(::NLCESolver, model::ManyBodyModel, request::Union{Observables.ObservableRequest,ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest},
                                  context::StructuralNLCEForwardContext)
    try
        _structural_nlce_support(model, context)
        inner = solver_compatibility(context.context.inner_solver, model, request, context.context.inner_method_context)
        reasons = inner.compatible ? ("induced-cluster closure is checked at execution",) : inner.reasons
        return inner.compatible, reasons, false
    catch error
        return false, (sprint(showerror, error),), true
    end
end

_context_approximations(context::StructuralNLCEForwardContext) = _context_approximations(context.context)

function _prepare_solver_input(::NLCESolver, model::ManyBodyModel, observable::Union{Observables.ObservableRequest,ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest},
                               args::Tuple, kwargs::NamedTuple, structural::StructuralNLCEForwardContext)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("structural NLCE solver arguments must be configured in the inner context"))
    source = _structural_nlce_support(model, structural)
    context = structural.context
    evaluator = function(cluster_model, cluster, index)
        metadata = Dict{Symbol,Any}(:nlce_cluster_key => cluster.key, :nlce_cluster_order => cluster.order, :nlce_cluster_index => index)
        calculation = ForwardProblem(cluster_model; transformation=context.inner_transformation, solver=context.inner_solver,
                                     solver_args=context.inner_solver_args, solver_kwargs=context.inner_solver_kwargs,
                                     method_context=context.inner_method_context, observable=observable, metadata=metadata)
        return _nlce_additive_observable(forward(calculation).observable_result)
    end
    builder = (material, cluster, spec, bonds) -> _induced_spin_model(model, source.reference, cluster, source.decomposition)
    spec = InducedStructuralSpinSpec(structural_id(structural.selection.family), structural_id(structural.selection))
    problem = NLCEProblem(source.material, source.reference, spec, evaluator; max_order=context.max_order,
                          bonds=source.bonds, min_order=context.min_order, max_clusters=context.max_clusters,
                          model_builder=builder, allow_periodic_reference=context.allow_periodic_reference, embedding_factor=source.embedding_factor)
    return _NLCEProblemSolverInput(problem, request_requirements(observable), context)
end
