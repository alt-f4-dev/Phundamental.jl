# Explicit, certified open-cluster reconstruction for fermionic linked-cluster calculations.
# Fermionic primitive ordering is retained; canonicalization handles anticommutation signs.

"""Opaque specification for induced fermionic NLCE clusters; the physical source remains the parent model."""
struct InducedFermionicClusterSpec <: AbstractHamiltonianSpec end

"""
    FermionicClusterDecomposition

A certified onsite/bond decomposition of a finite-reference fermionic Hamiltonian.
`site_modes[i]` gives the source fermion modes belonging to physical site `i`, in
local component order. Each physical bond contribution is supplied explicitly.
Only open finite references are supported; periodic-image decomposition is not inferred.
"""
struct FermionicClusterDecomposition
    site_modes::Vector{Vector{Int}}
    onsite::Vector{Vector{AbstractOperatorExpr}}
    bond_terms::Vector{Pair}
    bonds::Vector{Bond}
    source::ManyBodyModel
    reference::Supercell
end

_fermion_nlce_primitive(op::FermionCreate, map::Dict{Int,Int}) = haskey(map, op.mode) ? c(map[op.mode])' : nothing
_fermion_nlce_primitive(op::FermionAnnihilate, map::Dict{Int,Int}) = haskey(map, op.mode) ? c(map[op.mode]) : nothing
_fermion_nlce_primitive(op::NumberOperator, map::Dict{Int,Int}) = op.kind === :fermion && haskey(map, op.mode) ? nf(map[op.mode]) : nothing
_fermion_nlce_primitive(::IdentityOperator, map::Dict{Int,Int}) = IdentityOperator()
_fermion_nlce_primitive(op::AbstractPrimitiveOperator, map::Dict{Int,Int}) = throw(ArgumentError("unsupported fermionic cluster operator $(typeof(op))"))

function _fermion_nlce_map(expr::AbstractPrimitiveOperator, map::Dict{Int,Int})
    return _fermion_nlce_primitive(expr, map)
end

function _fermion_nlce_map(expr::ScaledOperator, map::Dict{Int,Int})
    item = _fermion_nlce_map(expr.operator, map)
    return isnothing(item) ? nothing : expr.coefficient * item
end

function _fermion_nlce_map(expr::OperatorProduct, map::Dict{Int,Int})
    result = AbstractOperatorExpr[]
    for factor in expr.factors
        mapped = _fermion_nlce_map(factor, map)
        isnothing(mapped) && return nothing
        push!(result, mapped)
    end
    return reduce(*, result)
end

function _fermion_nlce_map(expr::OperatorSum, map::Dict{Int,Int})
    result = AbstractOperatorExpr[]
    for term in expr.terms
        mapped = _fermion_nlce_map(term, map)
        isnothing(mapped) || push!(result, mapped)
    end
    return isempty(result) ? 0 * IdentityOperator() : reduce(+, result)
end

function _fermion_nlce_coefficients(expr::AbstractOperatorExpr, algebra::FermionAlgebra)
    values = PhundamentalCore._numeric_linear_terms(algebra, expr)
    isnothing(values) && throw(ArgumentError("fermionic cluster decomposition requires numeric canonical coefficients"))
    return values
end

function _fermion_nlce_support(expr::AbstractOperatorExpr, algebra::FermionAlgebra, mode_to_site::Dict{Int,Int})
    coefficients, operators = _fermion_nlce_coefficients(expr, algebra)
    sites = Set{Int}()
    for (key, value) in coefficients
        isfinite(real(value)) && isfinite(imag(value)) || throw(ArgumentError("nonfinite fermionic cluster coefficient"))
        supports = PhundamentalCore._structural_supports(operators[key])
        isempty(supports) && throw(ArgumentError("a nonzero identity offset has no certified extensive cluster assignment"))
        for support in supports
            support.family === :fermion || throw(ArgumentError("only fermionic primitive operators are supported"))
            haskey(mode_to_site, support.index) || throw(ArgumentError("operator references a mode absent from site_modes"))
            push!(sites, mode_to_site[support.index])
        end
    end
    return sites
end

"""
    certify_fermionic_clusters(model, reference; site_modes, onsite_terms, bond_terms, bonds, tolerance=1e-9)

Verify an explicit physical-site mapping and decomposition of a source symbolic
Hamiltonian into site-local terms and image-resolved two-site terms. All terms
must reproduce the full numeric canonical source exactly within `tolerance`.
Periodic references and ambiguous folded couplings are rejected.
"""
function certify_fermionic_clusters(model::ManyBodyModel, reference::Supercell;
                                    site_modes, onsite_terms, bond_terms, bonds=nothing, tolerance::Real=1e-9)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be positive and finite"))
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("fermionic cluster reconstruction requires FermionAlgebra"))
    model.representation.state_space isa FermionFockSpace || throw(ArgumentError("fermionic clusters require an unprojected fermion Fock space"))
    model.representation.basis isa FermionOccupationBasis || throw(ArgumentError("fermionic clusters require an occupation basis"))
    isnothing(model.representation.physical_subspace) && isnothing(model.representation.truncation) &&
        isempty(model.representation.constraints) && isnothing(model.representation.gauge_structure) ||
        throw(ArgumentError("constrained, gauged or truncated source representations require a separately certified cluster mapping"))
    all(!, reference.periodic) || throw(ArgumentError("fermionic NLCE currently requires an open finite reference; periodic image-resolved embedding is not yet certified"))
    reference === get(model.parameters, :cluster, nothing) || throw(ArgumentError("reference must be the exact Supercell used to build the source model"))
    nsites(reference) == length(site_modes) || throw(DimensionMismatch("site_modes length does not match the reference"))
    get(model.parameters, :material, nothing) isa Material ||
        throw(ArgumentError("source model must retain Material provenance"))
    nmodes = model.representation.algebra.nmodes
    local_modes = [Int.(collect(group)) for group in site_modes]
    all(!isempty, local_modes) || throw(ArgumentError("each physical site must own at least one fermionic mode"))
    all_modes = reduce(vcat, local_modes)
    sort(all_modes) == collect(1:nmodes) || throw(ArgumentError("site_modes must partition the complete fermionic mode set exactly once"))
    mode_to_site = Dict(mode => i for (i, group) in enumerate(local_modes) for mode in group)
    if haskey(model.parameters, :fermion_mode_map)
        for ((site, _), mode) in model.parameters[:fermion_mode_map]
            mode_to_site[mode] == site || throw(ArgumentError("site_modes disagree with the source fermion_mode_map"))
        end
    end
    source_basis = model.representation.basis.ordering
    source_basis == collect(1:nmodes) || throw(ArgumentError("source occupation basis must use canonical mode ordering"))
    isnothing(model.representation.ordering) || model.representation.ordering == collect(1:nmodes) ||
        throw(ArgumentError("noncanonical source ordering requires an explicit fermionic parity transformation"))
    model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("source requires a symbolic Hamiltonian"))

    onsite = [AbstractOperatorExpr[] for _ in 1:nsites(reference)]
    for (site, expression) in pairs(onsite_terms)
        site isa Integer && 1 <= site <= length(onsite) || throw(ArgumentError("onsite terms must use valid integer physical-site keys"))
        expression isa AbstractOperatorExpr || throw(ArgumentError("onsite terms must be operator expressions"))
        support = _fermion_nlce_support(expression, model.representation.algebra, mode_to_site)
        all(==(site), support) || throw(ArgumentError("onsite operator acts outside its declared physical site"))
        push!(onsite[site], expression)
    end

    declared = isnothing(bonds) ? [first(entry) for entry in bond_terms] : collect(bonds)
    all(bond -> bond isa Bond && 1 <= bond.i < bond.j <= nsites(reference), declared) ||
        throw(ArgumentError("declared connectivity must contain physical Bond objects with i < j"))
    images = [_structural_bond_image(reference, bond) for bond in declared]
    all(winding -> all(iszero, winding), images) || throw(ArgumentError("periodic-image bonds cannot be used in an open fermionic NLCE reference"))
    keys = [(bond.i, bond.j) for bond in declared]
    length(unique(keys)) == length(keys) || throw(ArgumentError("duplicate physical reference bonds"))
    indexed = Set(keys)
    contributions = Pair[]
    seen = Set{Tuple{Int,Int}}()
    for entry in bond_terms
        entry isa Pair && first(entry) isa Bond && last(entry) isa AbstractOperatorExpr ||
            throw(ArgumentError("bond_terms must contain Bond => fermion-operator pairs"))
        bond, expression = first(entry), last(entry)
        key = (bond.i, bond.j)
        key in indexed || throw(ArgumentError("bond term is absent from declared connectivity"))
        all(iszero, _structural_bond_image(reference, bond)) || throw(ArgumentError("bond term crosses the open reference boundary"))
        key in seen && throw(ArgumentError("duplicate physical bond contribution"))
        push!(seen, key)
        support = _fermion_nlce_support(expression, model.representation.algebra, mode_to_site)
        support == Set(key) || throw(ArgumentError("bond contribution must couple exactly its declared pair of sites"))
        push!(contributions, bond => expression)
    end
    all_terms = AbstractOperatorExpr[]
    foreach(terms -> append!(all_terms, terms), onsite)
    append!(all_terms, last.(contributions))
    reconstructed = isempty(all_terms) ? 0 * IdentityOperator() : reduce(+, all_terms)
    algebra = model.representation.algebra
    difference, _ = _fermion_nlce_coefficients(model.hamiltonian - reconstructed, algebra)
    source, _ = _fermion_nlce_coefficients(model.hamiltonian, algebra)
    scale = maximum(abs, values(source); init=1.0)
    maximum(abs, values(difference); init=0.0) <= tolerance * scale ||
        throw(ArgumentError("onsite and physical-bond terms do not reconstruct the source Hamiltonian"))
    return FermionicClusterDecomposition(local_modes, onsite, contributions, Bond[declared...], model, reference)
end

"""Rebuild a certified open fermion subcluster without approximating its algebra or changing local mode order."""
function regenerate_induced_fermion_cluster(decomposition::FermionicClusterDecomposition, cluster::Supercell)
    reference = decomposition.reference
    all(!, cluster.periodic) || throw(ArgumentError("induced fermion cluster must have open boundaries"))
    cluster.crystal === reference.crystal || throw(ArgumentError("induced cluster must preserve the source crystal"))
    original = Dict((Tuple(site.cell), site.basis_index) => i for (i, site) in enumerate(reference.sites))
    length(original) == nsites(reference) || throw(ArgumentError("source reference contains repeated physical site identities"))
    site_map = Dict{Int,Int}()
    for (j, site) in enumerate(cluster.sites)
        i = get(original, (Tuple(site.cell), site.basis_index), nothing)
        isnothing(i) && throw(ArgumentError("cluster site is absent from source reference"))
        reference.sites[i].species == site.species || throw(ArgumentError("cluster species mismatch"))
        isapprox(reference.sites[i].cartesian, site.cartesian; atol=1e-10, rtol=1e-10) ||
            throw(ArgumentError("cluster coordinate mismatch"))
        haskey(site_map, i) && throw(ArgumentError("cluster repeats a physical site"))
        site_map[i] = j
    end
    length(site_map) == nsites(cluster) || throw(ArgumentError("cluster contains repeated source sites"))
    map = Dict{Int,Int}()
    next_mode = 1
    for site in cluster.sites
        i = original[(Tuple(site.cell), site.basis_index)]
        for mode in decomposition.site_modes[i]
            map[mode] = next_mode
            next_mode += 1
        end
    end
    terms = AbstractOperatorExpr[]
    for i in sort!(collect(keys(site_map))), term in decomposition.onsite[i]
        push!(terms, _fermion_nlce_map(term, map))
    end
    for (bond, expression) in decomposition.bond_terms
        haskey(site_map, bond.i) && haskey(site_map, bond.j) || continue
        push!(terms, _fermion_nlce_map(expression, map))
    end
    expression = isempty(terms) ? 0 * IdentityOperator() : reduce(+, terms)
    source = decomposition.source
    nmodes = next_mode - 1
    representation = Representation(:induced_fermion_cluster, FermionFockSpace(nmodes), FermionAlgebra(nmodes),
                                    FermionOccupationBasis(collect(1:nmodes)); ordering=collect(1:nmodes),
                                    reference_state=:fermion_vacuum)
    observables = Dict{Symbol,Any}()
    for (key, value) in source.observables
        value isa AbstractOperatorExpr || continue
        supports = PhundamentalCore._structural_supports(value)
        all(support -> support.family === :fermion && haskey(map, support.index), supports) || continue
        mapped = _fermion_nlce_map(value, map)
        isnothing(mapped) || (observables[key] = mapped)
    end
    params = Dict{Symbol,Any}(:material => source.parameters[:material], :cluster => cluster,
                              :positions => site_positions(cluster), :parent_mode_map => copy(map),
                              :parent_sites => sort!(collect(keys(site_map))))
    for key in (:hbar, :coupling_convention)
        haskey(source.parameters, key) && (params[key] = source.parameters[key])
    end
    return ManyBodyModel(representation, expression; parameters=params, observables=observables,
                         provenance=[(construction=:certified_induced_fermion_cluster, parent_sites=params[:parent_sites],
                                      mode_ordering=:local_components, boundary=:open)])
end

"""
    fermionic_nlce_problem(decomposition, evaluator; max_order, min_order=1, max_clusters=100000, max_fock_dimension=4096)

Use the existing `NLCEProblem` engine with certified fermionic source-to-cluster
regeneration and a user-supplied additive cluster evaluator. The existing
`NLCESolver` performs enumeration, inclusion-exclusion and order sums.
"""
function fermionic_nlce_problem(decomposition::FermionicClusterDecomposition, evaluator; max_order::Integer,
                                min_order::Integer=1, max_clusters::Integer=100_000, max_fock_dimension::Integer=4096)
    source = decomposition.source
    material = source.parameters[:material]
    material isa Material || throw(ArgumentError("source requires physical Material provenance"))
    max_fock_dimension > 0 || throw(ArgumentError("max_fock_dimension must be positive"))
    builder = function(material, cluster, spec, bonds)
        model = regenerate_induced_fermion_cluster(decomposition, cluster)
        nmodes = model.representation.algebra.nmodes
        ndigits(max_fock_dimension; base=2) > nmodes ||
            throw(ArgumentError("cluster has $nmodes fermion modes; dimension 2^$nmodes exceeds max_fock_dimension=$max_fock_dimension"))
        return model
    end
    return NLCEProblem(material, decomposition.reference, InducedFermionicClusterSpec(), evaluator;
                       max_order=max_order, min_order=min_order, max_clusters=max_clusters,
                       bonds=decomposition.bonds, model_builder=builder)
end

"""A typed forward context for certified open-cluster fermionic NLCE."""
struct FermionicNLCEForwardContext{D,C} <: AbstractForwardMethodContext
    decomposition::D
    context::C
    max_fock_dimension::Int
end

function FermionicNLCEForwardContext(decomposition::FermionicClusterDecomposition, max_order::Integer;
                                     max_fock_dimension::Integer=4096, kwargs...)
    max_fock_dimension > 0 || throw(ArgumentError("max_fock_dimension must be positive"))
    context = NLCEForwardContext(max_order; kwargs...)
    return FermionicNLCEForwardContext(decomposition, context, Int(max_fock_dimension))
end

_context_approximations(context::FermionicNLCEForwardContext) = _context_approximations(context.context)

function _forward_context_support(::NLCESolver, model::ManyBodyModel,
                                  request::Union{Observables.ObservableRequest,ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest},
                                  context::FermionicNLCEForwardContext)
    model === context.decomposition.source || return false, ("source model differs from certified fermionic decomposition",), true
    inner = solver_compatibility(context.context.inner_solver, model, request, context.context.inner_method_context)
    return inner.compatible, inner.compatible ? ("finite-cluster reconstruction is checked at execution",) : inner.reasons, false
end

function _prepare_solver_input(::NLCESolver, model::ManyBodyModel,
                               request::Union{Observables.ObservableRequest,ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest},
                               args::Tuple, kwargs::NamedTuple, context::FermionicNLCEForwardContext)
    isempty(args) && isempty(kwargs) || throw(ArgumentError("NLCE solver arguments belong in the method context"))
    model === context.decomposition.source || throw(ArgumentError("source model differs from certified fermionic decomposition"))
    inner = context.context
    evaluator = function(cluster_model, cluster, index)
        metadata = Dict{Symbol,Any}(:nlce_cluster_key => cluster.key, :nlce_cluster_order => cluster.order,
                                    :nlce_cluster_index => index)
        calculation = ForwardProblem(cluster_model; transformation=inner.inner_transformation, solver=inner.inner_solver,
                                     solver_args=inner.inner_solver_args, solver_kwargs=inner.inner_solver_kwargs,
                                     method_context=inner.inner_method_context, observable=request, metadata=metadata)
        return _nlce_additive_observable(forward(calculation).observable_result)
    end
    problem = fermionic_nlce_problem(context.decomposition, evaluator;
                                    max_order=inner.max_order, min_order=inner.min_order, max_clusters=inner.max_clusters,
                                    max_fock_dimension=context.max_fock_dimension)
    return _NLCEProblemSolverInput(problem, request_requirements(request), inner)
end
