# Material- and crystallography-aware universal Hamiltonian-family generation.
# This layer coordinates existing geometry, crystallography, admissibility,
# canonical structure, and parameter-schema machinery without replacing them.

"""
    HamiltonianGenerationResult

Material-aware result of Batch 10 Hamiltonian-family generation. The object retains the crystallographic source, physical `GenerativeSpecification`, generated `HamiltonianSpace`, canonical `ModelStructure`, and constraint-bearing `ParameterSpace` as separate inspectable stages.
"""
struct HamiltonianGenerationResult{M,C,X,D,G,H,S,P,MD}
    material::M
    cluster::C
    crystal_structure::X
    dataset::D
    specification::G
    hamiltonian_space::H
    model_structure::S
    parameter_space::P
    metadata::MD
end

function _material_species_encoding(symbols)
    ids = Dict{Symbol,Int}()
    encoded = Int[]
    for symbol in symbols
        key = symbol
        id = get!(ids, key) do
            length(ids) + 1
        end
        push!(encoded, id)
    end
    return encoded, ids
end

function _require_three_dimensional_crystal(crystal::CrystalStructure)
    lattice_dimension(crystal.lattice) == 3 || throw(ArgumentError(
        "automatic crystallographic generation requires a three-dimensional lattice because the current Spglib backend is three-dimensional",
    ))
    return crystal
end

"""
    material_crystallographic_structure(material)
    material_crystallographic_structure(material, cluster)

Convert model-layer material geometry into the backend-neutral `Crystallography.CrystalStructure` used for symmetry discovery and invariant generation. The cluster overload requires a fully periodic three-dimensional `Supercell`; its lattice and coordinates are expressed in the supercell basis so crystallographic operations act on the same finite site indices as the generated operator algebra.
"""
function material_crystallographic_structure(material::Material)
    crystal = _require_three_dimensional_crystal(material.crystal)
    symbols = Symbol[site.species for site in crystal.basis]
    species, _ = _material_species_encoding(symbols)
    positions = hcat((Float64.(site.fractional) for site in crystal.basis)...)
    labels = Symbol[site.label for site in crystal.basis]
    return Crystallography.CrystalStructure(direct_matrix(crystal.lattice), positions, species; labels=labels)
end

function material_crystallographic_structure(material::Material, cluster::Supercell)
    _validate_material_cluster(material, cluster)
    _require_three_dimensional_crystal(cluster.crystal)
    all(cluster.periodic) || throw(ArgumentError(
        "crystallographic generation on an explicit Supercell requires periodic boundary conditions in every spatial direction",
    ))

    repetitions = Float64.(cluster.repetitions)
    lattice = direct_matrix(cluster.crystal.lattice) * Diagonal(cluster.repetitions)
    positions = hcat((Float64.(site.fractional) ./ repetitions for site in cluster.sites)...)
    symbols = Symbol[site.species for site in cluster.sites]
    species, _ = _material_species_encoding(symbols)
    labels = Symbol[Symbol(string(site.label), "_", string(site.index)) for site in cluster.sites]
    return Crystallography.CrystalStructure(lattice, positions, species; labels=labels)
end

function _site_local_space(material::Material, ::Nothing, algebra::SpinAlgebra, site::Integer)
    algebra.nsites == length(material.crystal.basis) || throw(ArgumentError(
        "automatic spin local-space construction requires SpinAlgebra.nsites to equal the material basis-site count",
    ))
    spin = material.basis_properties[Int(site)].spin
    isnothing(spin) && throw(ArgumentError("material basis site $(site) has no spin quantum number; supply local_spaces explicitly"))
    return SpinHilbertSpace([spin])
end

function _site_local_space(material::Material, cluster::Supercell, algebra::SpinAlgebra, site::Integer)
    algebra.nsites == nsites(cluster) || throw(ArgumentError(
        "automatic spin local-space construction requires SpinAlgebra.nsites to equal the cluster site count",
    ))
    spin = site_spin(material, cluster, site)
    isnothing(spin) && throw(ArgumentError("cluster site $(site) has no spin quantum number; supply local_spaces explicitly"))
    return SpinHilbertSpace([spin])
end

_site_local_space(::Material, ::Union{Nothing,Supercell}, ::FermionAlgebra, ::Integer) = FermionFockSpace(1)
_site_local_space(::Material, ::Union{Nothing,Supercell}, ::BosonAlgebra, ::Integer) = BosonFockSpace(1)

function _site_local_space(material::Material, cluster::Union{Nothing,Supercell}, algebra::CompositeAlgebra, site::Integer)
    factors = map(factor -> _site_local_space(material, cluster, factor, site), algebra.factors)
    return CompositeStateSpace(tuple(factors...))
end

function _site_local_space(::Material, ::Union{Nothing,Supercell}, ::MajoranaAlgebra, ::Integer)
    throw(ArgumentError("automatic local-space construction for crystallographic Majorana indices is ambiguous; supply local_spaces explicitly"))
end

function _site_local_space(::Material, ::Union{Nothing,Supercell}, ::CoordinateMomentumAlgebra, ::Integer)
    throw(ArgumentError("automatic local-space construction for coordinate/momentum component indices is ambiguous; supply local_spaces explicitly"))
end

function _default_local_spaces(material::Material, cluster, algebra::AbstractOperatorAlgebra, site_count::Integer)
    return Tuple(_site_local_space(material, cluster, algebra, site) for site in 1:Int(site_count))
end

function _standard_site_index_count(algebra::SpinAlgebra)
    return algebra.nsites
end

function _standard_site_index_count(algebra::BosonAlgebra)
    return algebra.nmodes
end

function _standard_site_index_count(algebra::FermionAlgebra)
    return algebra.nmodes
end

function _standard_site_index_count(algebra::MajoranaAlgebra)
    return algebra.ngenerators
end

_standard_site_index_count(::CoordinateMomentumAlgebra) = nothing

function _validate_standard_site_indexing(algebra::AbstractOperatorAlgebra, site_count::Integer)
    count = _standard_site_index_count(algebra)
    isnothing(count) && throw(ArgumentError(
        "the standard crystallographic symmetry action cannot infer coordinate/momentum vector components; supply a custom symmetry_action",
    ))
    count == site_count || throw(ArgumentError(
        "the standard crystallographic symmetry action requires one operator index per crystallographic site; algebra exposes $count indices for $site_count sites",
    ))
    return true
end

function _validate_standard_site_indexing(algebra::CompositeAlgebra, site_count::Integer)
    for factor in algebra.factors
        _validate_standard_site_indexing(factor, site_count)
    end
    return true
end

_local_space_signature(space::SpinHilbertSpace) = (:spin, Tuple(space.spins))
_local_space_signature(space::FermionFockSpace) = (:fermion, space.nmodes)
_local_space_signature(space::BosonFockSpace) = (:boson, space.nmodes)
_local_space_signature(space::CoordinateHilbertSpace) = (:coordinate, space.ndof)
_local_space_signature(space::CompositeStateSpace) = (:composite, Tuple(_local_space_signature(factor) for factor in space.factors))
_local_space_signature(space::AbstractStateSpace) = (nameof(typeof(space)), repr(space))

function _validate_standard_local_space_symmetry(
    local_spaces,
    crystal::Crystallography.CrystalStructure,
    dataset::Crystallography.CrystallographicDataset,
    options::Crystallography.CrystallographyOptions,
)
    length(local_spaces) == length(crystal) || throw(DimensionMismatch(
        "standard crystallographic symmetry requires one local state space per crystallographic site",
    ))
    signatures = Tuple(_local_space_signature(space) for space in local_spaces)
    for operation in dataset.operations
        permutation = Crystallography.site_permutation(crystal, operation; atol=options.site_tolerance)
        for site in eachindex(permutation)
            signatures[site] == signatures[permutation[site]] || throw(ArgumentError(
                "crystallographic symmetry maps site $(site) onto site $(permutation[site]) with an incompatible local state space; refine the crystallographic site classes or supply a custom symmetry_action/local_spaces convention",
            ))
        end
    end
    return true
end

# The nearest fractional-component image need not be the nearest Cartesian image for a skew lattice.
# The smallest singular value gives a rigorous finite enumeration radius for the exact periodic nearest-image problem.
function _minimum_periodic_cartesian_distance(lattice::AbstractMatrix{<:Real}, fractional_difference::AbstractVector{<:Real}; max_images::Integer=2_000_000)
    size(lattice) == (3, 3) && length(fractional_difference) == 3 ||
        throw(DimensionMismatch("periodic Cartesian distances require a 3x3 lattice and three fractional coordinates"))
    A = Matrix{Float64}(lattice)
    delta = Float64.(fractional_difference)
    sigma_min = minimum(svdvals(A))
    isfinite(sigma_min) && sigma_min > 0 || throw(ArgumentError("crystallographic lattice must be nonsingular"))
    nearest_component_image = round.(Int, delta)
    best_squared = sum(abs2, A * (delta .- nearest_component_image))
    # A cheap local stencil tightens the rigorous search radius before exhaustive enumeration.
    for offset in Iterators.product(-1:1, -1:1, -1:1)
        shift = nearest_component_image .+ collect(offset)
        best_squared = min(best_squared, sum(abs2, A * (delta .- shift)))
    end
    radius = sqrt(best_squared) / sigma_min
    isfinite(radius) || throw(ArgumentError("minimum-image search radius is not finite for the supplied lattice"))
    limits = ntuple(i -> (floor(Int, delta[i] - radius) - 1):(ceil(Int, delta[i] + radius) + 1), 3)
    count = prod(Int128(length(bound)) for bound in limits)
    count <= max_images || throw(ArgumentError("exact minimum-image distance requires $count periodic images, exceeding max_images=$max_images"))
    for offset in Iterators.product(limits...)
        candidate = delta .- collect(offset)
        best_squared = min(best_squared, sum(abs2, A * candidate))
    end
    return sqrt(best_squared)
end

function _term_geometric_range(term::StructuralTerm, crystal::Crystallography.CrystalStructure)
    indices = unique(Int[support.index for support in term.supports])
    isempty(indices) && return 0.0
    all(index -> 1 <= index <= length(crystal), indices) || throw(ArgumentError(
        "operator support contains an index that cannot be mapped to the crystallographic structure",
    ))
    length(indices) == 1 && return 0.0

    diameter = 0.0
    for a in 1:(length(indices) - 1), b in (a + 1):length(indices)
        difference = crystal.positions[:, indices[a]] .- crystal.positions[:, indices[b]]
        diameter = max(diameter, _minimum_periodic_cartesian_distance(crystal.lattice, difference))
    end
    return diameter
end

function _crystallographic_range_evaluator(crystal::Crystallography.CrystalStructure, algebra::AbstractOperatorAlgebra)
    return function (expression::AbstractOperatorExpr)
        model_structure = structure(expression, algebra)
        isempty(model_structure.terms) && return 0.0
        return maximum(_term_geometric_range(term, crystal) for term in model_structure.terms)
    end
end

function _crystallographic_orbits(dataset::Crystallography.CrystallographicDataset, crystal::Crystallography.CrystalStructure)
    isempty(dataset.crystallographic_orbits) && return Tuple(crystal.labels)
    length(dataset.crystallographic_orbits) == length(crystal) || throw(DimensionMismatch(
        "crystallographic orbit labels must contain one entry per crystal site",
    ))
    return Tuple(dataset.crystallographic_orbits)
end

function _material_specification(
    material::Material,
    cluster,
    algebra::AbstractOperatorAlgebra;
    local_spaces=nothing,
    dataset=nothing,
    symmetry_action=Crystallography.standard_symmetry_action,
    admissibility_conditions=(),
    max_range=nothing,
    max_body_order=nothing,
    range_evaluator=nothing,
    options::Crystallography.CrystallographyOptions=Crystallography.CrystallographyOptions(),
    metadata=Dict{Symbol,Any}(),
)
    crystal = isnothing(cluster) ? material_crystallographic_structure(material) : material_crystallographic_structure(material, cluster)
    site_count = length(crystal)
    symmetry_action === Crystallography.standard_symmetry_action && _validate_standard_site_indexing(algebra, site_count)

    resolved_spaces = if isnothing(local_spaces)
        _validate_standard_site_indexing(algebra, site_count)
        _default_local_spaces(material, cluster, algebra, site_count)
    else
        spaces = local_spaces isa AbstractStateSpace ? (local_spaces,) : Tuple(local_spaces)
        all(space -> space isa AbstractStateSpace, spaces) || throw(ArgumentError("local_spaces entries must subtype AbstractStateSpace"))
        spaces
    end
    resolved_dataset = if isnothing(dataset)
        Crystallography.discover_symmetry(crystal; options=options)
    else
        dataset isa Crystallography.CrystallographicDataset || throw(ArgumentError("dataset must be CrystallographicDataset or nothing"))
        dataset
    end

    if symmetry_action === Crystallography.standard_symmetry_action
        _validate_standard_local_space_symmetry(resolved_spaces, crystal, resolved_dataset, options)
    end

    resolved_range_evaluator = if !isnothing(range_evaluator)
        range_evaluator
    elseif !isnothing(max_range) && symmetry_action === Crystallography.standard_symmetry_action
        _crystallographic_range_evaluator(crystal, algebra)
    else
        nothing
    end

    symbols = isnothing(cluster) ? Symbol[site.species for site in material.crystal.basis] : Symbol[site.species for site in cluster.sites]
    _, species_ids = _material_species_encoding(symbols)
    merged_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata))
    merged_metadata[:crystal_structure] = crystal
    merged_metadata[:crystallography_options] = options
    merged_metadata[:crystallographic_backend] = get(resolved_dataset.metadata, :backend, :provided)
    merged_metadata[:material_name] = material.name
    merged_metadata[:material_species_ids] = species_ids
    merged_metadata[:generation_domain] = isnothing(cluster) ? :primitive_cell : :periodic_supercell
    merged_metadata[:operator_site_mapping] = symmetry_action === Crystallography.standard_symmetry_action ?
        :one_index_per_crystallographic_site : :custom
    if !isnothing(max_range) && isnothing(range_evaluator) && !isnothing(resolved_range_evaluator)
        merged_metadata[:range_evaluator_source] = :crystallographic_geometry
    end

    specification = GenerativeSpecification(
        resolved_dataset,
        _crystallographic_orbits(resolved_dataset, crystal),
        resolved_spaces,
        algebra,
        symmetry_action;
        admissibility_conditions=admissibility_conditions,
        max_range=max_range,
        max_body_order=max_body_order,
        range_evaluator=resolved_range_evaluator,
        metadata=merged_metadata,
    )
    return specification, crystal, resolved_dataset
end

"""
    material_generative_specification(material, algebra; kwargs...)
    material_generative_specification(material, cluster, algebra; kwargs...)

Construct the physical `GenerativeSpecification` used by the universal Hamiltonian generator from material geometry and crystallographic symmetry. The standard symmetry action requires one symbolic site/mode index per crystallographic site. More complicated orbital, vector-component, or internal-index conventions must supply a custom `symmetry_action` explicitly.
"""
function material_generative_specification(material::Material, algebra::AbstractOperatorAlgebra; kwargs...)
    specification, _, _ = _material_specification(material, nothing, algebra; kwargs...)
    return specification
end

function material_generative_specification(material::Material, cluster::Supercell, algebra::AbstractOperatorAlgebra; kwargs...)
    specification, _, _ = _material_specification(material, cluster, algebra; kwargs...)
    return specification
end

function _generated_parameter_space(
    space::HamiltonianSpace;
    parameter_values=nothing,
    parameter_bounds=nothing,
    parameter_units=nothing,
    parameter_fixed=nothing,
    parameter_priors=nothing,
    parameter_constraints=(),
    parameter_metadata=Dict{Symbol,Any}(),
)
    values = isnothing(parameter_values) ? zeros(Float64, length(space)) : collect(parameter_values)
    return ParameterSpace(
        space;
        values=values,
        bounds=parameter_bounds,
        units=parameter_units,
        fixed=parameter_fixed,
        priors=parameter_priors,
        constraints=parameter_constraints,
        metadata=parameter_metadata,
    )
end

function _generation_result(
    specification::GenerativeSpecification,
    crystal::Crystallography.CrystalStructure;
    material=nothing,
    cluster=nothing,
    candidates=nothing,
    max_operator_degree=nothing,
    basis_labels=nothing,
    options=nothing,
    generation_metadata=Dict{Symbol,Any}(),
    parameter_values=nothing,
    parameter_bounds=nothing,
    parameter_units=nothing,
    parameter_fixed=nothing,
    parameter_priors=nothing,
    parameter_constraints=(),
    parameter_metadata=Dict{Symbol,Any}(),
)
    space_metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(generation_metadata))
    space_metadata[:universal_generator] = :batch10_material_crystallographic
    space = Crystallography.generate_crystallographic_hamiltonian_space(
        specification,
        crystal;
        candidates=candidates,
        max_operator_degree=max_operator_degree,
        labels=basis_labels,
        options=options,
        metadata=space_metadata,
    )
    model_structure = structure(space)
    parameter_space = _generated_parameter_space(
        space;
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        parameter_units=parameter_units,
        parameter_fixed=parameter_fixed,
        parameter_priors=parameter_priors,
        parameter_constraints=parameter_constraints,
        parameter_metadata=parameter_metadata,
    )
    resolved_options = isnothing(options) ? get(specification.metadata, :crystallography_options, Crystallography.CrystallographyOptions()) : options
    resolved_options isa Crystallography.CrystallographyOptions || throw(ArgumentError("options must be CrystallographyOptions or nothing"))
    dataset = specification.space_group isa Crystallography.CrystallographicDataset ?
        specification.space_group : Crystallography.discover_symmetry(crystal; options=resolved_options)
    metadata = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(generation_metadata))
    metadata[:source] = isnothing(material) ? :generative_specification :
        (isnothing(cluster) ? :material_primitive_cell : :material_periodic_supercell)
    metadata[:spacegroup_number] = dataset.spacegroup_number
    metadata[:international_symbol] = dataset.international_symbol
    metadata[:model_space_dimension] = length(space)
    metadata[:parameter_dof] = parameter_dof(parameter_space)
    metadata[:structural_signature] = structural_signature(model_structure)
    metadata[:candidate_source] = get(space.metadata, :candidate_source, :unknown)
    metadata[:completeness_scope] = get(space.basis.metadata, :completeness_scope, :unknown)
    return HamiltonianGenerationResult(material, cluster, crystal, dataset, specification, space, model_structure, parameter_space, metadata)
end

"""
    generate_hamiltonian_family(specification, crystal; kwargs...)
    generate_hamiltonian_family(material, algebra; kwargs...)
    generate_hamiltonian_family(material, cluster, algebra; kwargs...)

Generate a crystallographically invariant Hamiltonian family and return all authoritative Batch 7--10 layers together: physical specification, admissible Hamiltonian space, canonical structure, and constraint-bearing parameter schema. Material overloads derive the crystallographic structure and specification before invoking the existing invariant generator.

The material-only overload addresses degrees of freedom in the primitive cell. Use the periodic-supercell overload when distinct translated sites must appear explicitly in candidate interactions.
"""
function generate_hamiltonian_family(specification::GenerativeSpecification, crystal::Crystallography.CrystalStructure; kwargs...)
    return _generation_result(specification, crystal; kwargs...)
end

function generate_hamiltonian_family(
    material::Material,
    algebra::AbstractOperatorAlgebra;
    candidates=nothing,
    max_operator_degree=nothing,
    basis_labels=nothing,
    generation_metadata=Dict{Symbol,Any}(),
    parameter_values=nothing,
    parameter_bounds=nothing,
    parameter_units=nothing,
    parameter_fixed=nothing,
    parameter_priors=nothing,
    parameter_constraints=(),
    parameter_metadata=Dict{Symbol,Any}(),
    specification_kwargs...,
)
    specification, crystal, _ = _material_specification(material, nothing, algebra; specification_kwargs...)
    options = get(specification.metadata, :crystallography_options, nothing)
    return _generation_result(
        specification,
        crystal;
        material=material,
        candidates=candidates,
        max_operator_degree=max_operator_degree,
        basis_labels=basis_labels,
        options=options,
        generation_metadata=generation_metadata,
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        parameter_units=parameter_units,
        parameter_fixed=parameter_fixed,
        parameter_priors=parameter_priors,
        parameter_constraints=parameter_constraints,
        parameter_metadata=parameter_metadata,
    )
end

function generate_hamiltonian_family(
    material::Material,
    cluster::Supercell,
    algebra::AbstractOperatorAlgebra;
    candidates=nothing,
    max_operator_degree=nothing,
    basis_labels=nothing,
    generation_metadata=Dict{Symbol,Any}(),
    parameter_values=nothing,
    parameter_bounds=nothing,
    parameter_units=nothing,
    parameter_fixed=nothing,
    parameter_priors=nothing,
    parameter_constraints=(),
    parameter_metadata=Dict{Symbol,Any}(),
    specification_kwargs...,
)
    specification, crystal, _ = _material_specification(material, cluster, algebra; specification_kwargs...)
    options = get(specification.metadata, :crystallography_options, nothing)
    return _generation_result(
        specification,
        crystal;
        material=material,
        cluster=cluster,
        candidates=candidates,
        max_operator_degree=max_operator_degree,
        basis_labels=basis_labels,
        options=options,
        generation_metadata=generation_metadata,
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        parameter_units=parameter_units,
        parameter_fixed=parameter_fixed,
        parameter_priors=parameter_priors,
        parameter_constraints=parameter_constraints,
        parameter_metadata=parameter_metadata,
    )
end

"""
    instantiate_model(result, representation, parameters=ParameterPoint(result.parameter_space); kwargs...)

Instantiate one `ManyBodyModel` from a generated Hamiltonian family while retaining a provenance record identifying the Batch 10 generator and structural signature. Physical/model-space and parameter-schema validation are delegated to the existing Foundation constructors.
"""
function instantiate_model(result::HamiltonianGenerationResult, representation::Representation, parameters=ParameterPoint(result.parameter_space);
                           observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}(), provenance=Any[])
    history = Any[provenance...]
    push!(history, (
        generator=:batch10_material_crystallographic,
        source=result.metadata[:source],
        spacegroup_number=result.dataset.spacegroup_number,
        structural_signature=structural_signature(result.model_structure),
    ))
    return instantiate_model(
        result.hamiltonian_space,
        representation,
        parameters;
        observables=observables,
        metadata_parameters=metadata_parameters,
        provenance=history,
    )
end
