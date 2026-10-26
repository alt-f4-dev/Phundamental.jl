# Model-aware construction of Foundation context carriers from certified Batch 10 provenance.

"""Derive material geometry only for generator-certified operator/site layouts with matching algebra and local spaces."""
function generation_context(result::HamiltonianGenerationResult, algebra::AbstractOperatorAlgebra)
    PhundamentalCore._same_model_algebra(algebra, result.specification.algebra) ||
        throw(ArgumentError("requested geometry algebra disagrees with the generated Hamiltonian family"))
    crystal = result.crystal_structure
    n = length(crystal)
    mapping = get(result.specification.metadata, :operator_site_mapping, :custom)
    action = result.specification.symmetry_action
    spaces = result.specification.local_spaces
    one_per_site = mapping === :one_index_per_crystallographic_site && (
        (algebra isa SpinAlgebra && algebra.nsites == n) ||
        (algebra isa FermionAlgebra && algebra.nmodes == n) ||
        (algebra isa BosonAlgebra && algebra.nmodes == n))
    layout = if mapping === :ordered_internal_components && action isa Crystallography.SpinfulFermionAction &&
                algebra isa FermionAlgebra && algebra.nmodes == 2 * action.norbitals * n &&
                length(spaces) == n && all(space -> space isa FermionFockSpace && space.nmodes == 2 * action.norbitals, spaces)
        ComponentIndexLayout(:spinful_fermion, n; norbitals=action.norbitals)
    elseif mapping === :ordered_internal_components && action isa Crystallography.VectorDisplacementAction &&
            algebra isa CoordinateMomentumAlgebra && algebra.ndof == 3 * n &&
            length(spaces) == n && all(space -> space isa CoordinateHilbertSpace && space.ndof == 3, spaces)
        ComponentIndexLayout(:cartesian_vector, n)
    else
        nothing
    end
    geometry = if one_per_site || !isnothing(layout)
        A = Matrix{Float64}(crystal.lattice)
        positions = [collect(A * crystal.positions[:, i]) for i in 1:n]
        masses = if isnothing(result.material)
            nothing
        elseif isnothing(result.cluster)
            [atomic_species(result.material, site.species).mass for site in result.material.crystal.basis]
        else
            [site_mass(result.material, site) for site in result.cluster.sites]
        end
        GeometryContext(positions, A, 2π * inv(A)', crystal.labels;
            indexing=isnothing(layout) ? :one_index_per_site : :site_component_layout,
            component_layout=layout, site_masses=masses)
    else
        nothing
    end
    return PhysicalModelContext(result.material, result.cluster, result.dataset, geometry)
end

function _fermionic_site_observables!(registry::Dict{Symbol,Any}, geometry::GeometryContext; hbar::Real=1.0)
    isfinite(hbar) && hbar > 0 || throw(ArgumentError("generated fermionic spin observables require finite positive hbar"))
    norbitals = geometry.component_layout.norbitals
    for site in eachindex(geometry.positions)
        density_terms = AbstractOperatorExpr[]
        double_terms = AbstractOperatorExpr[]
        sx_terms = AbstractOperatorExpr[]
        sy_terms = AbstractOperatorExpr[]
        sz_terms = AbstractOperatorExpr[]
        for orbital in 1:norbitals
            up = component_index(geometry, site, orbital, 1)
            down = component_index(geometry, site, orbital, 2)
            density = nf(up) + nf(down)
            double_occupancy = nf(up) * nf(down)
            push!(density_terms, density)
            push!(double_terms, double_occupancy)
            push!(sx_terms, (hbar / 2) * (c(up)' * c(down) + c(down)' * c(up)))
            push!(sy_terms, (hbar / (2im)) * (c(up)' * c(down) - c(down)' * c(up)))
            push!(sz_terms, (hbar / 2) * (nf(up) - nf(down)))
            registry[Symbol("fermion_density_", site, "_", orbital)] = density
            registry[Symbol("fermion_double_occupancy_", site, "_", orbital)] = double_occupancy
        end
        density = OperatorSum(density_terms)
        double_occupancy = OperatorSum(double_terms)
        spin_x = OperatorSum(sx_terms)
        spin_y = OperatorSum(sy_terms)
        spin_z = OperatorSum(sz_terms)
        registry[Symbol("fermion_density_", site)] = density
        registry[Symbol("fermion_double_occupancy_", site)] = double_occupancy
        registry[Symbol("fermion_spin_x_", site)] = spin_x
        registry[Symbol("fermion_spin_y_", site)] = spin_y
        registry[Symbol("fermion_spin_z_", site)] = spin_z
        registry[Symbol("density_", site)] = density
        registry[Symbol("double_occupancy_", site)] = double_occupancy
        registry[Symbol("spin_x_", site)] = spin_x
        registry[Symbol("spin_y_", site)] = spin_y
        registry[Symbol("spin_z_", site)] = spin_z
    end
    return registry
end

function _vector_site_observables!(registry::Dict{Symbol,Any}, geometry::GeometryContext)
    for site in eachindex(geometry.positions), axis in (:x, :y, :z)
        index = component_index(geometry, site, axis)
        registry[Symbol("phonon_displacement_", axis, "_", site)] = xop(index)
        registry[Symbol("phonon_momentum_", axis, "_", site)] = pop(index)
    end
    return registry
end

"""Build physical-site observables only for certified spin, spinful-fermion, and Cartesian-vector layouts."""
function _generated_observables(algebra::AbstractOperatorAlgebra, context::PhysicalModelContext; hbar::Real=1.0)
    registry = Dict{Symbol,Any}()
    geometry = context.geometry
    isnothing(geometry) && return registry
    if algebra isa SpinAlgebra && geometry.indexing === :one_index_per_site
        for site in eachindex(geometry.positions)
            registry[Symbol("spin_x_", site)] = Sx(site)
            registry[Symbol("spin_y_", site)] = Sy(site)
            registry[Symbol("spin_z_", site)] = Sz(site)
        end
    elseif algebra isa FermionAlgebra && !isnothing(geometry.component_layout) && geometry.component_layout.kind === :spinful_fermion
        _fermionic_site_observables!(registry, geometry; hbar=hbar)
    elseif algebra isa CoordinateMomentumAlgebra && !isnothing(geometry.component_layout) && geometry.component_layout.kind === :cartesian_vector
        _vector_site_observables!(registry, geometry)
    end
    return registry
end

function _validated_generated_metadata(context::PhysicalModelContext, metadata_parameters)
    data = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata_parameters))
    haskey(data, :physical_context) && throw(ArgumentError("the physical context is generated from the authoritative family and cannot be overridden"))
    haskey(data, :observable_context) && throw(ArgumentError("the observable context is generated from the validated registry and cannot be overridden"))
    data[:physical_context] = context
    for (key, expected) in ((:material, context.material), (:cluster, context.cluster))
        if haskey(data, key)
            !isnothing(expected) && (data[key] === expected || isequal(data[key], expected)) ||
                throw(ArgumentError("explicit :$key disagrees with the generated source"))
        elseif !isnothing(expected)
            data[key] = expected
        end
    end
    if !isnothing(context.geometry)
        positions = context.geometry.positions
        if haskey(data, :positions)
            provided = data[:positions]
            provided isa AbstractVector && length(provided) == length(positions) || throw(DimensionMismatch("provided Cartesian site positions disagree with the generated site ordering"))
            for (actual, expected) in zip(provided, positions)
                actual isa AbstractVector && length(actual) == length(expected) &&
                    all(value -> value isa Real && isfinite(value), actual) &&
                    isapprox(collect(actual), expected; atol=1e-10, rtol=1e-10) ||
                    throw(ArgumentError("provided Cartesian positions disagree with generated geometry"))
            end
        else
            data[:positions] = [copy(position) for position in positions]
        end
    end
    if !isnothing(context.geometry) && !isnothing(context.geometry.component_layout) &&
        context.geometry.component_layout.kind === :cartesian_vector && !isnothing(context.geometry.site_masses)
        layout = context.geometry.component_layout
        expected = [context.geometry.site_masses[component_site(context.geometry, index)] for index in 1:(layout.nsites * layout.components_per_site)]
        if haskey(data, :masses)
            actual = data[:masses]
            actual isa AbstractVector && length(actual) == length(expected) || throw(DimensionMismatch("phonon masses disagree with the certified Cartesian mode layout"))
            all(x -> x isa Real && isfinite(x), actual) && isapprox(Float64.(actual), expected; rtol=1e-10, atol=1e-10) ||
                throw(ArgumentError("phonon masses disagree with the generated material"))
        else
            data[:masses] = expected
        end
    end
    return data
end

function _require_context_label_separation(space::HamiltonianSpace)
    reserved = (:physical_context, :observable_context, :material, :cluster, :positions)
    space.specification.algebra isa CoordinateMomentumAlgebra && (reserved = (reserved..., :masses))
    any(name -> name in reserved, space.basis.labels) &&
        throw(ArgumentError("Hamiltonian parameter labels must not shadow reserved physical metadata keys"))
    return nothing
end

"""
    default_observable_context(result_or_family, representation; observables=Dict(), metadata_parameters=Dict())

Return certified material geometry and conventional operator registries while preserving the legacy metadata fields. Ambiguous operator indexing is marked without fabricating positions, and contradictory user overrides are rejected.
"""
function default_observable_context(result::HamiltonianGenerationResult, representation::Representation;
                                    observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}())
    provided = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(observables))
    context = generation_context(result, representation.algebra)
    geometry = context.geometry
    if !isnothing(geometry) && !isnothing(geometry.component_layout)
        layout = geometry.component_layout
        if layout.kind === :spinful_fermion
            representation.state_space isa FermionFockSpace && representation.state_space.nmodes == layout.nsites * layout.components_per_site ||
                throw(DimensionMismatch("fermion representation must match the certified site/orbital/spin modes"))
        elseif layout.kind === :cartesian_vector
            representation.state_space isa CoordinateHilbertSpace && representation.state_space.ndof == layout.nsites * layout.components_per_site ||
                throw(DimensionMismatch("coordinate representation must match the certified Cartesian displacement modes"))
        end
    end
    hbar = get(metadata_parameters, :hbar, 1.0)
    hbar isa Real && isfinite(hbar) && hbar > 0 || throw(ArgumentError("hbar must be finite, real, and positive"))
    registry = _generated_observables(representation.algebra, context; hbar=hbar)
    for (name, operator) in provided
        haskey(registry, name) && !isequal(registry[name], operator) &&
            throw(ArgumentError("observable :$name disagrees with the certified generated registry"))
        registry[name] = operator
    end
    metadata = _validated_generated_metadata(context, metadata_parameters)
    convention = if isnothing(geometry)
        :explicit_or_unregistered
    elseif representation.algebra isa SpinAlgebra && geometry.indexing === :one_index_per_site
        :spin_cartesian_site
    elseif !isnothing(geometry.component_layout) && geometry.component_layout.kind === :spinful_fermion
        :fermion_site_orbital_spin
    elseif !isnothing(geometry.component_layout) && geometry.component_layout.kind === :cartesian_vector
        :phonon_cartesian_site
    else
        :explicit_or_unregistered
    end
    metadata[:observable_context] = ObservableContext(copy(registry), convention)
    return (; observables=registry, metadata_parameters=metadata)
end

function default_observable_context(family::StructuralFamily, representation::Representation;
                                    observables=Dict{Symbol,Any}(), metadata_parameters=Dict{Symbol,Any}())
    source = get(family.metadata, :generation_result, nothing)
    isnothing(source) && return (; observables=Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(observables)),
                                 metadata_parameters=Dict{Symbol,Any}(Symbol(key) => value for (key, value) in pairs(metadata_parameters)))
    return default_observable_context(source, representation; observables=observables, metadata_parameters=metadata_parameters)
end
