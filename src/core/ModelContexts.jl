# Typed, operator-index-aware context carriers live in Foundation to avoid model/solver dependency cycles.

"""Certified, site-major operator ordering for spinful fermions or Cartesian vector coordinates."""
struct ComponentIndexLayout
    kind::Symbol
    nsites::Int
    components_per_site::Int
    norbitals::Int

    function ComponentIndexLayout(kind::Symbol, nsites::Integer; norbitals::Integer=1)
        nsites > 0 || throw(ArgumentError("component layouts require a positive site count"))
        norbitals > 0 || throw(ArgumentError("norbitals must be positive"))
        kind in (:spinful_fermion, :cartesian_vector) || throw(ArgumentError("unsupported certified component layout $kind"))
        kind === :cartesian_vector && norbitals != 1 && throw(ArgumentError("Cartesian vector layouts do not have orbital indices"))
        components = kind === :spinful_fermion ? 2 * Int(norbitals) : 3
        return new(kind, Int(nsites), components, Int(norbitals))
    end
end

"""Cartesian site geometry, optionally carrying a certified site-major component layout and physical site masses in amu."""
struct GeometryContext{T<:AbstractFloat}
    positions::Vector{Vector{T}}
    direct::Matrix{T}
    reciprocal::Matrix{T}
    labels::Vector{Symbol}
    indexing::Symbol
    length_unit::Union{Nothing,Symbol}
    component_layout::Union{Nothing,ComponentIndexLayout}
    site_masses::Union{Nothing,Vector{T}}
end

function GeometryContext(positions, direct, reciprocal, labels; indexing::Symbol=:one_index_per_site,
                         length_unit=nothing, component_layout=nothing, site_masses=nothing)
    indexing in (:one_index_per_site, :site_component_layout) || throw(ArgumentError("unsupported certified geometry indexing $indexing"))
    A = Matrix{Float64}(direct)
    B = Matrix{Float64}(reciprocal)
    size(A, 1) == size(A, 2) == size(B, 1) == size(B, 2) || throw(DimensionMismatch("direct and reciprocal matrices must be square and have equal dimension"))
    xyz = [Float64.(collect(point)) for point in positions]
    isempty(xyz) && throw(ArgumentError("geometry requires at least one position"))
    all(point -> length(point) == size(A, 1) && all(isfinite, point), xyz) || throw(ArgumentError("Cartesian site positions must be finite and match lattice dimension"))
    all(isfinite, A) && all(isfinite, B) || throw(ArgumentError("lattice matrices must be finite"))
    length(labels) == length(xyz) || throw(DimensionMismatch("site labels do not match positions"))
    names = Symbol.(collect(labels))
    length(unique(names)) == length(names) || throw(ArgumentError("site labels must be unique"))
    isapprox(A' * B, 2π * Matrix{Float64}(I, size(A, 1), size(A, 2)); atol=1e-8, rtol=1e-8) ||
        throw(ArgumentError("reciprocal matrix must obey A' * B = 2πI"))
    if indexing === :one_index_per_site
        isnothing(component_layout) || throw(ArgumentError("one-index-per-site geometry cannot carry a component layout"))
    else
        component_layout isa ComponentIndexLayout || throw(ArgumentError("component-aware geometry requires a certified ComponentIndexLayout"))
        component_layout.nsites == length(xyz) || throw(DimensionMismatch("component layout and geometry disagree on site count"))
        component_layout.kind !== :cartesian_vector || size(A, 1) == 3 || throw(DimensionMismatch("Cartesian vector layouts require three-dimensional geometry"))
    end
    masses = if isnothing(site_masses)
        nothing
    else
        values = Float64.(collect(site_masses))
        length(values) == length(xyz) || throw(DimensionMismatch("site masses and positions must have equal lengths"))
        all(m -> isfinite(m) && m > 0, values) || throw(ArgumentError("site masses must be finite and positive"))
        values
    end
    return GeometryContext(xyz, A, B, names, indexing, isnothing(length_unit) ? nothing : Symbol(length_unit), component_layout, masses)
end

# Legacy positional construction preserves the pre-F2 six-field geometry interface.
GeometryContext(positions, direct, reciprocal, labels, indexing::Symbol, length_unit) =
    GeometryContext(positions, direct, reciprocal, labels; indexing=indexing, length_unit=length_unit)

"""Map a certified operator component to its physical crystallographic site."""
function component_site(geometry::GeometryContext, mode::Integer)
    count = isnothing(geometry.component_layout) ? 1 : geometry.component_layout.components_per_site
    1 <= mode <= count * length(geometry.positions) || throw(BoundsError(1:(count * length(geometry.positions)), mode))
    return fld(Int(mode) - 1, count) + 1
end

"""Map a physical site and its one-based local component to the operator index."""
function component_index(geometry::GeometryContext, site::Integer, component::Integer)
    1 <= site <= length(geometry.positions) || throw(BoundsError(geometry.positions, site))
    count = isnothing(geometry.component_layout) ? 1 : geometry.component_layout.components_per_site
    1 <= component <= count || throw(BoundsError(1:count, component))
    return (Int(site) - 1) * count + Int(component)
end

"""Map `(site, orbital, spin)` to a spinful-fermion mode, with `spin=1` for up and `spin=2` for down."""
function component_index(geometry::GeometryContext, site::Integer, orbital::Integer, spin::Integer)
    layout = geometry.component_layout
    !isnothing(layout) && layout.kind === :spinful_fermion || throw(ArgumentError("orbital/spin indexing requires certified spinful-fermion geometry"))
    1 <= orbital <= layout.norbitals || throw(BoundsError(1:layout.norbitals, orbital))
    1 <= spin <= 2 || throw(BoundsError(1:2, spin))
    return component_index(geometry, site, 2 * (Int(orbital) - 1) + Int(spin))
end

"""Map `(site, :x/:y/:z)` to a Cartesian displacement or momentum mode."""
function component_index(geometry::GeometryContext, site::Integer, axis::Symbol)
    layout = geometry.component_layout
    !isnothing(layout) && layout.kind === :cartesian_vector || throw(ArgumentError("Cartesian-axis indexing requires certified vector geometry"))
    axis in (:x, :y, :z) || throw(ArgumentError("Cartesian component must be :x, :y, or :z"))
    return component_index(geometry, site, findfirst(==(axis), (:x, :y, :z)))
end

"""Return the physical site and internal labels of a certified operator index."""
function component_coordinates(geometry::GeometryContext, mode::Integer)
    site = component_site(geometry, mode)
    layout = geometry.component_layout
    isnothing(layout) && return (; site)
    component = Int(mode) - (site - 1) * layout.components_per_site
    if layout.kind === :spinful_fermion
        return (; site, orbital=fld(component - 1, 2) + 1, spin=mod(component - 1, 2) + 1)
    end
    return (; site, axis=(:x, :y, :z)[component])
end

"""Material and crystallographic provenance, with geometry only when operator-index ordering is certified."""
struct PhysicalModelContext{M,C,D,G}
    material::M
    cluster::C
    dataset::D
    geometry::G
    source::Symbol
end

function PhysicalModelContext(material, cluster, dataset, geometry::Union{Nothing,GeometryContext}; source::Symbol=:material_generation)
    return PhysicalModelContext(material, cluster, dataset, geometry, source)
end

"""Return the typed context of a model, or `nothing` for manually constructed legacy models."""
function model_context(model)
    context = get(model.parameters, :physical_context, nothing)
    isnothing(context) || context isa PhysicalModelContext || throw(ArgumentError("model.parameters[:physical_context] must be a PhysicalModelContext"))
    return context
end

"""Retrieve certified Cartesian geometry without guessing a multicomponent operator-to-site mapping."""
function geometry_context(model)
    context = model_context(model)
    return isnothing(context) ? nothing : context.geometry
end

"""Typed registry contract; conventional operator names retain their existing Dict-based forward compatibility."""
struct ObservableContext{R}
    registry::R
    convention::Symbol
end

"""Inspect the observable registry's declared convention, or return `nothing` for a legacy manually built model."""
function observable_context(model)
    context = get(model.parameters, :observable_context, nothing)
    isnothing(context) || context isa ObservableContext || throw(ArgumentError("model.parameters[:observable_context] must be an ObservableContext"))
    return context
end
