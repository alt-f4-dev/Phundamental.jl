# Coupled cluster construction, model building, thermal solution, and
# linked-cluster reconstruction.

function _apply_cluster_model_transform(model::ManyBodyModel, model_transform)
    isnothing(model_transform) && return model
    transformed = model_transform(model)
    transformed isa ManyBodyModel && return transformed
    hasproperty(transformed, :model) && getproperty(transformed, :model) isa ManyBodyModel &&
        return getproperty(transformed, :model)
    throw(ArgumentError(
        "model_transform must return ManyBodyModel or an object with a `.model::ManyBodyModel` field"
    ))
end

function _cluster_method(method::AbstractThermalMethod, cluster_index::Int)
    method isa ExactGibbs && return method
    # Decorrelate random streams deterministically between embedded clusters.
    return _sample_method_seed(method, method.seed + 1_000_003 * (cluster_index - 1))
end

"""
    cluster_thermodynamics(material, reference, spec;
        max_order, temperatures, method=ExactGibbs(), ...)

Coupled finite-reference linked-cluster calculation:

    reference geometry
      -> connected induced clusters
      -> Hamiltonian construction
      -> optional representation/model transform
      -> exact or stochastic thermal calculation
      -> linked-cluster inclusion-exclusion
      -> order-by-order per-site thermodynamics.

This is a finite-reference linked-cluster estimator.  It should be interpreted
as an infinite-lattice NLCE only when the chosen reference is large enough
that the retained cluster orders are unaffected by the reference boundary.

`model_builder` may be overridden for Hamiltonian specifications with
nonstandard site-dependent parameter indexing.  Its signature is

    model_builder(material, cluster, spec, bonds) -> ManyBodyModel

The default delegates to `build_model(...; bonds=bonds)`.
"""
function cluster_thermodynamics(
    material::Material,
    reference::Supercell,
    spec::AbstractHamiltonianSpec;
    max_order::Integer,
    temperatures,
    method::AbstractThermalMethod=ExactGibbs(),
    kB::Real=1.0,
    bonds=nothing,
    min_order::Integer=1,
    max_clusters::Integer=100_000,
    model_transform=nothing,
    model_builder=nothing,
    allow_periodic_reference::Bool=false,
)
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperature grid cannot be empty"))
    min_order == 1 || throw(ArgumentError(
        "linked-cluster inclusion-exclusion requires min_order=1"
    ))

    # Construct the reference model once to obtain the exact bond set selected
    # by the Hamiltonian specification when bonds were not supplied explicitly.
    reference_model = if isnothing(bonds)
        build_model(material, reference, spec)
    else
        build_model(material, reference, spec; bonds=collect(bonds))
    end
    ref_bonds = isnothing(bonds) ?
        get(reference_model.parameters, :bonds, nothing) :
        collect(bonds)
    isnothing(ref_bonds) && throw(ArgumentError(
        "Hamiltonian builder did not expose `parameters[:bonds]`; supply bonds explicitly"
    ))

    clusters = connected_clusters(
        reference,
        ref_bonds;
        max_order=max_order,
        min_order=min_order,
        max_clusters=max_clusters,
        allow_periodic_reference=allow_periodic_reference,
    )

    builder = if isnothing(model_builder)
        (mat, cl, sp, B) -> build_model(mat, cl, sp; bonds=B)
    else
        model_builder
    end

    raw = Dict{Tuple{Vararg{Int}},ThermodynamicCurve}()
    for (idx, cluster) in enumerate(clusters)
        model = builder(material, cluster.supercell, spec, cluster.bonds)
        model isa ManyBodyModel || throw(ArgumentError(
            "model_builder must return ManyBodyModel"
        ))
        working = _apply_cluster_model_transform(model, model_transform)
        local_method = _cluster_method(method, idx)
        raw[cluster.key] = thermal_curve(
            working,
            local_method,
            Ts;
            kB=kB,
        )
    end

    weights = linked_cluster_weights(
        raw, clusters, Ts, nsites(reference),
    )
    partial, convergence = _linked_partial_sums(
        weights, clusters, Int(max_order), length(Ts), nsites(reference),
    )

    metadata = Dict{Symbol,Any}(
        :method => :finite_reference_linked_cluster,
        :thermal_method => typeof(method),
        :max_order => Int(max_order),
        :min_order => Int(min_order),
        :cluster_count => length(clusters),
        :reference_sites => nsites(reference),
        :reference_periodic => collect(reference.periodic),
        :per_site_normalization => true,
        :infinite_lattice_claim => false,
        :interpretation => :convergence_before_reference_boundary_required,
        :stochastic_weight_uncertainties_propagated => false,
    )

    return ClusterThermodynamicsResult(
        Ts,
        clusters,
        raw,
        weights,
        partial,
        convergence,
        nsites(reference),
        metadata,
    )
end

"""Per-site linked-cluster estimate through the highest calculated order."""
function bulk_estimate(result::ClusterThermodynamicsResult, property::Symbol)
    haskey(result.partial_sums, property) ||
        throw(KeyError("unknown linked-cluster property $property"))
    return vec(result.partial_sums[property][end, :])
end

"""Per-site linked-cluster estimate through a specified cluster order."""
function bulk_estimate(
    result::ClusterThermodynamicsResult,
    property::Symbol,
    order::Integer,
)
    haskey(result.partial_sums, property) ||
        throw(KeyError("unknown linked-cluster property $property"))
    1 <= order <= size(result.partial_sums[property], 1) ||
        throw(BoundsError(result.partial_sums[property], order))
    return vec(result.partial_sums[property][Int(order), :])
end
