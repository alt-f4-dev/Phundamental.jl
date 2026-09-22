# Validation.jl

function _tensor_pair_symmetry_error(K)
    N = size(K, 1)
    worst = 0.0
    for i in 1:N, j in 1:N
        worst = max(
            worst,
            norm(view(K, i, j, :, :) - transpose(view(K, j, i, :, :))),
        )
    end
    return worst / max(norm(K), 1.0)
end

"""
    validate_long_range_result(result; tolerance=...)

Check pair symmetry and recorded Ewald cutoff convergence.
"""
function validate_long_range_result(
    result::ScalarInteractionResult;
    tolerance::Real=1e-9,
)
    symmetry = norm(result.total - transpose(result.total)) /
               max(norm(result.total), 1.0)
    recorded = get(result.metadata, :convergence_error_estimate, 0.0)
    converged = get(result.metadata, :converged, true)
    passed = symmetry <= tolerance &&
             converged !== false &&
             (isnan(recorded) || recorded <= max(tolerance, get(result.metadata, :convergence_tolerance, tolerance)))
    return Dict{Symbol,Any}(
        :passed => passed,
        :pair_symmetry_error => symmetry,
        :cutoff_converged => converged,
        :cutoff_change => recorded,
    )
end

function validate_long_range_result(
    result::TensorInteractionResult;
    tolerance::Real=1e-9,
)
    symmetry = _tensor_pair_symmetry_error(result.total)
    cartesian_symmetry = maximum(
        norm(view(result.total, i, i, :, :) - transpose(view(result.total, i, i, :, :)))
        for i in 1:size(result.total, 1)
    ) / max(norm(result.total), 1.0)
    recorded = get(result.metadata, :convergence_error_estimate, 0.0)
    converged = get(result.metadata, :converged, true)
    passed = max(symmetry, cartesian_symmetry) <= tolerance &&
             converged !== false &&
             (isnan(recorded) || recorded <= max(tolerance, get(result.metadata, :convergence_tolerance, tolerance)))
    return Dict{Symbol,Any}(
        :passed => passed,
        :pair_symmetry_error => symmetry,
        :diagonal_cartesian_symmetry_error => cartesian_symmetry,
        :cutoff_converged => converged,
        :cutoff_change => recorded,
    )
end

"""
    validate_ewald_alpha_independence(interaction, cluster; ...)

Repeat converged Ewald calculations with several splitting parameters.

For Coulomb interactions without a neutralizing background, the periodic
Green-function pair matrix is defined only on the neutral charge subspace.
Accordingly this validator compares energies of deterministic neutral test
charge vectors rather than raw matrix elements, avoiding a meaningless
alpha-dependent constant offset.

For dipoles the full tin-foil tensor is compared directly.
"""
function validate_ewald_alpha_independence(
    interaction::CoulombInteraction,
    cluster::Supercell;
    alpha_factors=(0.75, 1.0, 1.25),
    tolerance::Real=1e-7,
)
    base_method = EwaldSummation(tolerance=min(tolerance / 10, 1e-10))
    alpha0, _, _ = _auto_ewald_parameters(cluster, base_method)
    results = [
        interaction_matrix(
            interaction,
            cluster;
            method=EwaldSummation(
                alpha=alpha0 * f,
                tolerance=min(tolerance / 10, 1e-10),
                check_convergence=true,
                strict=true,
            ),
        )
        for f in alpha_factors
    ]

    N = nsites(cluster)
    test_vectors = Vector{Vector{Float64}}()
    if N >= 2
        q = zeros(Float64, N); q[1] = 1; q[2] = -1; push!(test_vectors, q)
    end
    if N >= 3
        q = zeros(Float64, N); q[1] = 1; q[2] = 1; q[3] = -2; push!(test_vectors, q)
    end
    isempty(test_vectors) && throw(ArgumentError("alpha-independence test needs at least two sites"))

    worst = 0.0
    energies = Vector{Vector{Float64}}()
    for result in results
        push!(energies, [interaction_energy(result, q) for q in test_vectors])
    end
    reference = energies[2]
    for e in energies
        worst = max(worst, norm(e - reference) / max(norm(reference), 1.0))
    end

    return Dict{Symbol,Any}(
        :passed => worst <= tolerance,
        :relative_energy_spread => worst,
        :alpha_values => alpha0 .* collect(alpha_factors),
        :energies => energies,
    )
end

function validate_ewald_alpha_independence(
    interaction::DipolarInteraction,
    cluster::Supercell;
    alpha_factors=(0.75, 1.0, 1.25),
    tolerance::Real=1e-7,
)
    base_method = EwaldSummation(tolerance=min(tolerance / 10, 1e-10))
    alpha0, _, _ = _auto_ewald_parameters(cluster, base_method)
    results = [
        interaction_tensor(
            interaction,
            cluster;
            method=EwaldSummation(
                alpha=alpha0 * f,
                tolerance=min(tolerance / 10, 1e-10),
                check_convergence=true,
                strict=true,
            ),
        )
        for f in alpha_factors
    ]

    reference = results[2].total
    worst = maximum(
        norm(r.total - reference) / max(norm(reference), 1.0)
        for r in results
    )
    return Dict{Symbol,Any}(
        :passed => worst <= tolerance,
        :relative_tensor_spread => worst,
        :alpha_values => alpha0 .* collect(alpha_factors),
    )
end
