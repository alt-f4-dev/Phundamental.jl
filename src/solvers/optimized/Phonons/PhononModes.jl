function _solve_phonon_dynamical_matrix(
    solver::HarmonicPhononSolver,
    model::ManyBodyModel,
    D::AbstractMatrix,
    q,
)
    hermitian_error = norm(D - adjoint(D)) / max(norm(D), 1.0)
    hermitian_error <= max(solver.tol, 1e-12) ||
        throw(ArgumentError("mass-weighted dynamical matrix is not Hermitian; error=$hermitian_error"))

    F = eigen(Hermitian(0.5 .* (D .+ adjoint(D))))
    raw_lambda = Float64.(real.(F.values))
    scale = max(maximum(abs, raw_lambda; init=0.0), 1.0)
    zero_tol = solver.zero_mode_factor * eps(Float64) * scale
    instability_tol = max(solver.tol * scale, zero_tol)
    minimum(raw_lambda; init=0.0) >= -instability_tol ||
        throw(ArgumentError("harmonic model contains an unstable mode with omega^2=$(minimum(raw_lambda))"))

    omega2 = copy(raw_lambda)
    omega2[abs.(omega2) .<= zero_tol] .= 0.0
    omega2 .= max.(omega2, 0.0)
    zero_mode_mask = iszero.(omega2)
    omega = sqrt.(omega2)

    masses = Float64.(collect(model.parameters[:masses]))
    physical_modes = Diagonal(1 ./ sqrt.(masses)) * F.vectors
    qcart = isnothing(q) ? nothing : _phonon_cartesian_wavevector(model, q)
    metadata = Dict{Symbol,Any}(
        :solver => :harmonic_phonon,
        :backend => haskey(model.parameters, :periodic_force_constants) ?
                    :bloch_dynamical_matrix : :dynamical_matrix,
        :dimension => length(omega),
        :omega_squared => omega2,
        :raw_omega_squared => raw_lambda,
        :zero_mode_mask => BitVector(zero_mode_mask),
        :zero_mode_eigenvalue_tolerance => zero_tol,
        :zero_mode_factor => solver.zero_mode_factor,
        :dynamical_matrix => Matrix(D),
        :physical_displacement_modes => Matrix(physical_modes),
        :hermiticity_error => hermitian_error,
        :symmetry_error => hermitian_error,
        :wavevector => qcart,
        :wavevector_basis => isnothing(q) ? nothing : _phonon_wavevector_basis(q),
        :wavevector_input => isnothing(q) ? nothing : Float64.(collect(q)),
        :periodic => haskey(model.parameters, :periodic_force_constants),
        :phonon_units => get(model.parameters, :phonon_units, nothing),
        :approximation => :none_for_harmonic_model,
        :full_fock_space_constructed => false,
    )
    return PhononModeResult(
        solver,
        model,
        omega,
        Matrix{ComplexF64}(F.vectors),
        metadata,
    )
end

function solve(solver::HarmonicPhononSolver, model::ManyBodyModel)
    return _solve_phonon_dynamical_matrix(
        solver,
        model,
        dynamical_matrix(model),
        nothing,
    )
end

"""
    solve(solver::HarmonicPhononSolver, model, q)

Solve a primitive-cell harmonic phonon problem. Raw vectors retain the legacy Cartesian convention; explicit `CartesianWaveVector` and `ReciprocalWaveVector` inputs are preferred.

The solver records `metadata[:zero_mode_mask]` from the eigenvalue-space numerical-zero policy so downstream thermal and scattering calculations can use the same classification.
"""
function solve(solver::HarmonicPhononSolver, model::ManyBodyModel, q)
    Dq = dynamical_matrix(model, q)
    return _solve_phonon_dynamical_matrix(solver, model, Dq, q)
end

function _phonon_degenerate_groups(lambda::AbstractVector{<:Real}, tol::Real)
    isempty(lambda) && return UnitRange{Int}[]
    scale = max(maximum(abs, lambda), 1.0)
    groups = UnitRange{Int}[]
    first_index = 1
    for i in 2:length(lambda)
        if abs(lambda[i] - lambda[i - 1]) > tol * scale
            push!(groups, first_index:(i - 1))
            first_index = i
        end
    end
    push!(groups, first_index:length(lambda))
    return groups
end

"""
    group_velocities(result; degeneracy_tol=1e-8, degeneracy_policy=:nan)

Cartesian group-velocity components for a `PhononModeResult`.

For a nondegenerate finite-frequency mode,

    v_nu,mu = e_nu' (dD/dq_mu) e_nu / (2 omega_nu).

At an exact/near finite-frequency degeneracy, a branch-resolved Cartesian
velocity vector is basis-dependent because the projected derivative operators
need not commute. The default `degeneracy_policy=:nan` therefore marks those
rows as `NaN`. Use `directional_group_velocities` for a well-defined
degenerate-perturbation result along a chosen direction. `:basis` returns the
expectation values in the eigensolver's current basis; `:error` raises.
"""
function group_velocities(
    result::PhononModeResult;
    degeneracy_tol::Real=1e-8,
    degeneracy_policy::Symbol=:nan,
)
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    degeneracy_policy in (:nan, :basis, :error) ||
        throw(ArgumentError("degeneracy_policy must be :nan, :basis, or :error"))
    get(result.metadata, :periodic, false) ||
        throw(ArgumentError("group velocities require a periodic q-resolved phonon result"))
    qcart = get(result.metadata, :wavevector, nothing)
    isnothing(qcart) && throw(ArgumentError("group velocities require a q-resolved phonon result"))

    dD = dynamical_gradient(result.model, CartesianWaveVector(qcart))
    lambda = Float64.(result.metadata[:omega_squared])
    omega = Float64.(result.frequencies)
    modes = result.modes
    Dspace = size(dD, 3)
    velocities = fill(NaN, length(omega), Dspace)
    groups = _phonon_degenerate_groups(lambda, Float64(degeneracy_tol))
    zero_tol = sqrt(max(Float64(get(result.metadata, :zero_mode_eigenvalue_tolerance, 0.0)), 0.0))

    for group in groups
        degenerate = length(group) > 1
        if degenerate && degeneracy_policy === :error
            throw(ArgumentError("group velocity is branch-ambiguous for degenerate modes $(collect(group)); use directional_group_velocities"))
        elseif degenerate && degeneracy_policy === :nan
            continue
        end

        for ν in group
            omega[ν] > zero_tol || continue
            e = view(modes, :, ν)
            for μ in 1:Dspace
                dH = Hermitian(0.5 .* (dD[:, :, μ] .+ adjoint(dD[:, :, μ])))
                velocities[ν, μ] = real(dot(e, dH * e)) / (2 * omega[ν])
            end
        end
    end
    return velocities
end

"""
    directional_group_velocities(result, direction; degeneracy_tol=1e-8)

Directional phonon slopes along a Cartesian unit direction. Degenerate
finite-frequency subspaces are treated with first-order degenerate perturbation
theory by diagonalizing the projected directional derivative `P' D_n' P`.

The values within a degenerate block are ordered by slope and are not intended
to provide a unique continuation label across a crossing.
"""
function directional_group_velocities(
    result::PhononModeResult,
    direction;
    degeneracy_tol::Real=1e-8,
)
    qcart = get(result.metadata, :wavevector, nothing)
    isnothing(qcart) && throw(ArgumentError("directional group velocities require a q-resolved result"))
    n = Float64.(collect(direction))
    Dspace = result.model.parameters[:spatial_dimension]
    length(n) == Dspace || throw(DimensionMismatch("direction must have $Dspace components"))
    norm(n) > 0 || throw(ArgumentError("direction must be nonzero"))
    n ./= norm(n)

    dD = dynamical_gradient(result.model, CartesianWaveVector(qcart))
    dDn = zeros(ComplexF64, size(dD, 1), size(dD, 2))
    for μ in 1:Dspace
        dDn .+= n[μ] .* dD[:, :, μ]
    end
    dDn .= 0.5 .* (dDn .+ adjoint(dDn))

    lambda = Float64.(result.metadata[:omega_squared])
    omega = Float64.(result.frequencies)
    groups = _phonon_degenerate_groups(lambda, Float64(degeneracy_tol))
    slopes = fill(NaN, length(omega))
    zero_tol = sqrt(max(Float64(get(result.metadata, :zero_mode_eigenvalue_tolerance, 0.0)), 0.0))

    for group in groups
        omega0 = maximum(omega[group])
        omega0 > zero_tol || continue
        P = result.modes[:, group]
        projected_matrix = adjoint(P) * dDn * P
        projected = Hermitian(0.5 .* (projected_matrix .+ adjoint(projected_matrix)))
        local_slopes = sort!(Float64.(real.(eigvals(projected))) ./ (2 * omega0))
        slopes[group] .= local_slopes
    end
    return slopes
end

"""
    phonon_dispersion(solver, model, qpath; derivatives=false,
                      degeneracy_policy=:nan)

Evaluate harmonic phonon modes along a wavevector sequence. Input points may be
raw Cartesian vectors or explicit wavevector wrappers. The stored `qpoints`
remain Cartesian for backward compatibility.

With `derivatives=true`, Cartesian group velocities are evaluated analytically and retained as `metadata[:group_velocities]` with shape `(Nq, Nbranch, spatial_dimension)`.

The dispersion also retains per-q solver zero-mode classifications in `metadata[:zero_mode_mask]`, together with the raw and projected squared frequencies and the eigenvalue-space tolerances used at each q point.
"""
function phonon_dispersion(
    solver::HarmonicPhononSolver,
    model::ManyBodyModel,
    qpath;
    derivatives::Bool=false,
    degeneracy_tol::Real=1e-8,
    degeneracy_policy::Symbol=:nan,
)
    qinputs = collect(qpath)
    isempty(qinputs) && throw(ArgumentError("phonon q path cannot be empty"))
    qs = [_phonon_cartesian_wavevector(model, q) for q in qinputs]
    first_result = solve(solver, model, first(qinputs))
    nbranch = length(first_result.frequencies)
    nq = length(qinputs)
    Dspace = model.parameters[:spatial_dimension]
    frequencies = Matrix{Float64}(undef, nq, nbranch)
    modes = Array{ComplexF64}(undef, nbranch, nbranch, nq)
    omega_squared = Matrix{Float64}(undef, nq, nbranch)
    raw_omega_squared = Matrix{Float64}(undef, nq, nbranch)
    zero_mode_mask = falses(nq, nbranch)
    zero_mode_eigenvalue_tolerances = Vector{Float64}(undef, nq)
    hermiticity_errors = Vector{Float64}(undef, nq)
    velocities = derivatives ? Array{Float64}(undef, nq, nbranch, Dspace) : nothing

    frequencies[1, :] .= first_result.frequencies
    modes[:, :, 1] .= first_result.modes
    omega_squared[1, :] .= first_result.metadata[:omega_squared]
    raw_omega_squared[1, :] .= first_result.metadata[:raw_omega_squared]
    zero_mode_mask[1, :] .= first_result.metadata[:zero_mode_mask]
    zero_mode_eigenvalue_tolerances[1] = first_result.metadata[:zero_mode_eigenvalue_tolerance]
    hermiticity_errors[1] = first_result.metadata[:hermiticity_error]
    if derivatives
        velocities[1, :, :] .= group_velocities(
            first_result;
            degeneracy_tol=degeneracy_tol,
            degeneracy_policy=degeneracy_policy,
        )
    end

    for iq in 2:nq
        result = solve(solver, model, qinputs[iq])
        length(result.frequencies) == nbranch ||
            throw(DimensionMismatch("phonon branch count changed along q path"))
        frequencies[iq, :] .= result.frequencies
        modes[:, :, iq] .= result.modes
        omega_squared[iq, :] .= result.metadata[:omega_squared]
        raw_omega_squared[iq, :] .= result.metadata[:raw_omega_squared]
        zero_mode_mask[iq, :] .= result.metadata[:zero_mode_mask]
        zero_mode_eigenvalue_tolerances[iq] = result.metadata[:zero_mode_eigenvalue_tolerance]
        hermiticity_errors[iq] = result.metadata[:hermiticity_error]
        if derivatives
            velocities[iq, :, :] .= group_velocities(
                result;
                degeneracy_tol=degeneracy_tol,
                degeneracy_policy=degeneracy_policy,
            )
        end
    end

    metadata = Dict{Symbol,Any}(
        :solver => :harmonic_phonon,
        :backend => :bloch_dynamical_matrix,
        :periodic => true,
        :branch_count => nbranch,
        :qpoint_count => nq,
        :omega_squared => omega_squared,
        :raw_omega_squared => raw_omega_squared,
        :zero_mode_mask => zero_mode_mask,
        :zero_mode_eigenvalue_tolerances => zero_mode_eigenvalue_tolerances,
        :zero_mode_factor => solver.zero_mode_factor,
        :hermiticity_errors => hermiticity_errors,
        :max_hermiticity_error => maximum(hermiticity_errors),
        :displacement_dimension => model.parameters[:displacement_dimension],
        :mass_weighted_force_constants => get(model.parameters, :mass_weighted_force_constants, false),
        :wavevector_input_bases => [_phonon_wavevector_basis(q) for q in qinputs],
        :derivatives => derivatives,
        :phonon_units => get(model.parameters, :phonon_units, nothing),
    )
    if derivatives
        metadata[:group_velocities] = velocities
        metadata[:group_velocity_degeneracy_policy] = degeneracy_policy
        metadata[:group_velocity_degeneracy_tolerance] = Float64(degeneracy_tol)
    end
    return PhononDispersionResult(solver, model, qs, frequencies, modes, metadata)
end

# -----------------------------------------------------------------------------
# High-level harmonic-phonon convenience API
# -----------------------------------------------------------------------------

function _require_periodic_harmonic_model(model::ManyBodyModel)
    haskey(model.parameters, :periodic_force_constants) || throw(ArgumentError(
        "this convenience method requires a primitive periodic harmonic phonon model",
    ))
    return model
end

function _convenience_phonon_q(model::ManyBodyModel, q::Real)
    D = model.parameters[:spatial_dimension]
    D == 1 || throw(ArgumentError(
        "scalar reciprocal coordinates are only unambiguous for one-dimensional models; use ReciprocalWaveVector for D=$D",
    ))
    return ReciprocalWaveVector([Float64(q)])
end

_convenience_phonon_q(::ManyBodyModel, q::AbstractWaveVector) = q

function _convenience_phonon_path(model::ManyBodyModel, qpath)
    points = collect(qpath)
    isempty(points) && throw(ArgumentError("phonon q path cannot be empty"))
    if all(point -> point isa Real, points)
        D = model.parameters[:spatial_dimension]
        D == 1 || throw(ArgumentError(
            "a scalar q path is only unambiguous for one-dimensional models; use ReciprocalWaveVector points for D=$D",
        ))
        return [ReciprocalWaveVector([Float64(point)]) for point in points]
    end
    return points
end

function _uniform_phonon_reciprocal_mesh(model::ManyBodyModel, shape::Tuple)
    D = model.parameters[:spatial_dimension]
    length(shape) == D || throw(DimensionMismatch("qmesh must contain one mesh size per spatial dimension"))
    all(value -> value isa Integer && value > 0, shape) || throw(ArgumentError("all qmesh dimensions must be positive integers"))
    dims = ntuple(index -> Int(shape[index]), D)
    axes = ntuple(index -> 0:(dims[index] - 1), D)
    return [
        ReciprocalWaveVector(Float64[index[d] / dims[d] for d in 1:D])
        for index in Iterators.product(axes...)
    ]
end

"""
    solve(model, q; solver=HarmonicPhononSolver())

Solve a periodic harmonic phonon model using the default harmonic solver. A
scalar `q` is interpreted as a reciprocal-lattice coordinate only for a
one-dimensional model. Higher-dimensional calls should use an explicit
`ReciprocalWaveVector` or `CartesianWaveVector`.
"""
function solve(
    model::ManyBodyModel,
    q::AbstractWaveVector;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
)
    _require_periodic_harmonic_model(model)
    return solve(solver, model, q)
end

function solve(
    model::ManyBodyModel,
    q::Real;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
)
    _require_periodic_harmonic_model(model)
    return solve(solver, model, _convenience_phonon_q(model, q))
end

"""
    phonon_dispersion(model, qpath; solver=HarmonicPhononSolver(), kwargs...)

Evaluate a periodic harmonic phonon dispersion without explicitly constructing
`HarmonicPhononSolver`. For a one-dimensional model, a scalar path is treated
as reciprocal-lattice coordinates. Other raw vectors retain the existing
legacy Cartesian convention; explicit wavevector wrappers remain preferred.
"""
function phonon_dispersion(
    model::ManyBodyModel,
    qpath;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    kwargs...,
)
    _require_periodic_harmonic_model(model)
    return phonon_dispersion(solver, model, _convenience_phonon_path(model, qpath); kwargs...)
end

"""
    phonon_zero_point_energy(model; qmesh, weights=nothing, solver=HarmonicPhononSolver(), hbar=1)

Return the harmonic zero-point ground-state energy per primitive cell,

    E0 / Ncell = (hbar / 2Nq) sum(q,nu) omega(q,nu).

A tuple `qmesh=(N1, N2, ...)` generates a uniform reciprocal mesh. An explicit
wavevector collection may be supplied instead. Optional nonnegative `weights`
are normalized internally; equal weights are used by default. `hbar=1` matches
the package's internal-unit convention; callers using frequency rather than
energy units should supply the appropriate conversion factor explicitly.
"""
function phonon_zero_point_energy(
    model::ManyBodyModel;
    qmesh,
    weights=nothing,
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    hbar::Real=1.0,
)
    _require_periodic_harmonic_model(model)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    qpoints = qmesh isa Tuple ? _uniform_phonon_reciprocal_mesh(model, qmesh) : _convenience_phonon_path(model, qmesh)
    dispersion = phonon_dispersion(solver, model, qpoints)
    qweights = if isnothing(weights)
        fill(inv(Float64(length(qpoints))), length(qpoints))
    else
        values = Float64.(collect(weights))
        length(values) == length(qpoints) || throw(DimensionMismatch("weights must contain one value per q point"))
        all(>=(0), values) || throw(ArgumentError("q-point weights must be nonnegative"))
        sum(values) > 0 || throw(ArgumentError("q-point weights must have positive total weight"))
        values ./ sum(values)
    end
    branch_sums = vec(sum(dispersion.frequencies; dims=2))
    return 0.5 * Float64(hbar) * dot(qweights, branch_sums)
end

function groundenergy(
    model::ManyBodyModel;
    qmesh,
    weights=nothing,
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    hbar::Real=1.0,
)
    return phonon_zero_point_energy(model; qmesh=qmesh, weights=weights, solver=solver, hbar=hbar)
end
