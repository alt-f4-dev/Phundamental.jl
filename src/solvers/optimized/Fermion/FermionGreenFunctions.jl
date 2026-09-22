# Sector-changing fermion operators, projected resolvents, and cluster perturbation theory.

struct ProjectedFermionResolvent
    alpha::Vector{Float64}
    beta::Vector{Float64}
    norm::Float64
    projections::Matrix{ComplexF64}
    tail_beta::Float64
    iterations::Int
end

"""Cached zero-temperature spin-resolved fermion Green-function data across the ground, particle-addition, and particle-removal sectors."""
struct FermionGreenWorkspace
    model::ManyBodyModel
    spin::Symbol
    sites::Vector{Int}
    modes::Vector{Int}
    ground_sector::FermionSector
    particle_sector::FermionSector
    hole_sector::FermionSector
    ground_energy::Float64
    ground_state::Vector{ComplexF64}
    ground_basis::ComputationalBasis
    particle_basis::ComputationalBasis
    hole_basis::ComputationalBasis
    ground_hamiltonian::SparseMatrixCSC{ComplexF64,Int}
    particle_hamiltonian::SparseMatrixCSC{ComplexF64,Int}
    hole_hamiltonian::SparseMatrixCSC{ComplexF64,Int}
    particle_resolvents::Vector{ProjectedFermionResolvent}
    hole_resolvents::Vector{ProjectedFermionResolvent}
    metadata::Dict{Symbol,Any}
end

function _fermion_mode_for_site(model::ManyBodyModel, site::Integer, spin::Symbol)
    spin in (:up, :down) || throw(ArgumentError("spin must be :up or :down"))
    fmap = get(model.parameters, :fermion_mode_map, nothing)
    isnothing(fmap) && throw(ArgumentError("fermion Green functions require model.parameters[:fermion_mode_map]"))
    haskey(fmap, (Int(site), spin)) || throw(ArgumentError("no fermion mode is registered for site $site and spin $spin"))
    return Int(fmap[(Int(site), spin)])
end

function _fermion_sites_and_modes(model::ManyBodyModel, spin::Symbol)
    fmap = get(model.parameters, :fermion_mode_map, nothing)
    isnothing(fmap) && throw(ArgumentError("fermion Green functions require model.parameters[:fermion_mode_map]"))
    sites = sort!(unique(Int(site) for (site, sigma) in keys(fmap) if sigma === spin))
    isempty(sites) && throw(ArgumentError("no $spin fermion modes are registered in this model"))
    modes = [_fermion_mode_for_site(model, site, spin) for site in sites]
    return sites, modes
end

function _fermion_target_sector(sector::FermionSector, spin::Symbol, delta::Int, nsites::Int)
    isnothing(sector.nup) && throw(ArgumentError("spin-resolved Green functions require FermionSector(nup=..., ndown=...)"))
    if spin === :up
        nup = sector.nup + delta
        ndown = sector.ndown
    elseif spin === :down
        nup = sector.nup
        ndown = sector.ndown + delta
    else
        throw(ArgumentError("spin must be :up or :down"))
    end
    0 <= nup <= nsites || throw(ArgumentError("target nup=$nup lies outside 0:$nsites"))
    0 <= ndown <= nsites || throw(ArgumentError("target ndown=$ndown lies outside 0:$nsites"))
    return FermionSector(nup=nup, ndown=ndown)
end

"""Apply one fermion creation or annihilation operator between two explicit computational bases."""
function fermion_transition_vector(
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_basis::ComputationalBasis,
    mode::Integer;
    create::Bool,
)
    length(state) == length(source_basis) || throw(DimensionMismatch("fermion transition source vector has the wrong dimension"))
    length(source_basis.layout.factors) == 1 || throw(ArgumentError("fermion transition vectors require a non-composite basis"))
    factor = source_basis.layout.factors[1]
    factor.family === :fermion || throw(ArgumentError("source basis is not fermionic"))
    output = zeros(ComplexF64, length(target_basis))
    @inbounds for column in eachindex(source_basis.states)
        amplitude = state[column]
        iszero(amplitude) && continue
        source_state = source_basis.states[column]
        source_state isa FermionBasisState || throw(ArgumentError("source basis contains a non-fermion state"))
        transition = _fermion_transition(create, Int(mode), source_state, factor)
        isnothing(transition) && continue
        target_state, sign = transition
        row = get(target_basis.index, target_state, 0)
        row == 0 && continue
        output[row] += ComplexF64(sign) * amplitude
    end
    return output
end

function fermion_transition_vector(
    model::ManyBodyModel,
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_basis::ComputationalBasis,
    site::Integer;
    spin::Symbol=:up,
    create::Bool,
)
    mode = _fermion_mode_for_site(model, site, spin)
    return fermion_transition_vector(state, source_basis, target_basis, mode; create=create)
end

function _packed_fermion_transition_vector!(
    output::AbstractVector{ComplexF64},
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_data::PackedFermionBasisData,
    mode::Int;
    create::Bool,
)
    length(state) == length(source_basis) || throw(DimensionMismatch("fermion transition source vector has the wrong dimension"))
    length(output) == length(target_data.masks) || throw(DimensionMismatch("fermion transition target vector has the wrong dimension"))
    factor = source_basis.layout.factors[1]
    pos = get(factor.positions, mode, 0)
    pos == 0 && throw(ArgumentError("fermion operator refers to unknown mode $mode"))
    flag = UInt64(1) << (pos - 1)
    fill!(output, 0.0 + 0.0im)
    @inbounds for column in eachindex(source_basis.states)
        amplitude = state[column]
        iszero(amplitude) && continue
        mask = source_basis.states[column].bits
        occupied = (mask & flag) != 0
        create == occupied && continue
        sign = _fermion_parity(mask, pos)
        target_mask = create ? (mask | flag) : (mask & ~flag)
        row = get(target_data.index, target_mask, 0)
        row == 0 && continue
        output[row] += sign * amplitude
    end
    return output
end

function _packed_fermion_transition_vector(
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_data::PackedFermionBasisData,
    mode::Int;
    create::Bool,
)
    output = zeros(ComplexF64, length(target_data.masks))
    return _packed_fermion_transition_vector!(output, state, source_basis, target_data, mode; create=create)
end

function _fermion_transition_seeds(
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_data::PackedFermionBasisData,
    modes::AbstractVector{<:Integer};
    create::Bool,
    parallel::Bool,
)
    seeds = zeros(ComplexF64, length(target_data.masks), length(modes))
    if parallel && Threads.nthreads() > 1 && length(modes) > 1
        Threads.@threads :static for column in eachindex(modes)
            _packed_fermion_transition_vector!(view(seeds, :, column), state, source_basis, target_data, Int(modes[column]); create=create)
        end
    else
        for column in eachindex(modes)
            _packed_fermion_transition_vector!(view(seeds, :, column), state, source_basis, target_data, Int(modes[column]); create=create)
        end
    end
    return seeds
end

function _sparse_packed_fermion_operator(A::PackedFermionOperator)
    dimension = length(A.data.masks)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    estimate = min(dimension * min(length(A.terms), 8), 25_000_000)
    sizehint!(rows, estimate)
    sizehint!(columns, estimate)
    sizehint!(values, estimate)

    @inbounds for column in 1:dimension
        input = A.data.masks[column]
        diagonal = 0.0 + 0.0im
        for term in A.terms
            output, amplitude, ok = _apply_fermion_term(term, input, A.data)
            ok || continue
            row = get(A.data.index, output, 0)
            row == 0 && continue
            if row == column
                diagonal += amplitude
            else
                push!(rows, row)
                push!(columns, column)
                push!(values, amplitude)
            end
        end
        if !iszero(diagonal)
            push!(rows, column)
            push!(columns, column)
            push!(values, diagonal)
        end
    end

    H = sparse(rows, columns, values, dimension, dimension)
    dropzeros!(H)
    return H
end

function _sparse_fermion_sector_hamiltonian(model::ManyBodyModel, sector::FermionSector; max_basis_dimension::Int)
    _fermion_sector_known_conserved(model, sector) ||
        throw(ArgumentError("requested fermion-number sector is not conserved by this Hamiltonian"))
    packed = _packed_fermion_operator(model, sector; max_dimension=max_basis_dimension)
    return _sparse_packed_fermion_operator(packed), packed.data
end

function _sparse_ground_state(
    H::SparseMatrixCSC{ComplexF64,Int};
    krylov_dim::Int,
    tol::Float64,
    seed::Int,
    reorthogonalize::Bool,
)
    dimension = size(H, 1)
    decomposition = _optimized_lanczos_decomposition(
        H,
        dimension;
        krylov_dim=min(dimension, krylov_dim),
        tol=tol,
        seed=seed,
        reorthogonalize=reorthogonalize,
    )
    energies, states, residuals = _optimized_ritz(decomposition, 1; tol=tol)
    state = ComplexF64.(vec(states[:, 1]))
    state ./= norm(state)
    return Float64(energies[1]), state, Float64(residuals[1]), decomposition.iterations
end

function _projected_fermion_resolvent_streaming(
    A,
    source::AbstractVector,
    sinks::AbstractMatrix;
    krylov_dim::Int,
    tol::Float64,
)
    source_norm = norm(source)
    nsinks = size(sinks, 2)
    source_norm > 0 || return ProjectedFermionResolvent(Float64[], Float64[], 0.0, zeros(ComplexF64, nsinks, 0), 0.0, 0)
    length(source) == size(A, 1) || throw(DimensionMismatch("projected fermion source has the wrong dimension"))
    size(sinks, 1) == size(A, 1) || throw(DimensionMismatch("projected fermion sinks have the wrong dimension"))

    mmax = min(size(A, 1), krylov_dim)
    q = Vector{ComplexF64}(undef, length(source))
    copyto!(q, source)
    q ./= source_norm
    # qprev is unread before the first rotation and work is overwritten by mul!, so neither buffer requires zero initialization.
    qprev = similar(q)
    work = similar(q)
    alpha = zeros(Float64, mmax)
    beta = zeros(Float64, max(mmax - 1, 0))
    projections = zeros(ComplexF64, nsinks, mmax)
    actual = 0
    tail_beta = 0.0

    for j in 1:mmax
        actual = j
        mul!(view(projections, :, j), adjoint(sinks), q)
        mul!(work, A, q)
        if j > 1
            work .-= beta[j - 1] .* qprev
        end
        aj = real(dot(q, work))
        alpha[j] = aj
        work .-= aj .* q
        bj = norm(work)
        tail_beta = bj
        if bj <= tol || j == mmax
            break
        end
        beta[j] = bj
        # Rotate buffer ownership so the residual becomes the next Krylov vector without copying a full sector vector.
        qprev, q, work = q, work, qprev
        q ./= bj
    end

    return ProjectedFermionResolvent(alpha[1:actual], beta[1:max(actual - 1, 0)], source_norm, projections[:, 1:actual], tail_beta, actual)
end

function _projected_fermion_resolvent_reorthogonalized(
    A,
    source::AbstractVector,
    sinks::AbstractMatrix;
    krylov_dim::Int,
    tol::Float64,
)
    source_vector = ComplexF64.(collect(source))
    source_norm = norm(source_vector)
    nsinks = size(sinks, 2)
    source_norm > 0 || return ProjectedFermionResolvent(Float64[], Float64[], 0.0, zeros(ComplexF64, nsinks, 0), 0.0, 0)
    decomposition = _optimized_lanczos_decomposition(
        A,
        length(source_vector);
        krylov_dim=min(length(source_vector), krylov_dim),
        tol=tol,
        initial_vector=source_vector,
        reorthogonalize=true,
    )
    projections = adjoint(sinks) * decomposition.basis
    return ProjectedFermionResolvent(
        decomposition.alpha,
        decomposition.beta,
        source_norm,
        Matrix{ComplexF64}(projections),
        decomposition.tail_beta,
        decomposition.iterations,
    )
end

function _projected_fermion_resolvents(
    A,
    seeds::AbstractMatrix;
    krylov_dim::Int,
    tol::Float64,
    reorthogonalize::Bool,
    parallel::Bool,
)
    nsource = size(seeds, 2)
    output = Vector{ProjectedFermionResolvent}(undef, nsource)
    run_parallel = parallel && !reorthogonalize && Threads.nthreads() > 1 && nsource > 1

    worker = function (column)
        source = view(seeds, :, column)
        if reorthogonalize
            output[column] = _projected_fermion_resolvent_reorthogonalized(A, source, seeds; krylov_dim=krylov_dim, tol=tol)
        else
            output[column] = _projected_fermion_resolvent_streaming(A, source, seeds; krylov_dim=krylov_dim, tol=tol)
        end
    end

    if run_parallel
        old_blas_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            Threads.@threads :static for column in 1:nsource
                worker(column)
            end
        finally
            BLAS.set_num_threads(old_blas_threads)
        end
    else
        for column in 1:nsource
            worker(column)
        end
    end
    return output
end

"""Construct a reusable projected-Lanczos workspace for the zero-temperature cluster Green matrix `G_ab(z)`."""
function fermion_green_workspace(
    model::ManyBodyModel;
    ground_sector::FermionSector,
    spin::Symbol=:up,
    ground_krylov_dim::Integer=192,
    green_krylov_dim::Integer=192,
    tol::Real=1e-10,
    ground_reorthogonalize::Bool=false,
    green_reorthogonalize::Bool=false,
    parallel::Bool=true,
    seed::Integer=0,
    max_basis_dimension::Integer=50_000_000,
)
    ground_krylov_dim > 1 || throw(ArgumentError("ground_krylov_dim must exceed one"))
    green_krylov_dim > 1 || throw(ArgumentError("green_krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("fermion_green_workspace requires a FermionAlgebra model"))

    sites, modes = _fermion_sites_and_modes(model, spin)
    nsites = length(sites)
    particle_sector = _fermion_target_sector(ground_sector, spin, 1, nsites)
    hole_sector = _fermion_target_sector(ground_sector, spin, -1, nsites)
    tolerance = Float64(tol)
    max_dimension = Int(max_basis_dimension)

    ground_build = @timed _sparse_fermion_sector_hamiltonian(model, ground_sector; max_basis_dimension=max_dimension)
    ground_hamiltonian, ground_data = ground_build.value
    ground_solve = @timed _sparse_ground_state(
        ground_hamiltonian;
        krylov_dim=Int(ground_krylov_dim),
        tol=tolerance,
        seed=Int(seed),
        reorthogonalize=ground_reorthogonalize,
    )
    ground_energy, ground_state, ground_residual, ground_iterations = ground_solve.value

    particle_build = @timed _sparse_fermion_sector_hamiltonian(model, particle_sector; max_basis_dimension=max_dimension)
    particle_hamiltonian, particle_data = particle_build.value
    hole_build = @timed _sparse_fermion_sector_hamiltonian(model, hole_sector; max_basis_dimension=max_dimension)
    hole_hamiltonian, hole_data = hole_build.value

    particle_seed_timing = @timed _fermion_transition_seeds(
        ground_state,
        ground_data.basis,
        particle_data,
        modes;
        create=true,
        parallel=parallel,
    )
    hole_seed_timing = @timed _fermion_transition_seeds(
        ground_state,
        ground_data.basis,
        hole_data,
        modes;
        create=false,
        parallel=parallel,
    )
    particle_seeds = particle_seed_timing.value
    hole_seeds = hole_seed_timing.value

    particle_chain_timing = @timed _projected_fermion_resolvents(
        particle_hamiltonian,
        particle_seeds;
        krylov_dim=Int(green_krylov_dim),
        tol=tolerance,
        reorthogonalize=green_reorthogonalize,
        parallel=parallel,
    )
    hole_chain_timing = @timed _projected_fermion_resolvents(
        hole_hamiltonian,
        hole_seeds;
        krylov_dim=Int(green_krylov_dim),
        tol=tolerance,
        reorthogonalize=green_reorthogonalize,
        parallel=parallel,
    )

    metadata = Dict{Symbol,Any}(
        :ground_dimension => size(ground_hamiltonian, 1),
        :particle_dimension => size(particle_hamiltonian, 1),
        :hole_dimension => size(hole_hamiltonian, 1),
        :ground_residual => ground_residual,
        :ground_iterations => ground_iterations,
        :ground_krylov_dim => Int(ground_krylov_dim),
        :green_krylov_dim => Int(green_krylov_dim),
        :green_reorthogonalize => green_reorthogonalize,
        :green_chain_backend => green_reorthogonalize ? :reorthogonalized_scalar : :streaming_scalar_rotating_buffers,
        :parallel => parallel,
        :threads => Threads.nthreads(),
        :ground_build_time => ground_build.time,
        :ground_solve_time => ground_solve.time,
        :particle_build_time => particle_build.time,
        :hole_build_time => hole_build.time,
        :particle_seed_time => particle_seed_timing.time,
        :hole_seed_time => hole_seed_timing.time,
        :particle_chain_time => particle_chain_timing.time,
        :hole_chain_time => hole_chain_timing.time,
        :ground_build_bytes => ground_build.bytes,
        :ground_solve_bytes => ground_solve.bytes,
        :particle_build_bytes => particle_build.bytes,
        :hole_build_bytes => hole_build.bytes,
        :particle_seed_bytes => particle_seed_timing.bytes,
        :hole_seed_bytes => hole_seed_timing.bytes,
        :particle_chain_bytes => particle_chain_timing.bytes,
        :hole_chain_bytes => hole_chain_timing.bytes,
    )
    metadata[:workspace_time] = sum(Float64(metadata[key]) for key in (
        :ground_build_time,
        :ground_solve_time,
        :particle_build_time,
        :hole_build_time,
        :particle_seed_time,
        :hole_seed_time,
        :particle_chain_time,
        :hole_chain_time,
    ))
    metadata[:workspace_bytes] = sum(Int(metadata[key]) for key in (
        :ground_build_bytes,
        :ground_solve_bytes,
        :particle_build_bytes,
        :hole_build_bytes,
        :particle_seed_bytes,
        :hole_seed_bytes,
        :particle_chain_bytes,
        :hole_chain_bytes,
    ))

    return FermionGreenWorkspace(
        model,
        spin,
        sites,
        modes,
        ground_sector,
        particle_sector,
        hole_sector,
        ground_energy,
        ground_state,
        ground_data.basis,
        particle_data.basis,
        hole_data.basis,
        ground_hamiltonian,
        particle_hamiltonian,
        hole_hamiltonian,
        particle_chain_timing.value,
        hole_chain_timing.value,
        metadata,
    )
end

function _projected_resolvent_dimension(chain::ProjectedFermionResolvent, krylov_dim)
    available = length(chain.alpha)
    isnothing(krylov_dim) && return available
    requested = Int(krylov_dim)
    requested > 0 || throw(ArgumentError("krylov_dim must be positive when supplied"))
    return min(requested, available)
end

function _projected_resolvent_column(
    chain::ProjectedFermionResolvent,
    z::ComplexF64,
    ground_energy::Float64,
    channel::Symbol;
    krylov_dim=nothing,
)
    chain.norm > 0 || return zeros(ComplexF64, size(chain.projections, 1))
    n = _projected_resolvent_dimension(chain, krylov_dim)
    n > 0 || return zeros(ComplexF64, size(chain.projections, 1))
    rhs = zeros(ComplexF64, n)
    rhs[1] = 1.0 + 0.0im
    alpha = view(chain.alpha, 1:n)
    beta = view(chain.beta, 1:max(n - 1, 0))
    if channel === :particle
        diagonal = ComplexF64.(z + ground_energy .- alpha)
        offdiagonal = ComplexF64.(-beta)
    elseif channel === :hole
        diagonal = ComplexF64.(z - ground_energy .+ alpha)
        offdiagonal = ComplexF64.(beta)
    else
        throw(ArgumentError("channel must be :particle or :hole"))
    end
    projected = Tridiagonal(offdiagonal, diagonal, offdiagonal) \ rhs
    return chain.norm .* (view(chain.projections, :, 1:n) * projected)
end

"""Evaluate the spin-resolved cluster Green matrix at complex spectral argument `z`. Set `krylov_dim` to evaluate a truncated projected-Lanczos chain without rebuilding the workspace."""
function cluster_green_matrix(workspace::FermionGreenWorkspace, z::Number; krylov_dim=nothing)
    complex_z = ComplexF64(z)
    nsites = length(workspace.sites)
    green = zeros(ComplexF64, nsites, nsites)
    for source in 1:nsites
        green[:, source] .+= _projected_resolvent_column(
            workspace.particle_resolvents[source], complex_z, workspace.ground_energy, :particle; krylov_dim=krylov_dim,
        )
        green[source, :] .+= _projected_resolvent_column(
            workspace.hole_resolvents[source], complex_z, workspace.ground_energy, :hole; krylov_dim=krylov_dim,
        )
    end
    return green
end

function cluster_green_matrix(workspace::FermionGreenWorkspace, axis; eta::Real, parallel::Bool=true, krylov_dim=nothing)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    frequencies = Float64.(collect(axis))
    nsites = length(workspace.sites)
    green = Array{ComplexF64}(undef, nsites, nsites, length(frequencies))
    if parallel && Threads.nthreads() > 1 && length(frequencies) > 1
        Threads.@threads :static for index in eachindex(frequencies)
            green[:, :, index] .= cluster_green_matrix(workspace, ComplexF64(frequencies[index], eta); krylov_dim=krylov_dim)
        end
    else
        for index in eachindex(frequencies)
            green[:, :, index] .= cluster_green_matrix(workspace, ComplexF64(frequencies[index], eta); krylov_dim=krylov_dim)
        end
    end
    return green
end

"""Return the one-dimensional inter-cluster hopping matrix `V(Q)` of the Senechal CPT construction."""
function cpt_intercluster_hopping(nsites::Integer, Q::Real; t_intercluster::Number)
    nsites >= 2 || throw(ArgumentError("CPT requires a cluster with at least two sites"))
    hopping = zeros(ComplexF64, Int(nsites), Int(nsites))
    hopping[end, 1] = -ComplexF64(t_intercluster) * cis(Float64(Q))
    hopping[1, end] = -conj(ComplexF64(t_intercluster)) * cis(-Float64(Q))
    return hopping
end

"""Reconstruct the mixed cluster/superlattice Green matrix from a cluster Green matrix and superlattice momentum `Q`."""
function cluster_perturbation_green_matrix(cluster_green::AbstractMatrix, Q::Real; t_intercluster::Number)
    size(cluster_green, 1) == size(cluster_green, 2) || throw(DimensionMismatch("cluster Green matrix must be square"))
    nsites = size(cluster_green, 1)
    hopping = cpt_intercluster_hopping(nsites, Q; t_intercluster=t_intercluster)
    identity_matrix = Matrix{ComplexF64}(I, nsites, nsites)
    return (identity_matrix - cluster_green * hopping) \ Matrix{ComplexF64}(cluster_green)
end

"""Periodize a mixed CPT Green matrix to a scalar Green function at original-lattice momentum `k`."""
function cpt_periodize(green::AbstractMatrix, k::Real; positions=nothing)
    size(green, 1) == size(green, 2) || throw(DimensionMismatch("CPT Green matrix must be square"))
    nsites = size(green, 1)
    coordinates = isnothing(positions) ? collect(0:(nsites - 1)) : Float64.(collect(positions))
    length(coordinates) == nsites || throw(DimensionMismatch("CPT positions must contain one coordinate per cluster site"))
    phase = ComplexF64[cis(Float64(k) * coordinate) for coordinate in coordinates]
    return dot(phase, green * phase) / nsites
end

"""Evaluate the periodized one-dimensional CPT Green function at original-lattice momentum `k`."""
function cluster_perturbation_green(
    cluster_green::AbstractMatrix,
    k::Real;
    t_intercluster::Number,
    cluster_translation::Real=size(cluster_green, 1),
    positions=nothing,
)
    Q = Float64(cluster_translation) * Float64(k)
    mixed_green = cluster_perturbation_green_matrix(cluster_green, Q; t_intercluster=t_intercluster)
    return cpt_periodize(mixed_green, k; positions=positions)
end
