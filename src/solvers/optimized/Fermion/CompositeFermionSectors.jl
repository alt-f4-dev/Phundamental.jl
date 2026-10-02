# Exact fermion-number sectors for composite representations with one fermion factor.

const _CompositeAlgebraRepresentation = Representation{S,A,B,P,G,R,O,T} where {S<:AbstractStateSpace,A<:CompositeAlgebra,B<:AbstractBasis,P,G,R,O,T}

struct CompositeFermionBasisData
    basis::ComputationalBasis
    fermion_factor::Int
    sector::FermionSector
end

function _composite_fermion_factor_index(model::ManyBodyModel; required::Bool=true)
    algebra = model.representation.algebra
    if !(algebra isa CompositeAlgebra)
        required && throw(ArgumentError("composite fermion-sector handling requires a CompositeAlgebra representation"))
        return nothing
    end
    indices = findall(factor -> factor isa FermionAlgebra, algebra.factors)
    if length(indices) != 1
        required && throw(ArgumentError("composite fermion-sector handling requires exactly one FermionAlgebra factor; found $(length(indices))"))
        return nothing
    end
    return only(indices)
end

function _composite_spinful_mode_labels(model::ManyBodyModel, factor::FactorLayout)
    fmap = get(model.parameters, :fermion_mode_map, nothing)
    isnothing(fmap) && throw(ArgumentError("nup/ndown sectors require model.parameters[:fermion_mode_map]"))

    up = Int[]
    down = Int[]
    for ((_, spin), mode0) in fmap
        mode = Int(mode0)
        haskey(factor.positions, mode) || throw(ArgumentError("fermion_mode_map refers to mode $mode outside the fermion tensor factor"))
        spin === :up && push!(up, mode)
        spin === :down && push!(down, mode)
    end
    sort!(unique!(up))
    sort!(unique!(down))
    isempty(up) && throw(ArgumentError("fermion_mode_map contains no spin-up modes"))
    isempty(down) && throw(ArgumentError("fermion_mode_map contains no spin-down modes"))
    isempty(intersect(up, down)) || throw(ArgumentError("spin-up and spin-down mode groups overlap"))

    covered = sort!(vcat(copy(up), copy(down)))
    expected = sort(copy(factor.ordering))
    covered == expected || throw(ArgumentError("nup/ndown sectors require fermion_mode_map to classify every fermion mode exactly once"))
    return up, down
end

function _composite_fermion_sector_constraints(model::ManyBodyModel, factor::FactorLayout, sector::FermionSector)
    modes = copy(factor.ordering)
    if !isnothing(sector.nparticles)
        0 <= sector.nparticles <= length(modes) || throw(ArgumentError("nparticles must lie in 0:$(length(modes))"))
        return [(kind=:single_occupancy, modes=modes, total=sector.nparticles)]
    end

    up, down = _composite_spinful_mode_labels(model, factor)
    0 <= sector.nup <= length(up) || throw(ArgumentError("nup must lie in 0:$(length(up))"))
    0 <= sector.ndown <= length(down) || throw(ArgumentError("ndown must lie in 0:$(length(down))"))
    return [
        (kind=:single_occupancy, modes=up, total=sector.nup),
        (kind=:single_occupancy, modes=down, total=sector.ndown),
    ]
end

function specialize_sector(model::ManyBodyModel{R}, sector::QuantumNumberSector) where {R<:_CompositeAlgebraRepresentation}
    isnothing(_composite_fermion_factor_index(model; required=false)) && return sector
    names = Set(number.name for number in sector.quantum_numbers)
    names == Set([:nparticles]) && return FermionSector(nparticles=Int(sector_value(sector, :nparticles)))
    names == Set([:nup, :ndown]) && return FermionSector(nup=Int(sector_value(sector, :nup)), ndown=Int(sector_value(sector, :ndown)))
    return sector
end

"""
    computational_basis(model, sector::FermionSector; max_basis_dimension=50_000_000)

Construct the finite product basis for a conserved fermion-number sector of a composite representation containing exactly one fermion factor. Other tensor factors retain their representation-level physical constraints and numerical truncations unchanged.
"""
function computational_basis(model::ManyBodyModel{R}, sector::FermionSector; max_basis_dimension::Integer=50_000_000) where {R<:_CompositeAlgebraRepresentation}
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    rep = model.representation
    factors = _representation_factors(rep)
    constraints = _all_physical_constraints(rep)
    fermion_factor_index = _composite_fermion_factor_index(model)

    factor_states = Any[]
    for (factor_index, factor) in enumerate(factors)
        family_constraints = _constraints_for_factor(constraints, factor.family, factor_index, factors)
        if factor_index == fermion_factor_index
            isempty(family_constraints) ||
                throw(ArgumentError("composite FermionSector currently does not combine with representation-level fermion occupancy constraints"))
            family_constraints = _composite_fermion_sector_constraints(model, factor, sector)
        end
        truncation = factor.family === :boson ? _truncation_for_factor(rep.truncation, factor_index) : nothing
        push!(factor_states, _factor_states(factor, family_constraints, truncation))
    end

    states = length(factor_states) == 1 ? factor_states[1] : _composite_states(factor_states)
    if !isnothing(rep.physical_subspace) && !isnothing(rep.physical_subspace.projector)
        projector = rep.physical_subspace.projector
        projector isa Function || throw(ArgumentError("matrix-valued physical projectors are not yet convertible to a sector product-state basis"))
        states = [state for state in states if projector(state)]
    end

    isempty(states) && throw(ArgumentError("the requested composite fermion sector has an empty computational basis"))
    _checked_dimension(length(states), Int(max_basis_dimension), "composite fermion sector")

    state_type = eltype(states)
    index = Dict{state_type,Int}()
    sizehint!(index, length(states))
    for (i, state) in enumerate(states)
        index[state] = i
    end

    layout = BasisLayout(
        factors,
        _family_map(factors),
        constraints,
        rep.truncation,
        !isempty(constraints) || !isnothing(rep.physical_subspace),
        !isnothing(rep.truncation),
    )
    return ComputationalBasis(rep, states, index, layout)
end

function fermion_transition_vector(
    state::AbstractVector,
    source_basis::ComputationalBasis{RS,CompositeBasisState},
    target_basis::ComputationalBasis{RT,CompositeBasisState},
    mode::Integer;
    create::Bool,
) where {RS,RT}
    length(state) == length(source_basis) || throw(DimensionMismatch("fermion transition source vector has the wrong dimension"))
    source_factor_index = get(source_basis.layout.family_to_factor, :fermion, 0)
    target_factor_index = get(target_basis.layout.family_to_factor, :fermion, 0)
    source_factor_index > 0 || throw(ArgumentError("source composite basis does not contain one unambiguous fermion factor"))
    target_factor_index == source_factor_index || throw(ArgumentError("source and target composite bases use different fermion-factor layouts"))

    factor = source_basis.layout.factors[source_factor_index]
    target_factor = target_basis.layout.factors[target_factor_index]
    factor.ordering == target_factor.ordering || throw(ArgumentError("source and target fermion basis orderings differ"))

    output = zeros(ComplexF64, length(target_basis))
    @inbounds for column in eachindex(source_basis.states)
        amplitude = state[column]
        iszero(amplitude) && continue
        source_state = source_basis.states[column]
        local_state = source_state.parts[source_factor_index]
        local_state isa FermionBasisState || throw(ArgumentError("composite fermion factor contains a non-fermion state"))
        transition = _fermion_transition(create, Int(mode), local_state, factor)
        isnothing(transition) && continue
        target_local_state, sign = transition
        target_state = _replace_factor_state(source_state, source_factor_index, target_local_state)
        row = get(target_basis.index, target_state, 0)
        row == 0 && continue
        output[row] += ComplexF64(sign) * amplitude
    end
    return output
end

function _fermion_transition_seeds(
    state::AbstractVector,
    source_basis::ComputationalBasis,
    target_data::CompositeFermionBasisData,
    modes::AbstractVector{<:Integer};
    create::Bool,
    parallel::Bool,
)
    seeds = zeros(ComplexF64, length(target_data.basis), length(modes))
    worker = function (column)
        seed = fermion_transition_vector(state, source_basis, target_data.basis, Int(modes[column]); create=create)
        copyto!(view(seeds, :, column), seed)
    end

    if parallel && Threads.nthreads() > 1 && length(modes) > 1
        Threads.@threads :static for column in eachindex(modes)
            worker(column)
        end
    else
        for column in eachindex(modes)
            worker(column)
        end
    end
    return seeds
end

function _sparse_fermion_sector_hamiltonian(model::ManyBodyModel{R}, sector::FermionSector; max_basis_dimension::Int) where {R<:_CompositeAlgebraRepresentation}
    _composite_fermion_factor_index(model)
    _fermion_sector_known_conserved(model, sector) || throw(ArgumentError("requested fermion-number sector is not conserved by this Hamiltonian"))
    basis = computational_basis(model, sector; max_basis_dimension=max_basis_dimension)
    realized = realize(model.hamiltonian, basis; sparse=true, check_physical_closure=true)
    factor_index = get(basis.layout.family_to_factor, :fermion, 0)
    factor_index > 0 || throw(ArgumentError("composite sector basis lost its fermion-factor mapping"))
    return realized.matrix, CompositeFermionBasisData(basis, factor_index, sector)
end

function _solve_composite_fermion(solver::OptimizedLanczos, model::ManyBodyModel, sector::FermionSector)
    _fermion_sector_known_conserved(model, sector) || throw(ArgumentError("requested fermion-number sector is not conserved by this Hamiltonian"))
    H, data = _sparse_fermion_sector_hamiltonian(model, sector; max_basis_dimension=solver.max_basis_dimension)
    dimension = size(H, 1)
    solver.nev <= dimension || throw(ArgumentError("nev=$(solver.nev) exceeds sector dimension $dimension"))

    hermitian_ok, hermiticity_error = _optimized_hermitian_check(H, dimension; seed=solver.seed + 19, tol=max(10 * solver.tol, 1e-9))
    hermitian_ok || throw(ArgumentError("composite fermion Lanczos requires a Hermitian projected Hamiltonian; error=$hermiticity_error"))

    decomposition = _optimized_lanczos_decomposition(
        H,
        dimension;
        krylov_dim=min(dimension, max(solver.krylov_dim, solver.nev + 2)),
        tol=solver.tol,
        seed=solver.seed,
        reorthogonalize=solver.reorthogonalize,
    )
    energies, right, residuals = _optimized_ritz(decomposition, solver.nev; tol=solver.tol)

    metadata = Dict{Symbol,Any}(
        :solver => :optimized_lanczos,
        :backend => :composite_fermion_sparse,
        :dimension => dimension,
        :sector => sector,
        :sector_observables => :same_sector_only,
        :hermitian => true,
        :hermiticity_error => hermiticity_error,
        :biorthogonal => false,
        :iterations => decomposition.iterations,
        :ritz_residuals => residuals,
        :converged => residuals .<= solver.tol,
        :matrix_free => false,
        :approximation => :none,
        :physical_representation_preserved => true,
    )
    return SolverResult(solver, model, energies, right, right, data.basis, metadata)
end

function solve(solver::OptimizedLanczos, model::ManyBodyModel{R}) where {R<:_CompositeAlgebraRepresentation}
    factor_index = _composite_fermion_factor_index(model; required=false)
    isnothing(factor_index) && return invoke(solve, Tuple{OptimizedLanczos,ManyBodyModel}, solver, model)

    sector = specialize_sector(model, solver.sector)
    selected = solver.backend
    if selected === :fermion || (selected === :auto && sector isa FermionSector)
        sector isa FermionSector || throw(ArgumentError("the composite fermion backend requires a FermionSector; supply nup/ndown or nparticles"))
        return _solve_composite_fermion(solver, model, sector)
    end
    return invoke(solve, Tuple{OptimizedLanczos,ManyBodyModel}, solver, model)
end


function _composite_fermion_workspace(
    model::ManyBodyModel{R};
    ground_sector::FermionSector,
    solution::Union{Nothing,SolverResult}=nothing,
    spin::Symbol=:up,
    ground_krylov_dim::Integer=192,
    green_krylov_dim::Integer=192,
    tol::Real=1e-10,
    ground_reorthogonalize::Bool=false,
    green_reorthogonalize::Bool=false,
    parallel::Bool=true,
    seed::Integer=0,
    max_basis_dimension::Integer=50_000_000,
) where {R<:_CompositeAlgebraRepresentation}
    ground_krylov_dim > 1 || throw(ArgumentError("ground_krylov_dim must exceed one"))
    green_krylov_dim > 1 || throw(ArgumentError("green_krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    _composite_fermion_factor_index(model)

    sites, modes = _fermion_sites_and_modes(model, spin)
    nsites = length(sites)
    particle_sector = _fermion_target_sector(ground_sector, spin, 1, nsites)
    hole_sector = _fermion_target_sector(ground_sector, spin, -1, nsites)
    tolerance = Float64(tol)
    max_dimension = Int(max_basis_dimension)

    ground_build = @timed _sparse_fermion_sector_hamiltonian(model, ground_sector; max_basis_dimension=max_dimension)
    ground_hamiltonian, ground_data = ground_build.value

    if isnothing(solution)
        ground_solve = @timed _sparse_ground_state(
            ground_hamiltonian;
            krylov_dim=Int(ground_krylov_dim),
            tol=tolerance,
            seed=Int(seed),
            reorthogonalize=ground_reorthogonalize,
        )
        ground_energy, ground_state, ground_residual, ground_iterations = ground_solve.value
        ground_solve_time = ground_solve.time
        ground_solve_bytes = ground_solve.bytes
        ground_reused = false
        ground_solution_solver = :internal_lanczos
    else
        solution.model === model || throw(ArgumentError("the supplied SolverResult must belong to the same ManyBodyModel instance"))
        solution.basis isa ComputationalBasis || throw(ArgumentError("fermion Green-function reuse requires an explicit ComputationalBasis"))
        length(solution) > 0 || throw(ArgumentError("the supplied SolverResult contains no eigenstates"))
        ground_data.basis.states == solution.basis.states ||
            throw(ArgumentError("reused solution basis ordering does not match the reconstructed composite fermion sector"))

        ground_energy = Float64(real(groundenergy(solution)))
        ground_state = ComplexF64.(collect(groundstate(solution)))
        ground_norm = norm(ground_state)
        ground_norm > 0 || throw(ArgumentError("the supplied ground state has zero norm"))
        ground_state ./= ground_norm
        ground_residual = _fermion_ground_residual_from_solution(solution)
        ground_iterations = get(solution.metadata, :iterations, nothing)
        ground_solve_time = 0.0
        ground_solve_bytes = 0
        ground_reused = true
        ground_solution_solver = solver_name(solution.solver)
    end

    particle_build = @timed _sparse_fermion_sector_hamiltonian(model, particle_sector; max_basis_dimension=max_dimension)
    particle_hamiltonian, particle_data = particle_build.value
    hole_build = @timed _sparse_fermion_sector_hamiltonian(model, hole_sector; max_basis_dimension=max_dimension)
    hole_hamiltonian, hole_data = hole_build.value

    particle_seed_timing = @timed _fermion_transition_seeds(ground_state, ground_data.basis, particle_data, modes; create=true, parallel=parallel)
    hole_seed_timing = @timed _fermion_transition_seeds(ground_state, ground_data.basis, hole_data, modes; create=false, parallel=parallel)
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
        :ground_reused => ground_reused,
        :ground_solution_solver => ground_solution_solver,
        :ground_krylov_dim => Int(ground_krylov_dim),
        :green_krylov_dim => Int(green_krylov_dim),
        :green_reorthogonalize => green_reorthogonalize,
        :green_chain_backend => green_reorthogonalize ? :reorthogonalized_scalar : :streaming_scalar_rotating_buffers,
        :parallel => parallel,
        :threads => Threads.nthreads(),
        :ground_build_time => ground_build.time,
        :ground_solve_time => ground_solve_time,
        :particle_build_time => particle_build.time,
        :hole_build_time => hole_build.time,
        :particle_seed_time => particle_seed_timing.time,
        :hole_seed_time => hole_seed_timing.time,
        :particle_chain_time => particle_chain_timing.time,
        :hole_chain_time => hole_chain_timing.time,
        :ground_build_bytes => ground_build.bytes,
        :ground_solve_bytes => ground_solve_bytes,
        :particle_build_bytes => particle_build.bytes,
        :hole_build_bytes => hole_build.bytes,
        :particle_seed_bytes => particle_seed_timing.bytes,
        :hole_seed_bytes => hole_seed_timing.bytes,
        :particle_chain_bytes => particle_chain_timing.bytes,
        :hole_chain_bytes => hole_chain_timing.bytes,
        :composite_fermion_factor => ground_data.fermion_factor,
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

"""Construct a zero-temperature fermion Green-function workspace for a composite model with one conserved fermion factor while retaining all spectator-factor truncations."""
function fermion_green_workspace(
    model::ManyBodyModel{R};
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
) where {R<:_CompositeAlgebraRepresentation}
    return _composite_fermion_workspace(
        model;
        ground_sector=ground_sector,
        spin=spin,
        ground_krylov_dim=ground_krylov_dim,
        green_krylov_dim=green_krylov_dim,
        tol=tol,
        ground_reorthogonalize=ground_reorthogonalize,
        green_reorthogonalize=green_reorthogonalize,
        parallel=parallel,
        seed=seed,
        max_basis_dimension=max_basis_dimension,
    )
end

"""Reuse an already solved composite fermion ground sector when constructing zero-temperature particle-addition and particle-removal Green-function sectors."""
function fermion_green_workspace(
    model::ManyBodyModel{R},
    solution::SolverResult;
    spin::Symbol=:up,
    green_krylov_dim::Integer=192,
    tol::Real=1e-10,
    green_reorthogonalize::Bool=false,
    parallel::Bool=true,
    max_basis_dimension::Integer=50_000_000,
) where {R<:_CompositeAlgebraRepresentation}
    ground_sector = _fermion_ground_sector_from_solution(solution)
    iterations = get(solution.metadata, :iterations, nothing)
    reused_krylov_dim = iterations isa Integer ? max(2, Int(iterations)) : 2
    return _composite_fermion_workspace(
        model;
        ground_sector=ground_sector,
        solution=solution,
        spin=spin,
        ground_krylov_dim=reused_krylov_dim,
        green_krylov_dim=green_krylov_dim,
        tol=tol,
        ground_reorthogonalize=false,
        green_reorthogonalize=green_reorthogonalize,
        parallel=parallel,
        seed=0,
        max_basis_dimension=max_basis_dimension,
    )
end
