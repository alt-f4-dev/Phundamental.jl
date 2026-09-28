# Reuse a previously solved canonical fermion ground sector when constructing a zero-temperature Green-function workspace.

function _fermion_ground_sector_from_solution(solution::SolverResult)
    sector = get(solution.metadata, :sector, nothing)
    sector isa FermionSector || throw(ArgumentError("fermion Green-function reuse requires a SolverResult carrying metadata[:sector]::FermionSector"))
    isnothing(sector.nup) && throw(ArgumentError("spin-resolved Green functions require FermionSector(nup=..., ndown=...)"))
    return sector
end

function _fermion_ground_residual_from_solution(solution::SolverResult)
    residuals = get(solution.metadata, :ritz_residuals, nothing)
    !isnothing(residuals) && !isempty(residuals) && return Float64(first(residuals))
    residual = get(solution.metadata, :ground_residual, nothing)
    return isnothing(residual) ? NaN : Float64(residual)
end

"""
    fermion_green_workspace(model, solution; spin=:up, green_krylov_dim=192, tol=1e-10, green_reorthogonalize=false, parallel=true, max_basis_dimension=50_000_000)

Construct a zero-temperature spin-resolved fermion Green-function workspace by reusing the canonical ground state already contained in `solution`. The supplied `SolverResult` must have been obtained from the same `model` in a spin-resolved `FermionSector(nup=..., ndown=...)`; only the particle-addition and particle-removal sectors and their projected resolvents are constructed anew.
"""
function fermion_green_workspace(
    model::ManyBodyModel,
    solution::SolverResult;
    spin::Symbol=:up,
    green_krylov_dim::Integer=192,
    tol::Real=1e-10,
    green_reorthogonalize::Bool=false,
    parallel::Bool=true,
    max_basis_dimension::Integer=50_000_000,
)
    green_krylov_dim > 1 || throw(ArgumentError("green_krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("fermion_green_workspace requires a FermionAlgebra model"))
    solution.model === model || throw(ArgumentError("the supplied SolverResult must belong to the same ManyBodyModel instance"))
    solution.basis isa ComputationalBasis || throw(ArgumentError("fermion Green-function reuse requires an explicit ComputationalBasis"))
    length(solution) > 0 || throw(ArgumentError("the supplied SolverResult contains no eigenstates"))

    ground_sector = _fermion_ground_sector_from_solution(solution)
    sites, modes = _fermion_sites_and_modes(model, spin)
    nsites = length(sites)
    particle_sector = _fermion_target_sector(ground_sector, spin, 1, nsites)
    hole_sector = _fermion_target_sector(ground_sector, spin, -1, nsites)
    tolerance = Float64(tol)
    max_dimension = Int(max_basis_dimension)

    ground_build = @timed _sparse_fermion_sector_hamiltonian(model, ground_sector; max_basis_dimension=max_dimension)
    ground_hamiltonian, ground_data = ground_build.value
    length(ground_data.basis) == length(solution.basis) || throw(DimensionMismatch("reused solution basis dimension does not match the reconstructed ground sector"))
    ground_data.basis.states == solution.basis.states || throw(ArgumentError("reused solution basis ordering does not match the canonical packed-fermion sector ordering"))

    ground_energy = Float64(real(groundenergy(solution)))
    ground_state = ComplexF64.(collect(groundstate(solution)))
    ground_norm = norm(ground_state)
    ground_norm > 0 || throw(ArgumentError("the supplied ground state has zero norm"))
    ground_state ./= ground_norm

    particle_build = @timed _sparse_fermion_sector_hamiltonian(model, particle_sector; max_basis_dimension=max_dimension)
    particle_hamiltonian, particle_data = particle_build.value
    hole_build = @timed _sparse_fermion_sector_hamiltonian(model, hole_sector; max_basis_dimension=max_dimension)
    hole_hamiltonian, hole_data = hole_build.value

    particle_seed_timing = @timed _fermion_transition_seeds(
        ground_state,
        solution.basis,
        particle_data,
        modes;
        create=true,
        parallel=parallel,
    )
    hole_seed_timing = @timed _fermion_transition_seeds(
        ground_state,
        solution.basis,
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
        :ground_residual => _fermion_ground_residual_from_solution(solution),
        :ground_iterations => get(solution.metadata, :iterations, nothing),
        :ground_reused => true,
        :ground_solution_solver => solver_name(solution.solver),
        :green_krylov_dim => Int(green_krylov_dim),
        :green_reorthogonalize => green_reorthogonalize,
        :green_chain_backend => green_reorthogonalize ? :reorthogonalized_scalar : :streaming_scalar_rotating_buffers,
        :parallel => parallel,
        :threads => Threads.nthreads(),
        :ground_build_time => ground_build.time,
        :ground_solve_time => 0.0,
        :particle_build_time => particle_build.time,
        :hole_build_time => hole_build.time,
        :particle_seed_time => particle_seed_timing.time,
        :hole_seed_time => hole_seed_timing.time,
        :particle_chain_time => particle_chain_timing.time,
        :hole_chain_time => hole_chain_timing.time,
        :ground_build_bytes => ground_build.bytes,
        :ground_solve_bytes => 0,
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
        solution.basis,
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
