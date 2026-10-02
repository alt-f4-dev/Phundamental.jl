using Phundamental
using Test
using LinearAlgebra

function _holstein_forward_fixture(; cutoff::Integer=6, coupling::Real=0.5)
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    crystal = LatticeCrystalStructure(lattice, [BasisSite(:X, :X, [0.0])])
    material = Material("Holstein dimer forward validation", crystal; species=Dict(:X => AtomicSpecies(:X; mass=1.0)))
    cluster = Supercell(crystal, [2]; periodic=false)
    model = build_model(material, cluster, HolsteinModel(1.0, 0.0, 1.0, coupling; phonon_cutoff=cutoff, shell=1))
    return model
end

function _holstein_dense_local_spectrum(model, sector::FermionSector, axis; spin::Symbol=:up, site::Integer=1, eta::Real=0.03)
    ground_basis = computational_basis(model, sector)
    ground_hamiltonian = hamiltonian_matrix(model; basis=ground_basis, sparse=false).matrix
    ground_system = eigen(Hermitian(ground_hamiltonian))
    ground_energy = Float64(ground_system.values[1])
    ground_state = ComplexF64.(ground_system.vectors[:, 1])

    if spin === :up
        particle_sector = FermionSector(nup=sector.nup + 1, ndown=sector.ndown)
        hole_sector = FermionSector(nup=sector.nup - 1, ndown=sector.ndown)
    elseif spin === :down
        particle_sector = FermionSector(nup=sector.nup, ndown=sector.ndown + 1)
        hole_sector = FermionSector(nup=sector.nup, ndown=sector.ndown - 1)
    else
        throw(ArgumentError("spin must be :up or :down"))
    end

    particle_basis = computational_basis(model, particle_sector)
    hole_basis = computational_basis(model, hole_sector)
    particle_hamiltonian = hamiltonian_matrix(model; basis=particle_basis, sparse=false).matrix
    hole_hamiltonian = hamiltonian_matrix(model; basis=hole_basis, sparse=false).matrix
    particle_system = eigen(Hermitian(particle_hamiltonian))
    hole_system = eigen(Hermitian(hole_hamiltonian))

    mode = model.parameters[:fermion_mode_map][(Int(site), spin)]
    particle_seed = fermion_transition_vector(ground_state, ground_basis, particle_basis, mode; create=true)
    hole_seed = fermion_transition_vector(ground_state, ground_basis, hole_basis, mode; create=false)
    particle_weights = abs2.(adjoint(particle_system.vectors) * particle_seed)
    hole_weights = abs2.(adjoint(hole_system.vectors) * hole_seed)
    particle_centers = Float64.(particle_system.values .- ground_energy)
    hole_centers = Float64.(-(hole_system.values .- ground_energy))

    centers = vcat(hole_centers, particle_centers)
    weights = vcat(hole_weights, particle_weights)
    intensity = zeros(Float64, length(axis))
    for (center, weight) in zip(centers, weights)
        intensity .+= weight .* (eta / π) ./ ((axis .- center).^2 .+ eta^2)
    end
    return (ground_energy=ground_energy, centers=centers, weights=weights, particle_weights=particle_weights, hole_weights=hole_weights, intensity=intensity)
end

@testset "Electron-phonon high-level forward API" begin
    cutoff = 6
    model = _holstein_forward_fixture(cutoff=cutoff)
    sector = FermionSector(nup=1, ndown=1)

    sector_basis = computational_basis(model, sector)
    @test length(sector_basis) == 4 * (cutoff + 1)^2
    solver = OptimizedLanczos(
        nev=1,
        krylov_dim=4 * (cutoff + 1)^2,
        tol=1e-12,
        sector=sector,
        backend=:fermion,
        reorthogonalize=true,
        seed=17,
    )
    axis = collect(range(-10.0, 10.0; length=2001))
    eta = 0.03
    request = LocalElectronSpectralFunctionRequest(
        axis;
        spin=:up,
        sites=[1, 2],
        eta=eta,
        normalization=:spectral_density,
        green_krylov_dim=2 * (cutoff + 1)^2,
        tol=1e-12,
        green_reorthogonalize=true,
        parallel=false,
    )

    calculation = forward(ForwardProblem(model; solver=solver, observable=request))
    result = calculation.result

    @test result isa LocalSpectrumResult
    @test calculation.solution.metadata[:backend] === :composite_fermion_sparse
    @test calculation.solution.metadata[:sector].nup == 1
    @test calculation.solution.metadata[:sector].ndown == 1
    @test length(calculation.solution.basis) == 4 * (cutoff + 1)^2
    @test result.metadata[:ground_solution_reused] === true
    @test result.metadata[:workspace_metadata][:ground_solve_time] == 0.0
    @test result.metadata[:workspace_metadata][:ground_dimension] == 4 * (cutoff + 1)^2
    @test result.metadata[:workspace_metadata][:particle_dimension] == 2 * (cutoff + 1)^2
    @test result.metadata[:workspace_metadata][:hole_dimension] == 2 * (cutoff + 1)^2

    reference_1 = _holstein_dense_local_spectrum(model, sector, axis; spin=:up, site=1, eta=eta)
    reference_2 = _holstein_dense_local_spectrum(model, sector, axis; spin=:up, site=2, eta=eta)

    @test isapprox(reference_1.ground_energy, -2.593448076046033; rtol=0.0, atol=1e-10)
    @test isapprox(groundenergy(calculation.solution), reference_1.ground_energy; rtol=0.0, atol=1e-10)
    @test isapprox(result.intensity[1, :], reference_1.intensity; rtol=1e-7, atol=1e-9)
    @test isapprox(result.intensity[2, :], reference_2.intensity; rtol=1e-7, atol=1e-9)
    @test maximum(abs.(result.intensity[1, :] .- result.intensity[2, :])) <= 1e-10
    @test isapprox(sum(reference_1.weights), 1.0; rtol=0.0, atol=1e-11)
    @test isapprox(sum(reference_1.hole_weights), 0.5; rtol=0.0, atol=1e-11)
    @test isapprox(sum(reference_1.particle_weights), 0.5; rtol=0.0, atol=1e-11)
    @test isapprox(sum(reference_2.weights), 1.0; rtol=0.0, atol=1e-11)

    low_level_workspace = fermion_green_workspace(
        model;
        ground_sector=sector,
        spin=:up,
        ground_krylov_dim=4 * (cutoff + 1)^2,
        green_krylov_dim=2 * (cutoff + 1)^2,
        tol=1e-12,
        ground_reorthogonalize=true,
        green_reorthogonalize=true,
        parallel=false,
        seed=17,
    )
    @test low_level_workspace.metadata[:ground_reused] === false
    @test low_level_workspace.metadata[:ground_dimension] == 4 * (cutoff + 1)^2
    @test isapprox(low_level_workspace.ground_energy, reference_1.ground_energy; rtol=0.0, atol=1e-10)

    generic_sector = QuantumNumberSector(:nup => 1, :ndown => 1)
    generic_solver = OptimizedLanczos(nev=1, krylov_dim=4 * (cutoff + 1)^2, tol=1e-12, sector=generic_sector, backend=:fermion, reorthogonalize=true, seed=17)
    generic_solution = solve(generic_solver, model)
    @test generic_solution.metadata[:sector] isa FermionSector
    @test isapprox(groundenergy(generic_solution), reference_1.ground_energy; rtol=0.0, atol=1e-10)
end
