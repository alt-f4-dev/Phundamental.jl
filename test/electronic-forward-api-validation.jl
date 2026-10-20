using Test
using LinearAlgebra

function _electronic_dimer_fixture()
    lattice = BravaisLattice(reshape([2.0], 1, 1))
    basis = BasisSite[
        BasisSite(:H1, :H, [0.0]),
        BasisSite(:H2, :H, [0.5]),
    ]
    crystal = LatticeCrystalStructure(lattice, basis)
    cluster = Supercell(crystal, (1,); periodic=false)
    material = Material(
        "Carrascal symmetric Hubbard dimer",
        crystal;
        species=Dict{Symbol,Any}(:H => lookup_species(:H)),
        metadata=Dict(
            :reference => "Carrascal et al., J. Phys.: Condens. Matter 27, 393001 (2015)",
            :doi => "10.1088/0953-8984/27/39/393001",
        ),
    )
    return material, cluster
end

function _manual_local_spectrum(model, sector, axis; spin=:up, site=1, eta=0.05)
    workspace = fermion_green_workspace(
        model;
        ground_sector=sector,
        spin=spin,
        ground_krylov_dim=4,
        green_krylov_dim=2,
        tol=1e-12,
        ground_reorthogonalize=true,
        green_reorthogonalize=true,
        parallel=false,
        seed=17,
    )
    green = cluster_green_matrix(workspace, axis; eta=eta, parallel=false)
    site_row = findfirst(==(site), workspace.sites)
    isnothing(site_row) && error("requested manual-spectrum site is absent from the workspace")
    values = -imag.(vec(green[site_row, site_row, :])) ./ π
    intensity = reshape(values, 1, length(values))
    return LocalSpectrumResult([site], collect(axis), intensity, Dict{Symbol,Any}(:source => :manual_low_level))
end

@testset "Electronic high-level forward API" begin
    material, cluster = _electronic_dimer_fixture()
    t = 0.5
    sector = FermionSector(nup=1, ndown=1)

    shell_model = build_model(material, cluster, HubbardModel(t, 1.0; shell=1))
    legacy_model = build_model(material, cluster, HubbardModel(t, 1.0; neighbor_cutoff=1.0 + 1e-8, max_shell=1))

    @test length(shell_model.parameters[:bonds]) == 1
    @test shell_model.parameters[:bonds][1].shell == 1
    @test_throws ArgumentError HubbardModel(t, 1.0)
    @test_throws ArgumentError HubbardModel(t, 1.0; shell=1, neighbor_cutoff=1.1)
    @test_throws ArgumentError HubbardModel(t, 1.0; shell=1, max_shell=1)

    shell_solution = solve(ExactDiagonalization(), shell_model)
    legacy_solution = solve(ExactDiagonalization(), legacy_model)
    @test isapprox(shell_solution.energies, legacy_solution.energies; rtol=1e-12, atol=1e-12)

    for U in (1.0, 5.0)
        model = build_model(material, cluster, HubbardModel(t, U; shell=1))
        axis = U == 1.0 ? collect(range(-2.0, 3.0; length=1001)) : collect(range(-1.0, 6.0; length=1401))
        eta = 0.05

        solver = OptimizedLanczos(
            nev=1,
            krylov_dim=4,
            tol=1e-12,
            sector=sector,
            backend=:fermion,
            reorthogonalize=true,
            seed=17,
        )
        request = LocalElectronSpectralFunctionRequest(
            axis;
            spin=:up,
            sites=[1],
            eta=eta,
            normalization=:spectral_density,
            green_krylov_dim=2,
            tol=1e-12,
            green_reorthogonalize=true,
            parallel=false,
        )

        calculation = forward(ForwardProblem(model; solver=solver, observable=request))
        manual = _manual_local_spectrum(model, sector, axis; spin=:up, site=1, eta=eta)

        @test calculation.observable_result isa LocalSpectrumResult
        @test calculation.result === calculation.observable_result
        @test calculation.result.sites == [1]
        @test calculation.result.metadata[:ground_solution_reused] === true
        @test calculation.result.metadata[:workspace_metadata][:ground_solve_time] == 0.0
        @test Phundamental.Observables.validate_observable_equivalence(calculation.result, manual; rtol=1e-10, atol=1e-12)

        weights = Phundamental.Observables.integrated_spectral_weight(calculation.result)
        @test length(weights) == 1
        @test real(weights[1]) > 0.95
        @test real(weights[1]) < 1.0

        resolution = GaussianResolution(0.08)
        resolved = forward(ForwardProblem(model; solver=solver, observable=request, resolution=resolution))
        manual_resolved = convolve_resolution(resolution, manual)

        @test resolved.result isa LocalSpectrumResult
        @test resolved.result.metadata[:resolution_applied] === true
        @test Phundamental.Observables.validate_observable_equivalence(resolved.result, manual_resolved; rtol=1e-10, atol=1e-12)
    end

    @testset "F7 local-electronic experimental contract" begin
        model = build_model(material, cluster, HubbardModel(t, 1.0; shell=1))
        axis = collect(range(-2.0, 3.0; length=301))
        solver = OptimizedLanczos(nev=1, krylov_dim=4, tol=1e-12, sector=sector,
                                  backend=:fermion, reorthogonalize=true, seed=17)
        request = LocalElectronSpectralFunctionRequest(axis; spin=:up, sites=[1], eta=0.05,
            normalization=:spectral_density, green_krylov_dim=2, tol=1e-12,
            green_reorthogonalize=true, parallel=false)
        calculation = forward(ForwardProblem(model; solver=solver, observable=request))
        prediction = portable_prediction(calculation; stage=:intrinsic, coordinate_frame=:site,
                                         coordinate_unit="site_index", spectral_unit="eV", intensity_unit="states/eV",
                                         normalization="local", spectral_kind=:energy)
        @test prediction.grid.coordinate_role === :site
        @test prediction.grid.coordinate_frame === :site
        @test import_result(export_result(prediction)).intensity ≈ prediction.intensity
        measured = experimental_dataset(prediction.grid, Float64.(real.(prediction.intensity));
                                        uncertainty=fill(0.04, size(prediction.intensity)),
                                        intensity_unit="states/eV", normalization="local")
        @test isapprox(compare_measurement(prediction, measured).chi_squared, 0.0; atol=1e-10)
        @test validate_forward_equivalence(calculation, calculation; compare=(:intrinsic, :instrument)).valid
        @test !solver_compatibility(DMFTSolver(), model, request).compatible
    end

end
