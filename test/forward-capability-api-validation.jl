using Test
using LinearAlgebra
using Phundamental

const FFC = Phundamental

@testset "Forward request and solver compatibility API" begin
    @testset "Declarative request requirements" begin
        representation = FFC.Representation(:request_requirements_spin, FFC.SpinHilbertSpace([1//2, 1//2]), FFC.SpinAlgebra(2), FFC.SpinProductBasis([1, 2]))
        hamiltonian = 0.7 * FFC.Sz(1) + 0.3 * FFC.Sx(1) * FFC.Sx(2)
        observables = Dict{Symbol,Any}(:sx_1 => FFC.Sx(1), :sx_2 => FFC.Sx(2))
        model = FFC.ManyBodyModel(representation, hamiltonian; observables=observables)
        field = FFC.MomentumField([:sx_1, :sx_2], [0.0, 1.0]; normalization=:sqrtN)
        request = FFC.DynamicalCorrelationRequest(field, [0.0, π], collect(range(-2.0, 2.0; length=101)); metadata=Dict(:channel => :xx))

        requirements = FFC.request_requirements(request)
        @test requirements.request_kind == :dynamical_correlation
        @test requirements.response_kind == :correlation_spectrum
        @test requirements.frequency_domain == :real
        @test requirements.momentum_scope == :momentum_resolved
        @test requirements.response_order == 2
        @test requirements.temperature == 0.0
        @test FFC.is_dynamical_request(requirements)

        solver = FFC.ExactDiagonalization()
        report = FFC.solver_compatibility(solver, model, request)
        @test report.model_supported
        @test report.request_supported
        @test report.compatible
        @test report.certified
        @test report.approximation == :none
        @test FFC.supports(solver, model)
        @test FFC.supports(solver, model, request)

        wrong_sector_solver = FFC.OptimizedLanczos(nev=1, krylov_dim=4, sector=FFC.FermionSector(nparticles=1), backend=:spin)
        wrong_sector_report = FFC.solver_compatibility(wrong_sector_solver, model, request)
        @test !wrong_sector_report.model_supported
        @test !wrong_sector_report.compatible

        custom = FFC.ObservableRequest(:ground_energy, (_, solution) -> FFC.groundenergy(solution); metadata=Dict(:response_kind => :ground_state_scalar))
        custom_report = FFC.solver_compatibility(solver, model, custom)
        @test custom_report.compatible
        @test !custom_report.certified
        @test FFC.request_requirements(custom).response_kind == :ground_state_scalar
    end

    @testset "Result-family compatibility is explicit" begin
        boson_representation = FFC.Representation(:compatibility_boson, FFC.BosonFockSpace(1), FFC.BosonAlgebra(1), FFC.BosonOccupationBasis([1]); ordering=[1], reference_state=:boson_vacuum)
        boson_hamiltonian = 1.4 * FFC.nb(1) + 0.2 * FFC.b(1)' * FFC.b(1)' + 0.2 * FFC.b(1) * FFC.b(1)
        boson_model = FFC.ManyBodyModel(boson_representation, boson_hamiltonian)
        boson_field = FFC.MomentumField([FFC.b(1) + FFC.b(1)'], [0.0]; normalization=:none)
        boson_request = FFC.DynamicalCorrelationRequest(boson_field, [0.0], collect(range(-2.0, 2.0; length=101)); convention=FFC.SpectrumConvention(fourier_normalization=:none))
        boson_solver = FFC.QuadraticBosonSolver(tol=1e-12)

        @test FFC.supports(boson_solver, boson_model)
        @test FFC.supports(boson_solver, boson_model, boson_request)

        fermion_representation = FFC.Representation(:compatibility_fermion, FFC.FermionFockSpace(1), FFC.FermionAlgebra(1), FFC.FermionOccupationBasis([1]); ordering=[1])
        fermion_model = FFC.ManyBodyModel(fermion_representation, 0.8 * FFC.nf(1))
        fermion_field = FFC.MomentumField([FFC.c(1) + FFC.c(1)'], [0.0]; normalization=:none)
        fermion_request = FFC.DynamicalCorrelationRequest(fermion_field, [0.0], collect(range(-2.0, 2.0; length=101)); convention=FFC.SpectrumConvention(fourier_normalization=:none))
        fermion_solver = FFC.QuadraticFermionSolver(tol=1e-12)

        @test FFC.supports(fermion_solver, fermion_model)
        @test !FFC.supports(fermion_solver, fermion_model, fermion_request)
        @test !FFC.supports(FFC.AutoSolver(tol=1e-12), fermion_model, fermion_request)
    end

    @testset "Approximation identity and caller-supplied candidate filtering" begin
        lattice = FFC.BravaisLattice(Matrix{Float64}(I, 3, 3))
        properties = FFC.SiteProperties(spin=0.5, g_tensor=2.0)
        crystal = FFC.LatticeCrystalStructure(lattice, FFC.BasisSite[FFC.BasisSite(:Cu, :Cu, [0.0, 0.0, 0.0]; properties=properties)])
        parent = FFC.Supercell(crystal, (1, 1, 1); periodic=true)
        material = FFC.Material("compatibility spin reference", crystal; species=Dict{Symbol,Any}(:Cu => FFC.lookup_species(:Cu)), basis_properties=[properties])
        cluster = FFC.bond_cluster(parent, first(FFC.neighbor_bonds(parent; cutoff=1.01, max_shell=1)))
        model = FFC.build_model(material, cluster, FFC.HeisenbergModel(1.0; shell=1))
        request = FFC.SpinTensorStructureFactorRequest([0.0, 0.0, 0.0], collect(range(-2.0, 2.0; length=101)))

        ed = FFC.ExactDiagonalization()
        lswt = FFC.LinearSpinWaveSolver()
        harmonic = FFC.HarmonicPhononSolver()
        lswt_report = FFC.solver_capabilities(lswt, model, request)

        @test lswt_report.compatible
        @test lswt_report.approximation == :linear_spin_wave
        @test FFC.supports(ed, model, request)
        @test !FFC.supports(harmonic, model, request)

        candidates = FFC.compatible_solvers(model, request, (harmonic, ed, lswt))
        @test candidates == [ed, lswt]
        @test_throws ArgumentError FFC.forward(FFC.ForwardProblem(model; solver=harmonic, observable=request))
    end

    @testset "Request-aware harmonic input construction remains explicit" begin
        lattice = FFC.BravaisLattice(reshape([1.0], 1, 1))
        basis = FFC.BasisSite[FFC.BasisSite(:M, :M, [0.0]), FFC.BasisSite(:m, :m, [0.5])]
        crystal = FFC.LatticeCrystalStructure(lattice, basis)
        species = Dict(:M => FFC.AtomicSpecies(:M; mass=2.0, nuclear_scattering_length=1.0), :m => FFC.AtomicSpecies(:m; mass=1.0, nuclear_scattering_length=1.0))
        material = FFC.Material("compatibility phonon reference", crystal; species=species)
        interactions = FFC.HarmonicBondInteraction[FFC.HarmonicBondInteraction(1, 2, [0]; longitudinal=4.0), FFC.HarmonicBondInteraction(1, 2, [-1]; longitudinal=4.0)]
        model = FFC.harmonic_model(material, interactions)
        request = FFC.PhononStructureFactorRequest([0.25, 0.75], collect(range(-4.0, 4.0; length=201)); convention=FFC.SpectrumConvention(spectral_axis=:omega))
        solver = FFC.HarmonicPhononSolver()

        report = FFC.solver_compatibility(solver, model, request)
        @test report.compatible
        @test report.certified
        @test !FFC.supports(FFC.ExactDiagonalization(), model, request)

        result = FFC.forward(FFC.ForwardProblem(model; solver=solver, observable=request))
        @test result.solution isa FFC.PhononDispersionResult
        @test result.observable_result isa FFC.PhononStructureFactorResult
    end
end
