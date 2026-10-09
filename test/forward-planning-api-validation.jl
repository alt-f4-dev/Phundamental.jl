using Test
using LinearAlgebra
using Phundamental

function _batch6_spin_fixture()
    lattice = BravaisLattice(Matrix{Float64}(I, 3, 3))
    properties = SiteProperties(spin=0.5, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0, 0.0, 0.0]; properties=properties)])
    parent = Supercell(crystal, (1, 1, 1); periodic=true)
    material = Material("Batch 6 spin reference", crystal; species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    cluster = bond_cluster(parent, first(neighbor_bonds(parent; cutoff=1.01, max_shell=1)))
    return build_model(material, cluster, HeisenbergModel(1.0; shell=1))
end

function _batch6_hubbard_fixture()
    lattice = BravaisLattice(reshape([2.0], 1, 1))
    basis = BasisSite[BasisSite(:H1, :H, [0.0]), BasisSite(:H2, :H, [0.5])]
    crystal = LatticeCrystalStructure(lattice, basis)
    cluster = Supercell(crystal, (1,); periodic=false)
    material = Material("Batch 6 Hubbard reference", crystal; species=Dict{Symbol,Any}(:H => lookup_species(:H)))
    return build_model(material, cluster, HubbardModel(1.0, 0.0; shell=1))
end

@testset "Policy-governed forward execution planning" begin
    @testset "Exact specialization is deterministic and inspectable" begin
        representation = Representation(:batch6_quadratic_boson, BosonFockSpace(1), BosonAlgebra(1), BosonOccupationBasis([1]); ordering=[1], reference_state=:boson_vacuum)
        hamiltonian = 1.4 * nb(1) + 0.2 * b(1)' * b(1)' + 0.2 * b(1) * b(1)
        model = ManyBodyModel(representation, hamiltonian)
        field = MomentumField([b(1) + b(1)'], [0.0]; normalization=:none)
        request = DynamicalCorrelationRequest(field, [0.0], collect(range(-2.0, 2.0; length=201)); convention=SpectrumConvention(fourier_normalization=:none))

        plan = plan_forward(model, request)
        report = planning_report(plan)
        @test plan.solver isa QuadraticBosonSolver
        @test report.selected_solver == :quadratic_boson
        @test report.selected_label == :quadratic_boson
        @test isempty(report.selected_approximations)
        @test report.selected_certified
        @test plan.metadata[:forward_planned]

        result = forward(plan)
        @test result.solution isa QuadraticModeResult
        @test result.solution.metadata[:forward_planned]
        @test result.solution.metadata[:planned_solver] == :quadratic_boson
        @test result.observable_result.metadata[:forward_planned]
        @test result.observable_result.metadata[:planned_approximations] == ()

        convenience = forward(model, request)
        @test convenience.solution isa QuadraticModeResult
        @test convenience.solution.metadata[:planned_solver] == :quadratic_boson
    end

    @testset "Exact routes outrank authorized physical approximations" begin
        model = _batch6_spin_fixture()
        request = SpinTensorStructureFactorRequest([0.0, 0.0, 0.0], collect(range(-2.0, 2.0; length=101)))
        policy = ForwardExecutionPolicy(allowed_approximations=(:linear_spin_wave,), include_default_exact_routes=false)
        routes = (ForwardRoute(LinearSpinWaveSolver()), ForwardRoute(ExactDiagonalization()))
        plan = plan_forward(model, request; policy=policy, routes=routes)
        @test plan.solver isa ExactDiagonalization
        @test planning_report(plan).selected_approximations == ()

        approximate_only = ForwardExecutionPolicy(allowed_approximations=(), include_default_exact_routes=false)
        error = try
            plan_forward(model, request; policy=approximate_only, routes=(ForwardRoute(LinearSpinWaveSolver()),))
            nothing
        catch caught
            caught
        end
        @test error isa ForwardPlanningError
        @test planning_report(error).assessments[1].compatible
        @test !planning_report(error).assessments[1].permitted
        @test :linear_spin_wave in planning_report(error).assessments[1].approximations
    end

    @testset "Exact representation transformations may be selected explicitly" begin
        representation = Representation(:batch6_spin_transform, SpinHilbertSpace([1//2, 1//2]), SpinAlgebra(2), SpinProductBasis([1, 2]))
        hamiltonian = 0.7 * Sz(1) - 0.2 * Sx(2) + 0.4 * (Sx(1) * Sx(2) + Sy(1) * Sy(2) + Sz(1) * Sz(2))
        observables = Dict{Symbol,Any}(:sx_1 => Sx(1), :sx_2 => Sx(2))
        model = ManyBodyModel(representation, hamiltonian; observables=observables)
        field = MomentumField([:sx_1, :sx_2], [0.0, 1.0]; normalization=:sqrtN)
        request = DynamicalCorrelationRequest(field, [0.0, π], collect(range(-2.0, 2.0; length=201)); convention=SpectrumConvention(fourier_normalization=:sqrtN))
        policy = ForwardExecutionPolicy(include_default_exact_routes=false)
        route = ForwardRoute(ExactDiagonalization(); transformation=JordanWigner(1:2), label=:jordan_wigner_ed)
        plan = plan_forward(model, request; policy=policy, routes=(route,))
        report = planning_report(plan)
        @test report.selected_label == :jordan_wigner_ed
        @test report.assessments[1].transformation == :jordan_wigner
        @test isempty(report.selected_approximations)
        @test plan.transformation isa JordanWigner
    end

    @testset "Uncertified opaque routes are policy-controlled" begin
        model = _batch6_spin_fixture()
        request = ObservableRequest(:opaque_energy, (_, solution) -> groundenergy(solution))
        route = ForwardRoute(ExactDiagonalization())
        permissive = ForwardExecutionPolicy(require_certified=false, include_default_exact_routes=false)
        plan = plan_forward(model, request; policy=permissive, routes=(route,))
        @test !planning_report(plan).selected_certified

        certified_only = ForwardExecutionPolicy(require_certified=true, include_default_exact_routes=false)
        @test_throws ForwardPlanningError plan_forward(model, request; policy=certified_only, routes=(route,))
    end

    @testset "Embedding approximations require explicit authorization and context" begin
        model = _batch6_hubbard_fixture()
        beta = 2.0
        fermionic_axis = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1, 2])
        lattice = periodic_momentum_lattice((2, 2), k -> -2 * (cos(k[1]) + cos(k[2])))
        dmft_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=0, backend=:dense), mixing=LinearMixing(1.0), tol=1e-10, maxiter=3)
        dmft_context = DMFTForwardContext(lattice; temperature=inv(beta), axis=fermionic_axis)
        request = LocalMatsubaraGreenFunctionRequest(metadata=Dict(:temperature => inv(beta)))
        route = ForwardRoute(dmft_solver; method_context=dmft_context)

        default_policy = ForwardExecutionPolicy(include_default_exact_routes=false)
        @test_throws ForwardPlanningError plan_forward(model, request; policy=default_policy, routes=(route,))

        dmft_policy = ForwardExecutionPolicy(allowed_approximations=(:single_site_dmft,), include_default_exact_routes=false)
        plan = plan_forward(model, request; policy=dmft_policy, routes=(route,))
        @test plan.solver === dmft_solver
        @test planning_report(plan).selected_approximations == (:single_site_dmft,)

        missing_context = ForwardRoute(dmft_solver)
        @test_throws ForwardPlanningError plan_forward(model, request; policy=dmft_policy, routes=(missing_context,))

        bosonic_axis = BosonicMatsubaraAxis(beta, [0])
        dga_context = LadderDGAForwardContext(lattice, bosonic_axis; dmft_solver=dmft_solver, impurity_backend=TauGridFourPoint(ntime=16, max_dimension=16, algorithm=:factored, parallel=:serial), fermionic_axis=fermionic_axis)
        dga_solver = DiagrammaticDMFTSolver(method=LadderDGA(pseudoinverse=true, reconstruct_self_energy=false), execution=DiagrammaticExecutionPolicy(parallel=:serial))
        susceptibility = MatsubaraSusceptibilityRequest(:charge; metadata=Dict(:temperature => inv(beta)))
        dga_route = ForwardRoute(dga_solver; method_context=dga_context)
        dga_only = ForwardExecutionPolicy(allowed_approximations=(:one_shot_ladder_dga,), include_default_exact_routes=false)
        @test_throws ForwardPlanningError plan_forward(model, susceptibility; policy=dga_only, routes=(dga_route,))

        complete_dga = ForwardExecutionPolicy(allowed_approximations=(:single_site_dmft, :one_shot_ladder_dga), include_default_exact_routes=false)
        dga_plan = plan_forward(model, susceptibility; policy=complete_dga, routes=(dga_route,))
        @test planning_report(dga_plan).selected_approximations == (:single_site_dmft, :one_shot_ladder_dga)
    end
end
