using Test
using LinearAlgebra
using Phundamental

function _batch5_hubbard_fixture()
    lattice = BravaisLattice(reshape([2.0], 1, 1))
    basis = BasisSite[
        BasisSite(:H1, :H, [0.0]),
        BasisSite(:H2, :H, [0.5]),
    ]
    crystal = LatticeCrystalStructure(lattice, basis)
    cluster = Supercell(crystal, (1,); periodic=false)
    material = Material("Batch 5 Hubbard reference", crystal; species=Dict{Symbol,Any}(:H => lookup_species(:H)))
    model = build_model(material, cluster, HubbardModel(1.0, 0.0; shell=1))
    return model
end

function _batch5_spin_fixture()
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    properties = SiteProperties(spin=0.5, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0]; properties=properties)])
    cluster = Supercell(crystal, (2,); periodic=false)
    material = Material("Batch 5 NLCE reference", crystal; species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    model = build_model(material, cluster, HeisenbergModel(1.0; shell=1))
    return model
end

@testset "High-level embedding and NLCE forward integration" begin
    hubbard = _batch5_hubbard_fixture()
    beta = 2.0
    fermionic_axis = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1, 2])
    momentum_lattice = periodic_momentum_lattice((2, 2), k -> -2 * (cos(k[1]) + cos(k[2])))
    dmft_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=0, backend=:dense), mixing=LinearMixing(1.0), tol=1e-10, maxiter=3)
    dmft_context = DMFTForwardContext(momentum_lattice; temperature=inv(beta), axis=fermionic_axis)
    green_request = LocalMatsubaraGreenFunctionRequest(metadata=Dict(:temperature => inv(beta)))
    self_energy_request = LocalMatsubaraSelfEnergyRequest(metadata=Dict(:temperature => inv(beta)))
    @test request_requirements(green_request).frequency_domain == :matsubara
    @test request_requirements(green_request).statistics == :fermionic
    @test request_requirements(green_request).momentum_scope == :local
    @test request_requirements(self_energy_request).response_kind == :self_energy

    @test !solver_compatibility(dmft_solver, hubbard, green_request).compatible
    @test solver_compatibility(dmft_solver, hubbard, green_request, dmft_context).compatible
    @test solver_compatibility(dmft_solver, hubbard, green_request, dmft_context).approximation == :single_site_dmft

    green_forward = forward(ForwardProblem(hubbard; solver=dmft_solver, method_context=dmft_context, observable=green_request))
    @test green_forward.solution isa DMFTResult
    @test green_forward.solution.converged
    @test green_forward.observable_result isa GreenFunction
    @test green_forward.result === green_forward.observable_result
    @test frequency_domain(green_forward.observable_result) == :matsubara
    @test requires_analytic_continuation(green_forward.observable_result)
    @test green_forward.solution.metadata[:forward_route] == :single_site_dmft

    self_energy_forward = forward(ForwardProblem(hubbard; solver=dmft_solver, method_context=dmft_context, observable=self_energy_request))
    @test self_energy_forward.observable_result isa SelfEnergy
    @test maximum(abs.(self_energy_forward.observable_result.values)) < 1e-8

    bosonic_axis = BosonicMatsubaraAxis(beta, [0])
    impurity_backend = TauGridFourPoint(ntime=16, max_dimension=16, algorithm=:factored, parallel=:serial)
    dga_context = LadderDGAForwardContext(momentum_lattice, bosonic_axis; dmft_solver=dmft_solver, impurity_backend=impurity_backend,
                                         fermionic_axis=fermionic_axis)
    dga_solver = DiagrammaticDMFTSolver(method=LadderDGA(pseudoinverse=true, reconstruct_self_energy=false),
                                        execution=DiagrammaticExecutionPolicy(parallel=:serial))
    susceptibility_request = MatsubaraSusceptibilityRequest(:charge; metadata=Dict(:temperature => inv(beta)))
    @test request_requirements(susceptibility_request).frequency_domain == :matsubara
    @test request_requirements(susceptibility_request).statistics == :bosonic
    @test request_requirements(susceptibility_request).operator_channels == (:charge,)

    @test !solver_compatibility(dga_solver, hubbard, susceptibility_request).compatible
    @test solver_compatibility(dga_solver, hubbard, susceptibility_request, dga_context).compatible
    @test solver_compatibility(dga_solver, hubbard, susceptibility_request, dga_context).approximation == :one_shot_ladder_dga
    opaque_dga_request = ObservableRequest(:opaque_dga, (_, solution) -> solution)
    @test !solver_compatibility(dga_solver, hubbard, opaque_dga_request, dga_context).compatible

    dga_forward = forward(ForwardProblem(hubbard; solver=dga_solver, method_context=dga_context, observable=susceptibility_request))
    @test dga_forward.solution isa LadderDGAResult
    @test dga_forward.observable_result isa SusceptibilityResult
    @test dga_forward.observable_result.metadata[:channel] == :charge
    @test frequency_domain(dga_forward.observable_result) == :matsubara
    @test requires_analytic_continuation(dga_forward.observable_result)
    @test dga_forward.solution.metadata[:dmft_reference_preserved]

    @test_throws ArgumentError forward(ForwardProblem(hubbard; transformation=ParticleHoleTransformation(), solver=dmft_solver,
                                                       method_context=dmft_context, observable=green_request))

    spin_model = _batch5_spin_fixture()
    nlce_solver = NLCESolver()
    ground_request = ObservableRequest(:ground_energy, (_, solution) -> groundenergy(solution); metadata=Dict(:response_kind => :ground_state_scalar))
    nlce_context = NLCEForwardContext(2; inner_solver=ExactDiagonalization())
    @test !solver_compatibility(nlce_solver, spin_model, ground_request).compatible
    @test solver_compatibility(nlce_solver, spin_model, ground_request, nlce_context).compatible
    @test solver_compatibility(nlce_solver, spin_model, ground_request, nlce_context).approximation == :finite_order_nlce

    nlce_forward = forward(ForwardProblem(spin_model; solver=nlce_solver, method_context=nlce_context, observable=ground_request))
    @test nlce_forward.solution isa NLCEResult
    @test nlce_forward.observable_result == bulk_estimate(nlce_forward.solution)
    @test nlce_forward.result == nlce_forward.observable_result
    @test nlce_forward.solution.metadata[:forward_route] == :finite_reference_nlce
    @test nlce_forward.solution.metadata[:observable_request] == :ground_energy
    @test nlce_forward.solution.metadata[:inner_solver] == :exact_diagonalization
    @test correlation_scope(nlce_forward.solution) == :short_intermediate_spatial
end
