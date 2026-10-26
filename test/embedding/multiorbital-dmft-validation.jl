#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiOrbitalDMFTValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _free_interaction(labels)
    return P.SpinOrbitalInteraction(zeros(ComplexF64, length(labels), length(labels), length(labels), length(labels)); labels=labels)
end

function _matrix_lattice(blocks, labels)
    return P.MatrixMomentumLattice(blocks; labels=labels)
end

function _dmft_solver(; bath_sites=0, tol=1e-9, maxiter=5)
    impurity = P.MultiOrbitalEDSolver(bath_sites=bath_sites, backend=:sectorized, max_full_dimension=4096,
                                      max_sector_dimension=4096, max_eigensystem_bytes=100_000_000)
    return P.MultiOrbitalDMFTSolver(impurity_solver=impurity, mixing=P.LinearMixing(1.0), tol=tol, maxiter=maxiter)
end

@testset "Matrix DMFT: noninteracting complex hopping and independent lattice Dyson" begin
    names = [:a, :b]
    h0 = ComplexF64[0.15 0.09+0.11im; 0.09-0.11im -0.2]
    t = ComplexF64[0.3 0.04im; -0.04im -0.25]
    lattice = _matrix_lattice(cat(h0+t, h0-t; dims=3), names)
    axis = P.FermionicMatsubaraAxis(8.0, 12; include_negative=true)
    interaction = _free_interaction(names)
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=1/8, axis=axis, one_body=h0)
    @test problem.one_body ≈ h0 atol=1e-14
    @test problem.labels == names
    sigma = P.zero_self_energy(axis, 2; labels=names)
    local_ref = P.lattice_green(lattice, axis, sigma)
    local_streamed = P.Embedding._multi_dmft_local_green(problem, sigma)
    @test local_ref.values ≈ local_streamed.values atol=1e-12 rtol=1e-12
    weiss = P.weiss_green(local_streamed, sigma)
    target = P.hybridization_from_weiss(weiss, h0)
    for i in 1:length(axis)
        direct = (axis[i] * Matrix{ComplexF64}(I, 2, 2) - h0 - inv(weiss[i]))
        @test target[i] ≈ direct atol=1e-12
    end
    solver = _dmft_solver()
    result = P.solve(solver, problem)
    @test result.converged
    @test result.metadata[:termination] === :converged
    @test result.metadata[:closure_verified] === true
    @test result.metadata[:bath_sites] == 0
    @test result.metadata[:bath_fit_residual] > 1e-5
    @test result.metadata[:bath_fit_absolute_residual] > 1e-8
    @test result.metadata[:bath_target_norm] > 1e-8
    @test result.residual <= solver.tol
    @test maximum(abs, result.self_energy.values) < 1e-9
    @test result.local_green_function.values ≈ local_ref.values atol=1e-9
    @test P.validate_dmft_result(result).valid
    @test length(result.history) >= 1
end

@testset "Matrix DMFT: atomic interacting fixed point and independent impurity solution" begin
    interaction = P.kanamori_interaction(2; U=2.5, J=0.35)
    names = interaction.labels
    h = Matrix(Diagonal(ComplexF64[0.1, 0.1, 0.18, 0.18]))
    h[1, 3] = 0.06im
    h[3, 1] = -0.06im
    lattice = _matrix_lattice(cat(h, h, h; dims=3), names)
    axis = P.FermionicMatsubaraAxis(5.0, 10)
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.2, chemical_potential=0.7, axis=axis)
    solver = _dmft_solver(maxiter=5, tol=1e-8)
    result = P.solve(solver, problem)
    @test result.converged
    @test result.iterations <= 5
    @test result.residual <= solver.tol
    # A zero physical bath has a roundoff-scale target, making relative error ill-conditioned.
    @test result.metadata[:bath_fit_absolute_residual] <= 1e-12
    @test result.metadata[:bath_target_norm] <= 1e-12
    @test isfinite(result.metadata[:bath_fit_residual])
    @test P.validate_dmft_result(result).valid
    atomic_bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, length(names), 0); labels=names)
    atomic_problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=0.7, bath=atomic_bath)
    atomic_result = P.solve(solver.impurity_solver, atomic_problem)
    @test result.impurity_result.green_function.values ≈ atomic_result.green_function.values atol=2e-9 rtol=2e-9
    @test result.self_energy.values ≈ atomic_result.self_energy.values atol=2e-8 rtol=2e-8
    @test P.impurity_local_residual(result) <= 2e-8
end


@testset "Matrix DMFT: finite matrix bath fit and warm-started closure" begin
    names = [:a, :b]
    onsite = ComplexF64[0.12 0; 0 -0.18]
    hopping = ComplexF64[0.24 0; 0 0.15]
    lattice = _matrix_lattice(cat(onsite+hopping, onsite-hopping; dims=3), names)
    problem = P.MultiOrbitalDMFTProblem(lattice, _free_interaction(names); temperature=0.16, nfrequencies=8)
    fit_options = P.BathFitOptions(maxiter=40, multistart=1, tol=1e-8, energy_window=(-2.0, 2.0))
    impurity = P.MultiOrbitalEDSolver(bath_sites=2, bath_fit=fit_options, max_full_dimension=4096,
                                      max_sector_dimension=4096, max_eigensystem_bytes=100_000_000)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=impurity, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=3)
    result = P.solve(solver, problem)
    @test result.converged
    @test length(result.impurity_result.bath) == 2
    @test result.metadata[:bath_sites] == 2
    @test result.impurity_result.metadata[:initial_bath_used] === true
    @test isfinite(result.metadata[:bath_fit_objective])
    @test isfinite(result.metadata[:bath_fit_residual])
    @test isfinite(result.metadata[:bath_fit_absolute_residual])
    @test result.metadata[:bath_fit_residual] >= 0
    @test result.metadata[:bath_fit_absolute_residual] >= 0
    @test maximum(abs, result.self_energy.values) <= 1e-8
    @test P.validate_dmft_result(result).valid
end

@testset "Matrix DMFT: mismatch rejection, warm-start and closure guard" begin
    names = [:a, :b]
    axis = P.FermionicMatsubaraAxis(6.0, 8)
    lattice = _matrix_lattice(cat([0.2 0.0; 0.0 -0.1], [0.2 0.0; 0.0 -0.1]; dims=3), names)
    interaction = _free_interaction(names)
    @test_throws ArgumentError P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=1/6, one_body=[0.5 0; 0 -0.1])
    @test_throws ArgumentError P.MultiOrbitalDMFTProblem(lattice, _free_interaction([:x, :y]); temperature=1/6)
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=1/6, axis=axis)
    @test_throws ArgumentError P.solve(_dmft_solver(), problem; initial_self_energy=42)
    @test_throws DimensionMismatch P.solve(_dmft_solver(), problem; initial_bath=P.MatrixDiscreteBath([0.1], ones(ComplexF64, 2, 1); labels=names))
    initial = P.zero_self_energy(axis, 2; labels=names)
    result = P.solve(_dmft_solver(maxiter=1), problem; initial_self_energy=initial)
    @test result.converged
    @test result.metadata[:continuation_self_energy] === true
    @test P.validate_dmft_result(result).valid
end

@testset "Matrix DMFT: explicit forward context, no implicit symbolic lowering" begin
    rep = P.Representation(:matrix_dmft_test, P.FermionFockSpace(2), P.FermionAlgebra(2),
                           P.FermionOccupationBasis([1, 2]); ordering=[1, 2])
    model = P.ManyBodyModel(rep, 0.1 * P.nf(1) - 0.2 * P.nf(2))
    other = P.ManyBodyModel(rep, 0.2 * P.nf(1) - 0.2 * P.nf(2))
    names = [:a, :b]
    lattice = _matrix_lattice(cat([0.1 0; 0 -0.2], [0.1 0; 0 -0.2]; dims=3), names)
    problem = P.MultiOrbitalDMFTProblem(lattice, _free_interaction(names); temperature=0.2, nfrequencies=8)
    solver = _dmft_solver(maxiter=2)
    context = P.MultiOrbitalDMFTForwardContext(model, problem)
    request = P.LocalMatsubaraGreenFunctionRequest()
    report = P.solver_compatibility(solver, model, request, context)
    @test report.compatible
    @test !report.certified
    @test !P.solver_compatibility(solver, other, request, context).compatible
    @test !P.solver_compatibility(solver, model, request).compatible
    computed = P.forward(P.ForwardProblem(model; solver=solver, method_context=context, observable=request))
    @test computed.solution.converged
    @test computed.observable_result.values ≈ computed.solution.local_green_function.values atol=1e-12
    @test computed.solution.metadata[:symbolic_lowering_certified] === false
    sigma_result = P.forward(P.ForwardProblem(model; solver=solver, method_context=context,
                                               observable=P.LocalMatsubaraSelfEnergyRequest()))
    @test sigma_result.result.values ≈ sigma_result.solution.self_energy.values atol=1e-12
end

end # module MultiOrbitalDMFTValidation
