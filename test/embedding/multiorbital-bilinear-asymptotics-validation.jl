using Test
using LinearAlgebra
using Phundamental

module OrbitalBilinearAsymptoticsValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _atomic_state(; U=1.1, backend=:sectorized)
    interaction = P.kanamori_interaction(1; U=U, J=0.0)
    h = ComplexF64[0.12 0.0; 0.0 0.12]
    axis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, 2, 0); labels=interaction.labels)
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; bath=bath, chemical_potential=0.1)
    return P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=backend), problem).thermal_state
end

@testset "Exact bilinear Lehmann response: interacting thermal covariance" begin
    state = _atomic_state()
    data = P.orbital_bilinear_lehmann(state; pairs=[(1, 1), (2, 2)])
    beta = state.beta
    bosonic = P.BosonicMatsubaraAxis(beta, [-2, 0, 1, 2])
    response = P.orbital_bilinear_susceptibility(data, bosonic)
    one = @view(data.operators[:, :, 1])
    two = @view(data.operators[:, :, 2])
    expected11 = beta * (sum(data.probabilities .* diag(one * one)) - data.expectations[1]^2)
    expected12 = beta * (sum(data.probabilities .* diag(one * two)) - data.expectations[1] * data.expectations[2])
    index0 = findfirst(==(0), bosonic.indices)
    @test isapprox(response.values[1, 1, index0], expected11; atol=1e-11)
    @test isapprox(response.values[1, 2, index0], expected12; atol=1e-11)
    @test response.values[2, 1, index0] ≈ response.values[1, 2, index0] atol=1e-11
    @test maximum(abs, response.values[:, :, [1, 3, 4]]) < 1e-10
    @test response.metadata[:bosonic_high_frequency_kernel_input]
    @test !response.metadata[:two_particle_vertex_tail_certified]
    full = P.orbital_bilinear_lehmann(state)
    channels = P.orbital_bilinear_spin_channels(full, bosonic)
    @test channels.density[1, 1, index0] ≈ expected11 + expected12 atol=1e-11
    @test channels.magnetic[1, 1, index0] ≈ expected11 - expected12 atol=1e-11
    @test channels.metadata[:longitudinal_residual] < 1e-10
    @test_throws ArgumentError P.orbital_bilinear_spin_channels(data, bosonic)
    @test_throws ArgumentError P.orbital_bilinear_lehmann(state; max_storage_bytes=1)
    @test_throws ArgumentError P.orbital_bilinear_susceptibility(data, P.BosonicMatsubaraAxis(5.0, [0]))
end

@testset "Exact bilinear response: full-Fock and sectorized complex hopping" begin
    interaction = P.kanamori_interaction(1; U=0.65, J=0.0)
    h = ComplexF64[0.14 0.09im; -0.09im -0.07]
    faxis = P.FermionicMatsubaraAxis(5.0, [-1, 0])
    bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, 2, 0); labels=interaction.labels)
    impurity = P.MultiOrbitalImpurityProblem(h, interaction, faxis; bath=bath, chemical_potential=0.07)
    sector = P.solve(P.MultiOrbitalEDSolver(bath_sites=0), impurity).thermal_state
    dense = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:dense), impurity).thermal_state
    chosen = [(1, 1), (2, 2), (1, 2), (2, 1)]
    a = P.orbital_bilinear_lehmann(sector; pairs=chosen)
    b = P.orbital_bilinear_lehmann(dense; pairs=chosen)
    frequencies = P.BosonicMatsubaraAxis(5.0, [-2, 0, 1, 2])
    xa = P.orbital_bilinear_susceptibility(a, frequencies)
    xb = P.orbital_bilinear_susceptibility(b, frequencies)
    @test xa.values ≈ xb.values atol=1e-10 rtol=1e-9
    @test any(abs(x) > 1e-7 for x in xa.values[:, :, 3])
    ma = P.orbital_bilinear_moments(a; order=4)
    mb = P.orbital_bilinear_moments(b; order=4)
    @test ma.coefficients ≈ mb.coefficients atol=1e-10
    @test ma.spectral_radius > 0
    high = P.BosonicMatsubaraAxis(5.0, [40, 60, 80])
    approximation = P.orbital_bilinear_tail(ma, high)
    exact = P.orbital_bilinear_susceptibility(a, high)
    for index in eachindex(high.indices)
        @test all(abs.(approximation.values[:, :, index] - exact.values[:, :, index]) .<=
            approximation.error_bounds[:, :, index] .+ 1e-11)
    end
    @test maximum(approximation.error_bounds[:, :, end]) < maximum(approximation.error_bounds[:, :, 1])
    @test approximation.certified_for === :finite_bath_bilinear_response_only
    @test !approximation.two_particle_vertex_tail_certified
    @test_throws ArgumentError P.orbital_bilinear_moments(a; order=0)
    @test_throws ArgumentError P.orbital_bilinear_moments(a; max_output_bytes=1)
    @test_throws ArgumentError P.orbital_bilinear_tail(ma, P.BosonicMatsubaraAxis(5.0, [0]))
    @test_throws ArgumentError P.orbital_bilinear_tail(ma, P.BosonicMatsubaraAxis(4.0, [60]))
end

@testset "Interacting, dispersive ladder convergence fixture" begin
    beta = 4.0
    interaction = P.kanamori_interaction(1; U=0.7, J=0.0)
    h = zeros(ComplexF64, 2, 2, 2)
    h[:, :, 1] .= Diagonal(ComplexF64[-0.2, -0.2])
    h[:, :, 2] .= Diagonal(ComplexF64[0.2, 0.2])
    lattice = P.MatrixMomentumLattice(h; labels=interaction.labels, kpoints=[:k0, :kpi], qmaps=[[1, 2], [2, 1]])
    axis = P.FermionicMatsubaraAxis(beta, [-1, 0])
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=inv(beta), axis=axis, chemical_potential=0.1)
    dmft = P.solve(P.MultiOrbitalDMFTSolver(impurity_solver=P.MultiOrbitalEDSolver(bath_sites=0),
        mixing=P.LinearMixing(1.0), maxiter=8), problem)
    @test dmft.converged
    @test P.validate_dmft_result(dmft).valid
    @test dmft.metadata[:bath_fit_absolute_residual] > 0
    @test norm(dmft.self_energy.values) > 1e-5
    small = P.OrbitalLadderWorkflowProblem(dmft, P.FermionicMatsubaraAxis(beta, -2:1),
        P.BosonicMatsubaraAxis(beta, -1:1); core_bosonic_indices=[0])
    large = P.OrbitalLadderWorkflowProblem(dmft, P.FermionicMatsubaraAxis(beta, -3:2),
        P.BosonicMatsubaraAxis(beta, -2:2); core_bosonic_indices=[0])
    solver = P.OrbitalLadderWorkflowSolver(four_point_backend=P.TauGridFourPoint(ntime=12, max_dimension=64),
        pseudoinverse=true, max_contractions=100_000_000)
    report = P.orbital_ladder_convergence_study(solver, small, large; ntime_coarse=12, ntime_fine=16)
    @test length(report.stages) == 4
    @test all(isfinite, (report.frequency_comparison.relative_residual,
        report.quadrature_comparison.relative_residual, report.coupled_difference.relative_residual))
    @test size(report.final_result.nonlocal_self_energy.values, 3) == 2
    @test !report.metadata[:two_particle_vertex_tail_certified]
    @test !report.metadata[:full_multiorbital_dga]
end

end # module OrbitalBilinearAsymptoticsValidation
