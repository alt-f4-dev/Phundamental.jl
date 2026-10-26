#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalTwoParticleValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _fixture(; U=1.1, backend=:sectorized, mixing=true)
    beta = 3.5
    faxis = P.FermionicMatsubaraAxis(beta, [-1, 0])
    baxis = P.BosonicMatsubaraAxis(beta, [0, 1])
    interaction = P.kanamori_interaction(1; U=U)
    h = mixing ? ComplexF64[0.18 0.13+0.07im; 0.13-0.07im -0.12] : ComplexF64[0.18 0; 0 -0.12]
    bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, 2, 0); labels=interaction.labels)
    problem = P.MultiOrbitalImpurityProblem(h, interaction, faxis; chemical_potential=0.26, bath=bath)
    solver = P.MultiOrbitalEDSolver(; bath_sites=0, backend=backend)
    result = P.solve(solver, problem)
    return (; result, faxis, baxis, h, U, mu=0.26)
end

function _independent_eigenoperators(fixture)
    # Explicit 4-state Fock basis and independently constructed operator matrices.
    C1 = zeros(ComplexF64, 4, 4)
    C2 = zeros(ComplexF64, 4, 4)
    C1[1, 2] = 1
    C1[3, 4] = 1
    C2[1, 3] = 1
    C2[2, 4] = -1
    C = [C1, C2]
    N1 = C1' * C1
    N2 = C2' * C2
    H = fixture.U * N1 * N2
    for b in 1:2, a in 1:2
        H += (fixture.h[a, b] - (a == b ? fixture.mu : 0)) * C[a]' * C[b]
    end
    eig = eigen(Hermitian(H))
    return eig.values, [eig.vectors' * op * eig.vectors for op in C]
end

@testset "Multiorbital impurity four-point: independent finite-Fock reference" begin
    fixture = _fixture()
    backend = P.TauGridFourPoint(; ntime=6, max_dimension=8, algorithm=:factored, equal_time=:symmetric)
    calculated = P.impurity_two_particle(fixture.result.thermal_state, fixture.faxis, fixture.baxis;
        backend=backend, pairs=[(1, 1), (2, 2), (1, 2)])
    @test calculated isa P.MultiOrbitalImpurityTwoParticleResult
    @test size(calculated.susceptibility.values) == (3, 3, 2, 2, 2)
    @test calculated.susceptibility.pairs == [(1, 1), (2, 2), (1, 2)]
    @test calculated.diagnostics[:pair_count] == 3
    @test calculated.diagnostics[:hilbert_dimension] == 4
    @test all(isfinite, calculated.susceptibility.values)
    @test all(isfinite, calculated.bubble.values)
    eig, C = _independent_eigenoperators(fixture)
    Cdagger = [Matrix(adjoint(operator)) for operator in C]
    taus = [(i - 0.5) * fixture.faxis.beta / backend.ntime for i in 1:backend.ntime]
    for (j, (c, d)) in enumerate(calculated.susceptibility.pairs),
        (i, (a, b)) in enumerate(calculated.susceptibility.pairs)
        cube = P.Greens._connected_tau_cube_direct(eig, Cdagger[a], C[b], Cdagger[c], C[d], fixture.faxis.beta, backend)
        expected = P.Greens._fourier_project_particle_hole(cube, taus, fixture.faxis, fixture.baxis)
        @test calculated.susceptibility.values[i, j, :, :, :] ≈ expected atol=1e-9 rtol=1e-9
    end
    block = P.orbital_pair_block(calculated.susceptibility, (1, 1), (2, 2))
    @test block.values ≈ calculated.susceptibility.values[1, 2, :, :, :] atol=0
    @test_throws ArgumentError P.orbital_pair_block(calculated.susceptibility, (2, 1), (1, 1))
    charge = P.project_orbital_channel(calculated.susceptibility, [1, 1, 0]; channel=:charge)
    magnetic = P.project_orbital_channel(calculated.susceptibility, [1, -1, 0]; channel=:magnetic)
    expected_charge = sum(calculated.susceptibility.values[i, j, :, :, :] for i in 1:2 for j in 1:2)
    expected_magnetic = calculated.susceptibility.values[1, 1, :, :, :] .-
                        calculated.susceptibility.values[1, 2, :, :, :] .-
                        calculated.susceptibility.values[2, 1, :, :, :] .+
                        calculated.susceptibility.values[2, 2, :, :, :]
    @test charge.values ≈ expected_charge atol=1e-12
    @test magnetic.values ≈ expected_magnetic atol=1e-12
    @test_throws DimensionMismatch P.project_orbital_channel(calculated.susceptibility, [1, 1])
end

@testset "Multiorbital impurity four-point: dense, sectorized and direct consistency" begin
    source = _fixture(; U=1.7)
    dense = _fixture(; U=1.7, backend=:dense)
    backend = P.TauGridFourPoint(; ntime=6, max_dimension=8, algorithm=:factored, parallel=:serial)
    selection = [(1, 1), (1, 2)]
    a = P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, pairs=selection)
    b = P.multiorbital_impurity_two_particle(dense.result.thermal_state, dense.faxis, dense.baxis;
        backend=backend, pairs=selection)
    direct = P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=P.TauGridFourPoint(; ntime=6, max_dimension=8, algorithm=:direct), pairs=selection)
    @test P.multiorbital_two_particle_residual(a, b) < 1e-10
    @test P.multiorbital_two_particle_residual(a, direct) < 1e-10
    @test a.bubble.values ≈ b.bubble.values atol=1e-11
    @test a.susceptibility.labels == source.result.hamiltonian.labels[1:2]
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, pairs=[(1, 1), (1, 1)])
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, pairs=[(3, 1)])
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, max_output_bytes=1)
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, max_operator_bytes=1)
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=backend, max_workspace_bytes=1)
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, source.baxis;
        backend=P.TauGridFourPoint(; ntime=6, max_dimension=3))
    wrong = P.BosonicMatsubaraAxis(4.1, [0])
    @test_throws ArgumentError P.multiorbital_impurity_two_particle(source.result.thermal_state, source.faxis, wrong; pairs=selection)
end

@testset "Multiorbital impurity four-point: complex finite matrix bath" begin
    fixture = _fixture(; U=1.3)
    interaction = P.kanamori_interaction(1; U=1.3)
    bath = P.MatrixDiscreteBath([0.35], reshape(ComplexF64[0.31+0.14im, -0.12+0.22im], 2, 1); labels=interaction.labels)
    problem = P.MultiOrbitalImpurityProblem(fixture.h, interaction, fixture.faxis;
        chemical_potential=fixture.mu, bath=bath)
    sector = P.solve(P.MultiOrbitalEDSolver(; bath_sites=1, backend=:sectorized), problem)
    dense = P.solve(P.MultiOrbitalEDSolver(; bath_sites=1, backend=:dense), problem)
    backend = P.TauGridFourPoint(; ntime=6, max_dimension=16, algorithm=:factored)
    selection = [(1, 1), (1, 2), (2, 2)]
    from_sectors = P.multiorbital_impurity_two_particle(sector.thermal_state, fixture.faxis, fixture.baxis;
        backend=backend, pairs=selection)
    from_dense = P.multiorbital_impurity_two_particle(dense.thermal_state, fixture.faxis, fixture.baxis;
        backend=backend, pairs=selection)
    @test from_sectors.diagnostics[:hilbert_dimension] == 8
    @test P.multiorbital_two_particle_residual(from_sectors, from_dense) < 1e-10
    @test from_sectors.bubble.values ≈ from_dense.bubble.values atol=1e-11
    @test any(abs.(from_sectors.susceptibility.values[2, :, :, :, :]) .> 1e-8)
end

@testset "Multiorbital impurity four-point: analytical matrix Wick bubble" begin
    fixture = _fixture(; U=0.0)
    result = P.multiorbital_impurity_two_particle(fixture.result.thermal_state, fixture.faxis, fixture.baxis;
        backend=P.TauGridFourPoint(; ntime=8, max_dimension=8), pairs=[(1, 1), (1, 2), (2, 1)])
    identity_matrix = Matrix{ComplexF64}(I, 2, 2)
    for (iΩ, m) in enumerate(fixture.baxis.indices), (iν, n) in enumerate(fixture.faxis.indices)
        z = im * (2n + 1) * π / fixture.faxis.beta
        zshift = im * (2(n+m) + 1) * π / fixture.faxis.beta
        G = ((z + fixture.mu) * identity_matrix - fixture.h) \ identity_matrix
        Gshift = ((zshift + fixture.mu) * identity_matrix - fixture.h) \ identity_matrix
        for (j, (c, d)) in enumerate(result.bubble.pairs), (i, (a, b)) in enumerate(result.bubble.pairs)
            expected = -fixture.faxis.beta * G[d, a] * Gshift[b, c]
            @test result.bubble.values[i, j, iν, iν, iΩ] ≈ expected atol=1e-11 rtol=1e-11
        end
    end
    @test result.bubble.metadata[:diagonal_in_fermionic_indices]
    @test maximum(abs, result.bubble.values[:, :, 1, 2, :]) == 0
end

@testset "Multiorbital impurity four-point: explicit vertex extraction and guards" begin
    fixture = _fixture()
    pairs = [(1, 1), (2, 2)]
    faxis = P.FermionicMatsubaraAxis(fixture.faxis.beta, [0])
    baxis = P.BosonicMatsubaraAxis(fixture.faxis.beta, [0])
    χ0 = zeros(ComplexF64, 2, 2, 1, 1, 1)
    χ = zeros(ComplexF64, 2, 2, 1, 1, 1)
    χ0[:, :, 1, 1, 1] = ComplexF64[2.0 0.1im; -0.1im 1.5]
    χ[:, :, 1, 1, 1] = ComplexF64[2.2 0.2im; -0.2im 1.4]
    bubble = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:up, :down], χ0)
    response = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:up, :down], χ)
    vertex = P.multiorbital_irreducible_vertex(bubble, response)
    expected = inv(χ0[:, :, 1, 1, 1]) - inv(χ[:, :, 1, 1, 1])
    @test vertex.values[:, :, 1, 1, 1] ≈ expected atol=1e-11
    @test vertex.metadata[:inverse_diagnostics][1].dimension == 2
    @test_throws ArgumentError P.multiorbital_irreducible_vertex(bubble, response; max_inversion_bytes=1)
    wrong = P.OrbitalPairSusceptibility(faxis, baxis, reverse(pairs), [:up, :down], χ)
    @test_throws ArgumentError P.multiorbital_irreducible_vertex(bubble, wrong)
    singular = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:up, :down], zeros(ComplexF64, size(χ)))
    @test_throws LinearAlgebra.SingularException P.multiorbital_irreducible_vertex(bubble, singular)
    @test all(isfinite, P.multiorbital_irreducible_vertex(bubble, singular; pseudoinverse=true).values)
end

end # module MultiorbitalTwoParticleValidation
