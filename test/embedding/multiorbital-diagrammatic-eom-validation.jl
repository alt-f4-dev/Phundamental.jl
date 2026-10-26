#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalDiagrammaticEOMValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _fixture(; atomic::Bool=false, interaction_strength::Float64=1.25)
    interaction = P.kanamori_interaction(1; U=interaction_strength, J=0.0)
    labels = interaction.labels
    onsite = Matrix(Diagonal(ComplexF64[0.12, 0.12]))
    displacement = atomic ? zeros(ComplexF64, 2, 2) : Matrix(Diagonal(ComplexF64[0.16, 0.16]))
    momentum = atomic ? [:k0] : [:k0, :kpi]
    qmaps = atomic ? [[1]] : [[1, 2], [2, 1]]
    blocks = atomic ? reshape(onsite, 2, 2, 1) : cat(onsite+displacement, onsite-displacement; dims=3)
    lattice = P.MatrixMomentumLattice(blocks; labels=labels, kpoints=momentum, qmaps=qmaps)
    axis = P.FermionicMatsubaraAxis(4.0, [-1, 0])
    baxis = P.BosonicMatsubaraAxis(4.0, [0, 1])
    problem = P.MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.25, axis=axis, chemical_potential=0.4)
    ed = P.MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized, max_full_dimension=256, max_sector_dimension=256)
    solver = P.MultiOrbitalDMFTSolver(impurity_solver=ed, mixing=P.LinearMixing(1.0), tol=1e-8, maxiter=5)
    dmft = P.solve(solver, problem)
    @test dmft.converged
    green = P.expanded_dmft_momentum_green(dmft, baxis)
    return (; dmft, green, interaction, axis, baxis, lattice, labels)
end

function _vertices(fixture; zero::Bool=false, contrast::Bool=true)
    n = length(fixture.labels)
    pairs = [(a, b) for a in 1:n for b in 1:n]
    nf, nb = length(fixture.axis), length(fixture.baxis)
    tensor = zeros(ComplexF64, n^2, n^2, nf, nf, nb)
    if !zero
        for iΩ in 1:nb, j in 1:n^2, i in 1:n^2, np in 1:nf, nu in 1:nf
            tensor[i, j, np, nu, iΩ] = (0.02 + 0.003im) / (1 + i + j + abs(np-nu) + iΩ)
        end
    end
    local_vertex = P.OrbitalPairSusceptibility(fixture.axis, fixture.baxis, pairs, fixture.labels, tensor)
    q_vertices = [P.OrbitalPairSusceptibility(fixture.axis, fixture.baxis, pairs, fixture.labels,
        (contrast ? 1 + 0.13 * (iq-1) : 1.0) .* tensor) for iq in eachindex(fixture.lattice.qmaps)]
    return (q_vertices, local_vertex)
end

# Independent direct contraction of the finite-frequency diagram, without
# pair-flattened matrices, BLAS multiplication, or a precomputed kernel.
function _reference(problem)
    dmft, green = problem.dmft, problem.green
    lattice = dmft.problem.lattice
    n, nk, nf = length(lattice.labels), length(lattice.kpoints), length(dmft.problem.axis)
    g = green.values
    beta = dmft.problem.axis.beta
    U = dmft.problem.interaction.tensor
    flookup = Dict(nu => i for (i, nu) in enumerate(green.axis.indices))
    gloc = dropdims(sum(reshape(lattice.weights, 1, 1, nk, 1) .* g; dims=3); dims=3)
    output = repeat(reshape(dmft.self_energy.values, n, n, 1, nf), 1, 1, nk, 1)
    for (iΩ, omega) in enumerate(problem.bosonic_axis.indices), (iprime, nuprime) in enumerate(dmft.problem.axis.indices), (inu, nu) in enumerate(dmft.problem.axis.indices)
        ip, shiftedprime = flookup[nuprime], flookup[nuprime+omega]
        shifted = flookup[nu+omega]
        for m in 1:n, mp in 1:n, h in 1:n, hp in 1:n, l in 1:n, u in 1:n, lp in 1:n, up in 1:n
            coefficient = U[m, l, h, u]
            iszero(coefficient) && continue
            p, r = (up-1)*n+lp, (hp-1)*n+mp
            local_bubble = -beta * gloc[up, u, ip] * gloc[l, lp, shiftedprime]
            local_f = problem.local_vertex.values[p, r, iprime, inu, iΩ]
            for ik in 1:nk
                output[m, mp, ik, inu] += coefficient * local_bubble * local_f * gloc[h, hp, shifted] / (2beta)
            end
            for iq in eachindex(problem.qmaps)
                mapping = problem.qmaps[iq]
                bubble = -beta * sum(green.weights[k] * g[up, u, k, ip] * g[l, lp, mapping[k], shiftedprime] for k in 1:nk)
                f = problem.q_vertices[iq].values[p, r, iprime, inu, iΩ]
                for ik in 1:nk
                    minusq = invperm(mapping)[ik]
                    output[m, mp, ik, inu] -= problem.qweights[iq] * coefficient * bubble * f * g[h, hp, minusq, shifted] / (2beta)
                end
            end
        end
    end
    return output
end

@testset "Diagrammatic multiorbital EOM: independent direct tensor contraction" begin
    fixture = _fixture()
    q, local_vertex = _vertices(fixture)
    problem = P.OrbitalDiagrammaticEOMProblem(fixture.dmft, fixture.green, q, local_vertex, fixture.baxis)
    result = P.orbital_diagrammatic_self_energy(problem)
    reference = _reference(problem)
    @test result.values ≈ reference atol=1e-11 rtol=1e-11
    @test result.metadata[:local_double_counting_scheme] === :matched_local_eom_diagram
    @test result.metadata[:frequency_box_complete]
    @test result.metadata[:high_frequency_completion] === :none
    @test result.metadata[:crossing_complete_certified_by_package] === false
    @test !result.metadata[:full_multiorbital_dga]
    @test all(isfinite, real.(result.values))
    @test P.dyson_momentum_green(fixture.lattice, result; chemical_potential=0.4) isa P.MatrixMomentumResolvedGreenFunction
    @test_throws ArgumentError P.orbital_diagrammatic_self_energy(problem; max_workspace_bytes=1)
    @test_throws ArgumentError P.orbital_diagrammatic_self_energy(problem; max_output_bytes=1)
end

@testset "Diagrammatic multiorbital EOM: atomic locality and zero-vertex reductions" begin
    atom = _fixture(atomic=true)
    q, local_vertex = _vertices(atom; contrast=false)
    problem = P.OrbitalDiagrammaticEOMProblem(atom.dmft, atom.green, q, local_vertex, atom.baxis)
    result = P.orbital_diagrammatic_self_energy(problem)
    @test result.values[:, :, 1, :] ≈ atom.dmft.self_energy.values atol=1e-11 rtol=1e-11
    @test result.metadata[:local_diagram_difference_norm] < 1e-11
    dispersive = _fixture()
    zq, zl = _vertices(dispersive; zero=true)
    zeros_problem = P.OrbitalDiagrammaticEOMProblem(dispersive.dmft, dispersive.green, zq, zl, dispersive.baxis)
    zeros_result = P.orbital_diagrammatic_self_energy(zeros_problem)
    @test zeros_result.values[:, :, 1, :] ≈ dispersive.dmft.self_energy.values atol=1e-13
    @test zeros_result.values[:, :, 2, :] ≈ dispersive.dmft.self_energy.values atol=1e-13
end

@testset "Diagrammatic multiorbital EOM: compatibility and frequency rejection" begin
    f = _fixture()
    q, lv = _vertices(f)
    @test_throws DimensionMismatch P.OrbitalDiagrammaticEOMProblem(f.dmft, f.green, [first(q)], lv, f.baxis)
    @test_throws ArgumentError P.OrbitalDiagrammaticEOMProblem(f.dmft, f.green, q, lv, f.baxis; qweights=[0.6, 0.4])
    incomplete = P.momentum_green(f.lattice, f.axis, f.dmft.self_energy; chemical_potential=0.4)
    @test_throws ArgumentError P.OrbitalDiagrammaticEOMProblem(f.dmft, incomplete, q, lv, f.baxis)
    wrong_labels = [:x, :y]
    invalid = P.OrbitalPairSusceptibility(f.axis, f.baxis, lv.pairs, wrong_labels, lv.values)
    @test_throws ArgumentError P.OrbitalDiagrammaticEOMProblem(f.dmft, f.green, q, invalid, f.baxis)
end


function _channel_fixture(f; support=[-2, -1, 0, 1, 2], transferred=[-2, -1, 0, 1, 2], contrast::Bool=false)
    faxis = P.FermionicMatsubaraAxis(f.axis.beta, support)
    baxis = P.BosonicMatsubaraAxis(f.axis.beta, transferred)
    nf, nb = length(faxis), length(baxis)
    pairs = [(1, 1)]
    d = Array{ComplexF64}(undef, 1, 1, nf, nf, nb)
    m = similar(d)
    for iΩ in 1:nb, j in 1:nf, i in 1:nf
        d[1, 1, i, j, iΩ] = 0.025 + 0.001im * (i-j) + 0.0002iΩ
        m[1, 1, i, j, iΩ] = -0.013 + 0.0007im * (j-i) + 0.0003iΩ
    end
    local_d = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:orb1], d)
    local_m = P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:orb1], m)
    qd = [P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:orb1], d .+ (contrast ? 0.002(iq-1) : 0.0)) for iq in eachindex(f.lattice.qmaps)]
    qm = [P.OrbitalPairSusceptibility(faxis, baxis, pairs, [:orb1], m .+ (contrast ? 0.001(iq-1) : 0.0)) for iq in eachindex(f.lattice.qmaps)]
    channels = P.OrbitalSpinLadderChannels(local_d, local_m, qd, qm, f.lattice.qmaps; spin_symmetry_certified=true)
    return P.OrbitalCrossingCore(channels, f.axis.indices, [0])
end

function _lift_local_vertex(f, core)
    pairs = [(a, b) for a in 1:2 for b in 1:2]
    values = zeros(ComplexF64, 4, 4, length(f.axis), length(f.axis), 1)
    for (j, (c, d)) in enumerate(pairs), (i, (a, b)) in enumerate(pairs), np in 1:length(f.axis), nu in 1:length(f.axis)
        ld = core.channels.local_density.values[1, 1, core.support_fermionic_locations[np], core.support_fermionic_locations[nu], core.support_bosonic_locations[1]]
        lm = core.channels.local_magnetic.values[1, 1, core.support_fermionic_locations[np], core.support_fermionic_locations[nu], core.support_bosonic_locations[1]]
        values[i, j, np, nu, 1] = P.Embedding._orbital_su2_spin_element(ld, lm, a, b, c, d)
    end
    axis = P.BosonicMatsubaraAxis(f.axis.beta, [0])
    return P.OrbitalPairSusceptibility(f.axis, axis, pairs, f.labels, values)
end

@testset "SU(2) crossed diagrammatic EOM: Pauli-spin structure and local reference" begin
    f = _fixture()
    core = _channel_fixture(f; contrast=false)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        value = P.orbital_spin_orbital_crossed_vertex(core, 1, 1, 1, 1, 1, 1, a, b, c, d)
        fd = core.channels.local_density.values[1, 1, core.support_fermionic_locations[1], core.support_fermionic_locations[1], core.support_bosonic_locations[1]]
        fm = core.channels.local_magnetic.values[1, 1, core.support_fermionic_locations[1], core.support_fermionic_locations[1], core.support_bosonic_locations[1]]
        expected = ((isodd(a) == isodd(b) && isodd(c) == isodd(d)) ? (fd-fm)/2 : 0) +
            ((isodd(a) == isodd(d) && isodd(c) == isodd(b)) ? fm : 0)
        @test value ≈ expected atol=1e-14
    end
    channel_sigma = P.orbital_channel_ladder_self_energy(f.dmft, f.green, core)
    local_full = _lift_local_vertex(f, core)
    direct_problem = P.OrbitalDiagrammaticEOMProblem(f.dmft, f.green, [local_full for _ in f.lattice.qmaps], local_full,
        P.BosonicMatsubaraAxis(f.axis.beta, [0]))
    direct_sigma = P.orbital_diagrammatic_self_energy(direct_problem)
    @test channel_sigma.values ≈ direct_sigma.values atol=1e-11 rtol=1e-11
    @test channel_sigma.metadata[:su2_hamiltonian_verified]
    @test channel_sigma.metadata[:local_double_counting_scheme] === :matched_local_eom_diagram
    @test !channel_sigma.metadata[:full_multiorbital_dga]
    @test_throws ArgumentError P.orbital_channel_ladder_self_energy(f.dmft, f.green, core; max_contractions=1)
    @test_throws ArgumentError P.orbital_channel_ladder_self_energy(f.dmft, f.green, core; max_output_bytes=1)
    @test_throws ArgumentError P.orbital_channel_ladder_self_energy(f.dmft, f.green, core; qweights=[0.75, 0.25])
end

@testset "SU(2) crossed diagrammatic EOM: atomic cancellation and nonlocal channel response" begin
    atom = _fixture(atomic=true)
    atomic_core = _channel_fixture(atom)
    result = P.orbital_channel_ladder_self_energy(atom.dmft, atom.green, atomic_core)
    @test result.values[:, :, 1, :] ≈ atom.dmft.self_energy.values atol=1e-11 rtol=1e-11
    nonlocal = _fixture()
    channels = _channel_fixture(nonlocal; contrast=true)
    output = P.orbital_channel_ladder_self_energy(nonlocal.dmft, nonlocal.green, channels)
    @test all(isfinite, real.(output.values))
    @test all(isfinite, imag.(output.values))
    @test output.metadata[:crossing_support_complete]
    @test output.metadata[:high_frequency_completion] === :none
    @test P.dyson_momentum_green(nonlocal.lattice, output; chemical_potential=0.4) isa P.MatrixMomentumResolvedGreenFunction
    incomplete = P.OrbitalCrossingCore(channels.channels, [-1, 0], [0]; require_complete=false)
    @test incomplete.diagnostics[:crossing_support_complete]
end

@testset "Diagrammatic EOM: explicit frequency-box comparison" begin
    f = _fixture()
    q, local_vertex = _vertices(f)
    problem = P.OrbitalDiagrammaticEOMProblem(f.dmft, f.green, q, local_vertex, f.baxis)
    result = P.orbital_diagrammatic_self_energy(problem)
    frequencies = P.FermionicMatsubaraAxis(f.axis.beta, [-2, -1, 0])
    expanded = zeros(ComplexF64, 2, 2, 2, 3)
    expanded[:, :, :, 2:3] .= result.values
    candidate = P.OrbitalMomentumSelfEnergy(frequencies, f.lattice.kpoints, f.labels, expanded)
    comparison = P.orbital_eom_frequency_box_comparison(result, candidate)
    @test comparison.converged
    @test comparison.maximum_absolute_difference == 0
    @test !comparison.high_frequency_asymptotics_certified
    candidate.values[1, 1, 1, 3] += 0.1
    shifted = P.orbital_eom_frequency_box_comparison(result, candidate)
    @test !shifted.converged
    @test shifted.relative_residual > 0
    @test_throws ArgumentError P.orbital_eom_frequency_box_comparison(result, result)
    @test_throws ArgumentError P.orbital_eom_frequency_box_comparison(candidate, result)
end

end # module MultiorbitalDiagrammaticEOMValidation
