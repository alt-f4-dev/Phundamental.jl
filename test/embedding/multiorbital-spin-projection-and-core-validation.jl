#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalSpinProjectionCoreValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _spin_fixture(; orbitals::Int=2, frequency_indices=[0], boson_indices=[0])
    axis = P.FermionicMatsubaraAxis(5.0, frequency_indices)
    bosons = P.BosonicMatsubaraAxis(5.0, boson_indices)
    n = 2 * orbitals
    pairs = [(a, b) for a in 1:n for b in 1:n]
    names = [Symbol("s", a) for a in 1:n]
    tensor = zeros(ComplexF64, n^2, n^2, length(axis), length(axis), length(bosons))
    for d in 1:orbitals, c in 1:orbitals, b in 1:orbitals, a in 1:orbitals
        uu = (0.13 + 0.01im) / (1 + a + b + c + d)
        ud = (-0.04 + 0.02im) / (1 + a + b + c + d)
        up_left = (2a - 2) * n + 2b - 1
        dn_left = (2a - 1) * n + 2b
        up_right = (2c - 2) * n + 2d - 1
        dn_right = (2c - 1) * n + 2d
        tensor[up_left, up_right, :, :, :] .= uu
        tensor[up_left, dn_right, :, :, :] .= ud
        tensor[dn_left, up_right, :, :, :] .= ud
        tensor[dn_left, dn_right, :, :, :] .= uu
    end
    return P.OrbitalPairSusceptibility(axis, bosons, pairs, names, tensor)
end

function _crossing_fixture(; supported=[-2, -1, 0, 1, 2], transfers=[-2, -1, 0, 1, 2])
    axis = P.FermionicMatsubaraAxis(4.0, supported)
    bosons = P.BosonicMatsubaraAxis(4.0, transfers)
    pairs = [(1, 1)]
    labels = [:orb1]
    cube = zeros(ComplexF64, 1, 1, length(axis), length(axis), length(bosons))
    for m in eachindex(transfers), np in eachindex(supported), n in eachindex(supported)
        cube[1, 1, n, np, m] = 0.1 + 0.01n + 0.001np + 0.002m
    end
    local_d = P.OrbitalPairSusceptibility(axis, bosons, pairs, labels, cube)
    local_m = P.OrbitalPairSusceptibility(axis, bosons, pairs, labels, 0.5 .* cube)
    maps = [[1, 2], [2, 1]]
    densities = [P.OrbitalPairSusceptibility(axis, bosons, pairs, labels, cube .+ (iq - 1) * 0.02) for iq in 1:2]
    magnetic = [P.OrbitalPairSusceptibility(axis, bosons, pairs, labels, 0.5 .* cube .+ (iq - 1) * 0.03) for iq in 1:2]
    return P.OrbitalSpinLadderChannels(local_d, local_m, densities, magnetic, maps; spin_symmetry_certified=true)
end

@testset "SU(2) impurity projection: independent spin-pair sums and storage guards" begin
    object = _spin_fixture()
    result = P.orbital_impurity_spin_channels(object; spatial_labels=[:a, :b])
    @test result.density.labels == [:a, :b]
    @test result.density.pairs == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test result.diagnostics[:longitudinal_residual] == 0
    nspin = 4
    for (j, (c, d)) in enumerate(result.density.pairs), (i, (a, b)) in enumerate(result.density.pairs)
        up_left = (2a - 2) * nspin + 2b - 1
        up_right = (2c - 2) * nspin + 2d - 1
        dn_right = (2c - 1) * nspin + 2d
        uu = object.values[up_left, up_right, 1, 1, 1]
        ud = object.values[up_left, dn_right, 1, 1, 1]
        @test result.density.values[i, j, 1, 1, 1] == uu + ud
        @test result.magnetic.values[i, j, 1, 1, 1] == uu - ud
    end
    invalid = copy(object.values)
    invalid[1, 6, 1, 1, 1] += 1
    broken = P.OrbitalPairSusceptibility(object.fermionic_axis, object.bosonic_axis, object.pairs, object.labels, invalid)
    @test_throws ArgumentError P.orbital_impurity_spin_channels(broken)
    @test_throws ArgumentError P.orbital_impurity_spin_channels(object; max_output_bytes=1)
    @test_throws ArgumentError P.orbital_impurity_spin_channels(object; spatial_labels=[:a, :a])
end

@testset "Finite frequency core: exact crossing support and fail-closed boundaries" begin
    channels = _crossing_fixture()
    core = P.OrbitalCrossingCore(channels, [0], [0])
    @test core.diagnostics[:crossing_support_complete]
    @test core.diagnostics[:high_frequency_tail] === :not_supplied
    @test core.diagnostics[:full_multiorbital_dga] === false
    direct = P.orbital_crossed_vertex(channels, 2, 1, 2, channels.fermion_lookup[0], channels.fermion_lookup[0], channels.boson_lookup[0], 1, 1, 1, 1)
    @test P.orbital_core_crossed_vertex(core, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1) == direct
    @test_throws BoundsError P.orbital_core_crossed_vertex(core, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1)
    @test_throws ArgumentError P.OrbitalCrossingCore(channels, [4], [0])
    @test_throws ArgumentError P.OrbitalCrossingCore(channels, [0], [0]; max_lookup_bytes=1)
    inadequate = _crossing_fixture(supported=[-1, 0], transfers=[0])
    @test_throws ArgumentError P.OrbitalCrossingCore(inadequate, [-1, 0], [0])
    partial = P.OrbitalCrossingCore(inadequate, [-1, 0], [0]; require_complete=false)
    @test !partial.diagnostics[:crossing_support_complete]
    @test partial.diagnostics[:missing_frequency_triples] > 0
end

@testset "Exact finite-bath EOM: independent Dyson reconstruction" begin
    axis = P.FermionicMatsubaraAxis(5.0, [-2, -1, 0, 1, 2])
    interaction = P.kanamori_interaction(1; U=2.2, J=0.0)
    labels = interaction.labels
    onsite = Matrix(Diagonal(ComplexF64[0.13, 0.13]))
    bath = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, length(labels), 0); labels=labels)
    problem = P.MultiOrbitalImpurityProblem(onsite, interaction, axis; chemical_potential=0.4, bath=bath)
    for backend in (:sectorized, :dense)
        solved = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=backend), problem)
        eom = P.orbital_impurity_equation_of_motion(solved)
        @test eom.diagnostics[:maximum_eom_residual] < 1e-10
        @test eom.diagnostics[:dyson_reference_residual] < 1e-9
        @test eom.self_energy.values ≈ solved.self_energy.values atol=1e-9 rtol=1e-9
        @test_throws ArgumentError P.orbital_impurity_equation_of_motion(solved; max_workspace_bytes=1)
    end
    kanamori = P.kanamori_interaction(2; U=2.0, J=0.25)
    orbital_labels = kanamori.labels
    h = Matrix(Diagonal(ComplexF64[0.1, 0.1, -0.15, -0.15]))
    h[1, 3] = 0.08 + 0.02im
    h[3, 1] = conj(h[1, 3])
    h[2, 4] = h[1, 3]
    h[4, 2] = conj(h[2, 4])
    bath2 = P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, 4, 0); labels=orbital_labels)
    problem2 = P.MultiOrbitalImpurityProblem(h, kanamori, axis; chemical_potential=0.3, bath=bath2)
    solved2 = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized), problem2)
    checked2 = P.orbital_impurity_equation_of_motion(solved2)
    @test checked2.diagnostics[:dyson_reference_residual] < 1e-9
    @test checked2.self_energy.values ≈ solved2.self_energy.values atol=1e-9 rtol=1e-9
    bath3 = P.MatrixDiscreteBath([0.25], reshape(ComplexF64[0.15 + 0.02im, 0.06 - 0.03im], 2, 1); labels=labels)
    problem3 = P.MultiOrbitalImpurityProblem(onsite, interaction, axis; chemical_potential=0.4, bath=bath3)
    solved3 = P.solve(P.MultiOrbitalEDSolver(bath_sites=1, backend=:sectorized), problem3)
    checked3 = P.orbital_impurity_equation_of_motion(solved3)
    @test checked3.diagnostics[:dyson_reference_residual] < 1e-9
    free_tensor = zeros(ComplexF64, 2, 2, 2, 2)
    free_interaction = P.SpinOrbitalInteraction(free_tensor; labels=labels)
    free_problem = P.MultiOrbitalImpurityProblem(onsite, free_interaction, axis; chemical_potential=0.4, bath=bath)
    free = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized), free_problem)
    reference = P.orbital_impurity_equation_of_motion(free)
    @test maximum(abs, reference.composite_green) < 1e-12
    @test maximum(abs, reference.self_energy.values) < 1e-12
end

end # module MultiorbitalSpinProjectionCoreValidation
