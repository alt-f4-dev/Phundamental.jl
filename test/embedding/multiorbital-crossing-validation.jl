#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalCrossingValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _fixture(; frequencies=[0], transfers=[0], orbitals=2)
    faxis = P.FermionicMatsubaraAxis(4.0, frequencies)
    baxis = P.BosonicMatsubaraAxis(4.0, transfers)
    pairs = [(a, b) for a in 1:orbitals for b in 1:orbitals]
    labels = [Symbol("orb", a) for a in 1:orbitals]
    values = zeros(ComplexF64, length(pairs), length(pairs), length(faxis), length(faxis), length(baxis))
    for i in 1:length(pairs)
        values[i, i, :, :, :] .= 0.12 + 0.02im
    end
    local_d = P.OrbitalPairSusceptibility(faxis, baxis, pairs, labels, values)
    local_m = P.OrbitalPairSusceptibility(faxis, baxis, pairs, labels, 0.5 .* values)
    density = [P.OrbitalPairSusceptibility(faxis, baxis, pairs, labels, values .+ shift) for shift in (0, 0.02)]
    magnetic = [P.OrbitalPairSusceptibility(faxis, baxis, pairs, labels, 0.5 .* values .+ shift) for shift in (0, 0.03)]
    maps = [[1, 2], [2, 1]]
    return (; local_d, local_m, density, magnetic, maps, faxis, baxis, pairs, labels)
end

@testset "Orbital ladder crossing: complete pair and spin-symmetry contracts" begin
    f = _fixture()
    channels = P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, f.maps;
        spin_symmetry_certified=true)
    @test channels.metadata[:crossing_relation] == :two_particle_hole_ladders
    @test channels.metadata[:full_multiorbital_dga] === false
    @test P.orbital_crossing_coverage(channels).complete
    @test_throws ArgumentError P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, f.maps)
    @test_throws ArgumentError P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, f.maps;
        spin_symmetry_certified=true, max_retained_bytes=1)
    @test_throws ArgumentError P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, [[1, 2], [1, 2]];
        spin_symmetry_certified=true)
    @test_throws DimensionMismatch P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density[1:1], f.magnetic, f.maps;
        spin_symmetry_certified=true)
end

@testset "Orbital ladder crossing: SU(2) channel coefficients and crossed orbital index" begin
    f = _fixture()
    channels = P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, f.maps;
        spin_symmetry_certified=true)
    # Crossed k'-k=π matches q=π. The q=π nonlocal differences are 0.02 and 0.03.
    value = P.orbital_crossed_vertex(channels, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1)
    @test value.density ≈ (0.12 + 0.02im) + 0.02 - 0.5 * 0.02 - 1.5 * 0.03 atol=1e-13
    @test value.magnetic ≈ (0.06 + 0.01im) + 0.03 - 0.5 * 0.02 + 0.5 * 0.03 atol=1e-13
    # With k'=k the crossed q=0 contribution vanishes, leaving only the direct ladder.
    direct = P.orbital_crossed_vertex(channels, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    @test direct.density ≈ (0.12 + 0.02im) + 0.02 atol=1e-13
    @test direct.magnetic ≈ (0.06 + 0.01im) + 0.03 atol=1e-13
    # For nontrivial orbital labels, the crossed tensor accesses (c,b),(a,d), not (a,b),(c,d).
    special = copy(f.density[2].values)
    special[3, 2, 1, 1, 1] += 0.4
    modified = [f.density[1], P.OrbitalPairSusceptibility(f.faxis, f.baxis, f.pairs, f.labels, special)]
    channels2 = P.OrbitalSpinLadderChannels(f.local_d, f.local_m, modified, f.magnetic, f.maps;
        spin_symmetry_certified=true)
    # a=1,b=1,c=2,d=2 => direct pair (1,1),(2,2), crossed pair (2,1),(1,2).
    reference = P.orbital_crossed_vertex(channels, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2)
    changed = P.orbital_crossed_vertex(channels2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2)
    @test changed.density - reference.density ≈ -0.2 atol=1e-13
    @test changed.magnetic - reference.magnetic ≈ -0.2 atol=1e-13
    @test_throws ArgumentError P.orbital_crossed_vertex(channels, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1)
end

@testset "Orbital ladder crossing: explicit incomplete Matsubara rejection" begin
    f = _fixture(frequencies=[-1, 0], transfers=[0])
    channels = P.OrbitalSpinLadderChannels(f.local_d, f.local_m, f.density, f.magnetic, f.maps;
        spin_symmetry_certified=true)
    @test !P.orbital_crossing_coverage(channels).complete
    @test P.orbital_crossing_coverage(channels).missing > 0
    @test_throws ArgumentError P.orbital_crossed_vertex(channels, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1)
end

@testset "Orbital ladder crossing: independently constructed Bethe-Salpeter channels" begin
    f = _fixture(orbitals=1)
    zeroset = zeros(ComplexF64, 1, 1, 1, 1, 1)
    bubble = P.OrbitalPairSusceptibility(f.faxis, f.baxis, f.pairs, f.labels, reshape(ComplexF64[-1.0], 1, 1, 1, 1, 1))
    q_bubbles = [bubble, P.OrbitalPairSusceptibility(f.faxis, f.baxis, f.pairs, f.labels,
        reshape(ComplexF64[-1.3], 1, 1, 1, 1, 1))]
    vertex_d = P.OrbitalPairIrreducibleVertex(f.faxis, f.baxis, f.pairs, f.labels, zeroset .+ 0.06, Dict{Symbol,Any}())
    vertex_m = P.OrbitalPairIrreducibleVertex(f.faxis, f.baxis, f.pairs, f.labels, zeroset .- 0.03, Dict{Symbol,Any}())
    ladder = P.orbital_spin_ladders(bubble, vertex_d, vertex_m, q_bubbles, f.maps;
        spin_symmetry_certified=true)
    @test ladder.density[1].values[1] ≈ 0.06 / 1.06 atol=1e-12
    @test ladder.magnetic[1].values[1] ≈ -0.03 / 0.97 atol=1e-12
    @test ladder.density[2].values[1] ≈ 0.06 / (1 + 1.3 * 0.06) atol=1e-12
    @test_throws ArgumentError P.orbital_spin_ladders(bubble, vertex_d, vertex_m, q_bubbles, f.maps;
        spin_symmetry_certified=true, max_retained_bytes=1)
end

@testset "Orbital ladder crossing: local SU(2) Hamiltonian certificate" begin
    U = P.kanamori_interaction(2; U=2.0, J=0.25)
    onsite = ComplexF64[0.3 0 0.1 0; 0 0.3 0 0.1; 0.1 0 -0.2 0; 0 0.1 0 -0.2]
    check = P.orbital_su2_symmetry(onsite, U)
    @test check.valid
    @test check.relative_residual <= 1e-13
    spin_split = copy(onsite)
    spin_split[1, 1] += 0.2
    @test !P.orbital_su2_symmetry(spin_split, U).valid
    @test_throws DimensionMismatch P.orbital_su2_symmetry(ones(3, 3), U)
end

end # module MultiorbitalCrossingValidation
