#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalBetheSalpeterValidation

using Test
using LinearAlgebra
const P = Main.Phundamental

function _lattice_fixture(; dimension=2)
    beta = 5.0
    extended = P.FermionicMatsubaraAxis(beta, [-1, 0, 1])
    selected = P.FermionicMatsubaraAxis(beta, [-1, 0])
    bosonic = P.BosonicMatsubaraAxis(beta, [0, 1])
    labels = [Symbol("mode_", i) for i in 1:dimension]
    cube = Array{ComplexF64}(undef, dimension, dimension, 2, length(extended))
    for ik in 1:2, n in 1:length(extended), b in 1:dimension, a in 1:dimension
        cube[a, b, ik, n] = if a == b
            (0.55 + 0.04a + 0.03ik) / (extended[n] + (0.7 + 0.09a))
        else
            (0.04 + 0.02im * (a-b)) / (extended[n] + 1.5 + 0.06ik)
        end
    end
    green = P.MatrixMomentumResolvedGreenFunction(extended, [:k1, :k2], cube; labels=labels, weights=[0.4, 0.6])
    maps = [[1, 2], [2, 1]]
    return (; green, maps, selected, bosonic, labels)
end

function _vertex(fixture; pairs=nothing, scale=0.02)
    chosen = isnothing(pairs) ? [(a, b) for a in 1:length(fixture.labels) for b in 1:length(fixture.labels)] : pairs
    np, nf, nb = length(chosen), length(fixture.selected), length(fixture.bosonic)
    values = zeros(ComplexF64, np, np, nf, nf, nb)
    for m in 1:nb, j in 1:np, i in 1:np, np2 in 1:nf, n in 1:nf
        values[i, j, n, np2, m] = scale * (i == j && n == np2 ? 1.0 : 0.04im * (i-j) + 0.01 * (n+np2+m))
    end
    return P.OrbitalPairIrreducibleVertex(fixture.selected, fixture.bosonic, chosen, fixture.labels, values, Dict{Symbol,Any}())
end

function _flatten(data, iΩ)
    np, _, nf = size(data)[1:3]
    output = Matrix{ComplexF64}(undef, np*nf, np*nf)
    for j in 1:np, i in 1:np, nprime in 1:nf, n in 1:nf
        output[(i-1)*nf+n, (j-1)*nf+nprime] = data[i, j, n, nprime, iΩ]
    end
    return output
end

@testset "Orbital ladder: independent complex lattice bubble and index convention" begin
    fixture = _lattice_fixture()
    responses = P.orbital_pair_lattice_bubble(fixture.green, fixture.maps, fixture.selected, fixture.bosonic)
    @test length(responses) == 2
    @test responses[1].pairs == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test responses[1].metadata[:frequency_box_complete]
    for (iq, mapping) in enumerate(fixture.maps), (iΩ, m) in enumerate(fixture.bosonic.indices),
        (iν, n) in enumerate(fixture.selected.indices), (j, (c, d)) in enumerate(responses[iq].pairs),
        (i, (a, b)) in enumerate(responses[iq].pairs)
        old = findfirst(==(n), fixture.green.axis.indices)
        shifted = findfirst(==(n + m), fixture.green.axis.indices)
        expected = -fixture.selected.beta * sum(fixture.green.weights[k] * fixture.green.values[d, a, k, old] *
            fixture.green.values[b, c, mapping[k], shifted] for k in 1:2)
        @test responses[iq].values[i, j, iν, iν, iΩ] ≈ expected atol=1e-13 rtol=1e-13
    end
    @test all(iszero, responses[1].values[:, :, 1, 2, :])
    legacy, _ = P.orbital_lattice_bubble(fixture.green, fixture.maps, fixture.bosonic; missing_frequency=:zero)
    adapter = P.orbital_pair_lattice_bubble(legacy[2], fixture.selected)
    @test adapter.values ≈ responses[2].values atol=1e-12 rtol=1e-12
    @test_throws ArgumentError P.orbital_pair_lattice_bubble(legacy[1], fixture.green.axis)
    @test_throws ArgumentError P.orbital_pair_lattice_bubble(fixture.green, fixture.maps, fixture.green.axis, fixture.bosonic)
    @test_throws ArgumentError P.orbital_pair_lattice_bubble(fixture.green, fixture.maps, fixture.selected, fixture.bosonic; max_output_bytes=1)
end

@testset "Orbital ladder: LU versus independent dense reference and zero vertex" begin
    fixture = _lattice_fixture()
    vertex = _vertex(fixture)
    bubbles = P.orbital_pair_lattice_bubble(fixture.green, fixture.maps, fixture.selected, fixture.bosonic)
    response = P.bethe_salpeter(bubbles[1], vertex)
    @test response isa P.OrbitalPairSusceptibility
    @test response.metadata[:inverse_formed] == false
    for iΩ in eachindex(fixture.bosonic.indices)
        B = _flatten(bubbles[1].values, iΩ)
        V = _flatten(vertex.values, iΩ)
        expected = (Matrix{ComplexF64}(I, size(B, 1), size(B, 2)) - B * V) \ B
        @test _flatten(response.values, iΩ) ≈ expected atol=1e-11 rtol=1e-11
        @test response.metadata[:diagnostics][iΩ].relative_residual < 1e-12
    end
    zero_vertex = _vertex(fixture; scale=0.0)
    @test P.bethe_salpeter(bubbles[1], zero_vertex).values ≈ bubbles[1].values atol=1e-13
    single_axis = P.FermionicMatsubaraAxis(5.0, [0])
    zero_transfer = P.BosonicMatsubaraAxis(5.0, [0])
    single_bubble = P.OrbitalPairSusceptibility(single_axis, zero_transfer, [(1, 1)], [:only], ones(ComplexF64, 1, 1, 1, 1, 1))
    singular_vertex = P.OrbitalPairIrreducibleVertex(single_axis, zero_transfer, [(1, 1)], [:only],
        ones(ComplexF64, 1, 1, 1, 1, 1), Dict{Symbol,Any}())
    @test_throws LinearAlgebra.SingularException P.bethe_salpeter(single_bubble, singular_vertex)
    @test_throws ArgumentError P.bethe_salpeter(bubbles[1], vertex; max_workspace_bytes=1)
    @test_throws ArgumentError P.bethe_salpeter(bubbles[1], vertex; max_output_bytes=1)
    wrong = P.OrbitalPairIrreducibleVertex(fixture.selected, fixture.bosonic, reverse(vertex.pairs), fixture.labels,
        vertex.values, Dict{Symbol,Any}())
    @test_throws ArgumentError P.bethe_salpeter(bubbles[1], wrong)
end

@testset "Orbital ladder: synthetic local vertex reconstruction" begin
    fixture = _lattice_fixture(; dimension=1)
    bubble = only(P.orbital_pair_lattice_bubble(fixture.green, [fixture.maps[1]], fixture.selected, fixture.bosonic))
    responses = similar(bubble.values)
    for iΩ in 1:length(fixture.bosonic)
        B = _flatten(bubble.values, iΩ)
        target = (Matrix{ComplexF64}(I, 2, 2) - 0.015 * B) \ B
        responses[1, 1, :, :, iΩ] .= target
    end
    target = P.OrbitalPairSusceptibility(fixture.selected, fixture.bosonic, bubble.pairs, bubble.labels, responses)
    extracted = P.multiorbital_irreducible_vertex(bubble, target)
    recovered = P.bethe_salpeter(bubble, extracted)
    @test recovered.values ≈ target.values atol=1e-10 rtol=1e-10
    @test all(item -> item.relative_residual < 1e-12, recovered.metadata[:diagnostics])
end

@testset "Orbital ladder: momentum streaming, contraction, resource guards" begin
    fixture = _lattice_fixture()
    vertex = _vertex(fixture)
    weights = ComplexF64[1, 0.3im, -0.2, 0.7]
    full = P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex;
        projection_weights=weights, qpoints=[:zero, :edge])
    streamed = P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex;
        projection_weights=weights, return_full=false, qpoints=[:zero, :edge])
    @test full.qpoints == [:zero, :edge]
    @test isnothing(streamed.susceptibilities)
    @test streamed.contracted ≈ full.contracted atol=1e-12 rtol=1e-12
    @test size(streamed.contracted) == (2, 2)
    for iq in 1:2, iΩ in 1:2
        computed = sum(conj(weights[i]) * weights[j] * sum(full.susceptibilities[iq].values[i, j, :, :, iΩ])
            for j in 1:4 for i in 1:4) / fixture.selected.beta^2
        @test full.contracted[iq, iΩ] ≈ computed atol=1e-12 rtol=1e-12
    end
    @test all(info -> info.relative_residual < 1e-12, vcat(streamed.diagnostics...))
    @test_throws ArgumentError P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex; return_full=false)
    @test_throws ArgumentError P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex; max_workspace_bytes=1)
    @test_throws ArgumentError P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex; max_output_bytes=1)
    @test_throws DimensionMismatch P.multiorbital_ladder_susceptibility(fixture.green, fixture.maps, vertex; qpoints=[:wrong])
end

@testset "Orbital ladder: scalar reduction and projected subspace contract" begin
    fixture = _lattice_fixture(; dimension=1)
    vertex = _vertex(fixture; scale=0.02)
    bubble = only(P.orbital_pair_lattice_bubble(fixture.green, [fixture.maps[1]], fixture.selected, fixture.bosonic))
    actual = P.bethe_salpeter(bubble, vertex)
    scalar_bubble = P.GeneralizedSusceptibility(fixture.selected, fixture.bosonic, dropdims(bubble.values; dims=(1, 2)))
    scalar_vertex = P.IrreducibleVertex(fixture.selected, fixture.bosonic, dropdims(vertex.values; dims=(1, 2)))
    scalar = P.bethe_salpeter(scalar_bubble, scalar_vertex)
    @test dropdims(actual.values; dims=(1, 2)) ≈ scalar.values atol=1e-11 rtol=1e-11
    mixed = _lattice_fixture()
    pairs = [(1, 1), (2, 2)]
    reduced = _vertex(mixed; pairs=pairs)
    projected_bubble = only(P.orbital_pair_lattice_bubble(mixed.green, [mixed.maps[1]], mixed.selected, mixed.bosonic; pairs=pairs))
    @test_throws ArgumentError P.bethe_salpeter(projected_bubble, reduced)
    @test P.bethe_salpeter(projected_bubble, reduced; allow_projected_pairs=true).metadata[:allow_projected_pairs]
    @test_throws ArgumentError P.multiorbital_ladder_susceptibility(mixed.green, mixed.maps, reduced)
end

end # module MultiorbitalBetheSalpeterValidation
