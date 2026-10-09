using Test
using Phundamental
using LinearAlgebra

function square_site_expansion(repetitions::Int, max_order::Int)
    lattice = BravaisLattice(Matrix{Float64}(I, 2, 2))
    basis = [BasisSite(:A, :spin, [0.0, 0.0])]
    crystal = LatticeCrystalStructure(lattice, basis)
    reference = Supercell(crystal, [repetitions, repetitions]; periodic=true)
    bonds = neighbor_bonds(reference; cutoff=1.01, max_shell=1)
    return site_nlce_expansion(reference, bonds; max_order=max_order, max_clusters=100_000, allow_periodic_reference=true)
end

@testset "site NLCE topology and published Rigol benchmark" begin
    expansion7 = square_site_expansion(7, 5)
    benchmark = validate_rigol_2007_square_site_counts(expansion7; max_order=5)
    @test benchmark.passed
    @test summarize_topological_expansion(expansion7).topological_cluster_count == [1, 1, 1, 3, 4]
    @test summarize_topological_expansion(expansion7).sum_lattice_constants ≈ [1.0, 2.0, 6.0, 19.0, 63.0]

    expansion8 = square_site_expansion(8, 5)
    stability = validate_embedding_stability(expansion7, expansion8)
    @test stability.stable
    @test stability.max_multiplicity_residual ≤ 1e-12
end

@testset "Wynn epsilon NLCE acceleration" begin
    partial_sums = Any[1.0, 0.5, 0.75, 0.625, 0.6875, 0.65625, 0.671875]
    result = NLCEResult(Any[], Dict{Any,Any}(), Dict{Any,Any}(), partial_sums, fill(NaN, length(partial_sums)), Dict{Symbol,Any}())
    accelerated = resum_nlce(result, WynnEpsilon(cycles=3))
    @test accelerated.metadata[:completed_cycles] ≥ 1
    @test abs(accelerated.estimate - 2 / 3) < abs(partial_sums[end] - 2 / 3)
end
