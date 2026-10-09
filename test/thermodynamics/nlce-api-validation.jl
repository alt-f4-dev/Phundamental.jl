using Test
using Phundamental

@testset "generic NLCE bookkeeping" begin
    classes = [
        TopologicalClusterClass((1,), 1, 1.0),
        TopologicalClusterClass((1, 2), 2, 1.0; subcluster_multiplicities=Dict((1,) => 2)),
    ]
    expansion = TopologicalExpansion(classes)
    result = topological_nlce(Dict((1,) => 2.0, (1, 2) => 5.0), expansion)
    @test result.weights[(1,)] == 2.0
    @test result.weights[(1, 2)] == 1.0
    @test bulk_estimate(result, 1) == 2.0
    @test bulk_estimate(result) == 3.0
    @test correlation_scope(result) == :short_intermediate_spatial
    @test validate_nlce_result(result).valid
end
