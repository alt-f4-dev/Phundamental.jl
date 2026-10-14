# Batch 11 exact structural-enumeration benchmark. Run from the package root with:
# julia --project=. benchmarks/structural-search-scaling-benchmark.jl

using Phundamental
using Printf

function benchmark_structural_enumeration(n::Int; max_structures::Int=4095)
    algebra = SpinAlgebra(n)
    specification = GenerativeSpecification(:C1, Tuple(1:n), Tuple(SpinHilbertSpace([1//2]) for _ in 1:n),
                                             algebra, identity; max_body_order=1)
    directions = AbstractOperatorExpr[Sz(i) for i in 1:n]
    basis = HamiltonianBasis(directions; metadata=Dict(:completeness_scope => :controlled_benchmark_basis))
    parent = generate_hamiltonian_space(specification, _ -> basis)
    family = structural_family(parent)
    expected = big(2)^n - 1
    expected <= max_structures || throw(ArgumentError("requested n=$(n) needs $(expected) structures, beyond max_structures"))
    selections = StructuralSelection[]
    elapsed = @elapsed (selections = enumerate_structures(family; max_structures=max_structures))
    observed = length(selections)
    observed == expected || error("expected $(expected) structures, got $(observed)")
    length(unique(selection.signature for selection in selections)) == expected || error("duplicate structural signatures")
    return (units=n, expected=Int(expected), observed=observed, seconds=elapsed)
end

println("Batch 11 structural-enumeration correctness and scaling")
@printf("%7s  %12s  %12s  %12s\n", "units", "expected", "observed", "seconds")
for n in (4, 8, 12)
    result = benchmark_structural_enumeration(n)
    @printf("%7d  %12d  %12d  %12.6f\n", result.units, result.expected, result.observed, result.seconds)
end
