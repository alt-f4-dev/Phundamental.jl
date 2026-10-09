using Phundamental
using Printf
using LinearAlgebra

max_order = parse(Int, get(ENV, "PHUNDAMENTAL_NLCE_RIGOL_MAX_ORDER", "5"))
repetitions = parse(Int, get(ENV, "PHUNDAMENTAL_NLCE_RIGOL_REFERENCE", string(max(7, max_order + 2))))
1 <= max_order <= 16 || error("PHUNDAMENTAL_NLCE_RIGOL_MAX_ORDER must lie in 1:16")
repetitions > max_order || error("reference linear size should exceed max_order to avoid periodic topological wrapping")

lattice = BravaisLattice(Matrix{Float64}(I, 2, 2))
basis = [BasisSite(:A, :spin, [0.0, 0.0])]
crystal = LatticeCrystalStructure(lattice, basis)
reference = Supercell(crystal, [repetitions, repetitions]; periodic=true)
bonds = neighbor_bonds(reference; cutoff=1.01, max_shell=1)
expansion = site_nlce_expansion(reference, bonds; max_order=max_order, max_clusters=2_000_000, allow_periodic_reference=true)
validation = validate_rigol_2007_square_site_counts(expansion; max_order=max_order)
summary = summarize_topological_expansion(expansion)
reference_data = rigol_2007_square_site_counts(; max_order=max_order)

println("Rigol-Bryant-Singh square-lattice site-NLCE benchmark")
println("Phys. Rev. E 75, 061118 (2007), DOI 10.1103/PhysRevE.75.061118, Table II")
println("reference = $(repetitions)x$(repetitions), max_order = $max_order")
println()
@printf("%5s %14s %14s %18s %18s\n", "order", "graphs", "published", "sum L(c)", "published")
for order in 1:max_order
    @printf("%5d %14d %14d %18.10g %18.10g\n", order, summary.topological_cluster_count[order],
            reference_data.topological_clusters.values[order], summary.sum_lattice_constants[order],
            reference_data.lattice_constants.values[order])
end
println()
println("PASS = ", validation.passed)
validation.passed || error("automatic site-NLCE topology/embedding benchmark does not reproduce Rigol et al. Table II")
