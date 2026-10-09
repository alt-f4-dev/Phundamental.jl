using Phundamental
using LinearAlgebra

# Canonical v0.0.8 high-level forward-architecture demo.
# The example keeps the physical model, observable request, execution plan, intrinsic response, probe contraction, and resolution stage separate.

const J = 1.0

# Physical material and two-site spin model.
lattice = BravaisLattice(Matrix{Float64}(I, 3, 3))
properties = SiteProperties(spin=0.5, g_tensor=2.0)
crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0, 0.0, 0.0]; properties=properties)])
parent = Supercell(crystal, (1, 1, 1); periodic=true)
material = Material("v0.0.8 high-level spin dimer", crystal; species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
cluster = bond_cluster(parent, first(neighbor_bonds(parent; cutoff=1.01, max_shell=1)))
model = build_model(material, cluster, HeisenbergModel(J; shell=1))

# Measurement definition.
positions = model.parameters[:positions]
dimer_vector = positions[2] - positions[1]
Q = (π / dot(dimer_vector, dimer_vector)) .* dimer_vector
axis = collect(range(0.0, 2.0; length=1001))
convention = SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=1.0)
request = SpinTensorStructureFactorRequest(Q, axis; temperature=0.0, convention=convention, broadening=LorentzianBroadening(0.03))
probe = MagneticNeutronProbe(material, cluster; site=1)
resolution = GaussianResolution(0.05)

# Deterministic policy-governed planning. The default policy considers only certified context-free exact routes.
plan = plan_forward(model, request; probe=probe, resolution=resolution)
report = planning_report(plan)

println("Selected route: ", report.selected_label)
println("Selected solver: ", report.selected_solver)
println("Authorized approximations: ", report.selected_approximations)
println("Certified route: ", report.selected_certified)

# Execute the already-inspected plan.
calculation = forward(plan)

println("Source model type: ", typeof(calculation.source_model))
println("Working model type: ", typeof(calculation.model))
println("Solver result type: ", typeof(calculation.solution))
println("Intrinsic response type: ", typeof(calculation.observable_result))
println("Probe response type: ", typeof(calculation.scattering_result))
println("Final response type: ", typeof(calculation.result))

intensity = vec(calculation.result.intensity)
peak_index = argmax(intensity)
println("Final peak energy: ", calculation.result.axis[peak_index])
println("Final peak intensity: ", intensity[peak_index])
