using Phundamental
using Printf

const t = 1.0
const U = 0.0
const omega = 1.0
const g = 0.5
const eta = 0.03
const phonon_cutoff = 6

lattice = BravaisLattice(reshape([1.0], 1, 1))
crystal = LatticeCrystalStructure(lattice, [BasisSite(:X, :X, [0.0])])
cluster = Supercell(crystal, [2]; periodic=false)
material = Material("Holstein dimer", crystal; species=Dict(:X => AtomicSpecies(:X; mass=1.0)))

model = build_model(material, cluster, HolsteinModel(t, U, omega, g; phonon_cutoff=phonon_cutoff, shell=1))
sector = FermionSector(nup=1, ndown=1)

axis = collect(range(-10.0, 10.0; length=2001))
ground_dimension = 4 * (phonon_cutoff + 1)^2
green_dimension = 2 * (phonon_cutoff + 1)^2

solver = OptimizedLanczos(nev=1, krylov_dim=ground_dimension, tol=1e-12, sector=sector, backend=:fermion, reorthogonalize=true, seed=17)
observable = LocalElectronSpectralFunctionRequest(axis; spin=:up, sites=[1, 2], eta=eta, normalization=:spectral_density, green_krylov_dim=green_dimension, tol=1e-12, green_reorthogonalize=true, parallel=false)

calculation = forward(ForwardProblem(model; solver=solver, observable=observable))
result = calculation.result
workspace = result.metadata[:workspace_metadata]

reference_E0 = -2.593448076046033
site_symmetry_error = maximum(abs.(result.intensity[1, :] .- result.intensity[2, :]))

println()
println(" --------------------------------")
println("| Holstein Dimer — Terse Forward |")
println(" --------------------------------")
@printf("t, U, omega, g             : %.3f, %.3f, %.3f, %.3f\n", t, U, omega, g)
@printf("phonon cutoff              : %d\n", phonon_cutoff)
println("canonical sector           : (Nup, Ndown) = (1, 1)")
@printf("ground energy              : %.12f\n", groundenergy(calculation.solution))
@printf("benchmark ground energy    : %.12f\n", reference_E0)
@printf("ground-energy error        : %.3e\n", abs(groundenergy(calculation.solution) - reference_E0))
println("sector dimensions          : ", (ground=workspace[:ground_dimension], particle=workspace[:particle_dimension], hole=workspace[:hole_dimension]))
println("sites                      : ", result.sites)
println("ground solution reused     : ", result.metadata[:ground_solution_reused])
@printf("site-spectrum symmetry err : %.3e\n", site_symmetry_error)
println("spectral-axis range        : ", extrema(result.axis))
println("intensity shape            : ", size(result.intensity))
