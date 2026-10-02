using Phundamental
using Printf

const t = 0.5
const U = 1.0
const eta = 0.05

lattice = BravaisLattice(reshape([2.0], 1, 1))

basis = BasisSite[BasisSite(:H1, :H, [0.0]),
                  BasisSite(:H2, :H, [0.5])]

crystal = LatticeCrystalStructure(lattice, basis)
cluster = Supercell(crystal, (1,); periodic=false)

material = Material("Carrascal symmetric Hubbard dimer", crystal;
                    species=Dict{Symbol,Any}(:H => lookup_species(:H)),
                    metadata=Dict(:reference => "Carrascal et al., J. Phys.: Condens. Matter 27, 393001 (2015)",
                                  :doi => "10.1088/0953-8984/27/39/393001"))

model = build_model(material, cluster, HubbardModel(t, U; shell=1))

sector = FermionSector(nup=1, ndown=1)

energy = collect(range(-2.0, 3.0; length=2501))

solver = OptimizedLanczos(nev=1, sector=sector, backend=:fermion, reorthogonalize=true)

observable = LocalElectronSpectralFunctionRequest(energy; spin=:up, sites=[1], eta=eta, green_krylov_dim=2, green_reorthogonalize=true, parallel=false)


problem = ForwardProblem(model; solver=solver, observable=observable)
calculation = forward(problem)


intensity = vec(calculation.result.intensity[1, :])

reference_E0 = U / 2 - sqrt((2t)^2 + (U / 2)^2)
reference_poles = sort([reference_E0 - t, 
                        reference_E0 + t,
                        U - t - reference_E0, 
                        U + t - reference_E0 ])

println()
println(" --------------------------------")
println("| Hubbard Dimer — Terse Forward  |")
println(" --------------------------------")
@printf("t                         : %.6f\n", t)
@printf("U                         : %.6f\n", U)
@printf("Lorentzian eta            : %.6f\n", eta)
println("canonical sector          : (Nup, Ndown) = (1, 1)")
println("reference poles           : ", round.(reference_poles; digits=9))
println("sites                     : ", calculation.result.sites)
println("ground solution reused    : ", calculation.result.metadata[:ground_solution_reused])
println("energy    (min, max)      : ", extrema(energy))
println("intensity (min, max)      : ", extrema(intensity))
