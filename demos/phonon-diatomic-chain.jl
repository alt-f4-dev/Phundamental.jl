using Phundamental
using LinearAlgebra

A = 1.0
M = 2.0
m = 1.0
k = 4.0

lattice = BravaisLattice(reshape([A], 1, 1))

basis = BasisSite[BasisSite(:M, :M, [0.0]), 
                  BasisSite(:m, :m, [0.5])]


crystal = LatticeCrystalStructure(lattice, basis)


species = Dict(:M => AtomicSpecies(:M; mass=M, nuclear_scattering_length=1.0), 
               :m => AtomicSpecies(:m; mass=m, nuclear_scattering_length=1.0))

material = Material("Diatomic chain", crystal; species=species)

interactions = HarmonicBondInteraction[HarmonicBondInteraction(1, 2, [0]; longitudinal=k), 
                                       HarmonicBondInteraction(1, 2, [-1]; longitudinal=k)]

model = harmonic_model(material, interactions)


Q = range(0.05, 1.95; length=161) 
axis = range(-4.5, 4.5; length=1801)
convention = SpectrumConvention(spectral_axis=:omega, hbar=1.0, kB=1.0)


observable = PhononStructureFactorRequest(Q, axis; temperature=1.0, broadening=GaussianBroadening(0.03), convention=convention)
solver = HarmonicPhononSolver()
probe = NuclearNeutronProbe()


problem = ForwardProblem(model; solver=solver, observable=observable, probe=probe, resolution=GaussianResolution(0.05))
calculation = forward(problem)

println("q-point count      : ", length(calculation.result.q))
println("spectral-axis count: ", length(calculation.result.axis))
println("intensity shape    : ", size(calculation.result.intensity))
