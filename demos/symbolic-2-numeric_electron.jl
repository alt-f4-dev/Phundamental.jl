# Compact symbolic-to-numerical model-space demo: inversion-symmetric spinless-electron dimer.
#
# Workflow:
#     G -> H(G) -> H -> R -> numerical matrix -> forward result
#
# The coefficients are supplied directly, so this demonstrates model construction and evaluation without an inference step.

include(joinpath(@__DIR__, "Phundamental.jl"))
using .Phundamental
using LinearAlgebra

const PC = Phundamental.PhundamentalCore # low-level representations/symbolics
const PS = Phundamental.Solvers          # numerical objects/solvers

# 1. Define G: a P-1 crystal with one scalar spinless-fermion orbital on each inversion-related site.
lattice = [4.17 0.83 0.37;
           0.00 5.11 1.29;
           0.00 0.00 6.23]

r = [0.123, 0.234, 0.345]

structure = CrystalStructure(lattice, [r, mod.(-r, 1.0)], [1, 1]; labels=[:A, :B])


algebra = PC.FermionAlgebra(2)
local_states = (PC.FermionFockSpace(1), PC.FermionFockSpace(2))
G = crystallographic_specification(structure, local_states, algebra; max_body_order=2)

@assert G.space_group.spacegroup_number == 2
println("G: ", G.space_group.international_symbol, " with ", length(G.space_group.operations), " symmetry operations")

# 2. Generate H(G) from a small number-conserving candidate space. Inversion removes the staggered density automatically.
hopping = PC.c(1)' * PC.c(2) + PC.c(2)' * PC.c(1)
density = PC.nf(1) + PC.nf(2)
staggered_density = PC.nf(1) - PC.nf(2)

Hspace = generate_crystallographic_hamiltonian_space(G, structure; candidates=[hopping, density, staggered_density])

@assert model_space_dimension(Hspace) == 2
println("H(G): dim = ", model_space_dimension(Hspace))
for (label, operator) in zip(Hspace.basis.labels, Hspace.basis.operators)
    println("  ", label, " = ", operator)
end

# 3. Insert known coefficients directly: H = -t(c1†c2 + c2†c1) - μ(n1 + n2).
t = 1.0; μ = 0.25; θ = [-t, -μ]

# 4. Define R and instantiate the selected member of H(G).
hilbert_space = PC.FermionFockSpace(2)

basis_states = PC.FermionOccupationBasis([1,2])

R = Representation(:fermion_occupation, hilbert_space, algebra, basis_states)

model = instantiate_model(Hspace, R, θ)

println("H = ", model.hamiltonian)

# 5. Lower the symbolic model to the finite fermion-occupation basis and realize its numerical Hamiltonian.
basis = PS.computational_basis(model)

H = PS.hamiltonian_matrix(model; basis=basis, sparse=false).matrix

println("numerical basis dimension = ", PS.basis_dimension(basis))
println("H matrix:")
show(stdout, "text/plain", real.(H))
println()
@assert isapprox(H, adjoint(H); atol=1e-12)

# 6. Evaluate the same model through the typed forward API.
ground_energy = ObservableRequest(:ground_energy, (_, solution) -> PS.groundenergy(solution))
result = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=ground_energy))

energies = sort(real.(result_energies(result.solution)))
expected = sort([0.0, -μ - t, -μ + t, -2μ])

println("spectrum = ", energies)
println("ground-state energy = ", result.observable_result)

@assert isapprox(energies, expected; atol=1e-12, rtol=1e-12)
println("Compact electron model-space pipeline: PASS")
