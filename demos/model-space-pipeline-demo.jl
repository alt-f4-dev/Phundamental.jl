#-----------------------------------------------------------------------------------------------#
#                                                                                               #
# Symbolic-to-numerical Hamiltonian model-space demonstration for Phundamental.jl.              #
#                                                                                               #
#                                                                                               #
#                                                                                               #
# The demonstration follows:                                                                    #
#                                                                                               #
#     G -> H(G) -> H -> R -> computational basis -> matrix realization -> numerical result      #
#                                                                                               #
#                                                                                               #
# The Hamiltonian-selection/inference problem is intentionally skipped.                         #
# A concrete coefficient vector is supplied directly after H(G) is generated.                   #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#


# Load module (local)
include(joinpath(@__DIR__, "Phundamental.jl"))
using .Phundamental



using LinearAlgebra #math



# Internal Namespaces
const PC = Phundamental.PhundamentalCore # low-level representations/symbolics
const PS = Phundamental.Solvers          # numerical objects/solvers


# Helper: Stage Title
function stage(number::Integer, title::AbstractString)
    println()
    println(repeat("=", 120))
    println("Stage $(number): $(title)")
    println(repeat("=", 120))
end

# Helper: Prints symbolic operator basis 
function show_operator_basis(space::HamiltonianSpace)
    for (index, (label, operator)) in enumerate(zip(space.basis.labels, space.basis.operators))
        println("  Phi[$(index)]  $(label) = $(operator)")
    end
end

# Helper: Stores a spin-product state for S=1/2
function spin_half_state_label(state::PS.SpinBasisState)
    labels = map(level -> level == 0 ? "↓" : "↑", state.levels)
    return "|" * join(labels) * ">"
end


#------------------------------#
#           Stage 1            #
#                              #
# Define the Crystal Structure #
#------------------------------#

stage(1, "Define the crystal structure")


# 
# Define a generic triclinic lattice with two identical sites occur at r and -r. 
# The spatial symmetry is inversion only such that the space group is P-1 (#2).
#            _          _
# lattice = | a₁  b₁  c₁ | = [aᵀ, bᵀ, cᵀ]
#           | a₂  b₂  c₂ |
#           | a₃  b₃  c₃ |
#            -          -
#
lattice = [4.17 0.83 0.37;
           0.00 5.11 1.29;
           0.00 0.00 6.23]

#
# Define two fractional positions with inversion symmetry (r₁ = -r₂)
# positions = [r₁, r₂]
#
r = [0.123, 0.234, 0.345]; positions = [r, mod.(-r, 1.0)]

#
# Spglib-compatible integer species identifiers
#
species = [1, 1]

#
# Construct a two site crystal structure
# 
#       (:A) r₁ = structure.positions[:,1]
#       (:B) r₂ = structure.positions[:,2]
#
structure = CrystalStructure(lattice, positions, species; labels=[:A, :B])

#
# Precision and tolerance policy for crystal symmetries
#
#       symprec → geometric symmetry precision tolerance for Spglib
#       
#       site_tolerance → precision tolerance for matching transformed fractional sites onto structure sites
#
#       linear_tolerance → precision tolerance for projected operators being numerically zero or linearly dependent
#
options = CrystallographyOptions(symprec=1e-6, site_tolerance=1e-8, linear_tolerance=1e-12)


# Helper: prints fractional positions 
println("Fractional positions:")
for (index, label) in enumerate(structure.labels)
    println("  $(label): ", structure.positions[:, index])
end


#-----------------------------------------------------------------------#
#                             Stage 2                                   #
#                                                                       #
# Define Local Quantum Algebra & Construct Crystallographic Generator 𝔊 #
#-----------------------------------------------------------------------#
stage(2, "Define the local quantum data and construct the generative specification G")

# 
# The local Hilbert-space descriptors belong to G. 
#
#       PC.SpinHilbertSpace[1//2] ∼ ℋᵢ = span{|↓⟩,|↑⟩}
#
#       local_spaces ∼ {ℋᵢ} = (ℋ₁, ℋ₂)
#
local_spaces = (PC.SpinHilbertSpace([1//2]), PC.SpinHilbertSpace([1//2]))

#
# Define the symbolic operator algebra for two spin sites
#       
#       Generators: {S₁ˣ, S₁ʸ, S₁ᶻ, S₂ˣ, S₂ʸ, S₂ᶻ}
#
spin_algebra = PC.SpinAlgebra(2)

# 
# crystallographic_specification(...) calls Spglib.jl, 
# copies the discovered symmetry into Phundamental-owned types, and constructs 𝔊.
#
# Inputs:
#       - crystal structure
#       - local Hilbert spaces
#       - operator algebra
#
#
# `max_body_order = 2` ⟹ intended model contains interactions with (at most) two local operators 
#
G = crystallographic_specification(structure, local_spaces, spin_algebra; symmetry_action=standard_symmetry_action, max_body_order=2, options=options, metadata=Dict(:demo => :spin_dimer))

#
# Extract space group information
#
dataset = G.space_group

println("Discovered space group number = ", dataset.spacegroup_number)
println("International symbol          = ", dataset.international_symbol)
println("Number of symmetry operations = ", length(dataset.operations))
println("Crystallographic orbits       = ", G.wyckoff_orbits)
println("Maximum body order            = ", G.max_body_order)

if dataset.spacegroup_number != 2
    error("The demonstration structure was expected to realize P-1 (#2), but Spglib found space group $(dataset.spacegroup_number).")
end
length(dataset.operations) == 2 || error("P-1 should provide exactly two spatial operations in this primitive setting.")



#--------------------------------------#
#               Stage 3                #
#                                      #
# Define Pre-Projection Operator Space #
#--------------------------------------#
stage(3, "Define the finite candidate operator space before symmetry projection")

#
# This candidate set defines the finite model-class scope for the demonstration. 
# The first four directions are inversion-even; the staggered field is inversion-odd and should be removed by projection.
#
exchange_x = PC.Sx(1) * PC.Sx(2) # X₁ = S₁ˣS₂ˣ
exchange_y = PC.Sy(1) * PC.Sy(2) # X₂ = S₁ʸS₂ʸ
exchange_z = PC.Sz(1) * PC.Sz(2) # X₃ = S₁ᶻS₂ᶻ
uniform_z = PC.Sz(1) + PC.Sz(2)  # X₄ = S₁ᶻ + S₂ᶻ
staggered_z = PC.Sz(1) - PC.Sz(2)# X₅ = S₁ᶻ - S₂ᶻ

#
# Candidate vector space 𝒱 = span{X₁, X₂, X₃, X₄, X₅}
#       
# Eventually, the API will allow this to be generated rather than defined explicitly.
#
candidates = PC.AbstractOperatorExpr[exchange_x, exchange_y, exchange_z, uniform_z, staggered_z]

println("Candidate operators:")
for (index, operator) in enumerate(candidates)
    println("  X[$(index)] = $(operator)")
end


#
# Explicit symmetry projection
#       
#       P[X] = ∑ᵧUᵧXUᵧ† / |𝔊|  for γ ∈ 𝔊
#
projected_staggered = reynolds_project(staggered_z, G, structure, dataset)
println()
println("Reynolds projection of the staggered field:")
println("  P_G[Sz(1) - Sz(2)] = ", projected_staggered)


#-------------------------------------------#
#                  Stage 4                  #
#                                           #
# Generate Permitted Hamiltonian Space ℋ(𝔊) #
#-------------------------------------------#
stage(4, "Generate the admissible Hamiltonian space H(G)")

basis_labels = [:exchange_x, :exchange_y, :exchange_z, :uniform_z]

#
# CrystallographicGenerator applies the spatial symmetry, enforces Hermiticity, removes linearly dependent directions, and returns a HamiltonianBasis.
#
generator = CrystallographicGenerator(structure, candidates; options=options, labels=basis_labels)

#
# generate_hamiltonian_space(G, generator; kwargs=...)
#       
#       This calls the generic model-space layer and executes 𝔊 + 𝒱 ↦ ℋ(𝔊)
#
Hspace = generate_hamiltonian_space(G, generator; metadata=Dict(:demo => :symbolic_to_numerical))




println("dim[H(G)] = ", model_space_dimension(Hspace))
println("Completeness scope = ", Hspace.basis.metadata[:completeness_scope])
println("Generated invariant basis:")
show_operator_basis(Hspace)

model_space_dimension(Hspace) == 4 || error("Expected four inversion-even Hamiltonian directions after projection.")
Hspace.basis.labels == basis_labels || error("Generated basis order differs from the demonstration's expected invariant ordering.")



#-----------------------------------------------------#
#                       Stage 5                       #
#                                                     #
# Embed Hamiltonian within Crystal Structure H ∈ ℋ(𝔊) #
#-----------------------------------------------------#

stage(5, "Embed a concrete Hamiltonian in H(G) without performing parameter inference")

# A numerical calculation requires one member of ℋ(𝔊), but no fitting or inference is performed here. 
# The physical coefficients are supplied directly.
J = 1.0
h = 0.30
theta = [J, J, J, -h]

#
# With the generated basis ordering, this coefficient vector defines the symbolic Hamiltonian:
#       
#       H = J S1.S2 - h(S1z + S2z).
#
H_symbolic = hamiltonian_from_coefficients(Hspace, theta)

println("Known coefficient vector θ = [J,J,J,-h]", theta)
println("Symbolic Hamiltonian:")
println("  H = ", H_symbolic)


#-----------------------------------------------#
#                     Stage 6                   #
#                                               #
# Define Full Symbolic Representation and Model #
#-----------------------------------------------#
stage(6, "Define the representation R and instantiate the model")

#
# Define the S=1/2 state space for the two-site system
#
state_space = PC.SpinHilbertSpace([1//2, 1//2])

#
# `representation` constructs the state space and basis for lowering symbolics to numerics
#
representation = Representation(:spin_product_z, state_space, spin_algebra, PC.SpinProductBasis([1, 2]))

#
# instantiate_model(...) binds the supplied coefficient vector θ to ℋ(𝔊) and carries 𝔊 and ℋ(𝔊) forward with the concrete model H ∈ ℋ(𝔊).
#
model = instantiate_model(Hspace, representation, theta)

println("Representation name = ", model.representation.name)
println("Parameter names     = ", parameter_names(model))
println("Parameter vector    = ", parameter_vector(model))
println("Model retains G     = ", model.specification === G)
println("Model retains H(G)  = ", model.model_space === Hspace)





#----------------------------------------------------#
#                       Stage 7                      #
#                                                    #
# Lower Symbolic Representation to Numerical Objects #
#----------------------------------------------------#
stage(7, "Lower the symbolic representation to a finite computational basis")


#
#convert representation to numerical basis
#
numerical_basis = PS.computational_basis(representation)

println("Computational-basis dimension = ", PS.basis_dimension(numerical_basis))
println("Basis ordering:")
for index in 1:PS.basis_dimension(numerical_basis)
    state = PS.basis_state(numerical_basis, index)
    println("  $(index): ", spin_half_state_label(state), "  levels=", state.levels)
end

PS.basis_dimension(numerical_basis) == 4 || error("Two spin-1/2 degrees of freedom should produce a four-dimensional basis.")



#------------------------------------#
#               Stage 8              #
#                                    #
# Evaluate the Numerical Hamiltonian #
#------------------------------------#
stage(8, "Realize the symbolic Hamiltonian as a numerical matrix")


#
# generate the Hamiltonian matrix with the computational basis 
#
matrix_result = PS.hamiltonian_matrix(model; basis=numerical_basis, sparse=false)
H_matrix = matrix_result.matrix

#
# compare numerical results to exact results
#
println("Numerical Hamiltonian matrix:")
show(stdout, "text/plain", real.(H_matrix))
println()

hermiticity_error = norm(H_matrix - adjoint(H_matrix)) / max(norm(H_matrix), eps(Float64))
println("Relative Hermiticity error = ", hermiticity_error)

# In the basis |↓↓>, |↓↑>, |↑↓>, |↑↑>, the Heisenberg dimer plus longitudinal field has this exact matrix.
H_expected = [J / 4 + h  0.0       0.0       0.0;
              0.0         -J / 4   J / 2     0.0;
              0.0          J / 2  -J / 4     0.0;
              0.0          0.0       0.0      J / 4 - h]

isapprox(H_matrix, H_expected; atol=1e-12, rtol=1e-12) || error("Numerical lowering does not match the analytic spin-dimer matrix.")


#-------------------------------------------------#
#                     Stage 9                     #
#                                                 #
# Evaluate Numerical Model via ForwardProblem API #
#-------------------------------------------------#
stage(9, "Evaluate the lowered model through the typed forward API")

#
# ObservableRequest(...) contains a symbolic name, evaluator function, and optional metadata.
#
# Here, the name is `:ground_energy` and the evaluator is `(_, solution) -> PS.groundenergy(solution)`
# 
# because the forward API expects `evaluator(model,solution)`. 
# 
# PS.groundenergy(solution) returns the first energy in the sorted solver result
#
ground_energy_request = ObservableRequest(:ground_energy, (_, solution) -> PS.groundenergy(solution))

#
# This constructs a typed description of the calculation before executing it.
# In this case, we request an ED is performed to produce the ground-state energy.
#
problem = ForwardProblem(model; solver=ExactDiagonalization(), observable=ground_energy_request)

# This executes the planned calculation of the problem
result = forward(problem)

# results
energies = sort(real.(result_energies(result.solution)))
expected_energies = sort([-3J / 4, J / 4 - h, J / 4, J / 4 + h])

println("Exact-diagonalization spectrum = ", energies)
println("Analytic spectrum              = ", expected_energies)
println("Ground-state energy            = ", result.observable_result)
println("Result capabilities            = ", result_capabilities(result.solution))

if !isapprox(energies, expected_energies; atol=1e-12, rtol=1e-12)
    error("Exact-diagonalization spectrum does not match the analytic dimer spectrum.")
end
if !isapprox(result.observable_result, minimum(expected_energies); atol=1e-12, rtol=1e-12)
    error("Ground-state energy does not match the analytic result.")
end

stage(10, "Pipeline summary")

println("𝔊                         : crystallographic + local quantum specification")
println("𝔊 -> ℋ(𝔊)                 : Spglib-backed symmetry discovery + Phundamental invariant projection")
println("H in ℋ(𝔊)                 : known coefficients inserted directly; no inference performed")
println("ℛ                         : two-spin product representation")
println("ℛ -> numerical basis      : finite spin computational basis")
println("H -> numerical matrix     : symbolic operator realization")
println("numerical matrix -> result: exact diagonalization + typed observable request")
println()
println("Symbolic-to-numerical model-space demonstration: PASS")
