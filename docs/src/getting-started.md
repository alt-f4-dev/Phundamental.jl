# Getting Started

This page gives the shortest path from a source checkout to a model, solver result, and high-level forward calculation. The module pages contain the detailed mathematical conventions and complete API organization.

## Loading Phundamental.jl

For an installed stable release, load the package normally:

```julia
using Phundamental
```

To work from a source checkout, instantiate the root environment and start Julia with that project active:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=.
```

Then load the package with `using Phundamental`. The public namespaces are `Foundation`, `Algebra`, `Crystallography`, `Transformations`, `Models`, `Solvers`, `Observables`, `Thermodynamics`, `Greens`, and `Embedding`.

## Construct a simple spin model

A minimal finite model requires geometry, material metadata, a Hamiltonian specification, and a finite cluster. The following one-dimensional example keeps those stages explicit:

```julia
const M = Phundamental.Models

lattice = BravaisLattice(reshape([1.0], 1, 1))

basis_sites = [BasisSite(:X, :X, [0.0])]
crystal = LatticeCrystalStructure(lattice, basis_sites)

species = Dict(:X => AtomicSpecies(:X; mass=1.0))
props = [SiteProperties(spin=1//2)]

material = Material("spin chain", crystal; species=species, basis_properties=props)

cluster = Supercell(crystal, [8]; periodic=[false])

model = build_model(material, cluster, XXZModel(1.0, 0.6; neighbor_cutoff=1.01))
```

`build_model` returns a `ManyBodyModel` that contains both the symbolic Hamiltonian and the representation required to interpret its operators. Geometry and material information are retained in model metadata so later solver and observable layers do not need to reconstruct it independently.

## Inspect the representation before solving

A `ManyBodyModel` carries a `Representation` rather than only a Hamiltonian expression:

```julia
rep = model.representation
rep.state_space
rep.algebra
rep.basis
rep.constraints
rep.truncation
```

Use `numerical_capabilities(model)` before a generic finite-basis calculation when the representation may contain bosons, coordinates, transported constraints, or other nontrivial numerical requirements:

```julia
caps = numerical_capabilities(model)
println(caps.numerically_realizable)
println(caps.reasons)
```

A numerical boson cutoff is attached with `with_truncation`; it is not encoded as a `PhysicalSubspace`:

```julia
truncated = with_truncation(model, NumericalTruncation(8))
```

## Inspect physical model-space admissibility

The model-space API can evaluate physical/model-space admissibility before any solver is selected. A `GenerativeSpecification` declares the operator algebra and physical restrictions, while `admissibility_report` records each evaluated criterion and distinguishes rejection from incomplete information.

```julia
space = SpinHilbertSpace([1//2])
algebra = SpinAlgebra(1)
spec = GenerativeSpecification(:C1, (:site_1,), (space,), algebra, identity; max_body_order=1)
report = admissibility_report(spec, Sz(1))
println(report.complete)
println(report.admissible)
```

A declared `max_range` requires an explicit `range_evaluator(candidate)` that returns a physical distance. The package does not reinterpret symbolic site or mode indices as geometry. `solver_compatibility` remains a later and independent computational question.

## Define a constraint-bearing parameter schema

The existing `ParameterSpace` serves as the coupling schema attached to a `HamiltonianSpace`. The schema can enforce bounds together with fixed, shared, or general linear relations among named real coefficients.

```julia
spin_space = SpinHilbertSpace([1//2])
algebra = SpinAlgebra(1)
spec = GenerativeSpecification(:C1, (:site_1,), (spin_space,), algebra, identity; max_body_order=1)
basis = HamiltonianBasis(AbstractOperatorExpr[Sz(1), Sx(1)]; labels=[:hz, :hx])
hamiltonian_space = generate_hamiltonian_space(spec, _ -> basis)

schema = ParameterSpace(
    hamiltonian_space;
    values=[0.5, 0.5],
    bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    constraints=(EqualParameterConstraint(:hz, :hx),),
)

point = ParameterPoint(schema)
println(parameter_constraint_report(point).valid)
println(parameter_dof(schema))
```

`parameter_binding(schema, :hz)` identifies the Hamiltonian-basis direction multiplied by `:hz` through its canonical `StructuralSignature`. `parameter_constraint_matrix(schema)` exposes the affine equalities as $A\boldsymbol\theta=\mathbf b$ for later inverse machinery. These constraints are independent of physical model-space admissibility and solver compatibility.

## Generate a Hamiltonian family from material crystallography

Material-aware Hamiltonian generation connects material geometry to the physical model-space, structural, and parameter layers. For a three-dimensional material, `generate_hamiltonian_family` converts the lattice geometry to the crystallography layer, discovers or accepts a space-group dataset, projects candidate operators into the invariant Hermitian subspace, audits physical admissibility, extracts canonical structure, and creates the corresponding `ParameterSpace`.

```julia
using LinearAlgebra

lattice3 = BravaisLattice(Matrix{Float64}(I, 3, 3))
crystal3 = LatticeCrystalStructure(lattice3, [BasisSite(:A, :A, [0.0, 0.0, 0.0]), BasisSite(:B, :B, [0.4, 0.0, 0.0])])
material3 = Material(
    "two-site magnet",
    crystal3;
    species=Dict(:A => AtomicSpecies(:A; mass=1.0), :B => AtomicSpecies(:B; mass=1.0)),
    basis_properties=[SiteProperties(spin=1//2), SiteProperties(spin=1//2)],
)

family = generate_hamiltonian_family(material3, SpinAlgebra(2); max_body_order=2, max_range=0.6)
family.specification
family.hamiltonian_space
family.model_structure
family.parameter_space
```

The standard symmetry action assumes one operator index per crystallographic site. Supply a custom `symmetry_action` and explicit `local_spaces` for multi-orbital or other internal-index conventions. If translated sites must be distinct Hamiltonian supports, use the `generate_hamiltonian_family(material, periodic_supercell, algebra; ...)` overload rather than the primitive-cell-only route. See [Universal Hamiltonian Generation](workflows/universal-hamiltonian-generation.md) for the full contract.

## Choose a solver

The generic reference solvers are useful for validation and small problems:

```julia
ed = solve(ExactDiagonalization(), model)
println(groundenergy(ed))
```

For larger exact finite systems, use the optimized path when the model has a supported packed backend:

```julia
result = solve(OptimizedLanczos(), model)
```

`AutoSolver` may select an exact optimized representation or an exact harmonic/quadratic mode solver when the input model already has that structure. It does not silently choose a physical approximation.

```julia
result = solve(AutoSolver(), model)
```

Approximations such as linear spin-wave theory are explicit solver choices:

```julia
sw = solve(LinearSpinWaveSolver(reference_signs=ones(Int, nsites(cluster))), model)
```

## Representation transformations

Transformations act on the model representation, Hamiltonian, and registered observables and return a `TransformationResult` with a mathematical certificate and validation metadata:

```julia
mapped = transform(JordanWigner(1:8), model)
fermion_model = mapped.model
mapped.certificate
mapped.validation
```

Use `transformation_capabilities` and `numerical_capabilities` when the target representation may require a target-basis cutoff or an explicit projector for transported constraints.

## Intrinsic observables, probes, and resolution

Phundamental keeps three stages separate:

$$\text{intrinsic observable}\rightarrow\text{probe response}\rightarrow\text{instrument response}$$

An intrinsic spectrum may be broadened numerically as part of its definition, then contracted with a physical probe, and finally convolved with an instrumental resolution model. This prevents numerical linewidth and experimental resolution from becoming the same parameter by accident.

## The high-level forward API

A typed `ForwardProblem` packages the whole calculation while keeping each stage inspectable:

```julia
obs = ObservableRequest(:ground_energy, (m, solution) -> groundenergy(solution))
problem = ForwardProblem(model; solver=ExactDiagonalization(), observable=obs)
result = forward(problem)
println(result.observable_result)
```

For generic scalar dynamical response, `DynamicalCorrelationRequest(field, qgrid, axis)` requests an autocorrelation and `DynamicalCorrelationRequest(field_a, field_b, qgrid, axis)` requests an ordered cross-correlation. These typed requests can be passed directly to the same forward pipeline.

For coupled models with several response channels, `CompositeDynamicalCorrelationRequest((channel_a=field_a, channel_b=field_b, ...), qgrid, axis)` returns the complete ordered channel matrix. `SpinPhononCorrelationRequest(qgrid, axis)` provides the standard spin-plus-displacement specialization and resolves its fields from the model's observable registry.

For scattering workflows, pass an observable request, probe, and resolution separately:

```julia
result = forward(model; solver=OptimizedLanczos(), observable=observable_request, probe=MagneticNeutronProbe(), resolution=GaussianResolution(0.08))
```

## Canonical forward demo

The repository-level script `demos/high-level-forward-api.jl` is the canonical executable example of the stable high-level API. It constructs a two-site Heisenberg model, creates a typed spin-tensor scattering request, asks `plan_forward` to select a certified exact route, inspects `planning_report(plan)`, executes `forward(plan)`, and keeps the intrinsic, probe-contracted, and resolution-convolved responses separate.

Run it from the repository root with:

```bash
julia --project=. demos/high-level-forward-api.jl
```

## Finite-temperature Green functions

A Matsubara axis stores both the inverse temperature and integer fermionic frequency indices:

```julia
axis = FermionicMatsubaraAxis(32.0, 128)
G = matsubara_green(model; axis=axis)
```

The temperature used by an exact thermal state must match the axis. Use `include_negative=true` when validating identities that require paired positive and negative Matsubara frequencies:

```julia
paired_axis = FermionicMatsubaraAxis(32.0, 64; include_negative=true)
```

## ED-DMFT

The validated embedding layer implements normal-state single-site ED-DMFT. A typical half-filled Bethe problem uses an explicit lattice, Matsubara axis, finite-bath impurity solver, convergence policy, and mixing policy:

```julia
beta = 32.0; axis = FermionicMatsubaraAxis(beta, 128)

problem = DMFTProblem(BetheLattice(sqrt(2)), 3.0; temperature=inv(beta), chemical_potential=1.5, impurity_energy=0.0, axis=axis)

impurity_solver = EDImpuritySolver(bath_sites=4, bath_fit=BathFitOptions(symmetry=:particle_hole), backend=:auto, threading=:auto)

solver = DMFTSolver(impurity_solver=impurity_solver, mixing=AdaptiveLinearMixing(), tol=1e-7, verify_closure=true)

result = solve(solver, problem)
```

Use `validate_dmft_result(result)` to collect fixed-point, closure, causality, particle-hole, bath-discretization, and limiting-case diagnostics.

!!! warning
    `quasiparticle_weight_estimate` is a low-frequency Fermi-liquid diagnostic. It should not be interpreted as a quasiparticle weight when the self-energy is insulating or otherwise non-Fermi-liquid.

## Running the validation suite

The frozen API is validated through the repository test suite:

```bash
julia --project=. test/runtests.jl
```

The ED-DMFT scientific benchmarks are intentionally separate from unit tests:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/ed-dmft-physical-regimes-benchmark.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/dmft/caffarel-krauth-1994-benchmark.jl
```

See [Validation](validation.md) for what these tests establish and what remains outside the documented validation scope.
