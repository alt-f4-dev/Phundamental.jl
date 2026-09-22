# Getting Started

This page gives the shortest path from the v0.0.7 source tree to a model, solver result, and high-level forward calculation. The module pages contain the detailed mathematical conventions and complete API organization.

## Loading v0.0.7

The frozen v0.0.7 repository is currently exercised in-place by loading `src/Phundamental.jl` under the active project environment:

```julia
include("src/Phundamental.jl")
using .Phundamental
```

The repository project contains the external `Spglib` dependency used by crystallographic symmetry discovery. From a fresh checkout, instantiate the root environment before running tests or calculations:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For interactive work from the repository root:

```bash
julia --project=.
```

then load the module with the `include` form above. The public namespaces are `Foundation`, `Algebra`, `Crystallography`, `Transformations`, `Models`, `Solvers`, `Observables`, `Thermodynamics`, `Greens`, and `Embedding`.

## Construct a simple spin model

A minimal finite model requires geometry, material metadata, a Hamiltonian specification, and a finite cluster. The following one-dimensional example keeps those stages explicit:

```julia
const M = Phundamental.Models

lattice = BravaisLattice(reshape([1.0], 1, 1))

basis_sites = [BasisSite(:X, :X, [0.0])
crystal = M.CrystalStructure(lattice, basis_sites)

species = Dict(:X => AtomicSpecies(:X; mass=1.0)
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

For scattering workflows, pass an observable request, probe, and resolution separately:

```julia
result = forward(model; solver=OptimizedLanczos(), observable=observable_request, probe=MagneticNeutronProbe(), resolution=GaussianResolution(0.08))
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

The v0.0.7 embedding layer implements normal-state single-site ED-DMFT. A typical half-filled Bethe problem uses an explicit lattice, Matsubara axis, finite-bath impurity solver, convergence policy, and mixing policy:

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

See [Validation](validation.md) for what these tests establish and what remains outside the v0.0.7 validation scope.
