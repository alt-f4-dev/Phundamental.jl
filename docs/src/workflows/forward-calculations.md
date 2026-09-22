# Forward Calculations

The forward-calculation layer is the package-level orchestration mechanism for composing a physical model, optional representation map, numerical solver, intrinsic observable, probe response, and instrument resolution without collapsing those stages into one result.

The workflow is

$$\text{source model}\rightarrow\text{optional transformation}\rightarrow\text{solver}\rightarrow\text{intrinsic observable}\rightarrow\text{probe}\rightarrow\text{resolution}.$$

`ForwardProblem` stores the complete plan. `forward(problem)` executes it and returns a `ForwardResult` that retains the intermediate products.

## Why use `ForwardProblem`

For short exploratory calculations, calling `solve` and observable functions directly is often sufficient. `ForwardProblem` becomes useful when the calculation itself should be inspectable or serializable: the transformation, solver, arguments, observable request, probe, resolution, and metadata are all explicit fields rather than implicit call-site context.

A forward plan also guarantees that the observable is evaluated on the working model after any requested representation transformation. Registered physical observables are transformed with the model, preventing a common error in which a source-representation operator is applied to a target-representation eigensystem.

## Build a model

The first stage is an ordinary `ManyBodyModel`. For example, a finite XXZ chain can be constructed from model geometry and material data:

```julia
include("src/Phundamental.jl")
using .Phundamental

const M = Phundamental.Models

lattice = BravaisLattice(reshape([1.0], 1, 1))
crystal = M.CrystalStructure(lattice, [BasisSite(:X, :X, [0.0])])

species = Dict(:X => AtomicSpecies(:X; mass=1.0)
props = [SiteProperties(spin=1//2)]


material = Material("spin chain", crystal; species=species, basis_properties=props)
cluster = Supercell(crystal, [8]; periodic=[false])

hamiltonian = XXZModel(1.0, 0.6; neighbor_cutoff=1.01)
model = build_model(material, cluster, hamiltonian)
```

At this point no solver has been selected and no representation transformation has been implied. The model remains a symbolic spin model with its geometry and provenance.

## Choose an optional representation transformation

A transformation may be supplied directly to `ForwardProblem`:

```julia
transformation = JordanWigner(1:8)
```

The forward layer calls the ordinary transformation machinery and stores the complete `TransformationResult`. It does not bypass applicability checks or transformation certificates.

If no transformation is needed, use `transformation=nothing` and the source model becomes the working model.

## Choose a solver

The solver is an ordinary `AbstractSolver` instance:

```julia
solver = OptimizedLanczos(nev=4)
```

The forward layer does not alter solver semantics. `ExactDiagonalization`, `OptimizedLanczos`, `AutoSolver`, mode solvers, or another compatible solver can be used according to the model representation and numerical requirements.

Solver positional and keyword arguments may be stored separately through `solver_args` and `solver_kwargs` when a concrete solver method requires additional information.

## Define an intrinsic observable request

`ObservableRequest` packages a symbolic name and an evaluator called as `evaluator(model, solution)`. A minimal ground-state-energy request is

```julia
energy_request = ObservableRequest(:ground_energy, (working_model, solution) -> groundenergy(solution))
```

The evaluator receives the post-transformation working model rather than the original source model. This is important for requests that resolve registered observables or construct momentum-space operators from the current representation.

More sophisticated requests can capture grids, temperatures, line broadenings, normalization conventions, or other observable-specific configuration:

```julia
spin_spectrum = (working_model, solution) -> dynamic_structure_factor(working_model, solution, field, qgrid, axis; broadening=GaussianBroadening(0.05))

spectrum_request = ObservableRequest(:spin_spectrum, spin_spectrum)
```

The exact argument list for a specialized observable should follow the corresponding [Observables](../modules/observables.md) API; `ObservableRequest` itself only standardizes how the request enters the forward pipeline.

## Add a probe only when appropriate

A probe operates on an intrinsic observable result. For magnetic neutron scattering, for example,

```julia
probe = MagneticNeutronProbe()
```

The forward constructor rejects a probe when no observable request was supplied. This enforces the architecture rather than allowing a probe to act directly on an eigensystem with no defined intrinsic response.

## Add instrument resolution separately

An instrument response acts after the intrinsic or probe-contracted spectrum:

```julia
resolution = GaussianResolution(0.08)
```

The resolution constructor is likewise rejected if no observable request exists. A numerical line broadening used to render Lehmann delta functions belongs inside the observable request; an experimental resolution belongs here.

## Construct and run the plan

```julia
problem = ForwardProblem(model; transformation=JordanWigner(1:8), solver=OptimizedLanczos(nev=4), observable=energy_request, metadata=Dict(:purpose => :representation_check))

result = forward(problem)
```

The result keeps the stages distinct:

```text
source_model
model
transformation_result
solution
observable_result
scattering_result
result
```

`result.model` is the working model after transformation. `result.solution` is the solver output. `result.observable_result` is the intrinsic response. `result.scattering_result` is populated only when a probe was requested. `result.result` is the final stage, which may be the intrinsic observable, probe response, or resolution-convolved response depending on the plan.

## Direct convenience form

The convenience form accepts the same conceptual stages without constructing `ForwardProblem` explicitly:

```julia
result = forward(model; transformation=JordanWigner(1:8), solver=OptimizedLanczos(nev=4), observable=energy_request)
```

Use the typed problem form when the calculation plan itself is part of the scientific record; use the convenience form when the plan is short and already clear from the surrounding code.

## Result capabilities

Consumers that operate on heterogeneous result types should prefer the root result-capability protocol over hard-coded type switches:

```julia
caps = result_capabilities(result.solution)
if supports(result.solution, :energies)
    E = result_energies(result.solution)
end
```

The capability protocol lets new solver/result types participate in generic workflows if they expose compatible structure.

## Reproducibility checklist

A forward calculation intended for publication or regression testing should record the model parameters and geometry, representation transformation and certificate, numerical truncation if any, solver type and controls, observable convention and broadening, probe metadata, resolution model, and package version.

`ForwardProblem`, `ForwardResult`, transformation provenance, and solver/observable metadata collectively provide these pieces, but the user remains responsible for preserving the final configuration used for a reported figure or table.

## See also

- [Architecture](../architecture.md) for the package-wide semantic separation.
- [Transformations](../modules/transformations.md) for exact maps and approximations.
- [Solvers](../modules/solvers.md) for numerical admissibility and backend selection.
- [Observables](../modules/observables.md) for intrinsic response, probes, and resolution.
