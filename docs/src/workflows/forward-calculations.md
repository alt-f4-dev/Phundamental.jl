# Forward Calculations

The forward-calculation layer is the package-level orchestration mechanism for composing a physical model, optional representation map, numerical solver, intrinsic observable, probe response, and instrument resolution without collapsing those stages into one result.

The workflow is

$$\text{source model}\rightarrow\text{optional transformation}\rightarrow\text{solver}\rightarrow\text{intrinsic observable}\rightarrow\text{probe}\rightarrow\text{resolution}.$$

`ForwardProblem` stores the complete plan. `forward(problem)` executes it and returns a `ForwardResult` that retains the intermediate products.

## Why use `ForwardProblem`

For short exploratory calculations, calling `solve` and observable functions directly is often sufficient. `ForwardProblem` becomes useful when the calculation itself should be inspectable or serializable: the transformation, solver, arguments, optional method-specific context, observable request, probe, resolution, and metadata are all explicit fields rather than implicit call-site context.

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

More sophisticated requests should use a typed observable request when one is available. Generic scalar autocorrelations and cross-correlations use `DynamicalCorrelationRequest`:

```julia
autocorrelation_request = DynamicalCorrelationRequest(field, qgrid, axis; broadening=GaussianBroadening(0.05))
cross_correlation_request = DynamicalCorrelationRequest(field_a, field_b, qgrid, axis; broadening=GaussianBroadening(0.05))
```

The one-field constructor represents $S_{AA}$ without requiring the same field twice. The two-field constructor preserves the ordered $S_{AB}$ definition and is the generic high-level entry point for coupled scalar response. `ObservableRequest` remains available for custom observables that do not yet have a dedicated typed request.

For more than two physical channels, `CompositeDynamicalCorrelationRequest` evaluates the complete ordered matrix in one calculation request:

```julia
fields = (spin_z=spin_z_field, displacement=displacement_field, density=density_field)
request = CompositeDynamicalCorrelationRequest(fields, qgrid, axis; broadening=GaussianBroadening(0.05))
```

The request is sector-agnostic: the labels only identify the rows and columns of the response matrix, while the fields determine the actual registered operators. `SpinPhononCorrelationRequest` is the corresponding high-level physical wrapper for the standard spin and displacement observable families registered by spin-boson and compatible hybrid models.

## Request requirements and computational compatibility

Typed requests expose their physical requirements through `request_requirements`. The returned `RequestRequirements` records the requested response kind, frequency domain, statistics when known, momentum scope, response order, operator channels, temperature, and request metadata. These descriptors state what the calculation asks for; they do not choose a numerical method.

Solver compatibility is queried separately through `solver_compatibility`, `solver_capabilities`, and the `supports(solver, model, request)` overload. A `SolverCompatibility` report distinguishes model support from request support, records whether the compatibility statement is statically certified, identifies an explicitly approximate solver such as linear spin-wave theory, and retains diagnostic reasons.

```julia
requirements = request_requirements(request)
report = solver_compatibility(solver, model, request)

if report.compatible
    result = forward(ForwardProblem(model; solver=solver, observable=request))
end
```

`compatible_solvers(model, request, candidates)` filters a caller-supplied collection of solver instances without ranking or selecting among them. Compatibility remains a factual implementation query; Batch 6 execution planning consumes these reports under an explicit policy and never turns compatibility itself into approximation authorization.

Physical/model-space admissibility is a separate question. A Hamiltonian can be physically admissible under a generative specification while remaining unsupported by every current solver. Conversely, a numerically executable Hamiltonian is not thereby certified as a valid member of a physical hypothesis space.

The forward layer checks typed solver/request compatibility before execution and isolates method-specific numerical input construction behind the request-aware solve boundary. Direct model solvers and the harmonic-phonon momentum path continue to use this boundary, while Batch 5 extends it to explicit single-site DMFT, ladder-DΓA, and finite-reference NLCE routes.

## Explicit method contexts for embedding and NLCE

Some physically selected methods require computational information that is not part of the finite `ManyBodyModel` itself. A single-site DMFT calculation needs a noninteracting lattice embedding and Matsubara domain; ladder DΓA additionally needs a momentum lattice and bosonic transfer-frequency box; finite-reference NLCE needs an expansion order and an inner forward solver. These data are supplied through `method_context`, which configures an already selected method and never changes physical/model-space admissibility.

For single-site DMFT, the local interaction and chemical potential remain authoritative on the source `HubbardModel`, while `DMFTForwardContext` supplies the embedding domain:

```julia
axis = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1, 2])
context = DMFTForwardContext(AtomicLattice(); temperature=1 / beta, axis=axis)
request = LocalMatsubaraGreenFunctionRequest(metadata=Dict(:temperature => 1 / beta))
result = forward(ForwardProblem(model; solver=dmft_solver, method_context=context, observable=request))
```

`LocalMatsubaraSelfEnergyRequest` exposes the corresponding local Matsubara self-energy. Neither request performs analytic continuation.

For the end-to-end one-shot ladder-DΓA route, `LadderDGAForwardContext` records the converged-DMFT prerequisite machinery and the explicit Matsubara boxes. `MatsubaraSusceptibilityRequest(:charge)` and `MatsubaraSusceptibilityRequest(:magnetic)` select the physical output channel without changing the diagrammatic method. The resulting susceptibility remains in the Matsubara domain and therefore advertises `requires_analytic_continuation(result) == true`.

Finite-reference NLCE uses `NLCEForwardContext(max_order; inner_solver=...)`. Each generated cluster is evaluated recursively through the same `ObservableRequest` and ordinary `ForwardProblem` machinery, after which `NLCESolver` performs linked-cluster subtraction and reconstruction. Batch 5 restricts this high-level route to custom requests whose cluster values are additive scalars or arrays; richer typed response reconstruction can be added without changing the method-context boundary. The raw `NLCEResult` remains available as `ForwardResult.solution`, while `ForwardResult.observable_result` is the reconstructed `bulk_estimate`.

These three high-level routes deliberately reject an outer `transformation` because their computational problems are constructed from the physical source-model specification. Any cluster-level transformation for NLCE must instead be supplied explicitly as `NLCEForwardContext(inner_transformation=...)`. This prevents an exact or approximate representation operation from being silently ignored during method-specific lowering.

## Policy-governed execution planning

`plan_forward` separates route selection from execution. It evaluates deterministic `ForwardRoute` candidates against Batch 4 compatibility, applies a `ForwardExecutionPolicy`, and returns the selected route as an ordinary inspectable `ForwardProblem`. The default policy supplies only context-free exact routes, authorizes no physical approximation, and requires static compatibility certification.

```julia
plan = plan_forward(model, request)
report = planning_report(plan)
result = forward(plan)
```

The default exact candidate set prefers dedicated harmonic or exactly quadratic solvers when their model/request contracts are satisfied, then dense exact diagonalization. Selection is deterministic: exact routes outrank approximation-bearing routes, certified routes outrank otherwise equivalent uncertified routes, the policy solver preference breaks the next tie, and caller order is the final stable tie-breaker.

`OptimizedLanczos` is intentionally not synthesized by the default planner because the current request contract does not yet distinguish a complete eigensystem requirement from a low-energy partial-eigensystem requirement. A caller who knows that a Lanczos route is sufficient may supply `ForwardRoute(OptimizedLanczos(...))` explicitly; the planner can then rank it without pretending that low-energy truncation is universally equivalent to dense diagonalization. The planner likewise does not estimate wall time or memory, so `include_default_exact_routes=false` can be used when the caller wants to restrict planning to an explicitly supplied computational envelope.

Approximate methods are never synthesized by the planner because their physical or computational context is not uniquely determined by the source model. To make one available, supply it explicitly as a `ForwardRoute` and authorize every approximation used by that route. For example, single-site DMFT requires both an explicit `DMFTForwardContext` and policy authorization:

```julia
route = ForwardRoute(dmft_solver; method_context=dmft_context)
policy = ForwardExecutionPolicy(allowed_approximations=(:single_site_dmft,))
plan = plan_forward(model, request; policy=policy, routes=(route,))
result = forward(plan)
```

A ladder-DΓA route carries both the single-site-DMFT reference approximation and the ladder-DΓA approximation, so both symbols must be authorized. NLCE similarly includes any approximation used by its explicitly configured inner solver in addition to `:finite_order_nlce`. This prevents nested approximate methods from being hidden behind one outer solver label.

`ForwardRoute` may carry an outer representation transformation, but automatic planning accepts that transformation only when its `TransformationCertificate` declares it exact and non-approximate. Approximate representation operations remain explicit calculations outside the planner. DMFT, DΓA, and NLCE retain their Batch 5 prohibition on outer transformations because their numerical problems are reconstructed from the physical source-model specification.

When no candidate is both computationally compatible and policy-permitted, `plan_forward` throws `ForwardPlanningError`. The attached `ForwardPlanningReport` retains every candidate assessment, including compatibility diagnostics, approximation identities, certification state, method-context identity, and policy rejection reasons. A planned calculation also propagates the selected solver, approximation tuple, and certification state into mutable solution/observable metadata when available.

The planner performs no analytic continuation. A Matsubara result remains a Matsubara result, and a real-frequency request must be satisfied by a route that directly implements the required response or by a future explicitly configured continuation stage.

## Canonical stable demo

`demos/high-level-forward-api.jl` is the executable reference for the high-level forward architecture. The script uses only public APIs and demonstrates the full stable sequence `ManyBodyModel + ObservableRequest -> plan_forward -> planning_report -> forward -> ForwardResult`, including a magnetic-neutron probe and a separate Gaussian instrument-resolution stage.

```bash
julia --project=. demos/high-level-forward-api.jl
```

Because the demo is intentionally small, the default conservative planner selects a certified exact route without authorizing a physical approximation. The printed `ForwardResult` stage types make the separation between intrinsic response, probe response, and final resolution-convolved result explicit.

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

The capability protocol lets new solver/result types participate in generic workflows if they expose compatible structure. Dynamical outputs additionally expose semantic capabilities:

```julia
kind = response_kind(result.observable_result)
domain = frequency_domain(result.observable_result)
stats = response_statistics(result.observable_result)
scope = momentum_scope(result.observable_result)
channels = operator_channels(result.observable_result)
```

`response_convention` collects these inferred semantics together with declared spectral-axis, normalization, operator-order, temperature, and channel metadata. Matsubara Green functions and susceptibilities report `requires_analytic_continuation(result) == true`; a real-frequency observable must therefore be obtained through an explicit continuation route before applying the causal-response conversion helpers.

## Reproducibility checklist

A forward calculation intended for publication or regression testing should record the model parameters and geometry, representation transformation and certificate, numerical truncation if any, solver type and controls, observable convention and broadening, probe metadata, resolution model, and package version.

`ForwardProblem`, `ForwardResult`, transformation provenance, and solver/observable metadata collectively provide these pieces, but the user remains responsible for preserving the final configuration used for a reported figure or table.

## See also

- [Architecture](../architecture.md) for the package-wide semantic separation.
- [Transformations](../modules/transformations.md) for exact maps and approximations.
- [Solvers](../modules/solvers.md) for numerical capability analysis and computational compatibility.
- [Observables](../modules/observables.md) for intrinsic response, probes, and resolution.
