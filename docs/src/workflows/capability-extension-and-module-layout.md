# Extending Solver Capabilities and Module Ownership

Milestone E consolidates high-level computational capability definitions without altering the physical model, numerical solver, or observable-result contracts. New solver types extend a small set of public generic functions, and the package continues to select routes through the same `ForwardExecutionPolicy` and `plan_forward` machinery.

## Extension contract

A third-party solver subtype may define `solver_model_capability(solver, model)`, `solver_output_family(solver, model)`, `solver_request_capability(solver, model, request)`, `solver_context_capability(solver, model, request, context)`, and `solver_approximation(solver)`. Each support method returns `CapabilityDecision(supported; certified=true, issues=())`, with an optional tuple of stage-specific `CompatibilityIssue` values. Defining these methods by type dispatch does not require editing `ForwardCapabilities.jl` or its built-in rule file. Runtime execution also requires the relevant `Solvers.solve` method and, for nontrivial method contexts, a lowering method.

```julia
using Phundamental

struct ExampleSolver <: AbstractSolver end

Phundamental.solver_model_capability(::ExampleSolver, ::ManyBodyModel) = CapabilityDecision(true)
Phundamental.solver_output_family(::ExampleSolver, ::ManyBodyModel) = :eigensystem

# A concrete solver also defines Solvers.solve(::ExampleSolver, model).
# Supply ForwardRoute(ExampleSolver()) explicitly when calling plan_forward.
```

An extension should report `certified=false` when an essential property can only be tested during execution. The planner's default `require_certified=true` policy is retained; new solver candidates are not globally registered or automatically selected without an explicit `ForwardRoute`. An approximation-bearing solver must implement `solver_approximation` and requires caller authorization through `ForwardExecutionPolicy`.

The built-in numerical capability rules are grouped in `src/forward/capabilities/BuiltinSolverCapabilities.jl`. The public extension hook contracts are declared in `src/forward/capabilities/ForwardCapabilityContracts.jl`, while `src/forward/capabilities/ForwardCapabilities.jl` retains request-family dispatch and the compatibility composition layer. Existing private underscore-prefixed hooks remain functional for legacy extensions but are not the recommended external contract.

## Typed compatibility diagnostics

The existing `SolverCompatibility.reasons` strings remain available. `compatibility_issues(report)` additionally returns typed `CompatibilityIssue(stage, code, message)` values, separating `:model`, `:request_model`, `:solver_request`, and `:method_context` outcomes. A `:deferred` diagnostic indicates that compatibility is not yet statically certified, not that the route has been rejected. The planner still makes its decision from `compatible`, `certified`, and explicit approximation authorization. Validation failure in a model or request does not imply that its Hamiltonian violates physical symmetry or parameter admissibility.

## Retired wrappers

The package's authoritative `PhundamentalCore`, `Algebra`, `Crystallography`, `Transformations`, `Solvers`, `Observables`, `Models`, and `Thermodynamics` modules are assembled directly by `src/Phundamental.jl`. Seven obsolete directory-local module wrappers had no inclusion or import sites in the v0.1.0 source/test/documentation tree and retained stale dependencies such as `..Core`. Milestone E removes those unused wrapper files without changing root exports or the public module objects. Direct inclusion of those wrappers was not a supported package entry point and should be replaced with `using Phundamental` or a documented submodule import.

## Capability limits

This consolidation does not remove all guarded compatibility checks. Spin-wave parameter extraction, specialized sector resolution, and a few probe-specific request checks still call numerical validation routines that can raise errors. These are retained until equivalent explicit, nonthrowing predicates exist. The common Hubbard and NLCE metadata checks now use direct typed predicates rather than invoking their execution-time lowering functions.
