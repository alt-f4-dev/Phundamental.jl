# Phundamental.jl

`Phundamental.jl` is a representation-aware framework for constructing, transforming, solving, and analyzing quantum many-body models in Julia. Version v0.0.8 stabilizes the model-to-measurement forward architecture: symbolic models and exact transformations feed generic or specialized numerical methods, typed observable requests describe the desired physical response, computational compatibility is inspected explicitly, DMFT/DΓA/NLCE can participate in the common forward pathway, and `plan_forward` selects only routes permitted by an explicit execution policy.

The central design rule is that the physical model, mathematical representation, numerical approximation, solver backend, observable definition, probe response, and instrumental resolution remain distinct objects. This separation makes transformations auditable, allows optimized backends to replace reference implementations without changing the represented physics, and keeps numerical approximations explicit in results and provenance.

$$\text{material/geometry}\rightarrow\text{Hamiltonian model}\rightarrow\text{representation}\rightarrow\text{optional transformation}\rightarrow\text{solver}\rightarrow\text{observable}\rightarrow\text{probe response}\rightarrow\text{instrument response}$$

The primary model object is `ManyBodyModel`, whose representation stores the ambient state space, exact physical sector, operator algebra, basis, constraints, gauge information, ordering, reference state, and any numerical truncation. A package-wide invariant is therefore

$$\boxed{\text{ambient state space}\neq\text{physical subspace}\neq\text{numerical truncation}}$$

Physical constraints define the theory being represented. Numerical truncations define how that theory is realized computationally and are never silently reinterpreted as physical constraints.

## Documentation map

The manual is organized first by architecture, then by the ten public submodules, and finally by cross-module workflows and validation. Each module page explains the mathematical role of the layer before presenting its API.

- [Getting Started](getting-started.md) covers loading the frozen v0.0.8 source tree, basic model construction, solver selection, and the high-level forward API.
- [Architecture](architecture.md) defines the package-wide data flow, result-capability protocol, provenance model, and the distinction between transformations, approximations, and numerical backends.
- [Core / Foundation](modules/core.md) defines state spaces, operator algebras, representations, model spaces, parameter spaces, reciprocal-space primitives, and `ManyBodyModel`.
- [Algebra](modules/algebra.md) implements noncommutative commutators, rewrite rules, normal ordering, and canonicalization.
- [Crystallography](modules/crystallography.md) discovers space-group symmetry and generates invariant Hamiltonian bases.
- [Transformations](modules/transformations.md) represents exact maps, constrained embeddings, gauge-redundant maps, approximations, and transformation graphs.
- [Models](modules/models.md) defines geometry, materials, interactions, force constants, and Hamiltonian constructors.
- [Solvers](modules/solvers.md) covers computational bases, matrix realization, ED/Lanczos/time evolution, optimized packed backends, mode solvers, classical sampling, and cluster-perturbation Green functions.
- [Observables](modules/observables.md) defines correlations, spectral functions, magnetic and nuclear neutron response, multiphonon scattering, and resolution convolution.
- [Thermodynamics](modules/thermodynamics.md) provides exact Gibbs ensembles, stochastic thermal methods, symmetry-sector decomposition, and linked-cluster reconstruction.
- [Greens](modules/greens.md) provides finite-temperature Matsubara propagators, Nambu indexing, Lehmann evaluation, Dyson relations, asymptotic moments, and validation diagnostics.
- [Embedding](modules/embedding.md) provides finite ED baths, impurity solvers, lattice/Weiss self-consistency, adaptive mixing, and the validated single-site ED-DMFT driver.
- [Workflows](workflows/forward-calculations.md) show how the layers compose without collapsing their mathematical distinctions.
- [Validation](validation.md) records the validation hierarchy, literature benchmarks, empirical checks, and the scope of the v0.0.8 claims.

## Version scope

This manual documents the frozen v0.0.8 API. Later development may extend the package, but statements in this manual are intended to describe the tagged v0.0.8 behavior rather than unreleased interfaces.

## High-level forward calculation

A complete typed forward calculation can be planned before execution so that the selected computational route is inspectable:

```julia
plan = plan_forward(model, observable_request; probe=MagneticNeutronProbe(), resolution=GaussianResolution(0.08))
report = planning_report(plan)
result = forward(plan)
```

The conservative default execution policy considers only certified context-free exact routes and does not silently authorize a physical approximation. Explicit `ForwardProblem` construction remains available when the caller wants to choose the solver, transformation, or method context directly.

`ForwardResult` stores the source model, transformed working model, transformation result, solver output, intrinsic observable, probe-contracted scattering response, and final resolution-convolved result separately. This is the canonical package-level expression of the architecture.

The executable reference for this stable interface is `demos/high-level-forward-api.jl`. It demonstrates the public model/request/planner/forward sequence and prints the selected exact route together with the retained forward stages.

!!! note
    `Phundamental.Foundation` is an alias of `Phundamental.PhundamentalCore`. The package does not define `Phundamental.Core` because `Core` is a built-in Julia module.

!!! note
    Version v0.0.8 validates normal-state, single-site, single-orbital ED-DMFT. The Green-function layer already supports Nambu indexing, but superconducting/Nambu ED-DMFT is outside the validated v0.0.8 embedding scope.
