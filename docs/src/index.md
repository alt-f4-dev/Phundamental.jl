# Phundamental.jl

`Phundamental.jl` is a representation-aware framework for constructing, transforming, solving, and analyzing quantum many-body models in Julia. Version v0.0.7 freezes the first stable API that combines symbolic model spaces, representation transformations, generic and optimized many-body solvers, scattering observables, thermodynamic sampling, finite-temperature Green functions, and single-site ED-DMFT embedding under a common set of mathematical abstractions.

The central design rule is that the physical model, mathematical representation, numerical approximation, solver backend, observable definition, probe response, and instrumental resolution remain distinct objects. This separation makes transformations auditable, allows optimized backends to replace reference implementations without changing the represented physics, and keeps numerical approximations explicit in results and provenance.

$$\text{material/geometry}\rightarrow\text{Hamiltonian model}\rightarrow\text{representation}\rightarrow\text{optional transformation}\rightarrow\text{solver}\rightarrow\text{observable}\rightarrow\text{probe response}\rightarrow\text{instrument response}$$

The primary model object is `ManyBodyModel`, whose representation stores the ambient state space, exact physical sector, operator algebra, basis, constraints, gauge information, ordering, reference state, and any numerical truncation. A package-wide invariant is therefore

$$\boxed{\text{ambient state space}\neq\text{physical subspace}\neq\text{numerical truncation}}$$

Physical constraints define the theory being represented. Numerical truncations define how that theory is realized computationally and are never silently reinterpreted as physical constraints.

## Documentation map

The manual is organized first by architecture, then by the ten public submodules, and finally by cross-module workflows and validation. Each module page explains the mathematical role of the layer before presenting its API.

- [Getting Started](getting-started.md) covers loading the frozen v0.0.7 source tree, basic model construction, solver selection, and the high-level forward API.
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
- [Validation](validation.md) records the validation hierarchy, literature benchmarks, empirical checks, and the scope of the v0.0.7 claims.

## Version scope

This manual documents the frozen v0.0.7 API. Later development may extend the package, but statements in this manual are intended to describe the tagged v0.0.7 behavior rather than unreleased interfaces.

## High-level forward calculation

A complete forward calculation may be represented as a typed `ForwardProblem` or invoked through `forward`:

```julia
problem = ForwardProblem(model; transformation=JordanWigner(1:N), solver=OptimizedLanczos(), observable=observable_request, probe=MagneticNeutronProbe(), resolution=GaussianResolution(0.08))

result = forward(problem)
```

`ForwardResult` stores the source model, transformed working model, transformation result, solver output, intrinsic observable, probe-contracted scattering response, and final resolution-convolved result separately. This is the canonical package-level expression of the architecture.

!!! note
    `Phundamental.Foundation` is an alias of `Phundamental.PhundamentalCore`. The package does not define `Phundamental.Core` because `Core` is a built-in Julia module.

!!! note
    Version v0.0.7 validates normal-state, single-site, single-orbital ED-DMFT. The Green-function layer already supports Nambu indexing, but superconducting/Nambu ED-DMFT is outside the validated v0.0.7 embedding scope.
