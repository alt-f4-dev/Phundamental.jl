# Representation Transformations

Representation-aware calculations in Phundamental separate an exact mathematical map from both numerical encoding and physical approximation. This workflow shows how to construct, inspect, execute, and validate a transformation without losing the physical observables or constraints attached to the source model.

## Start from a complete model

A transformation acts on a `ManyBodyModel`, not only on its Hamiltonian expression. The source model contains the representation, physical subspace, constraints, parameterization, observable registry, and provenance that the transformation must transport or reinterpret.

Before transforming a numerically nontrivial model, inspect its representation:

```julia
rep = model.representation
println(rep.name)
println(rep.state_space)
println(rep.algebra)
println(rep.constraints)
println(rep.truncation)
```

This establishes what mathematical information must survive the map.

## Inspect the certificate first

Every standard transformation provides a certificate:

```julia
T = JordanWigner(1:N)
cert = certificate(T)
```

Inspect at least:

```julia
cert.exact
cert.invertible
cert.relation
cert.locality
cert.requires_constraints
cert.gauge_redundant
cert.approximate
```

For example, a Jordan--Wigner map is exact but nonlocal because fermionic generators contain parity strings. A Holstein--Primakoff map is exact only when its physical occupancy restriction is retained. Schwinger-boson and Abrikosov-fermion maps enlarge the ambient space and introduce exact constraints/gauge redundancy.

## Check applicability

```julia
applicable(T, model) || error("transformation is not applicable")
```

Applicability checks should be treated as part of the mathematical contract, not as optional input validation. A transformation written for spin-$1/2$ degrees of freedom should not silently reinterpret an arbitrary spin representation.

## Inspect the target representation

For advanced workflows, inspect the target before constructing the full transformed model:

```julia
target = target_representation(T, model)
```

The target records state space, algebra, basis, constraints, gauge information, ordering, and truncation provenance. An exact target can still be numerically unusable until a target-basis cutoff or projector is provided.

## Transform the model

```julia
mapped = transform(T, model)
working_model = mapped.model
```

The result also stores the transformation, certificate, and validation metadata. The Hamiltonian and registered observables are mapped together and canonicalized in the target algebra.

This is safer than manually applying `transform_operator` to the Hamiltonian because a Hamiltonian-only substitution would omit the target representation and any observable maps required for physical response calculations.

## Registered observables follow the model

Suppose a spin observable is registered before a Jordan--Wigner transformation:

```julia
model_with_obs = register_observable(model, :Sz_1, Sz(1))
mapped = transform(JordanWigner(1:N), model_with_obs)
transformed_op = mapped.model.observables[:Sz_1]
```

`transformed_op` is the representation-correct fermionic operator. Downstream observable code should resolve this registered operator instead of constructing a new `Sz(1)` after the model has been transformed.

## Numerical closure is a separate question

After a symbolic transformation succeeds, ask whether the target model is numerically realizable:

```julia
caps = numerical_capabilities(mapped.model)
println(caps.numerically_realizable)
println(caps.reasons)
```

`transformation_capabilities(T, model)` combines transformation metadata with the numerical closure of the actual target model.

An exact map into an infinite bosonic Fock space may require a numerical truncation even though the transformation itself is exact. This is not a contradiction: exactness describes the mathematical relation, while truncation describes the computational realization.

## Supply a target-basis truncation explicitly

When a bosonic target has no exact finite occupancy bound, use `with_truncation` on the target model:

```julia
truncated_model = with_truncation(mapped.model, NumericalTruncation(cutoff_metadata))
```

The precise cutoff representation depends on the target factor layout. The important rule is that a truncation transported from the source basis is provenance only; it should not be silently reused in an inequivalent target basis.

## Exact Holstein--Primakoff versus LSWT

The distinction is especially important for spin-wave calculations. The exact Holstein--Primakoff transformation retains the square-root operators and the exact occupancy bound:

$$S_i^+=\hbar\sqrt{2S_i-n_i}\,b_i,$$

$$S_i^-=\hbar b_i^\dagger\sqrt{2S_i-n_i}.$$

Linear spin-wave theory additionally expands/truncates the transformed Hamiltonian to quadratic order around an ordered state. That quadratic truncation is an approximation. Phundamental therefore does not label the exact Holstein--Primakoff map itself as LSWT.

## Invertible transformations

A certificate may state that a mathematical map is invertible while the software does not provide an executable inverse. Check `has_inverse(T)` before calling `inverse(T)`:

```julia
if has_inverse(T)
    source_again = transform(inverse(T), mapped.model)
end
```

The representation graph uses the same rule when deciding whether an edge can be traversed backward.

## Compose transformations

```julia
T = compose(T1, T2, T3)
mapped = transform(T, model)
```

The composite certificate reflects the properties of every component. If any component is nonlocal, constrained, gauge redundant, or approximate, that property is visible in the composite metadata.

For debugging, transform through the components separately and compare intermediate representations before replacing the sequence by a composite transformation.

## Use the representation graph

A `RepresentationGraph` records named representations and executable transformation edges:

```julia
graph = RepresentationGraph()
add_representation!(graph, model.representation)
add_representation!(graph, mapped.model.representation)
add_transformation!(graph, model.representation.name, mapped.model.representation.name, T)
path = transformation_path(graph, model.representation.name, mapped.model.representation.name)
```

`representation_class` returns the orbit connected by exact invertible edges. `transform_along` executes a discovered path and applies edge-specific parameter maps.

The graph is useful when several equivalent model representations coexist and the desired route should be discovered rather than hard-coded.

## Approximations belong on separate edges

If a transformed Hamiltonian is projected into a restricted target `HamiltonianSpace`, represent that step with `ModelSpaceProjection` or another `AbstractApproximation` rather than changing the transformation certificate.

For a decomposition

$$T(H)=H_\parallel+H_\perp,$$

an exact transformation can coexist with a nonzero discarded $H_\perp$. The exactness of $T$ and the approximation introduced by discarding $H_\perp$ are different claims.

`transformation_closure_defect` is useful for diagnosing whether a finite target model class closes under a transformation before a projection is accepted.

## Validation workflow

A strong representation-validation sequence is:

1. validate the source model and transformation applicability;
2. inspect the transformation certificate;
3. transform the model and registered observables;
4. inspect the target representation and numerical capabilities;
5. realize source and target calculations independently when both are finite;
6. compare spectra for exact equivalences;
7. compare transformed observables or physical response, not only Hamiltonian eigenvalues;
8. vary numerical truncations when the exact mathematical map enters an infinite representation.

The v0.0.7 test suite follows this pattern for particle--hole, Majorana, Jordan--Wigner, Holstein--Primakoff, Dyson--Maleev, Schwinger bosons, Abrikosov fermions, Fourier maps, Bogoliubov maps, coordinate--ladder, bosonic displacement, Lang--Firsov, and spin-polaron transformations.

## See also

- [Transformations](../modules/transformations.md) for certificates, standard maps, approximations, and graph APIs.
- [Core / Foundation](../modules/core.md) for representations and physical versus numerical restrictions.
- [Solvers](../modules/solvers.md) for numerical capability analysis and exact backend specialization.
- [Validation](../validation.md) for the transformation-numerics and bosonic cutoff validation hierarchy.
