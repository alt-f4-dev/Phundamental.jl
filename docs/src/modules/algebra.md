```@meta
CurrentModule = Phundamental.Algebra
```

# Algebra

`Phundamental.Algebra` implements the noncommutative symbolic manipulations used by transformations, model construction, invariant generation, and numerical realization. It operates on the expression tree defined by [Core / Foundation](core.md) and keeps algebra-specific identities separate from the representation or solver layers.

The central principle is that an operator expression is not simplified as if its factors commute. Canonicalization must respect the declared spin, boson, fermion, Majorana, coordinate--momentum, or composite algebra.

## Commutators and anticommutators

The package exposes ordinary commutators and anticommutators,

$$[A,B]=AB-BA,$$

$$\{A,B\}=AB+BA.$$

`commutator` and `anticommutator` build and simplify these relations for symbolic expressions. `primitive_commutator` and `primitive_anticommutator` provide algebra-aware relations for primitive generators and are the low-level extension points used by rewrite logic.

Typical canonical relations include bosonic commutators, fermionic anticommutators, spin commutators, and the coordinate--momentum algebra. Composite algebras preserve factor identity so that operators acting on different tensor factors can be handled consistently.

## Rewrite rules

`rewrite_once` performs one algebraic rewrite pass. It is intentionally distinct from full canonicalization: callers that are implementing a transformation or debugging an algebraic relation can inspect the effect of a single rewrite step rather than requesting an entire normalization pipeline.

Rewrite rules encode local algebraic identities. They should not introduce a physical approximation. If a desired simplification changes the Hamiltonian rather than using an exact algebraic identity, it belongs in [Transformations](transformations.md) as an approximation or in a model-specific construction rather than in the Algebra layer.

## Normal ordering

`normal_order` rearranges creation and annihilation operators according to the declared algebra and introduces the required commutator or anticommutator terms. The operation is therefore algebra-dependent: bosonic and fermionic normal ordering have similar syntax but different signs and contraction terms.

Normal ordering is especially important after transformations such as Fourier, Bogoliubov, Lang--Firsov, or particle--hole maps, where direct substitution may generate expressions that are mathematically equivalent but not yet in a canonical operator order.

## Canonicalization

`canonicalize` is the primary normalization entry point. It recursively simplifies expression trees, applies algebra-specific ordering rules, combines scalar factors and like terms, removes algebraic zeros and identities, and produces a stable representation suitable for equality checks, transformation validation, or matrix realization.

Canonicalization is structural rather than numerical. It does not diagonalize the Hamiltonian, select a solver backend, truncate a bosonic space, or infer a physical approximation.

A useful workflow is

```julia
expr = c(2) * c(1)' + c(1)' * c(2)
canon = canonicalize(expr, FermionAlgebra())
```

The actual canonical result depends on the indices and the algebraic relation. The important point is that the algebra is supplied explicitly rather than inferred from generic multiplication.

## Operator-family classification

`operator_family` classifies a primitive or composite expression by its operator content. Higher-level code uses this information to decide which transformations, numerical backends, or symmetry actions are admissible.

Because `operator_family` is descriptive rather than transformative, it should not be used as a replacement for the full `Representation`. A mixed spin--boson Hamiltonian, for example, still needs a `CompositeAlgebra` and composite state-space structure even if individual terms can be classified by family.

## Algebra versus representation

The algebra answers how symbolic generators relate. The representation answers where those generators act and what physical constraints apply. These are deliberately different layers.

For example, the canonical bosonic relation does not itself enforce the exact Holstein--Primakoff bound $0\leq n_i\leq2S_i$. The bosonic algebra belongs here, while the occupancy restriction belongs to the target representation created by the transformation.

Likewise, fermionic anticommutation does not select a Jordan--Wigner ordering. Ordering information belongs to the representation and transformation certificate.

## Validation strategy

Algebraic validation is primarily exact symbolic validation. Transformation tests compare source and target Hamiltonians after canonicalization, and representation-numerics tests compare matrix spectra after the symbolic map has been realized numerically.

When a canonical expression appears unexpected, the useful diagnostic sequence is: inspect the primitive operator families, apply `rewrite_once`, inspect the intermediate tree, then apply `normal_order` and `canonicalize`. This separates a primitive relation error from an ordering or term-combination issue.

## Related modules

- [Core / Foundation](core.md) defines the expression tree and operator algebras manipulated here.
- [Transformations](transformations.md) performs representation changes and canonicalizes the transformed operators in the target algebra.
- [Crystallography](crystallography.md) transforms and projects symbolic candidate operators under symmetry.
- [Solvers](solvers.md) realizes the resulting canonical expressions as matrices or matrix-free operators.

## API reference

```@docs
commutator
anticommutator
primitive_commutator
primitive_anticommutator
rewrite_once
normal_order
canonicalize
```

`operator_family` is also part of the public Algebra API and provides lightweight operator-content classification for dispatch and diagnostics.
