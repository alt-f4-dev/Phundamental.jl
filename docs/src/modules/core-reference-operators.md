```@meta
CurrentModule = Phundamental.PhundamentalCore
```

# Foundation API — States, Operators, and Representations

## State spaces and exact restrictions

```@docs
AbstractStateSpace
AbstractHilbertSpace
AbstractFockSpace
SpinHilbertSpace
FermionFockSpace
BosonFockSpace
CoordinateHilbertSpace
CompositeStateSpace
BiorthogonalSpace
PhysicalSubspace
NumericalTruncation
spacedimension
FactorConstraint
FactorTruncation
```

## Operator algebras

```@docs
AbstractOperatorAlgebra
SpinAlgebra
BosonAlgebra
FermionAlgebra
MajoranaAlgebra
CoordinateMomentumAlgebra
CompositeAlgebra
algebra_kind
```

## Symbolic expressions

```@docs
IdentityOperator
NumberOperator
OperatorSum
OperatorProduct
ScaledOperator
FactorOperator
operator_degree
support_size
body_order
```

## Representations

```@docs
GaugeStructure
Representation
isconstrained
istruncated
ambientdimension
representationdimension
physicalspace
```
