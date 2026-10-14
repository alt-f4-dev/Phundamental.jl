# Bridge the Foundation structural protocol to the exact algebraic canonicalizer.

canonical_model_expression(algebra::AbstractOperatorAlgebra, expression::AbstractOperatorExpr) = canonicalize(algebra, expression)
