module Algebra

using ..Core
import ..Core: canonical_model_expression

include("Commutators.jl")
include("RewriteRules.jl")
include("Canonicalization.jl")
include("StructuralCanonicalization.jl")

export commutator,
       anticommutator,
       primitive_commutator,
       primitive_anticommutator,
       rewrite_once,
       normal_order,
       canonicalize,
       operator_family

end # module Algebra
