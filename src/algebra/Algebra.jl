module Algebra

using ..Core

include("Commutators.jl")
include("RewriteRules.jl")
include("Canonicalization.jl")

export commutator,
       anticommutator,
       primitive_commutator,
       primitive_anticommutator,
       rewrite_once,
       normal_order,
       canonicalize,
       operator_family

end # module Algebra
