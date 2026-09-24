using Phundamental 

include("generalized-api-validation.jl")
include("mixed-crystal-validation.jl")
include("transformation-representation-validation.jl")
include("greens/finite-temperature-green-validation.jl")
include("greens/greens-identities-validation.jl")
include("embedding/bath-discretization-validation.jl")
include("embedding/ed-impurity-sector-validation.jl")
include("embedding/ed-dmft-validation.jl")
include("embedding/dmft-reference-validation.jl")

ENV["PHUNDAMENTAL_BOSON_CONVERGENCE_STRICT"] = 1
include("bosonic-truncation-convergence-validation.jl")
