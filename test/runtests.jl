using Phundamental

include("generalized-api-validation.jl")
include("high-level-forward-api-validation.jl")
include("dynamical-correlation-api-validation.jl")
include("composite-response-api-validation.jl")
include("response-capability-api-validation.jl")
include("forward-capability-api-validation.jl")
include("forward-method-integration-validation.jl")
include("forward-planning-api-validation.jl")
include("electronic-forward-api-validation.jl")
include("electron-phonon-forward-api-validation.jl")
include("phonon-forward-api-validation.jl")
include("bond-selection-api-validation.jl")
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

include("thermodynamics/nlce-api-validation.jl")
include("thermodynamics/nlce-topology-validation.jl")
include("greens/lattice-bubble-validation.jl")

include("embedding/diagrammatic-dmft-api-validation.jl")
include("greens/impurity-four-point-validation.jl")
include("embedding/ladder-dga-end-to-end-validation.jl")
