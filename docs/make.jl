using Documenter
using Documenter.Remotes: GitHub

include(joinpath(@__DIR__, "..", "src", "Phundamental.jl"))
using .Phundamental

const DOCUMENTED_MODULES = [
    Phundamental,
    Phundamental.PhundamentalCore,
    Phundamental.Algebra,
    Phundamental.Crystallography,
    Phundamental.Transformations,
    Phundamental.Models,
    Phundamental.Solvers,
    Phundamental.Observables,
    Phundamental.Thermodynamics,
    Phundamental.Greens,
    Phundamental.Embedding,
]

const DOCUMENTATION_PAGES = [
    "Home" => "index.md",
    "Getting Started" => "getting-started.md",
    "Architecture" => "architecture.md",
    "Modules" => [
        "Core / Foundation" => [
            "Concepts" => "modules/core.md",
            "States, Operators, and Representations" => "modules/core-reference-operators.md",
            "Model Structures and Interchange" => "modules/core-reference-models.md",
        ],
        "Algebra" => "modules/algebra.md",
        "Crystallography" => "modules/crystallography.md",
        "Transformations" => "modules/transformations.md",
        "Models" => "modules/models.md",
        "Solvers" => [
            "Overview" => "modules/solvers.md",
            "Protocol, Capabilities, and Bases" => "modules/solvers-reference-foundation.md",
            "Quadratic and Collective Modes" => "modules/solvers-reference-modes.md",
            "Krylov, Fermionic, and Classical Methods" => "modules/solvers-reference-methods.md",
        ],
        "Observables" => [
            "Concepts" => "modules/observables.md",
            "Descriptors and Correlations" => "modules/observables-reference-core.md",
            "Probe and Instrument APIs" => "modules/observables-reference-neutron.md",
        ],
        "Thermodynamics" => "modules/thermodynamics.md",
        "Greens" => "modules/greens.md",
        "Embedding" => "modules/embedding.md",
    ],
    "Workflows" => [
        "Forward Calculations" => "workflows/forward-calculations.md",
        "Universal Generation and Structural Search" => "workflows/universal-hamiltonian-generation.md",
        "Structural Forward Closure" => "workflows/structural-forward-closure.md",
        "Physical Interaction Generation" => "workflows/physical-interaction-generation.md",
        "Model Context and Localization" => "workflows/model-context-and-localization.md",
        "Physical Symmetry and Transformed Specifications" => "workflows/physical-symmetry-and-transport.md",
        "Experimental Data and Instrument Resolution" => "workflows/experimental-data-resolution.md",
        "Extended Forward Physics" => "workflows/extended-forward-physics.md",
        "Noncollinear Spin Waves" => "workflows/noncollinear-spin-waves.md",
        "Site SU(N) Spin Waves" => "workflows/sun-spin-waves.md",
        "Solver Capability Extensions" => "workflows/capability-extension-and-module-layout.md",
        "Representation Transformations" => "workflows/representation-transformations.md",
        "Scattering" => "workflows/scattering.md",
        "Impurity Models and Local Response" => "embedding/impurity.md",
        "Scalar and Matrix DMFT" => "embedding/dmft.md",
        "Diagrammatic Extensions" => "embedding/dga.md",
        "Fermionic Linked Clusters" => "embedding/nlce.md",
    ],
    "Validation" => "validation.md",
]

const GITHUB_REPOSITORY = get(ENV, "GITHUB_REPOSITORY", "alt-f4-dev/Phundamental.jl")
const DOCUMENTATION_REPOSITORY = GitHub(GITHUB_REPOSITORY)

const MAKEDOCS_OPTIONS = (
    modules=DOCUMENTED_MODULES,
    sitename="Phundamental.jl",
    checkdocs=:none,
    doctest=true,
    warnonly=false,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        sidebar_sitename=true,
        collapselevel=1,
        edit_link="main",
        # Preserve searchable API entries while warning if the full index exceeds an explicit 800 KiB budget.
        search_size_threshold_warn=800 * 1024,
    ),
    pages=DOCUMENTATION_PAGES,
)

makedocs(; MAKEDOCS_OPTIONS..., repo=DOCUMENTATION_REPOSITORY)

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    deploydocs(
        repo="github.com/$(GITHUB_REPOSITORY).git",
        devbranch=get(ENV, "PHUNDAMENTAL_DOCS_DEVBRANCH", "main"),
        push_preview=true,
        versions=["stable" => "v^", "v#.#", "dev" => "dev"],
    )
end
