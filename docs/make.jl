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
        "Core / Foundation" => "modules/core.md",
        "Algebra" => "modules/algebra.md",
        "Crystallography" => "modules/crystallography.md",
        "Transformations" => "modules/transformations.md",
        "Models" => "modules/models.md",
        "Solvers" => "modules/solvers.md",
        "Observables" => "modules/observables.md",
        "Thermodynamics" => "modules/thermodynamics.md",
        "Greens" => "modules/greens.md",
        "Embedding" => "modules/embedding.md",
    ],
    "Workflows" => [
        "Forward Calculations" => "workflows/forward-calculations.md",
        "Representation Transformations" => "workflows/representation-transformations.md",
        "Scattering" => "workflows/scattering.md",
        "ED-DMFT" => "workflows/ed-dmft.md",
    ],
    "Validation" => "validation.md",
]

const GITHUB_REPOSITORY = get(ENV, "GITHUB_REPOSITORY", "")
const DOCUMENTATION_REPOSITORY = isempty(GITHUB_REPOSITORY) ? nothing : GitHub(GITHUB_REPOSITORY)

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
    ),
    pages=DOCUMENTATION_PAGES,
)

if isnothing(DOCUMENTATION_REPOSITORY)
    makedocs(; MAKEDOCS_OPTIONS..., remotes=nothing)
else
    makedocs(; MAKEDOCS_OPTIONS..., repo=DOCUMENTATION_REPOSITORY)
end

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    isempty(GITHUB_REPOSITORY) && error("GITHUB_REPOSITORY is required for documentation deployment")
    deploydocs(
        repo="github.com/$(GITHUB_REPOSITORY).git",
        devbranch=get(ENV, "PHUNDAMENTAL_DOCS_DEVBRANCH", "main"),
        push_preview=true,
        versions=["stable" => "v^", "v#.#", "dev" => "dev"],
    )
end
