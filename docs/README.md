# Phundamental.jl documentation maintenance

The documentation is built with Documenter.jl from the frozen v0.0.7 source tree. The documentation environment contains Documenter and the package's external `Spglib` dependency; `docs/make.jl` loads `src/Phundamental.jl` directly so the build does not require the repository root to be a registered Julia package project.

## Local build

From the repository root, instantiate the documentation environment and build the site:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve `docs/build/` with a local HTTP server rather than opening generated HTML files directly. Documenter is configured with `prettyurls=true` both locally and in CI so local and deployed URL behavior remains identical.

## GitHub Pages deployment

The supplied `.github/workflows/documentation.yml` follows the Documenter.jl GitHub Actions deployment model: documentation is built on pushes, tags, pull requests, and manual dispatches; release and development documentation is pushed by Documenter to the `gh-pages` branch; pull requests receive preview builds. In the repository's GitHub settings, configure Pages to deploy from the `gh-pages` branch at `/ (root)`.

The supplied workflow assumes that the development branch is `main`. If the repository uses another development branch, change the `on.push.branches` entry and the `PHUNDAMENTAL_DOCS_DEVBRANCH` value in `.github/workflows/documentation.yml` to the same branch name before enabling deployment.

## Documentation policy

Each public submodule has one conceptual reference page. Manual prose explains mathematical meaning, conventions, invariants, numerical assumptions, and workflows; curated `@docs` blocks expose the most important inline API docstrings. Expensive scientific calculations remain in the validation and benchmark scripts rather than executing during every documentation build.

The current build uses `checkdocs=:none` because v0.0.7 contains exported symbols whose behavior is documented in the manual but which do not all have normalized inline docstrings. Once source-level docstring coverage has been audited and normalized, change this to `checkdocs=:exports` and keep the documentation build strict.

Generated `docs/build/` content must not be committed to the development branch. The generated site belongs on `gh-pages` and is managed by Documenter deployment.
