# Benchmark environment and execution

The research benchmark suite is separate from `Pkg.test()`. It reproduces selected published models, exercises numerical convergence, and records performance measurements; it includes optional, memory-intensive workloads that should not become mandatory package regressions.

## One-time setup (from the repository root)

Use Julia 1.12 or 1.13 and create the benchmark environment against the current checkout:

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

The benchmark-only dependencies, including `GLMakie`, belong to `benchmarks/Project.toml` instead of the library's root environment. The generated `benchmarks/Manifest.toml` records the chosen dependency graph for this machine; keep it for repeatability and avoid committing machine-dependent local-path manifests to the package repository.

## Running the suite

```bash
./benchmarks/runtest.sh
```

The launcher resolves its paths from its own location, so it may be invoked from any working directory on Linux. Override `JULIA_EXECUTABLE`, `JULIA_NUM_THREADS`, or `OPENBLAS_NUM_THREADS` in the environment as needed. The default is Julia automatic threading with one OpenBLAS thread unless explicitly overridden. The script retains the existing physics parameters and benchmark order.

The comprehensive suite may allocate tens of GiB and produce large figures and CSVs. Set `PHUNDAMENTAL_TPQ_LARGE=1` only after confirming available system memory and the intended large-system benchmark configuration; the launcher currently defaults to `0` for paper-scale TPQ calculations.

## Interpreting results

A reported `PASS` only certifies the specific benchmark gates implemented by the script. Some figures are reconstructions rather than reproductions of original experimental arrays, and some published source conventions remain unresolved. Preserve outputs, Julia version, thread count, dependency manifest, hardware information, and any disabled optional studies when interpreting or publishing results.
