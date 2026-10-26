# Experimental prediction exchange and instrument resolution

Milestone C extends the established intrinsic observable → probe contraction → instrument response sequence without replacing `ForwardResult`, intrinsic spectrum types, or the optimized constant-width `GaussianResolution`. All instrument response functions act after intrinsic broadening; their parameters must come from an experiment-specific calibration or a clearly stated approximate resolution model.

## Versioned numerical interchange

`ExperimentalGrid` declares a numeric Q-coordinate matrix with rows ordered by measurement sample, a strictly increasing real spectral axis, the spectral kind (`:energy` or `:omega`), and explicit coordinate units and frames (`:cartesian`, `:reciprocal`, or `:dimensionless`). Local site spectra use `:site`. A reciprocal-lattice frame is not automatically interchangeable with Cartesian inverse-length coordinates. Explicitly pass `momentum_coordinates` when the original result uses wavevector objects; no unverified reciprocal-space conversion is performed.

`portable_prediction` requires an explicit `spectral_kind` when the original spectrum does not declare `metadata[:spectral_axis]`. It accepts a supported real-frequency scalar, tensor, composite, local, or scattering spectrum, and can select the `:intrinsic`, `:probe`, or `:instrument` stage of `ForwardResult`. Specify `coordinate_frame`, `coordinate_unit`, `spectral_unit`, `intensity_unit`, and `normalization`; the API does not infer calibration constants or count-rate normalization. `experimental_dataset` stores measured intensities, a valid-bin mask, strictly positive standard deviations or a positive-definite covariance restricted to valid bins, and optional portable provenance. Optional prediction uncertainties and covariance are exported only when provided by the result or explicitly supplied by the caller; they are not estimated automatically. For complex tensor response, probe contraction is required before a real-valued measurement comparison.

```julia
prediction = portable_prediction(result; coordinate_frame=:cartesian, coordinate_unit="1/Å", spectral_unit="meV", intensity_unit="counts/meV", normalization="per_cell")
record = export_result(prediction)
restored = import_result(record)
```

`export_result` creates a versioned `Dict{String,Any}` containing only neutral scalars, strings, dictionaries, lists, booleans, and `nothing`; no serialized Julia objects or package types are required by Python. An array is encoded as `{shape, order="column_major", real, imag}` with flattened Julia column-major ordering. A Python consumer reconstructs it using `numpy.reshape(data, shape, order="F")`. Complex components are carried separately and missing/masked values are `null` (`nothing` in Julia). This is an interchange structure, not a JSON encoder, HDF5 writer, or remote I/O mechanism. `import_result` validates the schema version and array ordering before reconstructing typed Julia objects. Source metadata not representable as neutral values are not silently serialized; supply explicit `provenance` for dataset/model identifiers, solver route and approximation settings, temperature, calibration, and reduction history. Prediction records additionally retain response conventions from the existing result-capability API. For a `ForwardResult`, the neutral `describe(result)` stage and structural-identity record is included in the explicit provenance map by default; pass `include_forward_description=false` only when deliberate provenance omission is desired.

`compare_measurement(prediction, dataset)` computes chi-square after checking exact grid, coordinate-frame, unit, normalization, shape, and validity-mask agreement. It uses the intersection of valid predicted and observed bins, with either measured diagonal standard deviations or measured full covariance restricted to those bins. Prediction uncertainty is preserved separately but is not automatically added to the experimental likelihood, since independence and covariance assumptions require explicit justification. It performs no energy regridding, intensity normalization, background subtraction, model fitting, or covariance estimation. This explicitly distinguishes experimental uncertainties from instrument broadening.

## Energy-dependent and tabulated resolution

`EnergyDependentGaussianResolution(sigma)` evaluates a user-supplied positive width at each **true** spectral energy; `TabulatedGaussianResolution(energies, widths)` linearly interpolates a positive tabulated true-energy width with no extrapolation. Both integrate $I_\mathrm{obs}(x_i)=\sum_j w_j R(x_i\mid x_j)I_\mathrm{true}(x_j)$ using the nonuniform-grid trapezoidal weights $w_j$. `TabulatedEnergyResolution(energy, density)` accepts a nonnegative response matrix indexed `(observed, true)` on the **same** spectral grid. The kernel is interpreted as a density and is multiplied by true-grid quadrature weights; do not pre-multiply its entries by bin widths.

The optimized `GaussianResolution(sigma)` retains its existing uniform-grid stencil and finite-window convention. General energy kernels do not artificially restore weight lost outside the available spectral window, and Gaussian widths are not renormalized individually near boundaries. A discrete approximation may require convergence testing when the resolution is smaller than the grid spacing or a truncated kernel extends beyond a finite energy window.

## Coupled momentum–energy response

`GaussianQEResolution(covariance, q_coordinates, q_weights; coordinate_frame, momentum_unit, energy_unit)` accepts a symmetric positive-definite covariance in the ordered coordinates `[Q..., energy]`, explicit momentum samples as a matrix `(nQ, dimension)`, and positive momentum integration weights. Its Gaussian kernel includes off-diagonal covariance and is evaluated without assembling a dense Q–energy response matrix. Energy integration uses the same nonuniform trapezoidal rule. It supports scalar `SpectrumResult` and `ScatteringResult` grids; general tensor-covariance convolution must be contracted before this stage. All supplied Q coordinates, Q integration weights, and covariance entries must already be expressed in a mutually consistent coordinate frame and units. The implementation does not guess the Jacobian from reciprocal-lattice units to inverse ångströms.

`SparseQEResolution(nQ, nE, observed, source, weights)` accepts a sparse collection of **already cell-integrated** transport coefficients on a common Q–energy grid, indexed in Julia column-major order (linear index `iQ + (iE-1)*nQ`). This supports tabulated or empirically simulated non-Gaussian instrument responses without storing a dense matrix. It does not impose a probability-column normalization because detector efficiency and finite-window losses may be physical; users must certify the meaning of their supplied weights.

`CoveredResolution(kernel, InstrumentCoverage(valid))` records the independent detector/kinematic coverage mask in the result metadata. Invalid bins retain computed numeric intensities; their values are excluded from portable records and statistical comparisons. A coverage mask is not an instrument simulation, does not provide a kinematic scattering law, and must be derived separately from the actual detector geometry and experimental settings.

## Assumptions and limitations

All new numerical convolutions act on a shared source/observed grid; interpolation onto independently sampled experimental coordinates is not performed. They do not infer a spectrometer-specific resolution model from sample structure, do not construct instrument trajectories, do not propagate predicted Monte Carlo covariance, and do not assume that reciprocal-lattice coordinates are Cartesian momentum coordinates. Cross-correlated Q–energy kernels are integrated with explicit sampling weights; reliable values require convergence and calibration checks. `ForwardResult` remains unchanged and continues to store intrinsic, probe, and instrument stages independently.

## F5: Raw-count Poisson likelihoods

`experimental_dataset(grid, counts; likelihood=:poisson, intensity_unit="counts", normalization="raw_counts")` represents **nonnegative integer raw detector counts**, without independently supplied Gaussian standard deviations or covariance. Its matching `PortablePrediction` must contain **expected detector counts per bin**, including any physically modeled background and exposure; no model-to-count conversion is inferred. Background-subtracted intensities, detector-normalized rates, and Gaussian-propagated reductions must continue using the Gaussian measurement mode unless an externally justified count-generating model is supplied.

`compare_measurement(prediction, measured)` keeps its original Gaussian result for `likelihood=:gaussian`. For `likelihood=:poisson`, the comparison returns the Cash/Poisson deviance $D=2\sum_i[\mu_i-n_i+n_i\log(n_i/\mu_i)]$, with the $n_i=0$ logarithmic term defined as zero. The comparison's `chi_squared` field is intentionally `NaN` in Poisson mode, while `deviance`, `likelihood`, and signed deviance residuals are populated. A predicted zero count in a bin with positive observed counts produces infinite deviance. `log_likelihood` omits the data-only $\log(n_i!)$ terms; it is appropriate for ranking predictions against an unchanged dataset, not for an absolute probability calculation. The schema-v1 measurement record adds the optional `likelihood` string; older records without that field are interpreted as Gaussian and must still satisfy Gaussian uncertainty requirements.

```julia
measured = experimental_dataset(prediction.grid, raw_counts; likelihood=:poisson, intensity_unit="counts", normalization="raw_counts")
report = compare_measurement(prediction, measured)
@show report.deviance report.nvalid
```

## F5: Source-resolved momentum–energy covariance and spatial search

`SpatialGaussianQEResolution(covariances, q_coordinates, energy, q_weights; coordinate_frame, momentum_unit, energy_unit)` accepts one positive-definite covariance per **true** momentum–energy source bin, indexed `covariances[:, :, iq, ie]`. Each covariance has `[Q..., energy]` ordering; the four-dimensional array shape is `(dimension+1, dimension+1, nQ, nE)`. Alternatively, pass a callable `covariance_function(qrow, true_energy)` to evaluate a source covariance on demand; set `storage=:dense` to materialize and validate the full tensor eagerly. Covariances, coordinates, and quadrature weights must use a consistent physical convention, and the recorded source-energy axis must match the source spectrum exactly.

The convolution computes $I_i^{\mathrm{obs}}=\sum_j w_jR(x_i\mid x_j)I_j^{\mathrm{true}}$, with the source covariance determining the Gaussian normalization and correlated Mahalanobis distance. The implementation uses a balanced multidimensional point index to restrict candidate output samples, then applies independent component bounds and the exact covariance quadratic form. It avoids a full exhaustive $n_Q^2$ momentum comparison when Gaussian support is spatially localized, but it does not promise subquadratic work when every covariance spans the entire momentum grid. Neither source kernels nor boundary-truncated outputs are artificially renormalized. Sparse and correlated kernels can both be wrapped in `CoveredResolution` without changing the detector-coverage convention.

```julia
resolution = SpatialGaussianQEResolution(covariances, q_coordinates, energies, q_weights;
                                         coordinate_frame=:cartesian, momentum_unit="1/Å", energy_unit="meV", nsigma=5.0)
convolved = convolve_resolution(resolution, intrinsic_spectrum)
```

The optimized constant-covariance `GaussianQEResolution` follows the same spatial-search method and retains its previous analytical Gaussian normalization, integration weights, and cross-covariance conventions. Performance depends on momentum-point ordering and local support; validate a particular spectrometer sampling density against an independent quadrature reference before interpreting fitted parameters.

## Scaling multidimensional Q–E kernels

`SpatialGaussianQEResolution` accepts per-source-bin covariance arrays with `storage=:dense` (the backwards-compatible default) or `storage=:lazy` (retaining the covariance while omitting the precomputed precision and normalization arrays). A callable covariance provider of the form `(q_vector, true_energy) -> covariance_matrix` defaults to `storage=:lazy` and allocates no per-bin covariance tensors; `storage=:dense` explicitly requests eager materialization. Lazy factorization occurs only for source bins with nonzero intensity. This trades repeated small matrix factorizations for lower resident memory. The convolution now caches its balanced multidimensional point index on the resolution instance, uses reusable Cholesky/precision workspaces for lazy covariance, and prunes the observed energy interval through the exact quadratic roots of the correlated Mahalanobis cutoff. The original full covariance, source-bin weights and Gaussian normalization are retained; the result is still an exact truncated quadrature, not a separable Gaussian approximation. The resolution's momentum coordinates must remain unchanged after construction because the cached tree is tied to those coordinates.

```julia
resolution = SpatialGaussianQEResolution((q, energy) -> covariance_model(q, energy), coordinates, energies, q_weights; coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", storage=:lazy)
```

Call `convolve_resolution(resolution, spectrum; threaded=true)` to opt into a source-parallel execution path. The default is serial, preserving source-major summation order. The threaded path prepares and copies source covariance and precision matrices in bounded 16,384-bin batches before executing parallel scattering, so factorization storage cannot be modified by a concurrent worker. A separate lock merges worker-local contribution buffers; no full output copy is allocated per thread. This approach also supports providers returning borrowed mutable covariance buffers. Parallel accumulation may change last-bit rounding and may be slower than serial execution when source support is small; measure both modes under your actual workload.

From `benchmarks/`, run `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. experimental-resolution-scaling.jl` for the serial baseline and prefix `PHUNDAMENTAL_QE_THREADED=true` for the optional threaded comparison. Use `PHUNDAMENTAL_QE_PROFILE=tof` for the larger safety-budgeted workloads. The script reports resolution construction time (including spatial-index construction), convolution time, cumulative allocations, resident-set snapshots, and numerical agreement on small grids.

The resolution provider must return a finite symmetric positive-definite matrix at every evaluated source bin. Geometry, quadrature weights, coordinate units, covariance parameterization and spectral-axis conventions remain caller responsibilities. Performance and agreement with dense reference convolution are tested in `test/phase-i-closure-validation.jl` and characterized by `benchmarks/experimental-resolution-scaling.jl`; do not infer validated full-instrument throughput from the synthetic scaling script alone.
