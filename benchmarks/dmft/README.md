# ED-DMFT validation and benchmark suite

This directory contains the scientific validation campaign for the first Phundamental finite-temperature ED-DMFT milestone. The scripts are intentionally separated from unit tests because finite-bath convergence, low-temperature behavior, and scaling can be computationally expensive and should be inspected quantitatively rather than reduced to one pass/fail threshold.

The primary published reference is M. Caffarel and W. Krauth, "Exact diagonalization approach to correlated fermions in infinite dimensions: Mott transition and superconductivity," *Physical Review Letters* **72**, 1545-1548 (1994), DOI: 10.1103/PhysRevLett.72.1545. The extended derivation is arXiv:cond-mat/9306057.

The general DMFT formal reference is A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, "Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions," *Reviews of Modern Physics* **68**, 13-125 (1996), DOI: 10.1103/RevModPhys.68.13.

The finite-temperature bath-size validation reference is A. Liebsch and H. Ishida, "Temperature and bath size in exact diagonalization dynamical mean field theory," *Journal of Physics: Condensed Matter* **24**, 053201 (2012), DOI: 10.1088/0953-8984/24/5/053201.

`bath-discretization-benchmark.jl` measures the quality of the finite-bath representation as a function of inverse temperature, bath size, and Matsubara weighting exponent. `ed-dmft-physical-regimes-benchmark.jl` follows the half-filled Bethe model across several interaction strengths, compares the lowest Matsubara self-energies across bath size, and exports the corresponding frequency-resolved curves. `caffarel-krauth-1994-benchmark.jl` matches the Bethe normalization and finite-temperature parameter choices used by Caffarel and Krauth, records inverse-Weiss and bath-symmetry diagnostics, and exports frequency-resolved $G$, $\Sigma$, $\mathcal{G}_0$, and $\Delta$ without inventing numerical targets from plotted curves. `ed-dmft-scaling-benchmark.jl` separates bath-fit, dense-ED, Lehmann, impurity-solve, and complete-driver costs.

Run the scripts from the repository root with `julia --project=. benchmarks/dmft/<script>.jl`. CSV outputs are written to `benchmarks/dmft/results/`.

Liebsch and Ishida also use comparison with continuous-time quantum Monte Carlo as an independent-solver accuracy criterion. That criterion is not implemented by this native ED-DMFT suite because Phundamental does not yet contain an independent CT-QMC impurity solver; it remains an external validation target rather than being silently replaced by a weaker check.
