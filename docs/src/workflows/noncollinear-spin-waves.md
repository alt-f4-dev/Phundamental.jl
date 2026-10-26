# Noncollinear SU(2) spin waves

`SpinWaveSolver` expands a spin Hamiltonian about a user-supplied stationary set of ordered directions. This is a quadratic Holstein–Primakoff approximation, not an exact basis transformation, and does not redefine model-space admissibility. The existing `LinearSpinWaveSolver` retains its legacy Heisenberg/XXZ and collinear-only API.

## Hamiltonian and units

The supported Hamiltonian is $H=\sum_{i<j}\mathbf s_i^T J_{ij}\mathbf s_j+\sum_i\mathbf s_i^T K_i\mathbf s_i-\sum_i\mathbf h_i^T\mathbf s_i+C$, where $\mathbf s_i=\mathbf S_i/\hbar$. The intersite $J_{ij}$ are real $3\times3$ tensors, including anisotropic exchange and antisymmetric DM terms. Each onsite $K_i$ is a real symmetric $3\times3$ tensor. Off-diagonal single-ion terms mean the Hermitian symmetrized operator $K_{i,\alpha\beta}\{s_i^\alpha,s_i^\beta\}/2$ summed over Cartesian indices. The solver extracts the coefficients from the actual ordered symbolic operator expression, not solely from construction metadata.

`TensorSpinModel(exchange; field=..., anisotropy=..., shell=...)` accepts `anisotropy=nothing` (default), a uniform real symmetric matrix, a vector with one matrix per site, or a callable `(i, site)` returning a matrix. A tensor with $K_{zz}=D$ adds $D(s_i^z)^2$. The exchange specification retains its existing matrix, bond-shell, or per-bond callable support. Real nonsymmetric single-ion tensors are rejected instead of silently converting antisymmetric onsite terms into an unphysical quadratic anisotropy. Symbolically supplied antisymmetric *ordered* products are instead reduced using the SU(2) commutator into effective linear fields, and the resulting Hamiltonian must be Hermitian.

```julia
K = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -0.3]
spec = TensorSpinModel(-Matrix{Float64}(I, 3, 3); anisotropy=K, field=(0.0, 0.0, 2.0), shell=1)
model = build_model(material, cluster, spec)
reference = MagneticReferenceState([[0.0, 0.0, 1.0] for _ in 1:nsites(cluster)])
solution = solve(SpinWaveSolver(reference=reference), model)
```

## Onsite spin quantum mechanics

Same-site operators do not commute. For real symmetric $K_i$, the implementation matches the exact local spin-$S_i$ matrix elements between the highest-weight state and its one- and two-spin-flip states. In a local frame with $\widetilde K=R_i^TK_iR_i$, this produces $E_i=S_i(S_i-\tfrac12)\widetilde K_{zz}+\tfrac{S_i}{2}\operatorname{tr}\widetilde K$, $A_{ii}^{\mathrm{ion}}=(2S_i-1)[(\widetilde K_{xx}+\widetilde K_{yy})/2-\widetilde K_{zz}]$, and $B_{ii}^{\mathrm{ion}}=\sqrt{S_i(2S_i-1)/2}\,[\widetilde K_{xx}-\widetilde K_{yy}+2i\widetilde K_{xy}]$. These are finite-spin kinematic corrections to the onsite Gaussian expansion; intersite interactions continue to use the standard leading Holstein–Primakoff expansion. This is not an exact treatment of interacting spin waves or higher-order $1/S$ terms.

The reference-state stationarity test uses the gradient of this same quantum coherent-state onsite energy: its contribution per unit spin is $(2S_i-1)K_i\mathbf n_i$. A symmetric onsite term for $S_i=\tfrac12$ is exactly proportional to the identity and contributes neither a torque nor a one-magnon gap. For $S_i=1$ with $D(s_i^z)^2$, the axial one-magnon shift is $-D$ when polarized along $+z$. An isotropic $K_i$ contributes only the exact constant $S_i(S_i+1)K_i$.

## Finite-cell reference and solution

Reference directions must be normalized; the library chooses right-handed transverse frames, and `transverse_axes` may set the local gauge. The solver checks transverse stationarity and quadratic energetic stability, then constructs Hermitian $A$ and symmetric $B$ for the established bosonic BdG backend. A nonstationary or energetically unstable reference raises an error. The result metadata identifies the onsite interaction scope and finite-spin truncation.

Strictly singular paired bosonic problems are rejected by default. With `zero_modes=:regularize`, the solver records an explicit positive regulator; these regulated modes do not certify physical Goldstone vectors or the corresponding scattering tensor.

## Tensor scattering and periodic bands

`SpinTensorStructureFactorRequest` accepts the resulting `SpinWaveResult` for laboratory-frame elastic and one-magnon tensor scattering. Gaussian longitudinal two-magnon continua are not implemented. `spinwave_dispersion(solver, model, qpoints)` includes the same onsite $A_{ii},B_{ii}$ contributions at each Bloch momentum, with certified translated-image intersite bonds from `TensorSpinModel`. Periodic one-magnon Bloch scattering is now available as an explicit `SpinTensorStructureFactorRequest` option with `metadata=Dict(:spinwave_momentum_scope => :magnetic_supercell_bloch)`. See [Site SU(N) generalized spin waves](sun-spin-waves.md) for the common `:dipole`/`:SUN` Bloch conventions, reference optimization, and their limitations. Incommensurate rotating-frame states remain unsupported.

## Capability contracts and tests

`SpinWaveSolver` continues to report approximation `:linear_spin_wave` and output family `:spin_wave`, with the existing explicit `ForwardExecutionPolicy` authorization rules. `solver_compatibility` tests operator extraction, tensor Hermiticity, reference dimensions, and stationary torques, while execution performs the final BdG stability checks. The `LinearSpinWaveSolver` contract remains unchanged.

The original `test/noncollinear-spin-wave-validation.jl` remains installed. Add `test/single-ion-spinwave-validation.jl` and the updated `test/runtests.jl`, and validate with `OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()'` followed by `julia --project=docs docs/make.jl`.
