# Impurity models, ED, and local response

These workflows retain the original numerical conventions, domain restrictions, and validation instructions.

## Multiorbital impurity models and matrix DMFT

Phundamental supports spin-orbital Anderson impurity Hamiltonians, finite matrix-valued baths, interacting finite-temperature exact diagonalization, and normal-state single-site dynamical mean-field theory (DMFT). Scalar single-orbital ED-DMFT remains available through its original API. No analytic continuation, cluster DMFT, or multiorbital ladder DΓA is implied.

### Spin-orbital interactions

`SpinOrbitalInteraction` stores the antisymmetrized four-index tensor $U_{abcd}$ with the fixed convention

$$H_{\mathrm{int}}=\frac14\sum_{abcd}U_{abcd}d_a^\dagger d_b^\dagger d_d d_c.$$

The tensor must be antisymmetric within each index pair and Hermitian under interchange of the two pairs. The constructor rejects inconsistent input rather than silently altering physical interactions.

`kanamori_interaction(norbitals; U, J, Uprime=U-2J)` constructs interleaved spin-orbital interactions including the Hund spin-flip and pair-hopping terms. The returned spin-orbital order is `orbital_1_up`, `orbital_1_down`, `orbital_2_up`, `orbital_2_down`, etc. General interactions are allowed, but all lattice, bath, Green-function and impurity labels must use the same order.

### Matrix finite baths and exact impurity solving

`MatrixDiscreteBath(energies, hybridizations; labels)` stores real bath energies and complex impurity-to-bath couplings $V_{ap}$. It evaluates

$$\Delta(z)=V\,\mathrm{diag}\!\left((z+\mu-\epsilon_p)^{-1}\right)V^\dagger.$$

Matrix bath fitting is available through `fit_matrix_bath(target, nbath; chemical_potential, options, initial_bath)` and through `fit_bath` dispatch for matrix targets. Its local nonlinear optimizer is deterministic, but neither the globally optimal fit nor exact representation of an arbitrary continuum hybridization is guaranteed. Check the fit objective and convergence with bath size; a successful nonlinear fit is not proof of small physical discretization error.

`MultiOrbitalImpurityProblem(one_body, interaction, axis; chemical_potential, bath)` defines a finite-bath problem. Alternatively supply `target_hybridization` instead of `bath`, and the solver fits the target using its configured bath-site count. `anderson_hamiltonian(problem)` assembles a structured impurity-plus-bath Hamiltonian with the chemical potential included. The interaction tensor acts on the impurity components only.

`MultiOrbitalEDSolver(; bath_sites, bath_fit, kB, backend=:sectorized, max_full_dimension, max_sector_dimension, max_eigensystem_bytes)` solves the complete finite-bath interacting problem. The default exact diagonalization uses conserved total-particle-number sectors; it does not assume that spin-up and spin-down populations are separately conserved. The alternative `backend=:dense` is an independent small-system full-Fock reference. Both methods require complete eigensystems, have exponential system-size costs, and enforce explicit dimension and memory guards.

`solve(solver::MultiOrbitalEDSolver, problem::MultiOrbitalImpurityProblem; initial_bath=nothing)` returns the matrix interacting and noninteracting Green functions, Dyson self-energy, impurity density matrix and occupations, bath, and thermal state. Thermal Lehmann sums include complex orbital off-diagonal terms. Matrix frequency wrappers support `object[i]` for a full frequency matrix, `object[a, b, i]` for a component, and `object.values[:, :, i]` for explicit array slicing.

### Matrix-valued single-site DMFT

`MultiOrbitalDMFTProblem(lattice::MatrixMomentumLattice, interaction::SpinOrbitalInteraction; temperature, chemical_potential=0, one_body=nothing, kB=1, nfrequencies=128, axis=nothing)` defines a Matsubara-axis single-site DMFT calculation. The matrix lattice and local interaction must have matching spin-orbital labels and dimensions. The periodic Bloch gauge is required. The impurity local one-body Hamiltonian is the normalized weighted Brillouin-zone mean $h_{\mathrm{loc}}=\sum_{\mathbf k}w_{\mathbf k}H_0(\mathbf k)$; if `one_body` is supplied it must agree with that mean. This prevents silently changing the onsite energy or double-counting a local term.

The local lattice propagator, Weiss function, and impurity hybridization satisfy

$$G_{\mathrm{loc}}(z)=\sum_{\mathbf k}w_{\mathbf k}\big[(z+\mu)I-H_0(\mathbf k)-\Sigma(z)\big]^{-1},\qquad \mathcal G_0^{-1}=G_{\mathrm{loc}}^{-1}+\Sigma,\qquad \Delta(z)=(z+\mu)I-h_{\mathrm{loc}}-\mathcal G_0^{-1}(z).$$

The implementation streams momentum integration one block at a time, avoiding storage of the full $N_{\mathrm{orb}}^2N_kN_\omega$ momentum-resolved Green-function tensor inside the iteration.

`MultiOrbitalDMFTSolver(; impurity_solver=MultiOrbitalEDSolver(), mixing=LinearMixing(0.5), tol=1e-8, maxiter=100, require_bath_fit_convergence=false, convergence_policy=DMFTConvergencePolicy())` solves the self-energy fixed point using the interacting matrix impurity solver. `LinearMixing` and `AdaptiveLinearMixing` are supported. An independent final impurity re-solve checks the closure residual at the final self-energy, and the result's lattice, Weiss, and hybridization functions are recomputed at that same self-energy.

`MultiOrbitalDMFTResult` contains the final matrix self-energy, local lattice Green function, Weiss Green function, target hybridization, final impurity result, iteration history, convergence status, and diagnostics. `validate_dmft_result(result)` checks Dyson/Weiss consistency, the reported closure residual, and the bath-fit residual; `impurity_local_residual(result)` compares the impurity Green function with the local lattice propagator. Finite-bath fit error is separate from the DMFT fixed-point residual, and reaching one tolerance does not imply meeting the other. The independent closure solve never substitutes for bath-size convergence studies.

Bath diagnostics include `metadata[:bath_fit_absolute_residual] = ‖Δ_bath − Δ_target‖_F`, `metadata[:bath_target_norm] = ‖Δ_target‖_F`, and the existing `metadata[:bath_fit_residual] = ‖Δ_bath − Δ_target‖_F / max(‖Δ_target‖_F, eps(Float64))`. The relative residual is meaningful for nonzero targets but becomes ill-conditioned in the atomic limit, where the exact hybridization is zero and floating-point cancellation may leave a tiny computed target. For an atomic calculation with `bath_sites=0`, validate the **absolute** residual and target norm against numerical precision; do not demand a small relative residual. For dispersing lattices, inspect both diagnostics and test convergence with bath size.

A minimal atomic example is:

```julia
using Phundamental
using LinearAlgebra

interaction = kanamori_interaction(2; U=2.5, J=0.35)
labels = interaction.labels
hlocal = Matrix(Diagonal(ComplexF64[0.1, 0.1, 0.18, 0.18]))
lattice = MatrixMomentumLattice([hlocal, hlocal]; labels=labels)
problem = MultiOrbitalDMFTProblem(lattice, interaction; temperature=0.2, chemical_potential=0.7, nfrequencies=12)
impurity = MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized)
solver = MultiOrbitalDMFTSolver(impurity_solver=impurity, mixing=LinearMixing(1.0), tol=1e-8, maxiter=5)
result = solve(solver, problem)
result.converged || error("DMFT did not converge: $(result.metadata[:termination])")
checks = validate_dmft_result(result)
checks.valid || error("matrix DMFT validation failed")
```

For a dispersing lattice a finite bath approximates the Weiss hybridization and introduces a discretization error. The `bath_sites=0` example is appropriate for the atomic limit, not a general lattice model.

### Explicit high-level forward execution

`MultiOrbitalDMFTForwardContext(model, dmft_problem)` binds a supplied `ManyBodyModel` to an explicitly defined matrix-DMFT problem. It may be used with `ForwardProblem(model; solver=dmft_solver, method_context=context, observable=LocalMatsubaraGreenFunctionRequest())` or `LocalMatsubaraSelfEnergyRequest()`. The execution result retains the full Matsubara matrix and its component labels. The context must be bound to that same model instance.

This explicit context is **not** an automatic symbolic-to-impurity lowering: the user must ensure that the matrix lattice and interaction tensor actually represent the chosen Hamiltonian. Compatibility is consequently marked as not statically certified. Multiorbital DMFT is a declared local approximation and is not automatically selected as an exact forward route. Multiorbital NLCE and ladder DΓA are separate extensions, not silently enabled by this solver.

### Tests and limitations

Run the focused tests with `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-dmft-validation.jl`. The impurity foundation and interacting ED tests are in `test/embedding/multiorbital-impurity-foundation-validation.jl` and `test/embedding/multiorbital-ed-validation.jl`. Check backward compatibility using `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. -e 'using Pkg; Pkg.test()'`, then build the documentation with `julia --project=docs docs/make.jl`.

Finite-temperature exact diagonalization is limited by exponential Fock-space growth and the nonlinear, generally nonunique finite-bath fit. Nonlocal self-energies, cluster DMFT, anomalous superconducting propagators, multiorbital two-particle ladder methods and real-frequency analytic continuation are not included. Report finite-bath, Matsubara-box, and mixing convergence independently; do not interpret a numerically small fixed-point residual as a certified thermodynamic-limit observable.

---

## Multiorbital impurity two-particle correlations

Phundamental can evaluate finite-temperature orbital-resolved particle-hole four-point functions using the same exact-diagonalization thermal states as its multiorbital impurity solver. This provides the local two-particle input for the [multiorbital lattice susceptibility workflow](dga.md). The impurity evaluator itself does not reconstruct nonlocal self-energies.

### Frequency and operator conventions

For ordered spin-orbital pairs $p=(a,b)$ and $q=(c,d)$, the measured correlator is $F_{p,q}(\tau_1,\tau_2,\tau_3)=\langle T_\tau d_a^\dagger(\tau_1)d_b(\tau_2)d_c^\dagger(\tau_3)d_d(0)\rangle$. The connected generalized susceptibility subtracts the **direct** disconnected contribution $\langle T_\tau d_a^\dagger(\tau_1)d_b(\tau_2)\rangle\langle T_\tau d_c^\dagger(\tau_3)d_d(0)\rangle$ before applying the phase $\exp[-i\nu\tau_1+i(\nu+\Omega)\tau_2-i(\nu'+\Omega)\tau_3]$. The exchange Wick contraction is retained. These conventions match the established single-band `TauGridFourPoint` implementation.

The noninteracting matrix bubble for this precise operator ordering is $\chi^0_{ab,cd}(\nu,\nu',\Omega)=-\beta G_{da}(\nu)G_{bc}(\nu+\Omega)\delta_{\nu\nu'}$. This index ordering is important: it differs from some particle-hole pair conventions in the literature and from the `OrbitalLatticeBubble` storage convention. Conversion must be explicit before using the output in a future lattice solver.

### Usage

```julia
axis = FermionicMatsubaraAxis(5.0, [-2, -1, 0, 1])
bosonic = BosonicMatsubaraAxis(5.0, [0, 1])
interaction = kanamori_interaction(1; U=2.0)
bath = MatrixDiscreteBath(Float64[], zeros(ComplexF64, 2, 0); labels=interaction.labels)
problem = MultiOrbitalImpurityProblem(zeros(ComplexF64, 2, 2), interaction, axis; bath=bath)
impurity = solve(MultiOrbitalEDSolver(bath_sites=0), problem)
backend = TauGridFourPoint(ntime=16, max_dimension=16, algorithm=:factored)
data = multiorbital_impurity_two_particle(impurity.thermal_state, axis, bosonic; backend=backend, pairs=[(1, 1), (2, 2)])
block = orbital_pair_block(data.susceptibility, (1, 1), (2, 2))
```

`data.susceptibility.values[i,j,n,np,m]` is a five-dimensional complex array indexed by selected orbital pair, selected orbital pair, incoming fermionic frequency, outgoing fermionic frequency, and bosonic frequency. `data.bubble` has the same shape and is built directly from the exact interacting ED one-particle Green function with all required shifted Matsubara frequencies included. In the interacting case this is the conventional bare particle-hole bubble built from *dressed* one-particle propagators, not the interacting four-point function itself. `orbital_pair_block` converts one component to the existing scalar-channel `GeneralizedSusceptibility` for inspection and comparison; it does not imply that an off-diagonal orbital block forms a closed physical scattering channel. `project_orbital_channel(data.susceptibility, [1, -1]; channel=:magnetic)` forms an explicit weighted sum of the selected pair channels. Use such charge/magnetic names only when the selected pairs describe corresponding spin densities and the model supports that interpretation; no spin symmetry is inferred automatically.

### Algorithms and resource limits

The evaluator caches impurity annihilation operators transformed into the complete ED energy eigenbasis and processes one orbital-pair block at a time. It reuses the existing deterministic direct and factored midpoint-time kernels, including symmetric equal-time averaging. The algorithm is exact with respect to the finite-bath ED spectrum but the **three-dimensional imaginary-time Fourier quadrature remains finite-grid approximate**. Increase `ntime` and compare results using `multiorbital_two_particle_residual` to quantify quadrature convergence. The code does not claim an exact continuous-frequency Lehmann four-point representation.

The complete dense output requires $16N_p^2N_\nu^2N_\Omega$ bytes for each complex susceptibility or bubble, where $N_p$ is the number of selected ordered orbital pairs. For all $d^2$ pairs this is $O(d^4N_\nu^2N_\Omega)$ elements. Explicit `max_output_bytes`, `max_operator_bytes`, and `max_workspace_bytes` parameters reject excessive requests before large allocations. The underlying `TauGridFourPoint.max_dimension` guards the complete Fock-space eigenoperator cache; this reference algorithm is intended for small and moderate finite-bath systems, not arbitrarily large impurity Hilbert spaces.

The exact bubble is diagonal in fermionic indices and uses an expanded frequency axis for $\nu+\Omega$ rather than zero-filling shifted values absent from the user-requested axis. The full tensor retains orbital mixing, complex hopping, and interactions as represented by the ED state; spin symmetry reductions are not silently applied.

### Local vertex extraction

`multiorbital_irreducible_vertex(data.bubble, data.susceptibility; regularization=0, pseudoinverse=false)` computes $\Gamma=(\chi^0)^{-1}-\chi^{-1}$ independently at each transfer frequency. The inversion uses pair-major/frequency-minor matrix indexing and records singular-value diagnostics. A truncated frequency box and finite quadrature can produce an apparent nonzero vertex even for a free impurity, so inspect convergence before interpreting the result physically. Inversion may be singular for an incomplete orbital-pair selection or nearly null frequency components; pseudoinverse or regularization changes the inferred vertex and must be reported. The extracted vertex can be passed to `bethe_salpeter` or `multiorbital_ladder_susceptibility` using the [lattice susceptibility workflow](dga.md). Schwinger–Dyson reconstruction remains separate.

### Validation

Run `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-two-particle-validation.jl`, then the full package regression suite and `julia --project=docs docs/make.jl`. Tests compare sectorized and dense ED backends, direct and factored time-grid traces, independently constructed full-Fock reference operators, the analytical complex matrix Wick bubble, vertex extraction, and resource guards. Measure elapsed time, allocations, and estimated live workspaces using `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/multiorbital-two-particle.jl`. Performance and frequency-grid convergence should be benchmarked for the intended physical model before treating numerical vertices as production-quality.

---

## Exact Orbital-Bilinear Correlations and High-Frequency Inputs

The multiorbital finite-bath ED solver can compute equal-time orbital-bilinear correlation functions and their entire bosonic Matsubara dependence without imaginary-time quadrature. These quantities are necessary inputs for the one-transfer-frequency sector of a multiorbital vertex asymptotic expansion. They are **not** sufficient to reconstruct the complete fermionic-frequency-dependent local vertex: its interaction-tensor contractions, channel permutations, and three-leg structures require additional derivation and validation.

### Definition and eigenbasis evaluation

For an ordered spin-orbital pair $(a,b)$, define the bilinear $B_{ab}=d_a^\dagger d_b$. The connected thermal response is

```math
\chi_{ab,cd}(i\Omega_m)=\int_0^\beta d\tau\,e^{i\Omega_m\tau}\left[\langle B_{ab}(\tau)B_{cd}(0)\rangle-\langle B_{ab}\rangle\langle B_{cd}\rangle\right].
```

For eigenstates $|s\rangle$, energies $E_s$, and Gibbs weights $p_s=e^{-\beta E_s}/Z$, the exact finite-bath result for nondegenerate transitions is

```math
\chi_{AB}(z)=\sum_{s,t}\frac{p_s-p_t}{E_t-E_s-z}A_{st}B_{ts},\qquad z=i\Omega_m.
```

At $m=0$, degenerate transitions contribute $\beta p_s A_{st}B_{ts}$ and the connected subtraction is $-\beta\langle A\rangle\langle B\rangle$. The zero-frequency elastic term must not be extrapolated from the nonzero-frequency asymptotic expansion. The code evaluates all sectors, including exactly degenerate levels, and preserves the ordering of the selected orbital pairs.

The `orbital_bilinear_spin_channels(data, axis)` method additionally projects the complete spin-interleaved correlator onto longitudinal density and magnetic channels. It verifies the two independent spin-label equalities and refuses to infer a transverse SU(2) Ward identity or an asymptotic fermionic vertex from them.

### Certified finite-bath bosonic moments

For $|z|>R$, where $R=\max_{s,t}|E_t-E_s|$, the nonzero-frequency response has a convergent Laurent series

```math
\chi_{AB}(z)=\sum_{k=0}^{p-1}\frac{M_{AB}^{(k)}}{z^{k+1}}+\mathcal R_{AB}^{(p)}(z),\qquad M_{AB}^{(k)}=-\sum_{s,t}(p_s-p_t)(E_t-E_s)^k A_{st}B_{ts}.
```

Define $S_{AB}=\sum_{s,t}|p_s-p_t|\,|A_{st}B_{ts}|$. The finite ED spectrum supplies the elementwise rigorous remainder bound

```math
|\mathcal R_{AB}^{(p)}(z)|\leq S_{AB}\frac{(R/|z|)^p}{|z|-R}.
```

The bound certifies the **bosonic bilinear correlator only**. It is not a vertex-asymptotic remainder estimate and cannot be used to set `two_particle_vertex_tail_certified=true` in a DΓA result.

### API example

```julia
using Phundamental

## `state` is a complete MultiOrbitalEDThermalState from finite-bath ED.
response_data = orbital_bilinear_lehmann(state; pairs=[(1, 1), (2, 2)])
axis = BosonicMatsubaraAxis(state.beta, [-1, 0, 1])
chi = orbital_bilinear_susceptibility(response_data, axis)

moments = orbital_bilinear_moments(response_data; order=4)
high_axis = BosonicMatsubaraAxis(state.beta, [50, 70, 90])
tail = orbital_bilinear_tail(moments, high_axis)
certified_elementwise_bounds = tail.error_bounds
```

The evaluator uses exact diagonalization transition operators already supported by the impurity solver. Operator-storage and result-storage limits are checked before allocation. For $P$ selected orbital pairs and full Fock dimension $D$, retaining the bilinear operators requires $O(PD^2)$ complex elements; the response requires $O(P^2N_\Omega)$ elements. Restricting `pairs` can substantially reduce memory usage.

### Interacting lattice convergence benchmark

The separate `benchmarks/multiorbital-interacting-convergence.jl` script uses a nonzero Hubbard interaction and a two-momentum dispersive lattice to execute the existing four-stage frequency-support and imaginary-time-quadrature comparison. It runs the complete ladder computation once before taking timed measurements and reports the residuals, allocations, nonlocal self-energy amplitude, and finite-bath fitting discrepancy.

The fixture deliberately uses zero bath sites so it remains a small, reproducible stress test. The resulting finite-bath approximation does **not** reproduce the dispersive target hybridization. Consequently, its result is a numerical integration benchmark, not a quantitatively converged DMFT solution for the dispersive lattice. The script separately reports whether the two finite grids satisfy the user-defined numerical tolerance; completing the comparison does not imply convergence.

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-bilinear-asymptotics-validation.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/multiorbital-interacting-convergence.jl
julia --project=docs docs/make.jl
```

The complete multiorbital vertex asymptotic construction requires the channel-dependent bare-interaction contractions, two-bilinear kernels, and three-leg kernels. These are described in Kaufmann, Gunacker, and Held, *Physical Review B* **96**, 035114 (2017), [DOI: 10.1103/PhysRevB.96.035114](https://doi.org/10.1103/PhysRevB.96.035114). They must be mapped into the package's established orbital and Matsubara conventions before enabling an analytic vertex-tail certification flag.

---
