# Diagrammatic extensions and ladder workflows

These workflows retain the original numerical conventions, domain restrictions, and validation instructions.

## Multiorbital lattice susceptibilities

Phundamental evaluates particle-hole lattice susceptibilities from an orbital-resolved Matsubara Green function and a local irreducible vertex. The calculation uses the existing multiorbital impurity two-particle convention and solves an orbital-pair/frequency Bethe–Salpeter equation independently at each momentum transfer and bosonic frequency. This is a ladder susceptibility calculation, **not** a multiorbital Schwinger–Dyson self-energy reconstruction or a self-consistent nonlocal embedding method.

### Operator and index conventions

An ordered orbital pair $p=(a,b)$ labels the local bilinear $d_a^\dagger d_b$, and $q=(c,d)$ labels $d_c^\dagger d_d$. The impurity convention is $\chi^0_{ab,cd}(\nu,\nu',\Omega)=-\beta\delta_{\nu\nu'}G_{da}(\nu)G_{bc}(\nu+\Omega)$. The corresponding momentum-dependent bubble is

```math
\chi^0_{ab,cd}(\mathbf q,\nu,\nu',\Omega)=-\beta\delta_{\nu\nu'}\sum_{\mathbf k}w_{\mathbf k}G_{da}(\mathbf k,\nu)G_{bc}(\mathbf k+\mathbf q,\nu+\Omega).
```

The older `OrbitalLatticeBubble` stores $-\beta\sum_{\mathbf k}w_{\mathbf k}G_{ac}(\mathbf k+\mathbf q,\nu+\Omega)G_{db}(\mathbf k,\nu)$ at indices `[a,b,c,d,ν,Ω]`. For the impurity pair ordering, its matching element is `legacy.values[b,a,c,d,ν,Ω]`, not `legacy.values[a,b,c,d,ν,Ω]`. `orbital_pair_lattice_bubble` applies this conversion explicitly.

At fixed $\Omega$, the composite matrix index is $I=(p-1)N_\nu+n$. The finite-box ladder equation and normalization are

```math
(\mathbf I-\boldsymbol\chi^0(\mathbf q,\Omega)\boldsymbol\Gamma(\Omega))\boldsymbol\chi(\mathbf q,\Omega)=\boldsymbol\chi^0(\mathbf q,\Omega).
```

This ordering is consistent with `multiorbital_irreducible_vertex`, which extracts $\Gamma=(\chi^0)^{-1}-\chi^{-1}$. The implementation factorizes the finite-box kernel with LU and solves its matrix right-hand side. It does not form an explicit matrix inverse. Every returned transfer-frequency block records a normalized equation residual and the minimum-to-maximum LU pivot magnitude ratio; the pivot ratio is a diagnostic, **not** a rigorous condition-number estimate.

### Required frequency data

The single-particle frequency axis must contain every requested fermionic index $n$ **and** shifted index $n+m$ for all selected incoming frequencies and bosonic transfers $m$. The two-particle frequency box may therefore be smaller than the single-particle axis. Incomplete shifted values raise `ArgumentError`, including when converting an existing zero-filled legacy bubble. The strict interface never treats missing values as zero.

Only a periodic Bloch gauge with explicit momentum index permutations is supported; nonperiodic embedded-orbital conventions require additional phase transport and must not be used without it. Momentum weights are normalized by `MatrixMomentumResolvedGreenFunction`.

### Example

```julia
using LinearAlgebra

beta = 5.0
single_particle = FermionicMatsubaraAxis(beta, [-2, -1, 0, 1, 2])
selected = FermionicMatsubaraAxis(beta, [-1, 0])
bosonic = BosonicMatsubaraAxis(beta, [0, 1])
labels = [:up, :down]
nk = 4
values = zeros(ComplexF64, 2, 2, nk, length(single_particle))
for ik in 1:nk, n in 1:length(single_particle)
    values[:, :, ik, n] .= inv(single_particle[n] * Matrix{ComplexF64}(I, 2, 2) -
        ComplexF64[0.2 0.05im; -0.05im -0.1])
end
green = MatrixMomentumResolvedGreenFunction(single_particle, collect(1:nk), values; labels=labels)
qmaps = periodic_momentum_shift_maps((nk,)).maps
pairs = [(a, b) for a in 1:2 for b in 1:2]
vertex_values = zeros(ComplexF64, 4, 4, length(selected), length(selected), length(bosonic))
vertex = OrbitalPairIrreducibleVertex(selected, bosonic, pairs, labels, vertex_values, Dict{Symbol,Any}())
bubbles = orbital_pair_lattice_bubble(green, qmaps, selected, bosonic; pairs=pairs)
susceptibility = bethe_salpeter(bubbles[1], vertex)
result = multiorbital_ladder_susceptibility(green, qmaps, vertex;
    projection_weights=ComplexF64[1, 0, 0, 1], return_full=false)
```

The example uses a zero vertex, so the returned susceptibilities equal the momentum-dependent bubbles; it is intended to demonstrate the interface, not provide an interacting impurity-vertex calculation. To compute an interacting vertex, use `multiorbital_irreducible_vertex` with a converged and sufficiently large impurity two-particle frequency/time grid, and check its inversion diagnostics before lattice calculations.

### Output and resource controls

`orbital_pair_lattice_bubble` returns one `OrbitalPairSusceptibility` per momentum-transfer map; `bethe_salpeter` solves one such input. `multiorbital_ladder_susceptibility` streams momentum transfers and returns `OrbitalLadderResult`. When `return_full=true`, `susceptibilities` holds the complete five-dimensional tensor for each momentum transfer. When `return_full=false`, the solver requires `projection_weights` and keeps only the contracted $(N_q,N_\Omega)$ array. The contraction is $\beta^{-2}\sum_{p,q,\nu,\nu'}w_p^*\chi_{p,q}(\nu,\nu',\Omega)w_q$; the weights must be selected for the intended physical observable, without assuming an orbital or spin symmetry. The optional `qpoints` keyword supplies explicit labels; otherwise transfer indices are used.

The full retained tensor scales as $16N_qN_p^2N_\nu^2N_\Omega$ bytes. Each streaming step also requires the current bubble and solved tensor and four dense $D\times D$ complex matrices with $D=N_pN_\nu$. `max_workspace_bytes` and `max_output_bytes` reject estimated oversize requests **before** these workspaces are allocated. These are storage estimates, not measured process peak memory, and exclude pre-existing single-particle and vertex inputs and Julia runtime overhead. The implementation does not precompute or retain momentum-by-momentum vertex matrices. `min_relative_pivot` is a heuristic rejection threshold; resolving nearly critical ladders requires a separate conditioning/convergence study.

A subset of orbital pairs generally does **not** form a closed interaction channel. By default a complete ordered-pair set is required; `allow_projected_pairs=true` explicitly permits projected finite-subspace calculations and labels their output accordingly. The method does not infer spin conservation, SU(2), or orbital block diagonalization.

### Validation

Run `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-bethe-salpeter-validation.jl`, then the full regression suite and `julia --project=docs docs/make.jl`. The tests compare independently constructed complex momentum bubbles, legacy index adaptation, frequency-grid rejection, dense linear solves, scalar-channel reduction, zero-vertex closure, and momentum streaming. Profile representative momentum meshes, orbital dimensions, and Matsubara boxes with `OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/multiorbital-bethe-salpeter.jl`; correctness and efficiency on large systems require independent Julia execution and measurements.

---

## Multiorbital Self-Energy Reconstruction

This workflow extends normal-state matrix DMFT with a momentum-dependent Schwinger–Dyson contraction, matrix Dyson reconstruction, and an explicitly identified screened particle-hole ladder approximation. The implementations use the same ordered spin-orbital labels as the impurity and momentum-lattice Green functions. They do not change the established scalar ladder DΓA route.

### Mathematical conventions

For the antisymmetrized interaction $U_{ab,cd}$, the local Hamiltonian contains $\frac14\sum_{abcd}U_{ab,cd}d_a^\dagger d_b^\dagger d_d d_c$. Using $\rho_{ab}=\langle d_b^\dagger d_a\rangle$, the static Hartree–Fock matrix is $\Sigma^{\mathrm{HF}}_{ab}=\sum_{cd}U_{ac,bd}\rho_{dc}$. The function `orbital_hartree_fock` evaluates this matrix without assuming orbital diagonality or collinear spin.

For a user-supplied, explicitly contracted four-index kernel $K_{ab,cd}(\mathbf q,\nu,\Omega)$, `orbital_schwinger_dyson` evaluates the defined one-shot convolution

$$C_{ab}(\mathbf k,\nu)=-\frac1\beta\sum_{\mathbf q,\Omega}w_{\mathbf q}\sum_{cd}K_{ab,cd}(\mathbf q,\nu,\Omega)G_{cd}(\mathbf k+\mathbf q,\nu+\Omega).$$

Its default local-projection prescription is $\Delta\Sigma_{ab}(\mathbf k,\nu)=C_{ab}(\mathbf k,\nu)-\sum_{\mathbf k'}w_{\mathbf k'}C_{ab}(\mathbf k',\nu)$ and $\Sigma_{ab}(\mathbf k,\nu)=\Sigma^{\mathrm{DMFT}}_{ab}(\nu)+\Delta\Sigma_{ab}(\mathbf k,\nu)$. This guarantees that the momentum-weighted correction vanishes, but it is **a specified local-projection approximation** and does not establish equivalence to diagrammatic double-counting cancellation in a crossing-completed multiorbital DΓA theory. The converged DMFT self-energy is included exactly once; no extra Hartree–Fock matrix is added on top of it.

The matrix Dyson propagator is $G(\mathbf k,z)=[(z+\mu)I-H_0(\mathbf k)-\Sigma(\mathbf k,z)]^{-1}$. The implementation performs a matrix linear solve for each momentum and frequency. Every shifted fermionic frequency $\nu+\Omega$ must be present on the supplied single-particle Matsubara grid.

### Two intentionally different kernel sources

`OrbitalSchwingerDysonKernel` accepts a callback `fill!(destination, iq, iν, iΩ)`, where the destination is an array of shape `(norbital, norbital, norbital, norbital)`. The callback is responsible for constructing the complete interaction/vertex/bubble contraction in this defined ordering. A provenance declaration of `:externally_certified_crossing` is **the caller's assertion**, not a verification by Phundamental. Orbital crossing, the transverse particle-hole contribution, local-vertex consistency, and all diagrammatic combinatorial coefficients must be independently established before identifying such an input with a full DΓA Schwinger–Dyson kernel.

`screened_orbital_ladder_kernel` instead builds an opt-in screened particle-hole approximation directly from the available `OrbitalLadderResult`. Defining the complete canonical orbital-pair index $P=(a,b)$, it computes the pair-channel matrix $U^{\mathrm{ph}}_{(a,b),(c,d)}=U_{ac,bd}$ and the fermion-frequency-contracted response $X_{PQ}(\mathbf q,\Omega)=\beta^{-2}\sum_{\nu,\nu'}\chi_{PQ}(\mathbf q,\nu,\nu',\Omega)$. Its screened interaction is

$$W(\mathbf q,\Omega)=U^{\mathrm{ph}}[X(\mathbf q,\Omega)-\langle X(\mathbf q,\Omega)\rangle_{\mathbf q}]U^{\mathrm{ph}}.$$

The callback maps $K_{ab,cd}=W_{(a,c),(b,d)}$. This is a well-defined **screened particle-hole/fluctuation-exchange-like approximation**, not the complete crossing-consistent multiorbital ladder DΓA self-energy. In particular, an impurity particle-hole vertex and its single-channel Bethe–Salpeter solution are not enough to reconstruct all parquet channel contributions. The result explicitly records `:full_multiorbital_dga => false`.

### One-shot screened-ladder execution

A converged `MultiOrbitalDMFTResult` is required. An `OrbitalPairIrreducibleVertex` must use the full canonical orbital-pair selection and the DMFT fermionic frequency box. The vertex is assumed to have been obtained separately from the impurity correlation calculation. The explicit momentum-shift maps must form valid permutations of the single-particle momentum grid.

```julia
bosonic_axis = BosonicMatsubaraAxis(dmft.problem.axis.beta, [0, 1])
expanded_green = expanded_dmft_momentum_green(dmft, bosonic_axis)
problem = OrbitalScreenedLadderProblem(dmft, expanded_green, vertex, bosonic_axis)
solver = OrbitalScreenedLadderSolver(max_workspace_bytes=512 * 1024^2)
result = solve(solver, problem)
sigma_k = result.nonlocal_self_energy
G_k = result.corrected_green
report = validate_orbital_schwinger_dyson(dmft, sigma_k)
```

`expanded_dmft_momentum_green` reevaluates the converged finite-bath impurity Green function and Weiss resolvent on an enlarged Matsubara grid using the stored ED thermal state. It does not extrapolate missing self-energy values or substitute zeros. The problem also has a constructor which creates this expanded Green function automatically.

For explicit high-level execution, bind `OrbitalScreenedLadderForwardContext(model, problem)` to the declaring `ManyBodyModel`, use `OrbitalScreenedLadderSolver`, and request `MomentumMatsubaraSelfEnergyRequest()`. This route is deliberately marked as not having certified symbolic-to-lattice lowering. The pre-existing `LocalMatsubaraSelfEnergyRequest` returns the retained DMFT local self-energy, not a momentum-resolved response.

### Error control and computational limits

The code rejects nonconverged DMFT references, mismatched spin-orbital labels, incompatible momentum maps, missing shifted Matsubara frequencies, nonfinite kernels, and memory estimates exceeding the caller's limits. The four-index Schwinger–Dyson workspace scales as $O(n^4)$ for each streamed $(q,\nu,\Omega)$ block, while the retained self-energy scales as $O(n^2N_kN_\nu)$. In the integrated screened approximation, the retained frequency-contracted ladder kernel scales as $O(n^4N_qN_\Omega)$; individual full Bethe–Salpeter susceptibilities are processed one momentum point at a time and discarded. The standalone `screened_orbital_ladder_kernel` accepts an existing fully retained `OrbitalLadderResult` when it is already available. Cumulative Julia allocations are not the same as measured peak resident memory.

Validation should check (i) Hartree–Fock signs and spin-orbital ordering, (ii) independent particle-hole full-vertex reconstruction, (iii) zero-kernel and atomic or q-independent cancellation, (iv) matrix Dyson reconstruction, (v) weighted local projection, and (vi) serial runtime and allocation scaling. **Neither these tests nor the screened-ladder route certify a crossing-complete DΓA approximation.** Frequency-box truncation, vertex asymptotics, and a consistent channel-combination prescription remain requirements before claiming that stronger result.

---

## Orbital-Resolved Particle-Hole Crossing

Phundamental provides density and magnetic particle-hole ladder reconstruction for normal-state, spin-rotation-invariant multiorbital Hamiltonians. This functionality retains both incoming fermionic frequencies, both spin channels, and all spatial-orbital-pair indices. It extends the orbital-pair Bethe–Salpeter solver without changing the existing screened particle-hole route.

### Physical assumptions and scope

The reconstruction assumes spin SU(2) symmetry and a local interaction. The spatial-orbital density and magnetic irreducible vertices must be supplied in mutually consistent spin and orbital conventions. For a spin-interleaved impurity Hamiltonian, `orbital_su2_symmetry(one_body, interaction)` verifies that the one-body matrix and antisymmetrized local interaction commute with the SU(2) generators; it does **not** validate the independently supplied frequency-dependent vertices or the momentum-dependent lattice Hamiltonian.

This implementation provides channel-resolved vertex reconstruction and exact finite-box rejection. **It does not by itself supply a complete, physically certified multiorbital DΓA self-energy.** In particular, converting the crossing-reconstructed vertices to the exact Schwinger–Dyson contraction and the published diagrammatic local subtraction, and implementing controlled high-frequency vertex tails, require further independent numerical validation. The original screened particle-hole solver remains a distinct approximation.

### Crossed-channel construction

Let $F_r^{\mathrm{loc}}(\nu,\nu',\Omega)$ denote a local full particle-hole vertex in the density ($r=d$) or magnetic ($r=m$) channel, and define $\Delta F_r(q,\nu,\nu',\Omega)=F_r^q-F_r^{\mathrm{loc}}$. For spatial orbital indices $(a,b,c,d)$, the transverse particle-hole contribution exchanges the labels $(a,b,c,d)\mapsto(c,b,a,d)$. Its transfer variables map to $(q',\nu' - \Omega,\nu',\nu'-\nu)$, with $q'=k'-k$.

The SU(2) ladder reconstruction reads

```math
\begin{aligned}
F_d &= F_d^{\mathrm{loc}}+\Delta F_d^{q}-\tfrac12\Delta F_d^{q'}\vert_{\mathrm{cross}}-\tfrac32\Delta F_m^{q'}\vert_{\mathrm{cross}},\\
F_m &= F_m^{\mathrm{loc}}+\Delta F_m^{q}-\tfrac12\Delta F_d^{q'}\vert_{\mathrm{cross}}+\tfrac12\Delta F_m^{q'}\vert_{\mathrm{cross}}.
\end{aligned}
```

The corresponding construction is described for SU(2)-invariant multiorbital ladder DΓA by Galler *et al.*, *Physical Review B* **95**, 115107 (2017), and in Peil, *Multi-Orbital Dynamical Vertex Approximation* (2025), Section 4.2. These equations are **not** a generic spin-mixing identity.

### API

`orbital_spin_ladders(local_bubble, vertex_d, vertex_m, q_bubbles, qmaps; spin_symmetry_certified=true)` solves each channel's Bethe–Salpeter equation and extracts its local and momentum-dependent full vertices. An alternative constructor, `OrbitalSpinLadderChannels(local_Fd, local_Fm, q_Fd, q_Fm, qmaps; spin_symmetry_certified=true)`, accepts already reconstructed full vertices and avoids duplicate inversion. In either case, use complete canonical spatial-orbital pairs `(1,1), (1,2), ..., (n,n)`.

`orbital_crossed_vertex(channels, iq, ik, ikprime, iν, iνprime, iΩ, a, b, c, d)` returns a named tuple with the density and magnetic full-vertex values. Momentum arguments refer to indices, not physical momentum coordinates. Every `qmap` must be a permutation of the full momentum grid, and `qmap[iq][ik]` must locate `k+q`.

`orbital_crossing_coverage(channels)` reports whether every requested crossed frequency and momentum transfer is represented. If a crossed transfer is absent, `orbital_crossed_vertex` throws `ArgumentError`; it never substitutes a zero or silently extrapolates. Typical small Matsubara boxes are not closed under crossing, so full-grid evaluation generally requires a separate asymptotic/completion method.

The constructors enforce explicit output-memory limits using `max_retained_bytes`. The Bethe–Salpeter route also enforces `max_workspace_bytes`. These budgets bound retained vertex arrays and solver workspaces, not all allocations made by Julia or peak resident memory.

### Correctness and performance

The test suite checks the density/magnetic crossing coefficients against an independent scalar construction, the `(c,b),(a,d)` orbital permutation, missing-frequency rejection, the noninteracting matrix-ladder equations, canonical-pair coverage, memory guards, and SU(2) Hamiltonian certification. Run `test/embedding/multiorbital-crossing-validation.jl` before the full suite. The separate benchmark `benchmarks/multiorbital-crossing.jl` measures the cost of selected crossing queries; it does not benchmark a complete Schwinger–Dyson calculation.

The remaining end-to-end acceptance is the physically derived SU(2) Schwinger–Dyson self-energy, its diagrammatic local subtraction, controlled frequency tails, the single-band reference limit, and full numerical benchmarks. A successful channel-crossing test is not equivalent to those acceptance gates passing.

---

## Impurity Spin Channels and Crossing-Frequency Support

Phundamental can project complete spin-orbital impurity four-point tensors into orbital-resolved longitudinal density and magnetic channels. It can also evaluate the previously available particle-hole crossing transformation on a small requested frequency core using a larger, explicitly populated Matsubara support grid. Neither operation is equivalent to a high-frequency vertex asymptotic construction or a complete Schwinger–Dyson reconstruction.

### Spin-orbital ordering and normalization

The impurity must use the interleaved basis `(orbital1_up, orbital1_down, orbital2_up, orbital2_down, ...)` and must supply all canonical spin-orbital pairs. With `χ_{σσ′}` denoting the connected particle-hole correlator of bilinears carrying the indicated spins, the projected channels use the normalization of the existing scalar charge/magnetic decomposition:

```math
\chi_d^{ab,cd} = \chi_{\uparrow\uparrow}^{ab,cd}+\chi_{\uparrow\downarrow}^{ab,cd},\qquad \chi_m^{ab,cd} = \chi_{\uparrow\uparrow}^{ab,cd}-\chi_{\uparrow\downarrow}^{ab,cd}.
```

`orbital_impurity_spin_channels(object; spatial_labels=nothing, atol=1e-9, rtol=1e-7)` returns the two `OrbitalPairSusceptibility` objects and projection diagnostics. It checks the independent longitudinal identities `χ↑↑≈χ↓↓` and `χ↑↓≈χ↓↑`. These checks alone do not certify every SU(2) Ward identity; additionally check the input Hamiltonian using `orbital_su2_symmetry` and verify the projected two-particle conventions for the intended physical calculation.

For an impurity two-particle result, `orbital_impurity_spin_channels(result; compute_vertices=false)` projects its susceptibility and independently computed Wick bubble. Setting `compute_vertices=true` additionally extracts both finite-box irreducible vertices; inversion parameters may be supplied in `vertex_kwargs=(; ...)`. Numerical regularization and finite imaginary-time quadrature errors are inherited from the underlying impurity result and must be checked independently.

### Crossing core and frequency support

`OrbitalCrossingCore(channels, core_fermionic, core_bosonic)` uses an existing `OrbitalSpinLadderChannels` object whose Matsubara axes contain the larger support box. Each requested external core frequency must exist on that support. For every core triple `(n,n′,m)`, the crossed vertex additionally requires the fermionic support value `n′-m` and bosonic support value `n′-n`.

A complete requested core can therefore be much smaller than the support box. The constructor checks these requirements without duplicating any vertex tensor. `orbital_core_crossed_vertex` evaluates the established crossed-channel formula on that core by mapping the requested external indices onto the support axes. An incomplete core is rejected by default; `require_complete=false` only creates a diagnostic object and does not silently fill missing values.

These are **finite support checks**, not an asymptotic vertex continuation or a guarantee that the Matsubara sums in a self-energy have converged. Summations outside the support box remain unresolved unless justified analytic tails or independent convergence tests are supplied.

### Computational limits and scientific status

The spin projection allocates two spatial-orbital tensors, with output storage proportional to `2 × 16 × norb^4 × Nν² × NΩ` bytes. The core object stores only frequency-index lookup arrays and its diagnostics; it reuses the existing momentum and vertex arrays. Both constructors check memory limits before allocating their own large outputs. Reported byte estimates are not measurements of process peak resident memory.

This functionality is a component of a normal-state multiorbital ladder calculation. The presently supported particle-hole screened approximation is distinct from a diagrammatically derived, crossing-consistent ladder DΓA self-energy. In particular, the remaining end-to-end validation requires a verified multiorbital Schwinger–Dyson contraction, a consistent diagrammatic local term and controlled high-frequency vertex and propagator asymptotics. The new projection and frequency-core functions do not mark that physics as complete.

The focused test file is `test/embedding/multiorbital-spin-projection-and-core-validation.jl`. Run it before the full package test suite and documentation build.

### Independent finite-impurity equation of motion

`orbital_impurity_equation_of_motion(result)` independently reconstructs the exact finite-bath local self-energy using the interaction current $J_a=[d_a,H_{\mathrm{int}}]$. For the package's antisymmetrized interaction convention, $J_a=\frac12\sum_{bcd}U_{abcd}d_b^\dagger d_d d_c$. Its Lehmann matrix correlator $R_{ab}(z)=\langle\langle J_a;d_b^\dagger\rangle\rangle_z$ obeys $R(z)=\Sigma(z)G(z)$, and the reconstruction evaluates $\Sigma(z)=R(z)G(z)^{-1}$ by a right matrix solve. The function returns the composite Green function, EOM self-energy, and residual against the independently computed Dyson self-energy. This provides a strong local benchmark for interaction-sign and index conventions, including complex orbital mixing; it does not compute a momentum-dependent ladder correction.

---

## Multiorbital diagrammatic equation of motion

Phundamental provides a finite-frequency, normal-state equation-of-motion contraction of a spin-orbital particle-hole full vertex, evaluated on a translation-complete uniform momentum grid. It is designed for controlled comparison of local and nonlocal diagrams, independent validation of tensor conventions, and future crossing-consistent ladder calculations. The routine does **not** infer crossing symmetry or reconstruct missing high-frequency vertex contributions.

### Definitions and physical conventions

The interaction is represented as $H_{\mathrm{int}}=\tfrac14\sum_{abcd}U_{abcd}d_a^\dagger d_b^\dagger d_d d_c$, with an antisymmetrized tensor $U$. The factor $1/2$ in the dynamic equation follows the exact fermionic interaction current $[d_m,H_{\mathrm{int}}]=\tfrac12\sum_{lhu}U_{mlhu}d_l^\dagger d_u d_h$; it must not be dropped when summing both antisymmetrized permutations. The full spin-orbital vertex has canonical pairs $(a,b)$, $(c,d)$, and values $F_{ab,cd}(\nu',\nu,\Omega)$. The particle-hole Wick bubble has the matching representation $\chi^0_{ab,cd}(\mathbf q,\nu',\Omega)=-\beta\sum_{\mathbf k}w_{\mathbf k}G_{da}(\mathbf k,\nu')G_{bc}(\mathbf k+\mathbf q,\nu'+\Omega)$.

The implemented connected equation-of-motion diagram is:

$$C_{mm'}(\mathbf k,\nu)=-\frac{1}{2\beta}\sum_{\mathbf q,\Omega,\nu'}w_{\mathbf q}\sum_{l,l',u,u',h,h'}U_{mlhu}\chi^0_{ul,l'u'}(\mathbf q,\nu',\Omega)F_{u'l',h'm'}(\mathbf q,\nu',\nu,\Omega)G_{hh'}(\mathbf k-\mathbf q,\nu+\Omega).$$

A separately evaluated local diagram uses the *same* interaction tensor, the local full vertex $F^{\mathrm{loc}}$, and the local Wick bubble and Green function constructed from the momentum-averaged one-particle propagator. The returned self-energy is $\Sigma(\mathbf k,\nu)=\Sigma_{\mathrm{DMFT}}(\nu)+C(\mathbf k,\nu)-C_{\mathrm{loc}}(\nu)$. This is a matched **diagrammatic counterterm**, distinct from subtracting $\langle C(\mathbf k,\nu)\rangle_{\mathbf k}$.

The implemented contraction is the finite-box counterpart of the multiorbital Schwinger–Dyson equation, notably the connected diagram in Eq. (61) of Galler *et al.*, *Phys. Rev. B* **95**, 115107 (2017), [DOI:10.1103/PhysRevB.95.115107](https://doi.org/10.1103/PhysRevB.95.115107). Its use as a ladder DΓA self-energy requires additional physical information: a verified crossing-consistent vertex assembled from the relevant spin channels, consistent high-frequency completion, and convergence against larger frequency boxes.

### API

- `OrbitalDiagrammaticEOMProblem(dmft, green, q_vertices, local_vertex, bosonic_axis; qmaps, qweights)` defines a converged matrix-DMFT reference and complete canonical spin-orbital full vertices at all momentum transfers.
- `orbital_diagrammatic_self_energy(problem; max_workspace_bytes, max_output_bytes)` returns an `OrbitalMomentumSelfEnergy` on the original DMFT fermionic grid.
- `dyson_momentum_green(lattice, self_energy; chemical_potential)` evaluates the matrix Dyson propagator using the resulting momentum-dependent self-energy.

Use `expanded_dmft_momentum_green(dmft, bosonic_axis)` to evaluate all shifted one-particle frequencies from the finite-bath Lehmann representation. The momentum maps must cover an entire uniform translation group; nonuniform quadrature is rejected because it changes the relationship between momentum convolution and the local counterterm. The `q_vertices` and `local_vertex` must share the exact ordered orbital-pair and Matsubara axes. Each q vertex retains both fermionic frequencies; a frequency-contracted screened susceptibility is **not** a suitable input.

The routine uses small dense BLAS matrices for interaction–bubble–vertex contraction and streams momentum transfers. It does not retain any additional momentum-resolved four-point arrays beyond the vertex objects provided by the caller. Its dynamic workspace is bounded independently of the number of q points, with memory checks performed before the large output allocation.

### Validation and limitations

The independent validation test compares the full contraction with a direct eight-orbital-index sum, tests the atomic local cancellation, checks the zero-full-vertex limit, and rejects incompatible labels, frequency support, and memory budgets. This is an algebraic validation of the chosen tensor convention; the tests do **not** establish that arbitrary supplied vertices satisfy crossing symmetry or the exact local impurity equation of motion.

Metadata deliberately records `:crossing_complete_certified_by_package => false`, `:high_frequency_completion => :none`, and `:full_multiorbital_dga => false`. The local counterterm is consistent with the finite-box EOM diagram; it does not establish control of missing high-frequency terms. End-to-end ladder DΓA acceptance requires independently tested spin-channel reconstruction, interaction and crossing prefactors, appropriate asymptotic tails, and convergence of the complete self-energy.

### SU(2) crossing-combined ladder contraction

`orbital_channel_ladder_self_energy(dmft, expanded_green, crossing_core; max_contractions, max_workspace_bytes, max_output_bytes)` evaluates the same finite-box EOM diagram using the density and magnetic spatial-orbital ladder channels with their explicit $(\mathbf q,\mathbf k,\mathbf k',\nu,\nu',\Omega)$ crossing transformation. The spin tensor is reconstructed by the Pauli identity $F_{\alpha\beta,\gamma\delta}=\tfrac12(F_d-F_m)\delta_{\alpha\beta}\delta_{\gamma\delta}+F_m\delta_{\alpha\delta}\delta_{\gamma\beta}$. All four spin-orbital interaction indices and both fermionic frequencies remain present in the dynamic contraction. This differs from a screened-interaction kernel, which integrates the two fermionic frequencies before constructing the self-energy.

The constructor checks the local and momentum-resolved Hamiltonians for SU(2) symmetry, requires the physical Matsubara core to coincide with the converged DMFT fermionic box, and rejects missing crossed-frequency support. The momentum quadrature must be a complete translation group. The EOM contraction streams $\mathbf q$, $\mathbf k'$, $\Omega$, and $\nu'$ while using fixed-size orbital matrices; `max_contractions` provides a runtime-complexity guard in addition to memory guards.

This route verifies the finite-core crossing algebra and spin contractions but still does **not** supply the missing analytic vertex tails. The returned metadata therefore retains `:full_multiorbital_dga => false` and `:high_frequency_completion => :none`. A finite-box run is not automatically a frequency-converged DΓA result, and it must not be presented as such without independent comparison to a larger frequency-box or an analytically controlled asymptotic completion.

### Frequency-box comparison

`orbital_eom_frequency_box_comparison(smaller, larger; atol, rtol)` compares two momentum-resolved self-energies on their common external frequencies, requiring the larger result to contain strictly more Matsubara points. This is an explicit convergence diagnostic, not an analytic high-frequency tail. The returned `high_frequency_asymptotics_certified` is always `false`, even if both finite boxes agree within tolerance.

---

## One-shot multiorbital ladder workflow

The one-shot orbital ladder workflow connects a converged matrix ED-DMFT reference to impurity two-particle correlations, spin-rotation-invariant density and magnetic channels, momentum-dependent Bethe–Salpeter ladders, a crossed-channel Schwinger–Dyson contraction, and a final matrix Dyson propagator. The input Hamiltonian must be normal-state and SU(2)-invariant in the declared spin-interleaved orbital ordering. This workflow is not parquet DΓA, is not self-consistent in the nonlocal self-energy, and does not infer an analytic high-frequency vertex tail.

### Data and conventions

`OrbitalLadderWorkflowProblem` requires a converged `MultiOrbitalDMFTResult`, a **support** fermionic Matsubara axis, a support bosonic Matsubara axis, and an optional smaller bosonic **core**. The external fermionic core is the original DMFT axis. For each requested core triple $(n,n',m)$, the support must contain the crossed fermionic index $n'-m$ and bosonic index $n'-n$. The support axes must also have the same inverse temperature as the DMFT reference.

The impurity two-particle evaluator retains the full pair-frequency structure $\chi_{ab,cd}(\nu,\nu',\Omega)$. The projected local density and magnetic responses are obtained from the explicitly spin-resolved impurity tensor, and the corresponding irreducible vertices are calculated with the established matrix Bethe–Salpeter convention. The lattice bubble is computed from the exact finite-bath one-particle Lehmann self-energy on the *expanded* single-particle Matsubara grid, including every required $\nu+\Omega$ index. Missing Green-function and crossing frequencies are errors, never zeros.

The nonlocal self-energy is reconstructed from the SU(2) channel-combined finite-frequency equation of motion, using the same local diagram as the double-counting counterterm. The full orbital-pair and two-fermionic-frequency structure is retained until the Schwinger–Dyson contraction. The final matrix Dyson propagator is evaluated with the corrected momentum-dependent self-energy.

### Julia example

```julia
using Phundamental

## dmft is a converged MultiOrbitalDMFTResult from an SU(2)-invariant
## spin-interleaved multiorbital Hamiltonian on a complete q-translation grid.
beta = dmft.problem.axis.beta
support_f = FermionicMatsubaraAxis(beta, -8:7)
support_b = BosonicMatsubaraAxis(beta, -6:6)
problem = OrbitalLadderWorkflowProblem(dmft, support_f, support_b; core_bosonic_indices=-2:2)
solver = OrbitalLadderWorkflowSolver(four_point_backend=TauGridFourPoint(ntime=24),
    max_retained_bytes=536870912, max_workspace_bytes=536870912)
result = solve(solver, problem)

sigma_k = result.nonlocal_self_energy
G_k = result.corrected_green
```

Increase the support until the desired core satisfies the exact crossing condition. The example assumes the original DMFT fermionic axis is contained in the support and that the finite impurity Hilbert space fits the four-point backend. The `max_contractions` guard rejects computationally impractical full orbital contractions before execution. Large runs may require a deliberately increased budget and a smaller physical frequency core.

The two optional `two_particle` and `momentum_green` inputs permit reuse of independently calculated, matching intermediate results. The workflow checks axes, labels, symmetry, and shifted-frequency coverage before using them. The two-particle evaluation uses finite midpoint imaginary-time quadrature; vary `ntime` and check its convergence independently of the Matsubara box.

### Convergence and limitations

For separate frequency-support and imaginary-time-quadrature diagnostics, use the [frequency bounds and convergence workflow](dga.md). A pair of runs on nested support grids and the same physical core may also be compared using `orbital_ladder_support_convergence(smaller, larger)`. Its returned numerical residual tests the change in the computed self-energy *on the same external grid*. This is a finite-box convergence diagnostic, not a rigorous upper bound on the omitted high-frequency contributions. Consequently, `result.metadata[:high_frequency_completion]` is `:not_certified` and `result.metadata[:full_multiorbital_dga]` remains `false`.

The complete analytic high-frequency vertex asymptotics require equal-time impurity correlation kernels beyond the one-particle Lehmann information. They have not been reconstructed or certified by this workflow. Until such asymptotics are implemented and independently validated, it would be incorrect to describe the finite-box numerical convergence test as proof of a fully frequency-completed DΓA calculation. Published asymptotic methods are detailed in Kaufmann, Gunacker, and Held, *Physical Review B* **96**, 035114 (2017), DOI: [10.1103/PhysRevB.96.035114](https://doi.org/10.1103/PhysRevB.96.035114).

The explicit high-level forward context is `OrbitalLadderWorkflowForwardContext(model, problem)`, paired with `OrbitalLadderWorkflowSolver` and `MomentumMatsubaraSelfEnergyRequest`. This context does not claim to reconstruct the supplied lattice and interaction automatically from a symbolic `ManyBodyModel`.

### Testing and profiling

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-ladder-workflow-validation.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/multiorbital-ladder-workflow.jl
julia --project=docs docs/make.jl
```

The benchmark reports wall-clock time and cumulative allocations for the complete pipeline. Allocated bytes are not peak resident memory. The complete package regression suite should be run at the final electronic-workflow acceptance gate.

---

## Finite-Bath Frequency Bounds and Ladder Convergence

The finite-temperature multiorbital solver provides two distinct frequency-control mechanisms. The first supplies rigorous **one-particle** high-frequency error bounds for an exact finite-bath Lehmann Green function. The second measures **two-particle** frequency-support and imaginary-time-quadrature sensitivity through four independent one-shot ladder calculations. A one-particle bound does not certify a multiorbital two-particle vertex asymptotic tail.

### Rigorous one-particle finite-bath expansion

For a number-conserving finite-bath ED solution, the impurity Green function admits the exact spectral representation $G_{ab}(z)=\sum_j (A_j)_{ab}/(z+\delta_j)$, where each residue matrix $A_j$ is positive semidefinite and each excitation energy $\delta_j$ is real. The moment expansion is $G_p(z)=\sum_{k=0}^{p-1}M_k/z^{k+1}$ with $M_k=\sum_j(-\delta_j)^kA_j$. These coefficients come directly from the ED eigenvectors and thermal probabilities, without fitting.

Let $R=\max_j|\delta_j|$. For $|z|>R$, the truncated one-particle expansion obeys the operator-norm bound

```math
\|G(z)-G_p(z)\|_2 \leq \|M_0\|_2 \frac{(R/|z|)^p}{|z|-R}.
```

The bound follows from the exact scalar geometric-series remainder and the positivity of the spectral residues: their sum is $M_0$. For canonical impurity fermions, $M_0=I$ up to floating-point error. The evaluator rejects frequencies with $|z|\leq R$ rather than extrapolating a bound beyond its domain.

```julia
using Phundamental

## state is the validated MultiOrbitalEDThermalState from the impurity solve.
moments = orbital_impurity_lehmann_moments(state; order=4)
high_axis = FermionicMatsubaraAxis(state.beta, [30, 50, 70])
tail = orbital_impurity_lehmann_tail(moments, high_axis)
G_approx = tail.green
guaranteed_operator_norm_errors = tail.error_bounds
```

The error bound is rigorous for the **finite-bath, one-particle propagator** only. It does not bound bath-discretization error, a lattice self-energy, the frequency-truncated two-particle vertex, or a DΓA correction.

### Independent ladder convergence dimensions

The `orbital_ladder_convergence_study` function evaluates the full one-shot ladder calculation at four points in a rectangular numerical-resolution grid: small and large Matsubara support, each with coarse and fine imaginary-time quadrature. It compares the two supports at fine quadrature and the two quadratures at large support. A difference-of-differences diagnostic measures the interaction between the two truncations. Keeping the external physical Matsubara core identical across all runs makes these comparisons meaningful.

```julia
using Phundamental

## dmft is a converged MultiOrbitalDMFTResult with complete momentum-shift maps.
beta = dmft.problem.axis.beta
small = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(beta, -2:1),
    BosonicMatsubaraAxis(beta, -1:1); core_bosonic_indices=[0])
large = OrbitalLadderWorkflowProblem(dmft, FermionicMatsubaraAxis(beta, -3:2),
    BosonicMatsubaraAxis(beta, -2:2); core_bosonic_indices=[0])
solver = OrbitalLadderWorkflowSolver(pseudoinverse=true)
report = orbital_ladder_convergence_study(solver, small, large;
    ntime_coarse=12, ntime_fine=16)

support_error = report.frequency_comparison
quadrature_error = report.quadrature_comparison
final_sigma = report.final_result.nonlocal_self_energy
```

The workflow checks strict support nesting, shared DMFT reference, matching physical core, and per-run memory/contraction budgets before evaluation. It also rejects midpoint imaginary-time grids whose sampled single-particle Fourier frequencies reach or exceed the discrete Nyquist limit. All four runs are independently evaluated; previously computed two-particle data cannot be reused across different `ntime` values. The report retains only the final full ladder result and compact copies of the other self-energy arrays, preventing four complete vertex tensors from remaining live.

Each stage reports execution time and cumulative allocations. The example benchmark warms up the full workflow before measuring the four stages so first-call compilation does not dominate its reported timing. Its noninteracting, single-momentum-point atomic fixture is only a smoke test: zero support and quadrature residuals are expected and do not establish convergence in interacting or momentum-dependent systems. These values are diagnostic and are not a measured peak resident-memory bound. A numerical plateau in either axis is evidence of finite-resolution stability, not a mathematical bound on the omitted infinite-frequency vertex. Accordingly, `report.metadata[:two_particle_vertex_tail_certified]` and `report.metadata[:full_multiorbital_dga]` are both `false`.

### What a complete vertex tail still requires

The asymptotic multiorbital four-point vertex contains one-frequency and two-frequency kernels related to equal-time two-particle and three-leg correlation functions, in addition to the bare interaction. The spectral-moment expansion above provides none of those missing kernels. An analytically completed vertex and an error estimate for the omitted remainder remain separate physics requirements; the software does not convert frequency-box agreement into a certification flag. See Kaufmann, Gunacker, and Held, *Physical Review B* **96**, 035114 (2017), [DOI: 10.1103/PhysRevB.96.035114](https://doi.org/10.1103/PhysRevB.96.035114), for the multiorbital vertex asymptotic structure.

### Validation and performance

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. test/embedding/multiorbital-frequency-audit-validation.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=. benchmarks/multiorbital-frequency-audit.jl
julia --project=docs docs/make.jl
```

### Exact equal-time vertex-asymptotic inputs

The finite-bath solver additionally provides an [exact orbital-bilinear response and bounded bosonic high-frequency expansion](impurity.md). These are physical equal-time inputs for the vertex asymptotic construction. They do not, by themselves, certify the full three-frequency vertex tail.

---
