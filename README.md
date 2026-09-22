# Phundamental.jl

`Phundamental.jl` is a representation-aware framework for constructing, transforming, solving, and analyzing quantum many-body models in Julia. Its central design goal is to keep the **physical model**, **mathematical representation**, **numerical approximation**, **solver backend**, **observable definition**, and **experimental response** conceptually separate.

The framework is intended to support calculations ranging from exact small-cluster benchmarks to thermally sampled clusters and specialized large-lattice mode calculations, while preserving enough symbolic information to validate transformations and compare different representations of the same physics.

---

# 1. Abstract top-down view

At the highest level, a calculation in `Phundamental` is organized as

$$
\text{material/geometry}
\longrightarrow
\text{Hamiltonian model}
\longrightarrow
\text{representation}
\longrightarrow
\text{optional transformation}
\longrightarrow
\text{solver}
\longrightarrow
\text{observable}
\longrightarrow
\text{probe response}
\longrightarrow
\text{instrument response}.
$$

The main model object is

\[\mathfrak M = \left(\mathfrak R,\,\hat H,\,\boldsymbol{\theta},\,\mathcal O,\,\text{provenance}\right),\]

represented by `ManyBodyModel`. Here:

- $\mathfrak{R}$ is the mathematical representation;
- $\hat{H}$ is the Hamiltonian;
- $\boldsymbol{\theta}$ is stored model/material metadata and parameter information;
- $\mathcal{O}$ is the registry of physical observables;
- provenance records how the model was constructed or transformed.

A `Representation` stores the ambient state space, physical subspace, operator algebra, basis, constraints, gauge structure, reference state, ordering, and any numerical truncation:

$$\mathfrak{R} = \left(\mathcal{S},\,\mathcal{S}_{\mathrm{phys}},\,\mathcal{A},\,\mathcal{B},\,\mathcal{C},\,\mathcal{G},\,|\Omega\rangle,\,\prec,\,\mathcal{T}_{\mathrm{num}}\right).$$

A central rule throughout the package is

$$\boxed{\text{ambient state space}\neq\text{physical subspace}\neq\text{numerical truncation}.}$$

Physical constraints are part of the mathematical definition of a representation. Numerical truncations are computational choices and are never silently reinterpreted as physical constraints.

## 1.1 Physical transformations, numerical backends, and approximations

`Phundamental` distinguishes three operations that are often conflated.

A **representation transformation** changes the algebraic description of the model. For example,

$$\text{spin-}\frac{1}{2} \xrightarrow{\text{Jordan--Wigner}} \text{spinless fermions}.$$

A **numerical backend** changes only how the same representation is encoded computationally. For example,

$$\text{spin representation}\longrightarrow\text{packed-bit spin backend}$$

does not constitute a Jordan--Wigner transformation and does not change the physical representation.

An **approximation** changes the model being solved. For example,

$$\hat{H}_{\mathrm{spin}} \xrightarrow{\mathrm{HP}} \hat{H}_{\mathrm{HP}} \xrightarrow{\text{quadratic truncation}} \hat{H}_{\mathrm{LSWT}}$$

is linear spin-wave theory and is explicitly marked as approximate.

This separation allows exact and approximate calculations to use a common high-level API without erasing their mathematical meaning.

## 1.2 Forward calculation API

The primary orchestration function is

```julia
result = forward(
    model;
    transformation=nothing,
    solver=ExactDiagonalization(),
    observable=nothing,
    probe=nothing,
    resolution=nothing,
)
```

The result is a `ForwardResult` containing each stage independently:

```text
source_model
model
transformation_result
solution
observable_result
scattering_result
result
```

This preserves the distinction

$$\text{intrinsic spectrum}\rightarrow\text{probe cross section}\rightarrow\text{instrument-convolved result}.$$

For example,

```julia
result = forward(
    model;
    transformation=JordanWigner(1:N),
    solver=OptimizedLanczos(),
    observable=obs,
    probe=MagneticNeutronProbe(),
    resolution=GaussianResolution(0.08),
)
```

performs the complete representation-aware calculation without requiring the observable or scattering layers to know how the Hamiltonian was solved.

A model may also be constructed directly inside `forward`:

```julia
result = forward(
    material,
    cluster,
    XXZModel(1.0, 0.6; neighbor_cutoff=1.01);
    solver=OptimizedLanczos(),
    observable=obs,
)
```

## 1.3 Module organization

The current implementation is organized as

```text
Phundamental.jl
├── Core/
├── Algebra/
├── Transformations/
├── Models/
├── Observables/
├── Solvers/
│   └── optimized/
├── Thermodynamics/
├── Greens/
└── Embedding/
```

The top-level namespaces are

```julia
Phundamental.Foundation
Phundamental.Algebra
Phundamental.Crystallography
Phundamental.Transformations
Phundamental.Models
Phundamental.Observables
Phundamental.Solvers
Phundamental.Thermodynamics
Phundamental.Greens
Phundamental.Embedding
```

`Foundation` is an alias of the internal `PhundamentalCore` module. This avoids collision with Julia's built-in `Core` module.

`Greens` provides finite-temperature Matsubara Green functions, Nambu indexing, Dyson relations, self-energies, high-frequency diagnostics, and symmetry/causality checks. `Embedding` provides finite Anderson baths, ED impurity solvers, lattice/Weiss self-consistency, and the first normal-state single-site ED-DMFT driver. Scientific validation and published-reference benchmarks for this layer are under `test/greens/`, `test/embedding/`, and `benchmarks/dmft/`.

During the current development layout, the package can be loaded with

```julia
include("Phundamental.jl")
using .Phundamental
```

The complete public `PhundamentalCore` and `Solvers` APIs are re-exported at the package root, so ordinary symbolic-model and numerical-solver workflows require only `using Phundamental`; the namespace modules remain available for explicit qualification and organization.

---

# 2. Symbolic model transformations, algebra, and representations

The symbolic layer defines the mathematical objects from which models and representation changes are built.

## 2.1 State spaces

The fundamental state-space types include

```julia
SpinHilbertSpace
FermionFockSpace
BosonFockSpace
CoordinateHilbertSpace
CompositeStateSpace
BiorthogonalSpace
PhysicalSubspace
NumericalTruncation
```

For $N$ local degrees of freedom,

$$\mathcal{H}=\bigotimes_{i=1}^{N}\mathcal{H}_{i}.$$

Examples include the spin product basis

$$|\mathbf{m}\rangle_{S} = |m_{1},m_{2},\ldots,m_{N}\rangle,$$

the fermionic occupation basis

$$|\mathbf{n}\rangle_{f} = |n_{1},n_{2},\ldots,n_{M}\rangle,\qquad n_{i}\in\{0,1\},$$

and a bosonic occupation basis

$$|\mathbf{n}\rangle_{b},\qquad n_{i}\in\mathbb{N}_{0}.$$

A finite bosonic cutoff is represented by `NumericalTruncation`; it is not treated as a physical constraint.

## 2.2 Operator algebras

The package currently supports

```julia
SpinAlgebra
BosonAlgebra
FermionAlgebra
MajoranaAlgebra
CoordinateMomentumAlgebra
CompositeAlgebra
```

Primitive symbolic operators include

```julia
Sx(i), Sy(i), Sz(i), Sp(i), Sm(i)

b(i), b(i)', nb(i)

c(i), c(i)', nf(i)

γ(i)

xop(i), pop(i)
```

Bosonic and fermionic creation operators use Julia adjoint syntax canonically: `b(i)'` and `c(i)'`. The former `bd(i)` and `cd(i)` constructors remain internal compatibility aliases but are no longer exported.

Along with these constructors, the operator layer provides `IdentityOperator`, `NumberOperator`, sums, products, and scaled expressions.

The operator tree is explicitly noncommutative. The algebra layer provides

```julia
commutator
anticommutator
rewrite_once
normal_order
canonicalize
operator_family
```

so that symbolic expressions can be transformed and then canonicalized in the target algebra.

## 2.3 Representations

A representation is not just an operator basis. `Representation` tracks

```text
name
state_space
physical_subspace
algebra
basis
constraints
gauge_structure
reference_state
ordering
truncation
```

This is important for transformations that enlarge the ambient space.

For example, the exact Holstein--Primakoff map sends a spin-$S_{i}$ degree of freedom into a constrained bosonic Fock space:

$$n_{i}=S_{i}-m_{i},\qquad0\leq n_{i}\leq 2S_{i},$$

with

$$S_{i}^{z} =\hbar(S_{i}-n_{i}),$$

$$S_{i}^{+} =\hbar\sqrt{2S_{i}-n_{i}}\,b_{i},$$

$$S_{i}^{-} = \hbar b_{i}^\dagger\sqrt{2S_{i}-n_{i}}.$$

The unconstrained bosonic Fock space is larger than the original spin Hilbert space, so the physical subspace must remain explicit.

## 2.4 Transformation interface

All transformations derive from

```julia
AbstractTransformation
```

and implement the common interface

```julia
certificate(T)
applicable(T, model)
target_representation(T, model)
transform_generator(T, op)
transform_operator(T, op)
transform_hamiltonian(T, H)
transform_observable(T, O)
transform(T, model)
```

A transformation acts on both the Hamiltonian and registered physical observables. Thus an observable defined in the source representation remains physically consistent after the model is transformed.

The main call is

```julia
mapped = transform(T, model)
```

which returns a `TransformationResult` containing

```text
model
transformation
certificate
validation
```

The source model is validated before transformation, the target representation is constructed, the Hamiltonian and observables are mapped and canonicalized, and provenance is retained.

Transformations may also be composed:

```julia
T = compose(T1, T2, T3)
mapped = transform(T, model)
```

## 2.5 Transformation certificates

Each known transformation carries a `TransformationCertificate` containing

```text
exact
invertible
relation
locality
requires_constraints
gauge_redundant
approximate
notes
```

Relation types include

```julia
UnitaryEquivalence
CanonicalMap
AlgebraIsomorphism
ConstrainedEmbedding
SimilarityEquivalence
GaugeRedundantEmbedding
CompositeRelation
```

This allows software to distinguish, for example, an exact algebra isomorphism from a constrained embedding or a gauge-redundant representation.

## 2.6 Available transformations

The current transformation layer includes

```julia
FourierTransformation
BogoliubovTransformation
ParticleHoleTransformation
MajoranaTransformation
JordanWigner
HolsteinPrimakoff
DysonMaleev
SchwingerBoson
AbrikosovFermion
SlaveParticleTransformation
CoordinateLadder
BosonicDisplacement
LangFirsov
SpinPolaron
```

### Jordan--Wigner

For the convention used by the package,

$$n_{j}=m_{j}+\frac{1}{2},$$

and

$$\mathcal{P}_{j} = \prod_{\ell<j}(1-2n_{\ell}).$$

Then

$$S_{j}^{+} = \hbar c_{j}^{\dagger}\mathcal{P}_{j}, \qquad S_{j}^{-} = \hbar\mathcal{P}_{j} c_{j},$$

$$S_{j}^{z} = \hbar\left(n_{j}-\frac{1}{2}\right).$$

This is an exact nonlocal algebra mapping. For periodic one-dimensional systems, the boundary term depends on total fermion parity.

### Dyson--Maleev

The Dyson--Maleev representation replaces the square-root structure of exact HP operators with finite polynomials. It is related to HP by a nonunitary similarity transformation on the physical subspace. The representation is therefore biorthogonal rather than conventionally Hermitian.

### Schwinger bosons and Abrikosov fermions

These transformations enlarge the ambient state space and impose exact local constraints. For example, Abrikosov fermions represent a spin-$\frac{1}{2}$ operator as

$$S_{i}^{\alpha} = \frac{\hbar}{2}f_{i}^{\dagger}\sigma^{\alpha} f_{i},$$

with the single-occupancy constraint

$$n_{i\uparrow}+n_{i\downarrow}=1.$$

### Coordinate/ladder operators

The coordinate--ladder transformation implements

$$x_{i} = \sqrt{\frac{\hbar}{2M_{i}\omega_{i}}}(b_{i}+b_{i}^{\dagger}),$$

$$p_i = -i\sqrt{\frac{\hbar M_{i}\omega_{i}}{2}}(b_{i}-b_{i}^{\dagger}).$$

### Lang--Firsov and spin-polaron transformations

The Lang--Firsov transformation reorganizes an electron--phonon model into dressed electronic and displaced phonon variables. The spin-polaron transformation performs the analogous displacement conditioned on spin operators.

Approximations are not introduced merely by performing these exact transformations. They arise only when the transformed Hamiltonian is subsequently truncated, expanded, averaged, or otherwise approximated.

## 2.7 Numerical closure after transformation

Exact symbolic transformation and finite numerical realization are separate capabilities. Use `transformation_capabilities(T, model)` before a representation change when the target may contain unbounded bosons, transported physical constraints, or a biorthogonal state space.

```julia
caps = transformation_capabilities(T, model)
```

For an already transformed model, `numerical_capabilities(model)` reports whether the Hamiltonian primitives have a matrix realization, whether the target basis is finite and constructible, whether a target-basis truncation is required, whether exact transported constraints still require a target projector, and which generic solvers are compatible.

```julia
caps = numerical_capabilities(mapped.model)
```

A numerical cutoff belongs to a particular basis. Basis-changing transformations therefore retain a source cutoff only as provenance and never reinterpret it silently as the same cutoff in the target basis. Select the target cutoff explicitly with

```julia
numerical_model = with_truncation(mapped.model, NumericalTruncation(12))
```

`with_truncation` changes only computational metadata; it does not alter the Hamiltonian or convert a numerical cutoff into a physical constraint. Exact transported physical constraints that cannot yet be represented by a target-basis projector are reported as unresolved and matrix realization is rejected rather than broadening the physical Hilbert space silently.

Dyson--Maleev targets are marked biorthogonal. `ExactDiagonalization()` automatically uses its non-Hermitian left/right eigensystem for that representation, forcing `hermitian=:yes` is rejected, and Hermitian Lanczos solvers reject the target explicitly. Exact spectral time evolution inherits the same biorthogonal eigensystem.

---

# 3. Models: materials, geometry, and Hamiltonians

The `Models` layer separates physical material/crystal information from a chosen Hamiltonian.

The intended relation is

\[\boxed{\text{material}\neq\text{Hamiltonian}.}\]

The same material and geometry may therefore be paired with several candidate models.

## 3.1 Lattices and reciprocal space

A Bravais lattice is defined by

```julia
BravaisLattice(vectors)
```

for spatial dimensions \(D=1,2,3\). Direct vectors are stored as columns of the direct-lattice matrix \(A\), while reciprocal vectors satisfy

\[A^TB=2\pi I.\]

The geometry layer provides

```julia
direct_matrix
reciprocal_matrix
cell_measure
fractional_to_cartesian
cartesian_to_fractional
reciprocal_vector
reciprocal_coordinates
```

A crystal is built from a Bravais lattice and basis sites:

```julia
crystal = M.CrystalStructure(
    M.BravaisLattice(reshape([1.0], 1, 1)),
    [M.BasisSite(:A, :X, [0.0])],
)
```

A finite calculation uses a `Supercell`:

```julia
cluster = Supercell(
    crystal,
    [8];
    periodic=false,
)
```

Each `CrystalSite` retains its global index, basis-site index, cell coordinate, species, fractional coordinate, and Cartesian coordinate.

## 3.2 Neighbor lists and bonds

Interactions are represented using `Bond` objects generated by

```julia
neighbor_bonds
bonds_in_shell
coordination_numbers
```

Hamiltonian couplings may be supplied as

- a scalar;
- a vector indexed by shell;
- a dictionary;
- a function of the bond.

This allows uniform couplings, shell-dependent exchange, or geometry-dependent custom interactions without changing the model constructor.

## 3.3 Material information

`Material` combines the crystal with a species database and basis-site physical properties.

Species are defined with

```julia
AtomicSpecies(
    :Fe;
    mass=...,
    nuclear_scattering_length=...,
)
```

Site properties may include

```text
spin
g_tensor
magnetic_form_factor
Debye-Waller tensor
arbitrary metadata
```

using

```julia
SiteProperties(...)
```

Magnetic form factors currently include

```julia
ConstantFormFactor
GaussianFormFactor
```

and a Debye--Waller amplitude is represented as

\[e^{-\frac12\mathbf Q^TU\mathbf Q}.\]

No tabulated atomic or magnetic data are hard-coded into the framework; material-specific coefficients can be supplied externally.

## 3.4 Building a model

The common constructor is

```julia
model = build_model(material, cluster, spec)
```

which returns a `ManyBodyModel`.

Construction records the material, cluster, positions, Hamiltonian specification, bond information when applicable, physical observables, and provenance.

For example,

```julia
model = build_model(
    material,
    cluster,
    XXZModel(
        1.0,
        0.6;
        field_z=0.2,
        neighbor_cutoff=1.01,
    ),
)
```

produces a spin representation and automatically registers local

```text
spin_x_i
spin_y_i
spin_z_i
```

observables.

## 3.5 Spin models

### Heisenberg model

The standard isotropic model is

\[H = \sum_{\langle ij\rangle}J_{ij}\frac{\mathbf S_i\cdot\mathbf S_j}{\hbar^2} - \sum_i \frac{\mathbf h\cdot\mathbf S_i}{\hbar}.\]

It is created using

```julia
HeisenbergModel(
    J;
    field=(hx, hy, hz),
    neighbor_cutoff=...,
    max_shell=...,
    hbar=1.0,
)
```

### XXZ model

The anisotropic model is

\[H =\sum_{\langle ij\rangle}\left[J_{ij}^{xy}\frac{S_i^xS_j^x+S_i^yS_j^y}{\hbar^2} + J_{ij}^{z}\frac{S_i^zS_j^z}{\hbar^2}\right]-\sum_ih_i^z\frac{S_i^z}{\hbar}.\]

It is created using

```julia
XXZModel(
    Jxy,
    Jz;
    field_z=0.0,
    neighbor_cutoff=...,
    max_shell=...,
    hbar=1.0,
)
```

## 3.6 Fermion model

The current standard fermionic model is the spinful single-band Hubbard model,

\[H=-\sum_{\langle ij\rangle,\sigma}t_{ij}\left(c_{i\sigma}^\dagger c_{j\sigma}+c_{j\sigma}^\dagger c_{i\sigma}\right)+\sum_i U_i n_{i\uparrow}n_{i\downarrow}-\sum_{i,\sigma}\mu_i n_{i\sigma}.\]

It is created with

```julia
HubbardModel(
    t,
    U;
    chemical_potential=0.0,
    neighbor_cutoff=...,
    max_shell=...,
)
```

The model registers local density, double occupancy, and spin observables.

## 3.7 Phonon models

### Harmonic coordinate-space phonons

`HarmonicPhononModel` represents

\[H=\frac12p^TM^{-1}p+\frac12x^T\Phi x.\]

The model retains masses and force constants explicitly, allowing the optimized solver layer to bypass Fock-space construction and solve the corresponding dynamical-matrix problem directly.

### Local phonons

`LocalPhononModel` represents independent bosonic modes,

\[H=\sum_i\hbar\omega_i\left(b_i^\dagger b_i+\frac{1}{2}\right),\]

with an explicit numerical occupation cutoff.

## 3.8 Hybrid models

The current hybrid constructors include

```julia
HolsteinModel
LongitudinalSpinBosonModel
```

which create composite fermion--boson or spin--boson representations.

`HolsteinModel` combines Hubbard-like electronic terms with local phonons and electron--phonon coupling.

`LongitudinalSpinBosonModel` combines an XXZ-like spin Hamiltonian with local bosons and longitudinal spin--boson coupling.

Because the representation, physical constraints, and numerical truncations remain explicit, hybrid models can subsequently be transformed using operations such as `LangFirsov` or `SpinPolaron`.

---

# 4. Observables, correlations, scattering, and resolution

The `Observables` layer converts solver output into physical correlation functions, spectra, and probe-specific responses.

The basic hierarchy is

\[\text{Hamiltonian solution}\rightarrow\text{correlation function}\rightarrow\text{intrinsic spectrum}\rightarrow\text{probe intensity}\rightarrow\text{instrument response}.\]

## 4.1 Observable registry

A `ManyBodyModel` contains a registry of named observables. They may be referenced symbolically using `ObservableRef` or directly as operator expressions.

Functions include

```julia
register_observable
resolve_observable
register_spin_tensor
```

The observable registry is transformed together with the Hamiltonian whenever an exact representation transformation is applied.

This means that if a spin observable is registered before a Jordan--Wigner transformation, the transformed model contains the corresponding fermionic operator rather than losing the physical definition of the measurement.

## 4.2 Momentum-space observables

`MomentumField` and `SpinTensorField` define how local operators are assembled in momentum space.

For a field \(O_i\),

\[O_{\mathbf q}=a_N\sum_ie^{-i\mathbf q\cdot\mathbf r_i}O_i,\]

where the normalization may be selected through the field definition.

The helper

```julia
momentum_operator(model, field, q)
```

constructs the corresponding symbolic operator.

## 4.3 Correlation functions

For operators \(A\) and \(B\),

\[C_{AB}(t)=\langle A(t)B(0)\rangle.\]

The package provides

```julia
correlation_function
eigenbasis_matrix
thermal_probabilities
```

and consistently supports ordinary Hermitian eigenbases and left/right biorthogonal results.

## 4.4 Dynamic structure factor

The Lehmann representation is used for exact eigenstate-based spectra:

\[S_{AB}(\mathbf q,\omega)=\frac{1}{Z}\sum_{mn}e^{-\beta E_m}\langle m|A_{\mathbf q}|n\rangle\langle n|B_{-\mathbf q}|m\rangle\delta\!\left(\omega-\frac{E_n-E_m}{\hbar}\right).\]

The API includes

```julia
lehmann_lines
broaden_lines
dynamic_structure_factor
spin_tensor_structure_factor
```

Intrinsic numerical broadening is represented independently using

```julia
LorentzianBroadening
GaussianBroadening
```

so it is not confused with an instrument-resolution function.

The complete spin tensor

\[S^{\alpha\beta}(\mathbf Q,\omega)\]

is retained rather than reducing the calculation prematurely to a scalar spectrum.

## 4.5 Static structure factor and spectral density

The module also provides

```julia
static_structure_factor
spectral_density
detailed_balance_residual
integrated_spectral_weight
dynamic_static_sumrule_residual
```

for static correlations, general spectral functions, and validation of sum rules or detailed balance.

## 4.6 Magnetic neutron scattering

`MagneticNeutronProbe` contracts the spin tensor with the magnetic polarization projector

\[P_{\alpha\beta}=\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta.\]

Schematically,

\[I(\mathbf Q,\omega)\propto|F(\mathbf Q)|^2\sum_{\alpha\beta}P_{\alpha\beta}\left(gSg^T\right)_{\alpha\beta}.\]

The corresponding API is

```julia
magnetic_neutron_intensity
scattering_intensity
```

Material-specific magnetic form factors and \(g\)-tensors may be included through the model or probe.

## 4.7 Coherent one-phonon neutron scattering in extended reciprocal space

The high-level one-phonon API accepts the full physical scattering vector $\mathbf Q$ and internally constructs the canonical decomposition $\mathbf Q=\mathbf q+\mathbf G$, where $\mathbf q$ is the first-Brillouin-zone momentum transfer and $\mathbf G$ is a reciprocal-lattice translation. Under the harmonic solver's atomic-position dynamical-matrix convention, the form-factor eigenvector is evaluated at $\mathbf q'=-\mathbf q$.

```julia
Q = ReciprocalWaveVector([0.0, 0.0, 2.37])
spectrum = one_phonon_neutron_intensity(
    HarmonicPhononSolver(), model, NuclearNeutronProbe(), Q, energy_axis;
    temperature=5.0, broadening=GaussianBroadening(0.2),
)
```

The reciprocal reduction uses the full reciprocal-lattice metric rather than componentwise wrapping, so the same API applies to nonorthogonal lattices. The harmonic dynamical matrix uses the atomic-position phase convention; consequently the high-level neutron API evaluates the mode eigenvector at $\mathbf q'=-\mathbf q$ and the coherent basis phase at $\mathbf G=\mathbf Q+\mathbf q'$, while the polarization projection and Debye-Waller factor retain the full physical $\mathbf Q$. `decompose_reciprocal_vector(model, Q)` exposes the retained $\mathbf Q$, reduced $\mathbf q$, reciprocal translation $\mathbf G$, integer reciprocal indices, convention, and reconstruction residual when explicit inspection is needed.

The existing result-based overloads remain available when the caller intentionally manages the reduced phonon momentum. The exact acoustic $\Gamma$ modes retain the normal zero-mode policy and are not converted into finite-energy inelastic lines.

## 4.8 Instrument resolution

Instrument broadening is applied only after a theoretical scattering spectrum is obtained:

\[I_{\mathrm{obs}}=R*I_{\mathrm{theory}}.\]

The current implementation provides

```julia
GaussianResolution
convolve_resolution
```

This separation is deliberate:

```text
intrinsic lifetime/numerical broadening
!=
experimental resolution convolution
```

## 4.9 Example

A complete spin-scattering calculation can be expressed as

```julia
obs = (model, solution) -> spin_tensor_structure_factor(
    model,
    solution,
    spinfield,
    qgrid,
    energy_grid;
    temperature=5.0,
    broadening=LorentzianBroadening(0.03),
)

result = forward(
    model;
    solver=OptimizedLanczos(),
    observable=obs,
    probe=MagneticNeutronProbe(),
    resolution=GaussianResolution(0.08),
)
```

---

# 5. Solvers: generic reference methods and optimized backends

The solver layer has two complementary purposes.

The generic implementation provides a representation-independent reference backend for correctness and validation.

The optimized implementation recognizes particular numerical structures and avoids generic operator-tree overhead or, where mathematically possible, avoids construction of the exponentially large many-body basis entirely.

## 5.1 Computational bases

The generic solver constructs `ComputationalBasis` objects for spin, fermion, boson, composite, constrained, and biorthogonal representations.

A basis is constructed from the model's `Representation`, including its physical constraints and numerical truncation.

Important functions include

```julia
computational_basis
basis_dimension
basis_state
```

## 5.2 Matrix realization

A symbolic operator can be converted to a sparse or dense matrix using

```julia
realize
matrix
hamiltonian_matrix
```

Products act in the ambient algebra before projection onto the retained basis.

This preserves the important rule

\[\boxed{PABP\neqPAPBP}\]

in general.

Therefore, for constrained or truncated representations, the implementation does not project after each primitive operator. The complete product acts first, and the final state is then tested against the physical subspace and numerical truncation.

Realization diagnostics distinguish

```text
physical_leakage
truncation_leakage
```

rather than combining them.

## 5.3 Exact diagonalization

`ExactDiagonalization()` constructs the Hamiltonian matrix and solves the complete eigensystem.

It is the primary reference method for small systems and supplies the complete spectrum required by `ExactGibbs`.

For Hilbert-space dimension \(D\), dense eigendecomposition has unfavorable asymptotic scaling and should be regarded as a small-cluster method.

## 5.4 Generic Lanczos

`LanczosSolver` provides a Krylov method for low-energy eigenstates and supports matrix-free symbolic Hamiltonian action through `SymbolicLinearOperator`.

This avoids storing a dense \(D\times D\) Hamiltonian, but the generic matrix-free action still traverses the explicit computational basis and symbolic expression tree.

It is therefore a general reference Krylov backend rather than the most aggressive high-performance implementation.

## 5.5 Time evolution

`TimeEvolutionSolver` and

```julia
evolve
```

provide time-dependent state propagation using the solver layer.

The optimized Krylov tools also expose

```julia
exp_action(A, state, tau)
```

for matrix-exponential action without explicitly constructing the exponential.

## 5.6 Optimized packed backends

The optimized layer is located at

```text
Solvers/optimized/
```

and is designed to preserve the same external `solve(...)` interface.

### Packed spin backend

For spin-\(\frac12\), a product state can be encoded in machine bits rather than heap-allocated state objects.

This turns operations such as

\[S_i^z,\qquadS_i^\pm\]

into bit tests and flips.

The interface is

```julia
result = solve(
    OptimizedLanczos(
        nev=8,
        sector=SpinSector(total_sz=0),
    ),
    model,
)
```

or

```julia
SpinSector(nup=...)
```

when an exact fixed-\(S^z_{\mathrm{tot}}\) sector is known.

### Packed fermion backend

Fermion occupations are likewise represented as bit strings. Creation and annihilation use parity from the number of occupied lower-index modes.

Fixed-number sectors include

```julia
FermionSector(nparticles=N)
```

and, for spinful models,

```julia
FermionSector(
    nup=Nup,
    ndown=Ndown,
)
```

This can reduce a Hubbard calculation from

\[4^N\]

states to

\[\binom{N}{N_\uparrow}\binom{N}{N_\downarrow}.\]

### Packed boson backend

Finite truncated boson bases use a mixed-radix numerical encoding.

Unlike the spin and fermion encodings, a bosonic occupation cutoff is a numerical approximation/truncation, so solver metadata records this explicitly.

## 5.7 Optimized Lanczos

The common exact packed interface is

```julia
OptimizedLanczos(
    nev=6,
    krylov_dim=64,
    tol=1e-10,
    sector=nothing,
    backend=:auto,
    reorthogonalize=true,
    seed=0,
    fallback=true,
)
```

The physical representation remains unchanged:

\[\text{physical representation}\neq\text{packed numerical encoding}.\]

Unsupported structures may fall back to the generic symbolic Lanczos implementation when `fallback=true`.

## 5.8 Quadratic fermion solver

For a quadratic fermionic Hamiltonian,

\[H=c^\dagger A c+\frac12\left(c^\dagger\Delta c^\dagger+\mathrm{h.c.}\right)+E_0,\]

`QuadraticFermionSolver` extracts the normal and anomalous blocks.

For number-conserving models, it diagonalizes the single-particle matrix \(A\). For paired systems, it solves the fermionic BdG problem.

The important scaling change is that the calculation acts in mode space rather than the full fermionic Fock space.

A system with \(M\) fermion modes is treated through matrices with dimension proportional to \(M\), rather than through a Hilbert space of dimension \(2^M\).

The result is a `QuadraticModeResult`, not a full many-body `SolverResult`.

## 5.9 Quadratic boson solver

`QuadraticBosonSolver` recognizes Hamiltonians of the form

\[H=b^\dagger A b+\frac12\left(b^\dagger B b^\dagger+\mathrm{h.c.}\right)+E_0.\]

It performs a bosonic BdG/symplectic mode calculation and checks stability through the reality and sign of the mode frequencies.

The result again lives in mode space and does not construct the complete bosonic Fock basis.

## 5.10 Harmonic phonon solver

For

\[H=\frac12p^TM^{-1}p+\frac12x^T\Phi x,\]

`HarmonicPhononSolver` constructs

\[D=M^{-1/2}\Phi M^{-1/2}\]

and solves

\[De_\nu=\omega_\nu^2e_\nu.\]

This is exact for the supplied harmonic model and avoids phonon Fock-space enumeration entirely.

The API is

```julia
result = solve(
    HarmonicPhononSolver(),
    harmonic_model,
)
```

and produces `PhononModeResult`.

## 5.11 Linear spin-wave solver

`LinearSpinWaveSolver` performs a quadratic Holstein--Primakoff expansion about a collinear \(\pm z\) reference state.

For example,

```julia
result = solve(
    LinearSpinWaveSolver(
        reference_signs=[1, -1, 1, -1],
    ),
    spin_model,
)
```

returns a `SpinWaveResult`.

Unlike packed spin/fermion acceleration or an exact harmonic normal-mode calculation, this method is explicitly approximate:

```julia
result.metadata[:approximation] == :linear_spin_wave
```

The solver also exposes

```julia
spinwave_zero_point_correction
spinwave_ground_energy
```

for the harmonic correction to the classical reference-state energy.

## 5.12 Automatic solver dispatch

`AutoSolver` chooses exact specializations when they can be inferred safely:

```julia
result = solve(AutoSolver(), model)
```

Its intended behavior is

```text
harmonic coordinate model -> HarmonicPhononSolver
quadratic fermion model   -> QuadraticFermionSolver
quadratic boson model     -> QuadraticBosonSolver
spin/fermion/boson model  -> optimized packed Lanczos when supported
otherwise                 -> generic fallback
```

`AutoSolver` does **not** automatically select linear spin-wave theory because LSWT introduces a physical approximation.

## 5.13 Krylov spectral tools

The optimized layer also exposes continued-fraction machinery:

```julia
cf = continued_fraction(
    A,
    seed;
    krylov_dim=128,
)

spectrum = continued_fraction_spectrum(
    cf,
    energy_axis;
    eta=0.02,
)
```

This provides the basis for large-system spectral functions of the form

\[G(z)=\langle\phi_{0}|(z-H)^{-1}|\phi_{0}\rangle,\]

without requiring a complete eigensystem.

## 5.14 Fermion Green functions and cluster perturbation theory

Spin-resolved zero-temperature one-particle Green matrices can be constructed directly between conserved fermion-number sectors. The workspace caches the half-filled ground state, particle and hole sector Hamiltonians, sector-changing fermion seeds, and projected Lanczos resolvents. The non-reorthogonalized production path streams recurrence vectors and sink projections rather than retaining a full many-body Krylov basis.

```julia
workspace = fermion_green_workspace(
    model;
    ground_sector=FermionSector(nup=6, ndown=6),
    spin=:up,
    ground_krylov_dim=160,
    green_krylov_dim=160,
)

Gc = cluster_green_matrix(workspace, 0.5 + 0.03im)
```

For convergence studies, a workspace constructed with a sufficiently long projected-Lanczos chain can be evaluated at a smaller effective Krylov dimension without rebuilding the sector Hamiltonians or recurrence data.

```julia
Gc128 = cluster_green_matrix(workspace, 0.5 + 0.03im; krylov_dim=128)
```

For a one-dimensional open cluster, the Senechal cluster-perturbation reconstruction is available through

```julia
Gk = cluster_perturbation_green(
    Gc,
    k;
    t_intercluster=t,
)

spectrum = cluster_perturbation_spectral_function(
    workspace,
    k_values,
    omega_values;
    eta=0.03,
    t_intercluster=t,
)
```

The spectral helper also accepts `krylov_dim=<n>` for inexpensive projected-Lanczos convergence tests using an existing workspace. The default spectral normalization is `-Im G/pi`, which integrates to unity per spin and momentum when the complete frequency axis is retained. Use `normalization=:senechal` for the `-2 Im G` convention used in Phys. Rev. Lett. 84, 522 (2000). The CPT reconstruction uses the open-cluster Green matrix, restores inter-cluster hopping at the one-body level, and periodizes the mixed cluster/superlattice Green function back to the original lattice momentum.

The Senechal empirical benchmark excludes the compilation-dominated `N=4` validation point from runtime scaling, measures `N=6,8,10` plus the production `N=12` workspace by default, reports ground-state/sector-build/seed/projected-chain timing separately, verifies projected-Lanczos map convergence, and checks spectral normalization as the retained frequency window is enlarged. Relevant environment controls include `PHUNDAMENTAL_CPT_SCALING_SIZES`, `PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS`, `PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_TOL`, `PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS`, and `PHUNDAMENTAL_CPT_OMEGA_SUM_RULE_TOL`.

## 5.15 Validation

The optimized layer includes

```julia
validate_optimized_against_generic
validate_phonon_modes
validate_mode_result
validate_spinwave_result
```

For small systems, packed backends can therefore be compared directly against the generic reference realization before being used at larger dimensions.

A representative workflow is

```julia
check = validate_optimized_against_generic(
    model;
    nev=6,
)

@assert check[:passed]
```

---

# 6. Thermodynamics, thermal sampling, and coupled/linked-cluster calculations

The thermodynamics layer provides equilibrium statistical mechanics without requiring every calculation to construct an explicit density matrix.

The mathematical canonical state is

\[\rho(\beta)=\frac{e^{-\beta H}}{Z},\qquadZ=\operatorname{Tr}(e^{-\beta H}),\qquad\beta=\frac{1}{k_BT}.\]

Thermal expectation values satisfy

\[\langle O\rangle_T=\operatorname{Tr}(\rho O).\]

The primary thermodynamic quantities are

\[F=-k_BT\ln Z,\]

\[U=\langle H\rangle,\]

\[S=\frac{U-F}{T},\]

and

\[C_V=\frac{\langle H^2\rangle-\langle H\rangle^2}{k_BT^2}.\]

The layer is organized around `AbstractThermalState` rather than around a mandatory dense matrix representation of \(\rho\).

## 6.1 Exact Gibbs thermodynamics

For a small system with a complete eigensystem,

```julia
thermal = thermal_state(
    model;
    temperature=50.0,
    method=ExactGibbs(),
    kB=kB,
)
```

constructs an `ExactGibbsState`.

The corresponding quantities are accessed with

```julia
log_partition_function(thermal)
partition_function(thermal)
free_energy(thermal)
internal_energy(thermal)
entropy(thermal)
heat_capacity(thermal)
```

An explicit density matrix may be constructed when the Hilbert space is small enough:

```julia
rho = density_matrix(
    thermal;
    max_dimension=4096,
)
```

The dimension guard is deliberate. A thermal ensemble does not need to become an explicit \(D\times D\) matrix merely because \(\rho\) is the formal statistical object.

## 6.2 Thermal expectation values

For an exact or sampled thermal state,

```julia
value = expectation(
    thermal,
    observable,
)
```

returns a `ThermalEstimate`.

Thermal fluctuations may be calculated using

```julia
thermal_variance(
    thermal,
    observable,
)
```

For example, a susceptibility can be constructed from

\[\chi=\frac{\beta}{N}\left(\langle M^2\rangle-\langle M\rangle^2\right).\]

## 6.3 Thermal typicality

For systems where a complete eigensystem is undesirable, `ThermalTypicality` uses random normalized states and thermally filtered vectors

\[|\psi_r(\beta)\rangle=e^{-\beta H/2}|r\rangle.\]

Then

\[\langle O\rangle_\beta\approx\frac{\sum_r\langle\psi_r|O|\psi_r\rangle}{\sum_r\langle\psi_r|\psi_r\rangle}.\]

Example:

```julia
thermal = thermal_state(
    model;
    temperature=50.0,
    method=ThermalTypicality(
        samples=32,
        krylov_dim=80,
        matrix_free=true,
        store_vectors=true,
        seed=1234,
    ),
    kB=kB,
)
```

When `store_vectors=true`, arbitrary thermal observables can be evaluated from the retained sampled thermal state.

Sampling results retain estimated standard errors.

## 6.4 Lower-memory thermal Lanczos

`ThermalLanczos` estimates the partition function and thermodynamic moments using Lanczos quadrature without retaining thermally filtered vectors.

It is intended for lower-memory temperature scans:

```julia
curve = thermal_curve(
    model,
    temperatures;
    method=ThermalLanczos(
        samples=32,
        krylov_dim=80,
        matrix_free=true,
        seed=1234,
    ),
    kB=kB,
)
```

A single set of stochastic Lanczos quadratures is reused across the entire temperature grid.

This makes the method substantially more economical than independently solving the thermal problem at every temperature.

Use `ThermalTypicality` when arbitrary noncommuting observable expectations are required. Use `ThermalLanczos` when the main goal is \(Z\), \(F\), \(U\), \(S\), or \(C_V\) over a temperature range.

## 6.5 Thermodynamic result types

A single-temperature result is stored in

```julia
ThermodynamicPoint
```

with fields

```text
temperature
kB
logZ
free_energy
internal_energy
entropy
heat_capacity
stderr
metadata
```

Temperature grids use

```julia
ThermodynamicCurve
```

with corresponding vectors of thermodynamic quantities and stochastic errors.

At \(T=0\), the framework uses the exact ground-state limit. `logZ` is intentionally not reported as a finite absolute quantity there, while the free energy, internal energy, entropy associated with ground-state degeneracy, and heat-capacity limit remain meaningful.

## 6.6 Connected-cluster thermodynamics

The package couples thermal sampling to connected finite clusters through

```julia
cluster_thermodynamics
```

The current implementation is a **finite-reference linked-cluster calculation**. The term “coupled cluster” in this context refers to coupling the cluster-generation and thermal-statistics workflows; it should not be confused with the quantum-chemistry coupled-cluster wavefunction hierarchy such as CCSD.

The computational sequence is

\[
\text{reference material/geometry}
\rightarrow
\text{connected induced clusters}
\rightarrow
H_c
\rightarrow
\text{thermal calculation}
\rightarrow
\text{linked-cluster weights}
\rightarrow
\text{order-by-order bulk estimate}.
\]

Connected clusters are generated from the reference bond graph rather than by averaging arbitrary random patches.

## 6.7 Linked-cluster weights

For an extensive thermodynamic quantity \(P\), the weight of an embedded connected cluster \(C\) is

\[W_P(C)=P(C)-\sum_{S\subsetneq C}W_P(S),\]

where disconnected subsets have zero linked weight.

The current extensive properties are

```text
free_energy
internal_energy
entropy
heat_capacity
```

The per-site finite-reference estimate through cluster order \(n\) is formed by summing all retained embedded-cluster weights and dividing by the number of sites in the reference lattice.

The result stores

```text
raw_curves
weights
partial_sums
convergence
reference_sites
metadata
```

in `ClusterThermodynamicsResult`.

## 6.8 Coupled cluster + thermal sampling API

A complete calculation is

```julia
bulk = cluster_thermodynamics(
    material,
    reference,
    XXZModel(
        1.0,
        0.6;
        neighbor_cutoff=1.01,
    );
    max_order=6,
    temperatures=5.0:5.0:300.0,
    method=ThermalLanczos(
        samples=16,
        krylov_dim=64,
        matrix_free=true,
    ),
    kB=kB,
)
```

The highest-order per-site estimate is obtained using

```julia
F  = bulk_estimate(bulk, :free_energy)
U  = bulk_estimate(bulk, :internal_energy)
S  = bulk_estimate(bulk, :entropy)
Cv = bulk_estimate(bulk, :heat_capacity)
```

A lower order may be requested explicitly:

```julia
Cv4 = bulk_estimate(
    bulk,
    :heat_capacity,
    4,
)
```

## 6.9 Convergence

The cluster result retains order-by-order partial sums

\[P^{(1)}(T),P^{(2)}(T),\ldots,P^{(n)}(T)\]

and differences

\[\Delta_n(T)=\left|P^{(n)}(T)-P^{(n-1)}(T)\right|.\]

These diagnostics are available through

```julia
bulk.partial_sums
bulk.convergence
```

A finite-reference calculation should only be interpreted as approximating the infinite-lattice thermodynamic limit when the retained orders converge before the boundary of the reference cluster affects the connected embeddings.

The implementation therefore records

```julia
metadata[:infinite_lattice_claim] == false
```

by default.

## 6.10 Cluster transformations and custom model builders

`cluster_thermodynamics` may apply a representation/model transformation to every generated cluster using `model_transform`.

It may also accept a custom `model_builder` for Hamiltonians whose site-dependent parameters require specialized indexing.

This makes it possible to combine

\[\text{cluster generation}+\text{representation transformation}+\text{thermal sampling}\]

within one controlled workflow.

## 6.11 Validation

Thermodynamic validation helpers include

```julia
validate_thermal_state
validate_sampling_against_exact
validate_linked_cluster_closure
```

A typical small-system validation is

```julia
check = validate_sampling_against_exact(
    model;
    temperature=1.0,
    sampled_method=ThermalLanczos(
        samples=32,
        krylov_dim=32,
    ),
)

println(check[:absolute_error])
```

The linked-cluster closure test verifies that, when the full connected reference graph has been included, the sum of all embedded linked weights reproduces the direct reference-cluster extensive quantity per site.

## 6.12 Overall role of the thermodynamics layer

The thermodynamics layer provides the bridge between exact or Krylov-solvable finite clusters and larger-material statistical predictions:

\[\boxed{\text{finite cluster}\rightarrow\text{thermal state / sampling}\rightarrow\text{connected-cluster reconstruction}\rightarrow\text{bulk thermodynamic estimate}.}\]

The approach does not make a small cluster automatically equivalent to a macroscopic material. Accuracy depends on thermal/quantum correlation lengths, cluster order, reference size, stochastic convergence, and the observable being calculated.

The design makes those approximations explicit rather than hiding them inside the solver.

---

`Phundamental.jl` is therefore structured around one consistent principle: preserve the mathematical meaning of the model while allowing the numerical method to change. Exact representation transformations, generic reference solvers, optimized backends, scattering observables, thermal sampling, and linked-cluster reconstruction are all exposed through compatible interfaces so that each stage can be validated independently.

## 7. Streamlined periodic harmonic-phonon workflow

The high-level harmonic API keeps physical choices explicit while absorbing deterministic primitive-cell, force-constant, solver, reciprocal-mesh, and full-Q bookkeeping. Low-level constructors remain available when those implementation details need to be controlled directly.

A one-dimensional diatomic model can be written as

```julia
using Phundamental

lattice = BravaisLattice(reshape([1.0], 1, 1))
crystal = LatticeCrystalStructure(
    lattice,
    [
        BasisSite(:A, :A, [0.0]),
        BasisSite(:B, :B, [0.5]),
    ],
)
material = Material(
    "Diatomic chain",
    crystal;
    species=Dict(
        :A => AtomicSpecies(:A; mass=1.0, nuclear_scattering_length=1.0),
        :B => AtomicSpecies(:B; mass=2.0, nuclear_scattering_length=1.5),
    ),
)
interactions = [
    HarmonicBondInteraction(1, 2, [0]; longitudinal=4.0),
    HarmonicBondInteraction(1, 2, [-1]; longitudinal=4.0),
]
model = harmonic_model(material, interactions)
```

`harmonic_model` compiles the interaction objects to `RealSpaceForceConstants`, constructs the primitive periodic cell, creates `HarmonicPhononModel`, and calls the existing `build_model` implementation. The explicit `force_constants`, `Supercell`, `HarmonicPhononModel`, and solver-first routes remain supported and are used as the reference implementation for validation.

The standard observables then require only the physical reciprocal-space requests:

```julia
E0 = groundenergy(model; qmesh=(256,))
bands = phonon_dispersion(model, range(-0.5, 0.5; length=101))
sqe = one_phonon_neutron_intensity(
    model,
    range(0.0, 2.0; length=161),
    range(0.0, 5.0; length=401);
    broadening=GaussianBroadening(0.03),
    temperature=0.0,
)
```

For one-dimensional models, scalar q and Q sequences are interpreted as reciprocal-lattice coordinates. In higher dimensions the API requires explicit `ReciprocalWaveVector` or `CartesianWaveVector` objects rather than guessing the coordinate convention.

The package root already uses `CrystalStructure` for the crystallographic invariant-generation type. To preserve that public API without a breaking rename, the lattice/material geometry type `Models.CrystalStructure` is re-exported as `LatticeCrystalStructure`. The namespace-qualified `Models.CrystalStructure` spelling remains valid.
