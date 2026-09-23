# Phundamental.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alt-f4-dev.github.io/Phundamental.jl/stable/) [![Documentation Build](https://github.com/alt-f4-dev/Phundamental.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/alt-f4-dev/Phundamental.jl/actions/workflows/documentation.yml) [![Julia](https://img.shields.io/badge/Julia-1.12%2B-9558B2.svg)](https://julialang.org/) [![Version](https://img.shields.io/badge/version-v0.0.7-informational.svg)](https://github.com/alt-f4-dev/Phundamental.jl/tree/v0.0.7)

`Phundamental.jl` is a representation-aware Julia framework for constructing, transforming, solving, and analyzing quantum many-body models. The design philosophy is to maintain a strict boundary between the physical model, mathematical representation, numerical approximation, solver backend, observable definition, probe response, and instrument response, keeping each of these concepts explicitly separated.

The package supports symbolic many-body operator algebra, representation transformations, crystallographic invariant generation, finite-cluster and specialized numerical solvers, scattering observables, equilibrium thermodynamics, finite-temperature Green functions, and finite-bath embedding workflows including normal-state single-site exact-diagonalization dynamical mean-field theory (ED-DMFT).

The current stable release is `v0.0.7`.

## Documentation

The complete documentation is the primary reference for the public API, mathematical conventions, workflows, and validation scope.

- [Stable documentation](https://alt-f4-dev.github.io/Phundamental.jl/stable/)
- [Development documentation](https://alt-f4-dev.github.io/Phundamental.jl/dev/)
- [Architecture](https://alt-f4-dev.github.io/Phundamental.jl/stable/architecture/)
- [Validation](https://alt-f4-dev.github.io/Phundamental.jl/stable/validation/)
- [ED-DMFT workflow](https://alt-f4-dev.github.io/Phundamental.jl/stable/workflows/ed-dmft/)


## Installation

`Phundamental.jl` is currently installed directly from GitHub. To install the stable `v0.0.7` release:

```julia
using Pkg
Pkg.add(url="https://github.com/alt-f4-dev/Phundamental.jl", rev="v0.0.7")
```

Then load the package normally:

```julia
using Phundamental
```

To work against the current development branch instead:

```julia
using Pkg
Pkg.develop(url="https://github.com/alt-f4-dev/Phundamental.jl")
```


For a source checkout:

```bash
git clone git@github.com:alt-f4-dev/Phundamental.jl.git
cd Phundamental.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Design

A typical forward calculation follows the conceptual pipeline

$$\text{material/geometry }\rightarrow\text{ Hamiltonian model }\rightarrow\text{ representation }\rightarrow\text{ transformation (optional) }\rightarrow\text{ solver }\rightarrow\text{ observable }\rightarrow\text{ probe response }\rightarrow\text{ instrument response}.$$

A `ManyBodyModel` stores the physical representation, Hamiltonian, parameters and metadata, observable registry, and provenance. A `Representation` separately tracks the ambient state space, physical subspace, operator algebra, basis, exact constraints, gauge structure, reference state, ordering, and numerical truncation.

A central package invariant is

$$\boxed{\text{ambient state space }\neq\text{ physical subspace }\neq\text{ numerical truncation}.}$$

Physical constraints define the model. Numerical truncations define a computational approximation. The framework does not silently reinterpret one as the other.

The package also distinguishes representation changes, numerical encodings, and physical approximations. A Jordan-Wigner map changes representation; a packed-bit spin backend changes numerical encoding; linear spin-wave theory introduces an approximation.

## Quick start

The following example constructs a finite one-dimensional spin-1/2 chain and solves it by exact diagonalization:

```julia
using Phundamental

#Bravais Lattice & Crystal Structure
lattice = BravaisLattice(reshape([1.0], 1, 1))
crystal = LatticeCrystalStructure(lattice, [BasisSite(:X, :X, [0.0])])

#Atomic Species & Basis Properties
species = Dict(:X => AtomicSpecies(:X; mass=1.0))
basis_properties = [SiteProperties(spin=1//2)]

#Material System
material = Material("spin chain", crystal; species=species, basis_properties=basis_properties)

#Finite Non-periodic Supercell Cluster
cluster = Supercell(crystal, [8]; periodic=false)

#XXZ Hamiltonian Model Embedded in Material System
model = build_model(material, cluster, XXZModel(1.0, 0.6; neighbor_cutoff=1.01))

#XXZ Exact Diagonalization Solution
solution = solve(ExactDiagonalization(), model)

#Print out ground-state energy
println(groundenergy(solution))
```

The same model can be passed through the high-level forward API:

```julia
result = forward(model; solver=ExactDiagonalization())
```

Representation transformations remain explicit:

```julia
#1D Chain Length
chain = 1:8

#Map Spin `model` via Jordan-Wigner Transformation
mapped = transform(JordanWigner(chain), model)

#Fermionic Representation
fermion_model = mapped.model
```

For numerical structures recognized by the optimized layer:

```julia
solution = solve(AutoSolver(), model)
```

`AutoSolver` may select an exact optimized backend when the model structure permits it, but it does not silently select a physical approximation such as linear spin-wave theory.

## Package organization

| Module | Role |
| --- | --- |
| `Phundamental.Foundation` | State spaces, operator algebras, representations, model spaces, parameters, and `ManyBodyModel`; this is an alias of the internal `PhundamentalCore` module. |
| `Phundamental.Algebra` | Noncommutative commutators, rewrite rules, normal ordering, and canonicalization. |
| `Phundamental.Crystallography` | Space-group operations, symmetry discovery, operator actions, and crystallographically invariant Hamiltonian generation. |
| `Phundamental.Transformations` | Exact and constrained representation maps, transformation certificates, approximation interfaces, registries, and representation graphs. |
| `Phundamental.Models` | Lattices, materials, interactions, spin/fermion/phonon/hybrid Hamiltonians, impurity models, and model construction. |
| `Phundamental.Solvers` | Generic exact/Krylov solvers, packed backends, symmetry sectors, quadratic-mode solvers, phonons, spin waves, and cluster Green-function tools. |
| `Phundamental.Observables` | Correlation functions, spectra, neutron scattering, multiphonon response, probe contraction, and instrument resolution. |
| `Phundamental.Thermodynamics` | Exact Gibbs calculations, thermal typicality, thermal Lanczos, sector sampling, and linked-cluster thermodynamics. |
| `Phundamental.Greens` | Fermionic Matsubara axes, Lehmann Green functions, Nambu structure, Dyson relations, self-energies, moments, and validation identities. |
| `Phundamental.Embedding` | Discrete baths, bath fitting, impurity problems, ED impurity solvers, lattice Green functions, mixing, and ED-DMFT self-consistency. |

Detailed documentation for each module is available under the [stable documentation](https://alt-f4-dev.github.io/Phundamental.jl/stable/).

## Core capabilities

### Representation-aware symbolic models

The symbolic layer provides spin, boson, fermion, Majorana, coordinate-momentum, composite, constrained, truncated, and biorthogonal representations. Hamiltonians and observables are represented as noncommutative operator expressions and retain enough structure for algebraic transformation and validation.

### Representation transformations

The transformation layer currently includes Fourier, Bogoliubov, particle-hole, Majorana, Jordan-Wigner, Holstein-Primakoff, Dyson-Maleev, Schwinger-boson, Abrikosov-fermion, slave-particle, coordinate-ladder, bosonic-displacement, Lang-Firsov, and spin-polaron transformations.

Transformation certificates distinguish unitary equivalence, canonical maps, algebra isomorphisms, constrained embeddings, nonunitary similarity relations, gauge-redundant embeddings, and composite relations.

### Models and geometry

The model layer supports direct and reciprocal lattices, basis sites, supercells, neighbor lists, materials, local frames, long-range Coulomb and dipolar interactions, harmonic force constants, and standard spin, fermion, phonon, hybrid, and impurity Hamiltonians.

Current high-level Hamiltonian specifications include Heisenberg, XXZ, Hubbard, Anderson impurity, harmonic phonon, local phonon, Holstein, and longitudinal spin-boson models.

### Solvers

Reference and optimized numerical paths include exact diagonalization, symbolic and optimized Lanczos, packed spin/fermion/boson backends, conserved sectors, time evolution, quadratic fermionic and bosonic mode solvers, harmonic phonons, linear spin-wave theory, continued fractions, projected fermion resolvents, and cluster perturbation theory.

Optimized numerical backends do not change the underlying physical representation. Approximate solvers are identified explicitly.

### Observables and scattering

The observable layer provides static and dynamic correlations, Lehmann spectra, spectral densities, spin tensors, magnetic neutron scattering, coherent one- and multiphonon neutron response, electron spectral functions, ensemble observables, numerical broadening, and instrument-resolution convolution.

Intrinsic spectra, probe cross sections, and instrument response are represented as separate stages.

### Thermodynamics

The thermodynamics layer provides exact Gibbs thermodynamics, thermal typicality, thermal Lanczos, thermal curves, sector-resolved sampling, and finite-reference linked-cluster reconstruction with explicit convergence diagnostics.

### Green functions and embedding

The Green-function layer provides finite-temperature fermionic Matsubara Green functions, Lehmann construction, Nambu indexing, Dyson relations, self-energies, high-frequency moments, causality checks, and particle-hole diagnostics.

The embedding layer provides finite discrete Anderson baths, constrained and unconstrained bath fitting, sectorized ED impurity solving, lattice and Weiss Green functions, adaptive self-energy mixing, fixed-point diagnostics, and normal-state single-site ED-DMFT.

## Representative workflows

### Forward calculations

`ForwardProblem`, `forward`, and `ForwardResult` coordinate model transformation, solution, observable evaluation, probe response, and resolution while retaining every intermediate stage.

```julia
problem = ForwardProblem(model; solver=ExactDiagonalization())
result = forward(problem)
```

See the [forward-calculation workflow](https://alt-f4-dev.github.io/Phundamental.jl/stable/workflows/forward-calculations/).

### Representation transformations

Transformations act on the representation, Hamiltonian, and registered observables and return a `TransformationResult` containing the transformed model, mathematical certificate, and validation metadata.

See the [representation-transformation workflow](https://alt-f4-dev.github.io/Phundamental.jl/stable/workflows/representation-transformations/).

### Scattering

Scattering calculations preserve the hierarchy

$$\text{intrinsic correlation function}\rightarrow\text{probe response}\rightarrow\text{instrument response}.$$

See the [scattering workflow](https://alt-f4-dev.github.io/Phundamental.jl/stable/workflows/scattering/).

### ED-DMFT

The `Embedding` layer implements normal-state single-site exact diagonalization dynamical mean field theory (ED-DMFT) using a finite Anderson bath, exact-diagonalization impurity solver, bath fitting, lattice self-consistency, mixing, convergence classification, and independent closure diagnostics.

See the [ED-DMFT workflow](https://alt-f4-dev.github.io/Phundamental.jl/stable/workflows/ed-dmft/).

## Validation

`Phundamental.jl` is developed around explicit cross-checks between symbolic definitions, generic numerical references, optimized implementations, limiting cases, and published many-body benchmarks.

The validation suite includes representation-transformation numerics, optimized-versus-reference solver comparisons, bosonic truncation convergence, crystallographic invariant generation, finite-temperature Green-function identities, bath discretization, dense-versus-sectorized impurity ED, threaded-versus-serial impurity ED, DMFT fixed-point and closure tests, and scientific ED-DMFT benchmarks.

Run the package test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Check strict package precompilation with:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile(strict=true)'
```

Build the documentation locally with:

```bash
julia --project=docs docs/make.jl
```

Scientific benchmarks are kept under `benchmarks/` and are distinct from the unit/regression suite. See the [validation documentation](https://alt-f4-dev.github.io/Phundamental.jl/stable/validation/) for the supported claims, benchmark scope, and known limitations of `v0.0.7`.

## Repository layout

```text
Phundamental.jl/
├── .github/
│   └── workflows/
├── benchmarks/
├── demos/
├── docs/
├── src/
│   ├── algebra/
│   ├── core/
│   ├── crystallography/
│   ├── embedding/
│   ├── greens/
│   ├── models/
│   ├── observables/
│   ├── solvers/
│   ├── thermodynamics/
│   └── transformations/
├── test/
├── Project.toml
└── README.md
```

## Release scope

`v0.0.7` is the current stable API baseline. It includes the representation-aware model architecture, transformation framework, generic and optimized solver layers, scattering and thermodynamic workflows, finite-temperature Green functions, and the first finite-bath embedding/ED-DMFT implementation.

The current ED-DMFT implementation targets the normal-state single-site single-band problem. Superconducting ED-DMFT and broader impurity geometries are outside the `v0.0.7` scope, but remain a target for future releases.

For complete API details and current development status, use the [stable documentation](https://alt-f4-dev.github.io/Phundamental.jl/stable/) or [development documentation](https://alt-f4-dev.github.io/Phundamental.jl/dev/).
