# Architecture

Phundamental is organized around semantic layers rather than around a single solver hierarchy. The package treats the mathematical model, its representation, any representation map, the numerical realization, the intrinsic observable, the probe-specific response, and instrumental resolution as distinct stages with independently inspectable results.

$$\text{model}\rightarrow\text{representation}\rightarrow\text{transformation}\rightarrow\text{solver}\rightarrow\text{observable}\rightarrow\text{probe}\rightarrow\text{resolution}$$

This architecture is the main compatibility contract of v0.0.8: changing a numerical backend must not silently change the physical representation, and applying a representation transformation must not be described as a mere storage optimization.

## Dependency order

The principal dependency direction is

$$\text{Foundation}\rightarrow\text{Algebra}\rightarrow\text{Crystallography}\rightarrow\text{Transformations}\rightarrow\text{Models/Solvers}\rightarrow\text{Observables}\rightarrow\text{Thermodynamics}\rightarrow\text{Greens}\rightarrow\text{Embedding}$$

The exact source-loading order differs slightly because `Models`, `Solvers`, and `Observables` contain mutually useful interfaces, but the conceptual direction remains acyclic at the level of mathematical responsibility. `Greens` depends on solver matrix realization and exact thermal states; `Embedding` depends on models, thermodynamics, Green functions, and solver primitives.

## Foundation as the package ontology

`Phundamental.PhundamentalCore` defines the objects that every later layer interprets: state spaces, algebras, symbolic operators, bases, representations, generative Hamiltonian spaces, typed parameter spaces, reciprocal wavevectors, and `ManyBodyModel`. `Phundamental.Foundation` is a semantic alias for this module.

A representation may be summarized as

$$\mathfrak R=(\mathcal S,\mathcal S_{\mathrm{phys}},\mathcal A,\mathcal B,\mathcal C,\mathcal G,|\Omega\rangle,\prec,\mathcal T_{\mathrm{num}})$$

where the ambient state space, physical sector, algebra, basis, exact constraints, gauge structure, reference state, ordering, and numerical truncation remain explicit. The split between physical and numerical restrictions is used by transformation validation and solver capability checks.

## Models are physical specifications, not solver inputs alone

The `Models` layer constructs material-aware `ManyBodyModel` objects from geometry, interactions, force constants, species metadata, and Hamiltonian specifications. A Hamiltonian specification such as `XXZModel`, `HubbardModel`, or `HarmonicPhononModel` describes a family of Hamiltonians; `build_model` combines that specification with a material and cluster to produce the concrete representation and symbolic operator expression supplied to later layers.

This separation permits the same solver API to act on models produced from direct symbolic construction, crystallographic invariant generation, or high-level material/geometry constructors.

## Transformations versus approximations versus backends

Three concepts that are frequently conflated are represented separately.

A representation transformation changes the algebraic description while carrying a `TransformationCertificate` that records exactness, invertibility, relation type, locality, constraints, gauge redundancy, and notes. Examples include Fourier, Jordan-Wigner, Majorana, coordinate-ladder, bosonic displacement, and Lang-Firsov maps.

An approximation intentionally changes the model or admissible model space. `AbstractApproximation`, `ApproximationCertificate`, `ApproximationResult`, and `ModelSpaceProjection` are the package-level mechanisms for making that change explicit.

A numerical backend changes only how an already specified representation is realized computationally. Packed spin, fermion, and boson backends therefore belong to `Solvers`, not `Transformations`.

$$\boxed{\text{representation transformation}\neq\text{approximation}\neq\text{numerical backend}}$$

## Numerical closure

Not every mathematically valid representation is immediately suitable for a finite-basis solver. `NumericalCapabilities` records whether the Hamiltonian is symbolically supported, whether the operator realization is implemented, whether the basis is finite and constructible, whether a numerical truncation is required, whether exact constraints remain unresolved in the target basis, whether a transported truncation must be replaced, and which solvers are compatible.

This prevents a transformation from being declared numerically usable merely because the symbolic map succeeds. For example, an unconstrained bosonic Fock space may be a valid exact target representation while still requiring an explicit numerical cutoff for matrix realization.

## Request requirements and solver compatibility

Typed observable requests declare the physical output they require through `RequestRequirements`. Numerical methods are checked against the working `ManyBodyModel` and request through `SolverCompatibility`. These objects answer different questions from model-space admissibility: request requirements describe the desired observable, while solver compatibility describes whether the current implementation has a computational route for that model/request combination.

$$\boxed{\text{physical admissibility}\neq\text{current computational compatibility}}$$

A future universal Hamiltonian generator may therefore produce physically admissible models for which every current solver reports incompatibility. Numerical limitations must not feed backward into the definition of the physical hypothesis space. Likewise, an approximate method such as linear spin-wave theory remains explicitly labeled as an approximation even when it is computationally compatible with a request.

Batch 5 extends this compatibility contract to methods whose numerical problem cannot be reconstructed from a finite `ManyBodyModel` alone. `DMFTForwardContext`, `LadderDGAForwardContext`, and `NLCEForwardContext` supply explicit computational context without becoming part of physical Hamiltonian admissibility. Batch 6 may rank fully specified routes, but only under an explicit `ForwardExecutionPolicy`; the method contexts themselves still neither rank methods nor authorize approximations.

The high-level DMFT and ladder-DΓA routes construct solver-specific embedding problems from a direct single-band `HubbardModel` plus explicit lattice/frequency context. The high-level NLCE route constructs connected cluster models from the source material, reference cluster, and Hamiltonian specification, then recursively evaluates the same custom observable request on each cluster through `ForwardProblem`. The raw embedding or NLCE result remains in `ForwardResult.solution`, preserving convergence and method provenance separately from the requested intrinsic observable.

Outer representation transformations are rejected for these method-specific lowering routes because their computational problems are derived from the physical source-model specification. This prevents a transformed Hamiltonian from being computed and then silently ignored while constructing DMFT, DΓA, or NLCE inputs. NLCE may still apply an explicitly declared cluster-level transformation through its method context.

## Solver results and capability-based introspection

Solver, observable, scattering, and thermodynamic result types do not inherit from one giant result union. The root module instead provides a capability protocol based on fields that results actually expose.

```julia
result_capabilities(result)
supports(result, :energies)
result_energies(result)
result_states(result)
result_axes(result)
result_values(result)
result_uncertainty(result)
result_provenance(result)
```

This lets new backends participate in generic consumers when they expose compatible result structure without requiring a central type switch to be edited.

## ForwardProblem and ForwardResult

`ForwardProblem` is the typed orchestration object for a complete calculation. It stores the source model, optional representation transformation, solver, solver arguments, optional explicit method-specific context, intrinsic observable request, probe, resolution model, and metadata.

```julia
problem = ForwardProblem(model; transformation=transformation, solver=solver, method_context=method_context, observable=observable, probe=probe, resolution=resolution)
```

`forward(problem)` executes each stage in order and returns a `ForwardResult` with separate fields:

```text
source_model
model
transformation_result
solution
observable_result
scattering_result
result
```

The final three fields preserve the distinction between intrinsic spectrum, probe cross section, and instrument response. Consumers that only need an intrinsic observable can stop before probe contraction; experimental comparisons can retain all intermediate results for diagnosis.

## Execution planning

`plan_forward(model, request; ...)` is the policy-governed layer above computational compatibility. It returns an ordinary `ForwardProblem`, preserving the existing orchestration object instead of introducing a competing physical-problem hierarchy. `ForwardRoute` represents a fully specified candidate method, while `ForwardExecutionPolicy` states which physical approximations may participate and whether static compatibility certification is mandatory.

The conservative default policy supplies only context-free exact numerical routes and requires static compatibility certification. Exact harmonic/quadratic specialization and equivalent backend selection may therefore be automatic, while LSWT, DMFT, ladder DΓA, NLCE, and other approximation-bearing routes must be supplied explicitly and authorized by approximation identity. Nested approximations are accumulated, so a DΓA route cannot hide its DMFT reference and an NLCE route cannot hide an approximate inner solver.

The default route catalog does not synthesize Lanczos because the current request requirements do not yet encode whether a partial low-energy eigensystem is sufficient. Lanczos remains available as an explicit caller-supplied exact numerical route. Planning currently ranks supported routes but does not predict memory use or wall time.

Automatic outer representation changes are limited to transformations whose certificates declare them exact and non-approximate. This rule preserves the architectural distinction

$$\text{exact representation map}\neq\text{physical approximation}\neq\text{numerical backend}.$$

A planning failure is a computational/policy failure, not a statement that the Hamiltonian is physically inadmissible. `ForwardPlanningReport` records the assessed candidates and their rejection reasons, and the selected route is retained in the returned `ForwardProblem` plus planning metadata. Analytic continuation remains outside automatic planning.

## Provenance

`ManyBodyModel.provenance` is carried through transformations and numerical-truncation updates. `with_provenance`, `remap_model`, `with_truncation`, and transformation machinery retain a history of how the working model was obtained. Provenance is intentionally attached to the model rather than inferred from the final solver type.

## Frequency-dependent propagation as an operator layer

`Greens` extends the architecture from static Hamiltonians to frequency-dependent operator-valued objects. `GreenFunction`, `NonInteractingGreenFunction`, `SelfEnergy`, and `HybridizationFunction` share an explicit frequency axis and component labels, allowing Dyson relations and validation identities to operate independently of the solver used to produce the propagator.

Nambu structure is implemented as index structure on a Green function rather than as a new Hamiltonian `Representation`. This keeps particle-hole doubling specific to the propagator object and avoids claiming that the underlying many-body Hilbert space has changed.

## Embedding as self-consistent composition

`Embedding` closes a self-consistency loop around the existing model, solver, thermodynamic, and Green-function layers:

$$\Sigma\rightarrow G_{\mathrm{lattice}}\rightarrow\mathcal G_0\rightarrow\Delta\rightarrow\text{finite bath}\rightarrow\text{impurity solve}\rightarrow\Sigma_{\mathrm{new}}$$

The ED-DMFT driver stores every iteration as `DMFTIteration`, records convergence history, separates the iteration residual from the independent final closure residual, supports fixed and adaptive mixing, carries continuation seeds explicitly, and validates finite-bath symmetry when requested.

## Exactness and validation metadata

The package consistently separates a numerical value from the evidence that the value is admissible. Transformation results carry certificates and validation dictionaries; solver results retain backend metadata; thermal estimates retain stochastic errors; DMFT results retain convergence histories and closure diagnostics. This is deliberate: a result should be inspectable without reconstructing the assumptions that produced it from call-site code.

## Public root versus namespace-qualified APIs

Many Foundation and Solvers names are re-exported at the package root for ordinary workflows. The named submodules remain the authoritative organizational units and should be used in documentation and explicit qualification when ambiguity matters.

The root name `CrystalStructure` refers to the crystallographic symmetry-layer structure. The lattice/material geometry type remains available as `Phundamental.Models.CrystalStructure` and is re-exported at the root under the compatibility alias `LatticeCrystalStructure`.

## API reference

```@meta
CurrentModule = Phundamental
```

```@docs
Phundamental.ForwardProblem
Phundamental.ForwardResult
Phundamental.SolverCompatibility
Phundamental.ForwardExecutionPolicy
Phundamental.ForwardRoute
Phundamental.ForwardRouteAssessment
Phundamental.ForwardPlanningReport
Phundamental.ForwardPlanningError
Phundamental.solver_compatibility
Phundamental.plan_forward
Phundamental.planning_report
Phundamental.forward
```

## Correlation-scale organization

The development API classifies correlation methods by the spatial information they add while retaining the existing module ownership. Single-site DMFT is the local-correlation method, ED/Lanczos with CPT or NLCE supplies short/intermediate spatial correlations, and diagrammatic DMFT supplies a route to nonlocal and critical correlations through two-particle vertex propagation. This classification is metadata and documentation, not an inheritance hierarchy, because the physical regimes overlap.

$$\text{local: DMFT}\qquad\text{short/intermediate: ED/Lanczos + CPT + NLCE}\qquad\text{long-range/critical: diagrammatic DMFT}$$

The observable layer remains method-independent. A susceptibility, spectral function, or structure factor retains provenance identifying the producing method, but probe and resolution code should consume the physical observable rather than dispatch on the correlation-scale method.

`correlation_scope(result)` reports the declared scope when result metadata supplies one. Current development results use `:local`, `:short_intermediate_spatial`, and `:long_range_critical`; results without an explicit declaration return `:unspecified`.

## Research-grade correlation-method validation

The three correlation-scale branches use different validation hierarchies. NLCE requires exact graph/isomorphism checks, independent embedding-size stability, bare-order convergence, optional resummation diagnostics, and reproduction of published cluster-count or thermodynamic-limit benchmarks. Diagrammatic DMFT requires a converged local DMFT reference, exact Bethe-Salpeter closure on controlled test tensors, explicit inversion conditioning, frequency-box and momentum-grid convergence, and comparison with independent published Hubbard-model benchmarks where numerical reference data are available.

Software tests and documentation builds establish API consistency but are not by themselves evidence of physical accuracy. Benchmark provenance, normalization conventions, model parameters, and numerical cutoffs must remain attached to every research validation result.
