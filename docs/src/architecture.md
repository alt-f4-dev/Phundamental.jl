# Architecture

Phundamental is organized around semantic layers rather than around a single solver hierarchy. The package treats the mathematical model, its representation, any representation map, the numerical realization, the intrinsic observable, the probe-specific response, and instrumental resolution as distinct stages with independently inspectable results.

$$\text{model}\rightarrow\text{representation}\rightarrow\text{transformation}\rightarrow\text{solver}\rightarrow\text{observable}\rightarrow\text{probe}\rightarrow\text{resolution}$$

This architecture is a central compatibility contract: changing a numerical backend must not silently change the physical representation, and applying a representation transformation must not be described as a mere storage optimization.

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

## Canonical model structure

The canonical structural layer sits between the physical model-space definition and numerical execution. A `ModelStructure` records coefficient-independent canonical operator terms, their local supports, an interaction hypergraph, and a physical representation signature.

$$\text{model space}\rightarrow\text{canonical structure}\rightarrow\text{parameterization}\rightarrow\text{forward execution}$$

Structural identity is intentionally representation-sensitive. Two Hamiltonians are `structurally_equivalent` only when their canonical operator topology agrees in the same physical representation context. Representation names and numerical truncations do not define that context, but state space, algebra, basis, exact constraints, gauge metadata, reference state, and ordering do.

An exact representation transformation defines a different relation. If $T$ is exact, then $H$ and $T(H)$ may represent the same physical theory while having different structural signatures, particularly for nonlocal maps such as Jordan--Wigner. The transformation layer therefore exposes `structurally_related(source, target, T)` rather than redefining structural equality across representations.

This distinction preserves locality and interaction topology for later structural inference while retaining exact transformation provenance.

## Constraint-bearing parameter schemas

The parameter schema represents the continuous coupling domain as an explicit layer between canonical structure and model instantiation. The existing `ParameterSpace` is the schema object; it is not replaced by a parallel `ParameterSchema` hierarchy.

$$\mathfrak G\longrightarrow\Theta(\mathfrak G)\longrightarrow\boldsymbol\theta\longrightarrow H_{\mathfrak G}(\boldsymbol\theta)$$

A `ParameterBinding` associates each named scalar coefficient with one Hamiltonian-basis direction and the canonical structural signature of that direction. Parameter values therefore remain outside structural identity while the schema still records exactly what each coupling multiplies. This is the intended Julia boundary for later Python parameter inference.

`ParameterSpace` can impose fixed values, shared/equal couplings, and general linear equality constraints. These relations define an affine manifold

$$A\boldsymbol\theta=\mathbf b$$

inside the ordinary bound box. `parameter_constraint_report` evaluates a concrete point, `parameter_constraint_matrix` exports the affine equality system, and `parameter_dof` reports its dimension. The existing `free_parameter_indices` function remains backward compatible and only reflects the `fixed` flag; it is not redefined to solve arbitrary equality constraints.

The parameter schema is distinct from physical admissibility and computational compatibility:

$$\boxed{\mathfrak G\in\mathcal M}\qquad\boxed{\boldsymbol\theta\in\Theta(\mathfrak G)}\qquad\boxed{\text{solver supports }H_{\mathfrak G}(\boldsymbol\theta)}$$

These are three different questions. A structurally admissible model can have an invalid parameter point, and a physically valid parameterized model can still lack a compatible numerical route.

Exact representation transformations preserve parameter names, values, and equality constraints while regenerating structural bindings against the transformed Hamiltonian basis. This keeps the physical coupling schema continuous across an exact map without collapsing representation-local structural identity.

## Material- and crystallography-aware Hamiltonian generation

Crystallographic Hamiltonian generation treats symmetry as an explicit upstream constraint rather than metadata attached after structural construction. Material geometry is converted to the backend-neutral crystallographic structure, the space-group action defines symmetry-invariant operator combinations, model-space admissibility filters the physical candidate space, canonical model structure records the resulting operator topology, and the parameter schema specifies continuous couplings.

$$\text{Material / CrystalStructure}\rightarrow\text{Crystallography}\rightarrow\mathcal G\rightarrow\mathcal H(\mathcal G)\rightarrow\mathfrak G\rightarrow\Theta(\mathfrak G).$$

`HamiltonianGenerationResult` retains each stage separately: the source material and optional periodic supercell, crystallographic structure and dataset, `GenerativeSpecification`, `HamiltonianSpace`, `ModelStructure`, and `ParameterSpace`. The generator therefore coordinates existing authoritative objects instead of creating a parallel model hierarchy.

The standard crystallographic operator action has an explicit site-index contract. One operator index must correspond to one crystallographic site for the automatically supported spin, boson, fermion, and Majorana actions. Richer orbital or internal-component conventions must provide a custom symmetry action rather than allowing the generator to guess an index map. Coordinate/momentum primitives likewise require a custom action because their bare mode indices do not encode Cartesian vector components.

When a material-derived specification declares `max_range`, `generate_hamiltonian_family` can construct a `range_evaluator` directly from crystallographic geometry. The interaction range of a canonical term is the maximum minimum-image Cartesian separation among its distinct site supports. This preserves the rule that symbolic indices themselves never imply physical distance.

The material-only route addresses primitive-cell-indexed degrees of freedom. An explicit fully periodic `Supercell` may instead be supplied when translated sites must occur as distinct finite operator indices. Open clusters are not passed to Spglib as periodic cells because that would silently change their boundary conditions.

Hamiltonian-family generation stops at family construction and optional `ManyBodyModel` instantiation. It does not search structural neighborhoods, optimize parameters, choose numerical solvers, or execute observables. Those responsibilities remain separated between structural search, structural forward closure, and the later Python inverse layer.

## Transformations versus approximations versus backends

Three concepts that are frequently conflated are represented separately.

A representation transformation changes the algebraic description while carrying a `TransformationCertificate` that records exactness, invertibility, relation type, locality, constraints, gauge redundancy, and notes. Examples include Fourier, Jordan-Wigner, Majorana, coordinate-ladder, bosonic displacement, and Lang-Firsov maps.

An approximation intentionally changes the model or admissible model space. `AbstractApproximation`, `ApproximationCertificate`, `ApproximationResult`, and `ModelSpaceProjection` are the package-level mechanisms for making that change explicit.

A numerical backend changes only how an already specified representation is realized computationally. Packed spin, fermion, and boson backends therefore belong to `Solvers`, not `Transformations`.

$$\boxed{\text{representation transformation}\neq\text{approximation}\neq\text{numerical backend}}$$

## Numerical closure

Not every mathematically valid representation is immediately suitable for a finite-basis solver. `NumericalCapabilities` records whether the Hamiltonian is symbolically supported, whether the operator realization is implemented, whether the basis is finite and constructible, whether a numerical truncation is required, whether exact constraints remain unresolved in the target basis, whether a transported truncation must be replaced, and which solvers are compatible.

This prevents a transformation from being declared numerically usable merely because the symbolic map succeeds. For example, an unconstrained bosonic Fock space may be a valid exact target representation while still requiring an explicit numerical cutoff for matrix realization.

## Physical model-space admissibility

The Foundation model-space layer owns the declared physical constraints used to decide whether a symbolic Hamiltonian is admissible under a `GenerativeSpecification`. `AbstractModelSpace` is the public root of this physical layer, with `HamiltonianSpace` as the current finite-dimensional implementation. `admissibility_report` produces an inspectable `AdmissibilityReport` containing algebra compatibility, body-order, physical-range, and custom-condition checks. The report distinguishes a rejected criterion from a criterion that is declared but cannot yet be evaluated. Symmetry invariance, Hermiticity, linear independence, and generator completeness remain separate certification questions. Material-aware generation records the crystallographic fixed-space generator evidence for generated families without broadening the meaning of the model-space admissibility report.

$$\boxed{\text{Model-space admissibility}=\text{evaluable declared model-space constraints}}$$

A physical range cutoff is meaningful only when the specification can evaluate physical range. `max_range` therefore requires an explicit `range_evaluator`; Foundation never converts site labels, mode indices, or operator support counts into distances implicitly. This keeps geometric assumptions outside the symbolic index convention. Material-aware generation supplies a geometry-aware evaluator only when the material/crystallographic site-index mapping is explicit.

Generated Hamiltonian spaces are audited before `generate_hamiltonian_space` returns them, and generated or rebuilt parameterized models are checked against their attached space. A `HamiltonianSpace` report also carries a `:basis_span` check. Canonical algebraic reduction certifies numeric linear-span membership after exact algebraic rewriting; coefficients that cannot be reduced to numeric linear coordinates remain explicitly unresolved. Direct `HamiltonianSpace` construction remains available for imported or partially certified bases so an invalid or incomplete space can be inspected rather than being impossible to represent. The resulting admissibility audit is deliberately narrower than a future model-space certificate: it does not yet prove Hermiticity, linear independence, symmetry completeness, or completeness of the candidate-generating procedure.

Exact representation transformations reconstruct a target `GenerativeSpecification` in the target algebra. Representation-specific physical-range evaluators are not assumed to survive that map automatically; if a transformed specification still declares `max_range` but has no valid target evaluator, its range admissibility remains explicitly incomplete rather than being silently inferred.

## Request requirements and solver compatibility

Typed observable requests declare the physical output they require through `RequestRequirements`. Numerical methods are checked against the working `ManyBodyModel` and request through `SolverCompatibility`. These objects answer different questions from model-space admissibility: request requirements describe the desired observable, while solver compatibility describes whether the current implementation has a computational route for that model/request combination.

$$\boxed{\text{physical admissibility}\neq\text{current computational compatibility}}$$

Universal Hamiltonian generation may therefore produce physically admissible models for which every current solver reports incompatibility. Numerical limitations must not feed backward into the definition of the physical hypothesis space. Likewise, an approximate method such as linear spin-wave theory remains explicitly labeled as an approximation even when it is computationally compatible with a request.

The forward compatibility contract also applies to methods whose numerical problem cannot be reconstructed from a finite `ManyBodyModel` alone. `DMFTForwardContext`, `LadderDGAForwardContext`, and `NLCEForwardContext` supply explicit computational context without becoming part of physical Hamiltonian admissibility. The forward planner ranks fully specified routes only under an explicit `ForwardExecutionPolicy`; the method contexts themselves still neither rank methods nor authorize approximations.

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

## Finite structural search over crystallographic invariants

Structural search organizes each finite crystallography-generated `HamiltonianSpace` into a `StructuralFamily` of indivisible symmetry-complete `StructuralUnit`s. These units contain whole invariant basis directions rather than individual canonical monomials. A `StructuralSelection` inherits the original physical specification and exposes a selected child Hamiltonian space, canonical operator support structure, and stable within-family signature. Structural operations include additions, removals, replacements, constrained one-move neighborhoods, bounded exhaustive enumeration, and within-family Hasse hierarchies.

$$\mathcal H(\mathcal G)\longrightarrow\{\mathcal H'_1,\mathcal H'_2,\ldots\},\qquad H(\mathbf z,\boldsymbol\theta)=\sum_a z_a\theta_a\Phi_a.$$

Each omitted invariant direction has parent coefficient exactly zero, independent of whether a retained direction's numerical coefficient happens to vanish. Parent bounds, fixed parameters, and supported affine equalities survive structural selection through explicit zero-substitution; incompatible parameter subspaces are rejected. A structurally admissible choice remains distinct from a valid parameter point and from the existence of a compatible numerical solver. The [universal generation and structural-search workflow](workflows/universal-hamiltonian-generation.md) describes this interface in detail.

## Julia-authoritative structural-to-forward closure

Material-aware crystallographic generation defines the finite invariant Hamiltonian family, structural search defines symmetry-complete subspaces, and structural forward closure composes structural instantiation with the existing numerical forward calculation. Julia owns physical admissibility, operator construction, parameter constraints, representation validation, numerical solutions, intrinsic observables, probe contractions, and resolution effects. An external Python inverse algorithm may search deterministic structural identifiers and numerical parameter values but must not rebuild or authenticate physical Hamiltonians independently.

The machine-readable `describe` interface exposes versioned neutral descriptions and SHA-based structural identity, while `reconstruct_model` resolves submitted selection IDs only against a live certified Julia `StructuralFamily`. Arbitrary user-defined callback semantics remain tied to their Julia implementation and must not be inferred from serialized metadata. Prepared evaluations do not reuse mutable solver state between parameter points.

## Milestone E — Capability and module consolidation

High-level solver compatibility now has a public dispatch extension contract in `ForwardCapabilityContracts.jl`; built-in solver-specific methods are separated into `forward/BuiltinSolverCapabilities.jl`. The planner consumes the same compatibility and approximation information, preserving the conservative policy boundary. Compatibility reports retain their former string diagnostics and additionally offer typed, stage-specific issues. The package root remains the only canonical module assembly entry point; obsolete alternative wrappers were retired after a reference audit. The corresponding [extension workflow](workflows/capability-extension-and-module-layout.md) documents the remaining guarded checks and the required solver-execution methods.

## Source-file organization (development v0.1.8)

`src/Phundamental.jl` is the only direct Julia source file in `src/`; it defines the canonical package module, constructs the established submodule hierarchy, and retains the existing public exports. Moving implementation files into directories does not introduce additional Julia `module` definitions, change import paths, or alter method ownership. The `include` order remains authoritative and is preserved by this relocation.

The `src/forward/` directory contains computational contexts, capability contracts, compatibility composition and built-in solver support, forward problems and result capabilities, planning, structural batch and NLCE forwarding, and extended forward routing. The built-in capabilities continue to be included by `ForwardCapabilities.jl` using a sibling-relative path. `src/embedding/GeometricEmbedding.jl` defines geometry-derived embedding helpers; `src/greens/AnalyticContinuation.jl` defines the explicit approximate Matsubara continuation; `src/experimental/ExperimentalData.jl` defines portable prediction, measurement and comparison records; and `src/api/ConvenienceAPI.jl` provides cross-layer constructors and aliases. These are physical source directories rather than new top-level namespaces.

The current root entry point remains comparatively large because it assembles ten established submodules and a large public export list. Reducing its size is a separate architectural change with considerably more include-order and scope risks and is not part of this source-file relocation.
