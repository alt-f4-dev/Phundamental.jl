# Fermionic linked-cluster workflows

These workflows retain the original numerical conventions, domain restrictions, and validation instructions.

## Fermionic and multiorbital linked-cluster calculations

The finite-reference numerical linked-cluster expansion (NLCE) supports fermionic Hamiltonians through an explicitly certified site-to-mode mapping. This route reuses Phundamental's existing connected-cluster enumeration, linked-cluster subtraction, thermal solvers, and forward API. It does not depend on the DMFT impurity approximation: the inner solver evaluates each finite fermionic cluster and the NLCE engine performs inclusion–exclusion.

### Mathematical contract

Let $C$ denote an open connected set of physical sites and $P(C)$ an extensive cluster observable. The linked-cluster weight is $W(C)=P(C)-\sum_{S\subsetneq C}W(S)$, with the sum taken over proper connected embedded subclusters. The finite-reference result sums the appropriate embedded weights over all available clusters and, by default, divides by the number of sites in the finite reference. Convergence with expansion order and reference size must be tested independently.

A source fermionic Hamiltonian is specified as $H=\sum_i H_i+\sum_{(i,j)\in\mathcal B}H_{ij}$. Every $H_i$ contains only modes belonging to physical site $i$, and each $H_{ij}$ has support on exactly the two declared endpoints of one physical bond. The supplied decomposition is checked against the canonical numerical source Hamiltonian before any clusters are generated. The code does not infer a bond contribution by dividing coefficients of a periodically folded Hamiltonian.

### Constructing the source decomposition

Pass the original `ManyBodyModel` and its exact open `Supercell` to `certify_fermionic_clusters`. The `site_modes` argument is a vector of physical-site mode lists, ordered by local component, for example `[[1,2,3,4], [5,6,7,8]]` for two sites with two spinful orbitals each. Modes must partition the complete source fermionic mode set. Provide `onsite_terms` as a dictionary mapping site indices to local fermionic expressions and `bond_terms` as `Bond => expression` pairs. Supply `bonds` separately when physical connectivity includes interactions with zero coupling.

```julia
source = build_model(material, reference, specification)
decomposition = certify_fermionic_clusters(source, reference;
    site_modes=site_modes,
    onsite_terms=onsite_terms,
    bond_terms=bond_terms,
    bonds=source.parameters[:bonds])
```

The decomposition cannot contain a nonzero scalar identity offset without an extensive site assignment. Symbolic coefficients must be numerically bound before certification. The source occupation basis must use canonical mode order; projected Fock spaces require a separate, explicit parity-preserving mapping. The initial implementation supports **open finite references only** and rejects periodic references, including potentially ambiguous physical-image couplings.

### NLCE and forward evaluation

A certified decomposition can be evaluated directly by supplying a cluster evaluator returning an additive scalar or array.

```julia
problem = fermionic_nlce_problem(decomposition, evaluator; max_order=3)
result = solve(NLCESolver(), problem)
```

The same decomposition integrates into the typed forward interface, so that the existing finite-temperature solver and request semantics are retained.

```julia
request = ThermodynamicCurveRequest([0.5, 1.0]; kB=1.0)
context = FermionicNLCEForwardContext(decomposition, 3;
    inner_solver=ThermodynamicForwardSolver(ExactGibbs()))
result = forward(ForwardProblem(source;
    solver=NLCESolver(), method_context=context, observable=request))
```

The cluster builder remaps every source fermion creation, annihilation, and number operator to the canonical local mode order in the induced cluster, preserving the order of noncommutative products. Complete source Hamiltonians are compared against regenerated clusters using independent finite-basis matrix realization in the acceptance tests. The regenerated models preserve physical-site provenance and supported source observable expressions whose modes are fully retained.

### Domain, numerical cost, and validation

The implementation supports finite interacting spinful and multiorbital fermionic source Hamiltonians built from explicitly assigned onsite and two-site operator contributions. It does not automatically derive decomposition from a generic generated model's periodic folded couplings, certify higher-than-two-site interactions, or perform cluster DMFT. In particular, a DMFT impurity self-energy must not be inserted into inclusion–exclusion without a separately justified extensive cluster functional.

The generic NLCE driver evaluates each embedded connected cluster once and caches its raw value for subtraction. The default `max_fock_dimension=4096` guard prevents constructing unexpectedly large Hilbert spaces; raise the limit explicitly only when the chosen inner solver and available memory permit it. Memory costs are dominated by the chosen inner numerical solver; the cluster mode count is the sum of its constituent local mode counts and full finite-basis ED therefore grows exponentially with cluster order. Test fermionic anticommutation signs, explicit onsite/bond reconstruction, independent direct cluster ED, atomic-limit linked-cluster cancellation, and complete package regressions before extending the accepted model domain.

---
