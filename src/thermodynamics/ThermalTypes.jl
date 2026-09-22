# Common thermal methods, states, estimates, workspaces, and result containers.

abstract type AbstractThermalMethod end
abstract type AbstractThermalState end

"""
    ExactGibbs(; solver=ExactDiagonalization())

Construct the canonical Gibbs ensemble from a complete spectrum. `thermal_state` requires a complete eigensystem because it retains eigenvectors; `thermal_curve` uses the energy-only exact spectrum path automatically when the solver is `ExactDiagonalization`.
"""
struct ExactGibbs{S<:AbstractSolver} <: AbstractThermalMethod
    solver::S
end
ExactGibbs(; solver::AbstractSolver=ExactDiagonalization()) = ExactGibbs(solver)

"""
    ThermalTypicality(; samples=16, krylov_dim=64, tol=1e-10, seed=0,
                        matrix_free=true, store_vectors=true,
                        reorthogonalize=true, hbar=1)

Stochastic canonical trace estimator based on random normalized states and Krylov approximations to `exp(-βH/2)|r>`. When `store_vectors=true`, thermally filtered vectors are retained so arbitrary thermal expectation values can be estimated without constructing a density matrix.

`tol` is a Lanczos-breakdown tolerance for the terminal off-diagonal recurrence coefficient. It is not an adaptive thermodynamic or exponential-propagation error tolerance.
"""
struct ThermalTypicality <: AbstractThermalMethod
    samples::Int
    krylov_dim::Int
    tol::Float64
    seed::Int
    matrix_free::Bool
    store_vectors::Bool
    reorthogonalize::Bool
    hbar::Float64
end

function ThermalTypicality(; samples::Integer=16, krylov_dim::Integer=64, tol::Real=1e-10, seed::Integer=0, matrix_free::Bool=true,
                           store_vectors::Bool=true, reorthogonalize::Bool=true, hbar::Real=1.0)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    krylov_dim > 1 || throw(ArgumentError("krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    return ThermalTypicality(Int(samples), Int(krylov_dim), Float64(tol), Int(seed), matrix_free, store_vectors, reorthogonalize, Float64(hbar))
end

"""
    ThermalLanczos(; samples=16, krylov_dim=64, tol=1e-10, seed=0,
                     matrix_free=true, reorthogonalize=false, hbar=1)

Low-memory finite-temperature stochastic Lanczos trace quadrature. It estimates partition functions and thermodynamic moments without retaining thermal vectors. Use `ThermalTypicality` when arbitrary noncommuting observable expectations are required from the same sampled state.

`tol` is a Lanczos-breakdown tolerance, not an adaptive thermodynamic convergence criterion.
"""
struct ThermalLanczos <: AbstractThermalMethod
    samples::Int
    krylov_dim::Int
    tol::Float64
    seed::Int
    matrix_free::Bool
    reorthogonalize::Bool
    hbar::Float64
end

function ThermalLanczos(; samples::Integer=16, krylov_dim::Integer=64, tol::Real=1e-10, seed::Integer=0, matrix_free::Bool=true, reorthogonalize::Bool=false, hbar::Real=1.0)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    krylov_dim > 1 || throw(ArgumentError("krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    return ThermalLanczos(Int(samples), Int(krylov_dim), Float64(tol), Int(seed), matrix_free, reorthogonalize, Float64(hbar))
end

"""
    SectorDecomposition(; symmetry=:total_sz, exploit_equivalent_sectors=true,
                          threaded_matvec=true, rank_lookup=:auto,
                          memory_fraction=0.70, memory_budget_bytes=0)

Exact symmetry decomposition policy for large stochastic thermal calculations. `symmetry` may be a symbolic backend label or a generic `AbstractSector` descriptor. The current thermal backend realizes only `:total_sz` for structurally certified spin-1/2 Hamiltonians and rejects other valid descriptors at execution time rather than at API construction time. All sectors required by the canonical trace remain included; `exploit_equivalent_sectors=true` only pairs sectors when the packed Hamiltonian is certified invariant under global spin inversion.

`threaded_matvec=true` parallelizes the race-free sector gather kernel rather than duplicating large Krylov workspaces across realizations. `rank_lookup` may be `:auto`, `:none`, or `:force`; a dense UInt32 mask-to-sector-rank lookup trades memory for O(1) off-diagonal indexing. In `:auto` mode it is allocated only when the estimated peak working set remains within the configured memory budget. `memory_budget_bytes=0` derives the budget as `memory_fraction * Sys.total_memory()`.
"""
struct SectorDecomposition{S}
    symmetry::S
    exploit_equivalent_sectors::Bool
    threaded_matvec::Bool
    rank_lookup::Symbol
    memory_fraction::Float64
    memory_budget_bytes::Int
end

function SectorDecomposition(; symmetry=:total_sz, exploit_equivalent_sectors::Bool=true, threaded_matvec::Bool=true,
                             rank_lookup::Symbol=:auto, memory_fraction::Real=0.70, memory_budget_bytes::Integer=0)
    symmetry isa Symbol || symmetry isa AbstractSector || throw(ArgumentError("symmetry must be a Symbol or AbstractSector descriptor"))
    rank_lookup in (:auto, :none, :force) || throw(ArgumentError("rank_lookup must be :auto, :none, or :force"))
    0 < memory_fraction <= 1 || throw(ArgumentError("memory_fraction must lie in (0,1]"))
    memory_budget_bytes >= 0 || throw(ArgumentError("memory_budget_bytes must be nonnegative"))
    return SectorDecomposition(symmetry, exploit_equivalent_sectors, threaded_matvec, rank_lookup, Float64(memory_fraction), Int(memory_budget_bytes))
end

"""
    CanonicalTPQ(; samples=16, krylov_dim=64, min_krylov_dim=32,
                   breakdown_tol=1e-10, adaptive_krylov=false,
                   convergence_tol=1e-8, convergence_check_interval=8,
                   convergence_consecutive=2, seed=0, matrix_free=true,
                   store_vectors=false, reorthogonalize=false, hbar=1,
                   propagator=:krylov, parallel=false, progress=false,
                   max_basis_dimension=typemax(Int), decomposition=nothing)

Explicit canonical thermal-pure-quantum-state method. The physical estimator is the canonical TPQ trace estimator; `propagator` identifies the numerical realization of `exp(-βH/2)`. The current production propagator is `:krylov`.

`decomposition=nothing` preserves the validated full-Hilbert-space path. Supplying `SectorDecomposition()` activates exact fixed-total-`Sᶻ` trace decomposition for compatible spin-1/2 Hamiltonians. Sector decomposition currently targets scalar thermodynamic curves and therefore requires `store_vectors=false`.

`breakdown_tol` terminates the Lanczos recurrence only when the Krylov space numerically closes. When `adaptive_krylov=true`, `krylov_dim` is the maximum recurrence dimension and the thermal quadrature is checked every `convergence_check_interval` iterations after `min_krylov_dim`. The recurrence stops after `convergence_consecutive` successive checkpoint changes fall below `convergence_tol` over the complete requested temperature grid. This separates numerical thermal convergence from Lanczos breakdown.

`parallel=true` requests deterministic realization-level parallelism for the full-space path. In a sector-decomposed calculation, threaded sector matvecs take precedence because duplicating the largest Krylov workspace across realizations can exceed memory.
"""
struct CanonicalTPQ <: AbstractThermalMethod
    samples::Int
    krylov_dim::Int
    min_krylov_dim::Int
    breakdown_tol::Float64
    adaptive_krylov::Bool
    convergence_tol::Float64
    convergence_check_interval::Int
    convergence_consecutive::Int
    seed::Int
    matrix_free::Bool
    store_vectors::Bool
    reorthogonalize::Bool
    hbar::Float64
    propagator::Symbol
    parallel::Bool
    progress::Bool
    max_basis_dimension::Int
    decomposition::Union{Nothing,SectorDecomposition}
end

function CanonicalTPQ(; samples::Integer=16, krylov_dim::Integer=64, min_krylov_dim::Integer=min(32, krylov_dim),
                      breakdown_tol::Real=1e-10, adaptive_krylov::Bool=false, convergence_tol::Real=1e-8,
                      convergence_check_interval::Integer=8, convergence_consecutive::Integer=2, seed::Integer=0,
                      matrix_free::Bool=true, store_vectors::Bool=false, reorthogonalize::Bool=false, hbar::Real=1.0,
                      propagator::Symbol=:krylov, parallel::Bool=false, progress::Bool=false, max_basis_dimension::Integer=typemax(Int),
                      decomposition::Union{Nothing,SectorDecomposition}=nothing)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    krylov_dim > 1 || throw(ArgumentError("krylov_dim must exceed one"))
    2 <= min_krylov_dim <= krylov_dim || throw(ArgumentError("min_krylov_dim must lie in 2:krylov_dim"))
    breakdown_tol > 0 || throw(ArgumentError("breakdown_tol must be positive"))
    convergence_tol > 0 || throw(ArgumentError("convergence_tol must be positive"))
    convergence_check_interval > 0 || throw(ArgumentError("convergence_check_interval must be positive"))
    convergence_consecutive > 0 || throw(ArgumentError("convergence_consecutive must be positive"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    propagator === :krylov || throw(ArgumentError("CanonicalTPQ currently supports propagator=:krylov"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    !isnothing(decomposition) && store_vectors && throw(ArgumentError("sector-decomposed CanonicalTPQ currently supports scalar thermal curves only; set store_vectors=false"))
    !isnothing(decomposition) && !matrix_free && throw(ArgumentError("sector-decomposed CanonicalTPQ requires matrix_free=true"))
    return CanonicalTPQ(Int(samples), Int(krylov_dim), Int(min_krylov_dim), Float64(breakdown_tol), adaptive_krylov, Float64(convergence_tol),
                        Int(convergence_check_interval), Int(convergence_consecutive), Int(seed), matrix_free, store_vectors, reorthogonalize,
                        Float64(hbar), propagator, parallel, progress, Int(max_basis_dimension), decomposition)
end

CanonicalTPQ(samples::Int, krylov_dim::Int, breakdown_tol::Float64, seed::Int, matrix_free::Bool, store_vectors::Bool, reorthogonalize::Bool,
             hbar::Float64, propagator::Symbol, parallel::Bool, progress::Bool, max_basis_dimension::Int) =
    CanonicalTPQ(samples, krylov_dim, breakdown_tol, seed, matrix_free, store_vectors, reorthogonalize, hbar, propagator, parallel, progress,
                 max_basis_dimension, nothing)

function CanonicalTPQ(samples::Int, krylov_dim::Int, breakdown_tol::Float64, seed::Int, matrix_free::Bool, store_vectors::Bool,
                      reorthogonalize::Bool, hbar::Float64, propagator::Symbol, parallel::Bool, progress::Bool, max_basis_dimension::Int,
                      decomposition::Union{Nothing,SectorDecomposition})
    return CanonicalTPQ(samples=samples, krylov_dim=krylov_dim, min_krylov_dim=min(32, krylov_dim), breakdown_tol=breakdown_tol,
                        adaptive_krylov=false, convergence_tol=1e-8, convergence_check_interval=8, convergence_consecutive=2, seed=seed,
                        matrix_free=matrix_free, store_vectors=store_vectors, reorthogonalize=reorthogonalize, hbar=hbar, propagator=propagator,
                        parallel=parallel, progress=progress, max_basis_dimension=max_basis_dimension, decomposition=decomposition)
end

"""
    ThermalWorkspace

Reusable Hamiltonian workspace for stochastic thermodynamics. `basis` is `nothing` when an optimized basis-implicit backend can operate directly on packed vector indices. This avoids constructing product-state objects and hash tables for scalar TPQ trace calculations.
"""
struct ThermalWorkspace{M,A,B}
    model::M
    operator::A
    basis::B
    dimension::Int
    backend::Symbol
    hermiticity_error::Float64
    hermitian_certified::Bool
    hbar::Float64
end

"""
    SectorThermalWorkspace

Reusable exact direct-sum workspace for a symmetry-decomposed canonical TPQ calculation. `sectors` contains only the sectors that are actually evaluated. `multiplicities` restores any exactly equivalent sectors omitted by symmetry pairing, and `dimensions` are the Hilbert-space dimensions of the representative sectors. `rank_lookup` is shared by all packed-spin sector operators when the optional O(1) indexing accelerator is enabled.
"""
struct SectorThermalWorkspace{M,D,O,L}
    model::M
    decomposition::D
    sectors::Vector{SpinSector}
    nup::Vector{Int}
    multiplicities::Vector{Int}
    dimensions::Vector{Int}
    operators::O
    rank_lookup::L
    total_dimension::Int
    backend::Symbol
    hbar::Float64
    memory_budget_bytes::Int
    estimated_peak_bytes::Int
    metadata::Dict{Symbol,Any}
end

"""A value with an estimated one-standard-error stochastic uncertainty. `stderr=NaN` means that sample-derived uncertainty is unavailable, for example for one realization."""
struct ThermalEstimate{T}
    value::T
    stderr::Float64
end

Base.show(io::IO, x::ThermalEstimate) = print(io, x.value, " ± ", x.stderr)

"""
Thermodynamic quantities at one temperature.

`logZ` is `NaN` at exactly zero temperature because the absolute partition function is energy-zero dependent in the β→∞ limit. The free energy and all ground-state quantities remain well defined.
"""
struct ThermodynamicPoint
    temperature::Float64
    kB::Float64
    logZ::Float64
    free_energy::Float64
    internal_energy::Float64
    entropy::Float64
    heat_capacity::Float64
    stderr::Dict{Symbol,Float64}
    metadata::Dict{Symbol,Any}
end

"""Thermodynamic quantities evaluated on a temperature grid."""
struct ThermodynamicCurve
    temperatures::Vector{Float64}
    kB::Float64
    logZ::Vector{Float64}
    free_energy::Vector{Float64}
    internal_energy::Vector{Float64}
    entropy::Vector{Float64}
    heat_capacity::Vector{Float64}
    stderr::Dict{Symbol,Vector{Float64}}
    metadata::Dict{Symbol,Any}
end

"""Exact canonical state represented in the solver eigenbasis."""
struct ExactGibbsState{M,S,P} <: AbstractThermalState
    model::M
    solution::S
    temperature::Float64
    kB::Float64
    beta::Float64
    probabilities::P
    ground_degeneracy::Int
    point::ThermodynamicPoint
    metadata::Dict{Symbol,Any}
end

"""
Sampled canonical state.

`filtered_vectors[:,r]` stores a Krylov approximation to `exp[-β(H-Eshift_r)/2]|r>` when vectors are retained. `sample_scales[r]` restores the relative Boltzmann scale between samples without numerical overflow.
"""
struct SampledThermalState{M,B,ME,V} <: AbstractThermalState
    model::M
    basis::B
    method::ME
    temperature::Float64
    kB::Float64
    beta::Float64
    filtered_vectors::V
    sample_norms::Vector{Float64}
    sample_scales::Vector{Float64}
    point::ThermodynamicPoint
    metadata::Dict{Symbol,Any}
end

"""One connected embedded cluster of a finite reference lattice."""
struct ThermalCluster{S,B}
    key::Tuple{Vararg{Int}}
    order::Int
    parent_sites::Vector{Int}
    supercell::S
    bonds::B
end

"""
Result of a finite-reference linked-cluster thermodynamics calculation.

`partial_sums[property][order, iT]` is the per-site linked-cluster sum through the specified order. The calculation becomes an infinite-lattice NLCE only when the reference embedding is sufficiently large that the retained orders are unaffected by its boundary.
"""
struct ClusterThermodynamicsResult{C}
    temperatures::Vector{Float64}
    clusters::C
    raw_curves::Dict{Tuple{Vararg{Int}},ThermodynamicCurve}
    weights::Dict{Symbol,Dict{Tuple{Vararg{Int}},Vector{Float64}}}
    partial_sums::Dict{Symbol,Matrix{Float64}}
    convergence::Dict{Symbol,Matrix{Float64}}
    reference_sites::Int
    metadata::Dict{Symbol,Any}
end
