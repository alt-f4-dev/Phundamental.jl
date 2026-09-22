# Representation-specific numerical backend descriptors and solver/result types.
#
# These are numerical encodings. They do NOT change the physical/algebraic
# Representation carried by ManyBodyModel.

abstract type AbstractOptimizedBackend end
abstract type AbstractSector end
abstract type AbstractModeSolver <: AbstractSolver end
abstract type AbstractModeResult end

struct PackedSpinBackend <: AbstractOptimizedBackend
    max_sites::Int
end
PackedSpinBackend(; max_sites::Integer=64) = PackedSpinBackend(Int(max_sites))

struct PackedFermionBackend <: AbstractOptimizedBackend
    max_modes::Int
end
PackedFermionBackend(; max_modes::Integer=64) = PackedFermionBackend(Int(max_modes))

struct PackedBosonBackend <: AbstractOptimizedBackend
    max_dimension::Int
end
PackedBosonBackend(; max_dimension::Integer=10_000_000) =
    PackedBosonBackend(Int(max_dimension))

struct QuadraticFermionBackend <: AbstractOptimizedBackend end
struct QuadraticBosonBackend <: AbstractOptimizedBackend end
struct HarmonicPhononBackend <: AbstractOptimizedBackend end
struct LinearSpinWaveBackend <: AbstractOptimizedBackend end

"""
    OptimizedLanczos(; nev=6, krylov_dim=64, tol=1e-10, sector=nothing,
                       backend=:auto, reorthogonalize=true, seed=0, hbar=1,
                       fallback=true, max_basis_dimension=50_000_000)

Lanczos using representation-specific packed kernels. The physical model
representation is preserved. If `sector` is supplied, the computational basis
is restricted before Krylov iteration.
"""
struct OptimizedLanczos <: AbstractSolver
    nev::Int
    krylov_dim::Int
    tol::Float64
    sector::Union{Nothing,AbstractSector}
    backend::Symbol
    reorthogonalize::Bool
    seed::Int
    hbar::Float64
    fallback::Bool
    max_basis_dimension::Int
end

function OptimizedLanczos(
    ; nev::Integer=6,
      krylov_dim::Integer=64,
      tol::Real=1e-10,
      sector::Union{Nothing,AbstractSector}=nothing,
      backend::Symbol=:auto,
      reorthogonalize::Bool=true,
      seed::Integer=0,
      hbar::Real=1.0,
      fallback::Bool=true,
      max_basis_dimension::Integer=50_000_000,
)
    nev > 0 || throw(ArgumentError("nev must be positive"))
    krylov_dim > 1 || throw(ArgumentError("krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    backend in (:auto, :spin, :fermion, :boson) ||
        throw(ArgumentError("backend must be :auto, :spin, :fermion, or :boson"))
    return OptimizedLanczos(
        Int(nev), Int(krylov_dim), Float64(tol), sector, backend,
        reorthogonalize, Int(seed), Float64(hbar), fallback,
        Int(max_basis_dimension),
    )
end
solver_name(::OptimizedLanczos) = :optimized_lanczos

"""
    AutoSolver(; ...)

Prefer specialized mode solvers when the model is recognized as harmonic or
quadratic; otherwise use `OptimizedLanczos`, with optional fallback to the
generic Lanczos implementation.
"""
struct AutoSolver <: AbstractSolver
    nev::Int
    krylov_dim::Int
    tol::Float64
    sector::Union{Nothing,AbstractSector}
    seed::Int
    hbar::Float64
    fallback::Bool
end

function AutoSolver(
    ; nev::Integer=6,
      krylov_dim::Integer=64,
      tol::Real=1e-10,
      sector::Union{Nothing,AbstractSector}=nothing,
      seed::Integer=0,
      hbar::Real=1.0,
      fallback::Bool=true,
)
    return AutoSolver(
        Int(nev), Int(krylov_dim), Float64(tol), sector,
        Int(seed), Float64(hbar), fallback,
    )
end
solver_name(::AutoSolver) = :auto_solver

"""Single-particle/BdG mode result. Not a full many-body eigensystem."""
struct QuadraticModeResult{S,M,E,V,D} <: AbstractModeResult
    solver::S
    model::M
    energies::E
    modes::V
    metadata::D
end

"""Harmonic normal-mode result with mass-weighted eigenvectors."""
struct PhononModeResult{S,M,E,V,D} <: AbstractModeResult
    solver::S
    model::M
    frequencies::E
    modes::V
    metadata::D
end

"""Wavevector-resolved harmonic phonon dispersion for a periodic primitive-cell model."""
struct PhononDispersionResult{S,M,Q,E,V,D} <: AbstractModeResult
    solver::S
    model::M
    qpoints::Q
    frequencies::E
    modes::V
    metadata::D
end

"""Linear-spin-wave result. This is an approximation, not an exact basis map."""
struct SpinWaveResult{S,M,E,V,A,B,D} <: AbstractModeResult
    solver::S
    model::M
    frequencies::E
    modes::V
    A::A
    B::B
    classical_energy::Float64
    metadata::D
end
