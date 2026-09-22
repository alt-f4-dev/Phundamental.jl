struct QuadraticFermionSolver <: AbstractModeSolver
    tol::Float64
end
QuadraticFermionSolver(; tol::Real=1e-10) = QuadraticFermionSolver(Float64(tol))
solver_name(::QuadraticFermionSolver) = :quadratic_fermion

"""
    BosonSubspaceProjection(basis; label=:projected, atol=1e-10)

Orthonormal single-particle basis retained by `QuadraticBosonSolver`. If the source boson vector is `b`, the projected coordinates `c` satisfy `b ≈ basis * c`. This is intended for exact symmetry or number-conserving subspace restrictions supplied by the caller, including projection of a known condensate orbital from the Bogoliubov quasiparticle sector.
"""
struct BosonSubspaceProjection
    basis::Matrix{ComplexF64}
    label::Symbol
    atol::Float64
end

function BosonSubspaceProjection(basis::AbstractMatrix; label::Symbol=:projected, atol::Real=1e-10)
    atol >= 0 || throw(ArgumentError("projection tolerance must be nonnegative"))
    Q = Matrix{ComplexF64}(basis)
    size(Q, 1) > 0 || throw(ArgumentError("boson projection basis must contain at least one source mode"))
    size(Q, 2) > 0 || throw(ArgumentError("boson projection basis must retain at least one mode"))
    size(Q, 2) <= size(Q, 1) || throw(DimensionMismatch("boson projection basis cannot contain more columns than rows"))
    gram = adjoint(Q) * Q
    error = norm(gram - I) / max(norm(gram), 1.0)
    error <= max(Float64(atol), 100eps(Float64)) || throw(ArgumentError("boson projection basis must have orthonormal columns; relative Gram error=$error"))
    return BosonSubspaceProjection(Q, label, Float64(atol))
end

"""
    orthogonal_boson_subspace(excluded; label=:number_conserving, atol=1e-10)

Construct an orthonormal complement to one or more excluded source-mode vectors. For a Bose condensate, passing the condensate orbital implements the single-particle projector `Q = I - |phi><phi|` underlying number-conserving Bogoliubov formulations. The returned object can be supplied as `projection=` to `QuadraticBosonSolver`.
"""
function orthogonal_boson_subspace(excluded::AbstractVecOrMat; label::Symbol=:number_conserving, atol::Real=1e-10)
    atol >= 0 || throw(ArgumentError("projection tolerance must be nonnegative"))
    X = excluded isa AbstractVector ? reshape(ComplexF64.(excluded), :, 1) : Matrix{ComplexF64}(excluded)
    M, nexcluded = size(X)
    M > 1 || throw(ArgumentError("orthogonal boson subspace requires at least two source modes"))
    nexcluded > 0 || throw(ArgumentError("at least one excluded mode vector is required"))

    F = svd(X; full=true)
    scale = isempty(F.S) ? 1.0 : max(maximum(F.S), 1.0)
    rank = count(s -> s > max(Float64(atol), 100eps(Float64)) * scale, F.S)
    rank > 0 || throw(ArgumentError("excluded boson subspace has zero numerical rank"))
    rank < M || throw(ArgumentError("excluded boson subspace spans the complete source space"))

    retained = Matrix{ComplexF64}(F.U[:, rank+1:M])
    return BosonSubspaceProjection(retained; label=label, atol=atol)
end

struct QuadraticBosonSolver <: AbstractModeSolver
    tol::Float64
    projection::Union{Nothing,BosonSubspaceProjection}
end

QuadraticBosonSolver(tol::Real) = QuadraticBosonSolver(Float64(tol), nothing)

function QuadraticBosonSolver(; tol::Real=1e-10, projection::Union{Nothing,BosonSubspaceProjection}=nothing)
    tol >= 0 || throw(ArgumentError("QuadraticBosonSolver tolerance must be nonnegative"))
    return QuadraticBosonSolver(Float64(tol), projection)
end

solver_name(::QuadraticBosonSolver) = :quadratic_boson

"""
    HarmonicPhononSolver(; tol=1e-10, zero_mode_factor=256.0)

Harmonic normal-mode solver with an eigenvalue-space numerical-zero tolerance `zero_mode_factor * eps(Float64) * scale`, where `scale` is the largest absolute dynamical-matrix eigenvalue or one, whichever is larger.
"""
struct HarmonicPhononSolver <: AbstractModeSolver
    tol::Float64
    zero_mode_factor::Float64
end

function HarmonicPhononSolver(; tol::Real=1e-10, zero_mode_factor::Real=256.0)
    tol >= 0 || throw(ArgumentError("HarmonicPhononSolver tolerance must be nonnegative"))
    zero_mode_factor >= 0 || throw(ArgumentError("zero_mode_factor must be nonnegative"))
    return HarmonicPhononSolver(Float64(tol), Float64(zero_mode_factor))
end
HarmonicPhononSolver(tol::Real) = HarmonicPhononSolver(tol=tol)
solver_name(::HarmonicPhononSolver) = :harmonic_phonon

"""
    LinearSpinWaveSolver(; reference_signs=nothing, tol=1e-10)

Quadratic Holstein-Primakoff expansion around a collinear ±z reference state.
`reference_signs[i]` is +1 or -1. If omitted, all spins are aligned +z.

This is an approximation and is marked as such in the returned metadata.
"""
struct LinearSpinWaveSolver <: AbstractModeSolver
    reference_signs::Union{Nothing,Vector{Int}}
    tol::Float64
end
function LinearSpinWaveSolver(; reference_signs=nothing, tol::Real=1e-10)
    signs = isnothing(reference_signs) ? nothing : Int.(collect(reference_signs))
    if !isnothing(signs) && !all(s -> s in (-1, 1), signs)
        throw(ArgumentError("reference_signs must contain only ±1"))
    end
    return LinearSpinWaveSolver(signs, Float64(tol))
end
solver_name(::LinearSpinWaveSolver) = :linear_spin_wave
