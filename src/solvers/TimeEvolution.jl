# Exact spectral time evolution for finite computational bases.

struct TimeEvolutionSolver{T<:Real} <: AbstractSolver
    times::Vector{T}
    hbar::T
    diagonalizer::ExactDiagonalization
end

function TimeEvolutionSolver(
    times::AbstractVector{T};
    hbar::Real=1,
    diagonalizer::ExactDiagonalization=ExactDiagonalization(),
) where {T<:Real}
    isempty(times) && throw(ArgumentError("at least one evolution time is required"))
    TT = promote_type(T, typeof(float(hbar)))
    return TimeEvolutionSolver{TT}(TT.(times), TT(hbar), diagonalizer)
end

solver_name(::TimeEvolutionSolver) = :exact_time_evolution

function _initial_vector(initial, basis::ComputationalBasis)
    dim = length(basis)
    if initial isa Integer
        1 <= initial <= dim || throw(BoundsError(basis.states, initial))
        ψ = zeros(ComplexF64, dim)
        ψ[Int(initial)] = 1
        return ψ
    elseif initial isa AbstractComputationalState
        idx = get(basis.index, initial, 0)
        idx == 0 && throw(ArgumentError("initial computational state is not in the basis"))
        ψ = zeros(ComplexF64, dim)
        ψ[idx] = 1
        return ψ
    elseif initial isa AbstractVector
        length(initial) == dim || throw(DimensionMismatch("initial vector length does not match basis"))
        ψ = ComplexF64.(initial)
        norm(ψ) > 0 || throw(ArgumentError("initial state has zero norm"))
        # Preserve the supplied normalization. This is particularly important
        # for nonunitary/biorthogonal representations, where Euclidean
        # normalization need not be the physical normalization.
        return ψ
    end
    throw(ArgumentError("initial state must be a basis index, computational state, or vector"))
end

function solve(solver::TimeEvolutionSolver, model::ManyBodyModel, initial)
    # Respect the requested hbar even if the stored ED object was constructed
    # with another value.
    ed = ExactDiagonalization(
        hermitian=solver.diagonalizer.hermitian,
        check_physical_closure=solver.diagonalizer.check_physical_closure,
        closure_tol=solver.diagonalizer.closure_tol,
        hbar=solver.hbar,
    )
    spectrum = solve(ed, model)
    ψ0 = _initial_vector(initial, spectrum.basis)

    R = spectrum.right_states
    L = spectrum.left_states
    coefficients = adjoint(L) * ψ0
    nt = length(solver.times)
    evolved = Matrix{ComplexF64}(undef, length(ψ0), nt)

    for (j, t) in enumerate(solver.times)
        phase = exp.((-im * t / solver.hbar) .* spectrum.energies)
        evolved[:, j] = R * (phase .* coefficients)
    end

    metadata = Dict{Symbol,Any}(
        :solver => :exact_time_evolution,
        :hbar => solver.hbar,
        :spectral_solver => spectrum,
        :biorthogonal => isbiorthogonal(spectrum),
    )
    return TimeEvolutionResult(solver, model, solver.times, evolved, spectrum.basis, metadata)
end

"""Convenience wrapper for exact spectral propagation."""
function evolve(
    model::ManyBodyModel,
    initial,
    times::AbstractVector{<:Real};
    hbar::Real=1,
    diagonalizer::ExactDiagonalization=ExactDiagonalization(),
)
    return solve(TimeEvolutionSolver(times; hbar=hbar, diagonalizer=diagonalizer), model, initial)
end
