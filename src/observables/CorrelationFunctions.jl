# Thermal weights and time-domain correlation functions.

function _real_energies(result::SolverResult; atol::Real=1e-10)
    E = result.energies
    scale = max(isempty(E) ? 0.0 : maximum(abs.(E)), 1.0)
    imagmax = isempty(E) ? 0.0 : maximum(abs.(imag.(E)))
    imagmax <= atol * scale ||
        throw(ArgumentError("thermal/spectral observables require an effectively real eigenspectrum"))
    return Float64.(real.(E))
end

"""
    thermal_probabilities(result; temperature=0, kB=1, degeneracy_tol=1e-10)

Stable canonical probabilities.  Energies are shifted by the ground energy
before exponentiation, so large absolute energy offsets do not cause overflow.
At exactly zero temperature, exactly-degenerate ground states (within
`degeneracy_tol`) are assigned equal statistical weight.
"""
function thermal_probabilities(
    result::SolverResult;
    temperature::Real=0.0,
    kB::Real=1.0,
    degeneracy_tol::Real=1e-10,
)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    E = _real_energies(result)
    Emin = minimum(E)

    if iszero(temperature)
        scale = max(isempty(E) ? 0.0 : maximum(abs.(E)), 1.0)
        ground = findall(e -> abs(e - Emin) <= degeneracy_tol * scale, E)
        p = zeros(Float64, length(E))
        p[ground] .= inv(length(ground))
        return p
    end

    β = inv(float(kB) * float(temperature))
    shifted = E .- Emin
    logw = .-β .* shifted
    # logw <= 0 by construction; exp is stable and underflow is harmless.
    w = exp.(logw)
    Zshift = sum(w)
    isfinite(Zshift) && Zshift > 0 ||
        throw(ArgumentError("failed to construct finite thermal partition weights"))
    return w ./ Zshift
end

function _validate_model_result(model::ManyBodyModel, result::SolverResult)
    result.basis.representation.name == model.representation.name || throw(ArgumentError(
        "solver result representation $(result.basis.representation.name) does not match observable model representation $(model.representation.name)"
    ))
    size(result.right_states, 1) == length(result.basis) ||
        throw(DimensionMismatch("solver right-state dimension does not match computational basis"))
    size(result.left_states, 1) == length(result.basis) ||
        throw(DimensionMismatch("solver left-state dimension does not match computational basis"))
    return nothing
end

function _operator_matrix(model::ManyBodyModel, result::SolverResult, op; hbar::Real=1.0)
    _validate_model_result(model, result)
    resolved = op isa Symbol || op isa ObservableRef ? resolve_observable(model, op) : op
    resolved isa AbstractOperatorExpr || resolved isa AbstractMatrix ||
        throw(ArgumentError("observable must resolve to an operator expression or matrix"))
    return matrix(resolved, result.basis; sparse=false, hbar=hbar)
end

"""Matrix elements `<L_m|O|R_n>` in the solver eigenbasis."""
function eigenbasis_matrix(
    model::ManyBodyModel,
    result::SolverResult,
    op;
    hbar::Real=1.0,
)
    O = _operator_matrix(model, result, op; hbar=hbar)
    return adjoint(result.left_states) * O * result.right_states
end

"""
    correlation_function(model, result, A, B, times; ...)

Calculate

    C_AB(t) = Tr[rho A(t) B(0)]

from the Lehmann representation.  For a Hermitian system this is

    sum_m p_m sum_n exp[-i(E_n-E_m)t/hbar] A_mn B_nm.

For similarity-transformed/biorthogonal results, left/right matrix elements are
used consistently.  This routine assumes an effectively real spectrum.
"""
function correlation_function(
    model::ManyBodyModel,
    result::SolverResult,
    A,
    B,
    times;
    temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(),
    degeneracy_tol::Real=1e-10,
    weight_tol::Real=0.0,
)
    E = _real_energies(result)
    p = thermal_probabilities(
        result;
        temperature=temperature,
        kB=convention.kB,
        degeneracy_tol=degeneracy_tol,
    )
    Ae = eigenbasis_matrix(model, result, A; hbar=convention.hbar)
    Be = eigenbasis_matrix(model, result, B; hbar=convention.hbar)

    t = collect(float.(times))
    values = zeros(ComplexF64, length(t))
    invh = inv(float(convention.hbar))

    @inbounds for m in eachindex(E)
        pm = p[m]
        pm <= weight_tol && continue
        for n in eachindex(E)
            wmn = pm * Ae[m, n] * Be[n, m]
            abs(wmn) <= weight_tol && continue
            Δ = (E[n] - E[m]) * invh
            values .+= wmn .* exp.(-1im .* Δ .* t)
        end
    end

    metadata = Dict{Symbol,Any}(
        :temperature => float(temperature),
        :kB => convention.kB,
        :hbar => convention.hbar,
        :definition => :thermal_two_point,
        :correlator => :A_t_B_0,
        :biorthogonal => isbiorthogonal(result),
        :ground_state_degeneracy_averaged => iszero(temperature),
    )
    return CorrelationResult(t, values, metadata)
end

function correlation_function(model, result::SolverResult, A, times; kwargs...)
    isbiorthogonal(result) && throw(ArgumentError(
        "for a biorthogonal/nonunitary representation, supply both A and the physically transformed partner B explicitly; ordinary adjoint(A) need not be the physical adjoint"
    ))
    Aresolved = resolve_observable(model, A)
    return correlation_function(model, result, Aresolved, adjoint(Aresolved), times; kwargs...)
end
