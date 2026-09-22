# Exact canonical ensembles and stable thermodynamic moments.

_result_dimension(result::SolverResult) = length(result.basis)
_result_dimension(result::EnergySpectrumResult) = result.dimension

function _thermal_real_energies(result::Union{SolverResult,EnergySpectrumResult}; atol::Real=1e-10)
    E = result.energies
    isempty(E) && throw(ArgumentError("spectrum contains no energies"))
    scale = max(maximum(abs.(E)), 1.0)
    imagmax = maximum(abs.(imag.(E)))
    imagmax <= atol * scale || throw(ArgumentError("canonical thermodynamics requires an effectively real eigenspectrum; maximum imaginary component is $imagmax"))
    return Float64.(real.(E))
end

function _require_complete_energy_spectrum(result::Union{SolverResult,EnergySpectrumResult})
    dim = _result_dimension(result)
    length(result.energies) == dim || throw(ArgumentError("ExactGibbs requires a complete energy spectrum: solver returned $(length(result.energies)) energies for Hilbert-space dimension $dim. Use CanonicalTPQ, ThermalLanczos, or ThermalTypicality for stochastic calculations."))
    return dim
end

function _require_complete_eigensystem(result::SolverResult)
    dim = _require_complete_energy_spectrum(result)
    size(result.right_states, 1) == dim && size(result.right_states, 2) == dim || throw(DimensionMismatch("right eigensystem is not complete"))
    size(result.left_states, 1) == dim && size(result.left_states, 2) == dim || throw(DimensionMismatch("left eigensystem is not complete"))
    return dim
end

function _exact_ground_probabilities(E::AbstractVector, degeneracy_tol::Real)
    Emin = minimum(E)
    scale = max(maximum(abs.(E)), 1.0)
    ground = findall(e -> abs(e - Emin) <= degeneracy_tol * scale, E)
    p = zeros(Float64, length(E))
    p[ground] .= inv(length(ground))
    return p, length(ground), Emin
end

function _exact_thermodynamic_point(E::AbstractVector, temperature::Real, kB::Real; degeneracy_tol::Real=1e-10)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    kB > 0 || throw(ArgumentError("kB must be positive"))

    T = Float64(temperature)
    kb = Float64(kB)

    if iszero(T)
        p, g, E0 = _exact_ground_probabilities(E, degeneracy_tol)
        S0 = kb * log(g)
        point = ThermodynamicPoint(
            T, kb, NaN, E0, E0, S0, 0.0,
            Dict{Symbol,Float64}(:free_energy => 0.0, :internal_energy => 0.0, :entropy => 0.0, :heat_capacity => 0.0, :logZ => NaN),
            Dict{Symbol,Any}(:method => :exact_gibbs, :ground_state_degeneracy => g, :zero_temperature_limit => true),
        )
        return point, p, g
    end

    β = inv(kb * T)
    Emin = minimum(E)
    shifted = E .- Emin
    w = exp.(-β .* shifted)
    zshift = sum(w)
    isfinite(zshift) && zshift > 0 || throw(ArgumentError("failed to construct finite canonical weights"))
    p = w ./ zshift

    logZ = -β * Emin + log(zshift)
    U = dot(p, E)
    E2 = dot(p, E .* E)
    varE = max(E2 - U^2, 0.0)
    F = -kb * T * logZ
    S = (U - F) / T
    Cv = varE / (kb * T^2)

    point = ThermodynamicPoint(
        T, kb, logZ, F, U, S, Cv,
        Dict{Symbol,Float64}(:free_energy => 0.0, :internal_energy => 0.0, :entropy => 0.0, :heat_capacity => 0.0, :logZ => 0.0),
        Dict{Symbol,Any}(:method => :exact_gibbs, :energy_variance => varE, :zero_temperature_limit => false),
    )
    return point, p, 0
end

"""
    thermal_state(result; temperature, kB=1, degeneracy_tol=1e-10)

Build an exact canonical Gibbs state from a complete `SolverResult`. A full eigensystem is required because the returned state retains the eigenvectors for subsequent observable calculations.
"""
function thermal_state(result::SolverResult; temperature::Real, kB::Real=1.0, degeneracy_tol::Real=1e-10)
    dim = _require_complete_eigensystem(result)
    E = _thermal_real_energies(result)
    point, p, g = _exact_thermodynamic_point(E, temperature, kB; degeneracy_tol=degeneracy_tol)
    β = iszero(temperature) ? Inf : inv(Float64(kB) * Float64(temperature))
    return ExactGibbsState(
        result.model,
        result,
        Float64(temperature),
        Float64(kB),
        β,
        p,
        g,
        point,
        Dict{Symbol,Any}(:method => :exact_gibbs, :solver => get(result.metadata, :solver, :unknown), :dimension => dim, :biorthogonal => isbiorthogonal(result)),
    )
end

"""
    thermal_state(model; temperature, method=ExactGibbs(), kB=1)

Construct a thermal state from a model. Method-specific overloads for stochastic Krylov sampling are defined in `ThermalSampling.jl`.
"""
function thermal_state(model::ManyBodyModel, method::ExactGibbs; temperature::Real, kB::Real=1.0, degeneracy_tol::Real=1e-10)
    result = solve(method.solver, model)
    return thermal_state(result; temperature=temperature, kB=kB, degeneracy_tol=degeneracy_tol)
end

function thermal_state(model::ManyBodyModel; temperature::Real, method::AbstractThermalMethod=ExactGibbs(), kwargs...)
    return thermal_state(model, method; temperature=temperature, kwargs...)
end

function _curve_from_points(points::Vector{ThermodynamicPoint}; metadata=Dict{Symbol,Any}())
    isempty(points) && throw(ArgumentError("thermodynamic curve requires at least one point"))
    Ts = [p.temperature for p in points]
    kb = points[1].kB
    all(p -> p.kB == kb, points) || throw(ArgumentError("all points must use the same kB"))
    keys_stderr = (:free_energy, :internal_energy, :entropy, :heat_capacity, :logZ)
    stderr = Dict{Symbol,Vector{Float64}}(key => [get(p.stderr, key, NaN) for p in points] for key in keys_stderr)
    return ThermodynamicCurve(Ts, kb, [p.logZ for p in points], [p.free_energy for p in points], [p.internal_energy for p in points], [p.entropy for p in points], [p.heat_capacity for p in points], stderr, Dict{Symbol,Any}(metadata))
end

"""
    thermal_curve(result, temperatures; kB=1)

Evaluate exact thermodynamics over a temperature grid while reusing one complete energy spectrum. `EnergySpectrumResult` is sufficient because canonical scalar thermodynamics does not require eigenvectors.
"""
function thermal_curve(result::Union{SolverResult,EnergySpectrumResult}, temperatures; kB::Real=1.0, degeneracy_tol::Real=1e-10)
    dim = _require_complete_energy_spectrum(result)
    E = _thermal_real_energies(result)
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperature grid cannot be empty"))
    all(>=(0), Ts) || throw(ArgumentError("temperatures must be nonnegative"))

    points = ThermodynamicPoint[]
    sizehint!(points, length(Ts))
    for T in Ts
        point, _, _ = _exact_thermodynamic_point(E, T, kB; degeneracy_tol=degeneracy_tol)
        push!(points, point)
    end
    return _curve_from_points(points; metadata=Dict{Symbol,Any}(
        :method => :exact_gibbs,
        :solver => get(result.metadata, :solver, :unknown),
        :dimension => dim,
        :spectrum_reused => true,
        :eigenvectors_required => false,
    ))
end

_exact_curve_solution(solver::ExactDiagonalization, model::ManyBodyModel) = energy_spectrum(solver, model)
_exact_curve_solution(solver::AbstractSolver, model::ManyBodyModel) = solve(solver, model)

function thermal_curve(model::ManyBodyModel, method::ExactGibbs, temperatures; kB::Real=1.0, degeneracy_tol::Real=1e-10)
    result = _exact_curve_solution(method.solver, model)
    return thermal_curve(result, temperatures; kB=kB, degeneracy_tol=degeneracy_tol)
end

function thermal_curve(model::ManyBodyModel, temperatures; method::AbstractThermalMethod=ExactGibbs(), kwargs...)
    return thermal_curve(model, method, temperatures; kwargs...)
end
