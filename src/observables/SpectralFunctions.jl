# General spectral-density helpers built on the Lehmann engine.

"""
    spectral_density(model, result, A, B, axis; ...)

General two-operator Lehmann spectral density without a momentum-field layer. This is the reusable backend for local spectra, transformed quasiparticle operators, density spectra, current spectra, and Green-function-facing interfaces.
"""
function spectral_density(model::ManyBodyModel, result::SolverResult, A, B, axis; temperature::Real=0.0,
                          convention::SpectrumConvention=SpectrumConvention(), broadening::AbstractBroadening=LorentzianBroadening(0.05),
                          degeneracy_tol::Real=1e-10, weight_tol::Real=0.0)
    lines = lehmann_lines(model, result, A, B; temperature=temperature, convention=convention,
                          degeneracy_tol=degeneracy_tol, weight_tol=weight_tol)
    x = _check_axis_grid(axis)
    values = broaden_lines(lines, x, broadening)
    metadata = copy(lines.metadata)
    metadata[:observable] = :spectral_density
    metadata[:broadening] = broadening
    metadata[:broadening_role] = :intrinsic_numerical_line_shape
    metadata[:resolution_applied] = false
    return SpectrumResult([nothing], x, reshape(values, 1, length(values)), metadata)
end

function spectral_density(model::ManyBodyModel, result::QuadraticModeResult, A, B, axis; temperature::Real=0.0,
                          convention::SpectrumConvention=SpectrumConvention(), broadening::AbstractBroadening=LorentzianBroadening(0.05),
                          degeneracy_tol::Real=1e-10, weight_tol::Real=0.0,
                          zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes())
    lines = lehmann_lines(model, result, A, B; temperature=temperature, convention=convention,
                          degeneracy_tol=degeneracy_tol, weight_tol=weight_tol, zero_mode_policy=zero_mode_policy)
    x = _check_axis_grid(axis)
    values = broaden_lines(lines, x, broadening)
    metadata = copy(lines.metadata)
    metadata[:observable] = :spectral_density
    metadata[:broadening] = broadening
    metadata[:broadening_role] = :intrinsic_numerical_line_shape
    metadata[:resolution_applied] = false
    return SpectrumResult([nothing], x, reshape(values, 1, length(values)), metadata)
end

function spectral_density(model, result::SolverResult, A, axis; kwargs...)
    isbiorthogonal(result) && throw(ArgumentError(
        "for a biorthogonal/nonunitary representation, supply both A and the physically transformed partner B explicitly; " *
        "ordinary adjoint(A) need not be the physical adjoint"
    ))
    Aresolved = resolve_observable(model, A)
    return spectral_density(model, result, Aresolved, adjoint(Aresolved), axis; kwargs...)
end

function spectral_density(model, result::QuadraticModeResult, A, axis; kwargs...)
    Aresolved = resolve_observable(model, A)
    return spectral_density(model, result, Aresolved, adjoint(Aresolved), axis; kwargs...)
end

"""
Check the thermal detailed-balance relation for an autocorrelation spectrum:

    S(-x) = exp(-beta*x_E) S(+x)

where `x_E=x` for an energy-transfer axis and `x_E=hbar*x` for an angular-frequency axis. The function returns a relative RMS residual over mirrored axis points and is intended primarily as a validation diagnostic.
"""
function detailed_balance_residual(spectrum::SpectrumResult; temperature::Real,
                                   convention::SpectrumConvention=SpectrumConvention(), atol::Real=1e-12)
    temperature > 0 || throw(ArgumentError("detailed balance requires positive temperature"))
    x = spectrum.axis
    size(spectrum.intensity, 1) == 1 || throw(ArgumentError("detailed_balance_residual currently expects a single spectrum"))
    y = vec(spectrum.intensity)
    β = inv(convention.kB * temperature)

    residuals = Float64[]
    for i in eachindex(x)
        j = argmin(abs.(x .+ x[i]))
        abs(x[j] + x[i]) <= max(atol, 10eps(Float64) * max(abs(x[i]), 1.0)) || continue
        xE = convention.spectral_axis === :energy ? x[i] : convention.hbar * x[i]
        lhs = y[j]
        rhs = exp(-β * xE) * y[i]
        denom = max(abs(lhs), abs(rhs), atol)
        push!(residuals, abs(lhs - rhs) / denom)
    end
    isempty(residuals) && return NaN
    return sqrt(sum(abs2, residuals) / length(residuals))
end
