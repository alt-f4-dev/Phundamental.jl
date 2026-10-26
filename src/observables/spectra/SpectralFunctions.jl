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
    metadata[:response_kind] = :spectral_density
    metadata[:frequency_domain] = :real
    metadata[:statistics] = :unspecified
    metadata[:momentum_scope] = :local
    metadata[:response_order] = 2
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
    metadata[:response_kind] = :spectral_density
    metadata[:frequency_domain] = :real
    metadata[:statistics] = :unspecified
    metadata[:momentum_scope] = :local
    metadata[:response_order] = 2
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

# Spin-resolved one-particle spectra reconstructed from finite-cluster Green matrices.

function _electron_spectral_intensity(value::Complex, normalization::Symbol)
    if normalization === :spectral_density || normalization === :minus_im_over_pi
        return -imag(value) / pi
    elseif normalization === :senechal
        return -2 * imag(value)
    end
    throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
end

"""
    cluster_perturbation_spectral_function(workspace, momenta, axis; eta, t_intercluster, ...)

Evaluate the spin-resolved one-particle spectral function of a one-dimensional cluster perturbation theory reconstruction. The default normalization is `-Im G/pi`, which integrates to unity per spin and momentum when the complete frequency axis is retained. Set `normalization=:senechal` to reproduce the `-2 Im G` convention used by Senechal et al.; set `krylov_dim` to evaluate a shorter prefix of the projected-Lanczos chains already stored in the workspace.
"""
function cluster_perturbation_spectral_function(
    workspace::FermionGreenWorkspace,
    momenta,
    axis;
    eta::Real,
    t_intercluster::Number,
    cluster_translation::Real=length(workspace.sites),
    positions=nothing,
    normalization::Symbol=:spectral_density,
    parallel::Bool=true,
    krylov_dim=nothing,
)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    k_values = Float64.(collect(momenta))
    frequency_axis = Float64.(collect(axis))
    isempty(k_values) && throw(ArgumentError("momenta must not be empty"))
    isempty(frequency_axis) && throw(ArgumentError("axis must not be empty"))
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) || throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))

    cluster_green = cluster_green_matrix(workspace, frequency_axis; eta=eta, parallel=parallel, krylov_dim=krylov_dim)
    intensity = zeros(Float64, length(k_values), length(frequency_axis))

    worker = function (frequency_index)
        Gc = view(cluster_green, :, :, frequency_index)
        for momentum_index in eachindex(k_values)
            Gk = cluster_perturbation_green(
                Gc,
                k_values[momentum_index];
                t_intercluster=t_intercluster,
                cluster_translation=cluster_translation,
                positions=positions,
            )
            intensity[momentum_index, frequency_index] = _electron_spectral_intensity(Gk, normalization)
        end
    end

    if parallel && Threads.nthreads() > 1 && length(frequency_axis) > 1
        Threads.@threads :static for frequency_index in eachindex(frequency_axis)
            worker(frequency_index)
        end
    else
        for frequency_index in eachindex(frequency_axis)
            worker(frequency_index)
        end
    end

    metadata = Dict{Symbol,Any}(
        :observable => :cluster_perturbation_spectral_function,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :correlation_scope => :short_intermediate_spatial,
        :eta => Float64(eta),
        :normalization => normalization,
        :spin => workspace.spin,
        :ground_sector => workspace.ground_sector,
        :particle_sector => workspace.particle_sector,
        :hole_sector => workspace.hole_sector,
        :t_intercluster => t_intercluster,
        :cluster_translation => Float64(cluster_translation),
        :cpt_equation => :senechal_eqs_6_and_9,
        :krylov_dim => krylov_dim,
        :resolution_applied => false,
    )
    return SpectrumResult(k_values, frequency_axis, intensity, metadata)
end

const _CompositeElectronRepresentation = Representation{S,A,B,P,G,R,O,T} where {S<:AbstractStateSpace,A<:CompositeAlgebra,B<:AbstractBasis,P,G,R,O,T}

"""Evaluate the existing local electronic spectral request for a composite representation by delegating conserved fermion-sector handling to `fermion_green_workspace`."""
function (request::LocalElectronSpectralFunctionRequest)(model::ManyBodyModel{R}, solution::SolverResult) where {R<:_CompositeElectronRepresentation}
    workspace = fermion_green_workspace(
        model,
        solution;
        spin=request.spin,
        green_krylov_dim=request.green_krylov_dim,
        tol=request.tol,
        green_reorthogonalize=request.green_reorthogonalize,
        parallel=request.parallel,
        max_basis_dimension=request.max_basis_dimension,
    )

    selected_sites = isnothing(request.sites) ? copy(workspace.sites) : request.sites
    site_rows = Int[]
    sizehint!(site_rows, length(selected_sites))
    for site in selected_sites
        row = findfirst(==(site), workspace.sites)
        isnothing(row) && throw(ArgumentError("site $site has no $(request.spin) fermion mode in the model"))
        push!(site_rows, row)
    end

    green = cluster_green_matrix(workspace, request.axis; eta=request.eta, parallel=request.parallel)
    intensity = zeros(Float64, length(selected_sites), length(request.axis))
    for (output_row, green_row) in enumerate(site_rows), axis_index in eachindex(request.axis)
        intensity[output_row, axis_index] = _electron_spectral_intensity(green[green_row, green_row, axis_index], request.normalization)
    end

    metadata = Dict{Symbol,Any}(
        :observable => :local_electron_spectral_function,
        :observable_request => :local_electron_spectral_function,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :local,
        :response_order => 2,
        :spin => request.spin,
        :eta => request.eta,
        :normalization => request.normalization,
        :ground_sector => workspace.ground_sector,
        :particle_sector => workspace.particle_sector,
        :hole_sector => workspace.hole_sector,
        :ground_solution_reused => get(workspace.metadata, :ground_reused, false),
        :workspace_metadata => copy(workspace.metadata),
        :resolution_applied => false,
        :request_metadata => copy(request.metadata),
    )
    return LocalSpectrumResult(copy(selected_sites), copy(request.axis), intensity, metadata)
end

# Explicit conversions between causal real-frequency response functions and physical spectral observables.

function _retarded_green_normalization(normalization::Symbol)
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) ||
        throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
    return normalization
end

function _retarded_green_intensity(values, normalization::Symbol)
    _retarded_green_normalization(normalization)
    if normalization === :senechal
        return -2 .* imag.(values)
    end
    return -imag.(values) ./ pi
end

function _response_metadata_copy(metadata)
    return metadata isa AbstractDict ? Dict{Symbol,Any}(metadata) : Dict{Symbol,Any}(pairs(metadata))
end

"""
    spectral_function_from_retarded_green(axis, values; ...)
    spectral_function_from_retarded_green(qgrid, axis, values; ...)

Convert a scalar retarded Green function into a real-frequency spectral function. The local form accepts one complex value per spectral point. The momentum-resolved form accepts a matrix indexed `(iq, ix)`.

For `normalization=:spectral_density` or `:minus_im_over_pi`, the conversion is `A = -Im(G^R)/pi`. `normalization=:senechal` returns the `-2 Im(G^R)` convention used by the CPT benchmark API. This operation requires retarded real-frequency data; Matsubara data must be analytically continued before calling this function.
"""
function spectral_function_from_retarded_green(axis, values::AbstractVector{<:Number};
                                               convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                               normalization::Symbol=:spectral_density, metadata=Dict{Symbol,Any}())
    x = _check_axis_grid(axis)
    length(values) == length(x) || throw(DimensionMismatch("retarded Green-function value count must match the spectral axis"))
    intensity = reshape(Float64.(_retarded_green_intensity(values, normalization)), 1, length(x))
    response_metadata = _response_metadata_copy(metadata)
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :spectral_function_from_retarded_green,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :local,
        :response_order => 2,
        :source_response => :retarded_green_function,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => normalization,
        :resolution_applied => false,
    ))
    return SpectrumResult([nothing], x, intensity, response_metadata)
end

function spectral_function_from_retarded_green(qgrid, axis, values::AbstractMatrix{<:Number};
                                               convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                               normalization::Symbol=:spectral_density, metadata=Dict{Symbol,Any}())
    q = collect(qgrid)
    x = _check_axis_grid(axis)
    size(values) == (length(q), length(x)) || throw(DimensionMismatch("retarded Green-function values must have shape (nq, nfrequency)"))
    intensity = Float64.(_retarded_green_intensity(values, normalization))
    response_metadata = _response_metadata_copy(metadata)
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :spectral_function_from_retarded_green,
        :response_kind => :spectral_function,
        :frequency_domain => :real,
        :statistics => :fermionic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :source_response => :retarded_green_function,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => normalization,
        :resolution_applied => false,
    ))
    return SpectrumResult(q, x, intensity, response_metadata)
end

function _real_susceptibility_axis(result::SusceptibilityResult)
    raw_axis = collect(result.axis)
    isempty(raw_axis) && throw(ArgumentError("susceptibility axis must not be empty"))
    all(value -> value isa Real, raw_axis) || throw(ArgumentError(
        "conversion from susceptibility to a real-frequency spectrum requires a real axis; Matsubara susceptibility data require explicit analytic continuation first"
    ))
    return _check_axis_grid(raw_axis)
end

function _hermitian_response_channel(result::SusceptibilityResult, hermitian_channel)
    metadata = result.metadata isa AbstractDict ? result.metadata : Dict{Symbol,Any}(pairs(result.metadata))
    declared = get(metadata, :hermitian_channel, nothing)
    flag = isnothing(hermitian_channel) ? declared : hermitian_channel
    flag === true || throw(ArgumentError(
        "scalar -Im(chi^R) conversion is valid here only for a Hermitian response channel; set metadata[:hermitian_channel]=true or pass hermitian_channel=true when that condition is known to hold"
    ))
    return true
end

function _commutator_axis_prefactor(convention::SpectrumConvention)
    return convention.spectral_axis === :energy ? 1.0 : Float64(convention.hbar)
end

"""
    commutator_spectral_density(result::SusceptibilityResult; convention=..., hermitian_channel=nothing)

Convert a scalar Hermitian retarded susceptibility into its commutator spectral density. With the package Fourier conventions, `rho_AB(x) = -s Im(chi^R_AB(x))/pi`, where `s=1` on an energy-transfer axis and `s=hbar` on an angular-frequency axis.

The input axis must already be real. The function intentionally rejects Matsubara susceptibilities because analytic continuation is a separate, method-dependent operation.
"""
function commutator_spectral_density(result::SusceptibilityResult;
                                     convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                     hermitian_channel=nothing, metadata=Dict{Symbol,Any}())
    x = _real_susceptibility_axis(result)
    _hermitian_response_channel(result, hermitian_channel)
    size(result.values) == (length(result.q), length(x)) || throw(DimensionMismatch("susceptibility values must have shape (nq, nfrequency)"))
    prefactor = _commutator_axis_prefactor(convention)
    values = -prefactor .* imag.(result.values) ./ pi
    response_metadata = _response_metadata_copy(result.metadata)
    merge!(response_metadata, _response_metadata_copy(metadata))
    merge!(response_metadata, Dict{Symbol,Any}(
        :observable => :commutator_spectral_density,
        :response_kind => :commutator_spectral_density,
        :frequency_domain => :real,
        :statistics => :bosonic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :source_response => :retarded_susceptibility,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :normalization => :unit_integral_delta,
        :resolution_applied => false,
    ))
    return SpectrumResult(copy(result.q), x, Float64.(values), response_metadata)
end

function _fdt_energy(x::Real, convention::SpectrumConvention)
    return convention.spectral_axis === :energy ? Float64(x) : Float64(convention.hbar) * Float64(x)
end

function _fdt_denominator(energy::Float64, temperature::Float64, kB::Float64)
    if iszero(temperature)
        energy > 0 && return 1.0
        energy < 0 && return -Inf
        return 0.0
    end
    beta = inv(kB * temperature)
    return -expm1(-beta * energy)
end

function _resolve_zero_frequency_value(zero_frequency_value, q, x, chi_value)
    isnothing(zero_frequency_value) && throw(ArgumentError(
        "the fluctuation-dissipation factor is singular at zero transfer; supply zero_frequency_value explicitly or evaluate the zero-frequency limit with a continuation/interpolation method appropriate to the response"
    ))
    return zero_frequency_value isa Function ? zero_frequency_value(q, x, chi_value) : zero_frequency_value
end

"""
    correlation_spectrum_from_retarded_susceptibility(result; temperature, convention=..., hermitian_channel=nothing, zero_frequency_value=nothing)

Convert a scalar Hermitian retarded susceptibility into the ordered equilibrium correlation spectrum `S_AB` using the fluctuation-dissipation relation. In the package normalization,

`S_AB(x) = rho_AB(x) / (1 - exp(-beta*x_E))`,

where `x_E=x` for an energy axis and `x_E=hbar*x` for an angular-frequency axis. `rho_AB` is the commutator spectral density returned by `commutator_spectral_density`.

The input must already be a real-frequency retarded susceptibility. Analytic continuation from Matsubara frequencies is deliberately not performed implicitly. At exactly zero transfer the Bose factor requires a limiting procedure; provide `zero_frequency_value` explicitly when the requested grid contains zero.
"""
function correlation_spectrum_from_retarded_susceptibility(result::SusceptibilityResult; temperature::Real,
                                                            convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                                            hermitian_channel=nothing, zero_frequency_value=nothing,
                                                            zero_frequency_tol::Real=0.0, metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    zero_frequency_tol >= 0 || throw(ArgumentError("zero_frequency_tol must be nonnegative"))
    rho = commutator_spectral_density(result; convention=convention, hermitian_channel=hermitian_channel)
    values = Matrix{Float64}(undef, size(rho.intensity))
    temperature_value = Float64(temperature)
    kB = Float64(convention.kB)

    for iq in axes(values, 1), ix in axes(values, 2)
        x = Float64(rho.axis[ix])
        energy = _fdt_energy(x, convention)
        if abs(energy) <= zero_frequency_tol
            values[iq, ix] = Float64(real(_resolve_zero_frequency_value(zero_frequency_value, rho.q[iq], x, result.values[iq, ix])))
            continue
        end
        denominator = _fdt_denominator(energy, temperature_value, kB)
        values[iq, ix] = isinf(denominator) ? 0.0 : rho.intensity[iq, ix] / denominator
    end

    response_metadata = copy(rho.metadata)
    merge!(response_metadata, _response_metadata_copy(metadata))
    response_metadata[:observable] = :correlation_spectrum_from_retarded_susceptibility
    response_metadata[:response_kind] = :correlation_spectrum
    response_metadata[:source_response] = :retarded_susceptibility
    response_metadata[:fluctuation_dissipation_applied] = true
    response_metadata[:temperature] = temperature_value
    response_metadata[:zero_frequency_policy] = isnothing(zero_frequency_value) ? :error : :explicit_value
    return SpectrumResult(copy(rho.q), copy(rho.axis), values, response_metadata)
end
