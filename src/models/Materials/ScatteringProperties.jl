# Probe-relevant material properties. No tabulated ion data are hard-coded;
# users may supply measured/tabulated coefficients without changing the model.

abstract type AbstractMagneticFormFactor end

struct ConstantFormFactor <: AbstractMagneticFormFactor
    value::Float64
end
ConstantFormFactor() = ConstantFormFactor(1.0)

"""
Gaussian sum form factor

    f(Q) = c + sum_k a_k exp[-b_k (Q / scale)^2].

`scale=4*pi` matches a common crystallographic parameterization when Q is an
inverse-length magnitude and the supplied `b_k` use the corresponding units.
"""
struct GaussianFormFactor <: AbstractMagneticFormFactor
    a::Vector{Float64}
    b::Vector{Float64}
    c::Float64
    scale::Float64
end

function GaussianFormFactor(a, b; c::Real=0.0, scale::Real=4pi)
    length(a) == length(b) || throw(DimensionMismatch("a and b must have equal lengths"))
    scale > 0 || throw(ArgumentError("form-factor scale must be positive"))
    return GaussianFormFactor(Float64.(a), Float64.(b), Float64(c), Float64(scale))
end

form_factor(ff::ConstantFormFactor, Q::Real) = ff.value
function form_factor(ff::GaussianFormFactor, Q::Real)
    q2 = (Float64(Q) / ff.scale)^2
    return ff.c + sum(ai * exp(-bi * q2) for (ai, bi) in zip(ff.a, ff.b))
end
form_factor(ff::AbstractMagneticFormFactor, Q) = form_factor(ff, norm(Q))

struct DebyeWallerTensor
    U::Matrix{Float64}
end

function DebyeWallerTensor(U::AbstractMatrix{<:Real})
    size(U, 1) == size(U, 2) || throw(DimensionMismatch("Debye-Waller tensor must be square"))
    issymmetric(Matrix(U)) || throw(ArgumentError("Debye-Waller tensor must be symmetric"))
    minimum(eigvals(Symmetric(Float64.(U)))) >= -1e-12 ||
        throw(ArgumentError("Debye-Waller tensor must be positive semidefinite"))
    return DebyeWallerTensor(Float64.(U))
end

"""Amplitude Debye-Waller factor exp[-1/2 Q' U Q]."""
function debye_waller_factor(dw::DebyeWallerTensor, Q)
    length(Q) == size(dw.U, 1) || throw(DimensionMismatch("Q dimension does not match U"))
    q = Float64.(collect(Q))
    return exp(-0.5 * dot(q, dw.U * q))
end


struct MeanSquareDisplacementResult
    temperature::Float64
    tensors::Vector{Matrix{Float64}}
    metadata::Dict{Symbol,Any}
end

function _phonon_dispersion_arrays(result)
    hasproperty(result, :frequencies) ||
        throw(ArgumentError("phonon thermal displacements require a phonon dispersion result"))
    hasproperty(result, :modes) ||
        throw(ArgumentError("phonon thermal displacements require retained phonon mode vectors"))
    hasproperty(result, :qpoints) ||
        throw(ArgumentError("phonon thermal displacements require a q-resolved dispersion result"))
    return result.frequencies, result.modes, result.qpoints
end

function _dispersion_zero_mode_mask(result, frequencies::AbstractMatrix{<:Real}, zero_mode_tol::Union{Nothing,Real})
    if isnothing(zero_mode_tol) && hasproperty(result, :metadata) && haskey(result.metadata, :zero_mode_mask)
        mask = Bool.(result.metadata[:zero_mode_mask])
        size(mask) == size(frequencies) || throw(DimensionMismatch("solver zero-mode mask does not match phonon dispersion"))
        return mask, :solver, nothing, nothing
    end

    tolerance = isnothing(zero_mode_tol) ? 1e-10 : Float64(zero_mode_tol)
    tolerance >= 0 || throw(ArgumentError("zero_mode_tol must be nonnegative"))
    threshold = tolerance * max(maximum(abs, frequencies; init=0.0), 1.0)
    source = isnothing(zero_mode_tol) ? :legacy_frequency_fallback : :explicit_frequency_tolerance
    return frequencies .<= threshold, source, tolerance, threshold
end

"""
    mean_square_displacement(result; temperature=0, weights=nothing, hbar=1, kB=1, displacement_prefactor=nothing, zero_mode_policy=:exclude, zero_mode_tol=nothing)

Compute per-basis-site displacement covariance tensors from a q-resolved harmonic phonon dispersion. `result.modes` are interpreted as mass-weighted eigenvectors, consistent with `PhononDispersionResult`.

`weights` are Brillouin-zone quadrature weights and are normalized internally. If omitted, all sampled q points receive equal weight. `hbar` controls the Bose exponent, while `displacement_prefactor` controls the covariance amplitude; its default is `hbar` for backward compatibility, but energy-valued phonon frequencies may supply the appropriate independent unit conversion.

The Γ translational zero modes make the naive finite-grid Bose factor divergent. By default, solver-provided zero-mode classifications in `result.metadata[:zero_mode_mask]` are used. Supplying `zero_mode_tol` overrides that classification with a relative frequency-space threshold; results without solver metadata retain the legacy `1e-10` fallback. `zero_mode_policy=:exclude` omits classified modes and `:error` raises instead.
"""
function mean_square_displacement(result; temperature::Real=0.0, weights=nothing, hbar::Real=1.0, kB::Real=1.0,
                                  displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_policy::Symbol=:exclude,
                                  zero_mode_tol::Union{Nothing,Real}=nothing)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    zero_mode_policy in (:exclude, :error) || throw(ArgumentError("zero_mode_policy must be :exclude or :error"))

    frequencies, modes, qpoints = _phonon_dispersion_arrays(result)
    nq, nbranch = size(frequencies)
    size(modes, 2) == nbranch || throw(DimensionMismatch("mode count does not match frequency branches"))
    size(modes, 3) == nq || throw(DimensionMismatch("mode q count does not match frequency q count"))
    length(qpoints) == nq || throw(DimensionMismatch("qpoint count does not match dispersion"))

    model = result.model
    D = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    ndof = nbasis * D
    size(modes, 1) == ndof || throw(DimensionMismatch("mode-vector dimension does not match model"))
    masses = Float64.(model.parameters[:masses])
    length(masses) == ndof || throw(DimensionMismatch("mass vector does not match phonon modes"))

    qweights = if isnothing(weights)
        fill(inv(float(nq)), nq)
    else
        w = Float64.(collect(weights))
        length(w) == nq || throw(DimensionMismatch("weights must contain one value per q point"))
        all(>=(0), w) || throw(ArgumentError("Brillouin-zone weights must be nonnegative"))
        sum(w) > 0 || throw(ArgumentError("Brillouin-zone weights must have positive total weight"))
        w ./ sum(w)
    end

    zero_mode_data = _dispersion_zero_mode_mask(result, frequencies, zero_mode_tol)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = zero_mode_data
    tensors = [zeros(Float64, D, D) for _ in 1:nbasis]
    excluded = Tuple{Int,Int}[]
    invsqrtm = 1 ./ sqrt.(masses)

    for iq in 1:nq
        for ν in 1:nbranch
            if zero_mask[iq, ν]
                zero_mode_policy === :error && throw(ArgumentError("zero/soft phonon mode encountered at q index $iq, branch $ν"))
                push!(excluded, (iq, ν))
                continue
            end

            ω = Float64(frequencies[iq, ν])
            occupation = if iszero(temperature)
                0.0
            else
                x = Float64(hbar) * ω / (Float64(kB) * Float64(temperature))
                inv(expm1(x))
            end
            quantum_factor = displacement_scale * (2occupation + 1) / (2ω)
            weight = qweights[iq] * quantum_factor

            @inbounds for site in 1:nbasis
                rows = ((site - 1) * D + 1):(site * D)
                polarization = ComplexF64.(modes[rows, ν, iq]) .* invsqrtm[rows]
                tensors[site] .+= weight .* real.(polarization * adjoint(polarization))
            end
        end
    end

    for U in tensors
        U .= 0.5 .* (U .+ transpose(U))
    end

    metadata = Dict{Symbol,Any}(
        :qpoint_count => nq,
        :branch_count => nbranch,
        :weights_normalized => true,
        :zero_mode_policy => zero_mode_policy,
        :zero_mode_source => zero_mode_source,
        :zero_mode_tolerance => zero_mode_tolerance,
        :zero_mode_frequency_threshold => zero_mode_frequency_threshold,
        :excluded_zero_modes => excluded,
        :hbar => Float64(hbar),
        :kB => Float64(kB),
        :displacement_prefactor => displacement_scale,
        :phonon_units => get(model.parameters, :phonon_units, nothing),
    )
    return MeanSquareDisplacementResult(Float64(temperature), tensors, metadata)
end

"""
    phonon_debye_waller_tensors(result; kwargs...)

Compute phonon-derived `DebyeWallerTensor` objects for each primitive-cell basis
site. Keywords are forwarded to `mean_square_displacement`.
"""
function phonon_debye_waller_tensors(result; kwargs...)
    msd = mean_square_displacement(result; kwargs...)
    return DebyeWallerTensor[DebyeWallerTensor(U) for U in msd.tensors]
end
