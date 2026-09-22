# Native spin-wave neutron-scattering tensors for the collinear LSWT backend.
#
# The current LinearSpinWaveSolver supports collinear ±z reference states for
# standard Heisenberg/XXZ Hamiltonians. This file evaluates the corresponding
# Gaussian spin-wave correlation tensor directly from SpinWaveResult without
# constructing many-body eigenstates or symbolic Holstein-Primakoff operators.

const _LSWT_CHANNELS = (:elastic, :one_magnon, :two_magnon)

function _lswt_field_operator(model::ManyBodyModel, key)
    if key isa ObservableRef
        return resolve_observable(model, key)
    elseif key isa Symbol
        return resolve_observable(model, key)
    elseif key isa AbstractOperatorExpr
        return key
    end
    throw(ArgumentError("SpinTensorField keys must be Symbol, ObservableRef, or operator expression"))
end

function _lswt_field_site(model::ManyBodyModel, key, component::Symbol)
    op = _lswt_field_operator(model, key)
    if component === :x
        op isa SpinX || throw(ArgumentError("native LSWT x field must contain local SpinX operators"))
    elseif component === :y
        op isa SpinY || throw(ArgumentError("native LSWT y field must contain local SpinY operators"))
    elseif component === :z
        op isa SpinZ || throw(ArgumentError("native LSWT z field must contain local SpinZ operators"))
    else
        throw(ArgumentError("unknown Cartesian spin component $component"))
    end
    return op.site
end

function _lswt_field_geometry(model::ManyBodyModel, field::SpinTensorField)
    norms = (field.x.normalization, field.y.normalization, field.z.normalization)
    all(norm -> norm == norms[1], norms) || throw(ArgumentError("all spin tensor fields must use the same Fourier normalization"))

    nx = length(field.x.keys)
    (length(field.y.keys) == nx && length(field.z.keys) == nx) || throw(DimensionMismatch("all spin tensor fields must contain the same number of sites"))
    (field.x.positions == field.y.positions && field.x.positions == field.z.positions) || throw(ArgumentError("all spin tensor components must use identical positions"))

    xsites = [_lswt_field_site(model, key, :x) for key in field.x.keys]
    ysites = [_lswt_field_site(model, key, :y) for key in field.y.keys]
    zsites = [_lswt_field_site(model, key, :z) for key in field.z.keys]
    (xsites == ysites && xsites == zsites) || throw(ArgumentError("native LSWT spin tensor requires x, y, and z fields to address the same sites in the same order"))
    length(unique(xsites)) == length(xsites) || throw(ArgumentError("native LSWT spin tensor fields must not repeat a site"))

    return xsites, field.x.positions, norms[1]
end

function _lswt_spin_data(model::ManyBodyModel, result::SpinWaveResult)
    model.representation.algebra isa SpinAlgebra || throw(ArgumentError("native LSWT spin tensor requires a SpinAlgebra model"))
    result.model.representation.algebra isa SpinAlgebra || throw(ArgumentError("SpinWaveResult does not retain a spin source model"))

    N = model.representation.algebra.nsites
    result.model.representation.algebra.nsites == N || throw(DimensionMismatch("model and SpinWaveResult contain different numbers of spin sites"))
    (size(result.A) == (N, N) && size(result.B) == (N, N)) || throw(DimensionMismatch("SpinWaveResult quadratic blocks do not match the source spin model"))

    rawspace = model.representation.state_space isa BiorthogonalSpace ? model.representation.state_space.ambient : model.representation.state_space
    rawspace isa SpinHilbertSpace || throw(ArgumentError("native LSWT spin tensor requires SpinHilbertSpace"))
    spins = Float64.(rawspace.spins)

    signs = get(result.metadata, :reference_signs, nothing)
    isnothing(signs) && throw(ArgumentError("SpinWaveResult does not retain reference_signs"))
    signs = Int.(signs)
    length(signs) == N || throw(DimensionMismatch("SpinWaveResult reference_signs do not match the source spin model"))
    all(sign -> sign in (-1, 1), signs) || throw(ArgumentError("SpinWaveResult reference_signs must contain only ±1"))

    frequencies = Float64.(result.frequencies)
    M = length(frequencies)
    modes = result.modes
    size(modes, 2) == M || throw(DimensionMismatch("SpinWaveResult mode-vector count does not match its frequency count"))

    if size(modes, 1) == N
        U = modes
        V = nothing
    elseif size(modes, 1) == 2N
        U = view(modes, 1:N, :)
        V = view(modes, N+1:2N, :)
    else
        throw(DimensionMismatch("SpinWaveResult modes must have N or 2N rows"))
    end

    return spins, signs, frequencies, U, V
end

function _lswt_bose_occupations(frequencies::Vector{Float64}, temperature::Real, kB::Real; tol::Real)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    occupations = zeros(Float64, length(frequencies))
    iszero(temperature) && return occupations

    isempty(frequencies) && return occupations
    scale = max(maximum(abs.(frequencies)), 1.0)
    minimum(frequencies) > tol * scale || throw(ArgumentError("finite-temperature LSWT requires strictly positive mode energies; zero/Goldstone modes require an infrared regularization or thermodynamic treatment"))
    β = inv(float(kB) * float(temperature))
    @inbounds for ν in eachindex(frequencies)
        occupations[ν] = inv(expm1(β * frequencies[ν]))
    end
    return occupations
end

function _lswt_local_depletion(U, V, occupations::Vector{Float64}, sites::Vector{Int})
    depletion = zeros(Float64, length(sites))
    thermal = any(value -> !iszero(value), occupations)

    @inbounds for (a, site) in enumerate(sites)
        value = 0.0
        for ν in axes(U, 2)
            u2 = abs2(U[site, ν])
            if isnothing(V)
                thermal && (value += u2 * occupations[ν])
            else
                v2 = abs2(V[site, ν])
                value += v2
                thermal && (value += (u2 + v2) * occupations[ν])
            end
        end
        depletion[a] = value
    end
    return depletion
end

@inline function _lswt_kernel_value(b::LorentzianBroadening, x::Real)
    return (b.width / π) / (x * x + b.width^2)
end

@inline function _lswt_kernel_value(b::GaussianBroadening, x::Real)
    return exp(-0.5 * (x / b.sigma)^2) / (sqrt(2π) * b.sigma)
end

@inline _lswt_kernel_value(b::AbstractBroadening, x::Real) = first(_kernel(b, [x]))

function _lswt_add_scalar_line!(out, iq::Int, axis::Vector{Float64}, center::Float64, weight::ComplexF64, broadening::AbstractBroadening, α::Int, β::Int)
    @inbounds for ix in eachindex(axis)
        out[iq, ix, α, β] += weight * _lswt_kernel_value(broadening, axis[ix] - center)
    end
    return nothing
end

function _lswt_add_transverse_line!(out, iq::Int, axis::Vector{Float64}, center::Float64, xamp::ComplexF64, yamp::ComplexF64, population::Float64, broadening::AbstractBroadening)
    wxx = population * xamp * conj(xamp)
    wxy = population * xamp * conj(yamp)
    wyx = population * yamp * conj(xamp)
    wyy = population * yamp * conj(yamp)

    @inbounds for ix in eachindex(axis)
        kernel = _lswt_kernel_value(broadening, axis[ix] - center)
        out[iq, ix, 1, 1] += wxx * kernel
        out[iq, ix, 1, 2] += wxy * kernel
        out[iq, ix, 2, 1] += wyx * kernel
        out[iq, ix, 2, 2] += wyy * kernel
    end
    return nothing
end

function _lswt_uniform_spacing(axis::Vector{Float64}; rtol::Real=1e-10)
    dE = axis[2] - axis[1]
    scale = max(abs(dE), 1.0)
    @inbounds for i in 2:length(axis)-1
        abs((axis[i + 1] - axis[i]) - dE) <= rtol * scale || return nothing
    end
    return dE
end

function _lswt_deposit_scalar!(raw, axis::Vector{Float64}, dE::Float64, center::Float64, weight::ComplexF64, α::Int, β::Int)
    (center < axis[1] || center > axis[end]) && return nothing
    t = (center - axis[1]) / dE + 1
    left = clamp(floor(Int, t), 1, length(axis))
    if left == length(axis)
        raw[left, α, β] += weight
        return nothing
    end
    fraction = t - left
    raw[left, α, β] += (1 - fraction) * weight
    raw[left + 1, α, β] += fraction * weight
    return nothing
end

function _lswt_deposit_transverse!(raw, axis::Vector{Float64}, dE::Float64, center::Float64, xamp::ComplexF64, yamp::ComplexF64, population::Float64)
    (center < axis[1] || center > axis[end]) && return nothing
    t = (center - axis[1]) / dE + 1
    left = clamp(floor(Int, t), 1, length(axis))
    right = min(left + 1, length(axis))
    fraction = right == left ? 0.0 : t - left
    wl = 1 - fraction
    wr = fraction

    wxx = population * xamp * conj(xamp)
    wxy = population * xamp * conj(yamp)
    wyx = population * yamp * conj(xamp)
    wyy = population * yamp * conj(yamp)

    raw[left, 1, 1] += wl * wxx
    raw[left, 1, 2] += wl * wxy
    raw[left, 2, 1] += wl * wyx
    raw[left, 2, 2] += wl * wyy
    if right != left
        raw[right, 1, 1] += wr * wxx
        raw[right, 1, 2] += wr * wxy
        raw[right, 2, 1] += wr * wyx
        raw[right, 2, 2] += wr * wyy
    end
    return nothing
end

function _lswt_convolve_binned!(out, iq::Int, raw, axis::Vector{Float64}, broadening::AbstractBroadening)
    @inbounds for source in eachindex(axis)
        wxx = raw[source, 1, 1]
        wxy = raw[source, 1, 2]
        wyx = raw[source, 2, 1]
        wyy = raw[source, 2, 2]
        wzz = raw[source, 3, 3]
        iszero(wxx) && iszero(wxy) && iszero(wyx) && iszero(wyy) && iszero(wzz) && continue
        center = axis[source]
        for ix in eachindex(axis)
            kernel = _lswt_kernel_value(broadening, axis[ix] - center)
            out[iq, ix, 1, 1] += wxx * kernel
            out[iq, ix, 1, 2] += wxy * kernel
            out[iq, ix, 2, 1] += wyx * kernel
            out[iq, ix, 2, 2] += wyy * kernel
            out[iq, ix, 3, 3] += wzz * kernel
        end
    end
    return nothing
end

function _lswt_line_domain(frequencies::Vector{Float64}, selected_channels, finite_temperature::Bool, axis_scale::Float64)
    ωmax = isempty(frequencies) ? 0.0 : maximum(frequencies) * axis_scale
    if :two_magnon in selected_channels
        return finite_temperature ? (-2ωmax, 2ωmax) : (0.0, 2ωmax)
    elseif :one_magnon in selected_channels
        return finite_temperature ? (-ωmax, ωmax) : (0.0, ωmax)
    end
    return (0.0, 0.0)
end

function _lswt_phase_vector!(phases::Vector{ComplexF64}, q, positions, normalization::Symbol)
    aN = _fourier_amplitude(normalization, length(positions))
    @inbounds for i in eachindex(positions)
        phases[i] = aN * _phase(q, positions[i])
    end
    return phases
end

function _lswt_prepare_transverse_modes(U, V, sites::Vector{Int}, ordered_moments::Vector{Float64}, signs::Vector{Int})
    nfield = length(sites)
    M = size(U, 2)
    X = Matrix{ComplexF64}(undef, nfield, M)
    Y = Matrix{ComplexF64}(undef, nfield, M)

    @inbounds for a in 1:nfield
        site = sites[a]
        coefficient = sqrt(max(ordered_moments[a], 0.0) / 2)
        η = signs[site]
        for ν in 1:M
            u = U[site, ν]
            v = isnothing(V) ? 0.0 + 0.0im : V[site, ν]
            X[a, ν] = coefficient * (u + v)
            Y[a, ν] = η * coefficient * (u - v)
        end
    end
    return X, Y
end

function _lswt_prepare_bare_transverse_modes(U, V, sites::Vector{Int}, spins::Vector{Float64}, signs::Vector{Int})
    nfield = length(sites)
    M = size(U, 2)
    X = Matrix{ComplexF64}(undef, nfield, M)
    Y = Matrix{ComplexF64}(undef, nfield, M)

    @inbounds for a in 1:nfield
        site = sites[a]
        coefficient = sqrt(spins[site] / 2)
        η = signs[site]
        for ν in 1:M
            u = U[site, ν]
            v = isnothing(V) ? 0.0 + 0.0im : V[site, ν]
            X[a, ν] = coefficient * (u + v)
            Y[a, ν] = η * coefficient * (u - v)
        end
    end
    return X, Y
end

function _lswt_mode_rows(U, V, sites::Vector{Int})
    Uf = Matrix{ComplexF64}(undef, length(sites), size(U, 2))
    Vf = isnothing(V) ? nothing : similar(Uf)
    @inbounds for (a, site) in enumerate(sites), ν in axes(U, 2)
        Uf[a, ν] = U[site, ν]
        isnothing(Vf) || (Vf[a, ν] = V[site, ν])
    end
    return Uf, Vf
end

function _lswt_validate_channels(channels)
    selected = Tuple(Symbol(channel) for channel in channels)
    isempty(selected) && throw(ArgumentError("channels cannot be empty"))
    all(channel -> channel in _LSWT_CHANNELS, selected) || throw(ArgumentError("LSWT channels must be chosen from $(_LSWT_CHANNELS)"))
    return selected
end

"""
    spin_tensor_structure_factor(model, result::SpinWaveResult, field, qgrid, axis; ...)

Compute the native Cartesian spin-wave tensor `S^{αβ}(q,x)` for the current
collinear `LinearSpinWaveSolver` regime. The calculation uses the Bogoliubov
mode vectors stored in `SpinWaveResult` directly and therefore does not build a
many-body basis or symbolic Holstein-Primakoff scattering operators.

For a reference sign `ηᵢ = ±1`, the laboratory-frame operators retained are

    Sᵢˣ = sqrt(mᵢ/2) (aᵢ + aᵢ†),
    Sᵢʸ = -i ηᵢ sqrt(mᵢ/2) (aᵢ - aᵢ†),
    Sᵢᶻ = ηᵢ (Sᵢ - aᵢ†aᵢ),

where `mᵢ = Sᵢ - ΔSᵢ` by default. This ordered-moment transverse
renormalization reproduces the noninteracting spin-wave neutron intensity and
sum-rule convention used by Huberman et al. for Rb₂MnF₄. Set
`renormalize_transverse=false` for the strict leading-order Holstein-Primakoff
factor `sqrt(Sᵢ/2)`.

The returned tensor contains the requested `channels` from `:elastic`,
`:one_magnon`, and `:two_magnon`. At zero temperature, `:two_magnon` is the
longitudinal pair-creation continuum. At finite temperature it additionally
contains pair-annihilation and one-magnon-in/one-magnon-out longitudinal
scattering. Mixed transverse-longitudinal components vanish at this Gaussian
collinear order.
"""
function spin_tensor_structure_factor(
    model::ManyBodyModel,
    result::SpinWaveResult,
    field::SpinTensorField,
    qgrid,
    axis;
    temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.x.normalization),
    broadening::AbstractBroadening=LorentzianBroadening(0.05),
    weight_tol::Real=0.0,
    channels=_LSWT_CHANNELS,
    renormalize_transverse::Bool=true,
    spectral_accumulation::Symbol=:auto,
)
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    selected_channels = _lswt_validate_channels(channels)
    sites, positions, normalization = _lswt_field_geometry(model, field)
    convention.fourier_normalization == normalization || throw(ArgumentError("SpectrumConvention Fourier normalization must match SpinTensorField"))

    spins, signs, frequencies, U, V = _lswt_spin_data(model, result)
    isempty(sites) && throw(ArgumentError("native LSWT spin tensor requires at least one field site"))
    all(site -> 1 <= site <= length(spins), sites) || throw(BoundsError(spins, sites))

    occupations = _lswt_bose_occupations(frequencies, temperature, convention.kB; tol=result.solver.tol)
    depletion = _lswt_local_depletion(U, V, occupations, sites)
    ordered_moments = Float64[spins[site] - depletion[a] for (a, site) in enumerate(sites)]
    minmoment = minimum(ordered_moments)
    minmoment >= -100 * result.solver.tol || throw(ArgumentError("LSWT ordered moment became negative; the harmonic expansion is outside its valid regime"))
    ordered_moments .= max.(ordered_moments, 0.0)

    X, Y = renormalize_transverse ? _lswt_prepare_transverse_modes(U, V, sites, ordered_moments, signs) : _lswt_prepare_bare_transverse_modes(U, V, sites, spins, signs)
    Uf, Vf = _lswt_mode_rows(U, V, sites)

    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    out = zeros(ComplexF64, length(qs), length(x), 3, 3)
    axis_scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    finite_temperature = !iszero(temperature)
    spectral_accumulation in (:auto, :direct, :binned) || throw(ArgumentError("spectral_accumulation must be :auto, :direct, or :binned"))
    dE = _lswt_uniform_spacing(x)
    line_min, line_max = _lswt_line_domain(frequencies, selected_channels, finite_temperature, axis_scale)
    covers_lines = x[1] <= line_min + 1e-12 && x[end] >= line_max - 1e-12
    use_binned = spectral_accumulation === :binned || (spectral_accumulation === :auto && !isnothing(dE) && covers_lines)
    use_binned && isnothing(dE) && throw(ArgumentError("binned LSWT spectral accumulation requires a uniformly spaced spectral axis"))
    bin_width = use_binned ? something(dE) : 0.0
    M = length(frequencies)
    nfield = length(sites)

    phases = zeros(ComplexF64, nfield)
    zphases = similar(phases)
    Lx = zeros(ComplexF64, M)
    Ly = similar(Lx)
    Cx = finite_temperature ? similar(Lx) : ComplexF64[]
    Cy = finite_temperature ? similar(Lx) : ComplexF64[]

    need_longitudinal = :elastic in selected_channels || :two_magnon in selected_channels
    weightedU = need_longitudinal ? similar(Uf) : Matrix{ComplexF64}(undef, 0, 0)
    weightedV = need_longitudinal && !isnothing(Vf) ? similar(Vf) : nothing
    weightedConjV = finite_temperature && :two_magnon in selected_channels && !isnothing(Vf) ? similar(Vf) : nothing
    D = :two_magnon in selected_channels && !isnothing(Vf) ? zeros(ComplexF64, M, M) : nothing
    P = finite_temperature && :two_magnon in selected_channels && !isnothing(Vf) ? similar(D) : nothing
    R = finite_temperature && :two_magnon in selected_channels ? zeros(ComplexF64, M, M) : nothing
    raw = use_binned ? zeros(ComplexF64, length(x), 3, 3) : nothing

    @inbounds for (iq, q) in enumerate(qs)
        isnothing(raw) || fill!(raw, 0.0 + 0.0im)
        _lswt_phase_vector!(phases, q, positions, normalization)

        if :one_magnon in selected_channels
            mul!(Lx, transpose(X), phases)
            mul!(Ly, transpose(Y), phases)
            Ly .*= -1im

            for ν in 1:M
                population = occupations[ν] + 1
                amplitude_scale = max(abs2(Lx[ν]), abs2(Ly[ν]))
                if population * amplitude_scale > weight_tol
                    center = frequencies[ν] * axis_scale
                    use_binned ? _lswt_deposit_transverse!(raw, x, bin_width, center, Lx[ν], Ly[ν], population) : _lswt_add_transverse_line!(out, iq, x, center, Lx[ν], Ly[ν], population, broadening)
                end
            end

            if finite_temperature
                mul!(Cx, adjoint(X), phases)
                mul!(Cy, adjoint(Y), phases)
                Cy .*= 1im
                for ν in 1:M
                    population = occupations[ν]
                    population <= 0 && continue
                    amplitude_scale = max(abs2(Cx[ν]), abs2(Cy[ν]))
                    if population * amplitude_scale > weight_tol
                        center = -frequencies[ν] * axis_scale
                        use_binned ? _lswt_deposit_transverse!(raw, x, bin_width, center, Cx[ν], Cy[ν], population) : _lswt_add_transverse_line!(out, iq, x, center, Cx[ν], Cy[ν], population, broadening)
                    end
                end
            end
        end

        if need_longitudinal
            for a in 1:nfield
                zphases[a] = phases[a] * signs[sites[a]]
            end

            if :elastic in selected_channels
                mz = 0.0 + 0.0im
                for a in 1:nfield
                    mz += zphases[a] * ordered_moments[a]
                end
                weight = ComplexF64(abs2(mz))
                if abs(weight) > weight_tol
                    use_binned ? _lswt_deposit_scalar!(raw, x, bin_width, 0.0, weight, 3, 3) : _lswt_add_scalar_line!(out, iq, x, 0.0, weight, broadening, 3, 3)
                end
            end

            if :two_magnon in selected_channels
                for a in 1:nfield, ν in 1:M
                    weightedU[a, ν] = zphases[a] * Uf[a, ν]
                    isnothing(weightedV) || (weightedV[a, ν] = zphases[a] * Vf[a, ν])
                end

                if !isnothing(Vf)
                    mul!(D, transpose(Vf), weightedU, -1.0 + 0im, 0.0 + 0im)
                    for μ in 1:M, ν in μ:M
                        amplitude = μ == ν ? sqrt(2.0) * D[μ, μ] : D[μ, ν] + D[ν, μ]
                        population = (occupations[μ] + 1) * (occupations[ν] + 1)
                        weight = population * abs2(amplitude)
                        weight <= weight_tol && continue
                        center = (frequencies[μ] + frequencies[ν]) * axis_scale
                        use_binned ? _lswt_deposit_scalar!(raw, x, bin_width, center, ComplexF64(weight), 3, 3) : _lswt_add_scalar_line!(out, iq, x, center, ComplexF64(weight), broadening, 3, 3)
                    end
                end

                if finite_temperature
                    if !isnothing(Vf)
                        for a in 1:nfield, ν in 1:M
                            weightedConjV[a, ν] = zphases[a] * conj(Vf[a, ν])
                        end
                        mul!(P, adjoint(Uf), weightedConjV, -1.0 + 0im, 0.0 + 0im)
                        for μ in 1:M, ν in μ:M
                            amplitude = μ == ν ? sqrt(2.0) * P[μ, μ] : P[μ, ν] + P[ν, μ]
                            population = occupations[μ] * occupations[ν]
                            weight = population * abs2(amplitude)
                            weight <= weight_tol && continue
                            center = -(frequencies[μ] + frequencies[ν]) * axis_scale
                            use_binned ? _lswt_deposit_scalar!(raw, x, bin_width, center, ComplexF64(weight), 3, 3) : _lswt_add_scalar_line!(out, iq, x, center, ComplexF64(weight), broadening, 3, 3)
                        end
                    end

                    mul!(R, adjoint(Uf), weightedU, -1.0 + 0im, 0.0 + 0im)
                    if !isnothing(Vf)
                        mul!(R, adjoint(Vf), weightedV, -1.0 + 0im, 1.0 + 0im)
                    end
                    for μ in 1:M, ν in 1:M
                        population = occupations[μ] * (occupations[ν] + 1)
                        population <= 0 && continue
                        weight = population * abs2(R[μ, ν])
                        weight <= weight_tol && continue
                        center = (frequencies[ν] - frequencies[μ]) * axis_scale
                        use_binned ? _lswt_deposit_scalar!(raw, x, bin_width, center, ComplexF64(weight), 3, 3) : _lswt_add_scalar_line!(out, iq, x, center, ComplexF64(weight), broadening, 3, 3)
                    end
                end
            end
        end

        use_binned && _lswt_convolve_binned!(out, iq, raw, x, broadening)
    end

    metadata = Dict{Symbol,Any}(
        :observable => :spin_tensor_dynamic_structure_factor,
        :components => (:x, :y, :z),
        :temperature => float(temperature),
        :fourier_normalization => normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :representation => model.representation.name,
        :solver => :linear_spin_wave,
        :tensor_definition => :S_alpha_beta,
        :native_lswt => true,
        :approximation => :gaussian_linear_spin_wave,
        :channels => selected_channels,
        :reference_signs => copy(signs),
        :mode_count => M,
        :field_sites => copy(sites),
        :zero_point_or_thermal_depletion => depletion,
        :ordered_moments => ordered_moments,
        :transverse_normalization => renormalize_transverse ? :ordered_moment : :bare_spin,
        :spectral_accumulation => use_binned ? :binned_linear_deposition : :direct_kernel_evaluation,
        :mixed_transverse_longitudinal_components => :zero_at_collinear_gaussian_order,
        :finite_temperature_longitudinal_channels => finite_temperature ? (:pair_creation, :pair_annihilation, :magnon_scattering) : (:pair_creation,),
    )
    return TensorSpectrumResult(qs, x, out, metadata)
end
