# Probe-specific scattering contractions.

"""
Magnetic neutron probe metadata. `prefactor` is deliberately user-controlled:
the tensor contraction and material form factors are separated from any chosen
absolute cross-section unit convention.

`form_factor(Q)` may return a scalar magnetic form factor. `g_tensor` is a
3x3 matrix acting on spin components before the transverse polarization
projection.
"""
struct MagneticNeutronProbe{F,G,T} <: AbstractProbe
    form_factor::F
    g_tensor::G
    prefactor::T
end

function MagneticNeutronProbe(
    ; form_factor=(Q -> 1.0),
      g_tensor=Matrix{Float64}(I, 3, 3),
      prefactor::Real=1.0,
)
    size(g_tensor) == (3, 3) || throw(DimensionMismatch("g_tensor must be 3x3"))
    return MagneticNeutronProbe(form_factor, Matrix(g_tensor), float(prefactor))
end

function _qvector(Q)
    if Q isa Number
        return Float64[float(Q), 0.0, 0.0]
    end
    q = Float64.(collect(Q))
    length(q) == 3 || throw(DimensionMismatch("magnetic neutron Q vectors must have three components"))
    return q
end

"""
    magnetic_neutron_intensity(probe, tensor_spectrum)

Contract the full tensor using

    P_ab(Q) = delta_ab - Qhat_a Qhat_b

and the supplied g tensor.  In matrix notation the contraction uses
`P : (g S g^T)`.  The scalar magnetic form factor enters as `|F(Q)|^2`.
No instrument resolution is applied here.
"""
function magnetic_neutron_intensity(
    probe::MagneticNeutronProbe,
    spectrum::TensorSpectrumResult,
)
    nq, nx, na, nb = size(spectrum.intensity)
    na == nb == 3 || throw(DimensionMismatch("spin tensor spectrum must have 3x3 Cartesian components"))
    length(spectrum.q) == nq || throw(DimensionMismatch("q grid does not match tensor spectrum"))

    out = zeros(Float64, nq, nx)
    g = ComplexF64.(probe.g_tensor)

    for iq in 1:nq
        Q = _qvector(spectrum.q[iq])
        qnorm = norm(Q)
        qnorm > 0 || throw(ArgumentError("magnetic polarization tensor is undefined at Q=0; supply a nonzero Q vector"))
        qhat = Q / qnorm
        P = Matrix{Float64}(I, 3, 3) - qhat * transpose(qhat)
        F2 = abs2(probe.form_factor(spectrum.q[iq]))

        for ix in 1:nx
            @views S = spectrum.intensity[iq, ix, :, :]
            Seff = g * S * adjoint(g)
            value = sum(P .* Seff)
            # A physical unpolarized cross section is real. Small imaginary
            # residue is numerical; a large one indicates an inconsistent tensor.
            scale = max(abs(real(value)), 1.0)
            abs(imag(value)) <= 1e-9 * scale ||
                throw(ArgumentError("magnetic tensor contraction produced a significant imaginary intensity"))
            out[iq, ix] = probe.prefactor * F2 * real(value)
        end
    end

    metadata = copy(spectrum.metadata)
    metadata[:probe] = :magnetic_neutron
    metadata[:polarization_projection] = :transverse
    metadata[:form_factor_applied] = true
    metadata[:g_tensor_applied] = true
    metadata[:absolute_prefactor] = probe.prefactor
    metadata[:resolution_applied] = false
    return ScatteringResult(spectrum.q, spectrum.axis, out, metadata)
end

scattering_intensity(probe::MagneticNeutronProbe, spectrum::TensorSpectrumResult) =
    magnetic_neutron_intensity(probe, spectrum)


"""
    NuclearNeutronProbe(; prefactor=1)

Coherent nuclear neutron probe for one-phonon scattering. Coherent scattering
lengths are read from the `AtomicSpecies` objects retained by the phonon model's
`Material`. `prefactor` is left explicit because absolute cross-section units
depend on the caller's length/energy convention.
"""
struct NuclearNeutronProbe{T<:Real} <: AbstractProbe
    prefactor::T
end

NuclearNeutronProbe(; prefactor::Real=1.0) =
    NuclearNeutronProbe{typeof(float(prefactor))}(float(prefactor))

const CoherentNeutronProbe = NuclearNeutronProbe

function _nuclear_qvector(model::ManyBodyModel, Q)
    D = model.parameters[:spatial_dimension]
    q = if Q isa CartesianWaveVector
        Float64.(Q.coordinates)
    elseif Q isa ReciprocalWaveVector
        B = get(model.parameters, :phonon_reciprocal_matrix, nothing)
        isnothing(B) && throw(ArgumentError("model does not retain reciprocal-lattice metadata"))
        Matrix{Float64}(B) * Float64.(Q.coordinates)
    else
        Float64.(collect(Q))
    end
    length(q) == D || throw(DimensionMismatch("nuclear neutron Q must have $D components"))
    return q
end

function _basis_coherent_scattering_lengths(model::ManyBodyModel)
    material = get(model.parameters, :material, nothing)
    isnothing(material) && throw(ArgumentError("phonon model does not retain Material metadata"))
    basis = material.crystal.basis
    return ComplexF64[
        material.species[site.species].nuclear_scattering_length for site in basis
    ]
end

function _basis_cartesian_positions(model::ManyBodyModel)
    direct = Matrix{Float64}(model.parameters[:phonon_direct_matrix])
    basis = model.parameters[:phonon_basis_fractional]
    return [direct * Float64.(f) for f in basis]
end

function _debye_waller_amplitude(dw, Q::AbstractVector{<:Real})
    isnothing(dw) && return 1.0
    U = dw isa AbstractMatrix ? Matrix{Float64}(dw) :
        hasproperty(dw, :U) ? Matrix{Float64}(getproperty(dw, :U)) :
        throw(ArgumentError("Debye-Waller entries must be matrices or objects with a U field"))
    size(U) == (length(Q), length(Q)) ||
        throw(DimensionMismatch("Debye-Waller tensor dimension does not match Q"))
    return exp(-0.5 * dot(Q, U * Q))
end

function _local_phonon_zero_mode_policy(result::PhononModeResult, zero_mode_tol)
    omega = Float64.(result.frequencies)
    if isnothing(zero_mode_tol)
        solver_mask = get(result.metadata, :zero_mode_mask, nothing)
        if !isnothing(solver_mask)
            length(solver_mask) == length(omega) || throw(DimensionMismatch("solver zero-mode mask does not match the phonon branch count"))
            threshold = sqrt(max(Float64(get(result.metadata, :zero_mode_eigenvalue_tolerance, 0.0)), 0.0))
            return Bool.(solver_mask), :solver, nothing, threshold
        end
        eigenvalue_tolerance = Float64(get(result.metadata, :zero_mode_eigenvalue_tolerance, 0.0))
        threshold = sqrt(max(eigenvalue_tolerance, 0.0))
        return omega .<= threshold, :solver, nothing, threshold
    end

    tolerance = Float64(zero_mode_tol)
    tolerance >= 0 || throw(ArgumentError("zero_mode_tol must be nonnegative"))
    threshold = tolerance * max(maximum(abs, omega; init=0.0), 1.0)
    return omega .<= threshold, :caller, tolerance, threshold
end

"""
    one_phonon_neutron_lines(probe, result, Q; temperature=0,
                             convention=SpectrumConvention(spectral_axis=:omega),
                             debye_waller=nothing, displacement_prefactor=nothing,
                             zero_mode_tol=nothing)

Return intrinsic coherent one-phonon creation and annihilation lines for an already q-resolved `PhononModeResult`.

This low-level method assumes that the caller has chosen the phonon momentum represented by `result`. The polarization projection and Debye-Waller factor use the supplied physical scattering vector `Q`. By default the historical basis phase also uses `Q`; `basis_phase_vector` is an advanced override used by the high-level full-`Q` route to enforce the harmonic solver's atomic-position phase convention. For extended-zone calculations, prefer the solver/model overload.

When `zero_mode_tol=nothing`, the harmonic solver's zero-mode classification is used. `displacement_prefactor` overrides the displacement scale entering `sqrt(prefactor / (2ω))`; when omitted, `convention.hbar` is used.
"""
function one_phonon_neutron_lines(
    probe::NuclearNeutronProbe,
    result::PhononModeResult,
    Q;
    temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
    debye_waller=nothing,
    displacement_prefactor::Union{Nothing,Real}=nothing,
    zero_mode_tol::Union{Nothing,Real}=nothing,
    basis_phase_vector=nothing,
)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    Qcart = _nuclear_qvector(result.model, Q)
    phase_cart = isnothing(basis_phase_vector) ? Qcart : _nuclear_qvector(result.model, basis_phase_vector)
    D = result.model.parameters[:displacement_dimension]
    length(Qcart) == D || throw(DimensionMismatch("one-phonon neutron scattering requires displacement and spatial dimensions to match"))

    nbasis = result.model.parameters[:phonon_basis_count]
    lengths = _basis_coherent_scattering_lengths(result.model)
    positions = _basis_cartesian_positions(result.model)
    length(lengths) == nbasis == length(positions) || throw(DimensionMismatch("material basis metadata is inconsistent"))

    dw = if isnothing(debye_waller)
        fill(nothing, nbasis)
    else
        values = collect(debye_waller)
        length(values) == nbasis || throw(DimensionMismatch("debye_waller must contain one tensor per primitive-cell basis site"))
        values
    end

    physical_modes = get(result.metadata, :physical_displacement_modes, nothing)
    isnothing(physical_modes) && throw(ArgumentError("PhononModeResult does not retain physical displacement modes"))
    omega = Float64.(result.frequencies)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = _local_phonon_zero_mode_policy(result, zero_mode_tol)

    centers = Float64[]
    weights = ComplexF64[]
    skipped = Int[]
    for ν in eachindex(omega)
        if zero_mask[ν]
            push!(skipped, ν)
            continue
        end

        ω = omega[ν]
        amplitude = 0.0 + 0.0im
        for site in 1:nbasis
            rows = ((site - 1) * D + 1):(site * D)
            polarization = view(physical_modes, rows, ν)
            phase = cis(dot(phase_cart, positions[site]))
            DW = _debye_waller_amplitude(dw[site], Qcart)
            amplitude += lengths[site] * DW * phase * dot(Qcart, polarization)
        end

        oscillator = sqrt(displacement_scale / (2ω))
        bare_weight = probe.prefactor * abs2(oscillator * amplitude)
        occupation = if iszero(temperature)
            0.0
        else
            x = Float64(convention.hbar) * ω / (Float64(convention.kB) * Float64(temperature))
            inv(expm1(x))
        end
        center = convention.spectral_axis === :energy ? Float64(convention.hbar) * ω : ω

        push!(centers, center)
        push!(weights, ComplexF64((occupation + 1) * bare_weight))
        if occupation > 0
            push!(centers, -center)
            push!(weights, ComplexF64(occupation * bare_weight))
        end
    end

    material = result.model.parameters[:material]
    basis_units = [
        get(material.species[site.species].metadata, :nuclear_scattering_length_unit, :unspecified)
        for site in material.crystal.basis
    ]
    scattering_length_unit = length(unique(basis_units)) == 1 ? first(basis_units) : :mixed

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :hbar => Float64(convention.hbar),
        :kB => Float64(convention.kB),
        :displacement_prefactor => displacement_scale,
        :coherent_scattering_lengths => lengths,
        :coherent_scattering_length_unit => scattering_length_unit,
        :debye_waller_applied => !isnothing(debye_waller),
        :zero_mode_source => zero_mode_source,
        :zero_mode_tolerance => zero_mode_tolerance,
        :zero_mode_frequency_threshold => zero_mode_frequency_threshold,
        :skipped_zero_modes => skipped,
        :Q_cartesian => Qcart,
        :basis_phase_cartesian => copy(phase_cart),
        :absolute_prefactor => probe.prefactor,
        :momentum_policy => :caller_supplied_local_q,
        :basis_phase_convention => isnothing(basis_phase_vector) ? :legacy_full_Q_basis_phase : :caller_supplied_basis_phase,
        :resolution_applied => false,
    )
    return LehmannLines(centers, weights, metadata)
end

function _reciprocal_decomposition_metadata!(metadata::Dict{Symbol,Any}, decomposition::ReciprocalDecomposition)
    metadata[:full_Q_cartesian] = wavevector_coordinates(decomposition.Q_cartesian)
    metadata[:full_Q_reciprocal] = wavevector_coordinates(decomposition.Q_reciprocal)
    metadata[:reduced_q_cartesian] = wavevector_coordinates(decomposition.q_cartesian)
    metadata[:reduced_q_reciprocal] = wavevector_coordinates(decomposition.q_reciprocal)
    metadata[:reciprocal_G_cartesian] = wavevector_coordinates(decomposition.G_cartesian)
    metadata[:reciprocal_G_reciprocal] = wavevector_coordinates(decomposition.G_reciprocal)
    metadata[:reciprocal_G_indices] = copy(decomposition.G_indices)
    metadata[:reciprocal_reduction_convention] = decomposition.convention
    metadata[:reciprocal_reconstruction_error] = decomposition.reconstruction_error
    metadata[:phonon_wavevector_role] = :reduced_q
    metadata[:scattering_wavevector_role] = :full_Q
    metadata[:basis_phase_convention] = :atomic_position_dynamical_matrix
    return metadata
end

"""
    one_phonon_neutron_lines(solver, model, probe, Q; reduction_convention=:first_bz,
                             reduction_atol=1e-12, temperature=0,
                             convention=SpectrumConvention(spectral_axis=:omega),
                             debye_waller=nothing, displacement_prefactor=nothing,
                             zero_mode_tol=nothing)

Evaluate coherent one-phonon neutron lines directly from the full physical scattering vector `Q`.

The API decomposes `Q = q + G` using the reciprocal-lattice metric. For the atomic-position dynamical-matrix convention used by the harmonic solver, the one-phonon form factor is evaluated with the eigenvector at `qprime = -q` and basis phase `G = Q + qprime`. The polarization projection and Debye-Waller factor retain the full physical `Q`. This combination is covariant when the reduced wavevector changes by a reciprocal-lattice vector and therefore avoids benchmark-level reciprocal wrapping.
"""
function one_phonon_neutron_lines(
    solver::HarmonicPhononSolver,
    model::ManyBodyModel,
    probe::NuclearNeutronProbe,
    Q::AbstractWaveVector;
    reduction_convention::Symbol=:first_bz,
    reduction_atol::Real=1e-12,
    temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
    debye_waller=nothing,
    displacement_prefactor::Union{Nothing,Real}=nothing,
    zero_mode_tol::Union{Nothing,Real}=nothing,
)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    decomposition = decompose_reciprocal_vector(model, Q; convention=reduction_convention, atol=reduction_atol)
    qprime = ReciprocalWaveVector(-wavevector_coordinates(decomposition.q_reciprocal))
    result = solve(solver, model, qprime)
    lines = one_phonon_neutron_lines(
        probe, result, Q; temperature=temperature, convention=convention, debye_waller=debye_waller,
        displacement_prefactor=displacement_prefactor, zero_mode_tol=zero_mode_tol, basis_phase_vector=decomposition.G_cartesian,
    )

    metadata = copy(lines.metadata)
    _reciprocal_decomposition_metadata!(metadata, decomposition)
    metadata[:momentum_transfer_q_reciprocal] = wavevector_coordinates(decomposition.q_reciprocal)
    metadata[:eigenvector_qprime_reciprocal] = wavevector_coordinates(qprime)
    metadata[:basis_phase_cartesian] = wavevector_coordinates(decomposition.G_cartesian)
    metadata[:basis_phase_reciprocal] = wavevector_coordinates(decomposition.G_reciprocal)
    metadata[:basis_phase_convention] = :atomic_position_dynamical_matrix
    metadata[:momentum_policy] = :full_Q_to_q_plus_G_with_qprime_equals_negative_q
    return LehmannLines(copy(lines.centers), copy(lines.weights), metadata)
end


"""
    one_phonon_neutron_intensity(probe, result, Q, axis;
                                 broadening, kwargs...)

Broaden the intrinsic coherent one-phonon lines on `axis`. `broadening` is a
spectral line-shape model, not an instrument resolution convolution.
"""
function one_phonon_neutron_intensity(
    probe::NuclearNeutronProbe,
    result::PhononModeResult,
    Q,
    axis;
    broadening::AbstractBroadening,
    kwargs...,
)
    lines = one_phonon_neutron_lines(probe, result, Q; kwargs...)
    values = broaden_lines(lines, axis, broadening)
    imag_scale = maximum(abs, imag.(values); init=0.0)
    real_scale = max(maximum(abs, real.(values); init=0.0), 1.0)
    imag_scale <= 1e-10 * real_scale ||
        throw(ArgumentError("coherent one-phonon spectrum acquired a significant imaginary component"))
    x = Float64.(collect(axis))
    Qcart = _nuclear_qvector(result.model, Q)
    metadata = copy(lines.metadata)
    metadata[:broadening] = typeof(broadening)
    intensity = reshape(Float64.(real.(values)), 1, length(values))
    return ScatteringResult([Qcart], x, intensity, metadata)
end

scattering_intensity(
    probe::NuclearNeutronProbe,
    result::PhononModeResult,
    Q,
    axis;
    kwargs...,
) = one_phonon_neutron_intensity(probe, result, Q, axis; kwargs...)


"""
    one_phonon_neutron_intensity(solver, model, probe, Q, axis; broadening, reduction_convention=:first_bz, reduction_atol=1e-12, kwargs...)

Evaluate a broadened coherent one-phonon spectrum from a full physical scattering vector. Reciprocal reduction and the `Q = q + G` bookkeeping are handled internally; callers do not need to solve the phonon problem at `q` themselves.
"""
function one_phonon_neutron_intensity(
    solver::HarmonicPhononSolver,
    model::ManyBodyModel,
    probe::NuclearNeutronProbe,
    Q::AbstractWaveVector,
    axis;
    broadening::AbstractBroadening,
    reduction_convention::Symbol=:first_bz,
    reduction_atol::Real=1e-12,
    kwargs...,
)
    lines = one_phonon_neutron_lines(
        solver, model, probe, Q; reduction_convention=reduction_convention, reduction_atol=reduction_atol, kwargs...,
    )
    values = broaden_lines(lines, axis, broadening)
    imag_scale = maximum(abs, imag.(values); init=0.0)
    real_scale = max(maximum(abs, real.(values); init=0.0), 1.0)
    imag_scale <= 1e-10 * real_scale || throw(ArgumentError("coherent one-phonon spectrum acquired a significant imaginary component"))
    x = Float64.(collect(axis))
    metadata = copy(lines.metadata)
    metadata[:broadening] = typeof(broadening)
    intensity = reshape(Float64.(real.(values)), 1, length(values))
    return ScatteringResult([copy(metadata[:full_Q_cartesian])], x, intensity, metadata)
end

scattering_intensity(
    solver::HarmonicPhononSolver,
    model::ManyBodyModel,
    probe::NuclearNeutronProbe,
    Q::AbstractWaveVector,
    axis;
    kwargs...,
) = one_phonon_neutron_intensity(solver, model, probe, Q, axis; kwargs...)

# -----------------------------------------------------------------------------
# High-level full-Q harmonic-scattering convenience API
# -----------------------------------------------------------------------------

function _convenience_full_Q(model::ManyBodyModel, Q::Real)
    D = model.parameters[:spatial_dimension]
    D == 1 || throw(ArgumentError(
        "scalar full-Q coordinates are only unambiguous for one-dimensional models; use ReciprocalWaveVector for D=$D",
    ))
    return ReciprocalWaveVector([Float64(Q)])
end

_convenience_full_Q(::ManyBodyModel, Q::AbstractWaveVector) = Q

function _convenience_full_Q_path(model::ManyBodyModel, Qs)
    points = collect(Qs)
    isempty(points) && throw(ArgumentError("full-Q path cannot be empty"))
    if all(point -> point isa Real, points)
        D = model.parameters[:spatial_dimension]
        D == 1 || throw(ArgumentError(
            "a scalar full-Q path is only unambiguous for one-dimensional models; use ReciprocalWaveVector points for D=$D",
        ))
        return [ReciprocalWaveVector([Float64(point)]) for point in points]
    end
    all(point -> point isa AbstractWaveVector, points) || throw(ArgumentError(
        "full-Q paths must contain explicit wavevector objects; raw multidimensional vectors are ambiguous between Cartesian and reciprocal coordinates",
    ))
    return points
end

"""
    one_phonon_neutron_lines(model, Q; solver=HarmonicPhononSolver(),
                             probe=NuclearNeutronProbe(), kwargs...)

Evaluate coherent one-phonon neutron lines from the full physical scattering
vector without explicitly constructing the harmonic solver or nuclear probe.
A scalar `Q` is reciprocal-lattice shorthand only for one-dimensional models.
"""
function one_phonon_neutron_lines(
    model::ManyBodyModel,
    Q::AbstractWaveVector;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    probe::NuclearNeutronProbe=NuclearNeutronProbe(),
    kwargs...,
)
    return one_phonon_neutron_lines(solver, model, probe, Q; kwargs...)
end

function one_phonon_neutron_lines(
    model::ManyBodyModel,
    Q::Real;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    probe::NuclearNeutronProbe=NuclearNeutronProbe(),
    kwargs...,
)
    return one_phonon_neutron_lines(solver, model, probe, _convenience_full_Q(model, Q); kwargs...)
end

"""
    one_phonon_neutron_intensity(model, Q, axis;
                                 solver=HarmonicPhononSolver(),
                                 probe=NuclearNeutronProbe(), broadening,
                                 kwargs...)

Evaluate a broadened coherent one-phonon spectrum directly from a full physical
scattering vector using the standard harmonic solver and nuclear probe.
"""
function one_phonon_neutron_intensity(
    model::ManyBodyModel,
    Q::AbstractWaveVector,
    axis;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    probe::NuclearNeutronProbe=NuclearNeutronProbe(),
    broadening::AbstractBroadening,
    kwargs...,
)
    return one_phonon_neutron_intensity(solver, model, probe, Q, axis; broadening=broadening, kwargs...)
end

function one_phonon_neutron_intensity(
    model::ManyBodyModel,
    Q::Real,
    axis;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    probe::NuclearNeutronProbe=NuclearNeutronProbe(),
    broadening::AbstractBroadening,
    kwargs...,
)
    Qpoint = _convenience_full_Q(model, Q)
    return one_phonon_neutron_intensity(solver, model, probe, Qpoint, axis; broadening=broadening, kwargs...)
end

"""
    one_phonon_neutron_intensity(model, Qpath, axis;
                                 solver=HarmonicPhononSolver(),
                                 probe=NuclearNeutronProbe(), broadening,
                                 parallel=false, kwargs...)

Evaluate a full coherent one-phonon `S(Q,E)` cut on a sequence of physical
scattering vectors. The returned `ScatteringResult.intensity` has shape
`(length(Qpath), length(axis))`. For one-dimensional models, a scalar sequence
is interpreted as reciprocal-lattice coordinates; higher-dimensional paths
must use explicit wavevector wrappers.
"""
function one_phonon_neutron_intensity(
    model::ManyBodyModel,
    Qpath,
    axis;
    solver::HarmonicPhononSolver=HarmonicPhononSolver(),
    probe::NuclearNeutronProbe=NuclearNeutronProbe(),
    broadening::AbstractBroadening,
    parallel::Bool=false,
    kwargs...,
)
    Qpoints = _convenience_full_Q_path(model, Qpath)
    first_result = one_phonon_neutron_intensity(
        solver, model, probe, first(Qpoints), axis; broadening=broadening, kwargs...,
    )
    results = Vector{typeof(first_result)}(undef, length(Qpoints))
    results[1] = first_result

    evaluate_point!(index) = begin
        results[index] = one_phonon_neutron_intensity(
            solver, model, probe, Qpoints[index], axis; broadening=broadening, kwargs...,
        )
        nothing
    end

    if length(Qpoints) > 1
        if parallel && Threads.nthreads() > 1
            Threads.@threads for index in 2:length(Qpoints)
                evaluate_point!(index)
            end
        else
            for index in 2:length(Qpoints)
                evaluate_point!(index)
            end
        end
    end

    x = copy(first_result.axis)
    nx = length(x)
    intensity = Matrix{Float64}(undef, length(results), nx)
    qcart = Vector{Vector{Float64}}(undef, length(results))
    point_metadata = Vector{Dict{Symbol,Any}}(undef, length(results))
    for index in eachindex(results)
        result = results[index]
        size(result.intensity) == (1, nx) || throw(DimensionMismatch("single-Q one-phonon result has an unexpected shape"))
        intensity[index, :] .= view(result.intensity, 1, :)
        qcart[index] = Float64.(only(result.q))
        point_metadata[index] = Dict{Symbol,Any}(result.metadata)
    end

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :vectorized_full_Q => true,
        :qpoint_count => length(Qpoints),
        :parallel => parallel && Threads.nthreads() > 1,
        :point_metadata => point_metadata,
        :broadening => typeof(broadening),
        :resolution_applied => false,
    )
    return ScatteringResult(qcart, x, intensity, metadata)
end
