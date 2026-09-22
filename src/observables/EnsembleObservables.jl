using Statistics

# EnsembleObservables.jl
#
# Observable methods for configuration ensembles. These extend the observable
# layer without changing SolverResult-based methods.

"""
Static Cartesian tensor structure factor from a classical configuration
ensemble. Values are indexed `(iq, alpha, beta)`.
"""
struct StaticTensorStructureFactorResult{Q,V,M}
    q::Q
    intensity::V
    metadata::M
end

function ensemble_expectation(
    ensemble::ConfigurationEnsemble,
    observable::Function,
)
    values = [
        observable(configuration(ensemble, r))
        for r in 1:nsamples(ensemble)
    ]
    value = mean(values)
    stderr = values[1] isa Real ? correlated_stderr(Float64.(values)) : NaN
    return (value=value, stderr=stderr)
end

ensemble_internal_energy(ensemble::ConfigurationEnsemble) = mean(ensemble.energies)

ensemble_heat_capacity(ensemble::ConfigurationEnsemble) =
    var(ensemble.energies; corrected=false) /
    (ensemble.kB * ensemble.temperature^2)

function ensemble_magnetization(
    ensemble::ConfigurationEnsemble;
    vector::Bool=false,
)
    problem = ensemble.problem
    if vector
        isnothing(problem.moment_vectors) && throw(ArgumentError(
            "vector magnetization requires IsingProblem(moment_vectors=...)"
        ))
        M = zeros(Float64, 3)
        for r in 1:nsamples(ensemble)
            s = view(ensemble.configurations, :, r)
            M .+= problem.moment_vectors * s
        end
        return M ./ nsamples(ensemble)
    end

    return mean(
        sum(view(ensemble.configurations, :, r)) / length(problem)
        for r in 1:nsamples(ensemble)
    )
end

function _ensemble_phase(q::Number, r::Number)
    return cis(-Float64(q) * Float64(r))
end

function _ensemble_phase(q, r)
    length(q) == length(r) ||
        throw(DimensionMismatch("q and position must have equal dimensions"))
    return cis(-sum(Float64(qi) * Float64(ri) for (qi, ri) in zip(q, r)))
end

function _ensemble_positions(problem::IsingProblem, positions)
    pos = isnothing(positions) ? problem.positions : positions
    isnothing(pos) && throw(ArgumentError(
        "positions must be supplied either to IsingProblem or static_structure_factor"
    ))
    length(pos) == length(problem) ||
        throw(DimensionMismatch("positions must contain one entry per site"))
    return pos
end

"""
    static_structure_factor(ensemble, qgrid; positions=nothing,
                              normalization=:sqrtN, connected=false)

Scalar Ising equal-time structure factor

    S(q) = < |a_N sum_i s_i exp(-iq.r_i)|^2 >.

With `connected=true`, the elastic/disconnected contribution
`|<S(q)>|^2` is subtracted.
"""
function static_structure_factor(
    ensemble::ConfigurationEnsemble,
    qgrid;
    positions=nothing,
    normalization::Symbol=:sqrtN,
    connected::Bool=false,
)
    problem = ensemble.problem
    problem isa IsingProblem ||
        throw(ArgumentError("this ensemble structure-factor method currently requires IsingProblem"))
    pos = _ensemble_positions(problem, positions)
    qs = collect(qgrid)
    N = length(problem)
    aN = _fourier_amplitude(normalization, N)
    values = zeros(Float64, length(qs))

    for (iq, q) in enumerate(qs)
        mean_amp = 0.0 + 0.0im
        mean_abs2 = 0.0
        for r in 1:nsamples(ensemble)
            s = view(ensemble.configurations, :, r)
            amp = 0.0 + 0.0im
            @inbounds for i in 1:N
                amp += aN * s[i] * _ensemble_phase(q, pos[i])
            end
            mean_amp += amp
            mean_abs2 += abs2(amp)
        end
        mean_amp /= nsamples(ensemble)
        mean_abs2 /= nsamples(ensemble)
        values[iq] = connected ? max(mean_abs2 - abs2(mean_amp), 0.0) : mean_abs2
    end

    metadata = Dict{Symbol,Any}(
        :observable => :ensemble_static_structure_factor,
        :temperature => ensemble.temperature,
        :fourier_normalization => normalization,
        :connected => connected,
        :samples => nsamples(ensemble),
        :classical_ensemble => true,
    )
    return StaticStructureFactorResult(qs, values, metadata)
end

"""
    spin_tensor_static_structure_factor(ensemble, qgrid; ...)

Classical Cartesian tensor

    S_ab(q) = < M_a(q) M_b(-q) >

for local-Ising configurations. `moment_vectors[:,i]` is the physical Cartesian
moment corresponding to `s_i=+1`.

The result can be passed directly to `scattering_intensity(MagneticNeutronProbe(), ...)`.
"""
function spin_tensor_static_structure_factor(
    ensemble::ConfigurationEnsemble,
    qgrid;
    positions=nothing,
    moment_vectors=nothing,
    normalization::Symbol=:sqrtN,
    connected::Bool=false,
)
    problem = ensemble.problem
    problem isa IsingProblem ||
        throw(ArgumentError("spin tensor ensemble method currently requires IsingProblem"))
    pos = _ensemble_positions(problem, positions)

    moments = isnothing(moment_vectors) ? problem.moment_vectors : moment_vectors
    isnothing(moments) && throw(ArgumentError(
        "moment_vectors must be supplied either to IsingProblem or this function"
    ))
    moments = Matrix{Float64}(moments)
    size(moments) == (3, length(problem)) ||
        throw(DimensionMismatch("moment_vectors must be 3×N"))

    qs = collect(qgrid)
    N = length(problem)
    aN = _fourier_amplitude(normalization, N)
    tensor = zeros(ComplexF64, length(qs), 3, 3)

    for (iq, q) in enumerate(qs)
        meanM = zeros(ComplexF64, 3)
        second = zeros(ComplexF64, 3, 3)

        for r in 1:nsamples(ensemble)
            s = view(ensemble.configurations, :, r)
            Mq = zeros(ComplexF64, 3)
            @inbounds for i in 1:N
                phase = aN * s[i] * _ensemble_phase(q, pos[i])
                Mq .+= phase .* view(moments, :, i)
            end
            meanM .+= Mq
            second .+= Mq * adjoint(Mq)
        end

        meanM ./= nsamples(ensemble)
        second ./= nsamples(ensemble)
        connected && (second .-= meanM * adjoint(meanM))
        @views tensor[iq, :, :] .= second
    end

    metadata = Dict{Symbol,Any}(
        :observable => :ensemble_spin_tensor_static_structure_factor,
        :temperature => ensemble.temperature,
        :fourier_normalization => normalization,
        :connected => connected,
        :samples => nsamples(ensemble),
        :classical_ensemble => true,
    )
    return StaticTensorStructureFactorResult(qs, tensor, metadata)
end

function magnetic_neutron_intensity(
    probe::MagneticNeutronProbe,
    spectrum::StaticTensorStructureFactorResult,
)
    nq, na, nb = size(spectrum.intensity)
    na == nb == 3 ||
        throw(DimensionMismatch("static spin tensor must have 3×3 Cartesian components"))

    out = zeros(Float64, nq)
    g = ComplexF64.(probe.g_tensor)

    for iq in 1:nq
        Q = _qvector(spectrum.q[iq])
        qnorm = norm(Q)
        qnorm > 0 || throw(ArgumentError(
            "magnetic polarization tensor is undefined at Q=0"
        ))
        qhat = Q / qnorm
        P = Matrix{Float64}(I, 3, 3) - qhat * transpose(qhat)
        F2 = abs2(probe.form_factor(spectrum.q[iq]))

        @views S = spectrum.intensity[iq, :, :]
        Seff = g * S * adjoint(g)
        value = sum(P .* Seff)
        scale = max(abs(real(value)), 1.0)
        abs(imag(value)) <= 1e-9 * scale ||
            throw(ArgumentError(
                "static magnetic tensor contraction produced significant imaginary intensity"
            ))
        out[iq] = probe.prefactor * F2 * real(value)
    end

    metadata = copy(spectrum.metadata)
    metadata[:probe] = :magnetic_neutron
    metadata[:polarization_projection] = :transverse
    metadata[:form_factor_applied] = true
    metadata[:g_tensor_applied] = true
    return StaticStructureFactorResult(spectrum.q, out, metadata)
end

scattering_intensity(
    probe::MagneticNeutronProbe,
    spectrum::StaticTensorStructureFactorResult,
) = magnetic_neutron_intensity(probe, spectrum)
