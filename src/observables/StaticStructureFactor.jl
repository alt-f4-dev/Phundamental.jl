# Equal-time/static structure factors.

"""
    static_structure_factor(model, result, field, qgrid; ...)

Calculate the equal-time quantity

    S(q) = < O_q O_-q >

with the same Fourier normalization used by the dynamic structure factor. For the normalization adopted here it also equals the full-axis integral of `S(q,x)` before finite-window loss and line-broadening discretization errors.
"""
function static_structure_factor(model::ManyBodyModel, result::SolverResult, field::MomentumField, qgrid;
                                 temperature::Real=0.0,
                                 convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.normalization),
                                 degeneracy_tol::Real=1e-10)
    convention.fourier_normalization == field.normalization || throw(ArgumentError("SpectrumConvention Fourier normalization must match MomentumField"))
    p = thermal_probabilities(result; temperature=temperature, kB=convention.kB, degeneracy_tol=degeneracy_tol)
    qs = collect(qgrid)
    values = zeros(ComplexF64, length(qs))

    for (iq, q) in enumerate(qs)
        A = eigenbasis_matrix(model, result, momentum_operator(model, field, q); hbar=convention.hbar)
        B = eigenbasis_matrix(model, result, momentum_operator(model, field, _negate_q(q)); hbar=convention.hbar)
        total = 0.0 + 0im
        @inbounds for m in eachindex(p), n in eachindex(p)
            total += p[m] * A[m, n] * B[n, m]
        end
        values[iq] = total
    end

    metadata = Dict{Symbol,Any}(
        :observable => :static_structure_factor,
        :temperature => float(temperature),
        :fourier_normalization => field.normalization,
        :sum_rule => :integral_of_dynamic_structure_factor,
        :representation => model.representation.name,
    )
    return StaticStructureFactorResult(qs, values, metadata)
end

"""
    static_structure_factor(model, result::QuadraticModeResult, field, qgrid; ...)

Calculate `S(q) = <O_q O_-q>` directly from a quadratic bosonic mode result. Momentum-field operators may be affine-linear or quadratic in the source bosons. The static result is obtained by summing the exact unbroadened Gaussian Lehmann weights, so it closes the dynamic/static sum rule before finite spectral-window and discretization effects.
"""
function static_structure_factor(model::ManyBodyModel, result::QuadraticModeResult, field::MomentumField, qgrid;
                                 temperature::Real=0.0,
                                 convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.normalization),
                                 degeneracy_tol::Real=1e-10, weight_tol::Real=0.0,
                                 zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes())
    convention.fourier_normalization == field.normalization || throw(ArgumentError("SpectrumConvention Fourier normalization must match MomentumField"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))

    qs = collect(qgrid)
    values = zeros(ComplexF64, length(qs))
    observable_order = :affine_linear
    projected_zero_modes = Set{Int}()

    for (iq, q) in enumerate(qs)
        source_q = _momentum_boson_quadratic_coefficients(model, field, q)
        source_minus = _momentum_boson_quadratic_coefficients(model, field, _negate_q(q))
        projected_q = _quadratic_mode_projection(model, result, source_q)
        projected_minus = _quadratic_mode_projection(model, result, source_minus)
        lines = _quadratic_mode_lehmann_lines(result, projected_q, projected_minus; temperature=temperature, convention=convention,
                                              degeneracy_tol=degeneracy_tol, weight_tol=weight_tol, zero_mode_policy=zero_mode_policy)
        values[iq] = isempty(lines.weights) ? 0.0 + 0.0im : sum(lines.weights)
        lines.metadata[:observable_order] === :quadratic && (observable_order = :quadratic)
        union!(projected_zero_modes, lines.metadata[:projected_zero_modes])
    end

    metadata = Dict{Symbol,Any}(
        :observable => :static_structure_factor,
        :temperature => float(temperature),
        :fourier_normalization => field.normalization,
        :sum_rule => :integral_of_dynamic_structure_factor,
        :representation => model.representation.name,
        :solver => get(result.metadata, :solver, :unknown),
        :mode_native => true,
        :mode_space => get(result.metadata, :mode_space, :unknown),
        :observable_class => observable_order === :quadratic ? :gaussian_boson_quadratic : :linear_boson,
        :observable_order => observable_order,
        :zero_mode_policy => Symbol(nameof(typeof(zero_mode_policy))),
        :projected_zero_modes => sort!(collect(projected_zero_modes)),
    )
    return StaticStructureFactorResult(qs, values, metadata)
end
