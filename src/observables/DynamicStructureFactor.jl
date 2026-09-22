# Lehmann spectral functions and dynamic structure factors.

struct LorentzianBroadening{T<:Real} <: AbstractBroadening
    width::T
    function LorentzianBroadening(width::Real)
        width > 0 || throw(ArgumentError("Lorentzian width must be positive"))
        new{typeof(float(width))}(float(width))
    end
end

struct GaussianBroadening{T<:Real} <: AbstractBroadening
    sigma::T
    function GaussianBroadening(sigma::Real)
        sigma > 0 || throw(ArgumentError("Gaussian sigma must be positive"))
        new{typeof(float(sigma))}(float(sigma))
    end
end

_kernel(b::LorentzianBroadening, x) = (b.width / π) ./ (x .* x .+ b.width^2)
_kernel(b::GaussianBroadening, x) = exp.(-0.5 .* (x ./ b.sigma).^2) ./ (sqrt(2π) * b.sigma)
_kernel_value(b::LorentzianBroadening, x::Real) = (b.width / π) / (x * x + b.width^2)
_kernel_value(b::GaussianBroadening, x::Real) = exp(-0.5 * (x / b.sigma)^2) / (sqrt(2π) * b.sigma)

mutable struct _BosonQuadraticPolynomial
    constant::ComplexF64
    annihilation::Vector{ComplexF64}
    creation::Vector{ComplexF64}
    normal::Union{Nothing,Matrix{ComplexF64}}
    pair_annihilation::Union{Nothing,Matrix{ComplexF64}}
    pair_creation::Union{Nothing,Matrix{ComplexF64}}
end

function _BosonQuadraticPolynomial(M::Int)
    M >= 0 || throw(ArgumentError("boson polynomial dimension must be nonnegative"))
    return _BosonQuadraticPolynomial(0.0 + 0.0im, zeros(ComplexF64, M), zeros(ComplexF64, M), nothing, nothing, nothing)
end

function _ensure_normal!(polynomial::_BosonQuadraticPolynomial)
    if isnothing(polynomial.normal)
        M = length(polynomial.annihilation)
        polynomial.normal = zeros(ComplexF64, M, M)
    end
    return polynomial.normal
end

function _ensure_pair_annihilation!(polynomial::_BosonQuadraticPolynomial)
    if isnothing(polynomial.pair_annihilation)
        M = length(polynomial.annihilation)
        polynomial.pair_annihilation = zeros(ComplexF64, M, M)
    end
    return polynomial.pair_annihilation
end

function _ensure_pair_creation!(polynomial::_BosonQuadraticPolynomial)
    if isnothing(polynomial.pair_creation)
        M = length(polynomial.annihilation)
        polynomial.pair_creation = zeros(ComplexF64, M, M)
    end
    return polynomial.pair_creation
end

function _numeric_boson_coefficient(value)
    try
        return ComplexF64(value)
    catch
        throw(ArgumentError("mode-native bosonic response requires numerical observable coefficients"))
    end
end

function _boson_factor_degree(factor::AbstractOperatorExpr)
    factor isa IdentityOperator && return 0
    factor isa BosonAnnihilate && return 1
    factor isa BosonCreate && return 1
    factor isa NumberOperator && factor.kind === :boson && return 2
    throw(ArgumentError("mode-native Gaussian bosonic response supports only boson constants, b, b†, and quadratic boson products; found $(typeof(factor))"))
end

function _expanded_boson_monomials(expr::AbstractOperatorExpr)
    if expr isa IdentityOperator || expr isa BosonAnnihilate || expr isa BosonCreate || expr isa NumberOperator
        return [(1.0 + 0.0im, AbstractOperatorExpr[expr])]
    elseif expr isa ScaledOperator
        factor = _numeric_boson_coefficient(expr.coefficient)
        return [(factor * coefficient, factors) for (coefficient, factors) in _expanded_boson_monomials(expr.operator)]
    elseif expr isa OperatorSum
        terms = Tuple{ComplexF64,Vector{AbstractOperatorExpr}}[]
        for term in expr.terms
            append!(terms, _expanded_boson_monomials(term))
        end
        return terms
    elseif expr isa OperatorProduct
        terms = Tuple{ComplexF64,Vector{AbstractOperatorExpr}}[(1.0 + 0.0im, AbstractOperatorExpr[])]
        for factor_expr in expr.factors
            parts = _expanded_boson_monomials(factor_expr)
            expanded = Tuple{ComplexF64,Vector{AbstractOperatorExpr}}[]
            for (left_coefficient, left_factors) in terms, (right_coefficient, right_factors) in parts
                factors = vcat(left_factors, right_factors)
                degree = sum(_boson_factor_degree(factor) for factor in factors)
                degree <= 2 || throw(ArgumentError("mode-native Gaussian bosonic response supports observables of degree at most two in b and b†"))
                push!(expanded, (left_coefficient * right_coefficient, factors))
            end
            terms = expanded
        end
        return terms
    end

    throw(ArgumentError("mode-native Gaussian bosonic response does not support observable node $(typeof(expr))"))
end

function _accumulate_pair_symmetric!(matrix::Matrix{ComplexF64}, i::Int, j::Int, coefficient::ComplexF64)
    if i == j
        matrix[i, i] += 2coefficient
    else
        matrix[i, j] += coefficient
        matrix[j, i] += coefficient
    end
    return matrix
end

function _accumulate_boson_expression!(polynomial::_BosonQuadraticPolynomial, expr::AbstractOperatorExpr, scale::ComplexF64=1.0 + 0.0im)
    M = length(polynomial.annihilation)
    for (coefficient0, factors0) in _expanded_boson_monomials(expr)
        coefficient = scale * coefficient0
        factors = [factor for factor in factors0 if !(factor isa IdentityOperator)]

        if isempty(factors)
            polynomial.constant += coefficient
            continue
        end

        if length(factors) == 1
            factor = factors[1]
            if factor isa BosonAnnihilate
                1 <= factor.mode <= M || throw(BoundsError(1:M, factor.mode))
                polynomial.annihilation[factor.mode] += coefficient
            elseif factor isa BosonCreate
                1 <= factor.mode <= M || throw(BoundsError(1:M, factor.mode))
                polynomial.creation[factor.mode] += coefficient
            elseif factor isa NumberOperator && factor.kind === :boson
                1 <= factor.mode <= M || throw(BoundsError(1:M, factor.mode))
                _ensure_normal!(polynomial)[factor.mode, factor.mode] += coefficient
            else
                throw(ArgumentError("unsupported bosonic observable factor $(typeof(factor))"))
            end
            continue
        end

        length(factors) == 2 || throw(ArgumentError("mode-native Gaussian bosonic response supports degree at most two"))
        first_factor, second_factor = factors
        first_factor isa Union{BosonAnnihilate,BosonCreate} || throw(ArgumentError("quadratic bosonic products must contain b or b† primitives"))
        second_factor isa Union{BosonAnnihilate,BosonCreate} || throw(ArgumentError("quadratic bosonic products must contain b or b† primitives"))
        i = first_factor.mode
        j = second_factor.mode
        1 <= i <= M || throw(BoundsError(1:M, i))
        1 <= j <= M || throw(BoundsError(1:M, j))

        if first_factor isa BosonCreate && second_factor isa BosonAnnihilate
            _ensure_normal!(polynomial)[i, j] += coefficient
        elseif first_factor isa BosonAnnihilate && second_factor isa BosonCreate
            i == j && (polynomial.constant += coefficient)
            _ensure_normal!(polynomial)[j, i] += coefficient
        elseif first_factor isa BosonAnnihilate && second_factor isa BosonAnnihilate
            _accumulate_pair_symmetric!(_ensure_pair_annihilation!(polynomial), i, j, coefficient)
        else
            _accumulate_pair_symmetric!(_ensure_pair_creation!(polynomial), i, j, coefficient)
        end
    end
    return polynomial
end

function _boson_quadratic_coefficients(model::ManyBodyModel, op::AbstractOperatorExpr)
    model.representation.algebra isa BosonAlgebra || throw(ArgumentError("mode-native Gaussian bosonic response requires a BosonAlgebra model"))
    polynomial = _BosonQuadraticPolynomial(model.representation.algebra.nmodes)
    return _accumulate_boson_expression!(polynomial, op)
end

function _momentum_boson_quadratic_coefficients(model::ManyBodyModel, field::MomentumField, q)
    model.representation.algebra isa BosonAlgebra || throw(ArgumentError("mode-native Gaussian bosonic response requires a BosonAlgebra model"))
    M = model.representation.algebra.nmodes
    polynomial = _BosonQuadraticPolynomial(M)
    amplitude = _fourier_amplitude(field.normalization, length(field.keys))

    for (key, position) in zip(field.keys, field.positions)
        op = key isa ObservableRef ? resolve_observable(model, key) : key isa Symbol ? resolve_observable(model, key) :
             key isa AbstractOperatorExpr ? key : throw(ArgumentError("MomentumField keys must be Symbol, ObservableRef, or operator expression"))
        scale = ComplexF64(amplitude * _phase(q, position))
        _accumulate_boson_expression!(polynomial, op, scale)
    end

    return polynomial
end

function _quadratic_mode_data(model::ManyBodyModel, result::QuadraticModeResult)
    model.representation.algebra isa BosonAlgebra || throw(ArgumentError("mode-native bosonic response requires a BosonAlgebra model"))
    result.solver isa QuadraticBosonSolver || throw(ArgumentError("mode-native bosonic response requires a QuadraticBosonSolver result"))
    result.model.representation.algebra isa BosonAlgebra || throw(ArgumentError("QuadraticModeResult does not retain a bosonic source model"))

    M = model.representation.algebra.nmodes
    result.model.representation.algebra.nmodes == M || throw(DimensionMismatch(
        "model and QuadraticModeResult contain different numbers of bosonic source modes"
    ))

    energies = Float64.(mode_energies(result))
    K = length(energies)
    modes = mode_vectors(result)
    size(modes, 2) == K || throw(DimensionMismatch("QuadraticModeResult mode-vector count does not match its energy count"))

    scale = isempty(energies) ? 1.0 : max(maximum(abs, energies), 1.0)
    if !isempty(energies)
        minimum(energies) >= -result.solver.tol * scale || throw(ArgumentError("mode-native bosonic response requires nonnegative mode energies"))
        energies .= max.(energies, 0.0)
    end

    if size(modes, 1) == M
        return energies, modes, nothing
    elseif size(modes, 1) == 2M
        return energies, view(modes, 1:M, :), view(modes, M+1:2M, :)
    end

    throw(DimensionMismatch("QuadraticModeResult modes must have M or 2M rows for a bosonic response"))
end

function _quadratic_mode_projection(model::ManyBodyModel, result::QuadraticModeResult, source::_BosonQuadraticPolynomial)
    energies, U, V = _quadratic_mode_data(model, result)
    K = length(energies)

    annihilation = transpose(U) * source.annihilation
    creation = adjoint(U) * source.creation
    if !isnothing(V)
        annihilation .+= transpose(V) * source.creation
        creation .+= adjoint(V) * source.annihilation
    end

    constant = source.constant
    normal = nothing
    pair_annihilation = nothing
    pair_creation = nothing
    has_quadratic = !isnothing(source.normal) || !isnothing(source.pair_annihilation) || !isnothing(source.pair_creation)

    if has_quadratic
        normal = zeros(ComplexF64, K, K)
        pair_annihilation = zeros(ComplexF64, K, K)
        pair_creation = zeros(ComplexF64, K, K)

        if !isnothing(source.normal)
            N = source.normal
            normal .+= adjoint(U) * N * U
            if !isnothing(V)
                C = transpose(V) * N * conj.(V)
                normal .+= transpose(C)
                annpair = transpose(V) * N * U
                crepair = adjoint(U) * N * conj.(V)
                pair_annihilation .+= annpair .+ transpose(annpair)
                pair_creation .+= crepair .+ transpose(crepair)
                constant += tr(C)
            end
        end

        if !isnothing(source.pair_annihilation)
            P = source.pair_annihilation
            pair_annihilation .+= transpose(U) * P * U
            if !isnothing(V)
                pair_creation .+= adjoint(V) * P * conj.(V)
                normal .+= adjoint(V) * P * U
                constant += 0.5 * tr(transpose(U) * P * conj.(V))
            end
        end

        if !isnothing(source.pair_creation)
            Q = source.pair_creation
            pair_creation .+= adjoint(U) * Q * conj.(U)
            if !isnothing(V)
                pair_annihilation .+= transpose(V) * Q * V
                normal .+= adjoint(U) * Q * V
                constant += 0.5 * tr(transpose(V) * Q * conj.(U))
            end
        end
    end

    return (constant=constant, annihilation=annihilation, creation=creation, normal=normal,
            pair_annihilation=pair_annihilation, pair_creation=pair_creation, energies=energies,
            order=has_quadratic ? :quadratic : :affine_linear)
end

function _quadratic_mode_projection(model::ManyBodyModel, result::QuadraticModeResult, op::AbstractOperatorExpr)
    return _quadratic_mode_projection(model, result, _boson_quadratic_coefficients(model, op))
end

function _quadratic_bose_thermal_data(energies::Vector{Float64}, temperature::Real, kB::Real;
                                      tol::Real, zero_mode_policy::AbstractBosonZeroModePolicy)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    occupations = zeros(Float64, length(energies))
    active = trues(length(energies))
    projected = Int[]
    isempty(energies) && return occupations, active, projected

    scale = max(maximum(abs, energies), 1.0)
    threshold = max(Float64(tol), _zero_mode_relative_tol(zero_mode_policy)) * scale
    zero_modes = findall(energy -> energy <= threshold, energies)

    if zero_mode_policy isa RejectBosonZeroModes
        if !iszero(temperature) && !isempty(zero_modes)
            throw(ArgumentError(
                "finite-temperature mode-native bosonic response contains zero/soft modes at indices $(zero_modes); " *
                "use a solver-side BosonSubspaceProjection or explicitly request NumberConservingZeroModeProjection when those modes are known " *
                "condensate/Goldstone directions"
            ))
        end
    elseif zero_mode_policy isa NumberConservingZeroModeProjection
        active[zero_modes] .= false
        append!(projected, zero_modes)
    else
        throw(ArgumentError("unsupported bosonic zero-mode policy $(typeof(zero_mode_policy))"))
    end

    iszero(temperature) && return occupations, active, projected
    β = inv(float(kB) * float(temperature))
    @inbounds for ν in eachindex(energies)
        active[ν] || continue
        occupations[ν] = inv(expm1(β * energies[ν]))
    end

    return occupations, active, projected
end

function _push_line!(centers::Vector{Float64}, weights::Vector{ComplexF64}, center::Real, weight, weight_tol::Real)
    abs(weight) <= weight_tol && return nothing
    push!(centers, Float64(center))
    push!(weights, ComplexF64(weight))
    return nothing
end

function _coalesce_lehmann_lines(centers::Vector{Float64}, weights::Vector{ComplexF64}; atol::Real, weight_tol::Real)
    isempty(centers) && return centers, weights
    order = sortperm(centers)
    merged_centers = Float64[]
    merged_weights = ComplexF64[]
    current_center = centers[order[1]]
    current_weight = weights[order[1]]

    for index in order[2:end]
        center = centers[index]
        weight = weights[index]
        if abs(center - current_center) <= atol
            total_weight = current_weight + weight
            if !iszero(total_weight)
                current_center = (current_center * abs(current_weight) + center * abs(weight)) / max(abs(current_weight) + abs(weight), eps(Float64))
            end
            current_weight = total_weight
        else
            abs(current_weight) > weight_tol && (push!(merged_centers, current_center); push!(merged_weights, current_weight))
            current_center = center
            current_weight = weight
        end
    end
    abs(current_weight) > weight_tol && (push!(merged_centers, current_center); push!(merged_weights, current_weight))
    return merged_centers, merged_weights
end

function _quadratic_mode_lehmann_lines(result::QuadraticModeResult, projected_A, projected_B;
                                       temperature::Real, convention::SpectrumConvention, degeneracy_tol::Real,
                                       weight_tol::Real, zero_mode_policy::AbstractBosonZeroModePolicy)
    projected_A.energies == projected_B.energies || throw(DimensionMismatch("projected mode energies differ between observables"))
    energies = projected_A.energies
    occupations, active, projected_zero_modes = _quadratic_bose_thermal_data(
        energies, temperature, convention.kB; tol=result.solver.tol, zero_mode_policy=zero_mode_policy,
    )
    axis_scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    centers = Float64[]
    weights = ComplexF64[]

    mean_A = projected_A.constant
    mean_B = projected_B.constant
    if !isnothing(projected_A.normal)
        @inbounds for i in eachindex(energies)
            active[i] && (mean_A += projected_A.normal[i, i] * occupations[i])
        end
    end
    if !isnothing(projected_B.normal)
        @inbounds for i in eachindex(energies)
            active[i] && (mean_B += projected_B.normal[i, i] * occupations[i])
        end
    end
    _push_line!(centers, weights, 0.0, mean_A * mean_B, weight_tol)

    @inbounds for i in eachindex(energies)
        active[i] || continue
        occupation = occupations[i]
        _push_line!(centers, weights, energies[i] * axis_scale,
                    (occupation + 1) * projected_A.annihilation[i] * projected_B.creation[i], weight_tol)
        _push_line!(centers, weights, -energies[i] * axis_scale,
                    occupation * projected_A.creation[i] * projected_B.annihilation[i], weight_tol)
    end

    if !isnothing(projected_A.normal) && !isnothing(projected_B.normal)
        @inbounds for i in eachindex(energies), j in eachindex(energies)
            active[i] && active[j] || continue
            weight = projected_A.normal[i, j] * projected_B.normal[j, i] * occupations[i] * (occupations[j] + 1)
            _push_line!(centers, weights, (energies[j] - energies[i]) * axis_scale, weight, weight_tol)
        end
    end

    if !isnothing(projected_A.pair_annihilation) && !isnothing(projected_B.pair_creation)
        @inbounds for i in eachindex(energies), j in eachindex(energies)
            active[i] && active[j] || continue
            weight = 0.5 * projected_A.pair_annihilation[i, j] * projected_B.pair_creation[i, j] *
                     (occupations[i] + 1) * (occupations[j] + 1)
            _push_line!(centers, weights, (energies[i] + energies[j]) * axis_scale, weight, weight_tol)
        end
    end

    if !isnothing(projected_A.pair_creation) && !isnothing(projected_B.pair_annihilation)
        @inbounds for i in eachindex(energies), j in eachindex(energies)
            active[i] && active[j] || continue
            weight = 0.5 * projected_A.pair_creation[i, j] * projected_B.pair_annihilation[i, j] * occupations[i] * occupations[j]
            _push_line!(centers, weights, -(energies[i] + energies[j]) * axis_scale, weight, weight_tol)
        end
    end

    line_scale = isempty(centers) ? 1.0 : max(maximum(abs, centers), 1.0)
    centers, weights = _coalesce_lehmann_lines(centers, weights; atol=degeneracy_tol * line_scale, weight_tol=weight_tol)
    metadata = Dict{Symbol,Any}(
        :temperature => float(temperature),
        :kB => convention.kB,
        :hbar => convention.hbar,
        :spectral_axis => convention.spectral_axis,
        :normalization => :unit_integral_delta,
        :definition => :quadratic_mode_gaussian_lehmann,
        :mode_native => true,
        :mode_space => get(result.metadata, :mode_space, :unknown),
        :operator_order => :A_then_B,
        :observable_order => projected_A.order === :quadratic || projected_B.order === :quadratic ? :quadratic : :affine_linear,
        :zero_mode_policy => Symbol(nameof(typeof(zero_mode_policy))),
        :projected_zero_modes => projected_zero_modes,
        :time_fourier_prefactor => convention.spectral_axis === :energy ? :one_over_2pi_hbar : :one_over_2pi,
    )
    return LehmannLines(centers, weights, metadata)
end

"""
    lehmann_lines(model, result, A, B; ...)

Return the discrete transitions for

    S_AB(x) = sum_mn p_m A_mn B_nm delta(x - x_nm),

where `x_nm = E_n-E_m` for `spectral_axis=:energy` and `x_nm=(E_n-E_m)/hbar` for `:omega`.

This convention satisfies `integral dx S_AB(x) = <A B>` before finite-grid windowing. For `B=A^dagger` in a Hermitian theory the weights are nonnegative up to floating-point error.
"""
function lehmann_lines(model::ManyBodyModel, result::SolverResult, A, B; temperature::Real=0.0,
                       convention::SpectrumConvention=SpectrumConvention(), degeneracy_tol::Real=1e-10,
                       weight_tol::Real=0.0)
    E = _real_energies(result)
    p = thermal_probabilities(result; temperature=temperature, kB=convention.kB, degeneracy_tol=degeneracy_tol)
    Ae = eigenbasis_matrix(model, result, A; hbar=convention.hbar)
    Be = eigenbasis_matrix(model, result, B; hbar=convention.hbar)

    centers = Float64[]
    weights = ComplexF64[]
    sizehint!(centers, length(E)^2)
    sizehint!(weights, length(E)^2)

    axis_scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))
    @inbounds for m in eachindex(E)
        pm = p[m]
        pm <= weight_tol && continue
        for n in eachindex(E)
            wmn = pm * Ae[m, n] * Be[n, m]
            abs(wmn) <= weight_tol && continue
            push!(centers, (E[n] - E[m]) * axis_scale)
            push!(weights, ComplexF64(wmn))
        end
    end

    metadata = Dict{Symbol,Any}(
        :temperature => float(temperature),
        :kB => convention.kB,
        :hbar => convention.hbar,
        :spectral_axis => convention.spectral_axis,
        :normalization => :unit_integral_delta,
        :definition => :lehmann,
        :time_fourier_prefactor => convention.spectral_axis === :energy ? :one_over_2pi_hbar : :one_over_2pi,
        :biorthogonal => isbiorthogonal(result),
    )
    return LehmannLines(centers, weights, metadata)
end

"""
    lehmann_lines(model, result::QuadraticModeResult, A, B; ...)

Return exact Gaussian Lehmann lines generated by bosonic observables of degree at most two in the source operators. Affine-linear observables generate one-quasiparticle lines, while quadratic observables additionally generate elastic, thermal scattering, and two-quasiparticle creation/annihilation channels. The calculation acts directly in mode space and constructs no truncated many-body Fock basis.
"""
function lehmann_lines(model::ManyBodyModel, result::QuadraticModeResult, A::Union{Symbol,ObservableRef,AbstractOperatorExpr},
                       B::Union{Symbol,ObservableRef,AbstractOperatorExpr}; temperature::Real=0.0,
                       convention::SpectrumConvention=SpectrumConvention(), degeneracy_tol::Real=1e-10,
                       weight_tol::Real=0.0, zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes())
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    projected_A = _quadratic_mode_projection(model, result, resolve_observable(model, A))
    projected_B = _quadratic_mode_projection(model, result, resolve_observable(model, B))
    return _quadratic_mode_lehmann_lines(result, projected_A, projected_B; temperature=temperature, convention=convention,
                                         degeneracy_tol=degeneracy_tol, weight_tol=weight_tol, zero_mode_policy=zero_mode_policy)
end

function _check_axis_grid(axis)
    x = collect(float.(axis))
    length(x) >= 2 || throw(ArgumentError("spectral grid requires at least two points"))
    all(diff(x) .> 0) || throw(ArgumentError("spectral grid must be strictly increasing"))
    return x
end

"""
    broaden_lines(lines, axis, broadening)

Numerically represent the intrinsic delta-function spectrum with a normalized line-shape kernel. This is spectral line broadening, not instrument resolution. Instrument response is applied separately by `convolve_resolution`.
"""
function broaden_lines(lines::LehmannLines, axis, broadening::AbstractBroadening)
    x = _check_axis_grid(axis)
    out = zeros(ComplexF64, length(x))
    @inbounds for (center, weight) in zip(lines.centers, lines.weights)
        for i in eachindex(x)
            out[i] += weight * _kernel_value(broadening, x[i] - center)
        end
    end
    return out
end

"""
    dynamic_structure_factor(model, result, field, qgrid, axis; ...)

Compute `S(q,x)` using the registered representation-correct local operators in `field`. The default `B=O_-q` yields the conventional two-point dynamic structure factor. The result is per unit of the selected spectral axis.
"""
function dynamic_structure_factor(model::ManyBodyModel, result::SolverResult, field::MomentumField, qgrid, axis;
                                  temperature::Real=0.0,
                                  convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.normalization),
                                  broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                  weight_tol::Real=0.0)
    convention.fourier_normalization == field.normalization || throw(ArgumentError(
        "SpectrumConvention Fourier normalization must match MomentumField normalization"
    ))

    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    values = zeros(ComplexF64, length(qs), length(x))

    for (iq, q) in enumerate(qs)
        Oq = momentum_operator(model, field, q)
        Ominus = momentum_operator(model, field, _negate_q(q))
        lines = lehmann_lines(model, result, Oq, Ominus; temperature=temperature, convention=convention,
                              degeneracy_tol=degeneracy_tol, weight_tol=weight_tol)
        values[iq, :] .= broaden_lines(lines, x, broadening)
    end

    metadata = Dict{Symbol,Any}(
        :observable => :dynamic_structure_factor,
        :temperature => float(temperature),
        :fourier_normalization => field.normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :representation => model.representation.name,
        :solver => get(result.metadata, :solver, :unknown),
    )
    return SpectrumResult(qs, x, values, metadata)
end

"""
    dynamic_structure_factor(model, result::QuadraticModeResult, field, qgrid, axis; ...)

Compute a Gaussian bosonic dynamic structure factor directly from `QuadraticModeResult`. Momentum-field operators may be affine-linear or quadratic in the source bosons. The route includes one-quasiparticle, thermal scattering, and two-quasiparticle channels as required by the observable order, and avoids any truncated many-body Fock basis.
"""
function dynamic_structure_factor(model::ManyBodyModel, result::QuadraticModeResult, field::MomentumField, qgrid, axis;
                                  temperature::Real=0.0,
                                  convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.normalization),
                                  broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                  weight_tol::Real=0.0, zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes())
    convention.fourier_normalization == field.normalization || throw(ArgumentError(
        "SpectrumConvention Fourier normalization must match MomentumField normalization"
    ))

    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    values = zeros(ComplexF64, length(qs), length(x))
    observable_order = :affine_linear
    projected_zero_modes = Set{Int}()

    for (iq, q) in enumerate(qs)
        source_q = _momentum_boson_quadratic_coefficients(model, field, q)
        source_minus = _momentum_boson_quadratic_coefficients(model, field, _negate_q(q))
        projected_q = _quadratic_mode_projection(model, result, source_q)
        projected_minus = _quadratic_mode_projection(model, result, source_minus)
        lines = _quadratic_mode_lehmann_lines(result, projected_q, projected_minus; temperature=temperature, convention=convention,
                                              degeneracy_tol=degeneracy_tol, weight_tol=weight_tol, zero_mode_policy=zero_mode_policy)
        values[iq, :] .= broaden_lines(lines, x, broadening)
        lines.metadata[:observable_order] === :quadratic && (observable_order = :quadratic)
        union!(projected_zero_modes, lines.metadata[:projected_zero_modes])
    end

    metadata = Dict{Symbol,Any}(
        :observable => :dynamic_structure_factor,
        :temperature => float(temperature),
        :fourier_normalization => field.normalization,
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :representation => model.representation.name,
        :solver => get(result.metadata, :solver, :unknown),
        :mode_native => true,
        :mode_space => get(result.metadata, :mode_space, :unknown),
        :observable_class => observable_order === :quadratic ? :gaussian_boson_quadratic : :linear_boson,
        :observable_order => observable_order,
        :zero_mode_policy => Symbol(nameof(typeof(zero_mode_policy))),
        :projected_zero_modes => sort!(collect(projected_zero_modes)),
    )
    return SpectrumResult(qs, x, values, metadata)
end

"""
    spin_tensor_structure_factor(model, result, field, qgrid, axis; ...)

Compute all nine Cartesian components `S^{alpha beta}(q,x)`, with component
ordering `(x,y,z)`.  For each q, the three operator matrices are realized and
transformed to the eigenbasis once, then reused for all nine tensor entries.
"""
function spin_tensor_structure_factor(
    model::ManyBodyModel,
    result::SolverResult,
    field::SpinTensorField,
    qgrid,
    axis;
    temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(
        fourier_normalization=field.x.normalization,
    ),
    broadening::AbstractBroadening=LorentzianBroadening(0.05),
    degeneracy_tol::Real=1e-10,
    weight_tol::Real=0.0,
)
    norms = (field.x.normalization, field.y.normalization, field.z.normalization)
    all(x -> x == norms[1], norms) || throw(ArgumentError("all spin tensor fields must use the same Fourier normalization"))
    convention.fourier_normalization == norms[1] ||
        throw(ArgumentError("SpectrumConvention Fourier normalization must match SpinTensorField"))

    E = _real_energies(result)
    p = thermal_probabilities(
        result;
        temperature=temperature,
        kB=convention.kB,
        degeneracy_tol=degeneracy_tol,
    )
    qs = collect(qgrid)
    x = _check_axis_grid(axis)
    out = zeros(ComplexF64, length(qs), length(x), 3, 3)
    axis_scale = convention.spectral_axis === :energy ? 1.0 : inv(float(convention.hbar))

    for (iq, q) in enumerate(qs)
        Oq = spin_tensor_operators(model, field, q)
        Om = spin_tensor_operators(model, field, _negate_q(q))
        Aq = ntuple(a -> eigenbasis_matrix(model, result, Oq[a]; hbar=convention.hbar), 3)
        Bm = ntuple(b -> eigenbasis_matrix(model, result, Om[b]; hbar=convention.hbar), 3)

        @inbounds for m in eachindex(E)
            pm = p[m]
            pm <= weight_tol && continue
            for n in eachindex(E)
                center = (E[n] - E[m]) * axis_scale
                K = _kernel(broadening, x .- center)
                for α in 1:3, β in 1:3
                    w = pm * Aq[α][m, n] * Bm[β][n, m]
                    abs(w) <= weight_tol && continue
                    @views out[iq, :, α, β] .+= w .* K
                end
            end
        end
    end

    metadata = Dict{Symbol,Any}(
        :observable => :spin_tensor_dynamic_structure_factor,
        :components => (:x, :y, :z),
        :temperature => float(temperature),
        :fourier_normalization => norms[1],
        :spectral_axis => convention.spectral_axis,
        :hbar => convention.hbar,
        :kB => convention.kB,
        :broadening => broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :resolution_applied => false,
        :representation => model.representation.name,
        :solver => get(result.metadata, :solver, :unknown),
        :tensor_definition => :S_alpha_beta,
    )
    return TensorSpectrumResult(qs, x, out, metadata)
end
