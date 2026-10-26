# Typed requests that delegate to existing static, thermodynamic, and CPT numerical methods.

"""Equal-time momentum correlation evaluated using the existing static structure-factor routine."""
struct StaticStructureFactorRequest{F,Q,C,Z,M} <: AbstractObservable
    field::F
    qgrid::Q
    temperature::Float64
    convention::C
    zero_mode_policy::Z
    metadata::M
end

function StaticStructureFactorRequest(field::MomentumField, qgrid; temperature::Real=0.0,
                                      convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field.normalization),
                                      zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes(), metadata=Dict{Symbol,Any}())
    isfinite(temperature) && temperature >= 0 || throw(ArgumentError("static structure-factor temperature must be finite and nonnegative"))
    convention.fourier_normalization == field.normalization || throw(ArgumentError("momentum-field normalization and spectrum convention disagree"))
    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("static structure-factor momentum grid must not be empty"))
    return StaticStructureFactorRequest(field, qs, Float64(temperature), convention, zero_mode_policy, Dict{Symbol,Any}(metadata))
end

function (request::StaticStructureFactorRequest)(model::ManyBodyModel, result::SolverResult)
    return static_structure_factor(model, result, request.field, request.qgrid;
                                   temperature=request.temperature, convention=request.convention)
end

function (request::StaticStructureFactorRequest)(model::ManyBodyModel, result::QuadraticModeResult)
    return static_structure_factor(model, result, request.field, request.qgrid;
                                   temperature=request.temperature, convention=request.convention,
                                   zero_mode_policy=request.zero_mode_policy)
end

"""Temperature-grid request evaluated by `ThermodynamicForwardSolver` using an existing thermal method."""
struct ThermodynamicCurveRequest{M} <: AbstractObservable
    temperatures::Vector{Float64}
    kB::Float64
    metadata::M
end

function ThermodynamicCurveRequest(temperatures; kB::Real=1.0, metadata=Dict{Symbol,Any}())
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("thermodynamic temperature grid must not be empty"))
    all(T -> isfinite(T) && T >= 0, Ts) || throw(ArgumentError("thermodynamic temperatures must be finite and nonnegative"))
    isfinite(kB) && kB > 0 || throw(ArgumentError("kB must be finite and positive"))
    return ThermodynamicCurveRequest(Ts, Float64(kB), Dict{Symbol,Any}(metadata))
end

"""Cluster perturbation theory spectrum from a spin-resolved fermion-sector solution; the intercluster hopping is explicit."""
struct CPTSpectralFunctionRequest{K,X,M} <: AbstractObservable
    momenta::K
    axis::X
    eta::Float64
    t_intercluster::ComplexF64
    spin::Symbol
    normalization::Symbol
    parallel::Bool
    metadata::M
end

function CPTSpectralFunctionRequest(momenta, axis; eta::Real, t_intercluster::Number, spin::Symbol=:up,
                                    normalization::Symbol=:spectral_density, parallel::Bool=false, metadata=Dict{Symbol,Any}())
    ks = Float64.(collect(momenta))
    xs = Float64.(collect(axis))
    isempty(ks) && throw(ArgumentError("CPT momentum grid must not be empty"))
    length(xs) >= 2 && all(diff(xs) .> 0) || throw(ArgumentError("CPT energy grid must be strictly increasing"))
    all(isfinite, ks) && all(isfinite, xs) || throw(ArgumentError("CPT grids must be finite"))
    isfinite(eta) && eta > 0 || throw(ArgumentError("CPT eta must be finite and positive"))
    spin in (:up, :down) || throw(ArgumentError("CPT spin must be :up or :down"))
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) || throw(ArgumentError("unsupported CPT normalization"))
    isfinite(real(t_intercluster)) && isfinite(imag(t_intercluster)) || throw(ArgumentError("CPT hopping must be finite"))
    return CPTSpectralFunctionRequest(ks, xs, Float64(eta), ComplexF64(t_intercluster), spin, normalization, parallel,
                                      Dict{Symbol,Any}(metadata))
end

function (request::CPTSpectralFunctionRequest)(model::ManyBodyModel, solution::SolverResult)
    workspace = fermion_green_workspace(model, solution; spin=request.spin, parallel=request.parallel)
    result = cluster_perturbation_spectral_function(workspace, request.momenta, request.axis;
                                                    eta=request.eta, t_intercluster=request.t_intercluster,
                                                    normalization=request.normalization, parallel=request.parallel)
    result.metadata[:request_metadata] = copy(request.metadata)
    return result
end

"""A coherent harmonic neutron slice evaluated through the existing one-, two-, or multiphonon engine."""
struct HarmonicMultiphononSliceRequest{P,H,V,S,W,B,C,M} <: AbstractObservable
    probe::P
    fixed_axis::Symbol
    fixed_value::Float64
    horizontal_axis::Symbol
    horizontal_values::H
    vertical_axis::Symbol
    vertical_values::V
    qmesh_shape::S
    energy_window::W
    order::Union{Int,Symbol}
    temperature::Float64
    broadening::B
    convention::C
    parallel::Symbol
    metadata::M
end

function HarmonicMultiphononSliceRequest(probe; fixed_axis::Symbol, fixed_value::Real, horizontal_axis::Symbol,
                                         horizontal_values, vertical_axis::Symbol, vertical_values, qmesh_shape,
                                         energy_window, order::Union{Integer,Symbol}=2, temperature::Real=0.0,
                                         broadening=nothing, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                         parallel::Symbol=:auto, metadata=Dict{Symbol,Any}())
    probe isa NuclearNeutronProbe || throw(ArgumentError("harmonic multiphonon slices require NuclearNeutronProbe"))
    axes = (fixed_axis, horizontal_axis, vertical_axis)
    length(unique(axes)) == 3 && Set(axes) == Set((:x, :y, :z)) ||
        throw(ArgumentError("harmonic neutron slice axes must be distinct :x, :y, and :z"))
    isfinite(fixed_value) || throw(ArgumentError("fixed reciprocal coordinate must be finite"))
    h = Float64.(collect(horizontal_values))
    v = Float64.(collect(vertical_values))
    !isempty(h) && !isempty(v) && all(isfinite, h) && all(isfinite, v) ||
        throw(ArgumentError("slice grids must be finite and nonempty"))
    mesh = Tuple(Int.(qmesh_shape))
    length(mesh) == 3 && all(>(0), mesh) || throw(ArgumentError("harmonic neutron slice requires a positive three-dimensional qmesh_shape"))
    window = Tuple(Float64.(energy_window))
    length(window) == 2 && all(isfinite, window) && window[1] < window[2] ||
        throw(ArgumentError("energy_window must contain two finite, increasing bounds"))
    order === :all || (order isa Integer && order >= 1) || throw(ArgumentError("order must be :all or a positive integer"))
    isfinite(temperature) && temperature >= 0 || throw(ArgumentError("temperature must be finite and nonnegative"))
    (order in (1, 2) || !isnothing(broadening)) ||
        throw(ArgumentError("multiphonon order :all or >2 requires explicit spectral broadening"))
    isnothing(broadening) || broadening isa AbstractBroadening ||
        throw(ArgumentError("broadening must implement AbstractBroadening"))
    parallel in (:auto, :serial, :map, :qmesh) || throw(ArgumentError("unsupported multiphonon parallel backend"))
    return HarmonicMultiphononSliceRequest(probe, fixed_axis, Float64(fixed_value), horizontal_axis, h, vertical_axis, v,
                                            mesh, window, order, Float64(temperature), broadening, convention, parallel,
                                            Dict{Symbol,Any}(metadata))
end

function (request::HarmonicMultiphononSliceRequest)(model::ManyBodyModel, result::PhononDispersionResult)
    output = harmonic_neutron_slice(request.probe, result; fixed_axis=request.fixed_axis, fixed_value=request.fixed_value,
                                    horizontal_axis=request.horizontal_axis, horizontal_values=request.horizontal_values,
                                    vertical_axis=request.vertical_axis, vertical_values=request.vertical_values,
                                    qmesh_shape=request.qmesh_shape, energy_window=request.energy_window, order=request.order,
                                    temperature=request.temperature, broadening=request.broadening,
                                    convention=request.convention, parallel=request.parallel)
    output.metadata[:request_metadata] = copy(request.metadata)
    return output
end

# Exact finite-basis magnetic response, with an explicit field-conjugate moment operator.

"""Static zero-field isothermal susceptibility versus temperature, using the complete Lehmann/Kubo response.

`moment` is the Hermitian physical moment component coupled as `H(B) = H(0) - B * moment`.
Its direction and units are explicit; no implicit g factor, Bohr magneton, or physical kB conversion is applied.
Temperatures must be strictly positive to avoid an ambiguous degenerate-ground-state derivative.
"""
struct MagneticSusceptibilityRequest{O,M} <: AbstractObservable
    moment::O
    temperatures::Vector{Float64}
    direction::Symbol
    kB::Float64
    field_unit::String
    moment_unit::String
    metadata::M
end

"""Field-dependent magnetization `<moment>` on the `(field, temperature)` grid from the exact Zeeman-shifted Hamiltonian."""
struct MagnetizationCurveRequest{O,M} <: AbstractObservable
    moment::O
    fields::Vector{Float64}
    temperatures::Vector{Float64}
    direction::Symbol
    kB::Float64
    field_unit::String
    moment_unit::String
    metadata::M
end

"""Exact static magnetic response; `values` has shape `(length(fields), length(temperatures))`."""
struct MagneticResponseCurve
    kind::Symbol
    direction::Symbol
    fields::Vector{Float64}
    temperatures::Vector{Float64}
    values::Matrix{Float64}
    moment_unit::String
    field_unit::String
    kB::Float64
    metadata::Dict{Symbol,Any}
end

function _magnetic_request_validation(temperatures, direction, kB, field_unit, moment_unit; allow_zero)
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("magnetic response requires at least one temperature"))
    all(T -> isfinite(T) && (allow_zero ? T >= 0 : T > 0), Ts) ||
        throw(ArgumentError("magnetic response temperatures must be finite and $(allow_zero ? "nonnegative" : "positive")"))
    direction in (:x, :y, :z) || throw(ArgumentError("magnetic direction must be :x, :y, or :z"))
    isfinite(kB) && kB > 0 || throw(ArgumentError("kB must be finite and positive in energy/temperature units"))
    !isempty(field_unit) && !isempty(moment_unit) || throw(ArgumentError("field and moment units must be explicit"))
    return Ts
end

function MagneticSusceptibilityRequest(moment, temperatures; direction::Symbol, kB::Real=1.0,
                                       field_unit::AbstractString, moment_unit::AbstractString,
                                       metadata=Dict{Symbol,Any}())
    Ts = _magnetic_request_validation(temperatures, direction, kB, field_unit, moment_unit; allow_zero=false)
    return MagneticSusceptibilityRequest(moment, Ts, direction, Float64(kB), String(field_unit), String(moment_unit), Dict{Symbol,Any}(metadata))
end

function MagnetizationCurveRequest(moment, fields, temperatures; direction::Symbol, kB::Real=1.0,
                                   field_unit::AbstractString, moment_unit::AbstractString,
                                   metadata=Dict{Symbol,Any}())
    Ts = _magnetic_request_validation(temperatures, direction, kB, field_unit, moment_unit; allow_zero=true)
    Bs = Float64.(collect(fields))
    !isempty(Bs) && all(isfinite, Bs) || throw(ArgumentError("magnetic fields must be finite and nonempty"))
    return MagnetizationCurveRequest(moment, Bs, Ts, direction, Float64(kB), String(field_unit), String(moment_unit), Dict{Symbol,Any}(metadata))
end

# A total-spin moment is rebuilt from the current cluster's registered physical-site operators.
# This is essential for NLCE: source-model site indices do not identify the same sites in a subcluster.
function _registered_total_spin_moment(model::ManyBodyModel, moment::Symbol)
    axis = moment === :total_spin_x ? :x : moment === :total_spin_y ? :y : moment === :total_spin_z ? :z : nothing
    isnothing(axis) && return nothing
    positions = get(model.parameters, :positions, nothing)
    positions isa AbstractVector && !isempty(positions) || throw(ArgumentError("total-spin moments require registered physical site positions"))
    operators = AbstractOperatorExpr[]
    for site in eachindex(positions)
        name = Symbol("spin_", axis, "_", site)
        haskey(model.observables, name) || throw(ArgumentError("total-spin moment requires registered operator $name"))
        push!(operators, model.observables[name])
    end
    return reduce(+, operators)
end

function _magnetic_eigensystem(model::ManyBodyModel, result::SolverResult, moment)
    isbiorthogonal(result) && throw(ArgumentError("static magnetic response requires a Hermitian physical eigensystem"))
    length(result.energies) == length(result.basis) || throw(ArgumentError("magnetic response requires a complete eigensystem"))
    V = result.right_states
    size(V) == (length(result.basis), length(result.basis)) || throw(DimensionMismatch("eigenvectors must form a complete basis"))
    isapprox(V' * V, I; atol=1e-9, rtol=1e-9) || throw(ArgumentError("magnetic response requires an orthonormal eigenbasis"))
    E = result.energies
    maximum(abs, imag.(E)) <= 1e-10 * max(maximum(abs, E), 1.0) || throw(ArgumentError("magnetic energies must be real"))
    hbar = hasproperty(result.solver, :hbar) ? result.solver.hbar : get(model.parameters, :hbar, 1.0)
    op = if moment isa Symbol
        total = _registered_total_spin_moment(model, moment)
        if !isnothing(total)
            total
        else
            haskey(model.observables, moment) || throw(ArgumentError("magnetic moment symbol is not a registered model observable"))
            model.observables[moment]
        end
    elseif moment isa ObservableRef
        resolve_observable(model, moment)
    else
        moment
    end
    M = Matrix{ComplexF64}(matrix(op, result.basis; sparse=false, hbar=hbar))
    isapprox(M, M'; atol=1e-10, rtol=1e-10) || throw(ArgumentError("magnetic moment must be Hermitian"))
    return Float64.(real.(E)), V, Hermitian((M + M') / 2)
end

function _magnetic_weights(E::AbstractVector{<:Real}, T::Float64, kB::Float64)
    shifted = E .- minimum(E)
    if T == 0
        tol = 1e-10 * max(maximum(abs, E), 1.0)
        valid = shifted .<= tol
        return Float64.(valid) ./ count(valid)
    end
    weights = exp.(-shifted ./ (kB * T))
    return weights ./ sum(weights)
end

function (request::MagneticSusceptibilityRequest)(model::ManyBodyModel, solution::SolverResult)
    E, V, M = _magnetic_eigensystem(model, solution, request.moment)
    Me = V' * M * V
    values = Matrix{Float64}(undef, 1, length(request.temperatures))
    tol = 1e-10 * max(maximum(abs, E), 1.0)
    for (it, T) in enumerate(request.temperatures)
        p = _magnetic_weights(E, T, request.kB)
        beta = inv(request.kB * T)
        meanM = sum(p .* real.(diag(Me)))
        response = 0.0
        for n in eachindex(E), m in eachindex(E)
            delta = E[n] - E[m]
            coeff = abs(delta) <= tol ? beta * (p[m] + p[n]) / 2 : (p[m] - p[n]) / delta
            response += coeff * abs2(Me[m, n])
        end
        values[1, it] = response - beta * meanM^2
    end
    metadata = merge(copy(request.metadata), Dict{Symbol,Any}(:method => :exact_lehmann_kubo,
                     :zeeman_convention => :minus_field_times_moment, :requires_complete_eigensystem => true, :response_kind => :static_susceptibility,
                     :frequency_domain => :static, :operator_channels => (request.direction,), :response_order => 2))
    return MagneticResponseCurve(:susceptibility, request.direction, [0.0], copy(request.temperatures), values,
                                 request.moment_unit, request.field_unit, request.kB, metadata)
end

function (request::MagnetizationCurveRequest)(model::ManyBodyModel, solution::SolverResult)
    E, V, M = _magnetic_eigensystem(model, solution, request.moment)
    H = Hermitian(V * Diagonal(E) * V')
    values = Matrix{Float64}(undef, length(request.fields), length(request.temperatures))
    for (ib, B) in enumerate(request.fields)
        F = eigen(Hermitian(H - B * M))
        Me = F.vectors' * M * F.vectors
        moments = real.(diag(Me))
        for (it, T) in enumerate(request.temperatures)
            values[ib, it] = dot(_magnetic_weights(F.values, T, request.kB), moments)
        end
    end
    metadata = merge(copy(request.metadata), Dict{Symbol,Any}(:method => :exact_field_diagonalization,
                     :zeeman_convention => :minus_field_times_moment, :requires_complete_eigensystem => true, :response_kind => :magnetization,
                     :frequency_domain => :static, :operator_channels => (request.direction,), :response_order => 1))
    return MagneticResponseCurve(:magnetization, request.direction, copy(request.fields), copy(request.temperatures), values,
                                 request.moment_unit, request.field_unit, request.kB, metadata)
end
