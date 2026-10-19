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
