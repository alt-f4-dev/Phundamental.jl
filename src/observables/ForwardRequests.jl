# Typed high-level observable requests for ForwardProblem.

"""
    DynamicalCorrelationRequest(field, qgrid, axis; ...)
    DynamicalCorrelationRequest(field_a, field_b, qgrid, axis; ...)

Describe an intrinsic ordered dynamical correlation `S_AB(q,x)` for `ForwardProblem`. The one-field constructor requests the same-field correlation built from `field` at `q` and `-q`. The two-field constructor requests the cross-correlation between independent momentum fields `A` and `B`; no adjoint or conjugation is inserted automatically.

The request remains representation-safe when its `MomentumField`s use registered observable keys because those observables are transformed with the `ManyBodyModel`. Numerical line broadening is intrinsic to the request and remains separate from probe contraction and instrument resolution.
"""
struct DynamicalCorrelationRequest{A,B,Q,X,T,C,R,Z,M} <: AbstractCorrelationObservable
    field_a::A
    field_b::B
    qgrid::Q
    axis::X
    temperature::T
    convention::C
    broadening::R
    degeneracy_tol::Float64
    weight_tol::Float64
    zero_mode_policy::Z
    metadata::M
end

function DynamicalCorrelationRequest(field::MomentumField, qgrid, axis; kwargs...)
    return DynamicalCorrelationRequest(field, field, qgrid, axis; kwargs...)
end

function DynamicalCorrelationRequest(field_a::MomentumField, field_b::MomentumField, qgrid, axis; temperature::Real=0.0,
                                     convention::SpectrumConvention=SpectrumConvention(fourier_normalization=field_a.normalization),
                                     broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                     weight_tol::Real=0.0, zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes(),
                                     metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    field_a.normalization == field_b.normalization || throw(ArgumentError("dynamical-correlation fields must use the same Fourier normalization"))
    convention.fourier_normalization == field_a.normalization || throw(ArgumentError("SpectrumConvention Fourier normalization must match both MomentumField normalizations"))

    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("qgrid must not be empty"))
    x = _check_axis_grid(axis)
    return DynamicalCorrelationRequest(field_a, field_b, qs, x, Float64(temperature), convention, broadening, Float64(degeneracy_tol),
                                        Float64(weight_tol), zero_mode_policy, Dict{Symbol,Any}(metadata))
end

function _finalize_dynamical_correlation_request(request::DynamicalCorrelationRequest, result::SpectrumResult)
    metadata = copy(result.metadata)
    metadata[:observable_request] = :dynamical_correlation
    metadata[:request_metadata] = copy(request.metadata)
    return SpectrumResult(result.q, result.axis, result.intensity, metadata)
end

function (request::DynamicalCorrelationRequest)(model::ManyBodyModel, solution)
    result = dynamical_correlation(model, solution, request.field_a, request.field_b, request.qgrid, request.axis;
                                   temperature=request.temperature, convention=request.convention, broadening=request.broadening,
                                   degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol)
    return _finalize_dynamical_correlation_request(request, result)
end

function (request::DynamicalCorrelationRequest)(model::ManyBodyModel, solution::QuadraticModeResult)
    result = dynamical_correlation(model, solution, request.field_a, request.field_b, request.qgrid, request.axis;
                                   temperature=request.temperature, convention=request.convention, broadening=request.broadening,
                                   degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol,
                                   zero_mode_policy=request.zero_mode_policy)
    return _finalize_dynamical_correlation_request(request, result)
end

"""
    CompositeDynamicalCorrelationRequest(fields, qgrid, axis; ...)

Describe the complete ordered dynamical-response matrix for a labeled `NamedTuple` of `MomentumField`s. The result contains `S_AB(q,x)` for every ordered pair of named channels and is returned as a `CompositeSpectrumResult`. This is the generic high-level request for arbitrary coupled/composite observables; it does not assume any particular physical sector or coupling mechanism.
"""
struct CompositeDynamicalCorrelationRequest{F,Q,X,T,C,R,Z,M} <: AbstractCorrelationObservable
    fields::F
    qgrid::Q
    axis::X
    temperature::T
    convention::C
    broadening::R
    degeneracy_tol::Float64
    weight_tol::Float64
    zero_mode_policy::Z
    metadata::M
end

function CompositeDynamicalCorrelationRequest(fields::NamedTuple, qgrid, axis; temperature::Real=0.0,
                                               convention::SpectrumConvention=SpectrumConvention(fourier_normalization=_composite_field_data(fields)[3]),
                                               broadening::AbstractBroadening=LorentzianBroadening(0.05), degeneracy_tol::Real=1e-10,
                                               weight_tol::Real=0.0, zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes(),
                                               metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    _, _, normalization = _composite_field_data(fields)
    convention.fourier_normalization == normalization ||
        throw(ArgumentError("SpectrumConvention Fourier normalization must match all composite response fields"))
    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("qgrid must not be empty"))
    x = _check_axis_grid(axis)
    return CompositeDynamicalCorrelationRequest(fields, qs, x, Float64(temperature), convention, broadening, Float64(degeneracy_tol),
                                                 Float64(weight_tol), zero_mode_policy, Dict{Symbol,Any}(metadata))
end

function _finalize_composite_dynamical_correlation_request(request::CompositeDynamicalCorrelationRequest, result::CompositeSpectrumResult)
    metadata = copy(result.metadata)
    metadata[:observable_request] = :composite_dynamical_correlation
    metadata[:request_metadata] = copy(request.metadata)
    return CompositeSpectrumResult(result.channels, result.q, result.axis, result.intensity, metadata)
end

function (request::CompositeDynamicalCorrelationRequest)(model::ManyBodyModel, solution)
    result = composite_dynamical_correlation(model, solution, request.fields, request.qgrid, request.axis; temperature=request.temperature,
                                             convention=request.convention, broadening=request.broadening,
                                             degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol)
    return _finalize_composite_dynamical_correlation_request(request, result)
end

function (request::CompositeDynamicalCorrelationRequest)(model::ManyBodyModel, solution::QuadraticModeResult)
    result = composite_dynamical_correlation(model, solution, request.fields, request.qgrid, request.axis; temperature=request.temperature,
                                             convention=request.convention, broadening=request.broadening,
                                             degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol,
                                             zero_mode_policy=request.zero_mode_policy)
    return _finalize_composite_dynamical_correlation_request(request, result)
end

"""
    SpinPhononCorrelationRequest(qgrid, axis; spin_components=(:x, :y, :z), sites=nothing, spin_prefix=:spin, displacement_prefix=:auto, ...)

Describe a high-level coupled spin-phonon response. At evaluation time the request resolves registered local spin observables and local phonon-displacement observables from the current `ManyBodyModel`, constructs representation-safe momentum fields, and returns the complete ordered response matrix spanning the selected spin components and the displacement channel.

`displacement_prefix=:auto` accepts exactly one complete registered family named either `boson_displacement_i` or `phonon_displacement_i` on the selected sites. Supply an explicit prefix when both families are present or when a custom registry convention is used.
"""
struct SpinPhononCorrelationRequest{Q,X,T,C,R,Z,S,CMP,M} <: AbstractCorrelationObservable
    qgrid::Q
    axis::X
    temperature::T
    convention::C
    broadening::R
    degeneracy_tol::Float64
    weight_tol::Float64
    zero_mode_policy::Z
    sites::S
    spin_components::CMP
    spin_prefix::Symbol
    displacement_prefix::Symbol
    metadata::M
end

function SpinPhononCorrelationRequest(qgrid, axis; temperature::Real=0.0, spin_components=(:x, :y, :z), sites=nothing,
                                      spin_prefix::Symbol=:spin, displacement_prefix::Symbol=:auto,
                                      convention::SpectrumConvention=SpectrumConvention(), broadening::AbstractBroadening=LorentzianBroadening(0.05),
                                      degeneracy_tol::Real=1e-10, weight_tol::Real=0.0,
                                      zero_mode_policy::AbstractBosonZeroModePolicy=RejectBosonZeroModes(), metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    components = _normalize_spin_components(spin_components)
    selected_sites = if isnothing(sites)
        nothing
    else
        values = sites isa Integer ? [Int(sites)] : Int.(collect(sites))
        isempty(values) && throw(ArgumentError("sites cannot be empty"))
        all(>(0), values) || throw(ArgumentError("sites must contain positive indices"))
        values
    end
    qs = collect(qgrid)
    isempty(qs) && throw(ArgumentError("qgrid must not be empty"))
    x = _check_axis_grid(axis)
    return SpinPhononCorrelationRequest(qs, x, Float64(temperature), convention, broadening, Float64(degeneracy_tol),
                                        Float64(weight_tol), zero_mode_policy, selected_sites, components, spin_prefix,
                                        displacement_prefix, Dict{Symbol,Any}(metadata))
end

function _spin_phonon_sites(request::SpinPhononCorrelationRequest, model::ManyBodyModel)
    positions = get(model.parameters, :positions, nothing)
    isnothing(positions) && throw(ArgumentError("SpinPhononCorrelationRequest requires model.parameters[:positions]"))
    sites = isnothing(request.sites) ? collect(eachindex(positions)) : request.sites
    all(site -> 1 <= site <= length(positions), sites) || throw(BoundsError(positions, sites))
    return sites, positions
end

function _spin_phonon_prefix_complete(model::ManyBodyModel, prefix::Symbol, sites)
    return all(site -> haskey(model.observables, Symbol(string(prefix), "_", string(site))), sites)
end

function _spin_phonon_displacement_prefix(request::SpinPhononCorrelationRequest, model::ManyBodyModel, sites)
    request.displacement_prefix !== :auto && return request.displacement_prefix
    candidates = Symbol[prefix for prefix in (:boson_displacement, :phonon_displacement) if _spin_phonon_prefix_complete(model, prefix, sites)]
    isempty(candidates) && throw(ArgumentError("SpinPhononCorrelationRequest could not find a complete boson_displacement_i or phonon_displacement_i observable family"))
    length(candidates) == 1 || throw(ArgumentError("multiple displacement observable families are registered; supply displacement_prefix explicitly"))
    return only(candidates)
end

function _spin_phonon_request_fields(request::SpinPhononCorrelationRequest, model::ManyBodyModel)
    sites, positions = _spin_phonon_sites(request, model)
    selected_positions = positions[sites]
    normalization = request.convention.fourier_normalization
    xkeys = Symbol[Symbol(string(request.spin_prefix), "_x_", string(site)) for site in sites]
    ykeys = Symbol[Symbol(string(request.spin_prefix), "_y_", string(site)) for site in sites]
    zkeys = Symbol[Symbol(string(request.spin_prefix), "_z_", string(site)) for site in sites]
    spin_field = SpinTensorField(xkeys=xkeys, ykeys=ykeys, zkeys=zkeys, positions=selected_positions, normalization=normalization)
    displacement_prefix = _spin_phonon_displacement_prefix(request, model, sites)
    displacement_keys = Symbol[Symbol(string(displacement_prefix), "_", string(site)) for site in sites]
    phonon_field = registered_momentum_field(model, displacement_keys; sites=sites, normalization=normalization)

    for component in request.spin_components
        field = _spin_component_field(spin_field, component)
        foreach(key -> resolve_observable(model, key), field.keys)
    end
    return spin_field, phonon_field, displacement_prefix, sites
end

function _finalize_spin_phonon_request(request::SpinPhononCorrelationRequest, result::CompositeSpectrumResult, displacement_prefix::Symbol, sites)
    metadata = copy(result.metadata)
    metadata[:observable_request] = :spin_phonon_correlation
    metadata[:response_family] = :spin_phonon
    metadata[:spin_components] = request.spin_components
    metadata[:spin_prefix] = request.spin_prefix
    metadata[:displacement_prefix] = displacement_prefix
    metadata[:sites] = copy(sites)
    metadata[:request_metadata] = copy(request.metadata)
    return CompositeSpectrumResult(result.channels, result.q, result.axis, result.intensity, metadata)
end

function (request::SpinPhononCorrelationRequest)(model::ManyBodyModel, solution)
    spin_field, phonon_field, displacement_prefix, sites = _spin_phonon_request_fields(request, model)
    result = spin_phonon_dynamical_correlation(model, solution, spin_field, phonon_field, request.qgrid, request.axis;
                                               spin_components=request.spin_components, temperature=request.temperature,
                                               convention=request.convention, broadening=request.broadening,
                                               degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol)
    return _finalize_spin_phonon_request(request, result, displacement_prefix, sites)
end

function (request::SpinPhononCorrelationRequest)(model::ManyBodyModel, solution::QuadraticModeResult)
    spin_field, phonon_field, displacement_prefix, sites = _spin_phonon_request_fields(request, model)
    result = spin_phonon_dynamical_correlation(model, solution, spin_field, phonon_field, request.qgrid, request.axis;
                                               spin_components=request.spin_components, temperature=request.temperature,
                                               convention=request.convention, broadening=request.broadening,
                                               degeneracy_tol=request.degeneracy_tol, weight_tol=request.weight_tol,
                                               zero_mode_policy=request.zero_mode_policy)
    return _finalize_spin_phonon_request(request, result, displacement_prefix, sites)
end

struct SpinTensorStructureFactorRequest{Q,X,T,C,B,S,M} <: AbstractObservable
    qgrid::Q
    axis::X
    temperature::T
    convention::C
    broadening::B
    degeneracy_tol::Float64
    weight_tol::Float64
    sites::S
    metadata::M
end

_spin_tensor_qgrid(Q::Number) = [[Q]]
_spin_tensor_qgrid(Q::AbstractVector{<:Number}) = [collect(Q)]
_spin_tensor_qgrid(Q::AbstractVector{<:AbstractVector}) = [collect(q) for q in Q]
_spin_tensor_qgrid(Q::Tuple) = collect(Q)

"""
    SpinTensorStructureFactorRequest(Q, axis; temperature=0, broadening=LorentzianBroadening(0.05), convention=SpectrumConvention(), sites=nothing, degeneracy_tol=1e-10, weight_tol=0, metadata=Dict())

Describe a full Cartesian spin-tensor structure-factor calculation for `ForwardProblem`. `Q` may be one scalar/vector momentum or a collection of momentum vectors. Local spin keys and positions are resolved from the model at evaluation time, so exact representation transformations automatically use the transformed registered observables.
"""
function SpinTensorStructureFactorRequest(Q, axis; temperature::Real=0.0, broadening::AbstractBroadening=LorentzianBroadening(0.05),
                                          convention::SpectrumConvention=SpectrumConvention(), sites=nothing, degeneracy_tol::Real=1e-10,
                                          weight_tol::Real=0.0, metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    weight_tol >= 0 || throw(ArgumentError("weight_tol must be nonnegative"))
    site_selection = if isnothing(sites)
        nothing
    else
        values = Int.(collect(sites))
        isempty(values) && throw(ArgumentError("sites cannot be empty"))
        all(>(0), values) || throw(ArgumentError("sites must contain positive indices"))
        values
    end
    return SpinTensorStructureFactorRequest(_spin_tensor_qgrid(Q), collect(axis), Float64(temperature), convention, broadening,
                                            Float64(degeneracy_tol), Float64(weight_tol), site_selection, Dict{Symbol,Any}(metadata))
end

function _spin_tensor_request_field(request::SpinTensorStructureFactorRequest, model::ManyBodyModel)
    positions = get(model.parameters, :positions, nothing)
    isnothing(positions) && throw(ArgumentError("SpinTensorStructureFactorRequest requires model.parameters[:positions]"))
    N = length(positions)
    sites = isnothing(request.sites) ? collect(1:N) : request.sites
    all(site -> 1 <= site <= N, sites) || throw(BoundsError(positions, sites))

    xkeys = Symbol[Symbol("spin_x_", site) for site in sites]
    ykeys = Symbol[Symbol("spin_y_", site) for site in sites]
    zkeys = Symbol[Symbol("spin_z_", site) for site in sites]
    for key in Iterators.flatten((xkeys, ykeys, zkeys))
        haskey(model.observables, key) || throw(KeyError("SpinTensorStructureFactorRequest requires registered observable $key"))
    end

    return SpinTensorField(xkeys=xkeys, ykeys=ykeys, zkeys=zkeys, positions=positions[sites], normalization=request.convention.fourier_normalization)
end

function (request::SpinTensorStructureFactorRequest)(model::ManyBodyModel, solution)
    field = _spin_tensor_request_field(request, model)
    result = spin_tensor_structure_factor(model, solution, field, request.qgrid, request.axis; temperature=request.temperature,
                                          convention=request.convention, broadening=request.broadening, degeneracy_tol=request.degeneracy_tol,
                                          weight_tol=request.weight_tol)
    metadata = copy(result.metadata)
    metadata[:observable_request] = :spin_tensor_structure_factor
    metadata[:request_metadata] = copy(request.metadata)
    return TensorSpectrumResult(result.q, result.axis, result.intensity, metadata)
end

struct LocalElectronSpectralFunctionRequest{X,S,M} <: AbstractObservable
    axis::X
    spin::Symbol
    sites::S
    eta::Float64
    normalization::Symbol
    green_krylov_dim::Int
    tol::Float64
    green_reorthogonalize::Bool
    parallel::Bool
    max_basis_dimension::Int
    metadata::M
end

"""
    LocalElectronSpectralFunctionRequest(axis; spin=:up, sites=nothing, eta=0.05, normalization=:spectral_density, green_krylov_dim=192, tol=1e-10, green_reorthogonalize=false, parallel=true, max_basis_dimension=50_000_000, metadata=Dict())

Describe a zero-temperature site-resolved one-particle spectral-function calculation for `ForwardProblem`. The canonical ground sector is supplied by the solver attached to the forward problem and is reused rather than solved a second time; the request constructs only the spin-resolved particle-addition and particle-removal Green-function sectors.
"""
function LocalElectronSpectralFunctionRequest(axis; spin::Symbol=:up, sites=nothing, eta::Real=0.05, normalization::Symbol=:spectral_density,
                                              green_krylov_dim::Integer=192, tol::Real=1e-10, green_reorthogonalize::Bool=false,
                                              parallel::Bool=true, max_basis_dimension::Integer=50_000_000, metadata=Dict{Symbol,Any}())
    spin in (:up, :down) || throw(ArgumentError("spin must be :up or :down"))
    eta > 0 || throw(ArgumentError("eta must be positive"))
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) ||
        throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
    green_krylov_dim > 1 || throw(ArgumentError("green_krylov_dim must exceed one"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))

    axis_values = Float64.(collect(axis))
    isempty(axis_values) && throw(ArgumentError("axis must not be empty"))
    site_selection = if isnothing(sites)
        nothing
    else
        values = Int.(collect(sites))
        isempty(values) && throw(ArgumentError("sites cannot be empty"))
        all(>(0), values) || throw(ArgumentError("sites must contain positive indices"))
        length(unique(values)) == length(values) || throw(ArgumentError("sites must be unique"))
        values
    end

    return LocalElectronSpectralFunctionRequest(axis_values, spin, site_selection, Float64(eta), normalization, Int(green_krylov_dim), Float64(tol),
                                                green_reorthogonalize, parallel, Int(max_basis_dimension), Dict{Symbol,Any}(metadata))
end

function (request::LocalElectronSpectralFunctionRequest)(model::ManyBodyModel, solution::SolverResult)
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("LocalElectronSpectralFunctionRequest requires a FermionAlgebra model"))

    workspace = fermion_green_workspace(model, solution; spin=request.spin, green_krylov_dim=request.green_krylov_dim, tol=request.tol,
                                        green_reorthogonalize=request.green_reorthogonalize, parallel=request.parallel,
                                        max_basis_dimension=request.max_basis_dimension)

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


"""
    LocalMatsubaraGreenFunctionRequest(; metadata=Dict())

Request the local fermionic Matsubara Green function produced by an embedding calculation. The request never performs analytic continuation; a real-frequency spectral function remains a separate explicit operation.
"""
struct LocalMatsubaraGreenFunctionRequest{M} <: AbstractObservable
    metadata::M
end

LocalMatsubaraGreenFunctionRequest(; metadata=Dict{Symbol,Any}()) = LocalMatsubaraGreenFunctionRequest(Dict{Symbol,Any}(metadata))

function (request::LocalMatsubaraGreenFunctionRequest)(::ManyBodyModel, solution)
    hasproperty(solution, :local_green_function) || throw(ArgumentError("LocalMatsubaraGreenFunctionRequest requires a solution with a local_green_function field"))
    return getproperty(solution, :local_green_function)
end

"""
    LocalMatsubaraSelfEnergyRequest(; metadata=Dict())

Request the local fermionic Matsubara self-energy produced by an embedding calculation. The request returns the solver-native Matsubara object and does not infer a real-frequency self-energy.
"""
struct LocalMatsubaraSelfEnergyRequest{M} <: AbstractObservable
    metadata::M
end

LocalMatsubaraSelfEnergyRequest(; metadata=Dict{Symbol,Any}()) = LocalMatsubaraSelfEnergyRequest(Dict{Symbol,Any}(metadata))

function (request::LocalMatsubaraSelfEnergyRequest)(::ManyBodyModel, solution)
    hasproperty(solution, :self_energy) || throw(ArgumentError("LocalMatsubaraSelfEnergyRequest requires a solution with a self_energy field"))
    value = getproperty(solution, :self_energy)
    isnothing(value) && throw(ArgumentError("the selected embedding calculation did not reconstruct a self-energy"))
    return value
end

"""
    MatsubaraSusceptibilityRequest(channel=:magnetic; metadata=Dict())

Request one bosonic Matsubara susceptibility channel from a diagrammatic embedding calculation. Supported high-level ladder-DΓA channels are `:charge` and `:magnetic`. The request is intentionally Matsubara-domain only and never performs analytic continuation.
"""
struct MatsubaraSusceptibilityRequest{M} <: AbstractObservable
    channel::Symbol
    metadata::M
end

function MatsubaraSusceptibilityRequest(channel::Symbol=:magnetic; metadata=Dict{Symbol,Any}())
    channel in (:charge, :magnetic) || throw(ArgumentError("channel must be :charge or :magnetic"))
    return MatsubaraSusceptibilityRequest(channel, Dict{Symbol,Any}(metadata))
end

function (request::MatsubaraSusceptibilityRequest)(::ManyBodyModel, solution)
    if hasproperty(solution, :observables)
        observables = getproperty(solution, :observables)
        observables isa AbstractDict || throw(ArgumentError("solution.observables must be a dictionary-like channel registry"))
        haskey(observables, request.channel) || throw(ArgumentError("requested susceptibility channel $(request.channel) is unavailable"))
        return observables[request.channel]
    elseif hasproperty(solution, :observable)
        observable = getproperty(solution, :observable)
        hasproperty(observable, :metadata) || throw(ArgumentError("diagrammatic observable does not expose channel metadata"))
        channel = get(getproperty(observable, :metadata), :channel, nothing)
        channel === request.channel || throw(ArgumentError("diagrammatic result contains channel $(channel), not $(request.channel)"))
        return observable
    end
    throw(ArgumentError("MatsubaraSusceptibilityRequest requires a diagrammatic solution exposing susceptibility channels"))
end


"""Intrinsic basis-resolved harmonic displacement structure factor with values indexed `(iQ, ix, a, b)`, where `a` and `b` label primitive-cell displacement degrees of freedom. Coherent nuclear scattering lengths, the neutron `Q` projection, and instrument resolution are not applied."""
struct PhononStructureFactorResult{M,Q,X,V,D}
    model::M
    q::Q
    axis::X
    intensity::V
    metadata::D
end

"""
    PhononStructureFactorRequest(Q, axis; temperature=0, broadening=GaussianBroadening(0.05), convention=SpectrumConvention(spectral_axis=:omega), displacement_prefactor=nothing, zero_mode_tol=nothing, reduction_convention=:first_bz, reduction_atol=1e-12, metadata=Dict())

Describe an intrinsic harmonic phonon structure-factor calculation for `ForwardProblem`. `Q` is the full physical momentum transfer: a scalar or scalar sequence is reciprocal-lattice shorthand only for one-dimensional models, while higher-dimensional calculations use explicit `ReciprocalWaveVector` or `CartesianWaveVector` objects. The request produces a basis-resolved displacement correlation tensor before probe-specific nuclear scattering lengths, neutron polarization factors, or instrument resolution are applied.
"""
struct PhononStructureFactorRequest{Q,X,C,B,M} <: AbstractObservable
    Q::Q
    axis::X
    temperature::Float64
    convention::C
    broadening::B
    displacement_prefactor::Union{Nothing,Float64}
    zero_mode_tol::Union{Nothing,Float64}
    reduction_convention::Symbol
    reduction_atol::Float64
    metadata::M
end

function PhononStructureFactorRequest(Q, axis; temperature::Real=0.0, broadening::AbstractBroadening=GaussianBroadening(0.05),
                                      convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                      displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing,
                                      reduction_convention::Symbol=:first_bz, reduction_atol::Real=1e-12, metadata=Dict{Symbol,Any}())
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    isnothing(displacement_prefactor) || displacement_prefactor > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    isnothing(zero_mode_tol) || zero_mode_tol >= 0 || throw(ArgumentError("zero_mode_tol must be nonnegative"))
    reduction_atol >= 0 || throw(ArgumentError("reduction_atol must be nonnegative"))
    x = _check_axis_grid(axis)
    displacement_scale = isnothing(displacement_prefactor) ? nothing : Float64(displacement_prefactor)
    zero_tolerance = isnothing(zero_mode_tol) ? nothing : Float64(zero_mode_tol)
    return PhononStructureFactorRequest(Q, x, Float64(temperature), convention, broadening, displacement_scale, zero_tolerance,
                                        reduction_convention, Float64(reduction_atol), Dict{Symbol,Any}(metadata))
end

function _phonon_structure_Qpoints(request::PhononStructureFactorRequest, model::ManyBodyModel)
    D = model.parameters[:spatial_dimension]
    if request.Q isa Real
        D == 1 || throw(ArgumentError("scalar full-Q coordinates are only unambiguous for one-dimensional models; use an explicit wavevector for D=$D"))
        return AbstractWaveVector[ReciprocalWaveVector([Float64(request.Q)])]
    elseif request.Q isa AbstractWaveVector
        return AbstractWaveVector[request.Q]
    end

    points = collect(request.Q)
    isempty(points) && throw(ArgumentError("phonon Q path cannot be empty"))
    if all(point -> point isa Real, points)
        D == 1 || throw(ArgumentError("a scalar full-Q path is only unambiguous for one-dimensional models; use explicit wavevectors for D=$D"))
        return AbstractWaveVector[ReciprocalWaveVector([Float64(point)]) for point in points]
    end
    all(point -> point isa AbstractWaveVector, points) ||
        throw(ArgumentError("phonon Q paths must contain explicit wavevector objects; raw multidimensional vectors are ambiguous between Cartesian and reciprocal coordinates"))
    return AbstractWaveVector[point for point in points]
end

function _phonon_structure_decompositions(request::PhononStructureFactorRequest, model::ManyBodyModel)
    Qpoints = _phonon_structure_Qpoints(request, model)
    return [decompose_reciprocal_vector(model, Q; convention=request.reduction_convention, atol=request.reduction_atol) for Q in Qpoints]
end

function _phonon_structure_solver_wavevectors(request::PhononStructureFactorRequest, model::ManyBodyModel)
    decompositions = _phonon_structure_decompositions(request, model)
    return [ReciprocalWaveVector(-wavevector_coordinates(decomposition.q_reciprocal)) for decomposition in decompositions]
end

function _phonon_structure_zero_mask(request::PhononStructureFactorRequest, solution::PhononDispersionResult, iq::Int)
    omega = view(solution.frequencies, iq, :)
    if isnothing(request.zero_mode_tol)
        mask = get(solution.metadata, :zero_mode_mask, nothing)
        isnothing(mask) && return falses(length(omega))
        size(mask) == size(solution.frequencies) || throw(DimensionMismatch("phonon dispersion zero-mode mask does not match the frequency array"))
        return Bool.(view(mask, iq, :))
    end
    threshold = request.zero_mode_tol * max(maximum(abs, omega; init=0.0), 1.0)
    return omega .<= threshold
end

function _phonon_structure_basis_positions(model::ManyBodyModel)
    direct = Matrix{Float64}(model.parameters[:phonon_direct_matrix])
    basis = model.parameters[:phonon_basis_fractional]
    return [direct * Float64.(fractional) for fractional in basis]
end

function _evaluate_phonon_structure_factor(request::PhononStructureFactorRequest, model::ManyBodyModel, solution::PhononDispersionResult)
    decompositions = _phonon_structure_decompositions(request, model)
    nq = length(decompositions)
    size(solution.frequencies, 1) == nq || throw(DimensionMismatch("phonon dispersion q-point count does not match the requested full-Q path"))
    size(solution.modes, 3) == nq || throw(DimensionMismatch("phonon dispersion mode count does not match the requested full-Q path"))

    x = request.axis
    nx = length(x)
    ndof = size(solution.modes, 1)
    size(solution.modes, 2) == ndof || throw(DimensionMismatch("phonon mode matrices must be square"))
    D = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    nbasis * D == ndof || throw(DimensionMismatch("phonon basis metadata does not match the normal-mode dimension"))
    positions = _phonon_structure_basis_positions(model)
    length(positions) == nbasis || throw(DimensionMismatch("phonon basis-position metadata is inconsistent"))

    displacement_scale = isnothing(request.displacement_prefactor) ? Float64(request.convention.hbar) : request.displacement_prefactor
    tensor = zeros(ComplexF64, nq, nx, ndof, ndof)
    skipped_zero_modes = Vector{Vector{Int}}(undef, nq)
    Qcartesian = Vector{Vector{Float64}}(undef, nq)
    reduced_q = Vector{Vector{Float64}}(undef, nq)
    reciprocal_G = Vector{Vector{Float64}}(undef, nq)
    eigenvector_qprime = Vector{Vector{Float64}}(undef, nq)

    for iq in 1:nq
        decomposition = decompositions[iq]
        Qcartesian[iq] = Float64.(wavevector_coordinates(decomposition.Q_cartesian))
        reduced_q[iq] = Float64.(wavevector_coordinates(decomposition.q_reciprocal))
        reciprocal_G[iq] = Float64.(wavevector_coordinates(decomposition.G_reciprocal))
        eigenvector_qprime[iq] = -reduced_q[iq]
        Gcart = Float64.(wavevector_coordinates(decomposition.G_cartesian))
        physical_modes = Solvers.physical_displacement_modes(solution, iq)
        zero_mask = _phonon_structure_zero_mask(request, solution, iq)
        skipped_zero_modes[iq] = findall(identity, zero_mask)

        for branch in axes(solution.frequencies, 2)
            zero_mask[branch] && continue
            omega = Float64(solution.frequencies[iq, branch])
            omega > 0 || throw(ArgumentError("nonzero phonon modes must have positive frequency"))
            mode = ComplexF64.(view(physical_modes, :, branch))
            for site in 1:nbasis
                phase = cis(dot(Gcart, positions[site]))
                rows = ((site - 1) * D + 1):(site * D)
                mode[rows] .*= phase
            end
            mode .*= sqrt(displacement_scale / (2omega))
            line_tensor = mode * adjoint(mode)
            occupation = if iszero(request.temperature)
                0.0
            else
                inv(expm1(Float64(request.convention.hbar) * omega / (Float64(request.convention.kB) * request.temperature)))
            end
            center = request.convention.spectral_axis === :energy ? Float64(request.convention.hbar) * omega : omega

            @inbounds for ix in eachindex(x)
                tensor[iq, ix, :, :] .+= ((occupation + 1) * _kernel_value(request.broadening, x[ix] - center)) .* line_tensor
                occupation > 0 && (tensor[iq, ix, :, :] .+= (occupation * _kernel_value(request.broadening, x[ix] + center)) .* line_tensor)
            end
        end
    end

    metadata = Dict{Symbol,Any}(
        :observable => :phonon_structure_factor,
        :observable_request => :phonon_structure_factor,
        :response_kind => :correlation_spectrum,
        :frequency_domain => :real,
        :statistics => :bosonic,
        :momentum_scope => :momentum_resolved,
        :response_order => 2,
        :temperature => request.temperature,
        :spectral_axis => request.convention.spectral_axis,
        :hbar => Float64(request.convention.hbar),
        :kB => Float64(request.convention.kB),
        :broadening => request.broadening,
        :broadening_role => :intrinsic_numerical_line_shape,
        :displacement_prefactor => displacement_scale,
        :zero_mode_tolerance => request.zero_mode_tol,
        :skipped_zero_modes => skipped_zero_modes,
        :full_Q_cartesian => Qcartesian,
        :reduced_q_reciprocal => reduced_q,
        :reciprocal_G_reciprocal => reciprocal_G,
        :eigenvector_qprime_reciprocal => eigenvector_qprime,
        :basis_phase_convention => :atomic_position_dynamical_matrix,
        :coherent_scattering_lengths_applied => false,
        :neutron_Q_projection_applied => false,
        :resolution_applied => false,
        :request_metadata => copy(request.metadata),
    )
    return PhononStructureFactorResult(model, Qcartesian, copy(x), tensor, metadata)
end

function (request::PhononStructureFactorRequest)(model::ManyBodyModel, solution)
    solution isa PhononDispersionResult ||
        throw(ArgumentError("PhononStructureFactorRequest requires a PhononDispersionResult; use HarmonicPhononSolver in ForwardProblem"))
    return _evaluate_phonon_structure_factor(request, model, solution)
end

export PhononStructureFactorRequest, PhononStructureFactorResult
