# Typed high-level observable requests for ForwardProblem.

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
    SpinTensorStructureFactorRequest(Q, axis; temperature=0,
                                     broadening=LorentzianBroadening(0.05),
                                     convention=SpectrumConvention(), sites=nothing,
                                     degeneracy_tol=1e-10, weight_tol=0, metadata=Dict())

Describe a full Cartesian spin-tensor structure-factor calculation for
`ForwardProblem`. `Q` may be one scalar/vector momentum or a collection of
momentum vectors. Local spin keys and positions are resolved from the model at
evaluation time, so exact representation transformations automatically use the
transformed registered observables.
"""
function SpinTensorStructureFactorRequest(
    Q,
    axis;
    temperature::Real=0.0,
    broadening::AbstractBroadening=LorentzianBroadening(0.05),
    convention::SpectrumConvention=SpectrumConvention(),
    sites=nothing,
    degeneracy_tol::Real=1e-10,
    weight_tol::Real=0.0,
    metadata=Dict{Symbol,Any}(),
)
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
    return SpinTensorStructureFactorRequest(
        _spin_tensor_qgrid(Q), collect(axis), Float64(temperature), convention, broadening,
        Float64(degeneracy_tol), Float64(weight_tol), site_selection, Dict{Symbol,Any}(metadata),
    )
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

    return SpinTensorField(
        xkeys=xkeys,
        ykeys=ykeys,
        zkeys=zkeys,
        positions=positions[sites],
        normalization=request.convention.fourier_normalization,
    )
end

function (request::SpinTensorStructureFactorRequest)(model::ManyBodyModel, solution)
    field = _spin_tensor_request_field(request, model)
    result = spin_tensor_structure_factor(
        model,
        solution,
        field,
        request.qgrid,
        request.axis;
        temperature=request.temperature,
        convention=request.convention,
        broadening=request.broadening,
        degeneracy_tol=request.degeneracy_tol,
        weight_tol=request.weight_tol,
    )
    metadata = copy(result.metadata)
    metadata[:observable_request] = :spin_tensor_structure_factor
    metadata[:request_metadata] = copy(request.metadata)
    return TensorSpectrumResult(result.q, result.axis, result.intensity, metadata)
end
