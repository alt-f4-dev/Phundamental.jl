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
    SpinTensorStructureFactorRequest(Q, axis; temperature=0, broadening=LorentzianBroadening(0.05), convention=SpectrumConvention(), sites=nothing, degeneracy_tol=1e-10, weight_tol=0, metadata=Dict())

Describe a full Cartesian spin-tensor structure-factor calculation for `ForwardProblem`. `Q` may be one scalar/vector momentum or a collection of momentum vectors. Local spin keys and positions are resolved from the model at evaluation time, so exact representation transformations automatically use the transformed registered observables.
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
function LocalElectronSpectralFunctionRequest(
    axis;
    spin::Symbol=:up,
    sites=nothing,
    eta::Real=0.05,
    normalization::Symbol=:spectral_density,
    green_krylov_dim::Integer=192,
    tol::Real=1e-10,
    green_reorthogonalize::Bool=false,
    parallel::Bool=true,
    max_basis_dimension::Integer=50_000_000,
    metadata=Dict{Symbol,Any}(),
)
    spin in (:up, :down) || throw(ArgumentError("spin must be :up or :down"))
    eta > 0 || throw(ArgumentError("eta must be positive"))
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) || throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
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

    return LocalElectronSpectralFunctionRequest(
        axis_values,
        spin,
        site_selection,
        Float64(eta),
        normalization,
        Int(green_krylov_dim),
        Float64(tol),
        green_reorthogonalize,
        parallel,
        Int(max_basis_dimension),
        Dict{Symbol,Any}(metadata),
    )
end

function (request::LocalElectronSpectralFunctionRequest)(model::ManyBodyModel, solution::SolverResult)
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("LocalElectronSpectralFunctionRequest requires a FermionAlgebra model"))

    workspace = fermion_green_workspace(
        model,
        solution;
        spin=request.spin,
        green_krylov_dim=request.green_krylov_dim,
        tol=request.tol,
        green_reorthogonalize=request.green_reorthogonalize,
        parallel=request.parallel,
        max_basis_dimension=request.max_basis_dimension,
    )

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
