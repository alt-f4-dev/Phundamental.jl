# Coherent harmonic multiphonon scattering for periodic phonon dispersions.

"""
    phonon_reciprocal_mesh(shape)

Construct a full uniform reciprocal-lattice mesh using explicit `ReciprocalWaveVector` points with coordinates `k_d / N_d`, where `k_d = 0, ..., N_d - 1`.
"""
function phonon_reciprocal_mesh(shape::NTuple{D,<:Integer}) where {D}
    dims = ntuple(d -> Int(shape[d]), D)
    all(>(0), dims) || throw(ArgumentError("all reciprocal-mesh dimensions must be positive"))
    axes = ntuple(d -> 0:(dims[d] - 1), D)
    return [ReciprocalWaveVector(Float64[index[d] / dims[d] for d in 1:D]) for index in Iterators.product(axes...)]
end

phonon_reciprocal_mesh(shape::AbstractVector{<:Integer}) = phonon_reciprocal_mesh(Tuple(Int.(shape)))


"""
    HarmonicMultiphononWorkspace

Reusable reciprocal-mesh and thermal data for coherent harmonic multiphonon calculations. Construct with `harmonic_multiphonon_workspace` and reuse the workspace across many scattering vectors on the same dispersion, temperature, spectrum convention, and zero-mode policy.
"""
struct HarmonicMultiphononWorkspace{D,R,M,I,P,O,Z}
    result::R
    qmesh_shape::NTuple{D,Int}
    mesh_map::M
    mesh_indices::I
    physical_modes::P
    occupations::O
    zero_mask::Z
    temperature::Float64
    hbar::Float64
    kB::Float64
    displacement_scale::Float64
    zero_mode_request::Union{Nothing,Float64}
    zero_mode_source::Symbol
    zero_mode_tolerance::Union{Nothing,Float64}
    zero_mode_frequency_threshold::Union{Nothing,Float64}
    mesh_tolerance::Float64
end

function _multiphonon_physical_modes(result::PhononDispersionResult)
    model = result.model
    D = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    masses = Float64.(model.parameters[:masses])
    length(masses) == nbasis * D ||
        throw(DimensionMismatch("phonon mass vector does not match primitive-cell displacement degrees of freedom"))
    nq, nbranch = size(result.frequencies)
    size(result.modes) == (nbranch, nbranch, nq) || throw(DimensionMismatch("phonon dispersion mode array has inconsistent dimensions"))
    physical_modes = Array{ComplexF64}(undef, D, nbasis, nbranch, nq)

    @inbounds for iq in 1:nq, ν in 1:nbranch, site in 1:nbasis
        offset = (site - 1) * D
        for d in 1:D
            row = offset + d
            physical_modes[d, site, ν, iq] = ComplexF64(result.modes[row, ν, iq]) / sqrt(masses[row])
        end
    end
    return physical_modes
end

function _multiphonon_physical_projections(workspace::HarmonicMultiphononWorkspace, Qcart::AbstractVector{<:Real})
    D, nbasis, nbranch, nq = size(workspace.physical_modes)
    length(Qcart) == D || throw(DimensionMismatch("scattering-vector dimension does not match the phonon displacement dimension"))
    projections = Array{ComplexF64}(undef, nq, nbranch, nbasis)

    @inbounds for iq in 1:nq, ν in 1:nbranch, site in 1:nbasis
        projection = 0.0 + 0.0im
        for d in 1:D
            projection += Float64(Qcart[d]) * workspace.physical_modes[d, site, ν, iq]
        end
        projections[iq, ν, site] = projection
    end
    return projections
end

function _multiphonon_zero_mask(result::PhononDispersionResult, zero_mode_tol::Union{Nothing,Real})
    frequencies = result.frequencies
    if isnothing(zero_mode_tol) && haskey(result.metadata, :zero_mode_mask)
        mask = BitMatrix(result.metadata[:zero_mode_mask])
        size(mask) == size(frequencies) || throw(DimensionMismatch("solver zero-mode mask does not match phonon dispersion"))
        return mask, :solver, nothing, nothing
    end

    tolerance = isnothing(zero_mode_tol) ? 1e-10 : Float64(zero_mode_tol)
    tolerance >= 0 || throw(ArgumentError("zero_mode_tol must be nonnegative"))
    threshold = tolerance * max(maximum(abs, frequencies; init=0.0), 1.0)
    source = isnothing(zero_mode_tol) ? :legacy_frequency_fallback : :explicit_frequency_tolerance
    return BitMatrix(frequencies .<= threshold), source, tolerance, threshold
end

function _multiphonon_physical_projections(result::PhononDispersionResult, Qcart::AbstractVector{<:Real})
    model = result.model
    D = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    masses = Float64.(model.parameters[:masses])
    length(masses) == nbasis * D ||
        throw(DimensionMismatch("phonon mass vector does not match primitive-cell displacement degrees of freedom"))
    nq, nbranch = size(result.frequencies)
    size(result.modes) == (nbranch, nbranch, nq) || throw(DimensionMismatch("phonon dispersion mode array has inconsistent dimensions"))
    projections = Array{ComplexF64}(undef, nq, nbranch, nbasis)

    for iq in 1:nq, ν in 1:nbranch, site in 1:nbasis
        rows = ((site - 1) * D + 1):(site * D)
        polarization = ComplexF64.(view(result.modes, rows, ν, iq)) ./ sqrt.(view(masses, rows))
        projections[iq, ν, site] = dot(Qcart, polarization)
    end
    return projections
end

function _multiphonon_occupations(result::PhononDispersionResult, temperature::Real, convention::SpectrumConvention)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    frequencies = Float64.(result.frequencies)
    occupations = zeros(Float64, size(frequencies))
    iszero(temperature) && return occupations
    scale = Float64(convention.hbar) / (Float64(convention.kB) * Float64(temperature))
    @inbounds for index in eachindex(frequencies)
        ω = frequencies[index]
        occupations[index] = ω > 0 ? inv(expm1(scale * ω)) : Inf
    end
    return occupations
end

function _multiphonon_debye_waller_amplitudes(
    result::PhononDispersionResult, Qcart, projections, occupations, zero_mask, displacement_scale, debye_waller,
)
    nbasis = result.model.parameters[:phonon_basis_count]
    if isnothing(debye_waller)
        return ones(Float64, nbasis), :none
    elseif debye_waller === :auto
        nq, nbranch = size(result.frequencies)
        W = zeros(Float64, nbasis)
        for iq in 1:nq, ν in 1:nbranch
            zero_mask[iq, ν] && continue
            ω = Float64(result.frequencies[iq, ν])
            thermal = 2 * occupations[iq, ν] + 1
            prefactor = displacement_scale * thermal / (4 * nq * ω)
            @inbounds for site in 1:nbasis
                W[site] += prefactor * abs2(projections[iq, ν, site])
            end
        end
        return exp.(-W), :harmonic_mesh
    end

    values = collect(debye_waller)
    length(values) == nbasis || throw(DimensionMismatch("debye_waller must contain one tensor per primitive-cell basis site"))
    return Float64[_debye_waller_amplitude(values[site], Qcart) for site in 1:nbasis], :supplied
end

function _reciprocal_mesh_index_map(result::PhononDispersionResult, shape::NTuple{D,<:Integer}; tolerance::Real=1e-8) where {D}
    dims = ntuple(d -> Int(shape[d]), D)
    all(>(0), dims) || throw(ArgumentError("all reciprocal-mesh dimensions must be positive"))
    prod(dims) == length(result.qpoints) || throw(DimensionMismatch(
        "reciprocal mesh shape contains $(prod(dims)) points but dispersion contains $(length(result.qpoints))",
    ))
    B = Matrix{Float64}(result.model.parameters[:phonon_reciprocal_matrix])
    size(B) == (D, D) || throw(DimensionMismatch("reciprocal-mesh dimension does not match phonon model"))
    mapping = Dict{NTuple{D,Int},Int}()

    for (iq, qcart) in enumerate(result.qpoints)
        reduced = B \ Float64.(qcart)
        index = ntuple(d -> mod(round(Int, reduced[d] * dims[d]), dims[d]), D)
        target = Float64[index[d] / dims[d] for d in 1:D]
        canonical = mod.(reduced, 1.0)
        residual = maximum(min(abs(canonical[d] - target[d]), 1.0 - abs(canonical[d] - target[d])) for d in 1:D)
        residual <= tolerance ||
            throw(ArgumentError("phonon dispersion q point $iq is not on the requested uniform reciprocal mesh; residual=$residual"))
        haskey(mapping, index) && throw(ArgumentError("phonon dispersion contains duplicate reciprocal-mesh index $index"))
        mapping[index] = iq
    end
    length(mapping) == prod(dims) || throw(ArgumentError("phonon dispersion does not cover the full reciprocal mesh"))
    return mapping
end


"""
    harmonic_multiphonon_workspace(dispersion; qmesh_shape, temperature=0, convention=SpectrumConvention(), displacement_prefactor=nothing, zero_mode_tol=nothing)

Precompute reciprocal-mesh indexing, physical polarization vectors, Bose occupations, and the zero-mode mask for repeated coherent multiphonon calculations on one phonon dispersion. The workspace is read-only after construction and is safe to share across map-point threads.
"""
function harmonic_multiphonon_workspace(
    result::PhononDispersionResult; qmesh_shape, temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega), displacement_prefactor::Union{Nothing,Real}=nothing,
    zero_mode_tol::Union{Nothing,Real}=nothing, mesh_tolerance::Real=1e-8,
)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    mesh_tolerance > 0 || throw(ArgumentError("mesh_tolerance must be positive"))
    dims = Tuple(Int.(qmesh_shape))
    D = length(dims)
    all(>(0), dims) || throw(ArgumentError("all qmesh dimensions must be positive"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    mesh_map = _reciprocal_mesh_index_map(result, dims; tolerance=mesh_tolerance)
    mesh_indices = Vector{NTuple{D,Int}}(undef, length(result.qpoints))
    for (index, iq) in mesh_map
        mesh_indices[iq] = index
    end
    occupations = _multiphonon_occupations(result, temperature, convention)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = _multiphonon_zero_mask(result, zero_mode_tol)
    physical_modes = _multiphonon_physical_modes(result)
    zero_mode_request = isnothing(zero_mode_tol) ? nothing : Float64(zero_mode_tol)
    return HarmonicMultiphononWorkspace(
        result, dims, mesh_map, mesh_indices, physical_modes, occupations, zero_mask, Float64(temperature), Float64(convention.hbar),
        Float64(convention.kB), displacement_scale, zero_mode_request, zero_mode_source, zero_mode_tolerance,
        zero_mode_frequency_threshold, Float64(mesh_tolerance),
    )
end

function _validate_multiphonon_workspace(
    workspace::HarmonicMultiphononWorkspace, result::PhononDispersionResult, dims, temperature::Real,
    convention::SpectrumConvention, displacement_scale::Real, zero_mode_tol, mesh_tolerance::Real,
)
    workspace.result === result || throw(ArgumentError("harmonic multiphonon workspace belongs to a different phonon dispersion"))
    workspace.qmesh_shape == dims || throw(DimensionMismatch("workspace reciprocal mesh does not match qmesh_shape"))
    workspace.temperature == Float64(temperature) || throw(ArgumentError("workspace temperature does not match requested temperature"))
    workspace.hbar == Float64(convention.hbar) || throw(ArgumentError("workspace hbar does not match spectrum convention"))
    workspace.kB == Float64(convention.kB) || throw(ArgumentError("workspace kB does not match spectrum convention"))
    workspace.displacement_scale == Float64(displacement_scale) ||
        throw(ArgumentError("workspace displacement scale does not match request"))
    requested_zero_tol = isnothing(zero_mode_tol) ? nothing : Float64(zero_mode_tol)
    isequal(workspace.zero_mode_request, requested_zero_tol) || throw(ArgumentError("workspace zero-mode policy does not match request"))
    workspace.mesh_tolerance == Float64(mesh_tolerance) || throw(ArgumentError("workspace mesh tolerance does not match request"))
    return workspace
end

function _with_blas_threads(f::F, target::Union{Nothing,Integer}) where {F}
    isnothing(target) && return f()
    requested = Int(target)
    requested >= 1 || throw(ArgumentError("blas_threads must be positive"))
    previous = LinearAlgebra.BLAS.get_num_threads()
    previous == requested && return f()
    LinearAlgebra.BLAS.set_num_threads(requested)
    try
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(previous)
    end
end

function _reduced_mesh_index(model::ManyBodyModel, Qcart, shape::NTuple{D,<:Integer}; tolerance::Real=1e-8) where {D}
    dims = ntuple(d -> Int(shape[d]), D)
    B = Matrix{Float64}(model.parameters[:phonon_reciprocal_matrix])
    reduced = B \ Float64.(Qcart)
    index = ntuple(d -> mod(round(Int, reduced[d] * dims[d]), dims[d]), D)
    target = Float64[index[d] / dims[d] for d in 1:D]
    canonical = mod.(reduced, 1.0)
    residual = maximum(min(abs(canonical[d] - target[d]), 1.0 - abs(canonical[d] - target[d])) for d in 1:D)
    residual <= tolerance || throw(ArgumentError("Q does not reduce to the requested reciprocal mesh; residual=$residual"))
    return index
end

@inline function _second_mesh_index(target::NTuple{D,Int}, first::NTuple{D,Int}, sign1::Int, sign2::Int, dims::NTuple{D,Int}) where {D}
    return ntuple(d -> mod(sign2 * (target[d] - sign1 * first[d]), dims[d]), D)
end

function _multiphonon_site_prefactors(result::PhononDispersionResult, Qcart, debye_waller_amplitudes)
    lengths = _basis_coherent_scattering_lengths(result.model)
    positions = _basis_cartesian_positions(result.model)
    nbasis = length(positions)
    return ComplexF64[
        lengths[site] * debye_waller_amplitudes[site] * cis(dot(Qcart, positions[site])) for site in 1:nbasis
    ]
end


@inline function _one_phonon_mesh_index(target::NTuple{D,Int}, sign::Int, dims::NTuple{D,Int}) where {D}
    sign in (-1, 1) || throw(ArgumentError("one-phonon process sign must be -1 or +1"))
    return ntuple(d -> mod(sign * target[d], dims[d]), D)
end

"""
    one_phonon_neutron_lines(probe, dispersion, Q; qmesh_shape, temperature=0, convention=SpectrumConvention(), debye_waller=:auto, zero_mode_tol=nothing)

Return coherent harmonic one-phonon creation and annihilation lines from a complete uniform reciprocal mesh. The creation channel uses the mode at reduced momentum `q = Q mod G`, while the annihilation channel uses `q = -Q mod G`; this distinction is required for finite-temperature coherent scattering in a multi-atom basis.

This mesh-aware overload is the reciprocal-space spectral counterpart of the first-order term returned by `harmonic_coherent_intermediate_scattering`. The existing `PhononModeResult` overload remains available for local single-q calculations.
"""
function one_phonon_neutron_lines(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q; qmesh_shape, temperature::Real=0.0,
    convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega), debye_waller=:auto,
    displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing,
    mesh_tolerance::Real=1e-8,
)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    dims = Tuple(Int.(qmesh_shape))
    D = length(dims)
    all(>(0), dims) || throw(ArgumentError("all qmesh dimensions must be positive"))
    Qcart = _nuclear_qvector(result.model, Q)
    length(Qcart) == D || throw(DimensionMismatch("qmesh dimension must match the phonon model"))
    mesh_map = _reciprocal_mesh_index_map(result, dims; tolerance=mesh_tolerance)
    target_index = _reduced_mesh_index(result.model, Qcart, dims; tolerance=mesh_tolerance)
    projections = _multiphonon_physical_projections(result, Qcart)
    occupations = _multiphonon_occupations(result, temperature, convention)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = _multiphonon_zero_mask(result, zero_mode_tol)
    dw_amplitudes, debye_waller_source = _multiphonon_debye_waller_amplitudes(
        result, Qcart, projections, occupations, zero_mask, displacement_scale, debye_waller,
    )
    site_prefactors = _multiphonon_site_prefactors(result, Qcart, dw_amplitudes)
    hbar = Float64(convention.hbar)
    centers = Float64[]
    weights = ComplexF64[]
    process_counts = Dict(:creation => 0, :annihilation => 0)

    for sign in (1, -1)
        process_index = _one_phonon_mesh_index(target_index, sign, dims)
        iq = get(mesh_map, process_index, 0)
        iszero(iq) && continue
        for ν in axes(result.frequencies, 2)
            zero_mask[iq, ν] && continue
            occupation = occupations[iq, ν]
            thermal = sign == 1 ? occupation + 1 : occupation
            iszero(thermal) && continue
            omega = Float64(result.frequencies[iq, ν])
            amplitude = 0.0 + 0.0im
            for site in eachindex(site_prefactors)
                projection = projections[iq, ν, site]
                leg = sign == 1 ? projection : conj(projection)
                amplitude += site_prefactors[site] * leg
            end
            weight = probe.prefactor * displacement_scale * thermal * abs2(amplitude) / (2 * omega)
            iszero(weight) && continue
            center_omega = sign * omega
            center = convention.spectral_axis === :energy ? hbar * center_omega : center_omega
            push!(centers, center)
            push!(weights, ComplexF64(weight))
            process_counts[sign == 1 ? :creation : :annihilation] += 1
        end
    end

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :harmonic_multiphonon => true,
        :order => 1,
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :hbar => hbar,
        :kB => Float64(convention.kB),
        :displacement_prefactor => displacement_scale,
        :reciprocal_mesh_shape => dims,
        :qpoint_count => size(result.frequencies, 1),
        :branch_count => size(result.frequencies, 2),
        :process_counts => process_counts,
        :debye_waller_source => debye_waller_source,
        :zero_mode_source => zero_mode_source,
        :zero_mode_tolerance => zero_mode_tolerance,
        :zero_mode_frequency_threshold => zero_mode_frequency_threshold,
        :Q_cartesian => Qcart,
        :normalization => :per_primitive_cell,
        :momentum_policy => :creation_q_equals_Q_annihilation_q_equals_minus_Q_mod_G,
        :resolution_applied => false,
    )
    return LehmannLines(centers, weights, metadata)
end

"""
    one_phonon_neutron_intensity(probe, dispersion, Q, axis; qmesh_shape, broadening, kwargs...)

Broaden the mesh-aware coherent one-phonon creation and annihilation lines on `axis`.
"""
function one_phonon_neutron_intensity(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, axis; qmesh_shape,
    broadening::AbstractBroadening, kwargs...,
)
    lines = one_phonon_neutron_lines(probe, result, Q; qmesh_shape=qmesh_shape, kwargs...)
    values = broaden_lines(lines, axis, broadening)
    imag_scale = maximum(abs, imag.(values); init=0.0)
    real_scale = max(maximum(abs, real.(values); init=0.0), 1.0)
    imag_scale <= 1e-10 * real_scale || throw(ArgumentError("coherent one-phonon spectrum acquired a significant imaginary component"))
    x = Float64.(collect(axis))
    Qcart = _nuclear_qvector(result.model, Q)
    metadata = copy(lines.metadata)
    metadata[:broadening] = typeof(broadening)
    intensity = reshape(Float64.(real.(values)), 1, length(values))
    return ScatteringResult([Qcart], x, intensity, metadata)
end

scattering_intensity(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, axis; qmesh_shape,
    kwargs...,
) = one_phonon_neutron_intensity(probe, result, Q, axis; qmesh_shape=qmesh_shape, kwargs...)

"""
    harmonic_coherent_intermediate_scattering(probe, dispersion, Q, times; qmesh_shape, order=:all, temperature=0, convention=SpectrumConvention(), debye_waller=:auto, zero_mode_tol=nothing)

Evaluate the coherent harmonic inelastic intermediate scattering function on a complete reciprocal-space phonon mesh. The Gaussian displacement identity is expanded either to a specific positive phonon order `order=n` through `C(t)^n / n!` or to all inelastic harmonic orders with `order=:all` through `exp(C(t)) - 1`.

The result is normalized per primitive cell. `debye_waller=:auto` derives the harmonic Debye-Waller amplitudes from the same mesh; `debye_waller=nothing` disables Debye-Waller attenuation; a supplied collection is contracted with the full Cartesian scattering vector. `mode_frequency_max` optionally restricts the displacement correlation to modes below a caller-selected frequency while leaving the Debye-Waller factor untruncated; this is an explicit approximation intended for controlled low-energy reconstructions.
"""
function harmonic_coherent_intermediate_scattering(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, times; qmesh_shape, order::Union{Integer,Symbol}=:all,
    temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega), debye_waller=:auto,
    displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing,
    mode_frequency_max::Union{Nothing,Real}=nothing,
)
    order === :all || (order isa Integer && order >= 1) || throw(ArgumentError("order must be :all or a positive integer"))
    isnothing(mode_frequency_max) || Float64(mode_frequency_max) > 0 || throw(ArgumentError("mode_frequency_max must be positive"))
    frequency_cutoff = isnothing(mode_frequency_max) ? Inf : Float64(mode_frequency_max)
    t = Float64.(collect(times))
    isempty(t) && throw(ArgumentError("times cannot be empty"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    Qcart = _nuclear_qvector(result.model, Q)
    D = result.model.parameters[:spatial_dimension]
    length(Qcart) == D || throw(DimensionMismatch("coherent multiphonon scattering requires displacement and spatial dimensions to match"))
    nq, nbranch = size(result.frequencies)
    projections = _multiphonon_physical_projections(result, Qcart)
    occupations = _multiphonon_occupations(result, temperature, convention)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = _multiphonon_zero_mask(result, zero_mode_tol)
    dw_amplitudes, debye_waller_source = _multiphonon_debye_waller_amplitudes(
        result, Qcart, projections, occupations, zero_mask, displacement_scale, debye_waller,
    )
    site_prefactors = _multiphonon_site_prefactors(result, Qcart, dw_amplitudes)
    nbasis = result.model.parameters[:phonon_basis_count]
    direct = Matrix{Float64}(result.model.parameters[:phonon_direct_matrix])
    dims = Tuple(Int.(qmesh_shape))
    length(dims) == size(direct, 2) || throw(DimensionMismatch("reciprocal-mesh dimension does not match phonon model"))
    prod(dims) == nq || throw(DimensionMismatch("reciprocal-mesh shape does not match dispersion q count"))
    _reciprocal_mesh_index_map(result, dims)
    translations = collect(Iterators.product((0:(dim - 1) for dim in dims)...))
    values = zeros(ComplexF64, length(t))
    C = Matrix{ComplexF64}(undef, nbasis, nbasis)

    for (it, time) in enumerate(t)
        total = 0.0 + 0.0im
        for cell_tuple in translations
            cell = Float64.(collect(cell_tuple))
            R = direct * cell
            fill!(C, 0.0 + 0.0im)
            for iq in 1:nq, ν in 1:nbranch
                zero_mask[iq, ν] && continue
                ω = Float64(result.frequencies[iq, ν])
                ω <= frequency_cutoff || continue
                qphase = cis(dot(result.qpoints[iq], R) - ω * time)
                thermal_creation = occupations[iq, ν] + 1
                thermal_annihilation = occupations[iq, ν]
                prefactor = displacement_scale / (2 * nq * ω)
                for a in 1:nbasis, b in 1:nbasis
                    va = projections[iq, ν, a]
                    vb = projections[iq, ν, b]
                    C[a, b] += prefactor * (
                        thermal_creation * va * conj(vb) * qphase + thermal_annihilation * conj(va) * vb * conj(qphase)
                    )
                end
            end

            lattice_phase = cis(-dot(Qcart, R))
            for a in 1:nbasis, b in 1:nbasis
                expansion = order === :all ? exp(C[a, b]) - 1 : C[a, b]^Int(order) / factorial(Int(order))
                total += lattice_phase * site_prefactors[a] * conj(site_prefactors[b]) * expansion
            end
        end
        values[it] = probe.prefactor * total
    end

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :harmonic_multiphonon => true,
        :order => order,
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :hbar => Float64(convention.hbar),
        :kB => Float64(convention.kB),
        :displacement_prefactor => displacement_scale,
        :reciprocal_mesh_shape => dims,
        :qpoint_count => nq,
        :branch_count => nbranch,
        :debye_waller_source => debye_waller_source,
        :zero_mode_source => zero_mode_source,
        :zero_mode_tolerance => zero_mode_tolerance,
        :zero_mode_frequency_threshold => zero_mode_frequency_threshold,
        :mode_frequency_max => isnothing(mode_frequency_max) ? nothing : frequency_cutoff,
        :Q_cartesian => Qcart,
        :normalization => :per_primitive_cell,
    )
    return CorrelationResult(t, values, metadata)
end

@inline function _multiphonon_axis_to_omega(convention::SpectrumConvention)
    return convention.spectral_axis === :energy ? inv(Float64(convention.hbar)) : 1.0
end

function _multiphonon_time_damping(broadening::GaussianBroadening, time::Real, axis_to_omega::Real)
    sigma_omega = Float64(broadening.sigma) * Float64(axis_to_omega)
    return exp(-0.5 * (sigma_omega * Float64(time))^2)
end

function _multiphonon_time_damping(broadening::LorentzianBroadening, time::Real, axis_to_omega::Real)
    width_omega = Float64(broadening.width) * Float64(axis_to_omega)
    return exp(-width_omega * abs(Float64(time)))
end

function _multiphonon_default_time_extent(broadening::GaussianBroadening, axis_to_omega::Real)
    sigma_omega = Float64(broadening.sigma) * Float64(axis_to_omega)
    sigma_omega > 0 || throw(ArgumentError("Gaussian multiphonon transform width must be positive"))
    return 6 / sigma_omega
end

function _multiphonon_default_time_extent(broadening::LorentzianBroadening, axis_to_omega::Real)
    width_omega = Float64(broadening.width) * Float64(axis_to_omega)
    width_omega > 0 || throw(ArgumentError("Lorentzian multiphonon transform width must be positive"))
    return 12 / width_omega
end

"""
    harmonic_multiphonon_neutron_intensity(probe, dispersion, Q, axis; qmesh_shape, order=:all, broadening, ...)

Fourier transform the Gaussian harmonic intermediate scattering function into a coherent inelastic neutron spectrum. `order=n` returns the selected harmonic phonon order and `order=:all` returns all inelastic harmonic orders through `exp(C)-1`.

The time-domain damping is the exact Fourier partner of the requested `GaussianBroadening` or `LorentzianBroadening`, so `broadening` has the same spectral-axis units used elsewhere in the observable API. `time_points` controls the symmetric quadrature grid and must be odd; `time_extent=nothing` chooses a damping-based default. The returned spectrum is normalized per primitive cell and per unit of the selected spectral axis.
"""
function harmonic_multiphonon_neutron_intensity(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, axis; qmesh_shape, order::Union{Integer,Symbol}=:all,
    broadening::AbstractBroadening, temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
    debye_waller=:auto, displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing,
    mode_frequency_max::Union{Nothing,Real}=nothing, time_points::Integer=257, time_extent::Union{Nothing,Real}=nothing,
)
    broadening isa Union{GaussianBroadening,LorentzianBroadening} || throw(ArgumentError(
        "harmonic multiphonon Fourier transforms currently support GaussianBroadening or LorentzianBroadening",
    ))
    time_points >= 3 || throw(ArgumentError("time_points must be at least 3"))
    isodd(time_points) || throw(ArgumentError("time_points must be odd so that the transform grid contains t=0"))
    x = Float64.(collect(axis))
    isempty(x) && throw(ArgumentError("spectral axis cannot be empty"))
    all(isfinite, x) || throw(ArgumentError("spectral axis must contain only finite values"))
    axis_to_omega = _multiphonon_axis_to_omega(convention)
    extent = isnothing(time_extent) ? _multiphonon_default_time_extent(broadening, axis_to_omega) : Float64(time_extent)
    extent > 0 || throw(ArgumentError("time_extent must be positive"))
    times = collect(range(-extent, extent; length=Int(time_points)))
    correlation = harmonic_coherent_intermediate_scattering(
        probe, result, Q, times; qmesh_shape=qmesh_shape, order=order, temperature=temperature, convention=convention,
        debye_waller=debye_waller, displacement_prefactor=displacement_prefactor, zero_mode_tol=zero_mode_tol,
        mode_frequency_max=mode_frequency_max,
    )
    dt = times[2] - times[1]
    quadrature = ones(Float64, length(times))
    quadrature[1] = 0.5
    quadrature[end] = 0.5
    damped = ComplexF64[
        quadrature[index] * _multiphonon_time_damping(broadening, times[index], axis_to_omega) * correlation.values[index]
        for index in eachindex(times)
    ]
    values = zeros(ComplexF64, length(x))
    normalization = axis_to_omega * dt / (2π)
    for ix in eachindex(x)
        omega = x[ix] * axis_to_omega
        values[ix] = normalization * sum(damped[index] * cis(omega * times[index]) for index in eachindex(times))
    end
    imag_scale = maximum(abs, imag.(values); init=0.0)
    real_scale = max(maximum(abs, real.(values); init=0.0), 1.0)
    imag_scale <= 1e-7 * real_scale || throw(ArgumentError(
        "harmonic multiphonon Fourier transform acquired a significant imaginary component; increase time_points or time_extent",
    ))
    Qcart = _nuclear_qvector(result.model, Q)
    metadata = copy(correlation.metadata)
    metadata[:broadening] = typeof(broadening)
    metadata[:time_points] = Int(time_points)
    metadata[:time_extent] = extent
    metadata[:fourier_quadrature] = :symmetric_trapezoidal
    metadata[:resolution_applied] = false
    intensity = reshape(Float64.(real.(values)), 1, length(values))
    return ScatteringResult([Qcart], x, intensity, metadata)
end

"""
    two_phonon_neutron_lines(probe, dispersion, Q; qmesh_shape, temperature=0, convention=SpectrumConvention(), debye_waller=:auto, zero_mode_tol=nothing)

Return the coherent harmonic two-phonon creation/annihilation/difference lines for a complete uniform reciprocal mesh. Momentum conservation is enforced modulo reciprocal lattice vectors, and the ordered mode-pair sum carries the `1 / 2!` expansion factor.
"""
function two_phonon_neutron_lines(probe::NuclearNeutronProbe, result::PhononDispersionResult, Q; qmesh_shape,
                                  temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
                                  debye_waller=:auto, displacement_prefactor::Union{Nothing,Real}=nothing,
                                  zero_mode_tol::Union{Nothing,Real}=nothing, mesh_tolerance::Real=1e-8)
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    dims = Tuple(Int.(qmesh_shape))
    D = length(dims)
    all(>(0), dims) || throw(ArgumentError("all qmesh dimensions must be positive"))
    Qcart = _nuclear_qvector(result.model, Q)
    length(Qcart) == D || throw(DimensionMismatch("qmesh dimension must match the phonon model"))
    mesh_map = _reciprocal_mesh_index_map(result, dims; tolerance=mesh_tolerance)
    target_index = _reduced_mesh_index(result.model, Qcart, dims; tolerance=mesh_tolerance)
    nq, nbranch = size(result.frequencies)
    projections = _multiphonon_physical_projections(result, Qcart)
    occupations = _multiphonon_occupations(result, temperature, convention)
    zero_mask, zero_mode_source, zero_mode_tolerance, zero_mode_frequency_threshold = _multiphonon_zero_mask(result, zero_mode_tol)
    dw_amplitudes, debye_waller_source = _multiphonon_debye_waller_amplitudes(
        result, Qcart, projections, occupations, zero_mask, displacement_scale, debye_waller,
    )
    site_prefactors = _multiphonon_site_prefactors(result, Qcart, dw_amplitudes)
    hbar = Float64(convention.hbar)
    centers = Float64[]
    weights = ComplexF64[]
    process_counts = Dict(
        :creation_creation => 0, :creation_annihilation => 0, :annihilation_creation => 0, :annihilation_annihilation => 0,
    )
    signs = (1, -1)

    for (index1, iq1) in mesh_map, s1 in signs
        for s2 in signs
            needed = _second_mesh_index(target_index, index1, s1, s2, dims)
            iq2 = get(mesh_map, needed, 0)
            iszero(iq2) && continue
            for ν1 in 1:nbranch, ν2 in 1:nbranch
                zero_mask[iq1, ν1] && continue
                zero_mask[iq2, ν2] && continue
                n1 = occupations[iq1, ν1]
                n2 = occupations[iq2, ν2]
                thermal1 = s1 == 1 ? n1 + 1 : n1
                thermal2 = s2 == 1 ? n2 + 1 : n2
                (iszero(thermal1) || iszero(thermal2)) && continue
                ω1 = Float64(result.frequencies[iq1, ν1])
                ω2 = Float64(result.frequencies[iq2, ν2])
                amplitude = 0.0 + 0.0im
                for site in eachindex(site_prefactors)
                    p1 = projections[iq1, ν1, site]
                    p2 = projections[iq2, ν2, site]
                    leg1 = s1 == 1 ? p1 : conj(p1)
                    leg2 = s2 == 1 ? p2 : conj(p2)
                    amplitude += site_prefactors[site] * leg1 * leg2
                end
                oscillator = displacement_scale^2 / (4 * nq * ω1 * ω2)
                weight = probe.prefactor * 0.5 * oscillator * thermal1 * thermal2 * abs2(amplitude)
                iszero(weight) && continue
                center_omega = s1 * ω1 + s2 * ω2
                center = convention.spectral_axis === :energy ? hbar * center_omega : center_omega
                push!(centers, center)
                push!(weights, ComplexF64(weight))
                process = s1 == 1 && s2 == 1 ? :creation_creation :
                          s1 == 1 && s2 == -1 ? :creation_annihilation :
                          s1 == -1 && s2 == 1 ? :annihilation_creation : :annihilation_annihilation
                process_counts[process] += 1
            end
        end
    end

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :harmonic_multiphonon => true,
        :order => 2,
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :hbar => hbar,
        :kB => Float64(convention.kB),
        :displacement_prefactor => displacement_scale,
        :reciprocal_mesh_shape => dims,
        :qpoint_count => nq,
        :branch_count => nbranch,
        :process_counts => process_counts,
        :debye_waller_source => debye_waller_source,
        :zero_mode_source => zero_mode_source,
        :zero_mode_tolerance => zero_mode_tolerance,
        :zero_mode_frequency_threshold => zero_mode_frequency_threshold,
        :Q_cartesian => Qcart,
        :normalization => :per_primitive_cell,
        :resolution_applied => false,
    )
    return LehmannLines(centers, weights, metadata)
end

@inline function _standard_normal_cdf(z::Real)
    x = abs(Float64(z))
    t = inv(1 + 0.2316419 * x)
    polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    upper_tail = exp(-0.5 * x^2) * polynomial / sqrt(2π)
    cdf = 1 - upper_tail
    return z >= 0 ? cdf : 1 - cdf
end

@inline function _spectral_window_fraction(center::Real, lower::Real, upper::Real, ::Nothing)
    return lower <= center <= upper ? 1.0 : 0.0
end

@inline function _spectral_window_fraction(center::Real, lower::Real, upper::Real, broadening::GaussianBroadening)
    invsigma = inv(Float64(broadening.sigma))
    return clamp(_standard_normal_cdf((upper - center) * invsigma) - _standard_normal_cdf((lower - center) * invsigma), 0.0, 1.0)
end

@inline function _spectral_window_fraction(center::Real, lower::Real, upper::Real, broadening::LorentzianBroadening)
    width = Float64(broadening.width)
    return clamp((atan((upper - center) / width) - atan((lower - center) / width)) / π, 0.0, 1.0)
end

@inline _spectral_window_margin(::Nothing, tail_cutoff::Real) = 0.0
@inline _spectral_window_margin(broadening::GaussianBroadening, tail_cutoff::Real) = Float64(tail_cutoff) * Float64(broadening.sigma)
@inline _spectral_window_margin(::LorentzianBroadening, tail_cutoff::Real) = Inf

function _mode_window_bounds(frequencies, s1::Int, ω1::Real, s2::Int, lower_omega::Real, upper_omega::Real)
    if s2 == 1
        lo = lower_omega - s1 * ω1
        hi = upper_omega - s1 * ω1
    else
        lo = s1 * ω1 - upper_omega
        hi = s1 * ω1 - lower_omega
    end
    hi < 0 && return 1:0
    lo = max(lo, 0.0)
    first_index = searchsortedfirst(frequencies, lo)
    last_index = searchsortedlast(frequencies, hi)
    if first_index <= last_index
        return first_index:last_index
    end
    return 1:0
end

function _integrate_lehmann_window(lines::LehmannLines, lower::Real, upper::Real, broadening)
    total = 0.0 + 0.0im
    for (center, weight) in zip(lines.centers, lines.weights)
        total += weight * _spectral_window_fraction(center, lower, upper, broadening)
    end
    return total
end

"""
    one_phonon_neutron_window(probe, dispersion, Q, lower, upper; qmesh_shape, broadening=nothing, kwargs...)

Integrate the coherent harmonic one-phonon response over `[lower, upper]`. With `broadening=nothing`, the intrinsic delta-line weights are selected by their centers. With Gaussian or Lorentzian broadening, each line contributes the analytically integrated normalized line shape over the requested window.
"""
function one_phonon_neutron_window(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, lower::Real, upper::Real; qmesh_shape,
    broadening::Union{Nothing,AbstractBroadening}=nothing, kwargs...
)
    lower < upper || throw(ArgumentError("one-phonon integration window must satisfy lower < upper"))
    isnothing(broadening) || broadening isa Union{GaussianBroadening,LorentzianBroadening} || throw(ArgumentError(
        "one-phonon window integration supports GaussianBroadening, LorentzianBroadening, or nothing",
    ))
    lines = one_phonon_neutron_lines(probe, result, Q; qmesh_shape=qmesh_shape, kwargs...)
    total = _integrate_lehmann_window(lines, Float64(lower), Float64(upper), broadening)
    imag_scale = abs(imag(total))
    real_scale = max(abs(real(total)), 1.0)
    imag_scale <= 1e-10 * real_scale || throw(ArgumentError("coherent one-phonon window acquired a significant imaginary component"))
    metadata = copy(lines.metadata)
    metadata[:integration_window] = (Float64(lower), Float64(upper))
    metadata[:window_broadening] = isnothing(broadening) ? :intrinsic : typeof(broadening)
    metadata[:resolution_applied] = !isnothing(broadening)
    return (intensity=Float64(real(total)), metadata=metadata)
end


function _resolve_two_phonon_parallel(parallel::Symbol, nq::Integer)
    parallel in (:auto, :serial, :qmesh) || throw(ArgumentError("two-phonon parallel policy must be :auto, :serial, or :qmesh"))
    Threads.nthreads() <= 1 && return :serial
    parallel === :serial && return :serial
    parallel === :qmesh && return :qmesh
    return nq >= 2 * Threads.nthreads() ? :qmesh : :serial
end

"""
    two_phonon_neutron_window(probe, dispersion, Q, lower, upper; qmesh_shape, temperature=0, convention=SpectrumConvention(), debye_waller=:auto, zero_mode_tol=nothing)

Integrate the coherent harmonic two-phonon response over the closed spectral window `[lower, upper]` without allocating the complete `LehmannLines` object. Momentum conservation and thermal factors are identical to `two_phonon_neutron_lines`. With `broadening=nothing`, intrinsic delta-line weights are selected by center; Gaussian or Lorentzian broadening integrates the normalized line shape over the window. Gaussian candidate pairs outside `broadening_tail` standard deviations of the window are skipped before the coherent basis amplitude is constructed.

The returned named tuple contains `intensity` and diagnostic `metadata`. This route is intended for reciprocal-space maps integrated over a finite spectral window.
"""
function two_phonon_neutron_window(
    probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, lower::Real, upper::Real; qmesh_shape,
    temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega), debye_waller=:auto,
    displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing, mesh_tolerance::Real=1e-8,
    broadening::Union{Nothing,AbstractBroadening}=nothing, broadening_tail::Real=8.0,
    workspace::Union{Nothing,HarmonicMultiphononWorkspace}=nothing, parallel::Symbol=:auto,
)
    lower < upper || throw(ArgumentError("two-phonon integration window must satisfy lower < upper"))
    temperature >= 0 || throw(ArgumentError("temperature must be nonnegative"))
    broadening_tail > 0 || throw(ArgumentError("broadening_tail must be positive"))
    isnothing(broadening) || broadening isa Union{GaussianBroadening,LorentzianBroadening} || throw(ArgumentError(
        "two-phonon window integration supports GaussianBroadening, LorentzianBroadening, or nothing",
    ))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    displacement_scale > 0 || throw(ArgumentError("displacement_prefactor must be positive"))
    dims = Tuple(Int.(qmesh_shape))
    D = length(dims)
    all(>(0), dims) || throw(ArgumentError("all qmesh dimensions must be positive"))
    Qcart = _nuclear_qvector(result.model, Q)
    length(Qcart) == D || throw(DimensionMismatch("qmesh dimension must match the phonon model"))
    active_workspace = isnothing(workspace) ? harmonic_multiphonon_workspace(
        result; qmesh_shape=dims, temperature=temperature, convention=convention, displacement_prefactor=displacement_scale,
        zero_mode_tol=zero_mode_tol, mesh_tolerance=mesh_tolerance,
    ) : _validate_multiphonon_workspace(
        workspace, result, dims, temperature, convention, displacement_scale, zero_mode_tol, mesh_tolerance,
    )
    mesh_map = active_workspace.mesh_map
    target_index = _reduced_mesh_index(result.model, Qcart, dims; tolerance=mesh_tolerance)
    nq, nbranch = size(result.frequencies)
    projections = _multiphonon_physical_projections(active_workspace, Qcart)
    occupations = active_workspace.occupations
    zero_mask = active_workspace.zero_mask
    dw_amplitudes, debye_waller_source = _multiphonon_debye_waller_amplitudes(
        result, Qcart, projections, occupations, zero_mask, displacement_scale, debye_waller,
    )
    site_prefactors = _multiphonon_site_prefactors(result, Qcart, dw_amplitudes)
    hbar = Float64(convention.hbar)
    lower_value = Float64(lower)
    upper_value = Float64(upper)
    margin = _spectral_window_margin(broadening, broadening_tail)
    search_lower = isfinite(margin) ? lower_value - margin : -Inf
    search_upper = isfinite(margin) ? upper_value + margin : Inf
    axis_to_omega = convention.spectral_axis === :energy ? inv(hbar) : 1.0
    lower_omega = search_lower * axis_to_omega
    upper_omega = search_upper * axis_to_omega
    signs = (1, -1)
    parallel_backend = _resolve_two_phonon_parallel(parallel, nq)

    function q_contribution(iq1::Int)
        index1 = active_workspace.mesh_indices[iq1]
        frequencies1 = view(result.frequencies, iq1, :)
        subtotal = 0.0
        accepted = 0
        considered = 0
        for s1 in signs, s2 in signs
            needed = _second_mesh_index(target_index, index1, s1, s2, dims)
            iq2 = get(mesh_map, needed, 0)
            iszero(iq2) && continue
            frequencies2 = view(result.frequencies, iq2, :)
            for ν1 in 1:nbranch
                zero_mask[iq1, ν1] && continue
                ω1 = Float64(frequencies1[ν1])
                n1 = occupations[iq1, ν1]
                thermal1 = s1 == 1 ? n1 + 1 : n1
                iszero(thermal1) && continue
                ν2_range = isfinite(lower_omega) && isfinite(upper_omega) ?
                           _mode_window_bounds(frequencies2, s1, ω1, s2, lower_omega, upper_omega) : (1:nbranch)
                for ν2 in ν2_range
                    zero_mask[iq2, ν2] && continue
                    considered += 1
                    ω2 = Float64(frequencies2[ν2])
                    n2 = occupations[iq2, ν2]
                    thermal2 = s2 == 1 ? n2 + 1 : n2
                    iszero(thermal2) && continue
                    center_omega = s1 * ω1 + s2 * ω2
                    center = convention.spectral_axis === :energy ? hbar * center_omega : center_omega
                    window_fraction = _spectral_window_fraction(center, lower_value, upper_value, broadening)
                    iszero(window_fraction) && continue
                    amplitude = 0.0 + 0.0im
                    @inbounds for site in eachindex(site_prefactors)
                        p1 = projections[iq1, ν1, site]
                        p2 = projections[iq2, ν2, site]
                        leg1 = s1 == 1 ? p1 : conj(p1)
                        leg2 = s2 == 1 ? p2 : conj(p2)
                        amplitude += site_prefactors[site] * leg1 * leg2
                    end
                    oscillator = displacement_scale^2 / (4 * nq * ω1 * ω2)
                    subtotal += window_fraction * probe.prefactor * 0.5 * oscillator * thermal1 * thermal2 * abs2(amplitude)
                    accepted += 1
                end
            end
        end
        return subtotal, accepted, considered
    end

    partial_totals = zeros(Float64, nq)
    partial_accepted = zeros(Int, nq)
    partial_considered = zeros(Int, nq)
    if parallel_backend === :qmesh
        Threads.@threads :static for iq1 in 1:nq
            partial_totals[iq1], partial_accepted[iq1], partial_considered[iq1] = q_contribution(iq1)
        end
    else
        @inbounds for iq1 in 1:nq
            partial_totals[iq1], partial_accepted[iq1], partial_considered[iq1] = q_contribution(iq1)
        end
    end
    total = sum(partial_totals)
    accepted_pairs = sum(partial_accepted)
    considered_pairs = sum(partial_considered)

    metadata = Dict{Symbol,Any}(
        :probe => :coherent_nuclear_neutron,
        :harmonic_multiphonon => true,
        :order => 2,
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :hbar => hbar,
        :kB => Float64(convention.kB),
        :displacement_prefactor => displacement_scale,
        :reciprocal_mesh_shape => dims,
        :qpoint_count => nq,
        :branch_count => nbranch,
        :integration_window => (lower_value, upper_value),
        :accepted_mode_pairs => accepted_pairs,
        :considered_mode_pairs => considered_pairs,
        :debye_waller_source => debye_waller_source,
        :zero_mode_source => active_workspace.zero_mode_source,
        :zero_mode_tolerance => active_workspace.zero_mode_tolerance,
        :zero_mode_frequency_threshold => active_workspace.zero_mode_frequency_threshold,
        :window_broadening => isnothing(broadening) ? :intrinsic : typeof(broadening),
        :broadening_tail => isnothing(broadening) ? nothing : Float64(broadening_tail),
        :Q_cartesian => Qcart,
        :normalization => :per_primitive_cell,
        :resolution_applied => !isnothing(broadening),
        :parallel_backend => parallel_backend,
        :julia_threads => Threads.nthreads(),
        :workspace_reused => !isnothing(workspace),
    )
    return (intensity=Float64(total), metadata=metadata)
end

struct HarmonicNeutronSliceResult{X,Y,V,M}
    fixed_axis::Symbol
    fixed_value::Float64
    horizontal_axis::Symbol
    horizontal_values::X
    vertical_axis::Symbol
    vertical_values::Y
    intensity::V
    metadata::M
end

@inline function _reciprocal_axis_index(axis::Symbol)
    axis === :H && return 1
    axis === :K && return 2
    axis === :L && return 3
    throw(ArgumentError("reciprocal slice axes must be :H, :K, or :L"))
end

function _reciprocal_slice_vector(fixed_axis, fixed_value, horizontal_axis, horizontal_value, vertical_axis, vertical_value)
    indices = (_reciprocal_axis_index(fixed_axis), _reciprocal_axis_index(horizontal_axis), _reciprocal_axis_index(vertical_axis))
    length(unique(indices)) == 3 || throw(ArgumentError("fixed, horizontal, and vertical reciprocal axes must be distinct"))
    q = zeros(Float64, 3)
    q[indices[1]] = Float64(fixed_value)
    q[indices[2]] = Float64(horizontal_value)
    q[indices[3]] = Float64(vertical_value)
    return ReciprocalWaveVector(q)
end

@inline function _trapz_uniform(x, y)
    length(x) == length(y) || throw(DimensionMismatch("trapezoidal axis and values must have equal lengths"))
    isempty(x) && throw(ArgumentError("trapezoidal integration requires at least one point"))
    length(x) == 1 && return Float64(y[1])
    total = 0.0
    @inbounds for index in 1:(length(x) - 1)
        total += 0.5 * (y[index] + y[index + 1]) * (x[index + 1] - x[index])
    end
    return total
end

function _harmonic_slice_integrated_intensity(slice::HarmonicNeutronSliceResult)
    horizontal = slice.horizontal_values
    vertical = slice.vertical_values
    row_integrals = Vector{Float64}(undef, length(vertical))
    @inbounds for iv in eachindex(vertical)
        row_integrals[iv] = _trapz_uniform(horizontal, view(slice.intensity, iv, :))
    end
    return _trapz_uniform(vertical, row_integrals)
end

function _resolve_slice_parallel(parallel::Symbol, order, npoints::Integer)
    parallel in (:auto, :serial, :map, :qmesh) ||
        throw(ArgumentError("slice parallel policy must be :auto, :serial, :map, or :qmesh"))
    Threads.nthreads() <= 1 && return :serial
    parallel === :serial && return :serial
    if parallel === :qmesh
        order == 2 || throw(ArgumentError("parallel=:qmesh is currently supported only for order=2 slices"))
        return :qmesh
    end
    parallel === :map && return :map
    npoints >= Threads.nthreads() && return :map
    return order == 2 ? :qmesh : (npoints > 1 ? :map : :serial)
end

"""
    harmonic_neutron_slice(probe, dispersion; fixed_axis, fixed_value, horizontal_axis, horizontal_values, vertical_axis, vertical_values, qmesh_shape, energy_window, order=2, parallel=:auto, ...)

Evaluate a fixed reciprocal-space plane integrated over an energy window. `order=1` uses the coherent one-phonon response, `order=2` uses the explicit coherent two-phonon response, and `order=:all` uses the Gaussian harmonic intermediate-scattering function. The result stores the map as `intensity[vertical, horizontal]`.

`parallel=:auto` uses independent map-point threading when the map contains enough points, otherwise q-mesh threading for `order=2`. `parallel=:map`, `:qmesh`, and `:serial` select the backend explicitly. Nested threading is disabled: map-threaded calculations evaluate each pixel with a serial inner q-mesh sum. A reusable `HarmonicMultiphononWorkspace` can be supplied for `order=2`; otherwise it is constructed once per slice.
"""
function harmonic_neutron_slice(
    probe::NuclearNeutronProbe, result::PhononDispersionResult; fixed_axis::Symbol, fixed_value::Real,
    horizontal_axis::Symbol, horizontal_values, vertical_axis::Symbol, vertical_values, qmesh_shape, energy_window,
    order::Union{Integer,Symbol}=2, temperature::Real=0.0, convention::SpectrumConvention=SpectrumConvention(spectral_axis=:omega),
    debye_waller=:auto, displacement_prefactor::Union{Nothing,Real}=nothing, zero_mode_tol::Union{Nothing,Real}=nothing,
    broadening::Union{Nothing,AbstractBroadening}=nothing, energy_points::Integer=129, time_points::Integer=257,
    mode_frequency_max::Union{Nothing,Real}=nothing, mesh_tolerance::Real=1e-8,
    workspace::Union{Nothing,HarmonicMultiphononWorkspace}=nothing, parallel::Symbol=:auto, blas_threads::Union{Nothing,Integer}=nothing,
)
    order === :all || (order isa Integer && order >= 1) || throw(ArgumentError("order must be :all or a positive integer"))
    lower, upper = Float64.(energy_window)
    lower < upper || throw(ArgumentError("energy_window must satisfy lower < upper"))
    hvalues = Float64.(collect(horizontal_values))
    vvalues = Float64.(collect(vertical_values))
    isempty(hvalues) && throw(ArgumentError("horizontal_values cannot be empty"))
    isempty(vvalues) && throw(ArgumentError("vertical_values cannot be empty"))
    intensity = zeros(Float64, length(vvalues), length(hvalues))
    dims = Tuple(Int.(qmesh_shape))
    displacement_scale = isnothing(displacement_prefactor) ? Float64(convention.hbar) : Float64(displacement_prefactor)
    active_workspace = nothing
    if order == 2
        active_workspace = isnothing(workspace) ? harmonic_multiphonon_workspace(
            result; qmesh_shape=dims, temperature=temperature, convention=convention, displacement_prefactor=displacement_scale,
            zero_mode_tol=zero_mode_tol, mesh_tolerance=mesh_tolerance,
        ) : _validate_multiphonon_workspace(
            workspace, result, dims, temperature, convention, displacement_scale, zero_mode_tol, mesh_tolerance,
        )
    elseif !isnothing(workspace)
        throw(ArgumentError("a harmonic multiphonon workspace is currently used only by order=2 slice calculations"))
    end

    n_horizontal = length(hvalues)
    n_vertical = length(vvalues)
    npoints = n_horizontal * n_vertical
    parallel_backend = _resolve_slice_parallel(parallel, order, npoints)
    inner_parallel = parallel_backend === :qmesh ? :qmesh : :serial

    function evaluate_pixel(linear_index::Int)
        iv = fld(linear_index - 1, n_horizontal) + 1
        ih = mod(linear_index - 1, n_horizontal) + 1
        Q = _reciprocal_slice_vector(fixed_axis, fixed_value, horizontal_axis, hvalues[ih], vertical_axis, vvalues[iv])
        if order == 1
            window = one_phonon_neutron_window(
                probe, result, Q, lower, upper; qmesh_shape=dims, broadening=broadening, temperature=temperature,
                convention=convention, debye_waller=debye_waller, displacement_prefactor=displacement_prefactor,
                zero_mode_tol=zero_mode_tol,
            )
            intensity[iv, ih] = window.intensity
        elseif order == 2
            window = two_phonon_neutron_window(
                probe, result, Q, lower, upper; qmesh_shape=dims, broadening=broadening, temperature=temperature,
                convention=convention, debye_waller=debye_waller, displacement_prefactor=displacement_prefactor,
                zero_mode_tol=zero_mode_tol, mesh_tolerance=mesh_tolerance, workspace=active_workspace, parallel=inner_parallel,
            )
            intensity[iv, ih] = window.intensity
        else
            isnothing(broadening) && throw(ArgumentError("order=:all and order>2 slice calculations require spectral broadening"))
            energy_points >= 3 || throw(ArgumentError("energy_points must be at least 3"))
            axis = collect(range(lower, upper; length=Int(energy_points)))
            spectrum = harmonic_multiphonon_neutron_intensity(
                probe, result, Q, axis; qmesh_shape=dims, order=order, broadening=broadening, temperature=temperature,
                convention=convention, debye_waller=debye_waller, displacement_prefactor=displacement_prefactor,
                zero_mode_tol=zero_mode_tol, mode_frequency_max=mode_frequency_max, time_points=time_points,
            )
            intensity[iv, ih] = _trapz_uniform(axis, vec(spectrum.intensity))
        end
        return nothing
    end

    requested_blas_threads = isnothing(blas_threads) && parallel_backend !== :serial ? 1 : blas_threads
    blas_threads_during = isnothing(requested_blas_threads) ? LinearAlgebra.BLAS.get_num_threads() : Int(requested_blas_threads)
    run_slice = function ()
        if parallel_backend === :map
            Threads.@threads :static for linear_index in 1:npoints
                evaluate_pixel(linear_index)
            end
        else
            @inbounds for linear_index in 1:npoints
                evaluate_pixel(linear_index)
            end
        end
        return nothing
    end
    _with_blas_threads(run_slice, requested_blas_threads)

    metadata = Dict{Symbol,Any}(
        :order => order,
        :energy_window => (lower, upper),
        :temperature => Float64(temperature),
        :spectral_axis => convention.spectral_axis,
        :reciprocal_mesh_shape => dims,
        :broadening => isnothing(broadening) ? :intrinsic : typeof(broadening),
        :normalization => :per_primitive_cell,
        :parallel_backend => parallel_backend,
        :julia_threads => Threads.nthreads(),
        :blas_threads_during => blas_threads_during,
        :workspace_reused => !isnothing(workspace),
    )
    return HarmonicNeutronSliceResult(
        fixed_axis, Float64(fixed_value), horizontal_axis, hvalues, vertical_axis, vvalues, intensity, metadata,
    )
end

function _harmonic_slice_residual(previous::HarmonicNeutronSliceResult, current::HarmonicNeutronSliceResult)
    size(previous.intensity) == size(current.intensity) || throw(DimensionMismatch("slice maps must have equal dimensions"))
    total_previous = _harmonic_slice_integrated_intensity(previous)
    total_current = _harmonic_slice_integrated_intensity(current)
    total_residual = abs(total_current - total_previous) / max(abs(total_current), abs(total_previous), eps(Float64))
    if total_previous <= eps(Float64) && total_current <= eps(Float64)
        shape_residual = 0.0
    elseif total_previous <= eps(Float64) || total_current <= eps(Float64)
        shape_residual = Inf
    else
        previous_shape = previous.intensity ./ total_previous
        current_shape = current.intensity ./ total_current
        shape_residual = norm(current_shape - previous_shape) / max(norm(current_shape), norm(previous_shape), eps(Float64))
    end
    return (total=total_residual, shape=shape_residual, maximum=max(total_residual, shape_residual))
end

function _harmonic_power_law_fit(history, field::Symbol)
    valid = [entry for entry in history if getproperty(entry, field) > 0 && isfinite(getproperty(entry, field))]
    length(valid) >= 3 || return nothing
    fit_entries = valid[max(1, length(valid) - 2):end]
    x = log.([Float64(entry.nq) for entry in fit_entries])
    y = log.([Float64(getproperty(entry, field)) for entry in fit_entries])
    xmean = sum(x) / length(x)
    ymean = sum(y) / length(y)
    denominator = sum((value - xmean)^2 for value in x)
    denominator > 0 || return nothing
    exponent = sum((x[i] - xmean) * (y[i] - ymean) for i in eachindex(x)) / denominator
    intercept = ymean - exponent * xmean
    predicted = intercept .+ exponent .* x
    ss_residual = sum((y .- predicted).^2)
    ss_total = sum((y .- ymean).^2)
    r2 = ss_total > 0 ? 1 - ss_residual / ss_total : 1.0
    return (exponent=exponent, r2=r2, points=length(fit_entries), nq_range=(fit_entries[1].nq, fit_entries[end].nq))
end

function _harmonic_convergence_scaling(history)
    return (
        dispersion=_harmonic_power_law_fit(history, :dispersion_time),
        workspace=_harmonic_power_law_fit(history, :workspace_time),
        slice=_harmonic_power_law_fit(history, :slice_time),
        total=_harmonic_power_law_fit(history, :total_time),
        allocation=_harmonic_power_law_fit(history, :total_alloc_bytes),
    )
end

"""
    converge_harmonic_neutron_slice(probe, solver, model; mesh_sizes, convergence_tol=0.05, required_consecutive=1, parallel=:auto, kwargs...)

Refine a uniform reciprocal phonon mesh until a finite reciprocal-space slice is stable in both integrated intensity and normalized map shape. Each mesh is evaluated once: the same accuracy-convergence trajectory records phonon-dispersion time, reusable-workspace time, slice time, total wall time, and allocated bytes. Runtime and allocation power-law fits use the three largest completed meshes when available.

`convergence_tol` remains the backward-compatible default for both residuals. `total_tol` and `shape_tol` can override the integrated-intensity and normalized-map tolerances independently. A mesh counts as converged only when both criteria pass.
"""
function converge_harmonic_neutron_slice(
    probe::NuclearNeutronProbe, solver::HarmonicPhononSolver, model::ManyBodyModel; mesh_sizes,
    convergence_tol::Real=0.05, total_tol::Real=convergence_tol, shape_tol::Real=convergence_tol,
    required_consecutive::Integer=1, parallel::Symbol=:auto, blas_threads::Union{Nothing,Integer}=nothing, kwargs...,
)
    convergence_tol > 0 || throw(ArgumentError("convergence_tol must be positive"))
    total_tol > 0 || throw(ArgumentError("total_tol must be positive"))
    shape_tol > 0 || throw(ArgumentError("shape_tol must be positive"))
    required_consecutive >= 1 || throw(ArgumentError("required_consecutive must be positive"))
    haskey(kwargs, :workspace) && throw(ArgumentError("convergence builds one workspace per mesh; do not pass workspace explicitly"))
    haskey(kwargs, :qmesh_shape) && throw(ArgumentError("convergence determines qmesh_shape from mesh_sizes"))
    sizes = Int.(collect(mesh_sizes))
    length(sizes) >= 2 || throw(ArgumentError("mesh_sizes must contain at least two refinements"))
    all(>(0), sizes) || throw(ArgumentError("mesh sizes must be positive"))
    issorted(sizes) || throw(ArgumentError("mesh_sizes must be sorted from coarse to fine"))
    length(unique(sizes)) == length(sizes) || throw(ArgumentError("mesh_sizes must not contain duplicates"))
    history = NamedTuple[]
    previous = nothing
    consecutive = 0
    final_slice = nothing
    converged_mesh = nothing
    order = get(kwargs, :order, 2)
    temperature = get(kwargs, :temperature, 0.0)
    convention = get(kwargs, :convention, SpectrumConvention(spectral_axis=:omega))
    displacement_prefactor = get(kwargs, :displacement_prefactor, nothing)
    zero_mode_tol = get(kwargs, :zero_mode_tol, nothing)
    mesh_tolerance = get(kwargs, :mesh_tolerance, 1e-8)

    for mesh_size in sizes
        shape = (mesh_size, mesh_size, mesh_size)
        dispersion_timing = @timed phonon_dispersion(solver, model, phonon_reciprocal_mesh(shape))
        dispersion = dispersion_timing.value
        workspace = nothing
        workspace_time = 0.0
        workspace_alloc_bytes = 0
        if order == 2
            workspace_timing = @timed harmonic_multiphonon_workspace(
                dispersion; qmesh_shape=shape, temperature=temperature, convention=convention,
                displacement_prefactor=displacement_prefactor, zero_mode_tol=zero_mode_tol, mesh_tolerance=mesh_tolerance,
            )
            workspace = workspace_timing.value
            workspace_time = workspace_timing.time
            workspace_alloc_bytes = workspace_timing.bytes
        end
        slice_timing = @timed harmonic_neutron_slice(
            probe, dispersion; qmesh_shape=shape, workspace=workspace, parallel=parallel, blas_threads=blas_threads, kwargs...,
        )
        slice = slice_timing.value
        residual = isnothing(previous) ? (total=Inf, shape=Inf, maximum=Inf) : _harmonic_slice_residual(previous, slice)
        accuracy_pass = !isnothing(previous) && residual.total <= total_tol && residual.shape <= shape_tol
        consecutive = accuracy_pass ? consecutive + 1 : 0
        integrated_intensity = _harmonic_slice_integrated_intensity(slice)
        total_time = dispersion_timing.time + workspace_time + slice_timing.time
        total_alloc_bytes = dispersion_timing.bytes + workspace_alloc_bytes + slice_timing.bytes
        push!(history, (
            mesh_size=mesh_size, nq=mesh_size^3, residual=residual, integrated_intensity=integrated_intensity,
            dispersion_time=dispersion_timing.time, workspace_time=workspace_time, slice_time=slice_timing.time,
            total_time=total_time, dispersion_alloc_bytes=dispersion_timing.bytes, workspace_alloc_bytes=workspace_alloc_bytes,
            slice_alloc_bytes=slice_timing.bytes, total_alloc_bytes=total_alloc_bytes,
            parallel_backend=slice.metadata[:parallel_backend], julia_threads=slice.metadata[:julia_threads],
            blas_threads=slice.metadata[:blas_threads_during], accuracy_pass=accuracy_pass,
        ))
        final_slice = slice
        if consecutive >= required_consecutive
            converged_mesh = mesh_size
            scaling = _harmonic_convergence_scaling(history)
            return (
                result=final_slice, history=history, converged=true, converged_mesh=converged_mesh, scaling=scaling,
                tolerances=(total=Float64(total_tol), shape=Float64(shape_tol), required_consecutive=Int(required_consecutive)),
            )
        end
        previous = slice
    end
    scaling = _harmonic_convergence_scaling(history)
    return (
        result=final_slice, history=history, converged=false, converged_mesh=converged_mesh, scaling=scaling,
        tolerances=(total=Float64(total_tol), shape=Float64(shape_tol), required_consecutive=Int(required_consecutive)),
    )
end


"""
    two_phonon_neutron_intensity(probe, dispersion, Q, axis; broadening, kwargs...)

Broaden the intrinsic coherent two-phonon lines on `axis`. The broadening is a spectral line-shape model and does not by itself represent a multidimensional instrument-resolution convolution.
"""
function two_phonon_neutron_intensity(probe::NuclearNeutronProbe, result::PhononDispersionResult, Q, axis;
                                      broadening::AbstractBroadening, kwargs...)
    lines = two_phonon_neutron_lines(probe, result, Q; kwargs...)
    values = broaden_lines(lines, axis, broadening)
    imag_scale = maximum(abs, imag.(values); init=0.0)
    real_scale = max(maximum(abs, real.(values); init=0.0), 1.0)
    imag_scale <= 1e-10 * real_scale || throw(ArgumentError("coherent two-phonon spectrum acquired a significant imaginary component"))
    x = Float64.(collect(axis))
    Qcart = _nuclear_qvector(result.model, Q)
    metadata = copy(lines.metadata)
    metadata[:broadening] = typeof(broadening)
    intensity = reshape(Float64.(real.(values)), 1, length(values))
    return ScatteringResult([Qcart], x, intensity, metadata)
end
