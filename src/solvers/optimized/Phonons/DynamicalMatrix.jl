@inline _phonon_mode_index(site::Integer, component::Integer, D::Integer) =
    (Int(site) - 1) * Int(D) + Int(component)

function _phonon_cartesian_wavevector(model::ManyBodyModel, q)
    D = model.parameters[:spatial_dimension]
    qv = if q isa CartesianWaveVector
        Float64.(q.coordinates)
    elseif q isa ReciprocalWaveVector
        hkl = Float64.(q.coordinates)
        length(hkl) == D || throw(DimensionMismatch("reciprocal wavevector must have $D components"))
        B = get(model.parameters, :phonon_reciprocal_matrix, nothing)
        isnothing(B) && throw(ArgumentError("model does not retain a reciprocal-lattice matrix"))
        Matrix{Float64}(B) * hkl
    else
        Float64.(collect(q))
    end
    length(qv) == D || throw(DimensionMismatch("wavevector must have $D components"))
    all(isfinite, qv) || throw(ArgumentError("wavevector components must be finite"))
    return qv
end

_phonon_wavevector_basis(q) =
    q isa ReciprocalWaveVector ? :reciprocal :
    q isa CartesianWaveVector ? :cartesian : :cartesian_legacy

"""
    dynamical_matrix(model)

Return the mass-weighted harmonic dynamical matrix for a coordinate phonon
model. For a periodic real-space IFC model this is the Γ-point matrix.
"""
function dynamical_matrix(model::ManyBodyModel)
    if haskey(model.parameters, :periodic_force_constants)
        return dynamical_matrix(
            model,
            CartesianWaveVector(zeros(Float64, model.parameters[:spatial_dimension])),
        )
    end

    cached = get(model.parameters, :mass_weighted_dynamical_matrix, nothing)
    !isnothing(cached) && return Matrix{Float64}(cached)

    Φ = get(model.parameters, :force_constants, nothing)
    masses = get(model.parameters, :masses, nothing)
    isnothing(Φ) && throw(ArgumentError("model does not carry force_constants"))
    isnothing(masses) && throw(ArgumentError("model does not carry phonon masses"))
    Φ = Matrix{Float64}(real.(Φ))
    masses = Float64.(collect(masses))
    size(Φ, 1) == size(Φ, 2) == length(masses) ||
        throw(DimensionMismatch("force constants and masses have inconsistent dimensions"))
    all(>(0), masses) || throw(ArgumentError("phonon masses must be positive"))
    invsqrtm = Diagonal(1 ./ sqrt.(masses))
    return Matrix(invsqrtm * Φ * invsqrtm)
end

function _periodic_block_separations(model::ManyBodyModel, ifcs)
    cached = get(model.parameters, :phonon_block_separations, nothing)
    if !isnothing(cached)
        length(cached) == length(ifcs.blocks) ||
            throw(DimensionMismatch("cached phonon block separations do not match periodic force constants"))
        return cached
    end

    direct = model.parameters[:phonon_direct_matrix]
    basis_fractional = model.parameters[:phonon_basis_fractional]
    return [
        direct * (
            Float64.(block.translation) .+
            basis_fractional[block.to_site] .-
            basis_fractional[block.from_site]
        )
        for block in ifcs.blocks
    ]
end

function _periodic_phonon_dimensions(model::ManyBodyModel)
    ifcs = get(model.parameters, :periodic_force_constants, nothing)
    isnothing(ifcs) &&
        throw(ArgumentError("q-resolved phonon operations require periodic real-space force constants"))
    Ddisp = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    ndof = nbasis * Ddisp
    masses = Float64.(collect(model.parameters[:masses]))
    length(masses) == ndof ||
        throw(DimensionMismatch("phonon mass vector does not match primitive-cell displacement dimension"))
    return ifcs, Ddisp, nbasis, ndof, masses
end

@inline function _weighted_block_entry(block, α, β, ia, ib, masses, mass_weighted)
    return mass_weighted ?
        ComplexF64(block.tensor[α, β]) :
        ComplexF64(block.tensor[α, β]) / sqrt(masses[ia] * masses[ib])
end

"""
    dynamical_matrix!(Dq, model, q)

Fill `Dq` with the primitive-cell Bloch dynamical matrix. `q` may be a raw
vector (legacy Cartesian convention), `CartesianWaveVector`, or
`ReciprocalWaveVector`.
"""
function dynamical_matrix!(Dq::AbstractMatrix{ComplexF64}, model::ManyBodyModel, q)
    ifcs, Ddisp, nbasis, ndof, masses = _periodic_phonon_dimensions(model)
    size(Dq) == (ndof, ndof) || throw(DimensionMismatch("Dq must be $ndof×$ndof"))
    qv = _phonon_cartesian_wavevector(model, q)
    mass_weighted = get(model.parameters, :mass_weighted_force_constants, false)
    separations = _periodic_block_separations(model, ifcs)
    fill!(Dq, 0)

    for (block, separation) in zip(ifcs.blocks, separations)
        phase = cis(dot(qv, separation))
        @inbounds for α in 1:Ddisp, β in 1:Ddisp
            ia = _phonon_mode_index(block.from_site, α, Ddisp)
            ib = _phonon_mode_index(block.to_site, β, Ddisp)
            Dq[ia, ib] += phase * _weighted_block_entry(
                block, α, β, ia, ib, masses, mass_weighted,
            )
        end
    end
    return Dq
end

"""
    dynamical_matrix(model, q)

Construct

    D_{κα,κ'β}(q) =
        Σ_R Φ_{κα,κ'β}(R) exp[i q·(R+τκ'-τκ)] / √(mκ mκ').

Raw vectors retain the historical Cartesian convention. Prefer explicit
`CartesianWaveVector(q)` or `ReciprocalWaveVector(hkl)` for new code.
"""
function dynamical_matrix(model::ManyBodyModel, q)
    _, _, _, ndof, _ = _periodic_phonon_dimensions(model)
    Dq = zeros(ComplexF64, ndof, ndof)
    return dynamical_matrix!(Dq, model, q)
end

"""
    dynamical_gradient!(dD, model, q)
    dynamical_gradient(model, q)

Analytic first derivatives of the mass-weighted dynamical matrix. `dD` has
shape `(ndof, ndof, spatial_dimension)` and

    ∂D/∂qμ = i Σ_b r_bμ D_b exp(i q·r_b).
"""
function dynamical_gradient!(dD::AbstractArray{ComplexF64,3}, model::ManyBodyModel, q)
    ifcs, Ddisp, _, ndof, masses = _periodic_phonon_dimensions(model)
    Dspace = model.parameters[:spatial_dimension]
    size(dD) == (ndof, ndof, Dspace) ||
        throw(DimensionMismatch("dD must have shape ($ndof,$ndof,$Dspace)"))
    qv = _phonon_cartesian_wavevector(model, q)
    mass_weighted = get(model.parameters, :mass_weighted_force_constants, false)
    separations = _periodic_block_separations(model, ifcs)
    fill!(dD, 0)

    for (block, separation) in zip(ifcs.blocks, separations)
        phase = cis(dot(qv, separation))
        @inbounds for α in 1:Ddisp, β in 1:Ddisp
            ia = _phonon_mode_index(block.from_site, α, Ddisp)
            ib = _phonon_mode_index(block.to_site, β, Ddisp)
            weighted = _weighted_block_entry(block, α, β, ia, ib, masses, mass_weighted)
            for μ in 1:Dspace
                dD[ia, ib, μ] += (1im * separation[μ]) * phase * weighted
            end
        end
    end
    return dD
end

function dynamical_gradient(model::ManyBodyModel, q)
    _, _, _, ndof, _ = _periodic_phonon_dimensions(model)
    Dspace = model.parameters[:spatial_dimension]
    dD = zeros(ComplexF64, ndof, ndof, Dspace)
    return dynamical_gradient!(dD, model, q)
end

"""
    dynamical_hessian!(d2D, model, q)
    dynamical_hessian(model, q)

Analytic second derivatives of the mass-weighted dynamical matrix. `d2D` has
shape `(ndof, ndof, spatial_dimension, spatial_dimension)` and

    ∂²D/(∂qμ∂qν) =
        -Σ_b r_bμ r_bν D_b exp(i q·r_b).
"""
function dynamical_hessian!(
    d2D::AbstractArray{ComplexF64,4},
    model::ManyBodyModel,
    q,
)
    ifcs, Ddisp, _, ndof, masses = _periodic_phonon_dimensions(model)
    Dspace = model.parameters[:spatial_dimension]
    size(d2D) == (ndof, ndof, Dspace, Dspace) ||
        throw(DimensionMismatch("d2D must have shape ($ndof,$ndof,$Dspace,$Dspace)"))
    qv = _phonon_cartesian_wavevector(model, q)
    mass_weighted = get(model.parameters, :mass_weighted_force_constants, false)
    separations = _periodic_block_separations(model, ifcs)
    fill!(d2D, 0)

    for (block, separation) in zip(ifcs.blocks, separations)
        phase = cis(dot(qv, separation))
        @inbounds for α in 1:Ddisp, β in 1:Ddisp
            ia = _phonon_mode_index(block.from_site, α, Ddisp)
            ib = _phonon_mode_index(block.to_site, β, Ddisp)
            weighted = _weighted_block_entry(block, α, β, ia, ib, masses, mass_weighted)
            for μ in 1:Dspace, ν in 1:Dspace
                d2D[ia, ib, μ, ν] +=
                    -(separation[μ] * separation[ν]) * phase * weighted
            end
        end
    end
    return d2D
end

function dynamical_hessian(model::ManyBodyModel, q)
    _, _, _, ndof, _ = _periodic_phonon_dimensions(model)
    Dspace = model.parameters[:spatial_dimension]
    d2D = zeros(ComplexF64, ndof, ndof, Dspace, Dspace)
    return dynamical_hessian!(d2D, model, q)
end

"""
    acoustic_sum_rule_residual(model)

Return the largest normalized Γ-point residual of the rigid-translation vectors
for a periodic harmonic phonon model.
"""
function acoustic_sum_rule_residual(model::ManyBodyModel)
    haskey(model.parameters, :periodic_force_constants) ||
        throw(ArgumentError("acoustic_sum_rule_residual requires periodic real-space force constants"))
    D0 = dynamical_matrix(model)
    masses = Float64.(collect(model.parameters[:masses]))
    Ddisp = model.parameters[:displacement_dimension]
    nbasis = model.parameters[:phonon_basis_count]
    scale = max(norm(D0), 1.0)
    residual = 0.0

    for α in 1:Ddisp
        translation = zeros(ComplexF64, nbasis * Ddisp)
        @inbounds for site in 1:nbasis
            index = _phonon_mode_index(site, α, Ddisp)
            translation[index] = sqrt(masses[index])
        end
        residual = max(
            residual,
            norm(D0 * translation) /
            (scale * max(norm(translation), eps(Float64))),
        )
    end
    return residual
end
