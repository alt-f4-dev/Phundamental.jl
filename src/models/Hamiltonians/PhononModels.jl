# Harmonic lattice models in coordinate/momentum or local-boson form.

"""A periodic real-space interatomic-force-constant block coupling primitive-cell basis site `from_site` to basis site `to_site` in the cell displaced by the integer Bravais translation `translation`."""
struct ForceConstantBlock{T<:Number}
    from_site::Int
    to_site::Int
    translation::Vector{Int}
    tensor::Matrix{T}
end

function ForceConstantBlock(from_site::Integer, to_site::Integer, translation, tensor::AbstractMatrix{<:Number})
    from_site >= 1 || throw(ArgumentError("from_site must be positive"))
    to_site >= 1 || throw(ArgumentError("to_site must be positive"))
    t = Int.(collect(translation))
    isempty(t) && throw(ArgumentError("force-constant translation cannot be empty"))
    size(tensor, 1) == size(tensor, 2) || throw(DimensionMismatch("force-constant tensor must be square"))
    size(tensor, 1) >= 1 || throw(ArgumentError("force-constant tensor cannot be empty"))
    T = promote_type(Float64, eltype(tensor))
    return ForceConstantBlock{T}(Int(from_site), Int(to_site), t, Matrix{T}(tensor))
end

"""Collection of periodic real-space force-constant blocks with independent lattice-translation and displacement dimensions."""
struct RealSpaceForceConstants{T<:Number}
    blocks::Vector{ForceConstantBlock{T}}
    lattice_dimension::Int
    displacement_dimension::Int
end

function RealSpaceForceConstants(blocks; displacement_dimension=nothing)
    raw = collect(blocks)
    isempty(raw) && throw(ArgumentError("real-space force constants require at least one block"))
    all(block -> block isa ForceConstantBlock, raw) || throw(ArgumentError("real-space force constants must contain ForceConstantBlock entries"))

    lattice_dimension = length(first(raw).translation)
    all(length(block.translation) == lattice_dimension for block in raw) ||
        throw(DimensionMismatch("all force-constant translations must have the same lattice dimension"))
    inferred_dimension = size(first(raw).tensor, 1)
    all(size(block.tensor) == (inferred_dimension, inferred_dimension) for block in raw) ||
        throw(DimensionMismatch("all force-constant tensors must have the same displacement dimension"))

    D = isnothing(displacement_dimension) ? inferred_dimension : Int(displacement_dimension)
    D == inferred_dimension || throw(DimensionMismatch("displacement_dimension does not match the force-constant tensors"))
    T = foldl(promote_type, (eltype(block.tensor) for block in raw); init=Float64)
    entries = ForceConstantBlock{T}[
        ForceConstantBlock{T}(block.from_site, block.to_site, copy(block.translation), Matrix{T}(block.tensor)) for block in raw
    ]
    return RealSpaceForceConstants{T}(entries, lattice_dimension, D)
end

function _aggregate_force_constant_blocks(ifcs::RealSpaceForceConstants)
    blocks = Dict{Tuple{Int,Int,Tuple},Matrix{ComplexF64}}()
    for block in ifcs.blocks
        key = (block.from_site, block.to_site, Tuple(block.translation))
        value = ComplexF64.(block.tensor)
        if haskey(blocks, key)
            blocks[key] .+= value
        else
            blocks[key] = Matrix(value)
        end
    end
    return blocks
end

"""
    project_force_constants(ifcs; translational=true)

Return a new physical (not mass-weighted) real-space IFC set with the acoustic
sum rule enforced by replacing each onsite `(i,i,0)` block with the negative
sum of all other blocks leaving basis site `i`.

This operation is explicit and never occurs automatically during model
construction. Rotational projection is intentionally not performed because a
unique correction requires an additional metric/optimization prescription.
"""
function project_force_constants(ifcs::RealSpaceForceConstants; translational::Bool=true)
    translational || return RealSpaceForceConstants(copy(ifcs.blocks); displacement_dimension=ifcs.displacement_dimension)
    blocks = _aggregate_force_constant_blocks(ifcs)
    Dlat = ifcs.lattice_dimension
    Z = Tuple(zeros(Int, Dlat))
    sites = sort!(unique(first(key) for key in keys(blocks)))

    for i in sites
        total = zeros(ComplexF64, ifcs.displacement_dimension, ifcs.displacement_dimension)
        for ((ii, j, R), block) in blocks
            ii == i || continue
            (j == i && R == Z) && continue
            total .+= block
        end
        blocks[(i, i, Z)] = -total
    end

    entries = ForceConstantBlock[]
    for key in sort!(collect(keys(blocks)); by=x -> (x[1], x[2], x[3]))
        i, j, R = key
        block = blocks[key]
        imag_scale = maximum(abs, imag.(block); init=0.0)
        real_scale = max(maximum(abs, real.(block); init=0.0), 1.0)
        tensor = imag_scale <= 64eps(Float64) * real_scale ? real.(block) : block
        push!(entries, ForceConstantBlock(i, j, collect(R), tensor))
    end
    return RealSpaceForceConstants(entries; displacement_dimension=ifcs.displacement_dimension)
end

enforce_acoustic_sum_rule(ifcs::RealSpaceForceConstants) =
    project_force_constants(ifcs; translational=true)

"""
    PhononUnitConvention(; length=:unspecified, mass=:unspecified,
                          force_constant=:unspecified, frequency=:internal)

Metadata describing the numerical unit convention used by a harmonic phonon
model. The solver remains unit-agnostic; this object records conventions rather
than performing implicit conversions.
"""
struct PhononUnitConvention
    length::Symbol
    mass::Symbol
    force_constant::Symbol
    frequency::Symbol
end

PhononUnitConvention(
    ; length::Symbol=:unspecified,
      mass::Symbol=:unspecified,
      force_constant::Symbol=:unspecified,
      frequency::Symbol=:internal,
) = PhononUnitConvention(length, mass, force_constant, frequency)

"""Harmonic coordinate model. For `RealSpaceForceConstants`, `mass_weighted=true` means each stored tensor already contains the `1/sqrt(m_i m_j)` factors of the Bloch dynamical matrix; otherwise those factors are applied from `masses`."""
struct HarmonicPhononModel{F,M} <: AbstractHamiltonianSpec
    force_constants::F
    masses::M
    displacement_dimension::Union{Nothing,Int}
    mass_weighted::Bool
    symmetry_tolerance::Float64
    units::PhononUnitConvention
end

HarmonicPhononModel(force_constants, masses, symmetry_tolerance::Real) =
    HarmonicPhononModel(force_constants; masses=masses, symmetry_tolerance=symmetry_tolerance)

# Backward-compatible fieldwise constructor from the pre-unit-metadata API.
HarmonicPhononModel(force_constants, masses, displacement_dimension, mass_weighted::Bool, symmetry_tolerance::Real) =
    HarmonicPhononModel(
        force_constants, masses, displacement_dimension, mass_weighted,
        Float64(symmetry_tolerance), PhononUnitConvention(),
    )

function HarmonicPhononModel(
    force_constants;
    masses=nothing,
    displacement_dimension=nothing,
    mass_weighted::Bool=false,
    symmetry_tolerance::Real=1e-10,
    units::PhononUnitConvention=PhononUnitConvention(),
)
    symmetry_tolerance >= 0 || throw(ArgumentError("symmetry_tolerance must be nonnegative"))
    D = isnothing(displacement_dimension) ? nothing : Int(displacement_dimension)
    isnothing(D) || D >= 1 || throw(ArgumentError("displacement_dimension must be positive"))
    if force_constants isa RealSpaceForceConstants && !isnothing(D)
        D == force_constants.displacement_dimension ||
            throw(DimensionMismatch("HarmonicPhononModel displacement_dimension disagrees with RealSpaceForceConstants"))
    end
    return HarmonicPhononModel(force_constants, masses, D, mass_weighted, Float64(symmetry_tolerance), units)
end

struct LocalPhononModel{W} <: AbstractHamiltonianSpec
    frequencies::W
    cutoff::Union{Int,Vector{Int}}
    hbar::Float64
    include_zero_point::Bool
end

function LocalPhononModel(frequencies; cutoff, hbar::Real=1.0, include_zero_point::Bool=true)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cut = cutoff isa Integer ? Int(cutoff) : Int.(collect(cutoff))
    if cut isa Int
        cut >= 0 || throw(ArgumentError("boson cutoff must be nonnegative"))
    else
        all(>=(0), cut) || throw(ArgumentError("boson cutoffs must be nonnegative"))
    end
    return LocalPhononModel(frequencies, cut, Float64(hbar), include_zero_point)
end

function _phonon_masses(material::Material, cluster::Supercell, D::Int, supplied)
    N = nsites(cluster)
    ndof = N * D
    if isnothing(supplied)
        masses = Vector{Float64}(undef, ndof)
        @inbounds for i in 1:N
            M = site_mass(material, cluster, i)
            for alpha in 1:D
                masses[mode_index(i, alpha, D)] = M
            end
        end
        return masses
    end
    if supplied isa Number
        supplied > 0 || throw(ArgumentError("mass must be positive"))
        return fill(Float64(supplied), ndof)
    end
    masses = Float64.(collect(supplied))
    length(masses) == ndof || throw(DimensionMismatch("mass vector must have one value per displacement degree of freedom"))
    all(>(0), masses) || throw(ArgumentError("all masses must be positive"))
    return masses
end

function _coordinate_phonon_model(
    material::Material,
    cluster::Supercell,
    spec::HarmonicPhononModel,
    Phi::AbstractMatrix{<:Real},
    masses::Vector{Float64},
    D::Int;
    parameters=Dict{Symbol,Any}(),
    provenance_record=(construction=:harmonic_phonon, material=material.name, sites=nsites(cluster), ndof=size(Phi, 1)),
)
    N = nsites(cluster)
    ndof = N * D
    size(Phi) == (ndof, ndof) || throw(DimensionMismatch("force-constant matrix must be $ndof×$ndof for this model"))
    norm(Phi - transpose(Phi)) <= spec.symmetry_tolerance * max(norm(Phi), 1.0) ||
        throw(ArgumentError("force-constant matrix is not symmetric within tolerance"))
    Phi = 0.5 .* (Matrix{Float64}(Phi) .+ transpose(Matrix{Float64}(Phi)))

    terms = AbstractOperatorExpr[]
    sizehint!(terms, ndof + (ndof * (ndof + 1)) ÷ 2)
    @inbounds for a in 1:ndof
        push!(terms, inv(2 * masses[a]) * (pop(a) * pop(a)))
        phiaa = Phi[a, a]
        !iszero(phiaa) && push!(terms, (phiaa / 2) * (xop(a) * xop(a)))
        for bmode in (a + 1):ndof
            phiab = Phi[a, bmode]
            iszero(phiab) && continue
            push!(terms, phiab * (xop(a) * xop(bmode)))
        end
    end

    rep = Representation(
        :coordinate_phonon,
        CoordinateHilbertSpace(ndof),
        CoordinateMomentumAlgebra(ndof),
        CoordinateBasis(collect(1:ndof));
        ordering=collect(1:ndof),
        reference_state=nothing,
    )

    obs = Dict{Symbol,Any}()
    for i in 1:N, alpha in 1:D
        mode = mode_index(i, alpha, D)
        obs[Symbol("displacement_", i, "_", alpha)] = xop(mode)
        obs[Symbol("momentum_", i, "_", alpha)] = pop(mode)
    end

    params = _base_parameters(material, cluster, spec)
    merge!(params, parameters)
    params[:masses] = masses
    params[:force_constants] = Phi
    params[:spatial_dimension] = lattice_dimension(cluster.crystal.lattice)
    params[:displacement_dimension] = D
    params[:phonon_direct_matrix] = Matrix{Float64}(direct_matrix(cluster.crystal.lattice))
    params[:phonon_reciprocal_matrix] = Matrix{Float64}(reciprocal_matrix(cluster.crystal.lattice))
    params[:phonon_units] = spec.units
    params[:phonon_basis_fractional] = [Float64.(site.fractional) for site in cluster.crystal.basis]
    params[:phonon_basis_count] = length(cluster.crystal.basis)
    params[:dof_mode_map] = Dict((i, alpha) => mode_index(i, alpha, D) for i in 1:N for alpha in 1:D)

    return ManyBodyModel(rep, _operator_sum(terms); parameters=params, observables=obs, provenance=[provenance_record])
end

function _periodic_force_constant_separations(crystal::CrystalStructure, ifcs::RealSpaceForceConstants)
    direct = Matrix{Float64}(direct_matrix(crystal.lattice))
    basis_fractional = [Float64.(site.fractional) for site in crystal.basis]
    return [
        direct * (Float64.(block.translation) .+ basis_fractional[block.to_site] .- basis_fractional[block.from_site]) for block in ifcs.blocks
    ]
end

function _periodic_gamma_matrix(
    material::Material,
    cluster::Supercell,
    spec::HarmonicPhononModel,
    ifcs::RealSpaceForceConstants,
    masses::Vector{Float64},
)
    crystal = cluster.crystal
    nbasis = length(crystal.basis)
    D = ifcs.displacement_dimension
    ndof = nbasis * D
    weighted = zeros(ComplexF64, ndof, ndof)

    for block in ifcs.blocks
        1 <= block.from_site <= nbasis || throw(BoundsError(crystal.basis, block.from_site))
        1 <= block.to_site <= nbasis || throw(BoundsError(crystal.basis, block.to_site))
        rows = ((block.from_site - 1) * D + 1):(block.from_site * D)
        cols = ((block.to_site - 1) * D + 1):(block.to_site * D)
        if spec.mass_weighted
            weighted[rows, cols] .+= block.tensor
        else
            @inbounds for alpha in 1:D, beta in 1:D
                ia = mode_index(block.from_site, alpha, D)
                ib = mode_index(block.to_site, beta, D)
                weighted[ia, ib] += block.tensor[alpha, beta] / sqrt(masses[ia] * masses[ib])
            end
        end
    end

    hermitian_error = norm(weighted - adjoint(weighted)) / max(norm(weighted), 1.0)
    hermitian_error <= spec.symmetry_tolerance ||
        throw(ArgumentError("periodic Γ-point dynamical matrix is not Hermitian within tolerance; error=$hermitian_error"))
    weighted = 0.5 .* (weighted .+ adjoint(weighted))
    imag_error = norm(imag.(weighted)) / max(norm(weighted), 1.0)
    imag_error <= spec.symmetry_tolerance ||
        throw(ArgumentError("periodic Γ-point dynamical matrix is not real within tolerance; error=$imag_error"))

    sqrtm = Diagonal(sqrt.(masses))
    return Matrix{Float64}(real.(sqrtm * weighted * sqrtm)), Matrix(weighted)
end

function build_model(material::Material, cluster::Supercell, spec::HarmonicPhononModel; kwargs...)
    _validate_material_cluster(material, cluster)
    lattice_D = lattice_dimension(cluster.crystal.lattice)

    if spec.force_constants isa RealSpaceForceConstants
        ifcs = spec.force_constants
        ifcs.lattice_dimension == lattice_D || throw(DimensionMismatch("force-constant translations must match the lattice dimension"))
        all(==(1), cluster.repetitions) || throw(ArgumentError("periodic real-space force constants require a primitive one-cell Supercell"))
        all(cluster.periodic) || throw(ArgumentError("periodic real-space force constants require periodic boundary conditions in every lattice direction"))
        nbasis = length(cluster.crystal.basis)
        nsites(cluster) == nbasis || throw(DimensionMismatch("primitive periodic phonon model must contain exactly the crystal basis sites"))
        D = isnothing(spec.displacement_dimension) ? ifcs.displacement_dimension : spec.displacement_dimension
        D == ifcs.displacement_dimension || throw(DimensionMismatch("displacement dimension disagrees with periodic force constants"))
        masses = _phonon_masses(material, cluster, D, spec.masses)
        Phi_gamma, D_gamma = _periodic_gamma_matrix(material, cluster, spec, ifcs, masses)
        parameters = Dict{Symbol,Any}(
            :periodic_force_constants => ifcs,
            :phonon_block_separations => _periodic_force_constant_separations(cluster.crystal, ifcs),
            :mass_weighted_force_constants => spec.mass_weighted,
            :gamma_dynamical_matrix => D_gamma,
            :coordinate_hamiltonian_wavevector => :gamma,
            :primitive_periodic_phonon => true,
        )
        provenance = (
            construction=:periodic_harmonic_phonon,
            material=material.name,
            basis_sites=nbasis,
            displacement_dimension=D,
            force_constant_blocks=length(ifcs.blocks),
            mass_weighted=spec.mass_weighted,
        )
        return _coordinate_phonon_model(material, cluster, spec, Phi_gamma, masses, D; parameters=parameters, provenance_record=provenance)
    end

    D = isnothing(spec.displacement_dimension) ? lattice_D : spec.displacement_dimension
    N = nsites(cluster)
    ndof = N * D
    supplied = Matrix(spec.force_constants)
    size(supplied) == (ndof, ndof) || throw(DimensionMismatch("force-constant matrix must be $ndof×$ndof for this cluster"))
    norm(supplied - transpose(supplied)) <= spec.symmetry_tolerance * max(norm(supplied), 1.0) ||
        throw(ArgumentError("force-constant matrix is not symmetric within tolerance"))
    supplied = 0.5 .* (supplied .+ transpose(supplied))
    masses = _phonon_masses(material, cluster, D, spec.masses)

    if spec.mass_weighted
        D0 = Matrix{Float64}(real.(supplied))
        sqrtm = Diagonal(sqrt.(masses))
        Phi = Matrix{Float64}(sqrtm * D0 * sqrtm)
        parameters = Dict{Symbol,Any}(:mass_weighted_dynamical_matrix => D0, :mass_weighted_force_constants => true)
        return _coordinate_phonon_model(material, cluster, spec, Phi, masses, D; parameters=parameters)
    end

    Phi = Matrix{Float64}(real.(supplied))
    return _coordinate_phonon_model(material, cluster, spec, Phi, masses, D; parameters=Dict{Symbol,Any}(:mass_weighted_force_constants => false))
end


# -----------------------------------------------------------------------------
# High-level periodic harmonic-model convenience API
# -----------------------------------------------------------------------------

function _primitive_periodic_cluster(crystal::CrystalStructure)
    D = lattice_dimension(crystal.lattice)
    return Supercell(crystal, ones(Int, D); periodic=true)
end

"""
    build_model(material, spec::HarmonicPhononModel)

Build a primitive-cell periodic harmonic phonon model without requiring the
caller to construct the one-cell periodic `Supercell` explicitly. This
convenience method is available only for `RealSpaceForceConstants`, where the
primitive periodic geometry is unambiguous.
"""
function build_model(material::Material, spec::HarmonicPhononModel; kwargs...)
    spec.force_constants isa RealSpaceForceConstants || throw(ArgumentError(
        "two-argument harmonic build_model requires RealSpaceForceConstants; " *
        "supply an explicit Supercell for finite or Gamma-only force-constant matrices",
    ))
    return build_model(material, _primitive_periodic_cluster(material.crystal), spec; kwargs...)
end

"""
    harmonic_model(material, ifcs::RealSpaceForceConstants; kwargs...)

Construct the standard primitive-cell periodic harmonic phonon model from
real-space force constants. The primitive periodic cell is inferred from
`material.crystal`.
"""
function harmonic_model(
    material::Material,
    ifcs::RealSpaceForceConstants;
    masses=nothing,
    mass_weighted::Bool=false,
    symmetry_tolerance::Real=1e-10,
    units::PhononUnitConvention=PhononUnitConvention(),
)
    spec = HarmonicPhononModel(
        ifcs;
        masses=masses,
        displacement_dimension=ifcs.displacement_dimension,
        mass_weighted=mass_weighted,
        symmetry_tolerance=symmetry_tolerance,
        units=units,
    )
    return build_model(material, spec)
end

"""
    harmonic_model(material, interactions; kwargs...)

Construct a primitive-cell periodic harmonic phonon model directly from
`AbstractHarmonicInteraction` objects. Interaction geometry is compiled to
`RealSpaceForceConstants` with `force_constants`, so reverse and onsite blocks
required by the interaction definitions are generated by the existing harmonic
compiler rather than entered manually.
"""
function harmonic_model(material::Material, interactions; kwargs...)
    terms = interactions isa AbstractHarmonicInteraction ? [interactions] : collect(interactions)
    isempty(terms) && throw(ArgumentError("harmonic_model requires at least one harmonic interaction"))
    all(term -> term isa AbstractHarmonicInteraction, terms) || throw(ArgumentError(
        "harmonic_model interactions must all be AbstractHarmonicInteraction objects",
    ))
    ifcs = force_constants(material.crystal, terms)
    return harmonic_model(material, ifcs; kwargs...)
end

function build_model(material::Material, cluster::Supercell, spec::LocalPhononModel; kwargs...)
    _validate_material_cluster(material, cluster)
    N = nsites(cluster)
    omega = spec.frequencies isa Number ? fill(spec.frequencies, N) : collect(spec.frequencies)
    length(omega) == N || throw(DimensionMismatch("local-phonon frequencies must contain one value per site"))
    all(>(0), omega) || throw(ArgumentError("all phonon frequencies must be positive"))

    cutoffs = spec.cutoff isa Int ? fill(spec.cutoff, N) : copy(spec.cutoff)
    length(cutoffs) == N || throw(DimensionMismatch("phonon cutoffs must contain one value per site"))
    truncation = NumericalTruncation(cutoffs; description="local phonon occupation cutoffs")

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 2 * N)
    for i in 1:N
        energy = spec.hbar * omega[i]
        push!(terms, energy * nb(i))
        spec.include_zero_point && push!(terms, (energy / 2) * IdentityOperator())
    end

    rep = Representation(
        :local_boson,
        BosonFockSpace(N),
        BosonAlgebra(N),
        BosonOccupationBasis(collect(1:N));
        ordering=collect(1:N),
        reference_state=:boson_vacuum,
        truncation=truncation,
    )
    obs = Dict{Symbol,Any}()
    for i in 1:N
        obs[Symbol("boson_number_", i)] = nb(i)
        obs[Symbol("dimensionless_displacement_", i)] = b(i) + b(i)'
    end

    params = _base_parameters(material, cluster, spec)
    params[:frequencies] = omega
    params[:hbar] = spec.hbar
    return ManyBodyModel(rep, _operator_sum(terms); parameters=params, observables=obs, provenance=[(construction=:local_phonon, material=material.name, sites=N)])
end
