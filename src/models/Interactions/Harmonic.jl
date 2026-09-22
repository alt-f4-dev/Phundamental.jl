# Harmonic short-range interactions compiled into periodic force constants.

abstract type AbstractHarmonicInteraction end

"""
    HarmonicBondInteraction(from_site, to_site, translation;
                            longitudinal, transverse=0)

Central harmonic interaction between primitive-cell basis sites. `translation` is the Bravais-cell offset of `to_site` relative to `from_site`.

The Cartesian bond stiffness is `K = kL * e*e' + kT * (I - e*e')`, where `e` is the equilibrium bond direction. `transverse=0` is the unstressed central-force limit; nonzero transverse stiffness represents prestress or an effective transverse restoring term.
"""
struct HarmonicBondInteraction{T<:Real} <: AbstractHarmonicInteraction
    from_site::Int
    to_site::Int
    translation::Vector{Int}
    longitudinal::T
    transverse::T
end

function HarmonicBondInteraction(from_site::Integer, to_site::Integer, translation; longitudinal::Real, transverse::Real=0.0)
    from_site >= 1 || throw(ArgumentError("from_site must be positive"))
    to_site >= 1 || throw(ArgumentError("to_site must be positive"))
    R = Int.(collect(translation))
    isempty(R) && throw(ArgumentError("bond translation cannot be empty"))
    T = promote_type(typeof(float(longitudinal)), typeof(float(transverse)))
    return HarmonicBondInteraction{T}(Int(from_site), Int(to_site), R, T(longitudinal), T(transverse))
end

"""
    HarmonicVectorCoordinate(sites, translations, coefficients; translation_tolerance=1e-12)

Translation-invariant linear Cartesian coordinate of the form `r = sum(c_a * r_a)`. `sites[a]`, `translations[a]`, and `coefficients[a]` identify one primitive-cell basis site, its Bravais-cell offset, and its scalar coefficient.

The coefficients must sum to zero within `translation_tolerance`, so the coordinate is independent of the arbitrary Cartesian origin. Duplicate `(site, translation)` terms are combined deterministically and exact zero coefficients are removed.

Uniform rescaling of all coefficients rescales the coordinate and is mathematically degenerate with an inverse rescaling of a quadratic stiffness. Coefficient maps should therefore encode coordinate structure, not hide a force-constant refit.
"""
struct HarmonicVectorCoordinate{T<:Real}
    sites::Vector{Int}
    translations::Vector{Vector{Int}}
    coefficients::Vector{T}
end

function HarmonicVectorCoordinate(sites, translations, coefficients; translation_tolerance::Real=1e-12)
    translation_tolerance >= 0 || throw(ArgumentError("translation_tolerance must be nonnegative"))
    site_values = Int.(collect(sites))
    translation_values = [Int.(collect(R)) for R in translations]
    coefficient_values = Float64.(collect(coefficients))
    nterms = length(site_values)
    nterms > 0 || throw(ArgumentError("harmonic vector coordinate requires at least one term"))
    length(translation_values) == nterms || throw(DimensionMismatch("translations must contain one entry per site"))
    length(coefficient_values) == nterms || throw(DimensionMismatch("coefficients must contain one entry per site"))
    all(>=(1), site_values) || throw(ArgumentError("coordinate basis-site indices must be positive"))
    all(isfinite, coefficient_values) || throw(ArgumentError("coordinate coefficients must be finite"))
    dimension = length(first(translation_values))
    dimension > 0 || throw(ArgumentError("coordinate translations cannot be empty"))
    all(R -> length(R) == dimension, translation_values) || throw(DimensionMismatch("all coordinate translations must have equal dimensions"))

    combined = Dict{Tuple{Int,Tuple},Float64}()
    for (site, R, coefficient) in zip(site_values, translation_values, coefficient_values)
        key = (site, Tuple(R))
        combined[key] = get(combined, key, 0.0) + coefficient
    end
    keys_sorted = sort!(collect(keys(combined)); by=key -> (key[1], key[2]))
    compact_sites = Int[]
    compact_translations = Vector{Int}[]
    compact_coefficients = Float64[]
    for key in keys_sorted
        coefficient = combined[key]
        iszero(coefficient) && continue
        push!(compact_sites, key[1])
        push!(compact_translations, collect(key[2]))
        push!(compact_coefficients, coefficient)
    end
    isempty(compact_sites) && throw(ArgumentError("harmonic vector coordinate cannot reduce to zero"))
    coefficient_scale = max(sum(abs, compact_coefficients), 1.0)
    abs(sum(compact_coefficients)) <= translation_tolerance * coefficient_scale ||
        throw(ArgumentError("harmonic vector coordinate coefficients must sum to zero for translation invariance"))
    return HarmonicVectorCoordinate{Float64}(compact_sites, compact_translations, compact_coefficients)
end

"""
    HarmonicBendingInteraction(f, g; stiffness, collinear_tol=1e-12)

General harmonic bending interaction between two translation-invariant linear Cartesian coordinates `f` and `g`. If `r_f` and `r_g` are their equilibrium vectors, the interaction energy is `stiffness * |delta_alpha_parallel|^2 / 2` using the vector-coordinate bending construction implemented by [`bending_force_constant_matrix`](@ref).

This is the general internal-coordinate primitive. [`HarmonicAngleInteraction`](@ref) remains the conventional three-site convenience specialization and compiles through the same implementation.
"""
struct HarmonicBendingInteraction{T<:Real} <: AbstractHarmonicInteraction
    f::HarmonicVectorCoordinate{Float64}
    g::HarmonicVectorCoordinate{Float64}
    stiffness::T
    collinear_tol::Float64
end

function HarmonicBendingInteraction(f::HarmonicVectorCoordinate, g::HarmonicVectorCoordinate; stiffness::Real, collinear_tol::Real=1e-12)
    isfinite(stiffness) || throw(ArgumentError("bending stiffness must be finite"))
    collinear_tol >= 0 || throw(ArgumentError("collinear_tol must be nonnegative"))
    dimension_f = length(first(f.translations))
    dimension_g = length(first(g.translations))
    dimension_f == dimension_g || throw(DimensionMismatch("bending coordinates must have equal translation dimensions"))
    return HarmonicBendingInteraction{typeof(float(stiffness))}(f, g, float(stiffness), Float64(collinear_tol))
end

"""
    HarmonicAngleInteraction(i, j, k, Rji, Rjk; stiffness)

Harmonic bond-bending interaction centered on primitive-cell basis site `j`. `Rji` and `Rjk` are the Bravais-cell offsets of neighbors `i` and `k` relative to the cell containing `j`. The stiffness multiplies `(delta_theta)^2/2`.

This type preserves the existing three-site API. Compilation converts it exactly to two linear vector coordinates, `r_f = r_i(Rji) - r_j(0)` and `r_g = r_k(Rjk) - r_j(0)`, then delegates to [`HarmonicBendingInteraction`](@ref). Collinear triplets use the same two-normal transverse-plane average as the general bending primitive.
"""
struct HarmonicAngleInteraction{T<:Real} <: AbstractHarmonicInteraction
    i::Int
    j::Int
    k::Int
    Rji::Vector{Int}
    Rjk::Vector{Int}
    stiffness::T
end

function HarmonicAngleInteraction(i::Integer, j::Integer, k::Integer, Rji, Rjk; stiffness::Real)
    all(>=(1), (i, j, k)) || throw(ArgumentError("angle basis-site indices must be positive"))
    rji = Int.(collect(Rji))
    rjk = Int.(collect(Rjk))
    length(rji) == length(rjk) || throw(DimensionMismatch("Rji and Rjk must have equal dimensions"))
    isempty(rji) && throw(ArgumentError("angle translations cannot be empty"))
    return HarmonicAngleInteraction{typeof(float(stiffness))}(Int(i), Int(j), Int(k), rji, rjk, float(stiffness))
end

function HarmonicBendingInteraction(interaction::HarmonicAngleInteraction; collinear_tol::Real=1e-12)
    dimension = length(interaction.Rji)
    zero_translation = zeros(Int, dimension)
    f = HarmonicVectorCoordinate((interaction.i, interaction.j), (interaction.Rji, zero_translation), (1.0, -1.0))
    g = HarmonicVectorCoordinate((interaction.k, interaction.j), (interaction.Rjk, zero_translation), (1.0, -1.0))
    return HarmonicBendingInteraction(f, g; stiffness=interaction.stiffness, collinear_tol=collinear_tol)
end

@inline _ifc_key(i::Integer, j::Integer, R) = (Int(i), Int(j), Tuple(Int.(R)))

function _add_ifc_block!(blocks::Dict, i::Integer, j::Integer, R, block::AbstractMatrix{<:Real})
    key = _ifc_key(i, j, R)
    if haskey(blocks, key)
        blocks[key] .+= block
    else
        blocks[key] = Matrix{Float64}(block)
    end
    return blocks
end

function _harmonic_bond_vector(crystal::CrystalStructure, interaction::HarmonicBondInteraction)
    nbasis = length(crystal.basis)
    1 <= interaction.from_site <= nbasis || throw(BoundsError(crystal.basis, interaction.from_site))
    1 <= interaction.to_site <= nbasis || throw(BoundsError(crystal.basis, interaction.to_site))
    dimension = lattice_dimension(crystal.lattice)
    length(interaction.translation) == dimension || throw(DimensionMismatch("bond translation dimension does not match crystal"))
    fi = crystal.basis[interaction.from_site].fractional
    fj = crystal.basis[interaction.to_site].fractional
    return direct_matrix(crystal.lattice) * (Float64.(interaction.translation) .+ Float64.(fj) .- Float64.(fi))
end

function _compile_harmonic_bond!(blocks::Dict, crystal::CrystalStructure, interaction::HarmonicBondInteraction)
    r = _harmonic_bond_vector(crystal, interaction)
    distance = norm(r)
    distance > sqrt(eps(Float64)) || throw(ArgumentError("harmonic bond has zero equilibrium length"))
    e = r / distance
    dimension = length(r)
    projector = e * transpose(e)
    K = interaction.longitudinal .* projector .+ interaction.transverse .* (Matrix{Float64}(I, dimension, dimension) .- projector)
    zero_translation = zeros(Int, length(interaction.translation))
    R = interaction.translation
    _add_ifc_block!(blocks, interaction.from_site, interaction.to_site, R, -K)
    _add_ifc_block!(blocks, interaction.to_site, interaction.from_site, -R, -transpose(K))
    _add_ifc_block!(blocks, interaction.from_site, interaction.from_site, zero_translation, K)
    _add_ifc_block!(blocks, interaction.to_site, interaction.to_site, zero_translation, K)
    return blocks
end

"""
    harmonic_coordinate_vector(crystal, coordinate)

Evaluate the equilibrium Cartesian vector represented by a [`HarmonicVectorCoordinate`](@ref) in `crystal`. Because the coordinate coefficients are translation invariant, the result is independent of the arbitrary Cartesian origin.
"""
function harmonic_coordinate_vector(crystal::CrystalStructure, coordinate::HarmonicVectorCoordinate)
    nbasis = length(crystal.basis)
    dimension = lattice_dimension(crystal.lattice)
    dimension == length(first(coordinate.translations)) || throw(DimensionMismatch("coordinate translation dimension does not match crystal"))
    direct = direct_matrix(crystal.lattice)
    result = zeros(Float64, dimension)
    for (site, R, coefficient) in zip(coordinate.sites, coordinate.translations, coordinate.coefficients)
        1 <= site <= nbasis || throw(BoundsError(crystal.basis, site))
        fractional = Float64.(crystal.basis[site].fractional) .+ Float64.(R)
        result .+= coefficient .* (direct * fractional)
    end
    return result
end

"""
    bending_force_constant_matrix(rf, rg; stiffness, collinear_tol=1e-12)

Return the `6x6` force-constant matrix for the two displacement coordinates `u_f` and `u_g` of a harmonic bending interaction, ordered as `[u_f; u_g]`. The implementation follows the vector-coordinate construction `delta_alpha = u_f x r_f/|r_f|^2 - u_g x r_g/|r_g|^2`.

For a noncollinear angle, the bend normal is `n = (r_f x r_g)/|r_f x r_g|`. For a collinear angle, the force-constant matrix is the average over two orthogonal normals spanning the plane perpendicular to the bond axis, equivalent to the continuous transverse-normal average.
"""
function bending_force_constant_matrix(rf, rg; stiffness::Real, collinear_tol::Real=1e-12)
    collinear_tol >= 0 || throw(ArgumentError("collinear_tol must be nonnegative"))
    f = Float64.(collect(rf))
    g = Float64.(collect(rg))
    length(f) == 3 || throw(DimensionMismatch("rf must contain three Cartesian components"))
    length(g) == 3 || throw(DimensionMismatch("rg must contain three Cartesian components"))
    rf_norm = norm(f)
    rg_norm = norm(g)
    rf_norm > sqrt(eps(Float64)) || throw(ArgumentError("rf must be nonzero"))
    rg_norm > sqrt(eps(Float64)) || throw(ArgumentError("rg must be nonzero"))
    beta = Float64(stiffness)
    cross_fg = cross(f, g)
    cross_scale = rf_norm * rg_norm

    function matrix_for_normal(n)
        wf = cross(f / rf_norm^2, n)
        wg = cross(g / rg_norm^2, n)
        return [wf * transpose(wf) -wf * transpose(wg); -wg * transpose(wf) wg * transpose(wg)]
    end

    if norm(cross_fg) > collinear_tol * cross_scale
        normal = cross_fg / norm(cross_fg)
        return beta .* matrix_for_normal(normal)
    end

    axis = f / rf_norm
    trial_axis = argmin(abs.(axis))
    reference = zeros(Float64, 3)
    reference[trial_axis] = 1.0
    n1 = normalize(cross(axis, reference))
    n2 = normalize(cross(axis, n1))
    return (beta / 2) .* (matrix_for_normal(n1) .+ matrix_for_normal(n2))
end

function _bending_participants(interaction::HarmonicBendingInteraction)
    participant_map = Dict{Tuple{Int,Tuple},Vector{Float64}}()
    for (coordinate_index, coordinate) in enumerate((interaction.f, interaction.g))
        for (site, R, coefficient) in zip(coordinate.sites, coordinate.translations, coordinate.coefficients)
            key = (site, Tuple(R))
            coefficients = get!(participant_map, key) do
                zeros(Float64, 2)
            end
            coefficients[coordinate_index] += coefficient
        end
    end
    keys_sorted = sort!(collect(keys(participant_map)); by=key -> (key[1], key[2]))
    return [(site=key[1], translation=collect(key[2]), f=participant_map[key][1], g=participant_map[key][2]) for key in keys_sorted]
end

function _compile_harmonic_bending!(blocks::Dict, crystal::CrystalStructure, interaction::HarmonicBendingInteraction)
    beta = Float64(interaction.stiffness)
    iszero(beta) && return blocks
    dimension = lattice_dimension(crystal.lattice)
    dimension == 3 || throw(ArgumentError("HarmonicBendingInteraction currently requires a three-dimensional crystal"))
    rf = harmonic_coordinate_vector(crystal, interaction.f)
    rg = harmonic_coordinate_vector(crystal, interaction.g)
    K = bending_force_constant_matrix(rf, rg; stiffness=beta, collinear_tol=interaction.collinear_tol)
    Kff = view(K, 1:3, 1:3)
    Kfg = view(K, 1:3, 4:6)
    Kgf = view(K, 4:6, 1:3)
    Kgg = view(K, 4:6, 4:6)
    participants = _bending_participants(interaction)
    block = zeros(Float64, 3, 3)

    for first in participants, second in participants
        @inbounds for row in 1:3, column in 1:3
            block[row, column] = first.f * second.f * Kff[row, column] + first.f * second.g * Kfg[row, column] +
                                 first.g * second.f * Kgf[row, column] + first.g * second.g * Kgg[row, column]
        end
        translation = second.translation .- first.translation
        _add_ifc_block!(blocks, first.site, second.site, translation, block)
    end
    return blocks
end

function _compile_harmonic_angle!(blocks::Dict, crystal::CrystalStructure, interaction::HarmonicAngleInteraction)
    return _compile_harmonic_bending!(blocks, crystal, HarmonicBendingInteraction(interaction))
end

"""
    force_constants(crystal, interactions)

Compile harmonic bond, conventional angle, and generalized bending interactions into `RealSpaceForceConstants`. Returned blocks are ordinary unweighted physical IFCs; mass weighting is applied later by `HarmonicPhononModel`/`HarmonicPhononSolver`.
"""
function force_constants(crystal::CrystalStructure, interactions)
    terms = collect(interactions)
    isempty(terms) && throw(ArgumentError("at least one harmonic interaction is required"))
    all(term -> term isa AbstractHarmonicInteraction, terms) || throw(ArgumentError("all entries must be AbstractHarmonicInteraction objects"))

    blocks = Dict{Tuple{Int,Int,Tuple},Matrix{Float64}}()
    for interaction in terms
        if interaction isa HarmonicBondInteraction
            _compile_harmonic_bond!(blocks, crystal, interaction)
        elseif interaction isa HarmonicAngleInteraction
            _compile_harmonic_angle!(blocks, crystal, interaction)
        elseif interaction isa HarmonicBendingInteraction
            _compile_harmonic_bending!(blocks, crystal, interaction)
        else
            throw(ArgumentError("unsupported harmonic interaction $(typeof(interaction))"))
        end
    end

    entries = ForceConstantBlock[]
    for key in sort!(collect(keys(blocks)); by=value -> (value[1], value[2], value[3]))
        i, j, translation_tuple = key
        push!(entries, ForceConstantBlock(i, j, collect(translation_tuple), blocks[key]))
    end
    return RealSpaceForceConstants(entries; displacement_dimension=lattice_dimension(crystal.lattice))
end

force_constants(crystal::CrystalStructure, interaction::AbstractHarmonicInteraction) = force_constants(crystal, (interaction,))
