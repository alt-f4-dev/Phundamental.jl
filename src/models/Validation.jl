# Lightweight geometry/material/model validation helpers.

function validate_lattice(lattice::BravaisLattice; atol::Real=1e-10, rtol::Real=1e-10)
    D = lattice_dimension(lattice)
    target = 2pi .* Matrix{eltype(lattice.direct)}(I, D, D)
    residual = norm(transpose(lattice.direct) * lattice.reciprocal - target)
    scale = max(norm(target), 1.0)
    return (
        passed = residual <= atol + rtol * scale,
        reciprocal_residual = residual,
        dimension = D,
        measure = lattice.measure,
    )
end

function validate_bonds(cluster::Supercell, bonds::AbstractVector{<:Bond}; atol::Real=1e-10, rtol::Real=1e-10)
    N = nsites(cluster)
    max_distance_error = 0.0
    valid_indices = true
    valid_shells = true
    for bond in bonds
        valid_indices &= 1 <= bond.i <= N && 1 <= bond.j <= N
        valid_shells &= bond.shell >= 1
        err = abs(norm(bond.displacement) - bond.distance)
        max_distance_error = max(max_distance_error, err)
    end
    scale = isempty(bonds) ? 1.0 : max(maximum(b.distance for b in bonds), 1.0)
    passed = valid_indices && valid_shells && max_distance_error <= atol + rtol * scale
    return (
        passed = passed,
        valid_indices = valid_indices,
        valid_shells = valid_shells,
        max_distance_error = max_distance_error,
        nbonds = length(bonds),
        coordination = coordination_numbers(cluster, bonds),
    )
end

function validate_material_model(model::ManyBodyModel; atol::Real=1e-10, rtol::Real=1e-10)
    material = get(model.parameters, :material, nothing)
    cluster = get(model.parameters, :cluster, nothing)
    positions = get(model.parameters, :positions, nothing)
    material isa Material || return (passed=false, reason=:missing_material)
    cluster isa Supercell || return (passed=false, reason=:missing_cluster)

    _validate_material_cluster(material, cluster)
    position_ok = !isnothing(positions) && length(positions) == nsites(cluster)
    lattice_check = validate_lattice(material.crystal.lattice; atol=atol, rtol=rtol)
    bond_check = if haskey(model.parameters, :bonds)
        validate_bonds(cluster, model.parameters[:bonds]; atol=atol, rtol=rtol)
    else
        nothing
    end
    passed = position_ok && lattice_check.passed && (isnothing(bond_check) || bond_check.passed)
    return (
        passed = passed,
        positions = position_ok,
        lattice = lattice_check,
        bonds = bond_check,
        representation = model.representation.name,
    )
end


struct ForceConstantValidationResult
    passed::Bool
    hermiticity_residual::Float64
    permutation_residual::Float64
    translational_residual::Float64
    rotational_residual::Union{Nothing,Float64}
    metadata::Dict{Symbol,Any}
end

function _force_constant_separations(crystal::CrystalStructure, ifcs::RealSpaceForceConstants)
    direct = Matrix{Float64}(direct_matrix(crystal.lattice))
    basis = [Float64.(site.fractional) for site in crystal.basis]
    return [
        direct * (Float64.(block.translation) .+ basis[block.to_site] .- basis[block.from_site])
        for block in ifcs.blocks
    ]
end

function _physical_force_constant_blocks(model::ManyBodyModel)
    ifcs = get(model.parameters, :periodic_force_constants, nothing)
    ifcs isa RealSpaceForceConstants ||
        throw(ArgumentError("force-constant validation requires periodic RealSpaceForceConstants"))
    !get(model.parameters, :mass_weighted_force_constants, false) && return ifcs

    D = ifcs.displacement_dimension
    masses = Float64.(model.parameters[:masses])
    entries = ForceConstantBlock[]
    for block in ifcs.blocks
        tensor = Matrix{ComplexF64}(undef, D, D)
        @inbounds for α in 1:D, β in 1:D
            ia = mode_index(block.from_site, α, D)
            ib = mode_index(block.to_site, β, D)
            tensor[α, β] = block.tensor[α, β] * sqrt(masses[ia] * masses[ib])
        end
        push!(entries, ForceConstantBlock(block.from_site, block.to_site, block.translation, tensor))
    end
    return RealSpaceForceConstants(entries; displacement_dimension=D)
end

"""
    validate_force_constants(crystal, ifcs; atol=1e-10, rtol=1e-10,
                             rotational=true)

Validate real-space IFC reciprocity/Hermiticity, the translational acoustic sum
rule, and (optionally) the Born-Huang rotational first-moment condition.

Residuals are normalized and dimensionless. The rotational condition is
evaluated from the actual equilibrium separation vectors and therefore requires
the crystal geometry.
"""
function validate_force_constants(
    crystal::CrystalStructure,
    ifcs::RealSpaceForceConstants;
    atol::Real=1e-10,
    rtol::Real=1e-10,
    rotational::Bool=true,
)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    ifcs.lattice_dimension == lattice_dimension(crystal.lattice) ||
        throw(DimensionMismatch("IFC lattice dimension does not match crystal"))
    D = ifcs.displacement_dimension
    spatial_D = lattice_dimension(crystal.lattice)
    if rotational && D != spatial_D
        throw(ArgumentError(
            "rotational IFC validation requires displacement_dimension == lattice dimension; " *
            "set rotational=false when validating reduced-dimensional or embedded models",
        ))
    end
    nbasis = length(crystal.basis)
    blocks = _aggregate_force_constant_blocks(ifcs)

    block_scale = max(
        maximum((norm(block) for block in values(blocks)); init=0.0),
        1.0,
    )
    pair_error = 0.0
    onsite_error = 0.0
    Z = Tuple(zeros(Int, ifcs.lattice_dimension))
    for ((i, j, R), block) in blocks
        1 <= i <= nbasis || throw(BoundsError(crystal.basis, i))
        1 <= j <= nbasis || throw(BoundsError(crystal.basis, j))
        reverse_key = (j, i, Tuple(-collect(R)))
        reverse = get(blocks, reverse_key, zeros(ComplexF64, D, D))
        pair_error = max(pair_error, norm(block - adjoint(reverse)))
        if i == j && R == Z
            onsite_error = max(onsite_error, norm(block - adjoint(block)))
        end
    end
    permutation_residual = pair_error / block_scale
    hermiticity_residual = max(pair_error, onsite_error) / block_scale

    row_sums = [zeros(ComplexF64, D, D) for _ in 1:nbasis]
    row_scales = zeros(Float64, nbasis)
    for ((i, _, _), block) in blocks
        row_sums[i] .+= block
        row_scales[i] += norm(block)
    end
    translational_residual = maximum(
        norm(row_sums[i]) / max(row_scales[i], 1.0) for i in 1:nbasis
    )

    rotational_residual = if rotational
        direct = Matrix{Float64}(direct_matrix(crystal.lattice))
        basis = [Float64.(site.fractional) for site in crystal.basis]
        max_error = 0.0
        max_scale = 0.0
        for i in 1:nbasis, α in 1:D, β in 1:D, γ in 1:D
            value = 0.0 + 0.0im
            local_scale = 0.0
            for ((ii, j, R), block) in blocks
                ii == i || continue
                r = direct * (Float64.(collect(R)) .+ basis[j] .- basis[i])
                term1 = block[α, β] * r[γ]
                term2 = block[α, γ] * r[β]
                value += term1 - term2
                local_scale += abs(term1) + abs(term2)
            end
            max_error = max(max_error, abs(value))
            max_scale = max(max_scale, local_scale)
        end
        max_error / max(max_scale, 1.0)
    else
        nothing
    end

    threshold = Float64(atol + rtol)
    passed = hermiticity_residual <= threshold &&
             permutation_residual <= threshold &&
             translational_residual <= threshold &&
             (isnothing(rotational_residual) || rotational_residual <= threshold)

    return ForceConstantValidationResult(
        passed,
        hermiticity_residual,
        permutation_residual,
        translational_residual,
        rotational_residual,
        Dict{Symbol,Any}(
            :atol => Float64(atol),
            :rtol => Float64(rtol),
            :block_count => length(ifcs.blocks),
            :basis_count => nbasis,
            :displacement_dimension => D,
            :rotational_checked => rotational,
        ),
    )
end

function validate_force_constants(
    model::ManyBodyModel;
    atol::Real=1e-10,
    rtol::Real=1e-10,
    rotational::Bool=true,
)
    material = get(model.parameters, :material, nothing)
    isnothing(material) && throw(ArgumentError("model does not retain Material metadata"))
    ifcs = _physical_force_constant_blocks(model)
    result = validate_force_constants(
        material.crystal,
        ifcs;
        atol=atol,
        rtol=rtol,
        rotational=rotational,
    )
    result.metadata[:source_mass_weighted] = get(model.parameters, :mass_weighted_force_constants, false)
    return result
end
