"""A symmetry-generated crystallographic orbit in fractional coordinates."""
struct SymmetryOrbit
    representative::Vector{Float64}
    positions::Vector{Vector{Float64}}
    operation_indices::Vector{Vector{Int}}

    function SymmetryOrbit(representative, positions, operation_indices)
        rep = Float64.(collect(representative))
        length(rep) == 3 || throw(DimensionMismatch("symmetry-orbit representatives must contain three fractional coordinates"))
        pos = [Float64.(collect(position)) for position in positions]
        all(length(position) == 3 for position in pos) || throw(DimensionMismatch("symmetry-orbit positions must contain three fractional coordinates"))
        ops = [Int.(collect(indices)) for indices in operation_indices]
        length(pos) == length(ops) || throw(DimensionMismatch("operation provenance must contain one entry per orbit position"))
        all(indices -> !isempty(indices), ops) || throw(ArgumentError("every orbit position must retain at least one generating operation"))
        new(rep, pos, ops)
    end
end

Base.length(orbit::SymmetryOrbit) = length(orbit.positions)
Base.getindex(orbit::SymmetryOrbit, i::Integer) = orbit.positions[Int(i)]
Base.iterate(orbit::SymmetryOrbit, state...) = iterate(orbit.positions, state...)

_normalize_space_group_choice(choice) = isnothing(choice) ? nothing : string(choice)

function _resolve_hall_number(spacegroup_number::Integer; hall_number::Union{Nothing,Integer}=nothing, choice=nothing)
    number = Int(spacegroup_number)
    1 <= number <= 230 || throw(ArgumentError("space-group number must lie between 1 and 230"))
    !isnothing(hall_number) && !isnothing(choice) && throw(ArgumentError("specify either hall_number or choice, not both"))

    if !isnothing(hall_number)
        hall = Int(hall_number)
        1 <= hall <= 530 || throw(ArgumentError("Hall number must lie between 1 and 530"))
        group_type = Spglib.get_spacegroup_type(hall)
        Int(group_type.number) == number || throw(ArgumentError("Hall number $hall belongs to space group $(group_type.number), not $number"))
        return hall
    end

    requested_choice = _normalize_space_group_choice(choice)
    available_choices = String[]
    for hall in 1:530
        group_type = Spglib.get_spacegroup_type(hall)
        Int(group_type.number) == number || continue
        setting_choice = String(group_type.choice)
        isnothing(requested_choice) && return hall
        setting_choice == requested_choice && return hall
        setting_choice in available_choices || push!(available_choices, setting_choice)
    end

    if isnothing(requested_choice)
        throw(ArgumentError("no Hall setting was found for space group $number"))
    end
    throw(ArgumentError("space group $number has no Spglib setting choice $(repr(requested_choice)); available choices are $(join(repr.(available_choices), ", "))"))
end

"""
    space_group_operations(spacegroup_number; hall_number=nothing, choice=nothing)
    space_group_operations(; hall_number)

Return ordinary crystallographic operations from the Spglib database as Phundamental `SpaceGroupOperation` objects. Supplying only an international space-group number selects Spglib's first standard Hall setting. Use `choice` for a human-readable Spglib setting/origin choice such as `choice=2`, or use `hall_number` when the exact Hall setting is already known. `choice` and `hall_number` are mutually exclusive.
"""
function space_group_operations(spacegroup_number::Integer; hall_number::Union{Nothing,Integer}=nothing, choice=nothing)
    hall = _resolve_hall_number(spacegroup_number; hall_number=hall_number, choice=choice)
    rotations, translations = Spglib.get_symmetry_from_database(hall)
    return SpaceGroupOperation[SpaceGroupOperation(rotation, translation) for (rotation, translation) in zip(rotations, translations)]
end

function space_group_operations(; hall_number::Integer)
    hall = Int(hall_number)
    1 <= hall <= 530 || throw(ArgumentError("Hall number must lie between 1 and 530"))
    rotations, translations = Spglib.get_symmetry_from_database(hall)
    return SpaceGroupOperation[SpaceGroupOperation(rotation, translation) for (rotation, translation) in zip(rotations, translations)]
end

"""
    expand_symmetry_orbit(position, operations; atol=1e-8)

Generate the unique periodic fractional positions obtained by applying `operations` to one representative position. Symmetry operations that map onto the same special-position image are retained in `operation_indices`, preserving stabilizer/provenance information instead of silently discarding duplicate generators.
"""
function expand_symmetry_orbit(position::AbstractVector{<:Real}, operations::AbstractVector{<:SpaceGroupOperation}; atol::Real=1e-8)
    length(position) == 3 || throw(DimensionMismatch("symmetry-orbit representatives must contain three fractional coordinates"))
    isempty(operations) && throw(ArgumentError("at least one space-group operation is required"))
    atol > 0 || throw(ArgumentError("atol must be positive"))
    representative = wrap_fractional(position)
    positions = Vector{Vector{Float64}}()
    operation_indices = Vector{Vector{Int}}()

    for (operation_index, operation) in enumerate(operations)
        transformed = wrap_fractional(operation.rotation * representative + operation.translation)
        match = findfirst(existing -> norm(fractional_displacement(transformed, existing)) <= atol, positions)
        if isnothing(match)
            push!(positions, transformed)
            push!(operation_indices, [operation_index])
        else
            push!(operation_indices[match], operation_index)
        end
    end

    return SymmetryOrbit(representative, positions, operation_indices)
end

expand_symmetry_orbit(position::AbstractVector{<:Real}, dataset::CrystallographicDataset; kwargs...) = expand_symmetry_orbit(position, dataset.operations; kwargs...)

function expand_symmetry_orbit(position::AbstractVector{<:Real}, spacegroup_number::Integer; hall_number::Union{Nothing,Integer}=nothing, choice=nothing, atol::Real=1e-8)
    operations = space_group_operations(spacegroup_number; hall_number=hall_number, choice=choice)
    return expand_symmetry_orbit(position, operations; atol=atol)
end
