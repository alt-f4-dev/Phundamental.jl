import Spglib

"""Return a `Spglib.SpglibCell` for a backend-neutral `CrystalStructure`."""
function spglib_cell(structure::CrystalStructure)
    positions = [collect(structure.positions[:, i]) for i in 1:length(structure)]
    return Spglib.SpglibCell(structure.lattice, positions, structure.species)
end

"""Discover ordinary crystallographic symmetry using Spglib.jl and copy the result into Phundamental-owned types."""
function discover_symmetry(structure::CrystalStructure; options::CrystallographyOptions=CrystallographyOptions(), hall_number::Union{Nothing,Integer}=nothing)
    cell = spglib_cell(structure)
    dataset = isnothing(hall_number) ? Spglib.get_dataset(cell, options.symprec) : Spglib.get_dataset_with_hall_number(cell, Int(hall_number), options.symprec)
    operations = SpaceGroupOperation[]
    sizehint!(operations, length(dataset.rotations))
    for (rotation, translation) in zip(dataset.rotations, dataset.translations)
        push!(operations, SpaceGroupOperation(rotation, translation))
    end
    metadata = Dict{Symbol,Any}(:backend => :spglib, :choice => string(dataset.choice), :symprec => options.symprec)
    return CrystallographicDataset(spacegroup_number=dataset.spacegroup_number, hall_number=dataset.hall_number, international_symbol=dataset.international_symbol, hall_symbol=dataset.hall_symbol, pointgroup_symbol=dataset.pointgroup_symbol, operations=operations, wyckoffs=string.(dataset.wyckoffs), site_symmetry_symbols=string.(dataset.site_symmetry_symbols), equivalent_atoms=Int.(dataset.equivalent_atoms), crystallographic_orbits=Int.(dataset.crystallographic_orbits), metadata=metadata)
end
