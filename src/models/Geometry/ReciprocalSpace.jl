# Reciprocal-space conversion utilities.

function reciprocal_vector(lattice::BravaisLattice, hkl)
    D = lattice_dimension(lattice)
    length(hkl) == D || throw(DimensionMismatch("reciprocal coordinates must have $D components"))
    return lattice.reciprocal * collect(hkl)
end

function reciprocal_coordinates(lattice::BravaisLattice, Q)
    D = lattice_dimension(lattice)
    length(Q) == D || throw(DimensionMismatch("Q must have $D Cartesian components"))
    return lattice.reciprocal \ collect(Q)
end

reciprocal_vector(crystal::CrystalStructure, hkl) = reciprocal_vector(crystal.lattice, hkl)
reciprocal_coordinates(crystal::CrystalStructure, Q) = reciprocal_coordinates(crystal.lattice, Q)
