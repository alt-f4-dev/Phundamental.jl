# Certified single-band periodic embeddings from explicitly declared real-space hopping displacements.

"""One directed Bravais-cell hopping with its Hamiltonian matrix element and conjugate reverse hopping implied."""
struct BravaisHopping{D}
    displacement::NTuple{D,Int}
    amplitude::ComplexF64

    function BravaisHopping{D}(displacement::NTuple{D,Int}, amplitude::ComplexF64) where {D}
        D > 0 || throw(ArgumentError("hopping displacement must not be empty"))
        any(!iszero, displacement) || throw(ArgumentError("onsite energy belongs in onsite_energy, not a BravaisHopping"))
        isfinite(real(amplitude)) && isfinite(imag(amplitude)) || throw(ArgumentError("hopping amplitude must be finite"))
        return new{D}(displacement, amplitude)
    end
end

function BravaisHopping(displacement::NTuple{D,<:Integer}, amplitude::Number) where {D}
    return BravaisHopping{D}(Int.(displacement), ComplexF64(amplitude))
end

"""
    bravais_momentum_lattice(material, shape, hoppings; onsite_energy=0, weights=nothing)

Produce the existing `MomentumLattice` and exact periodic momentum-shift maps for a one-orbital Bravais model. Hopping amplitudes are the directed Hamiltonian matrix elements, so both the declared hop and its Hermitian conjugate contribute. `kpoints` are reduced radians, not Cartesian inverse lengths.
"""
function bravais_momentum_lattice(material::Material, shape::NTuple{D,<:Integer}, hoppings;
                                  onsite_energy::Real=0.0, weights=nothing) where {D}
    lattice_dimension(material.crystal.lattice) == D || throw(DimensionMismatch("momentum mesh dimension differs from material lattice"))
    length(material.crystal.basis) == 1 || throw(ArgumentError("scalar MomentumLattice requires exactly one orbital per primitive cell"))
    all(>(0), shape) || throw(ArgumentError("momentum mesh dimensions must be positive"))
    isfinite(onsite_energy) || throw(ArgumentError("onsite energy must be finite"))
    bonds = collect(hoppings)
    all(hop -> hop isa BravaisHopping{D}, bonds) || throw(ArgumentError("hoppings must be BravaisHopping objects of matching dimension"))
    directions = [hop.displacement for hop in bonds]
    length(unique(directions)) == length(directions) || throw(ArgumentError("duplicate directed hopping displacements are ambiguous"))
    reverse = Set(Tuple(-x for x in direction) for direction in directions)
    any(direction -> direction in reverse, directions) && throw(ArgumentError("supply one directed hopping per Hermitian bond pair; reverse hoppings are implicit"))
    dispersion = function(k)
        energy = Float64(onsite_energy)
        for hop in bonds
            phase = sum(k[d] * hop.displacement[d] for d in 1:D)
            energy += 2 * real(hop.amplitude * cis(phase))
        end
        return energy
    end
    return periodic_momentum_lattice(shape, dispersion; weights=weights)
end

# For generated Hamiltonians, require every nearest-shell Bravais direction exactly once.
# Matching only -t without checking the geometric shell could silently change the band structure.
function _certify_generated_bravais_hoppings(model::ManyBodyModel, hoppings::AbstractVector{<:BravaisHopping}; tolerance::Real=1e-8)
    reference = get(model.parameters, :cluster, nothing)
    material = get(model.parameters, :material, nothing)
    reference isa Supercell && material isa Material || throw(ArgumentError("generated Bravais certification requires source material and reference"))
    lattice = Matrix{Float64}(material.crystal.lattice.direct)
    directions = Set{Tuple}()
    canonical(direction) = begin
        first_nonzero = findfirst(!iszero, direction)
        isnothing(first_nonzero) ? direction : direction[first_nonzero] < 0 ? Tuple(-value for value in direction) : direction
    end
    for bond in bonds_in_shell(reference, 1)
        fractional = lattice \ bond.displacement
        integers = Tuple(round(Int, entry) for entry in fractional)
        all(i -> isapprox(fractional[i], integers[i]; atol=tolerance, rtol=tolerance), eachindex(integers)) ||
            throw(ArgumentError("the generated nearest-shell bond is not a Bravais lattice translation"))
        all(iszero, integers) && throw(ArgumentError("a nearest-shell Bravais bond has zero cell displacement"))
        push!(directions, canonical(integers))
    end
    declared = Set(canonical(hop.displacement) for hop in hoppings)
    declared == directions || throw(ArgumentError(
        "declared Bravais hopping directions do not exactly match the generated nearest-shell Hubbard reference"))
    return nothing
end

"""Construct the same geometric lattice after verifying the supported single-band Hubbard source model and explicit hopping convention."""
function bravais_momentum_lattice(model::ManyBodyModel, shape::NTuple{D,<:Integer}, hoppings;
                                  onsite_energy::Real=0.0, weights=nothing) where {D}
    parameters = _single_band_hubbard_forward_parameters(model)
    material = get(model.parameters, :material, nothing)
    material isa Material || throw(ArgumentError("geometric Hubbard embedding requires model material provenance"))
    parameters.specification.t isa Number || throw(ArgumentError("geometric Hubbard embedding requires a scalar hopping coupling"))
    # The Hamiltonian convention is -t(c†_i c_j + h.c.); an explicit Bravais hopping must use that sign.
    hops = collect(hoppings)
    all(hop -> hop isa BravaisHopping && isapprox(hop.amplitude, -parameters.specification.t; atol=1e-12, rtol=1e-12), hops) ||
        throw(ArgumentError("Bravais hopping amplitude disagrees with the Hubbard -t convention"))
    if parameters.certified_from === :structural_generation
        iszero(onsite_energy) || throw(ArgumentError("a generated Hubbard source uses chemical_potential for onsite energies; do not add an unverified Bravais onsite offset"))
        all(hop -> hop isa BravaisHopping{D}, hops) || throw(DimensionMismatch("generated Bravais hopping dimension is inconsistent"))
        _certify_generated_bravais_hoppings(model, BravaisHopping{D}[hops...])
    end
    return bravais_momentum_lattice(material, shape, hops; onsite_energy=onsite_energy, weights=weights)
end

"""Single-site DMFT context obtained from certified geometric hopping data; the electronic approximation is unchanged."""
function DMFTForwardContext(model::ManyBodyModel, shape::NTuple{D,<:Integer}, hoppings; kwargs...) where {D}
    lattice = bravais_momentum_lattice(model, shape, hoppings)
    return DMFTForwardContext(lattice; kwargs...)
end

"""Ladder-DΓA context with exact mesh addition maps and a shared geometrically constructed DMFT reference."""
function LadderDGAForwardContext(model::ManyBodyModel, shape::NTuple{D,<:Integer}, hoppings,
                                 bosonic_axis::BosonicMatsubaraAxis; kwargs...) where {D}
    lattice = bravais_momentum_lattice(model, shape, hoppings)
    return LadderDGAForwardContext(lattice, bosonic_axis; kwargs...)
end


"""One directed inter-cell hopping matrix; the reverse displacement and Hermitian conjugate are implicit."""
struct OrbitalBravaisHopping{D}
    displacement::NTuple{D,Int}
    amplitude::Matrix{ComplexF64}

    function OrbitalBravaisHopping{D}(displacement::NTuple{D,Int}, amplitude::Matrix{ComplexF64}) where {D}
        D > 0 || throw(ArgumentError("orbital hopping requires a nonempty displacement"))
        any(!iszero, displacement) || throw(ArgumentError("on-cell orbital mixing belongs in the onsite matrix"))
        n, m = size(amplitude)
        n > 0 && n == m || throw(DimensionMismatch("orbital hopping amplitudes must be square"))
        all(z -> isfinite(real(z)) && isfinite(imag(z)), amplitude) || throw(ArgumentError("orbital hopping matrix must be finite"))
        return new{D}(displacement, amplitude)
    end
end

function OrbitalBravaisHopping(displacement::NTuple{D,<:Integer}, amplitude::AbstractMatrix{<:Number}) where {D}
    return OrbitalBravaisHopping{D}(Int.(displacement), ComplexF64.(amplitude))
end

"""Construct a matrix Bloch Hamiltonian from explicit directed cell hoppings and optional fractional orbital centers.

An orbital center is a vector of fractional direct-lattice coordinates; omitted centers mean the periodic Bloch gauge with all internal orbitals at the cell origin. No orbital geometry is inferred from the crystal's atomic basis.
"""
function matrix_bravais_momentum_lattice(material::Material, shape::NTuple{D,<:Integer}, hoppings;
                                          onsite::AbstractMatrix{<:Number}, orbital_positions=nothing,
                                          weights=nothing, labels=nothing, tolerance::Real=1e-10) where {D}
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("orbital lattice tolerance must be finite and nonnegative"))
    lattice_dimension(material.crystal.lattice) == D || throw(DimensionMismatch("mesh dimension differs from material lattice dimension"))
    all(>(0), shape) || throw(ArgumentError("momentum mesh dimensions must be positive"))
    n, m = size(onsite)
    n > 0 && n == m || throw(DimensionMismatch("orbital onsite matrix must be square"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), onsite) || throw(ArgumentError("onsite matrix must be finite"))
    H0 = ComplexF64.(onsite)
    isapprox(H0, H0'; atol=tolerance, rtol=tolerance) || throw(ArgumentError("onsite matrix must be Hermitian"))
    bonds = collect(hoppings)
    all(h -> h isa OrbitalBravaisHopping{D} && size(h.amplitude) == (n, n), bonds) ||
        throw(DimensionMismatch("all hopping matrices must have the onsite orbital dimension and lattice dimension"))
    directions = [hop.displacement for hop in bonds]
    length(unique(directions)) == length(directions) || throw(ArgumentError("duplicate orbital hopping displacements"))
    direction_set = Set(directions)
    any(direction -> Tuple(-x for x in direction) in direction_set, directions) &&
        throw(ArgumentError("each Hermitian hopping pair must be supplied in only one direction"))
    centers = if isnothing(orbital_positions)
        [ntuple(_ -> 0.0, D) for _ in 1:n]
    else
        coords = [Tuple(Float64.(collect(position))) for position in orbital_positions]
        length(coords) == n && all(pos -> length(pos) == D && all(isfinite, pos), coords) ||
            throw(DimensionMismatch("fractional orbital center count or dimension disagrees with onsite orbitals"))
        coords
    end
    shifted = periodic_momentum_shift_maps(shape)
    dims = Int.(collect(shape))
    ks = [ntuple(d -> 2π * q[d] / dims[d], D) for q in shifted.qpoints]
    values = Array{ComplexF64}(undef, n, n, length(ks))
    for (ik, k) in enumerate(ks)
        block = Matrix{ComplexF64}(undef, n, n)
        for a in 1:n, b in 1:n
            intracell_phase = sum(k[d] * (centers[b][d] - centers[a][d]) for d in 1:D)
            block[a, b] = H0[a, b] * cis(intracell_phase)
        end
        for hop in bonds, a in 1:n, b in 1:n
            phase = sum(k[d] * (hop.displacement[d] + centers[b][d] - centers[a][d]) for d in 1:D)
            value = hop.amplitude[a, b] * cis(phase)
            block[a, b] += value
            block[b, a] += conj(value)
        end
        values[:, :, ik] .= block
    end
    gauge = all(center -> all(x -> isapprox(x, round(x); atol=tolerance, rtol=0.0), center), centers) ? :periodic : :embedded
    return MatrixMomentumLattice(values; kpoints=ks, weights=weights, qmaps=shifted.maps,
                                 qpoints=ks, labels=labels, gauge=gauge, tolerance=tolerance)
end

# A matrix Bloch dispersion is not an independently validated multiorbital impurity solver.
# Fail before DMFT or DΓA attempts to run scalar fixed-point iterations.
function DMFTForwardContext(lattice::MatrixMomentumLattice; kwargs...)
    throw(ArgumentError("multiorbital matrix embeddings support Dyson propagation and orbital bubbles, not single-orbital ED-DMFT"))
end

function LadderDGAForwardContext(lattice::MatrixMomentumLattice, bosonic_axis::BosonicMatsubaraAxis; kwargs...)
    throw(ArgumentError("matrix orbital bubbles do not provide a certified multiorbital ladder-DΓA vertex solver"))
end


# Certify a generated source against an explicit, image-resolved quadratic Bloch representation.
# This requires a periodic reference large enough to separate all declared hopping images.
function _certify_generated_matrix_hopping(model::ManyBodyModel, onsite::AbstractMatrix{<:Number},
                                            hops::AbstractVector{<:OrbitalBravaisHopping}; tolerance::Real=1e-9)
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("structural certification tolerance must be finite and nonnegative"))
    !isnothing(Models._structural_provenance(model)) ||
        throw(ArgumentError("matrix embedding of a ManyBodyModel requires certified structural-generation provenance"))
    geometry = geometry_context(model)
    isnothing(geometry) && throw(ArgumentError("generated matrix embedding requires a certified geometry context"))
    layout = geometry.component_layout
    !isnothing(layout) && layout.kind === :spinful_fermion ||
        throw(ArgumentError("generated matrix embedding requires a spinful-fermion component layout"))
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("matrix embedding requires direct fermionic operators"))
    reference = get(model.parameters, :cluster, nothing)
    material = get(model.parameters, :material, nothing)
    reference isa Supercell && material isa Material || throw(ArgumentError("generated matrix embedding requires original material and Supercell"))
    all(reference.periodic) || throw(ArgumentError("the certified quadratic Bloch reference must be periodic in every dimension"))
    model.model_space isa HamiltonianSpace || throw(ArgumentError("generated orbital embedding requires a certified HamiltonianSpace"))
    maximum_range = model.model_space.specification.max_range
    !isnothing(maximum_range) && isfinite(maximum_range) && maximum_range > 0 ||
        throw(ArgumentError("generated orbital embedding requires an explicit finite positive structural max_range"))
    translation = reference.crystal.lattice.direct * Diagonal(reference.repetitions)
    2 * maximum_range < minimum(svdvals(translation)) ||
        throw(ArgumentError("reference cell is too small to separate the declared interaction range from periodic images"))
    dimension = length(reference.repetitions)
    all(h -> h isa OrbitalBravaisHopping{dimension}, hops) || throw(DimensionMismatch("hopping and reference dimensions disagree"))
    nbasis = length(material.crystal.basis)
    modes_per_site = layout.components_per_site
    nmodes_cell = nbasis * modes_per_site
    size(onsite) == (nmodes_cell, nmodes_cell) || throw(DimensionMismatch("onsite matrix must cover every site/orbital/spin mode in one primitive cell"))
    layout.nsites == length(reference.sites) && model.representation.algebra.nmodes == length(reference.sites) * modes_per_site ||
        throw(DimensionMismatch("generated fermionic modes disagree with source Supercell"))
    for d in 1:dimension
        span = maximum((abs(hop.displacement[d]) for hop in hops); init=0)
        reference.repetitions[d] > 2 * span || throw(ArgumentError("reference is too small to distinguish periodic-image hopping directions"))
    end
    terms = AbstractOperatorExpr[]
    mode(cell, index) = begin
        basis = fld(index - 1, modes_per_site) + 1
        internal = mod(index - 1, modes_per_site) + 1
        component_index(geometry, site_index(reference, cell, basis), internal)
    end
    ranges = (0:(n - 1) for n in reference.repetitions)
    for cell_tuple in Iterators.product(ranges...)
        cell = collect(cell_tuple)
        for a in 1:nmodes_cell, b in 1:nmodes_cell
            coefficient = onsite[a, b]
            iszero(coefficient) || push!(terms, coefficient * c(mode(cell, a))' * c(mode(cell, b)))
        end
        for hop in hops
            shifted = [mod(cell[d] + hop.displacement[d], reference.repetitions[d]) for d in 1:dimension]
            for a in 1:nmodes_cell, b in 1:nmodes_cell
                coefficient = hop.amplitude[a, b]
                iszero(coefficient) && continue
                forward = c(mode(cell, a))' * c(mode(shifted, b))
                push!(terms, coefficient * forward + conj(coefficient) * forward')
            end
        end
    end
    expected = isempty(terms) ? 0 * IdentityOperator() : OperatorSum(terms)
    algebra = model.representation.algebra
    coefficients = PhundamentalCore._numeric_linear_terms(algebra, Algebra.canonicalize(algebra, model.hamiltonian - expected))
    isnothing(coefficients) && throw(ArgumentError("generated Hamiltonian has nonnumeric or uncertifiable operator coefficients"))
    difference = maximum((abs(value) for value in values(first(coefficients))); init=0.0)
    source = PhundamentalCore._numeric_linear_terms(algebra, Algebra.canonicalize(algebra, model.hamiltonian))
    isnothing(source) && throw(ArgumentError("source Hamiltonian could not be reduced to canonical numeric coefficients"))
    scale = max(1.0, maximum((abs(value) for value in values(first(source))); init=0.0))
    difference <= tolerance * scale || throw(ArgumentError("generated Hamiltonian disagrees with the supplied image-resolved orbital hopping and onsite matrices"))
    return nothing
end

"""Construct a matrix Bloch dispersion only after exact canonical certification against a generated periodic fermion reference."""
function matrix_bravais_momentum_lattice(model::ManyBodyModel, shape::NTuple{D,<:Integer}, hoppings;
                                          onsite::AbstractMatrix{<:Number}, orbital_positions=nothing,
                                          weights=nothing, labels=nothing, tolerance::Real=1e-9) where {D}
    material = get(model.parameters, :material, nothing)
    material isa Material || throw(ArgumentError("generated orbital embedding requires source material provenance"))
    hops = collect(hoppings)
    all(h -> h isa OrbitalBravaisHopping{D}, hops) || throw(DimensionMismatch("matrix hoppings must match momentum-mesh dimension"))
    _certify_generated_matrix_hopping(model, onsite, OrbitalBravaisHopping{D}[hops...]; tolerance=tolerance)
    return matrix_bravais_momentum_lattice(material, shape, hops; onsite=onsite, orbital_positions=orbital_positions,
                                            weights=weights, labels=labels, tolerance=tolerance)
end
