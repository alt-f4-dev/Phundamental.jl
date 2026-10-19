# Certified single-band periodic embeddings from explicitly declared real-space hopping displacements.

"""One directed Bravais-cell hopping with its Hamiltonian matrix element and conjugate reverse hopping implied."""
struct BravaisHopping{D}
    displacement::NTuple{D,Int}
    amplitude::ComplexF64
end

function BravaisHopping(displacement::NTuple{D,<:Integer}, amplitude::Number) where {D}
    D > 0 || throw(ArgumentError("hopping displacement must not be empty"))
    any(x -> !iszero(x), displacement) || throw(ArgumentError("onsite energy belongs in onsite_energy, not a BravaisHopping"))
    isfinite(real(amplitude)) && isfinite(imag(amplitude)) || throw(ArgumentError("hopping amplitude must be finite"))
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
