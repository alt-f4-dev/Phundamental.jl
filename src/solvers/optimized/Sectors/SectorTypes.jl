"""
    SpinSector(; total_sz=nothing, nup=nothing)

Fixed spin-1/2 magnetization sector. Supply exactly one of `total_sz` or `nup`.
With N spin-1/2 sites, `nup = total_sz + N/2`.
"""
struct SpinSector <: AbstractSector
    total_sz::Union{Nothing,Float64}
    nup::Union{Nothing,Int}
end

function SpinSector(; total_sz=nothing, nup=nothing)
    (isnothing(total_sz) ⊻ isnothing(nup)) ||
        throw(ArgumentError("supply exactly one of total_sz or nup"))
    if !isnothing(nup)
        Int(nup) >= 0 || throw(ArgumentError("nup must be nonnegative"))
        return SpinSector(nothing, Int(nup))
    end
    return SpinSector(Float64(total_sz), nothing)
end

"""
    FermionSector(; nparticles=nothing, nup=nothing, ndown=nothing)

Fixed fermion-number sector. Use either total `nparticles`, or both `nup` and
`ndown` for spinful models carrying `parameters[:fermion_mode_map]`.
"""
struct FermionSector <: AbstractSector
    nparticles::Union{Nothing,Int}
    nup::Union{Nothing,Int}
    ndown::Union{Nothing,Int}
end

function FermionSector(; nparticles=nothing, nup=nothing, ndown=nothing)
    total_mode = !isnothing(nparticles)
    spin_mode = !isnothing(nup) || !isnothing(ndown)
    total_mode ⊻ spin_mode ||
        throw(ArgumentError("use either nparticles or the pair (nup, ndown)"))
    if total_mode
        Int(nparticles) >= 0 || throw(ArgumentError("nparticles must be nonnegative"))
        return FermionSector(Int(nparticles), nothing, nothing)
    end
    (!isnothing(nup) && !isnothing(ndown)) ||
        throw(ArgumentError("both nup and ndown are required"))
    Int(nup) >= 0 && Int(ndown) >= 0 ||
        throw(ArgumentError("nup and ndown must be nonnegative"))
    return FermionSector(nothing, Int(nup), Int(ndown))
end

# ---------------------------------------------------------------------------
# Generic symmetry-sector descriptors
# ---------------------------------------------------------------------------

"""One conserved quantum-number or irrep label used to identify a sector."""
struct QuantumNumber{V}
    name::Symbol
    value::V
end

QuantumNumber(name::AbstractString, value) = QuantumNumber(Symbol(name), value)

"""
    QuantumNumberSector(labels...; metadata=Dict())

Composable sector descriptor for simultaneous conserved quantities or irrep
labels. Numerical backends may specialize on subsets of these labels without
changing the public sector representation.
"""
struct QuantumNumberSector{Q<:Tuple,M} <: AbstractSector
    quantum_numbers::Q
    metadata::M
end

function QuantumNumberSector(labels::Pair...; metadata=Dict{Symbol,Any}())
    names = Symbol[first(label) for label in labels]
    length(unique(names)) == length(names) || throw(ArgumentError("quantum-number labels must be unique"))
    values = tuple((QuantumNumber(Symbol(first(label)), last(label)) for label in labels)...)
    return QuantumNumberSector(values, metadata)
end

function QuantumNumberSector(numbers::QuantumNumber...; metadata=Dict{Symbol,Any}())
    names = getfield.(numbers, :name)
    length(unique(names)) == length(names) || throw(ArgumentError("quantum-number labels must be unique"))
    return QuantumNumberSector(tuple(numbers...), metadata)
end

"""Intersection of several compatible sector descriptors."""
struct CompositeSector{S<:Tuple} <: AbstractSector
    sectors::S
end

CompositeSector(sectors::AbstractSector...) = CompositeSector(tuple(sectors...))

quantum_numbers(sector::QuantumNumberSector) = sector.quantum_numbers
quantum_numbers(sector::CompositeSector) = tuple((q for s in sector.sectors for q in quantum_numbers(s))...)

function quantum_numbers(sector::SpinSector)
    !isnothing(sector.nup) && return (QuantumNumber(:nup, sector.nup),)
    return (QuantumNumber(:total_sz, sector.total_sz),)
end

function quantum_numbers(sector::FermionSector)
    !isnothing(sector.nparticles) && return (QuantumNumber(:nparticles, sector.nparticles),)
    return (QuantumNumber(:nup, sector.nup), QuantumNumber(:ndown, sector.ndown))
end

function sector_value(sector::AbstractSector, name::Symbol)
    for number in quantum_numbers(sector)
        number.name === name && return number.value
    end
    throw(KeyError(name))
end

has_quantum_number(sector::AbstractSector, name::Symbol) = any(number -> number.name === name, quantum_numbers(sector))

function _merged_quantum_number_sector(sector::CompositeSector)
    numbers = QuantumNumber[]
    for component in sector.sectors
        append!(numbers, collect(quantum_numbers(component)))
    end
    names = getfield.(numbers, :name)
    length(unique(names)) == length(names) || throw(ArgumentError("composite sector contains duplicate quantum-number labels"))
    return QuantumNumberSector(numbers...)
end

"""
    specialize_sector(model, sector)

Map a generic quantum-number sector to an optimized backend sector when the
current implementation recognizes the requested conserved quantities. Unknown
labels remain representable but are rejected later by backends that do not yet
implement them.
"""
specialize_sector(::ManyBodyModel, sector::Union{Nothing,SpinSector,FermionSector}) = sector
specialize_sector(model::ManyBodyModel, sector::CompositeSector) = specialize_sector(model, _merged_quantum_number_sector(sector))

function specialize_sector(model::ManyBodyModel, sector::QuantumNumberSector)
    names = Set(number.name for number in sector.quantum_numbers)
    algebra = model.representation.algebra
    if algebra isa SpinAlgebra
        names == Set([:nup]) && return SpinSector(nup=Int(sector_value(sector, :nup)))
        names == Set([:total_sz]) && return SpinSector(total_sz=sector_value(sector, :total_sz))
    elseif algebra isa FermionAlgebra
        names == Set([:nparticles]) && return FermionSector(nparticles=Int(sector_value(sector, :nparticles)))
        names == Set([:nup, :ndown]) && return FermionSector(nup=Int(sector_value(sector, :nup)), ndown=Int(sector_value(sector, :ndown)))
    end
    return sector
end
