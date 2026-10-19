# High-level, algebra-neutral physical interaction declarations and certified coupling coordinates.
# The existing crystallographic generator remains authoritative for projection and independence.

"""One named physical operator sum over explicitly supplied supports. The operator builder receives each support as positional arguments."""
struct OperatorOrbit{S,F,W}
    name::Symbol
    group::Symbol
    supports::S
    operator::F
    weights::W
end

function OperatorOrbit(name::Symbol, supports; operator, group::Symbol=name, weights=nothing)
    entries = Tuple(support isa Tuple ? support : (support isa Integer || support isa Symbol) ? (support,) : Tuple(support) for support in supports)
    isempty(entries) && throw(ArgumentError("operator orbit :$name must contain at least one support"))
    all(entry -> !isempty(entry), entries) || throw(ArgumentError("operator supports must be nonempty"))
    all(support -> all(index -> !(index isa Integer) || index > 0, support), entries) ||
        throw(ArgumentError("integer operator indices must be positive"))
    coefficients = isnothing(weights) ? ntuple(_ -> 1.0, length(entries)) : Tuple(weights)
    length(coefficients) == length(entries) || throw(DimensionMismatch("orbit weights must match the support count"))
    all(weight -> weight isa Number && isfinite(real(weight)) && isfinite(imag(weight)), coefficients) ||
        throw(ArgumentError("orbit weights must be finite numbers"))
    return OperatorOrbit(name, group, entries, operator, coefficients)
end

"""Several independent physical channels that must be selected together as one structural unit."""
struct InteractionGroup{O}
    name::Symbol
    orbits::O
end

function InteractionGroup(name::Symbol, orbits::OperatorOrbit...)
    isempty(orbits) && throw(ArgumentError("interaction group :$name cannot be empty"))
    all(orbit -> orbit.group == name, orbits) || throw(ArgumentError("group and orbit names must agree"))
    return InteractionGroup(name, orbits)
end

"""A finite physical interaction domain, distinct from the crystallographic invariant basis generated from it."""
struct InteractionSpecification
    orbits::Vector{OperatorOrbit}

    function InteractionSpecification(orbits::Vector{OperatorOrbit})
        isempty(orbits) && throw(ArgumentError("interaction specification must contain at least one operator orbit"))
        labels = getfield.(orbits, :name)
        length(unique(labels)) == length(labels) || throw(ArgumentError("physical interaction channel labels must be unique"))
        groups = getfield.(orbits, :group)
        for group in unique(groups)
            positions = findall(==(group), groups)
            maximum(positions) - minimum(positions) + 1 == length(positions) ||
                throw(ArgumentError("channels in interaction group :$group must be contiguous"))
        end
        return new(copy(orbits))
    end
end

function InteractionSpecification(entries::Union{OperatorOrbit,InteractionGroup}...)
    orbits = OperatorOrbit[]
    for entry in entries
        append!(orbits, entry isa OperatorOrbit ? (entry,) : entry.orbits)
    end
    return InteractionSpecification(orbits)
end

InteractionSpecification(entries::AbstractVector{<:Union{OperatorOrbit,InteractionGroup}}) = InteractionSpecification(entries...)

"""Two-spin XXZ exchange channels with an explicit physical bond-sum convention."""
function ExchangeOrbit(name::Symbol, bonds; channels=(:xy, :zz))
    implementations = Dict{Symbol,Function}(
        :xy => ((i, j) -> Sx(i) * Sx(j) + Sy(i) * Sy(j)),
        :zz => ((i, j) -> Sz(i) * Sz(j)),
        :xx => ((i, j) -> Sx(i) * Sx(j)),
        :yy => ((i, j) -> Sy(i) * Sy(j)),
    )
    chosen = Tuple(channels)
    isempty(chosen) && throw(ArgumentError("exchange channels cannot be empty"))
    all(channel -> haskey(implementations, channel), chosen) || throw(ArgumentError("unsupported exchange channel"))
    orbits = [OperatorOrbit(Symbol(name, "_", channel), bonds; group=name, operator=implementations[channel]) for channel in chosen]
    return InteractionGroup(name, orbits...)
end

"""A named local spin field with explicit Hamiltonian sign (default `-h Sᶻ`)."""
function FieldOrbit(name::Symbol, sites; axis::Symbol=:z, sign::Real=-1)
    spin = get(Dict(:x => Sx, :y => Sy, :z => Sz), axis, nothing)
    isnothing(spin) && throw(ArgumentError("field axis must be :x, :y or :z"))
    return OperatorOrbit(name, sites; operator=i -> sign * spin(i))
end

"""Hermitian fermionic or bosonic hopping with physical bond-sum coefficients."""
function HoppingOrbit(name::Symbol, bonds; statistics::Symbol=:fermion)
    lowering = statistics === :fermion ? c : statistics === :boson ? b :
        throw(ArgumentError("hopping statistics must be :fermion or :boson"))
    return OperatorOrbit(name, bonds; operator=(i, j) -> lowering(i)' * lowering(j) + lowering(j)' * lowering(i))
end

"""Density-density interactions in fermionic or bosonic operator algebras."""
function DensityInteractionOrbit(name::Symbol, bonds; statistics::Symbol=:fermion)
    number = statistics === :fermion ? nf : statistics === :boson ? nb :
        throw(ArgumentError("density statistics must be :fermion or :boson"))
    return OperatorOrbit(name, bonds; operator=(i, j) -> number(i) * number(j))
end

"""Hermitian spinless-fermion pairing channel. The ordered support fixes the pairing orientation."""
function PairingOrbit(name::Symbol, bonds)
    return OperatorOrbit(name, bonds; operator=(i, j) -> c(i)' * c(j)' + c(j) * c(i))
end

"""Assemble the user-defined weighted physical interaction sum without symmetry projection or normalization."""
function physical_operator_sum(orbit::OperatorOrbit)
    terms = AbstractOperatorExpr[]
    for (support, weight) in zip(orbit.supports, orbit.weights)
        operator = orbit.operator(support...)
        operator isa AbstractOperatorExpr || throw(ArgumentError("operator builder for :$(orbit.name) did not return an operator expression"))
        push!(terms, weight * operator)
    end
    return reduce(+, terms)
end

"""Certified real coordinate map from user-specified physical operator sums to the generated Hermitian invariant basis."""
struct PhysicalCoefficientMap
    physical_labels::Vector{Symbol}
    basis_labels::Vector{Symbol}
    transformation::Matrix{Float64}
    symmetry_policy::Symbol
    residuals::Vector{Float64}
end

"""Complete high-level interaction-generation result with the authoritative Batch 10 and 11 objects retained."""
struct InteractionGenerationResult{G,F,I,P}
    generation::G
    family::F
    interactions::I
    coefficient_map::P
end

function _physical_coefficient_matrix(basis, physical, algebra; tolerance::Real=1e-10)
    result = PhundamentalCore.structural_span_coordinates(basis, physical, algebra; tolerance=tolerance)
    result === missing && throw(ArgumentError("physical coordinate mapping requires numeric canonical coefficients"))
    for column in eachindex(result.within_span)
        result.within_span[column] || throw(ArgumentError(
            "physical channel $(column) lies outside the generated invariant span (residual=$(result.residuals[column]))"))
    end
    return result.coefficients, result.residuals
end

function _physical_require_hermitian(operator::AbstractOperatorExpr, algebra::AbstractOperatorAlgebra, name::Symbol; tolerance::Real)
    residual = operator - adjoint(operator)
    terms = PhundamentalCore._numeric_linear_terms(algebra, residual)
    isnothing(terms) && throw(ArgumentError("physical channel :$name has nonnumeric coefficients; numerical Hermiticity cannot be certified"))
    maximum((abs(v) for v in values(first(terms))); init=0.0) <= tolerance ||
        throw(ArgumentError("physical channel :$name is not Hermitian; supply the conjugate terms explicitly"))
    return operator
end

"""
    generate_interaction_family(material, algebra, interactions; symmetry_policy=:require, kwargs...)
    generate_interaction_family(material, cluster, algebra, interactions; symmetry_policy=:require, kwargs...)

Expand physical operator sums, use the authoritative Batch 10 crystallographic generator, group full invariant directions into Batch 11 units, and certify the real coefficient map. `:require` rejects symmetry-breaking physical interactions; `:project` explicitly uses their Reynolds projections and records that change of model.
"""
function _generate_interaction_family(source::Tuple, algebra::AbstractOperatorAlgebra, interactions::InteractionSpecification;
                                      symmetry_policy::Symbol=:require, tolerance::Real=1e-9, kwargs...)
    symmetry_policy in (:require, :project) || throw(ArgumentError("symmetry_policy must be :require or :project"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be positive and finite"))
    names = Symbol[orbit.name for orbit in interactions.orbits]
    physical = AbstractOperatorExpr[]
    for orbit in interactions.orbits
        candidate = canonicalize(algebra, physical_operator_sum(orbit))
        _physical_require_hermitian(candidate, algebra, orbit.name; tolerance=tolerance)
        push!(physical, candidate)
    end
    normalized_candidates = AbstractOperatorExpr[(1 / length(orbit.supports)) * expression for (orbit, expression) in zip(interactions.orbits, physical)]
    generation = try
        generate_hamiltonian_family(source..., algebra; candidates=normalized_candidates, basis_labels=names, kwargs...)
    catch error
        if error isa DimensionMismatch && occursin("basis labels", sprint(showerror, error))
            throw(ArgumentError("physical interaction channels are zero or linearly dependent after symmetry projection; merge redundant channels or narrow the declared domain"))
        end
        rethrow()
    end
    count = length(generation.hamiltonian_space)
    count == length(names) || throw(ArgumentError("interaction channels produce $count independent directions from $(length(names)) physical channels; remove redundant/zero channels or split the candidate domain"))
    projected = AbstractOperatorExpr[Crystallography.reynolds_project(operator, generation.specification, generation.crystal_structure, generation.dataset) for operator in physical]
    targets = symmetry_policy === :project ? projected : physical
    if symmetry_policy === :require
        for (name, source_operator, projected_operator) in zip(names, physical, projected)
            residual = PhundamentalCore._numeric_linear_terms(algebra, source_operator - projected_operator)
            isnothing(residual) && throw(ArgumentError("cannot certify symmetry invariance for :$name"))
            maximum((abs(v) for v in values(first(residual))); init=0.0) <= tolerance ||
                throw(ArgumentError("physical channel :$name is not invariant under the declared symmetry; use symmetry_policy=:project only if the projected Hamiltonian is intended"))
        end
    end
    T, residuals = _physical_coefficient_matrix(generation.hamiltonian_space.basis.operators, targets, algebra; tolerance=tolerance)
    group_ids = unique(orbit.group for orbit in interactions.orbits)
    groups = Pair{Symbol,Vector{Symbol}}[id => Symbol[orbit.name for orbit in interactions.orbits if orbit.group == id] for id in group_ids]
    metadata = Dict{Symbol,Any}(:interaction_generation => true, :symmetry_policy => symmetry_policy,
        :physical_channel_labels => Tuple(names), :physical_support_counts => Tuple(length(orbit.supports) for orbit in interactions.orbits))
    family = structural_family(generation; groups=groups, metadata=metadata)
    mapping = PhysicalCoefficientMap(names, copy(generation.hamiltonian_space.basis.labels), T, symmetry_policy, residuals)
    return InteractionGenerationResult(generation, family, interactions, mapping)
end

"""Generate an invariant Hamiltonian family, structural units and a certified real physical-coupling map from a Material."""
function generate_interaction_family(material::Material, algebra::AbstractOperatorAlgebra, interactions::InteractionSpecification; kwargs...)
    return _generate_interaction_family((material,), algebra, interactions; kwargs...)
end

function generate_interaction_family(material::Material, cluster::Supercell, algebra::AbstractOperatorAlgebra, interactions::InteractionSpecification; kwargs...)
    return _generate_interaction_family((material, cluster), algebra, interactions; kwargs...)
end

# Positional overloads compose the existing generator without changing its established keyword contract.
generate_hamiltonian_family(material::Material, algebra::AbstractOperatorAlgebra, interactions::InteractionSpecification; kwargs...) =
    generate_interaction_family(material, algebra, interactions; kwargs...)
generate_hamiltonian_family(material::Material, cluster::Supercell, algebra::AbstractOperatorAlgebra, interactions::InteractionSpecification; kwargs...) =
    generate_interaction_family(material, cluster, algebra, interactions; kwargs...)

"""Named physical coupling values, distinct from coefficients of normalized invariant basis directions."""
struct PhysicalCouplings
    values::Dict{Symbol,Float64}
    unit::Union{Nothing,Symbol}
end

function PhysicalCouplings(values::Union{AbstractDict,NamedTuple}; unit=nothing)
    converted = Dict{Symbol,Float64}()
    for (label, value) in pairs(values)
        value isa Real && isfinite(value) || throw(ArgumentError("physical couplings must be finite real numbers"))
        converted[Symbol(label)] = Float64(value)
    end
    return PhysicalCouplings(converted, isnothing(unit) ? nothing : Symbol(unit))
end

function PhysicalCouplings(; exchange=Dict(), fields=Dict(), values=Dict(), unit=nothing)
    collected = Dict{Symbol,Float64}()
    for (name, pair) in pairs(exchange)
        pair isa Tuple && length(pair) == 2 || throw(ArgumentError("exchange coefficients must be (xy, zz) pairs"))
        collected[Symbol(name, "_xy")] = pair[1]
        collected[Symbol(name, "_zz")] = pair[2]
    end
    for (name, value) in pairs(fields)
        collected[Symbol(name)] = value
    end
    for (name, value) in pairs(values)
        haskey(collected, Symbol(name)) && throw(ArgumentError("duplicate coupling :$name"))
        collected[Symbol(name)] = value
    end
    return PhysicalCouplings(collected; unit=unit)
end

"""Return exact generated-basis coefficients, keyed by basis label and validated against the physical channel schema."""
function physical_parameters(map::PhysicalCoefficientMap, couplings::PhysicalCouplings)
    Set(keys(couplings.values)) == Set(map.physical_labels) ||
        throw(ArgumentError("physical coupling labels must match the interaction specification exactly"))
    J = Float64[couplings.values[name] for name in map.physical_labels]
    theta = map.transformation * J
    all(isfinite, theta) || throw(ArgumentError("mapped invariant coefficients must be finite"))
    return Dict(name => theta[i] for (i, name) in enumerate(map.basis_labels))
end

function physical_parameters(result::InteractionGenerationResult, couplings::PhysicalCouplings)
    if !isnothing(couplings.unit)
        specs = result.generation.parameter_space.specifications
        all(spec -> !isnothing(spec.unit) && spec.unit == couplings.unit, specs) ||
            throw(ArgumentError("physical coupling unit does not match every generated parameter unit; no implicit unit conversion is performed"))
    end
    return physical_parameters(result.coefficient_map, couplings)
end
physical_parameters(result::InteractionGenerationResult, values::Union{AbstractDict,NamedTuple}) = physical_parameters(result, PhysicalCouplings(values))

"""Retrieve the authoritative full-direction structural family created by interaction generation."""
structural_family(result::InteractionGenerationResult) = result.family

"""Describe the interaction generation result without attempting to serialize arbitrary Julia operator callbacks."""
function describe(result::InteractionGenerationResult)
    return Dict{String,Any}(
        "schema_version" => STRUCTURAL_SCHEMA_VERSION,
        "kind" => "interaction_generation_result",
        "generation" => describe(result.generation),
        "family" => describe(result.family),
        "physical_channels" => String.(result.coefficient_map.physical_labels),
        "basis_channels" => String.(result.coefficient_map.basis_labels),
        "coefficient_transformation" => [collect(row) for row in eachrow(result.coefficient_map.transformation)],
        "symmetry_policy" => String(result.coefficient_map.symmetry_policy),
        "mapping_residuals" => copy(result.coefficient_map.residuals),
        "callback_certification" => "operator builders remain live Julia callbacks; their semantics are not serialized",
    )
end
