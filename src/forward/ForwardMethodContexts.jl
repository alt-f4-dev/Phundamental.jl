# Explicit method-specific context configures an already selected high-level computational route without determining physical/model-space admissibility or ranking candidate methods.

using LinearAlgebra: svdvals, Diagonal

abstract type AbstractForwardMethodContext end

"""
    DMFTForwardContext(lattice; temperature, impurity_energy=0, kB=1, nfrequencies=128, axis=nothing)

High-level configuration required to lower a single-band Hubbard `ManyBodyModel` into a `DMFTProblem`. The local interaction and chemical potential remain authoritative on the model's `HubbardModel` specification; `lattice` supplies the noninteracting embedding used by single-site DMFT.
"""
struct DMFTForwardContext{L<:AbstractLatticeEmbedding,A<:FermionicMatsubaraAxis} <: AbstractForwardMethodContext
    lattice::L
    temperature::Float64
    impurity_energy::Float64
    kB::Float64
    axis::A
end

function DMFTForwardContext(lattice::AbstractLatticeEmbedding; temperature::Real, impurity_energy::Real=0.0, kB::Real=1.0,
                            nfrequencies::Integer=128, axis=nothing)
    temperature > 0 || throw(ArgumentError("DMFTForwardContext requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    nfrequencies > 0 || throw(ArgumentError("nfrequencies must be positive"))
    beta = inv(Float64(kB) * Float64(temperature))
    frequency_axis = isnothing(axis) ? FermionicMatsubaraAxis(beta, nfrequencies) : axis
    frequency_axis isa FermionicMatsubaraAxis || throw(ArgumentError("axis must be a FermionicMatsubaraAxis"))
    isapprox(frequency_axis.beta, beta; rtol=1e-12, atol=1e-12 * max(abs(beta), 1.0)) || throw(ArgumentError("provided fermionic Matsubara axis does not match temperature and kB"))
    return DMFTForwardContext(lattice, Float64(temperature), Float64(impurity_energy), Float64(kB), frequency_axis)
end

"""
    LadderDGAForwardContext(lattice, bosonic_axis; dmft_solver=DMFTSolver(), impurity_backend=TauGridFourPoint(), impurity_two_particle=nothing, impurity_energy=0, kB=1, nfrequencies=128, fermionic_axis=nothing, dmft_solver_kwargs=(;))

High-level context for the end-to-end ladder-DΓA route. The same `MomentumLattice` is used for the local DMFT reference and the nonlocal ladder construction. The context makes the prerequisite DMFT solver and Matsubara frequency boxes explicit; no analytic continuation or self-energy feedback is implied beyond the selected `LadderDGA` method.
"""
struct LadderDGAForwardContext{L<:MomentumLattice,D<:DMFTSolver,A<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis,I,T,K<:NamedTuple} <: AbstractForwardMethodContext
    lattice::L
    dmft_solver::D
    fermionic_axis::A
    bosonic_axis::B
    impurity_backend::I
    impurity_two_particle::T
    temperature::Float64
    impurity_energy::Float64
    kB::Float64
    dmft_solver_kwargs::K
end

function LadderDGAForwardContext(lattice::MomentumLattice, bosonic_axis::BosonicMatsubaraAxis; dmft_solver::DMFTSolver=DMFTSolver(),
                                 impurity_backend=TauGridFourPoint(), impurity_two_particle=nothing, impurity_energy::Real=0.0,
                                 kB::Real=1.0, nfrequencies::Integer=128, fermionic_axis=nothing, dmft_solver_kwargs::NamedTuple=(;))
    !isnothing(lattice.qmaps) || throw(ArgumentError("LadderDGAForwardContext requires a MomentumLattice with explicit q maps"))
    impurity_backend isa AbstractImpurityTwoParticleBackend || throw(ArgumentError("impurity_backend must be an AbstractImpurityTwoParticleBackend"))
    !isnothing(impurity_two_particle) && !(impurity_two_particle isa ImpurityTwoParticleResult) && throw(ArgumentError("impurity_two_particle must be ImpurityTwoParticleResult or nothing"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    nfrequencies > 0 || throw(ArgumentError("nfrequencies must be positive"))
    beta = bosonic_axis.beta
    frequency_axis = isnothing(fermionic_axis) ? FermionicMatsubaraAxis(beta, nfrequencies) : fermionic_axis
    frequency_axis isa FermionicMatsubaraAxis || throw(ArgumentError("fermionic_axis must be a FermionicMatsubaraAxis"))
    isapprox(frequency_axis.beta, beta; rtol=1e-12, atol=1e-12 * max(abs(beta), 1.0)) || throw(ArgumentError("fermionic and bosonic Matsubara axes must carry the same inverse temperature"))
    temperature = inv(Float64(kB) * beta)
    return LadderDGAForwardContext(lattice, dmft_solver, frequency_axis, bosonic_axis, impurity_backend, impurity_two_particle,
                                   temperature, Float64(impurity_energy), Float64(kB), dmft_solver_kwargs)
end

"""
    NLCEForwardContext(max_order; inner_solver=ExactDiagonalization(), inner_transformation=nothing, inner_method_context=nothing, inner_solver_args=(), inner_solver_kwargs=(;), bonds=nothing, min_order=1, max_clusters=100_000, model_transform=nothing, model_builder=nothing, allow_periodic_reference=false)

High-level configuration for finite-reference NLCE. Each generated cluster is evaluated through the ordinary `ForwardProblem` machinery using the same outer observable request and the explicitly selected `inner_solver`. The outer `NLCESolver` performs only linked-cluster inclusion-exclusion and normalization.
"""
struct NLCEForwardContext{S<:AbstractSolver,T,C,A<:Tuple,K<:NamedTuple,B,MT,MB} <: AbstractForwardMethodContext
    max_order::Int
    inner_solver::S
    inner_transformation::T
    inner_method_context::C
    inner_solver_args::A
    inner_solver_kwargs::K
    bonds::B
    min_order::Int
    max_clusters::Int
    model_transform::MT
    model_builder::MB
    allow_periodic_reference::Bool
end

function NLCEForwardContext(max_order::Integer; inner_solver::AbstractSolver=ExactDiagonalization(), inner_transformation=nothing,
                            inner_method_context=nothing, inner_solver_args::Tuple=(), inner_solver_kwargs::NamedTuple=(;), bonds=nothing,
                            min_order::Integer=1, max_clusters::Integer=100_000, model_transform=nothing, model_builder=nothing,
                            allow_periodic_reference::Bool=false)
    1 <= min_order <= max_order || throw(ArgumentError("require 1 <= min_order <= max_order"))
    max_clusters > 0 || throw(ArgumentError("max_clusters must be positive"))
    inner_solver isa NLCESolver && throw(ArgumentError("nested NLCESolver execution is not supported by NLCEForwardContext"))
    !isnothing(inner_transformation) && !(inner_transformation isa AbstractTransformation) && throw(ArgumentError("inner_transformation must be AbstractTransformation or nothing"))
    !isnothing(inner_method_context) && !(inner_method_context isa AbstractForwardMethodContext) && throw(ArgumentError("inner_method_context must be AbstractForwardMethodContext or nothing"))
    return NLCEForwardContext(Int(max_order), inner_solver, inner_transformation, inner_method_context, inner_solver_args, inner_solver_kwargs,
                              bonds, Int(min_order), Int(max_clusters), model_transform, model_builder, allow_periodic_reference)
end

# A generated family can reach single-site embedding only when its entire canonical Hamiltonian
# lies in the three-dimensional nearest-shell Hubbard subspace of the same physical reference.
function _generated_single_band_hubbard_parameters(model::ManyBodyModel; tolerance::Real=1e-9)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("generated Hubbard certification requires a finite positive tolerance"))
    isnothing(Models._structural_provenance(model)) && throw(ArgumentError(
        "automatic Hubbard lowering requires certified structural-selection provenance"))
    material = get(model.parameters, :material, nothing)
    reference = get(model.parameters, :cluster, nothing)
    material isa Material && reference isa Supercell || throw(ArgumentError(
        "generated Hubbard lowering requires the original material and finite reference Supercell"))
    length(material.crystal.basis) == 1 || throw(ArgumentError("scalar Hubbard embedding requires one basis site per primitive cell"))
    maximum_range = model.model_space.specification.max_range
    isnothing(maximum_range) && throw(ArgumentError(
        "generated Hubbard embedding requires an explicit max_range certifying the intended physical interaction domain"))
    translation = reference.crystal.lattice.direct * Diagonal(reference.repetitions)
    2 * maximum_range < minimum(svdvals(translation)) || throw(ArgumentError(
        "the periodic reference is too small to distinguish declared-range structural interactions from their periodic images"))
    context = geometry_context(model)
    isnothing(context) && throw(ArgumentError("generated Hubbard lowering requires certified component geometry"))
    layout = context.component_layout
    !isnothing(layout) && layout.kind === :spinful_fermion && layout.norbitals == 1 || throw(ArgumentError(
        "only generated single-orbital spinful-fermion layouts admit scalar Hubbard embedding"))
    layout.nsites == nsites(reference) && model.representation.algebra.nmodes == 2 * nsites(reference) ||
        throw(DimensionMismatch("certified fermion layout does not match the reference Supercell"))
    model.hamiltonian isa AbstractOperatorExpr || throw(ArgumentError("generated Hubbard certification requires a symbolic Hamiltonian"))
    hopping = build_model(material, reference, HubbardModel(1.0, 0.0; shell=1)).hamiltonian
    interaction = build_model(material, reference, HubbardModel(0.0, 1.0; shell=1)).hamiltonian
    chemical = build_model(material, reference, HubbardModel(0.0, 0.0; shell=1, chemical_potential=1.0)).hamiltonian
    decomposition = PhundamentalCore.structural_span_coordinates(AbstractOperatorExpr[hopping, interaction, chemical],
        model.hamiltonian, model.representation.algebra; tolerance=tolerance)
    decomposition === missing && throw(ArgumentError("generated Hubbard certification requires numeric Hamiltonian coefficients"))
    decomposition.within_span || throw(ArgumentError(
        "the generated Hamiltonian is not an exact scalar nearest-shell Hubbard model (residual=$(decomposition.residual))"))
    t, U, mu = decomposition.coefficients
    specification = HubbardModel(t, U; shell=1, chemical_potential=mu)
    return (interaction=U, chemical_potential=mu, hopping=t, specification=specification,
            canonical_residual=decomposition.residual, certified_from=:structural_generation)
end

function _single_band_hubbard_forward_parameters(model::ManyBodyModel)
    model.representation.algebra isa FermionAlgebra || throw(ArgumentError("single-site DMFT forward integration requires a direct FermionAlgebra representation"))
    iseven(model.representation.algebra.nmodes) || throw(ArgumentError("spinful single-band Hubbard models require an even number of fermion modes"))
    !isnothing(Models._structural_provenance(model)) && return _generated_single_band_hubbard_parameters(model)
    specification = get(model.parameters, :hamiltonian_spec, nothing)
    specification isa HubbardModel || throw(ArgumentError("single-site DMFT requires a certified structural single-band Hubbard model or a typed HubbardModel specification"))
    specification.U isa Real || throw(ArgumentError("single-site DMFT currently requires a scalar Hubbard interaction"))
    specification.chemical_potential isa Real || throw(ArgumentError("single-site DMFT currently requires a scalar chemical potential"))
    return (interaction=Float64(specification.U), chemical_potential=Float64(specification.chemical_potential),
            hopping=specification.t, specification=specification, certified_from=:explicit_hubbard_specification)
end

function _nlce_forward_model_components(model::ManyBodyModel)
    material = get(model.parameters, :material, nothing)
    reference = get(model.parameters, :cluster, nothing)
    specification = get(model.parameters, :hamiltonian_spec, nothing)
    material isa Material || throw(ArgumentError("NLCE forward integration requires model.parameters[:material]"))
    reference isa Supercell || throw(ArgumentError("NLCE forward integration requires model.parameters[:cluster] to be a Supercell"))
    specification isa AbstractHamiltonianSpec || throw(ArgumentError("NLCE forward integration requires model.parameters[:hamiltonian_spec]"))
    return (material=material, reference=reference, specification=specification)
end
