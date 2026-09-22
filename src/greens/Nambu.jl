# Nambu particle-hole indexing layered on the generic Green-function object.

"""
    NambuIndex(modes)

Particle-hole index convention for the Nambu spinor `Ψ = (c₁,…,cₘ,c₁†,…,cₘ†)ᵀ`. Nambu is represented as Green-function index structure rather than as a Hamiltonian representation.
"""
struct NambuIndex
    modes::Vector{Int}
    particle::UnitRange{Int}
    hole::UnitRange{Int}
end

function NambuIndex(modes::AbstractVector{<:Integer})
    selected = Int.(collect(modes))
    isempty(selected) && throw(ArgumentError("Nambu indexing requires at least one fermion mode"))
    length(unique(selected)) == length(selected) || throw(ArgumentError("Nambu modes must be unique"))
    n = length(selected)
    return NambuIndex(selected, 1:n, n+1:2n)
end

function _nambu_labels(index::NambuIndex)
    particles = [Symbol("c_", mode) for mode in index.modes]
    holes = [Symbol("cdag_", mode) for mode in index.modes]
    return vcat(particles, holes)
end

"""
    nambu_green(state, axis; modes=nothing)

Evaluate the exact finite-temperature Nambu Green function using the convention `Ψ = (c,c†)ᵀ` and `𝒢 = -⟨Tτ Ψ(τ)Ψ†(0)⟩`.
"""
function nambu_green(state::ExactGibbsState, axis::FermionicMatsubaraAxis; modes=nothing)
    selected = isnothing(modes) ? _default_fermion_modes(state.model) : Int.(collect(modes))
    index = NambuIndex(selected)
    components = AbstractOperatorExpr[c(mode) for mode in selected]
    append!(components, AbstractOperatorExpr[adjoint(c(mode)) for mode in selected])
    metadata = Dict{Symbol,Any}(:nambu => true, :nambu_index => index, :modes => selected)
    return lehmann_green(state, components, axis; labels=_nambu_labels(index), metadata=metadata)
end

function nambu_green(model::ManyBodyModel, method::ExactGibbs=ExactGibbs(); axis::FermionicMatsubaraAxis, modes=nothing, kB::Real=1.0)
    temperature = axis_temperature(axis; kB=kB)
    state = thermal_state(model, method; temperature=temperature, kB=kB)
    return nambu_green(state, axis; modes=modes)
end

function nambu_index(green::GreenFunction)
    get(green.metadata, :nambu, false) || throw(ArgumentError("Green function does not carry Nambu indexing"))
    return green.metadata[:nambu_index]
end

normal_block(green::GreenFunction, i::Integer) = begin
    index = nambu_index(green)
    view(green.values, index.particle, index.particle, Int(i))
end

anomalous_block(green::GreenFunction, i::Integer) = begin
    index = nambu_index(green)
    view(green.values, index.particle, index.hole, Int(i))
end

hole_block(green::GreenFunction, i::Integer) = begin
    index = nambu_index(green)
    view(green.values, index.hole, index.hole, Int(i))
end
