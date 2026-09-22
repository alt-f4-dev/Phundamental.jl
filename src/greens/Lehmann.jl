# Exact finite-temperature Lehmann Green functions.

function _validate_axis_temperature(state::ExactGibbsState, axis::FermionicMatsubaraAxis)
    isfinite(state.beta) || throw(ArgumentError("finite-temperature Matsubara Green functions require temperature > 0"))
    scale = max(abs(state.beta), abs(axis.beta), 1.0)
    isapprox(state.beta, axis.beta; rtol=1e-12, atol=1e-12 * scale) ||
        throw(ArgumentError("thermal-state beta=$(state.beta) does not match Matsubara-axis beta=$(axis.beta)"))
    return true
end

function _eigenbasis_operator(state::ExactGibbsState, operator)
    O = matrix(operator, state.solution.basis; sparse=false, hbar=get(state.model.parameters, :hbar, 1.0))
    return adjoint(state.solution.left_states) * O * state.solution.right_states
end

"""
    lehmann_green(state, components, axis; labels=nothing, metadata=Dict())

Evaluate the exact finite-temperature fermionic Lehmann representation for a vector of odd-parity operator components `Aₐ`, using `Gₐᵦ(iωₙ) = Σₘₗ (pₘ+pₗ)⟨m|Aₐ|l⟩⟨l|Aᵦ†|m⟩/(iωₙ+Eₘ-Eₗ)`.
"""
function lehmann_green(state::ExactGibbsState, components::AbstractVector, axis::FermionicMatsubaraAxis; labels=nothing, metadata=Dict{Symbol,Any}())
    _validate_axis_temperature(state, axis)
    isempty(components) && throw(ArgumentError("at least one Green-function component is required"))

    energies = Float64.(real.(state.solution.energies))
    probabilities = Float64.(state.probabilities)
    dimension = length(energies)
    length(probabilities) == dimension || throw(DimensionMismatch("thermal probabilities and eigenspectrum have different lengths"))

    components_eigen = [_eigenbasis_operator(state, component) for component in components]
    adjoints_eigen = [_eigenbasis_operator(state, adjoint(component)) for component in components]
    ncomponents = length(components)
    values = zeros(ComplexF64, ncomponents, ncomponents, length(axis))

    @inbounds for a in 1:ncomponents
        A = components_eigen[a]
        for b in 1:ncomponents
            Bdagger = adjoints_eigen[b]
            for m in 1:dimension
                pm = probabilities[m]
                Em = energies[m]
                for n in 1:dimension
                    amplitude = (pm + probabilities[n]) * A[m, n] * Bdagger[n, m]
                    iszero(amplitude) && continue
                    energy_difference = Em - energies[n]
                    for iw in eachindex(axis.values)
                        values[a, b, iw] += amplitude / (axis[iw] + energy_difference)
                    end
                end
            end
        end
    end

    info = _metadata_dict(metadata)
    info[:method] = :exact_lehmann
    info[:temperature] = state.temperature
    info[:beta] = state.beta
    info[:hilbert_dimension] = dimension
    return GreenFunction(axis, values; labels=labels, metadata=info)
end

function _default_fermion_modes(model::ManyBodyModel)
    algebra = model.representation.algebra
    algebra isa FermionAlgebra || throw(ArgumentError("automatic Matsubara components require a FermionAlgebra representation"))
    return collect(1:algebra.nmodes)
end

"""
    matsubara_green(state, axis; modes=nothing, labels=nothing)

Construct the normal finite-temperature fermionic Green function for the requested annihilation modes. When `modes=nothing`, all modes of a non-composite `FermionAlgebra` are included.
"""
function matsubara_green(state::ExactGibbsState, axis::FermionicMatsubaraAxis; modes=nothing, labels=nothing)
    selected = isnothing(modes) ? _default_fermion_modes(state.model) : Int.(collect(modes))
    isempty(selected) && throw(ArgumentError("modes cannot be empty"))
    components = AbstractOperatorExpr[c(mode) for mode in selected]
    component_names = isnothing(labels) ? [Symbol("c_", mode) for mode in selected] : labels
    return lehmann_green(state, components, axis; labels=component_names, metadata=Dict{Symbol,Any}(:nambu => false, :modes => selected))
end

"""
    matsubara_green(model, method=ExactGibbs(); axis, modes=nothing, kB=1, labels=nothing)

Solve `model` with an exact Gibbs method at the temperature encoded by `axis` and evaluate its finite-temperature Matsubara Green function.
"""
function matsubara_green(model::ManyBodyModel, method::ExactGibbs=ExactGibbs(); axis::FermionicMatsubaraAxis, modes=nothing, kB::Real=1.0, labels=nothing)
    temperature = axis_temperature(axis; kB=kB)
    state = thermal_state(model, method; temperature=temperature, kB=kB)
    return matsubara_green(state, axis; modes=modes, labels=labels)
end
