# Finite Anderson impurity models used by embedding solvers.

"""
    AndersonImpurityModel(U; chemical_potential=0, impurity_energy=0, bath_energies=[], hybridizations=[])

Spin-degenerate single-orbital Anderson impurity model. `bath_energies` are physical one-particle bath energies, so the grand-canonical bath term is `(εₚ-μ)nₚ` and `Δ(z)=Σₚ |Vₚ|²/(z+μ-εₚ)`.
"""
struct AndersonImpurityModel <: AbstractHamiltonianSpec
    U::Float64
    chemical_potential::Float64
    impurity_energy::Float64
    bath_energies::Vector{Float64}
    hybridizations::Vector{ComplexF64}
end

function AndersonImpurityModel(U::Real; chemical_potential::Real=0.0, impurity_energy::Real=0.0,
                               bath_energies=Float64[], hybridizations=ComplexF64[])
    energies = Float64.(collect(bath_energies))
    couplings = ComplexF64.(collect(hybridizations))
    length(energies) == length(couplings) || throw(DimensionMismatch("bath_energies and hybridizations must have equal length"))
    all(isfinite, energies) || throw(ArgumentError("bath energies must be finite"))
    all(v -> isfinite(real(v)) && isfinite(imag(v)), couplings) || throw(ArgumentError("hybridizations must be finite"))
    return AndersonImpurityModel(Float64(U), Float64(chemical_potential), Float64(impurity_energy), energies, couplings)
end

"""Number of finite bath orbitals per spin in an Anderson impurity specification."""
impurity_bath_sites(spec::AndersonImpurityModel) = length(spec.bath_energies)

@inline _impurity_mode(spin::Symbol) = spin === :up ? 1 : spin === :down ? 2 : throw(ArgumentError("spin must be :up or :down"))
@inline _bath_mode(site::Integer, spin::Symbol) = 2 * Int(site) + (spin === :up ? 1 : spin === :down ? 2 : throw(ArgumentError("spin must be :up or :down")))

"""
    build_model(spec::AndersonImpurityModel)

Construct the finite spinful impurity Hamiltonian directly, without requiring crystallographic material metadata.
"""
function build_model(spec::AndersonImpurityModel)
    nbath = impurity_bath_sites(spec)
    nmodes = 2 * (1 + nbath)
    up = _impurity_mode(:up)
    down = _impurity_mode(:down)
    terms = AbstractOperatorExpr[]

    impurity_level = spec.impurity_energy - spec.chemical_potential
    !iszero(impurity_level) && push!(terms, impurity_level * nf(up), impurity_level * nf(down))
    !iszero(spec.U) && push!(terms, spec.U * (nf(up) * nf(down)))

    for p in 1:nbath
        bath_level = spec.bath_energies[p] - spec.chemical_potential
        coupling = spec.hybridizations[p]
        for spin in (:up, :down)
            impurity = _impurity_mode(spin)
            bath = _bath_mode(p, spin)
            !iszero(bath_level) && push!(terms, bath_level * nf(bath))
            if !iszero(coupling)
                push!(terms, coupling * (c(impurity)' * c(bath)))
                push!(terms, conj(coupling) * (c(bath)' * c(impurity)))
            end
        end
    end

    representation = Representation(:anderson_impurity, FermionFockSpace(nmodes), FermionAlgebra(nmodes),
                                    FermionOccupationBasis(collect(1:nmodes)); ordering=collect(1:nmodes), reference_state=:fermion_vacuum)
    observables = Dict{Symbol,Any}(
        :impurity_density => nf(up) + nf(down),
        :impurity_double_occupancy => nf(up) * nf(down),
        :impurity_spin_z => 0.5 * (nf(up) - nf(down)),
    )
    fermion_mode_map = Dict{Tuple{Int,Symbol},Int}((1, :up) => up, (1, :down) => down)
    for p in 1:nbath
        fermion_mode_map[(p + 1, :up)] = _bath_mode(p, :up)
        fermion_mode_map[(p + 1, :down)] = _bath_mode(p, :down)
    end
    parameters = Dict{Symbol,Any}(
        :hamiltonian_spec => spec,
        :chemical_potential => spec.chemical_potential,
        :impurity_energy => spec.impurity_energy,
        :bath_energies => copy(spec.bath_energies),
        :hybridizations => copy(spec.hybridizations),
        :impurity_modes => Dict(:up => up, :down => down),
        :fermion_mode_map => fermion_mode_map,
        :hbar => 1.0,
    )
    return ManyBodyModel(representation, _operator_sum(terms); parameters=parameters, observables=observables,
                         provenance=[(construction=:anderson_impurity, bath_sites=nbath, fermion_modes=nmodes)])
end
