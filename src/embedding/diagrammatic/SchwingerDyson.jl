# Ladder vertices and Schwinger-Dyson reconstruction for the single-band Hubbard model.

"""Momentum-resolved scalar self-energy `Σ(k,iν)` with explicit provenance and convergence metadata."""
struct MomentumResolvedSelfEnergy{A<:FermionicMatsubaraAxis,K,M}
    axis::A
    kpoints::K
    values::Matrix{ComplexF64}
    metadata::M
end

function MomentumResolvedSelfEnergy(axis::A, kpoints, values; metadata=Dict{Symbol,Any}()) where {A<:FermionicMatsubaraAxis}
    ks = collect(kpoints)
    matrix = ComplexF64.(values)
    size(matrix) == (length(ks), length(axis)) || throw(DimensionMismatch("momentum self-energy values must have shape (nk, nfermionic)"))
    return MomentumResolvedSelfEnergy{A,typeof(ks),typeof(metadata)}(axis, ks, matrix, metadata)
end

"""Fully reducible ladder vertex in the same rescaled Bethe-Salpeter convention as `IrreducibleVertex`."""
struct FullVertex{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis,M} <: AbstractTwoParticleFrequencyObject
    fermionic_axis::F
    bosonic_axis::B
    values::Array{ComplexF64,3}
    channel::Symbol
    metadata::M
end

function FullVertex(fermionic_axis::F, bosonic_axis::B, values; channel::Symbol=:charge, metadata=Dict{Symbol,Any}(),
                    copy_values::Bool=true) where {F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    cube = _two_particle_values(fermionic_axis, bosonic_axis, values; copy_values=copy_values)
    return FullVertex{F,B,typeof(metadata)}(fermionic_axis, bosonic_axis, cube, channel, metadata)
end

"""
    full_vertex(bubble, susceptibility)

Recover the fully reducible ladder vertex from `χ = χ₀ + χ₀ F χ₀` in the package's rescaled two-particle convention. Particle-hole bubbles marked as fermion-frequency diagonal use the specialized diagonal inverse path rather than a dense SVD.
"""
function full_vertex(bubble::GeneralizedSusceptibility, susceptibility::GeneralizedSusceptibility; inverse_rtol::Real=sqrt(eps(Float64)),
                     inverse_atol::Real=0.0, pseudoinverse::Bool=false)
    _compatible_two_particle(bubble, susceptibility)
    values = zeros(ComplexF64, size(susceptibility.values))
    diagnostics = Vector{NamedTuple}(undef, size(values, 3))
    active_masks = Vector{Vector{Int}}(undef, size(values, 3))
    for iΩ in axes(values, 3)
        active = _common_active_two_particle_indices(bubble, susceptibility, iΩ)
        bubble_inverse, left = _bubble_inverse(bubble, active, iΩ; rtol=inverse_rtol, atol=inverse_atol,
                                               pseudoinverse=pseudoinverse, return_diagnostics=true)
        values[active, active, iΩ] .= bubble_inverse * (view(susceptibility.values, active, active, iΩ) -
                                         view(bubble.values, active, active, iΩ)) * bubble_inverse
        diagnostics[iΩ] = (bubble=left, active_dimension=length(active))
        active_masks[iΩ] = active
    end
    metadata = Dict{Symbol,Any}(:construction => :ladder_full_vertex, :equation => :chi_equals_chi0_plus_chi0_F_chi0,
                                :vertex_scaling => :bethe_salpeter_rescaled, :inverse_diagnostics => diagnostics,
                                :active_indices_by_bosonic => active_masks)
    return FullVertex(bubble.fermionic_axis, bubble.bosonic_axis, values; channel=bubble.channel, metadata=metadata, copy_values=false)
end

function _opposite_spin_full_vertex(charge::FullVertex, magnetic::FullVertex)
    charge.fermionic_axis.indices == magnetic.fermionic_axis.indices || throw(ArgumentError("charge and magnetic fermionic axes differ"))
    charge.bosonic_axis.indices == magnetic.bosonic_axis.indices || throw(ArgumentError("charge and magnetic bosonic axes differ"))
    values = 0.5 .* (charge.values .- magnetic.values)
    metadata = Dict{Symbol,Any}(:construction => :spin_opposite_from_charge_magnetic,
                                :identity => :F_updown_equals_half_F_charge_minus_F_magnetic)
    return FullVertex(charge.fermionic_axis, charge.bosonic_axis, values; channel=:updown, metadata=metadata, copy_values=false)
end

function _shift_lookup(axis::FermionicMatsubaraAxis, bosonic_axis::BosonicMatsubaraAxis)
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    return [Int[get(lookup, index + bosonic_axis.indices[iΩ], 0) for index in axis.indices] for iΩ in eachindex(bosonic_axis.indices)]
end

function _schwinger_dyson_contractions(updown_vertices::AbstractVector{<:FullVertex}, updown_bubbles::AbstractVector{<:GeneralizedSusceptibility},
                                       bosonic_axis::BosonicMatsubaraAxis, parallel::Symbol)
    nq = length(updown_vertices)
    nν = length(first(updown_vertices).fermionic_axis)
    nΩ = length(bosonic_axis)
    contractions = zeros(ComplexF64, nq, nν, nΩ)
    function process_q(iq::Int)
        vertex = updown_vertices[iq]
        bubble = updown_bubbles[iq]
        for iΩ in 1:nΩ
            active = _common_active_two_particle_indices(vertex, bubble, iΩ)
            for iν in active
                accumulator = zero(ComplexF64)
                @inbounds @simd for iνp in active
                    accumulator += vertex.values[iν, iνp, iΩ] * bubble.values[iνp, iνp, iΩ]
                end
                contractions[iq, iν, iΩ] = accumulator
            end
        end
        return nothing
    end
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for iq in 1:nq
            process_q(iq)
        end
    else
        for iq in 1:nq
            process_q(iq)
        end
    end
    return contractions
end

"""
    schwinger_dyson_self_energy(dmft_result, lattice, momentum_green, updown_vertices, updown_bubbles, bosonic_axis; require_complete_frequency_box=true, parallel=:serial)

One-shot ladder Schwinger-Dyson reconstruction for the paramagnetic single-band Hubbard model. The expensive vertex-bubble contraction is evaluated once for each `(q,ν,Ω)` and reused for every external momentum. `parallel=:threads` distributes both q-local contractions and independent external-momentum rows across Julia threads. The static Hartree term is fixed by the converged impurity density.
"""
function schwinger_dyson_self_energy(dmft_result::DMFTResult, lattice::MomentumLattice, green::MomentumResolvedGreenFunction,
                                     updown_vertices::AbstractVector{<:FullVertex}, updown_bubbles::AbstractVector{<:GeneralizedSusceptibility},
                                     bosonic_axis::BosonicMatsubaraAxis; require_complete_frequency_box::Bool=true, parallel::Symbol=:serial)
    length(updown_vertices) == length(updown_bubbles) == length(lattice.qmaps) || throw(DimensionMismatch("q-resolved vertex, bubble, and q-map counts differ"))
    parallel in (:serial, :threads) || throw(ArgumentError("parallel must be :serial or :threads"))
    axis = green.axis
    isapprox(axis.beta, bosonic_axis.beta; rtol=0.0, atol=16eps(Float64) * max(axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic axes must share beta"))
    beta = axis.beta
    U = dmft_result.problem.interaction
    density = dmft_result.impurity_result.density
    hartree = U * density / 2
    values = fill(ComplexF64(hartree), length(lattice.kpoints), length(axis))
    shifts = _shift_lookup(axis, bosonic_axis)
    missing_per_q = sum(count(iszero, shift) for shift in shifts)
    missing = missing_per_q * length(lattice.qmaps)
    if require_complete_frequency_box && missing > 0
        for (iΩ, shift) in enumerate(shifts), iν in eachindex(axis.indices)
            iszero(shift[iν]) || continue
            throw(ArgumentError("Schwinger-Dyson frequency box does not contain ν+Ω for ν=$(axis.indices[iν]), Ω=$(bosonic_axis.indices[iΩ])"))
        end
    end
    contractions = _schwinger_dyson_contractions(updown_vertices, updown_bubbles, bosonic_axis, parallel)
    nq = length(lattice.qmaps)
    factor = U / (beta * nq)
    function process_k(ik::Int)
        @inbounds for iν in eachindex(axis.indices)
            correction = zero(ComplexF64)
            for iq in 1:nq, iΩ in eachindex(bosonic_axis.indices)
                shifted_index = shifts[iΩ][iν]
                iszero(shifted_index) && continue
                correction += contractions[iq, iν, iΩ] * green.values[lattice.qmaps[iq][ik], shifted_index]
            end
            values[ik, iν] -= factor * correction
        end
        return nothing
    end
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for ik in eachindex(lattice.kpoints)
            process_k(ik)
        end
    else
        for ik in eachindex(lattice.kpoints)
            process_k(ik)
        end
    end
    metadata = Dict{Symbol,Any}(:construction => :ladder_schwinger_dyson, :hartree => hartree, :interaction => U, :density => density,
                                :frequency_box_complete => iszero(missing), :missing_frequency_entries => missing,
                                :vertex_convention => :chi_equals_chi0_plus_chi0_F_chi0, :one_shot => true,
                                :parallel => parallel, :julia_threads => Base.Threads.nthreads(), :contractions_precomputed => true)
    return MomentumResolvedSelfEnergy(axis, lattice.kpoints, values; metadata=metadata)
end

"""Reconstruct `G(k,iν)` from a momentum-resolved self-energy and a `MomentumLattice`."""
function dyson_momentum_green(lattice::MomentumLattice, sigma::MomentumResolvedSelfEnergy; chemical_potential::Real=0.0)
    length(lattice.kpoints) == length(sigma.kpoints) || throw(DimensionMismatch("lattice and self-energy momentum counts differ"))
    values = Matrix{ComplexF64}(undef, length(lattice.kpoints), length(sigma.axis))
    @inbounds for ik in eachindex(lattice.kpoints), iν in eachindex(sigma.axis.indices)
        values[ik, iν] = inv(sigma.axis[iν] + chemical_potential - lattice.energies[ik] - sigma.values[ik, iν])
    end
    metadata = Dict{Symbol,Any}(:construction => :momentum_dyson, :chemical_potential => Float64(chemical_potential))
    return MomentumResolvedGreenFunction(sigma.axis, lattice.kpoints, values; metadata=metadata)
end

"""Maximum relative norm of the nonlocal self-energy correction with respect to the converged local DMFT self-energy."""
function nonlocal_self_energy_residual(result::MomentumResolvedSelfEnergy, local_self_energy::SelfEnergy)
    matrix_dimension(local_self_energy) == 1 || throw(DimensionMismatch("local reference self-energy must be scalar"))
    _compatible_frequency_objects(local_self_energy, zero_self_energy(result.axis, 1; labels=[:local]))
    reference = reshape([local_self_energy[1, 1, iν] for iν in eachindex(result.axis.indices)], 1, length(result.axis))
    difference = result.values .- reference
    return norm(difference) / max(norm(result.values), norm(reference), eps(Float64))
end
