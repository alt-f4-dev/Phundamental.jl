# ============================================================================
# SchwingerDyson.jl: preserved implementation from src/embedding/diagrammatic/SchwingerDyson.jl
# ============================================================================


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

# ============================================================================
# DiagrammaticDMFT.jl: preserved implementation from src/embedding/diagrammatic/DiagrammaticDMFT.jl
# ============================================================================


# Ladder dynamical-vertex extension of a converged single-site DMFT solution.


abstract type AbstractDiagrammaticDMFTMethod end

"""
    LadderDGA(; regularization=0, inverse_rtol=sqrt(eps()), inverse_atol=0, pseudoinverse=false,
                reconstruct_self_energy=true, require_complete_frequency_box=false)

One-shot single-band ladder DΓA method in the charge and magnetic particle-hole channels. `reconstruct_self_energy=true` activates the Schwinger-Dyson stage for an end-to-end `LadderDGAProblem`. The method retains explicit matrix-conditioning controls and does not hide finite-frequency-box truncation behind unconditional inverses.
"""
struct LadderDGA <: AbstractDiagrammaticDMFTMethod
    regularization::Float64
    inverse_rtol::Float64
    inverse_atol::Float64
    pseudoinverse::Bool
    reconstruct_self_energy::Bool
    require_complete_frequency_box::Bool
end

function LadderDGA(; regularization::Real=0.0, inverse_rtol::Real=sqrt(eps(Float64)), inverse_atol::Real=0.0, pseudoinverse::Bool=false,
                   reconstruct_self_energy::Bool=true, require_complete_frequency_box::Bool=false)
    regularization >= 0 || throw(ArgumentError("regularization must be nonnegative"))
    inverse_rtol >= 0 || throw(ArgumentError("inverse_rtol must be nonnegative"))
    inverse_atol >= 0 || throw(ArgumentError("inverse_atol must be nonnegative"))
    return LadderDGA(Float64(regularization), Float64(inverse_rtol), Float64(inverse_atol), pseudoinverse,
                     reconstruct_self_energy, require_complete_frequency_box)
end

# Backward-compatible positional construction from the first diagrammatic-DMFT milestone.
LadderDGA(regularization::Real) = LadderDGA(; regularization=regularization)

"""
    DiagrammaticDMFTProblem(dmft_result, local_vertex, lattice_bubbles; qpoints=nothing)

Low-level ladder problem built from a converged local DMFT reference, one local irreducible vertex, and explicitly supplied momentum-resolved lattice bubbles. This constructor is retained for backward compatibility and for controlled unit tests; `LadderDGAProblem` is the end-to-end research workflow.
"""
struct DiagrammaticDMFTProblem{D,V,B,Q} <: AbstractEmbeddingProblem
    dmft_result::D
    local_vertex::V
    lattice_bubbles::B
    qpoints::Q
end

function DiagrammaticDMFTProblem(dmft_result::DMFTResult, local_vertex::IrreducibleVertex, lattice_bubbles; qpoints=nothing)
    bubbles = collect(lattice_bubbles)
    isempty(bubbles) && throw(ArgumentError("diagrammatic DMFT requires at least one lattice bubble"))
    all(b -> b isa GeneralizedSusceptibility, bubbles) || throw(ArgumentError("lattice_bubbles must contain GeneralizedSusceptibility objects"))
    for bubble in bubbles
        _compatible_two_particle(bubble, local_vertex)
    end
    qs = isnothing(qpoints) ? collect(1:length(bubbles)) : collect(qpoints)
    length(qs) == length(bubbles) || throw(DimensionMismatch("qpoint count must equal lattice-bubble count"))
    return DiagrammaticDMFTProblem(dmft_result, local_vertex, bubbles, qs)
end

function DiagrammaticDMFTProblem(dmft_result::DMFTResult, local_bubble::GeneralizedSusceptibility,
                                 local_susceptibility::GeneralizedSusceptibility, lattice_bubbles;
                                 qpoints=nothing, regularization::Real=0.0, inverse_rtol::Real=sqrt(eps(Float64)),
                                 inverse_atol::Real=0.0, pseudoinverse::Bool=false)
    vertex = irreducible_vertex(local_bubble, local_susceptibility; regularization=regularization, inverse_rtol=inverse_rtol,
                                inverse_atol=inverse_atol, pseudoinverse=pseudoinverse)
    return DiagrammaticDMFTProblem(dmft_result, vertex, lattice_bubbles; qpoints=qpoints)
end

function DiagrammaticDMFTProblem(dmft_result::DMFTResult, local_vertex::IrreducibleVertex, green::MomentumResolvedGreenFunction,
                                 qmaps, bosonic_axis::BosonicMatsubaraAxis; qpoints=nothing, missing_frequency::Symbol=:error)
    bubbles, diagnostics = lattice_bubble(green, qmaps, bosonic_axis; channel=local_vertex.channel, missing_frequency=missing_frequency)
    for bubble in bubbles
        bubble.metadata[:frequency_box_complete] = diagnostics.frequency_box_complete
        bubble.metadata[:missing_frequency_entries] = diagnostics.missing_frequency_entries
    end
    return DiagrammaticDMFTProblem(dmft_result, local_vertex, bubbles; qpoints=qpoints)
end

"""
    LadderDGAProblem(dmft_result, lattice, bosonic_axis; impurity_backend=TauGridFourPoint(), impurity_two_particle=nothing)

End-to-end one-shot ladder-DΓA problem. Starting from a converged single-site `DMFTResult`, the solver constructs or consumes the local impurity four-point function, extracts charge and magnetic irreducible vertices, forms `G(k,iν)` and momentum-resolved particle-hole bubbles, solves both ladder Bethe-Salpeter equations, reconstructs the opposite-spin full vertex, and optionally evaluates the Schwinger-Dyson momentum-dependent self-energy.

When `impurity_two_particle=nothing`, a dense exact-Gibbs impurity state is reconstructed and evaluated with `impurity_backend`. `TauGridFourPoint` is deliberately a small-system reference backend; production calculations may supply an independently converged `ImpurityTwoParticleResult` from a higher-performance impurity vertex solver without changing the downstream DΓA API.
"""
struct LadderDGAProblem{D,L,B,I,T} <: AbstractEmbeddingProblem
    dmft_result::D
    lattice::L
    bosonic_axis::B
    impurity_backend::I
    impurity_two_particle::T
end

function LadderDGAProblem(dmft_result::DMFTResult, lattice::MomentumLattice, bosonic_axis::BosonicMatsubaraAxis;
                          impurity_backend::AbstractImpurityTwoParticleBackend=TauGridFourPoint(), impurity_two_particle=nothing)
    dmft_result.converged || throw(ArgumentError("LadderDGAProblem requires a converged DMFT reference"))
    isnothing(lattice.qmaps) && throw(ArgumentError("MomentumLattice requires qmaps for ladder DΓA"))
    isapprox(dmft_result.problem.axis.beta, bosonic_axis.beta; rtol=1e-12, atol=1e-12 * max(bosonic_axis.beta, 1.0)) ||
        throw(ArgumentError("DMFT and bosonic Matsubara axes have different inverse temperatures"))
    !isnothing(impurity_two_particle) && !(impurity_two_particle isa ImpurityTwoParticleResult) &&
        throw(ArgumentError("impurity_two_particle must be an ImpurityTwoParticleResult or nothing"))
    return LadderDGAProblem(dmft_result, lattice, bosonic_axis, impurity_backend, impurity_two_particle)
end

"""Execution policy for diagrammatic-DMFT calculations. `parallel=:threads` parallelizes independent momentum-transfer work without changing the physical method."""
struct DiagrammaticExecutionPolicy
    parallel::Symbol
end

function DiagrammaticExecutionPolicy(; parallel::Symbol=:serial)
    parallel in (:serial, :threads) || throw(ArgumentError("parallel must be :serial or :threads"))
    return DiagrammaticExecutionPolicy(parallel)
end

"""Solver descriptor for low-level and end-to-end ladder diagrammatic-DMFT calculations."""
struct DiagrammaticDMFTSolver{M<:AbstractDiagrammaticDMFTMethod,E<:DiagrammaticExecutionPolicy} <: AbstractSolver
    method::M
    execution::E
end

DiagrammaticDMFTSolver(method::AbstractDiagrammaticDMFTMethod) = DiagrammaticDMFTSolver(method, DiagrammaticExecutionPolicy())
DiagrammaticDMFTSolver(; method::AbstractDiagrammaticDMFTMethod=LadderDGA(), execution::DiagrammaticExecutionPolicy=DiagrammaticExecutionPolicy(),
                       parallel::Union{Nothing,Symbol}=nothing) =
    DiagrammaticDMFTSolver(method, isnothing(parallel) ? execution : DiagrammaticExecutionPolicy(; parallel=parallel))
solver_name(::DiagrammaticDMFTSolver) = :ladder_dga

"""Momentum-resolved low-level diagrammatic-DMFT susceptibility result."""
struct DiagrammaticDMFTResult{P,S,G,O,M}
    problem::P
    solver::S
    generalized_susceptibilities::G
    observable::O
    metadata::M
end

"""End-to-end ladder-DΓA result with local two-particle data, both particle-hole channels, nonlocal self-energy, and reconstructed lattice Green function."""
struct LadderDGAResult{P,S,I,V,B,X,F,O,SE,G,M}
    problem::P
    solver::S
    impurity_two_particle::I
    local_vertices::V
    lattice_bubbles::B
    generalized_susceptibilities::X
    full_vertices::F
    observables::O
    self_energy::SE
    momentum_green_function::G
    metadata::M
end

function _maximum_inverse_condition(susceptibilities)
    maximum_condition = 0.0
    for susceptibility_value in susceptibilities
        diagnostics = get(susceptibility_value.metadata, :inverse_diagnostics, NamedTuple[])
        for diagnostic in diagnostics
            condition = diagnostic.kernel.condition_number
            maximum_condition = max(maximum_condition, condition)
        end
    end
    return maximum_condition
end

function _solve_channel_bethe_salpeter(bubbles, vertex::IrreducibleVertex, method::LadderDGA, parallel::Symbol)
    susceptibilities = Vector{GeneralizedSusceptibility}(undef, length(bubbles))
    solve_one(iq) = bethe_salpeter(bubbles[iq], vertex; regularization=method.regularization, inverse_rtol=method.inverse_rtol,
                                   inverse_atol=method.inverse_atol, pseudoinverse=method.pseudoinverse)
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for iq in eachindex(bubbles)
            susceptibilities[iq] = solve_one(iq)
        end
    else
        for iq in eachindex(bubbles)
            susceptibilities[iq] = solve_one(iq)
        end
    end
    return susceptibilities
end

function _solve_full_vertices(bubbles, susceptibilities, method::LadderDGA, parallel::Symbol)
    vertices = Vector{FullVertex}(undef, length(bubbles))
    solve_one(iq) = full_vertex(bubbles[iq], susceptibilities[iq]; inverse_rtol=method.inverse_rtol, inverse_atol=method.inverse_atol,
                                pseudoinverse=method.pseudoinverse)
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for iq in eachindex(bubbles)
            vertices[iq] = solve_one(iq)
        end
    else
        for iq in eachindex(bubbles)
            vertices[iq] = solve_one(iq)
        end
    end
    return vertices
end

_hermitian_susceptibility_channel(channel::Symbol) = channel in (:density, :charge, :magnetic, :spin, :spin_x, :spin_y, :spin_z)

function _solve_diagrammatic_dmft(solver::DiagrammaticDMFTSolver, problem::DiagrammaticDMFTProblem)
    method = solver.method
    parallel = solver.execution.parallel
    susceptibilities = _solve_channel_bethe_salpeter(problem.lattice_bubbles, problem.local_vertex, method, parallel)
    nΩ = length(problem.local_vertex.bosonic_axis)
    values = Matrix{ComplexF64}(undef, length(susceptibilities), nΩ)
    beta = inverse_temperature(problem.local_vertex.bosonic_axis)
    for (iq, susceptibility_value) in enumerate(susceptibilities)
        values[iq, :] .= physical_susceptibility(susceptibility_value; beta_normalization=beta^2)
    end
    frequency_complete = all(bubble -> get(bubble.metadata, :frequency_box_complete, true), problem.lattice_bubbles)
    metadata = Dict{Symbol,Any}(:method => :ladder_dga, :approximation => :single_channel_ladder_bethe_salpeter,
                                :response_kind => :susceptibility, :frequency_domain => :matsubara, :statistics => :bosonic,
                                :momentum_scope => :momentum_resolved, :response_order => 2, :spectral_axis => :bosonic_matsubara,
                                :channel => problem.local_vertex.channel, :operator_channels => (problem.local_vertex.channel,),
                                :hermitian_channel => _hermitian_susceptibility_channel(problem.local_vertex.channel),
                                :correlation_scope => :long_range_critical,
                                :local_reference => :single_site_dmft, :nonlocal_self_energy_reconstructed => false,
                                :regularization => method.regularization, :inverse_rtol => method.inverse_rtol,
                                :inverse_atol => method.inverse_atol, :pseudoinverse => method.pseudoinverse,
                                :maximum_inverse_condition => _maximum_inverse_condition(susceptibilities),
                                :frequency_box_complete => frequency_complete, :converged_dmft_reference => problem.dmft_result.converged,
                                :parallel => parallel, :julia_threads => Base.Threads.nthreads())
    observable = SusceptibilityResult(problem.qpoints, problem.local_vertex.bosonic_axis, values, metadata)
    return DiagrammaticDMFTResult(problem, solver, susceptibilities, observable, metadata)
end

function _dense_reference_state(dmft_result::DMFTResult, backend::TauGridFourPoint)
    state = dmft_result.impurity_result.thermal_state
    if state isa ExactGibbsState
        length(state.solution.energies) <= backend.max_dimension || throw(ArgumentError("existing dense impurity state exceeds TauGridFourPoint.max_dimension"))
        return state
    end
    model = dmft_result.impurity_result.model
    temperature = dmft_result.problem.temperature
    dense = thermal_state(model, ExactGibbs(solver=ExactDiagonalization()); temperature=temperature, kB=dmft_result.problem.kB)
    length(dense.solution.energies) <= backend.max_dimension || throw(ArgumentError("dense two-particle reference state exceeds TauGridFourPoint.max_dimension; supply precomputed impurity two-particle data"))
    return dense
end

function _local_channel_bubble(base::GeneralizedSusceptibility, channel::Symbol)
    metadata = copy(base.metadata)
    metadata[:channel_relabel] = channel
    metadata[:shared_values] = true
    return GeneralizedSusceptibility(base.fermionic_axis, base.bosonic_axis, base.values; channel=channel, metadata=metadata, copy_values=false)
end

function _channel_observable(qpoints, bosonic_axis, susceptibilities, channel::Symbol)
    beta = bosonic_axis.beta
    values = Matrix{ComplexF64}(undef, length(susceptibilities), length(bosonic_axis))
    for (iq, susceptibility_value) in enumerate(susceptibilities)
        values[iq, :] .= physical_susceptibility(susceptibility_value; beta_normalization=beta^2)
    end
    metadata = Dict{Symbol,Any}(:method => :ladder_dga, :response_kind => :susceptibility, :frequency_domain => :matsubara,
                                :statistics => :bosonic, :momentum_scope => :momentum_resolved, :response_order => 2,
                                :spectral_axis => :bosonic_matsubara, :channel => channel, :operator_channels => (channel,),
                                :hermitian_channel => _hermitian_susceptibility_channel(channel),
                                :channel_normalization => :upup_plusminus_updown)
    return SusceptibilityResult(qpoints, bosonic_axis, values, metadata)
end

function _solve_ladder_dga(solver::DiagrammaticDMFTSolver, problem::LadderDGAProblem)
    method = solver.method
    parallel = solver.execution.parallel
    dmft = problem.dmft_result
    axis = dmft.problem.axis
    impurity = if isnothing(problem.impurity_two_particle)
        problem.impurity_backend isa TauGridFourPoint || throw(ArgumentError("automatic impurity evaluation currently requires TauGridFourPoint"))
        state = _dense_reference_state(dmft, problem.impurity_backend)
        impurity_two_particle(state, axis, problem.bosonic_axis; backend=problem.impurity_backend)
    else
        problem.impurity_two_particle
    end

    local_bubble_charge = _local_channel_bubble(impurity.local_bubble, :charge)
    local_bubble_magnetic = _local_channel_bubble(impurity.local_bubble, :magnetic)
    charge_vertex = irreducible_vertex(local_bubble_charge, impurity.charge; regularization=method.regularization,
                                       inverse_rtol=method.inverse_rtol, inverse_atol=method.inverse_atol, pseudoinverse=method.pseudoinverse)
    magnetic_vertex = irreducible_vertex(local_bubble_magnetic, impurity.magnetic; regularization=method.regularization,
                                         inverse_rtol=method.inverse_rtol, inverse_atol=method.inverse_atol, pseudoinverse=method.pseudoinverse)

    lattice = problem.lattice
    green = momentum_green(lattice, axis, dmft.self_energy; chemical_potential=dmft.problem.chemical_potential)
    charge_bubbles, frequency_box = lattice_bubble(green, lattice.qmaps, problem.bosonic_axis; channel=:charge, missing_frequency=:zero,
                                                    parallel=parallel)
    magnetic_bubbles = [_local_channel_bubble(bubble, :magnetic) for bubble in charge_bubbles]
    charge_susceptibilities = _solve_channel_bethe_salpeter(charge_bubbles, charge_vertex, method, parallel)
    magnetic_susceptibilities = _solve_channel_bethe_salpeter(magnetic_bubbles, magnetic_vertex, method, parallel)
    charge_full = _solve_full_vertices(charge_bubbles, charge_susceptibilities, method, parallel)
    magnetic_full = _solve_full_vertices(magnetic_bubbles, magnetic_susceptibilities, method, parallel)
    updown_full = Vector{FullVertex}(undef, length(charge_full))
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for iq in eachindex(charge_full)
            updown_full[iq] = _opposite_spin_full_vertex(charge_full[iq], magnetic_full[iq])
        end
    else
        for iq in eachindex(charge_full)
            updown_full[iq] = _opposite_spin_full_vertex(charge_full[iq], magnetic_full[iq])
        end
    end
    updown_bubbles = [_local_channel_bubble(bubble, :updown) for bubble in charge_bubbles]

    sigma_nonlocal = method.reconstruct_self_energy ?
        schwinger_dyson_self_energy(dmft, lattice, green, updown_full, updown_bubbles, problem.bosonic_axis;
                                    require_complete_frequency_box=method.require_complete_frequency_box, parallel=parallel) : nothing
    corrected_green = isnothing(sigma_nonlocal) ? green : dyson_momentum_green(lattice, sigma_nonlocal; chemical_potential=dmft.problem.chemical_potential)
    qpoints = lattice.qpoints
    observables = Dict{Symbol,Any}(:charge => _channel_observable(qpoints, problem.bosonic_axis, charge_susceptibilities, :charge),
                                   :magnetic => _channel_observable(qpoints, problem.bosonic_axis, magnetic_susceptibilities, :magnetic))
    local_vertices = Dict{Symbol,Any}(:charge => charge_vertex, :magnetic => magnetic_vertex)
    lattice_bubbles = Dict{Symbol,Any}(:charge => charge_bubbles, :magnetic => magnetic_bubbles)
    susceptibilities = Dict{Symbol,Any}(:charge => charge_susceptibilities, :magnetic => magnetic_susceptibilities)
    full_vertices = Dict{Symbol,Any}(:charge => charge_full, :magnetic => magnetic_full, :updown => updown_full)
    metadata = Dict{Symbol,Any}(:method => :ladder_dga, :approximation => :one_shot_charge_magnetic_particle_hole_ladders,
                                :correlation_scope => :long_range_critical, :converged_dmft_reference => dmft.converged,
                                :impurity_backend => get(impurity.diagnostics, :backend, :precomputed),
                                :frequency_box_complete => frequency_box.frequency_box_complete,
                                :missing_frequency_entries => frequency_box.missing_frequency_entries,
                                :regularization => method.regularization, :inverse_rtol => method.inverse_rtol, :inverse_atol => method.inverse_atol,
                                :pseudoinverse => method.pseudoinverse, :nonlocal_self_energy_reconstructed => !isnothing(sigma_nonlocal),
                                :self_energy_feedback => false, :channels => (:charge, :magnetic), :parallel => parallel,
                                :julia_threads => Base.Threads.nthreads(), :shared_lattice_bubble_storage => true)
    return LadderDGAResult(problem, solver, impurity, local_vertices, lattice_bubbles, susceptibilities,
                           full_vertices, observables, sigma_nonlocal, corrected_green, metadata)
end

function solve(solver::DiagrammaticDMFTSolver, problem::DiagrammaticDMFTProblem)
    problem.dmft_result.converged || throw(ArgumentError("diagrammatic DMFT requires a converged DMFT reference result"))
    return _solve_diagrammatic_dmft(solver, problem)
end

function solve(solver::DiagrammaticDMFTSolver, problem::LadderDGAProblem)
    solver.method isa LadderDGA || throw(ArgumentError("LadderDGAProblem requires a LadderDGA method"))
    return _solve_ladder_dga(solver, problem)
end

"""Return the low-level susceptibility produced by a `DiagrammaticDMFTResult`."""
susceptibility(result::DiagrammaticDMFTResult) = result.observable

"""Return a charge or magnetic observable from an end-to-end `LadderDGAResult`."""
function susceptibility(result::LadderDGAResult, channel::Symbol=:magnetic)
    haskey(result.observables, channel) || throw(ArgumentError("available ladder-DΓA channels are $(collect(keys(result.observables)))"))
    return result.observables[channel]
end

"""Maximum norm difference between two low-level diagrammatic-DMFT susceptibility grids with identical q and Ω dimensions."""
function diagrammatic_convergence(a::DiagrammaticDMFTResult, b::DiagrammaticDMFTResult)
    size(a.observable.values) == size(b.observable.values) || throw(DimensionMismatch("diagrammatic susceptibility grids have different shapes"))
    return Float64(norm(a.observable.values .- b.observable.values))
end

function diagrammatic_convergence(a::LadderDGAResult, b::LadderDGAResult)
    residuals = Float64[]
    for channel in (:charge, :magnetic)
        left = susceptibility(a, channel).values
        right = susceptibility(b, channel).values
        size(left) == size(right) || throw(DimensionMismatch("diagrammatic susceptibility grids have different shapes"))
        push!(residuals, norm(left .- right) / max(norm(left), norm(right), eps(Float64)))
    end
    if !isnothing(a.self_energy) && !isnothing(b.self_energy)
        size(a.self_energy.values) == size(b.self_energy.values) || throw(DimensionMismatch("diagrammatic self-energy grids have different shapes"))
        push!(residuals, norm(a.self_energy.values .- b.self_energy.values) / max(norm(a.self_energy.values), norm(b.self_energy.values), eps(Float64)))
    end
    return maximum(residuals)
end

"""Validate Bethe-Salpeter closure, finite values, frequency-box provenance, and Schwinger-Dyson output availability for a ladder result."""
function validate_diagrammatic_dmft_result(result::DiagrammaticDMFTResult; atol::Real=1e-10, rtol::Real=1e-8, require_complete_frequency_box::Bool=false)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    length(result.generalized_susceptibilities) == length(result.problem.lattice_bubbles) ||
        return (valid=false, max_bethe_salpeter_residual=Inf, checked_momenta=0, frequency_box_complete=false)
    method = result.solver.method
    residual_max = 0.0
    for (bubble, susceptibility_value) in zip(result.problem.lattice_bubbles, result.generalized_susceptibilities)
        expected = bethe_salpeter(bubble, result.problem.local_vertex; regularization=method.regularization, inverse_rtol=method.inverse_rtol,
                                  inverse_atol=method.inverse_atol, pseudoinverse=method.pseudoinverse)
        residual = norm(expected.values .- susceptibility_value.values)
        scale = max(norm(expected.values), norm(susceptibility_value.values), 1.0)
        residual_max = max(residual_max, Float64(residual))
        residual <= atol + rtol * scale || return (valid=false, max_bethe_salpeter_residual=residual_max,
                                                     checked_momenta=length(result.generalized_susceptibilities),
                                                     frequency_box_complete=get(result.metadata, :frequency_box_complete, false))
    end
    finite = all(isfinite, real.(result.observable.values)) && all(isfinite, imag.(result.observable.values))
    frequency_complete = get(result.metadata, :frequency_box_complete, false)
    valid = finite && (!require_complete_frequency_box || frequency_complete)
    return (valid=valid, max_bethe_salpeter_residual=residual_max, checked_momenta=length(result.generalized_susceptibilities),
            frequency_box_complete=frequency_complete)
end

function validate_diagrammatic_dmft_result(result::LadderDGAResult; require_self_energy::Bool=true, require_complete_frequency_box::Bool=false)
    finite_susceptibility = all(channel -> all(isfinite, real.(susceptibility(result, channel).values)) &&
                                               all(isfinite, imag.(susceptibility(result, channel).values)), (:charge, :magnetic))
    self_energy_available = !isnothing(result.self_energy)
    finite_self_energy = !self_energy_available || (all(isfinite, real.(result.self_energy.values)) && all(isfinite, imag.(result.self_energy.values)))
    frequency_complete = get(result.metadata, :frequency_box_complete, false)
    valid = finite_susceptibility && finite_self_energy && (!require_self_energy || self_energy_available) &&
            (!require_complete_frequency_box || frequency_complete)
    return (valid=valid, finite_susceptibility=finite_susceptibility, finite_self_energy=finite_self_energy,
            self_energy_available=self_energy_available, frequency_box_complete=frequency_complete,
            impurity_diagnostics=result.impurity_two_particle.diagnostics)
end

# ============================================================================
# Benchmarking.jl: preserved implementation from src/embedding/diagrammatic/Benchmarking.jl
# ============================================================================


# Peer-reviewed benchmark descriptors and residual reports for diagrammatic-DMFT observables.

"""Peer-reviewed susceptibility benchmark with explicit model parameters, axes, values, and uncertainty provenance."""
struct SusceptibilityBenchmark{Q,V,U,P,M}
    name::Symbol
    citation::String
    doi::String
    channel::Symbol
    qpoints::Q
    bosonic_indices::Vector{Int}
    values::V
    uncertainty::U
    parameters::P
    metadata::M
end

function SusceptibilityBenchmark(name::Symbol, citation::AbstractString, doi::AbstractString, channel::Symbol, qpoints,
                                 bosonic_indices, values; uncertainty=nothing, parameters=Dict{Symbol,Any}(), metadata=Dict{Symbol,Any}())
    qs = collect(qpoints)
    indices = Int.(collect(bosonic_indices))
    matrix = ComplexF64.(values)
    size(matrix) == (length(qs), length(indices)) || throw(DimensionMismatch("benchmark values must have shape (nq, nbosonic)"))
    errors = isnothing(uncertainty) ? fill(NaN, size(matrix)) : Float64.(uncertainty)
    size(errors) == size(matrix) || throw(DimensionMismatch("benchmark uncertainty must match the value matrix"))
    return SusceptibilityBenchmark(name, String(citation), String(doi), channel, qs, indices, matrix, errors,
                                   Dict{Symbol,Any}(parameters), Dict{Symbol,Any}(metadata))
end

"""Residual report for a susceptibility calculation compared with a peer-reviewed numerical benchmark."""
struct SusceptibilityBenchmarkResult{B,R,N,M}
    benchmark::B
    residuals::R
    normalized_residuals::N
    passed::Bool
    metadata::M
end

function _benchmark_axis_positions(axis::BosonicMatsubaraAxis, indices::AbstractVector{<:Integer})
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    positions = Int[]
    for index in indices
        haskey(lookup, index) || throw(ArgumentError("result bosonic axis does not contain benchmark Matsubara index $index"))
        push!(positions, lookup[index])
    end
    return positions
end

"""
    compare_susceptibility_benchmark(result, benchmark; atol=0, rtol=0, uncertainty_sigma=3)

Compare a `SusceptibilityResult` or `DiagrammaticDMFTResult` against explicitly stored peer-reviewed values. The comparison never digitizes or interpolates published figures internally; reference values and their uncertainty provenance must be supplied when constructing `SusceptibilityBenchmark`.
"""
function compare_susceptibility_benchmark(result::SusceptibilityResult, benchmark::SusceptibilityBenchmark;
                                            atol::Real=0.0, rtol::Real=0.0, uncertainty_sigma::Real=3.0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    uncertainty_sigma > 0 || throw(ArgumentError("uncertainty_sigma must be positive"))
    length(result.qpoints) == length(benchmark.qpoints) || throw(DimensionMismatch("result and benchmark q-point counts differ"))
    result.qpoints == benchmark.qpoints || throw(ArgumentError("result and benchmark q-point labels differ; explicit remapping is required"))
    positions = _benchmark_axis_positions(result.axis, benchmark.bosonic_indices)
    predicted = result.values[:, positions]
    residuals = abs.(predicted .- benchmark.values)
    normalized = fill(NaN, size(residuals))
    passed = true
    for index in eachindex(residuals)
        uncertainty = benchmark.uncertainty[index]
        scale = Float64(atol) + Float64(rtol) * max(abs(predicted[index]), abs(benchmark.values[index]), 1.0)
        if isfinite(uncertainty) && uncertainty > 0
            normalized[index] = residuals[index] / uncertainty
            scale = max(scale, uncertainty_sigma * uncertainty)
        end
        residuals[index] <= scale || (passed = false)
    end
    finite_normalized = filter(isfinite, vec(normalized))
    metadata = Dict{Symbol,Any}(
        :max_absolute_residual => isempty(residuals) ? 0.0 : maximum(residuals),
        :max_normalized_residual => isempty(finite_normalized) ? NaN : maximum(finite_normalized),
        :atol => Float64(atol),
        :rtol => Float64(rtol),
        :uncertainty_sigma => Float64(uncertainty_sigma),
        :benchmark_doi => benchmark.doi,
    )
    return SusceptibilityBenchmarkResult(benchmark, residuals, normalized, passed, metadata)
end

compare_susceptibility_benchmark(result::DiagrammaticDMFTResult, benchmark::SusceptibilityBenchmark; kwargs...) =
    compare_susceptibility_benchmark(susceptibility(result), benchmark; kwargs...)

"""Return the peer-reviewed publications selected as acceptance targets for nonlocal Hubbard-model validation."""
function diagrammatic_dmft_benchmark_references()
    return [
        (name=:gukelberger_2017_dual_fermion, doi="10.1103/PhysRevB.96.035152",
         citation="J. Gukelberger, E. Kozik, and H. Hafermann, Phys. Rev. B 96, 035152 (2017)",
         target=:nonlocal_self_energy_and_convergence_regimes,
         status=:reference_values_not_embedded),
        (name=:leblanc_2019_magnetic_susceptibility, doi="10.1103/PhysRevB.100.075123",
         citation="J. P. F. LeBlanc et al., Phys. Rev. B 100, 075123 (2019)",
         target=:momentum_resolved_magnetic_susceptibility,
         status=:reference_values_not_embedded),
        (name=:leblanc_2015_hubbard_benchmark, doi="10.1103/PhysRevX.5.041041",
         citation="J. P. F. LeBlanc et al., Phys. Rev. X 5, 041041 (2015)",
         target=:independent_hubbard_reference_data,
         status=:reference_values_not_embedded),
    ]
end

"""Peer-reviewed Matsubara self-energy benchmark with explicit momentum/frequency coordinates and uncertainty provenance."""
struct SelfEnergyBenchmark{K,V,U,P,M}
    name::Symbol
    citation::String
    doi::String
    kpoints::K
    fermionic_indices::Vector{Int}
    values::V
    uncertainty::U
    parameters::P
    metadata::M
end

function SelfEnergyBenchmark(name::Symbol, citation::AbstractString, doi::AbstractString, kpoints, fermionic_indices, values;
                             uncertainty=nothing, parameters=Dict{Symbol,Any}(), metadata=Dict{Symbol,Any}())
    ks = collect(kpoints)
    indices = Int.(collect(fermionic_indices))
    matrix = ComplexF64.(values)
    size(matrix) == (length(ks), length(indices)) || throw(DimensionMismatch("self-energy benchmark values must have shape (nk, nfermionic)"))
    errors = isnothing(uncertainty) ? fill(NaN, size(matrix)) : Float64.(uncertainty)
    size(errors) == size(matrix) || throw(DimensionMismatch("self-energy benchmark uncertainty must match the value matrix"))
    return SelfEnergyBenchmark(name, String(citation), String(doi), ks, indices, matrix, errors,
                               Dict{Symbol,Any}(parameters), Dict{Symbol,Any}(metadata))
end

"""Residual report for a momentum-resolved self-energy compared with a peer-reviewed numerical benchmark."""
struct SelfEnergyBenchmarkResult{B,R,N,M}
    benchmark::B
    residuals::R
    normalized_residuals::N
    passed::Bool
    metadata::M
end

function _fermionic_benchmark_positions(axis::FermionicMatsubaraAxis, indices::AbstractVector{<:Integer})
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    return Int[get(lookup, index, 0) == 0 ? throw(ArgumentError("result fermionic axis does not contain benchmark Matsubara index $index")) : lookup[index] for index in indices]
end

function compare_self_energy_benchmark(result::MomentumResolvedSelfEnergy, benchmark::SelfEnergyBenchmark;
                                       atol::Real=0.0, rtol::Real=0.0, uncertainty_sigma::Real=3.0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    uncertainty_sigma > 0 || throw(ArgumentError("uncertainty_sigma must be positive"))
    result.kpoints == benchmark.kpoints || throw(ArgumentError("result and benchmark momentum labels differ; explicit remapping is required"))
    positions = _fermionic_benchmark_positions(result.axis, benchmark.fermionic_indices)
    predicted = result.values[:, positions]
    residuals = abs.(predicted .- benchmark.values)
    normalized = fill(NaN, size(residuals))
    passed = true
    for index in eachindex(residuals)
        uncertainty = benchmark.uncertainty[index]
        scale = Float64(atol) + Float64(rtol) * max(abs(predicted[index]), abs(benchmark.values[index]), 1.0)
        if isfinite(uncertainty) && uncertainty > 0
            normalized[index] = residuals[index] / uncertainty
            scale = max(scale, uncertainty_sigma * uncertainty)
        end
        residuals[index] <= scale || (passed = false)
    end
    finite_normalized = filter(isfinite, vec(normalized))
    metadata = Dict{Symbol,Any}(:max_absolute_residual => isempty(residuals) ? 0.0 : maximum(residuals),
                                :max_normalized_residual => isempty(finite_normalized) ? NaN : maximum(finite_normalized),
                                :atol => Float64(atol), :rtol => Float64(rtol), :uncertainty_sigma => Float64(uncertainty_sigma),
                                :benchmark_doi => benchmark.doi)
    return SelfEnergyBenchmarkResult(benchmark, residuals, normalized, passed, metadata)
end

function compare_self_energy_benchmark(result::LadderDGAResult, benchmark::SelfEnergyBenchmark; kwargs...)
    isnothing(result.self_energy) && throw(ArgumentError("LadderDGAResult does not contain a reconstructed nonlocal self-energy"))
    return compare_self_energy_benchmark(result.self_energy, benchmark; kwargs...)
end

compare_susceptibility_benchmark(result::LadderDGAResult, benchmark::SusceptibilityBenchmark; kwargs...) =
    compare_susceptibility_benchmark(susceptibility(result, benchmark.channel), benchmark; kwargs...)
