# Ladder dynamical-vertex extension of a converged single-site DMFT solution.

include("MomentumLattice.jl")
include("SchwingerDyson.jl")

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

include("Benchmarking.jl")
