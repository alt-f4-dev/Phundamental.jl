# Independent two-axis convergence diagnostics for finite-support ladder calculations.
# A nested box test and a quadrature test never certify the uncomputed vertex asymptotics.

"""Four-run numerical convergence report with a complete fine-grid reference and stage diagnostics."""
struct OrbitalLadderConvergenceReport{R}
    final_result::R
    frequency_comparison::NamedTuple
    quadrature_comparison::NamedTuple
    coupled_difference::NamedTuple
    stages::Vector{NamedTuple}
    metadata::Dict{Symbol,Any}
end

function _orbital_ladder_resolution(ntime::Int, problem::OrbitalLadderWorkflowProblem)
    shifted = (n + m for n in problem.support_fermionic.indices for m in problem.support_bosonic.indices)
    maximum_odd_frequency = maximum(abs(2n + 1) for n in Iterators.flatten((problem.support_fermionic.indices, shifted)))
    return (maximum_odd_frequency=maximum_odd_frequency, ntime=ntime,
        below_midpoint_nyquist=maximum_odd_frequency < ntime)
end

function _orbital_ladder_with_ntime(solver::OrbitalLadderWorkflowSolver, ntime::Int)
    backend = solver.four_point_backend
    new_backend = TauGridFourPoint(; ntime=ntime, max_dimension=backend.max_dimension,
        subtract_disconnected=backend.subtract_disconnected, algorithm=backend.algorithm,
        parallel=backend.parallel, equal_time=backend.equal_time)
    return OrbitalLadderWorkflowSolver(; four_point_backend=new_backend,
        max_two_particle_bytes=solver.max_two_particle_bytes, max_operator_bytes=solver.max_operator_bytes,
        max_workspace_bytes=solver.max_workspace_bytes, max_retained_bytes=solver.max_retained_bytes,
        max_contractions=solver.max_contractions, regularization=solver.regularization,
        pseudoinverse=solver.pseudoinverse, min_relative_pivot=solver.min_relative_pivot,
        spin_tolerance=solver.spin_tolerance)
end

function _orbital_ladder_array_comparison(a::AbstractArray, b::AbstractArray, atol::Float64, rtol::Float64)
    size(a) == size(b) || throw(DimensionMismatch("comparison arrays require identical physical cores"))
    delta = a .- b
    maximum_difference = maximum(abs, delta)
    scale = max(maximum(abs, a), maximum(abs, b))
    relative = norm(delta) / max(norm(a), norm(b), eps(Float64))
    bound = atol + rtol * scale
    return (converged=maximum_difference <= bound, maximum_absolute_difference=maximum_difference,
        relative_residual=relative, tolerance=bound)
end

"""Resolve Matsubara-support and imaginary-time quadrature errors independently.

Executes the rectangular grid `(small_support, large_support) × (coarse_ntime, fine_ntime)`.
Only the final full workflow result is retained; the other runs retain self-energy
arrays and timing/allocation summaries. The support and quadrature residuals are
*empirical*, not certified error bounds on the missing high-frequency vertex.

`ntime_coarse` and `ntime_fine` must place every single-particle Fourier factor
below the midpoint-grid Nyquist frequency, avoiding unreported Fourier aliasing.
"""
function orbital_ladder_convergence_study(solver::OrbitalLadderWorkflowSolver,
                                            smaller::OrbitalLadderWorkflowProblem,
                                            larger::OrbitalLadderWorkflowProblem;
                                            ntime_coarse::Integer, ntime_fine::Integer,
                                            atol::Real=1e-9, rtol::Real=1e-5,
                                            max_total_contractions::Integer=400_000_000,
                                            max_retained_output_bytes::Integer=268435456)
    smaller.dmft === larger.dmft || throw(ArgumentError("support runs must use the identical converged DMFT result"))
    smaller.core_bosonic_indices == larger.core_bosonic_indices ||
        throw(ArgumentError("support runs must have identical physical bosonic cores"))
    small_f = smaller.support_fermionic.indices
    large_f = larger.support_fermionic.indices
    small_b = smaller.support_bosonic.indices
    large_b = larger.support_bosonic.indices
    issubset(small_f, large_f) && issubset(small_b, large_b) &&
        (length(large_f) > length(small_f) || length(large_b) > length(small_b)) ||
        throw(ArgumentError("the second support must be a strictly larger nested Matsubara box"))
    isnothing(smaller.two_particle) && isnothing(larger.two_particle) ||
        throw(ArgumentError("quadrature studies must independently recompute impurity four-point data"))
    all(x -> isfinite(x) && x >= 0, (atol, rtol)) || throw(ArgumentError("invalid convergence tolerances"))
    ntime_coarse >= 4 && ntime_fine > ntime_coarse ||
        throw(ArgumentError("require 4 <= ntime_coarse < ntime_fine"))
    max_total_contractions > 0 && max_retained_output_bytes > 0 ||
        throw(ArgumentError("total resource limits must be positive"))
    resolution = _orbital_ladder_resolution(Int(ntime_coarse), larger)
    resolution.below_midpoint_nyquist ||
        throw(ArgumentError("coarse imaginary-time grid aliases a requested Matsubara mode; increase ntime_coarse"))
    p = smaller.dmft.problem
    lattice = p.lattice
    n = length(lattice.labels)
    estimated_each = big(n)^8 * length(lattice.kpoints)^2 * length(lattice.qmaps) *
        length(p.axis)^2 * length(smaller.core_bosonic_indices)
    4 * estimated_each <= max_total_contractions ||
        throw(ArgumentError("four-run convergence study exceeds max_total_contractions"))
    output_bytes = big(16) * big(n)^2 * length(lattice.kpoints) * length(p.axis)
    4 * output_bytes <= max_retained_output_bytes ||
        throw(ArgumentError("convergence comparison arrays exceed max_retained_output_bytes"))
    coarse_solver = _orbital_ladder_with_ntime(solver, Int(ntime_coarse))
    fine_solver = _orbital_ladder_with_ntime(solver, Int(ntime_fine))
    stages = NamedTuple[]
    function run_stage(name::Symbol, current_solver::OrbitalLadderWorkflowSolver, problem::OrbitalLadderWorkflowProblem)
        measured = @timed solve(current_solver, problem)
        result = measured.value
        push!(stages, (name=name, time_seconds=measured.time, allocated_bytes=measured.bytes,
            ntime=current_solver.four_point_backend.ntime,
            fermionic_count=length(problem.support_fermionic), bosonic_count=length(problem.support_bosonic),
            estimated_eom_contractions=result.metadata[:estimated_eom_contractions]))
        return result
    end
    small_coarse_result = run_stage(:small_coarse, coarse_solver, smaller)
    small_coarse = copy(small_coarse_result.nonlocal_self_energy.values)
    small_coarse_result = nothing
    small_fine_result = run_stage(:small_fine, fine_solver, smaller)
    small_fine = copy(small_fine_result.nonlocal_self_energy.values)
    small_fine_result = nothing
    large_coarse_result = run_stage(:large_coarse, coarse_solver, larger)
    large_coarse = copy(large_coarse_result.nonlocal_self_energy.values)
    large_coarse_result = nothing
    final_result = run_stage(:large_fine, fine_solver, larger)
    large_fine = final_result.nonlocal_self_energy.values
    absolute_tolerance, relative_tolerance = Float64(atol), Float64(rtol)
    frequency = _orbital_ladder_array_comparison(small_fine, large_fine, absolute_tolerance, relative_tolerance)
    quadrature = _orbital_ladder_array_comparison(large_coarse, large_fine, absolute_tolerance, relative_tolerance)
    difference = (large_fine .- large_coarse) .- (small_fine .- small_coarse)
    interaction = (maximum_absolute_difference=maximum(abs, difference),
        relative_residual=norm(difference) / max(norm(large_fine), eps(Float64)))
    metadata = Dict{Symbol,Any}(:method => :nested_support_and_quadrature,
        :finite_box_numerically_converged => frequency.converged && quadrature.converged,
        :midpoint_aliasing_preflight_passed => true, :one_particle_analytic_tail_applied => false,
        :two_particle_vertex_tail_certified => false, :full_multiorbital_dga => false,
        :total_allocated_bytes => sum(stage.allocated_bytes for stage in stages),
        :total_time_seconds => sum(stage.time_seconds for stage in stages),
        :peak_resident_memory_measured => false)
    return OrbitalLadderConvergenceReport(final_result, frequency, quadrature, interaction, stages, metadata)
end
