# Explicit, one-shot multiorbital ladder workflow. Frequency-box and channel
# approximations remain visible; no unverified high-frequency tail is inferred.

"""Inputs for a one-shot, SU(2)-invariant multiorbital ladder calculation.

`support_fermionic` and `support_bosonic` specify the full two-particle vertex
box; `core_bosonic_indices` specifies the external EOM transfer frequencies.
The external fermionic core is the converged DMFT axis. Every crossed index
required on that core must exist in the support box. A precomputed two-particle
result or expanded lattice Green function can be supplied to avoid recomputation.
"""
struct OrbitalLadderWorkflowProblem{D,F,B,T,G} <: AbstractEmbeddingProblem
    dmft::D
    support_fermionic::F
    support_bosonic::B
    core_bosonic_indices::Vector{Int}
    two_particle::T
    momentum_green::G
end

function OrbitalLadderWorkflowProblem(dmft::MultiOrbitalDMFTResult, support_fermionic::FermionicMatsubaraAxis,
    support_bosonic::BosonicMatsubaraAxis;
    core_bosonic_indices=support_bosonic.indices,
    two_particle=nothing, momentum_green=nothing)
    dmft.converged || throw(ArgumentError("one-shot ladder execution requires a converged DMFT reference"))
    beta = dmft.problem.axis.beta
    all(axis -> isapprox(axis.beta, beta; atol=1e-12 * max(beta, 1.0), rtol=1e-12),
        (support_fermionic, support_bosonic)) || throw(ArgumentError("support Matsubara axes must match the DMFT temperature"))
    core = Int.(collect(core_bosonic_indices))
    !isempty(core) && length(unique(core)) == length(core) || throw(ArgumentError("core bosonic frequencies must be unique and nonempty"))
    all(in(support_bosonic.indices), core) || throw(ArgumentError("core bosonic frequencies are missing from support"))
    external = dmft.problem.axis.indices
    all(in(support_fermionic.indices), external) || throw(ArgumentError("DMFT fermionic core is missing from support"))
    for m in core, np in external, n in external
        (np - m in support_fermionic.indices && np - n in support_bosonic.indices) ||
            throw(ArgumentError("crossed frequency (n=$n, np=$np, m=$m) is missing from the declared support"))
    end
    isnothing(dmft.problem.lattice.qmaps) && throw(ArgumentError("a complete momentum-translation grid is required"))
    return OrbitalLadderWorkflowProblem(dmft, support_fermionic, support_bosonic, core, two_particle, momentum_green)
end

"""Numerical and resource policy for the fully frequency-resolved one-shot ladder workflow.

`four_point_backend` retains the finite imaginary-time quadrature convention.
The workflow does not invent high-frequency vertex values and never labels a
finite-box result as an analytically asymptotic-completed DΓA result.
"""
struct OrbitalLadderWorkflowSolver <: AbstractSolver
    four_point_backend::TauGridFourPoint
    max_two_particle_bytes::Int
    max_operator_bytes::Int
    max_workspace_bytes::Int
    max_retained_bytes::Int
    max_contractions::Int
    regularization::Float64
    pseudoinverse::Bool
    min_relative_pivot::Float64
    spin_tolerance::Float64
end

function OrbitalLadderWorkflowSolver(; four_point_backend::TauGridFourPoint=TauGridFourPoint(),
    max_two_particle_bytes::Integer=536870912,
    max_operator_bytes::Integer=268435456,
    max_workspace_bytes::Integer=536870912,
    max_retained_bytes::Integer=536870912,
    max_contractions::Integer=100_000_000,
    regularization::Real=0.0, pseudoinverse::Bool=false,
    min_relative_pivot::Real=1e-14, spin_tolerance::Real=1e-8)
    all(>(0), (max_two_particle_bytes, max_operator_bytes, max_workspace_bytes, max_retained_bytes, max_contractions)) ||
        throw(ArgumentError("resource limits must all be positive"))
    all(x -> isfinite(x) && x >= 0, (regularization, min_relative_pivot, spin_tolerance)) ||
        throw(ArgumentError("regularization, pivot and symmetry tolerances must be finite and nonnegative"))
    min_relative_pivot < 1 || throw(ArgumentError("min_relative_pivot must be smaller than one"))
    return OrbitalLadderWorkflowSolver(four_point_backend, Int(max_two_particle_bytes), Int(max_operator_bytes),
        Int(max_workspace_bytes), Int(max_retained_bytes), Int(max_contractions), Float64(regularization),
        pseudoinverse, Float64(min_relative_pivot), Float64(spin_tolerance))
end

solver_name(::OrbitalLadderWorkflowSolver) = :orbital_su2_ladder_workflow

"""One-shot ladder output, including all intermediate vertices and a finite-box provenance report."""
struct OrbitalLadderWorkflowResult{P,T,C,V,L,X,S,G}
    problem::P
    two_particle::T
    spin_channels::C
    irreducible_vertices::V
    ladders::L
    crossing_core::X
    nonlocal_self_energy::S
    corrected_green::G
    metadata::Dict{Symbol,Any}
end

"""Re-evaluate the converged finite-bath DMFT self-energy on every shifted support Matsubara index.

The impurity Lehmann solution is reused, with no zero padding or interpolation.
This evaluation is exact for the *finite-bath impurity*, not an analytic
continuation or an exact infinite-frequency two-particle completion.
"""
function orbital_ladder_expanded_green(dmft::MultiOrbitalDMFTResult, faxis::FermionicMatsubaraAxis,
                                        baxis::BosonicMatsubaraAxis; max_output_bytes::Integer=536870912)
    dmft.converged || throw(ArgumentError("expanded lattice Green function requires converged DMFT"))
    isapprox(faxis.beta, dmft.problem.axis.beta; atol=1e-12, rtol=1e-12) &&
        isapprox(baxis.beta, dmft.problem.axis.beta; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("expanded Green and DMFT inverse temperatures differ"))
    indices = sort!(unique(vcat(faxis.indices, [n + m for n in faxis.indices for m in baxis.indices],
        dmft.problem.axis.indices)))
    axis = FermionicMatsubaraAxis(faxis.beta, indices)
    lattice = dmft.problem.lattice
    output_bytes = big(16) * big(length(lattice.labels))^2 * length(lattice.kpoints) * length(axis)
    max_output_bytes > 0 && output_bytes <= max_output_bytes || throw(ArgumentError("expanded Green output exceeds max_output_bytes"))
    impurity = dmft.impurity_result
    p = dmft.problem
    finite_bath = MultiOrbitalImpurityProblem(p.one_body, p.interaction, axis;
        chemical_potential=p.chemical_potential, bath=impurity.bath)
    sigma = self_energy(noninteracting_impurity_green(finite_bath),
        multiorbital_impurity_green(impurity.thermal_state, axis))
    output = momentum_green(lattice, axis, sigma; chemical_potential=p.chemical_potential)
    output.metadata[:expanded_finite_bath_lehmann] = true
    output.metadata[:estimated_output_bytes] = Int(output_bytes)
    return output
end

function _orbital_ladder_workflow_su2(problem::OrbitalLadderWorkflowProblem, tolerance::Float64)
    p = problem.dmft.problem
    local_check = orbital_su2_symmetry(p.one_body, p.interaction; tolerance=tolerance)
    local_check.valid || throw(ArgumentError("local Hamiltonian is not SU(2) invariant"))
    for ik in eachindex(p.lattice.kpoints)
        orbital_su2_symmetry(view(p.lattice.hamiltonians, :, :, ik), p.interaction; tolerance=tolerance).valid ||
            throw(ArgumentError("lattice Bloch Hamiltonian violates SU(2) spin symmetry"))
    end
    return local_check
end

"""Execute impurity four-point → SU(2) projection → matrix ladders → crossed EOM → matrix Dyson.

Every vertex retains the two fermionic frequencies throughout the EOM. This
is a finite-support, one-shot calculation with no self-energy feedback. The
result is not certified for infinite-frequency DΓA unless an independent
vertex-tail completion and convergence analysis are supplied.
"""
function solve(solver::OrbitalLadderWorkflowSolver, problem::OrbitalLadderWorkflowProblem)
    dmft = problem.dmft
    p = dmft.problem
    lattice = p.lattice
    _orbital_ladder_workflow_su2(problem, solver.spin_tolerance)
    nf, nb, n = length(problem.support_fermionic), length(problem.support_bosonic), length(lattice.labels)
    nq = length(lattice.qmaps)
    nk = length(lattice.kpoints)
    core_nf = length(p.axis)
    core_nb = length(problem.core_bosonic_indices)
    estimated_contractions = big(n)^8 * nk^2 * nq * core_nf^2 * core_nb
    estimated_contractions <= solver.max_contractions ||
        throw(ArgumentError("channel EOM estimated contractions exceed max_contractions before impurity evaluation"))
    np = div(n, 2)^2
    retained = big(16) * big(np)^2 * nf^2 * nb * (2 + 2nq)
    retained <= solver.max_retained_bytes || throw(ArgumentError("retained channel vertices exceed max_retained_bytes"))
    !isnothing(problem.two_particle) && !(problem.two_particle isa MultiOrbitalImpurityTwoParticleResult) &&
        throw(ArgumentError("two_particle must be a MultiOrbitalImpurityTwoParticleResult"))
    two = isnothing(problem.two_particle) ? multiorbital_impurity_two_particle(
        dmft.impurity_result.thermal_state, problem.support_fermionic, problem.support_bosonic;
        backend=solver.four_point_backend, max_output_bytes=solver.max_two_particle_bytes,
        max_operator_bytes=solver.max_operator_bytes, max_workspace_bytes=solver.max_workspace_bytes) : problem.two_particle
    _orbital_pair_compatible(two.susceptibility, two.bubble)
    two.susceptibility.fermionic_axis.indices == problem.support_fermionic.indices &&
        two.susceptibility.bosonic_axis.indices == problem.support_bosonic.indices &&
        two.susceptibility.labels == p.labels || throw(ArgumentError("precomputed impurity two-particle data do not match support axes and labels"))
    projected = orbital_impurity_spin_channels(two; compute_vertices=true, atol=solver.spin_tolerance,
        rtol=solver.spin_tolerance, max_output_bytes=solver.max_two_particle_bytes,
        vertex_kwargs=(; regularization=solver.regularization, pseudoinverse=solver.pseudoinverse,
            max_inversion_bytes=solver.max_workspace_bytes))
    local_asymmetry = norm(projected.bubble.density.values - projected.bubble.magnetic.values) /
        max(norm(projected.bubble.density.values), norm(projected.bubble.magnetic.values), eps(Float64))
    local_asymmetry <= solver.spin_tolerance ||
        throw(ArgumentError("local density/magnetic Wick bubbles are inconsistent with SU(2)"))
    green = isnothing(problem.momentum_green) ? orbital_ladder_expanded_green(dmft, problem.support_fermionic,
        problem.support_bosonic; max_output_bytes=solver.max_retained_bytes) : problem.momentum_green
    green isa MatrixMomentumResolvedGreenFunction || throw(ArgumentError("momentum_green must be a MatrixMomentumResolvedGreenFunction"))
    green.labels == p.labels && green.kpoints == lattice.kpoints ||
        throw(ArgumentError("expanded Green function and lattice spin-orbital/momentum labels differ"))
    get(green.metadata, :expanded_finite_bath_lehmann, false) === true ||
        throw(ArgumentError("the supplied Green function must carry finite-bath Lehmann provenance"))
    maps, shifts = _orbital_ladder_maps(green, lattice.qmaps, problem.support_fermionic, problem.support_bosonic)
    spin_pairs = [(a, b) for a in 1:n for b in 1:n]
    spatial_bubbles = OrbitalPairSusceptibility[]
    maximum_bubble_asymmetry = 0.0
    for mapping in maps
        bubble = _orbital_ladder_bubble_at(green, mapping, problem.support_fermionic,
            problem.support_bosonic, spin_pairs, shifts)
        channels = orbital_impurity_spin_channels(bubble; atol=solver.spin_tolerance,
            rtol=solver.spin_tolerance, max_output_bytes=solver.max_two_particle_bytes)
        deviation = norm(channels.density.values - channels.magnetic.values)
        scale = max(norm(channels.density.values), norm(channels.magnetic.values), eps(Float64))
        maximum_bubble_asymmetry = max(maximum_bubble_asymmetry, deviation / scale)
        maximum_bubble_asymmetry <= solver.spin_tolerance ||
            throw(ArgumentError("lattice density/magnetic Wick bubbles differ beyond the SU(2) tolerance"))
        push!(spatial_bubbles, channels.density)
    end
    ladder = orbital_spin_ladders(projected.bubble.density, projected.density_vertex,
        projected.magnetic_vertex, spatial_bubbles, lattice.qmaps;
        spin_symmetry_certified=true, max_workspace_bytes=solver.max_workspace_bytes,
        max_retained_bytes=solver.max_retained_bytes, min_relative_pivot=solver.min_relative_pivot)
    core = OrbitalCrossingCore(ladder, p.axis.indices, problem.core_bosonic_indices;
        max_lookup_bytes=solver.max_workspace_bytes)
    sigma = orbital_channel_ladder_self_energy(dmft, green, core;
        max_workspace_bytes=solver.max_workspace_bytes, max_output_bytes=solver.max_retained_bytes,
        max_contractions=solver.max_contractions, symmetry_tolerance=solver.spin_tolerance)
    corrected = dyson_momentum_green(lattice, sigma; chemical_potential=p.chemical_potential)
    metadata = Dict{Symbol,Any}(:construction => :one_shot_orbital_su2_ladder,
        :dmft_reference_converged => true, :frequency_core_indices => copy(p.axis.indices),
        :core_bosonic_indices => copy(problem.core_bosonic_indices),
        :support_fermionic_indices => copy(problem.support_fermionic.indices),
        :support_bosonic_indices => copy(problem.support_bosonic.indices),
        :impurity_quadrature => get(two.diagnostics, :backend, :precomputed),
        :impurity_ntime => get(two.diagnostics, :ntime, missing),
        :bubble_spin_asymmetry => maximum_bubble_asymmetry,
        :local_bubble_spin_asymmetry => local_asymmetry,
        :transverse_spin_ward_identity_certified => false,
        :crossing_support_complete => true,
        :local_double_counting_scheme => :matched_local_eom_diagram,
        :high_frequency_completion => :not_certified,
        :full_multiorbital_dga => false,
        :end_to_end_forward_integrated => true,
        :self_energy_feedback => false,
        :estimated_retained_vertex_bytes => Int(retained),
        :estimated_eom_contractions => estimated_contractions)
    return OrbitalLadderWorkflowResult(problem, two, projected, (density=projected.density_vertex,
        magnetic=projected.magnetic_vertex), ladder, core, sigma, corrected, metadata)
end

"""Compare two completed ladder calculations on an identical physical core.

The second run must use a strictly larger nested two-particle support grid.
This is empirical box-convergence evidence, *not* an analytic tail-error bound.
"""
function orbital_ladder_support_convergence(smaller::OrbitalLadderWorkflowResult,
    larger::OrbitalLadderWorkflowResult;
    atol::Real=1e-10, rtol::Real=1e-6)
    smaller.problem.dmft === larger.problem.dmft ||
        throw(ArgumentError("support-convergence runs must share one converged DMFT reference"))
    a, b = smaller.metadata, larger.metadata
    a[:frequency_core_indices] == b[:frequency_core_indices] &&
        a[:core_bosonic_indices] == b[:core_bosonic_indices] ||
        throw(ArgumentError("support convergence requires identical physical frequency cores"))
    smaller.problem.dmft.problem.labels == larger.problem.dmft.problem.labels &&
        smaller.problem.dmft.problem.lattice.kpoints == larger.problem.dmft.problem.lattice.kpoints ||
        throw(ArgumentError("support convergence requires identical orbital and momentum labels"))
    issubset(a[:support_fermionic_indices], b[:support_fermionic_indices]) &&
        issubset(a[:support_bosonic_indices], b[:support_bosonic_indices]) ||
        throw(ArgumentError("larger support must contain every smaller-support frequency"))
    (length(b[:support_fermionic_indices]) > length(a[:support_fermionic_indices]) ||
        length(b[:support_bosonic_indices]) > length(a[:support_bosonic_indices])) ||
        throw(ArgumentError("second support box must be strictly larger"))
    all(x -> isfinite(x) && x >= 0, (atol, rtol)) || throw(ArgumentError("invalid convergence tolerances"))
    difference = smaller.nonlocal_self_energy.values - larger.nonlocal_self_energy.values
    absolute = maximum(abs, difference)
    reference = max(maximum(abs, smaller.nonlocal_self_energy.values),
        maximum(abs, larger.nonlocal_self_energy.values))
    bound = atol + rtol * reference
    return (converged=absolute <= bound, maximum_absolute_difference=absolute,
        relative_residual=norm(difference) / max(norm(smaller.nonlocal_self_energy.values),
            norm(larger.nonlocal_self_energy.values), eps(Float64)), tolerance=bound,
        high_frequency_tail_certified=false, local_subtraction=:matched_local_eom_diagram,
        compared_external_frequency_count=length(smaller.problem.dmft.problem.axis),
        compared_momentum_count=length(smaller.problem.dmft.problem.lattice.kpoints))
end
