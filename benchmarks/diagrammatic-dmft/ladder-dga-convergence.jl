#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Phundamental

benchmark_int(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))
benchmark_float(name::AbstractString, default::Real) = parse(Float64, get(ENV, name, string(default)))
benchmark_bool(name::AbstractString, default::Bool) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
benchmark_symbol(name::AbstractString, default::Symbol) = Symbol(lowercase(get(ENV, name, String(default))))

function square_dispersion(t::Float64, tprime::Float64)
    return k -> -2t * (cos(k[1]) + cos(k[2])) - 4tprime * cos(k[1]) * cos(k[2])
end

function solve_square_hubbard_dmft(; U::Real, beta::Real, chemical_potential::Real, t::Real=1.0, tprime::Real=0.0,
                                   shape::Tuple{Int,Int}=(8, 8), nfermionic::Integer=12, bath_sites::Integer=2,
                                   bath_fit_maxiter::Integer=300, bath_fit_tol::Real=1e-8, bath_fit_multistart::Integer=5,
                                   dmft_tol::Real=1e-7, dmft_maxiter::Integer=80, mixing::Real=0.5)
    U = Float64(U)
    beta = Float64(beta)
    chemical_potential = Float64(chemical_potential)
    t = Float64(t)
    tprime = Float64(tprime)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    nfermionic > 1 || throw(ArgumentError("nfermionic must exceed one"))
    lattice = periodic_momentum_lattice(shape, square_dispersion(t, tprime))
    fermionic_indices = collect(-Int(nfermionic):Int(nfermionic)-1)
    axis = FermionicMatsubaraAxis(beta, fermionic_indices)
    bandwidth = 8abs(t) + 8abs(tprime) + max(abs(U), 1.0)
    fit_options = BathFitOptions(maxiter=bath_fit_maxiter, tol=bath_fit_tol, energy_window=(-bandwidth, bandwidth), multistart=bath_fit_multistart)
    impurity_solver = EDImpuritySolver(bath_sites=bath_sites, bath_fit=fit_options, backend=:dense)
    dmft_problem = DMFTProblem(lattice, U; temperature=inv(beta), chemical_potential=chemical_potential, nfrequencies=nfermionic, axis=axis)
    dmft_solver = DMFTSolver(impurity_solver=impurity_solver, mixing=LinearMixing(mixing), tol=dmft_tol, maxiter=dmft_maxiter, verify_closure=true)
    dmft = solve(dmft_solver, dmft_problem)
    dmft.converged || throw(ErrorException("DMFT reference did not converge: termination=$(get(dmft.metadata, :termination, :unknown)), residual=$(dmft.residual)"))
    return (dmft=dmft, lattice=lattice, axis=axis)
end

function solve_square_hubbard_ladder_dga(; U::Real, beta::Real, chemical_potential::Real, t::Real=1.0, tprime::Real=0.0,
                                         shape::Tuple{Int,Int}=(8, 8), nfermionic::Integer=12, nbosonic::Integer=4,
                                         bath_sites::Integer=2, bath_fit_maxiter::Integer=300, bath_fit_tol::Real=1e-8,
                                         bath_fit_multistart::Integer=5, dmft_tol::Real=1e-7, dmft_maxiter::Integer=80,
                                         mixing::Real=0.5, tau_points::Integer=64, max_impurity_dimension::Integer=256,
                                         regularization::Real=0.0, inverse_rtol::Real=sqrt(eps(Float64)), inverse_atol::Real=0.0,
                                         pseudoinverse::Bool=true, reconstruct_self_energy::Bool=true, prepared_reference=nothing,
                                         prepared_impurity=nothing, parallel::Symbol=:serial, tau_algorithm::Symbol=:factored,
                                         tau_equal_time::Symbol=:symmetric)
    nbosonic > 0 || throw(ArgumentError("nbosonic must be positive"))
    reference = isnothing(prepared_reference) ?
        solve_square_hubbard_dmft(U=U, beta=beta, chemical_potential=chemical_potential, t=t, tprime=tprime, shape=shape, nfermionic=nfermionic,
                                  bath_sites=bath_sites, bath_fit_maxiter=bath_fit_maxiter, bath_fit_tol=bath_fit_tol,
                                  bath_fit_multistart=bath_fit_multistart, dmft_tol=dmft_tol, dmft_maxiter=dmft_maxiter, mixing=mixing) : prepared_reference
    bosonic_axis = BosonicMatsubaraAxis(beta, collect(0:Int(nbosonic)-1))
    backend = TauGridFourPoint(ntime=tau_points, max_dimension=max_impurity_dimension, algorithm=tau_algorithm, parallel=parallel,
                               equal_time=tau_equal_time)
    problem = LadderDGAProblem(reference.dmft, reference.lattice, bosonic_axis; impurity_backend=backend, impurity_two_particle=prepared_impurity)
    method = LadderDGA(regularization=regularization, inverse_rtol=inverse_rtol, inverse_atol=inverse_atol, pseudoinverse=pseudoinverse,
                       reconstruct_self_energy=reconstruct_self_energy, require_complete_frequency_box=false)
    result = solve(DiagrammaticDMFTSolver(method=method, parallel=parallel), problem)
    diagnostics = validate_diagrammatic_dmft_result(result; require_self_energy=reconstruct_self_energy, require_complete_frequency_box=false)
    diagnostics.valid || throw(ErrorException("ladder-DΓA result failed internal validation: $diagnostics"))
    return result
end

function nearest_point_index(points, target; atol::Real=1e-10)
    target_tuple = Tuple(Float64.(collect(target)))
    best_index = 0
    best_distance = Inf
    for (index, point) in enumerate(points)
        point_tuple = Tuple(Float64.(collect(point)))
        length(point_tuple) == length(target_tuple) || continue
        distance = maximum(abs(point_tuple[i] - target_tuple[i]) for i in eachindex(target_tuple))
        if distance < best_distance
            best_distance = distance
            best_index = index
        end
    end
    best_index > 0 && best_distance <= atol || throw(ArgumentError("no momentum point matches $target within atol=$atol; nearest distance=$best_distance"))
    return best_index
end

function axis_position(indices::AbstractVector{<:Integer}, target::Integer)
    position = findfirst(==(Int(target)), indices)
    isnothing(position) && throw(ArgumentError("frequency index $target is absent from the calculated axis"))
    return position
end

function select_self_energy(result::LadderDGAResult, kpoints, fermionic_indices; momentum_atol::Real=1e-10)
    isnothing(result.self_energy) && throw(ArgumentError("ladder-DΓA result contains no reconstructed self-energy"))
    values = Matrix{ComplexF64}(undef, length(kpoints), length(fermionic_indices))
    for (ik, kpoint) in enumerate(kpoints), (iν, index) in enumerate(fermionic_indices)
        source_k = nearest_point_index(result.self_energy.kpoints, kpoint; atol=momentum_atol)
        source_ν = axis_position(result.self_energy.axis.indices, index)
        values[ik, iν] = result.self_energy.values[source_k, source_ν]
    end
    return values
end

function select_susceptibility(result::LadderDGAResult, channel::Symbol, qpoints, bosonic_indices; momentum_atol::Real=1e-10)
    observable = susceptibility(result, channel)
    qgrid = hasproperty(observable, :qpoints) ? getproperty(observable, :qpoints) : getproperty(observable, :q)
    values = Matrix{ComplexF64}(undef, length(qpoints), length(bosonic_indices))
    for (iq, qpoint) in enumerate(qpoints), (iΩ, index) in enumerate(bosonic_indices)
        source_q = nearest_point_index(qgrid, qpoint; atol=momentum_atol)
        source_Ω = axis_position(observable.axis.indices, index)
        values[iq, iΩ] = observable.values[source_q, source_Ω]
    end
    return values
end

function select_two_particle(object, fermionic_indices, bosonic_indices)
    values = Array{ComplexF64}(undef, length(fermionic_indices), length(fermionic_indices), length(bosonic_indices))
    for (iν, ν) in enumerate(fermionic_indices), (iνp, νp) in enumerate(fermionic_indices), (iΩ, Ω) in enumerate(bosonic_indices)
        source_ν = axis_position(object.fermionic_axis.indices, ν)
        source_νp = axis_position(object.fermionic_axis.indices, νp)
        source_Ω = axis_position(object.bosonic_axis.indices, Ω)
        values[iν, iνp, iΩ] = object.values[source_ν, source_νp, source_Ω]
    end
    return values
end

function relative_norm_change(previous, current)
    size(previous) == size(current) || throw(DimensionMismatch("compared arrays must have equal sizes"))
    numerator = norm(current .- previous)
    denominator = max(norm(previous), norm(current), eps(Float64))
    return Float64(numerator / denominator)
end

function tau_harmonic_requirement(nfermionic::Integer, nbosonic::Integer)
    nfermionic > 0 || throw(ArgumentError("nfermionic must be positive"))
    nbosonic > 0 || throw(ArgumentError("nbosonic must be positive"))
    return 2Int(nfermionic) + 2Int(nbosonic) - 3
end

function tau_grid_resolved(tau_points::Integer, nfermionic::Integer, nbosonic::Integer)
    return Int(tau_points) > tau_harmonic_requirement(nfermionic, nbosonic)
end

function inverse_diagnostic_summary(result::LadderDGAResult, channel::Symbol)
    vertex = result.local_vertices[channel]
    vertex_diagnostics = get(vertex.metadata, :inverse_diagnostics, NamedTuple[])
    local_conditions = Float64[]
    local_rank_fractions = Float64[]
    for diagnostic in vertex_diagnostics
        for inverse_diagnostic in (diagnostic.bubble, diagnostic.susceptibility)
            push!(local_conditions, Float64(inverse_diagnostic.condition_number))
            push!(local_rank_fractions, inverse_diagnostic.dimension == 0 ? 0.0 : inverse_diagnostic.numerical_rank / inverse_diagnostic.dimension)
        end
    end
    kernel_conditions = Float64[]
    kernel_rank_fractions = Float64[]
    for susceptibility_value in result.generalized_susceptibilities[channel]
        diagnostics = get(susceptibility_value.metadata, :inverse_diagnostics, NamedTuple[])
        for diagnostic in diagnostics
            push!(kernel_conditions, Float64(diagnostic.kernel.condition_number))
            dimension = diagnostic.kernel.dimension
            push!(kernel_rank_fractions, dimension == 0 ? 0.0 : diagnostic.kernel.numerical_rank / dimension)
        end
    end
    maximum_local_condition = isempty(local_conditions) ? NaN : maximum(local_conditions)
    maximum_kernel_condition = isempty(kernel_conditions) ? NaN : maximum(kernel_conditions)
    minimum_local_rank_fraction = isempty(local_rank_fractions) ? NaN : minimum(local_rank_fractions)
    minimum_kernel_rank_fraction = isempty(kernel_rank_fractions) ? NaN : minimum(kernel_rank_fractions)
    return (maximum_local_condition=maximum_local_condition, maximum_kernel_condition=maximum_kernel_condition,
            minimum_local_rank_fraction=minimum_local_rank_fraction, minimum_kernel_rank_fraction=minimum_kernel_rank_fraction)
end

# Controlled internal-convergence benchmark for the one-shot ladder-DΓA implementation. Each numerical study varies one control at a time. Observable convergence is gated by a finite residual envelope rather than strict monotonic contraction. Local-vertex convergence is tracked and gated separately where it is informative. This remains an internal convergence benchmark, not a replacement for external literature validation.

const U = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_U", 4.0)
const BETA = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_BETA", 2.0)
const T_HOP = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_T", 1.0)
const TPRIME = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_TPRIME", 0.0)
const CHEMICAL_POTENTIAL = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_MU", U / 2)
const TOLERANCE = benchmark_float("PHUNDAMENTAL_DGA_CONVERGENCE_TOL", 5e-2)
const VERTEX_TOLERANCE = benchmark_float("PHUNDAMENTAL_DGA_VERTEX_CONVERGENCE_TOL", TOLERANCE)
const ENVELOPE_STEPS = benchmark_int("PHUNDAMENTAL_DGA_ENVELOPE_STEPS", 2)
const REPORT_CONTRACTION = benchmark_bool("PHUNDAMENTAL_DGA_REQUIRE_CONTRACTION", false)
const GATE_VERTEX_CONVERGENCE = benchmark_bool("PHUNDAMENTAL_DGA_GATE_VERTEX_CONVERGENCE", true)
const PROFILE = benchmark_symbol("PHUNDAMENTAL_DGA_BENCHMARK_PROFILE", :convergence)
PROFILE in (:quick, :convergence, :full) || throw(ArgumentError("PHUNDAMENTAL_DGA_BENCHMARK_PROFILE must be quick, convergence, or full"))
const PARALLEL = benchmark_symbol("PHUNDAMENTAL_DGA_PARALLEL", :serial)
PARALLEL in (:serial, :threads) || throw(ArgumentError("PHUNDAMENTAL_DGA_PARALLEL must be serial or threads"))
const TAU_ALGORITHM = benchmark_symbol("PHUNDAMENTAL_DGA_TAU_ALGORITHM", :factored)
TAU_ALGORITHM in (:factored, :direct) || throw(ArgumentError("PHUNDAMENTAL_DGA_TAU_ALGORITHM must be factored or direct"))
const TAU_EQUAL_TIME = benchmark_symbol("PHUNDAMENTAL_DGA_TAU_EQUAL_TIME", :symmetric)
TAU_EQUAL_TIME in (:symmetric, :ordered) || throw(ArgumentError("PHUNDAMENTAL_DGA_TAU_EQUAL_TIME must be symmetric or ordered"))
const REPORT_TIMING = benchmark_bool("PHUNDAMENTAL_DGA_REPORT_TIMING", true)
const OUTPUT_MODE = benchmark_symbol("PHUNDAMENTAL_DGA_OUTPUT", :compact)
OUTPUT_MODE in (:compact, :verbose) || throw(ArgumentError("PHUNDAMENTAL_DGA_OUTPUT must be compact or verbose"))
const VERBOSE_OUTPUT = OUTPUT_MODE === :verbose
const VALIDATE_PARALLEL = benchmark_bool("PHUNDAMENTAL_DGA_VALIDATE_PARALLEL", PARALLEL === :threads && TAU_ALGORITHM === :factored)
const PARALLEL_VALIDATION_TOL = benchmark_float("PHUNDAMENTAL_DGA_PARALLEL_VALIDATION_TOL", 1e-10)
const RUN_BATH_STUDY = benchmark_bool("PHUNDAMENTAL_DGA_RUN_BATH_STUDY", PROFILE === :full)
const GATE_BATH_STUDY = benchmark_bool("PHUNDAMENTAL_DGA_GATE_BATH_STUDY", false)
const RUN_INVERSION_STUDY = benchmark_bool("PHUNDAMENTAL_DGA_RUN_INVERSION_STUDY", PROFILE === :full)

ENVELOPE_STEPS > 0 || throw(ArgumentError("PHUNDAMENTAL_DGA_ENVELOPE_STEPS must be positive"))

const BATH_FIT_MAXITER = benchmark_int("PHUNDAMENTAL_DGA_BATH_FIT_MAXITER", 300)
const BATH_FIT_TOL = benchmark_float("PHUNDAMENTAL_DGA_BATH_FIT_TOL", 1e-8)
const BATH_FIT_MULTISTART = benchmark_int("PHUNDAMENTAL_DGA_BATH_FIT_MULTISTART", 5)
const DMFT_TOL = benchmark_float("PHUNDAMENTAL_DGA_DMFT_TOL", 1e-7)
const DMFT_MAXITER = benchmark_int("PHUNDAMENTAL_DGA_DMFT_MAXITER", 80)
const MIXING = benchmark_float("PHUNDAMENTAL_DGA_MIXING", 0.5)
const REGULARIZATION = benchmark_float("PHUNDAMENTAL_DGA_REGULARIZATION", 0.0)
const INVERSE_RTOL = benchmark_float("PHUNDAMENTAL_DGA_INVERSE_RTOL", sqrt(eps(Float64)))
const INVERSE_ATOL = benchmark_float("PHUNDAMENTAL_DGA_INVERSE_ATOL", 0.0)
const PSEUDOINVERSE = benchmark_bool("PHUNDAMENTAL_DGA_PSEUDOINVERSE", true)
const BASE_TAU_POINTS = benchmark_int("PHUNDAMENTAL_DGA_BASE_TAU_POINTS", 96)
const BASE_NFERMIONIC = benchmark_int("PHUNDAMENTAL_DGA_BASE_NFERMIONIC", 10)
const BASE_NBOSONIC = benchmark_int("PHUNDAMENTAL_DGA_BASE_NBOSONIC", 4)

const BASE = (
    shape=(8, 8),
    nfermionic=BASE_NFERMIONIC,
    nbosonic=BASE_NBOSONIC,
    bath_sites=2,
    tau_points=BASE_TAU_POINTS,
    max_impurity_dimension=256,
    regularization=REGULARIZATION,
    inverse_rtol=INVERSE_RTOL,
    inverse_atol=INVERSE_ATOL,
    pseudoinverse=PSEUDOINVERSE,
)

const QPOINTS = [(0.0, 0.0), (π, 0.0), (π, π)]
const BOSONIC_INDICES = [0, 1]
const KPOINTS = [(0.0, 0.0), (π, 0.0), (π, π)]
const FERMIONIC_INDICES = [-1, 0, 1]
const TWO_PARTICLE_FERMIONIC_INDICES = [-2, -1, 0, 1]
const TWO_PARTICLE_BOSONIC_INDICES = [0, 1]

const STUDIES = if PROFILE === :quick
    [
        (name=:momentum, label="momentum grid", levels=[(8, 8), (10, 10), (12, 12)], gate=true, vertex_gate=true),
        (name=:fermionic, label="fermionic Matsubara box", levels=[10, 12, 14], gate=true, vertex_gate=true),
        (name=:bosonic, label="bosonic Matsubara box", levels=[3, 4, 6], gate=true, vertex_gate=false),
        (name=:tau, label="four-point τ grid", levels=[48, 64, 96], gate=true, vertex_gate=true),
    ]
elseif PROFILE === :convergence
    [
        (name=:momentum, label="momentum grid", levels=[(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)], gate=true, vertex_gate=true),
        (name=:fermionic, label="fermionic Matsubara box", levels=[6, 8, 10, 12, 14], gate=true, vertex_gate=true),
        (name=:bosonic, label="bosonic Matsubara box", levels=[2, 3, 4, 6], gate=true, vertex_gate=false),
        (name=:tau, label="four-point τ grid", levels=[48, 64, 96, 128], gate=true, vertex_gate=true),
    ]
else
    [
        (name=:momentum, label="momentum grid", levels=[(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)], gate=true, vertex_gate=true),
        (name=:fermionic, label="fermionic Matsubara box", levels=[6, 8, 10, 12, 14], gate=true, vertex_gate=true),
        (name=:bosonic, label="bosonic Matsubara box", levels=[2, 3, 4, 6], gate=true, vertex_gate=false),
        (name=:tau, label="four-point τ grid", levels=[32, 48, 64, 96, 128, 192, 256], gate=true, vertex_gate=true),
    ]
end

struct Observation
    magnetic::Matrix{ComplexF64}
    charge::Matrix{ComplexF64}
    self_energy::Matrix{ComplexF64}
    impurity_magnetic::Array{ComplexF64,3}
    impurity_charge::Array{ComplexF64,3}
    vertex_magnetic::Array{ComplexF64,3}
    vertex_charge::Array{ComplexF64,3}
    local_condition_magnetic::Float64
    local_condition_charge::Float64
    kernel_condition_magnetic::Float64
    kernel_condition_charge::Float64
    local_rank_fraction_magnetic::Float64
    local_rank_fraction_charge::Float64
    kernel_rank_fraction_magnetic::Float64
    kernel_rank_fraction_charge::Float64
    missing_frequency_entries::Int
    missing_frequency_fraction::Float64
    frequency_box_complete::Bool
    probe_frequency_core_complete::Bool
end

struct Comparison
    magnetic::Float64
    charge::Float64
    self_energy::Float64
    impurity_magnetic::Float64
    impurity_charge::Float64
    vertex_magnetic::Float64
    vertex_charge::Float64
    observable_maximum::Float64
    vertex_maximum::Float64
    internal_maximum::Float64
end

function parameter_set(study::Symbol, level)
    if study === :momentum
        return merge(BASE, (shape=level,))
    elseif study === :fermionic
        return merge(BASE, (nfermionic=level,))
    elseif study === :bosonic
        return merge(BASE, (nbosonic=level,))
    elseif study === :tau
        return merge(BASE, (tau_points=level,))
    elseif study === :bath
        return merge(BASE, (bath_sites=level,))
    elseif study === :inverse_rtol
        return merge(BASE, (inverse_rtol=Float64(level),))
    end
    throw(ArgumentError("unknown convergence study: $study"))
end

function cache_key(parameters)
    return (parameters.shape, parameters.nfermionic, parameters.nbosonic, parameters.bath_sites, parameters.tau_points,
            parameters.max_impurity_dimension, parameters.regularization, parameters.inverse_rtol, parameters.inverse_atol, parameters.pseudoinverse)
end

function probe_frequency_core_complete(parameters)
    available = Set(-Int(parameters.nfermionic):Int(parameters.nfermionic)-1)
    for ν in FERMIONIC_INDICES, Ω in 0:Int(parameters.nbosonic)-1
        ν + Ω in available || return false
    end
    return true
end

function missing_frequency_fraction(parameters, missing::Integer)
    total = prod(parameters.shape) * (2Int(parameters.nfermionic)) * Int(parameters.nbosonic)
    total > 0 || return NaN
    return Float64(missing) / total
end

const REFERENCE_CACHE = Dict{Any,Any}()
const IMPURITY_CACHE = Dict{Any,ImpurityTwoParticleResult}()
const RESULT_CACHE = Dict{Any,Observation}()
const PARALLEL_VALIDATED = Ref(false)

reference_cache_key(parameters) = (parameters.shape, parameters.nfermionic, parameters.bath_sites)
impurity_cache_key(parameters) = (reference_cache_key(parameters), parameters.nbosonic, parameters.tau_points, parameters.max_impurity_dimension, TAU_ALGORITHM, TAU_EQUAL_TIME)

function status_text(value::Bool)
    return value ? "PASS" : "FAIL"
end

function yesno(value::Bool)
    return value ? "yes" : "no"
end

function format_level(name::Symbol, level)
    if name === :momentum
        return "$(level[1])×$(level[2])"
    elseif name === :fermionic
        return "Nν=$(level)"
    elseif name === :bosonic
        return "NΩ=$(level)"
    elseif name === :tau
        return "Nτ=$(level)"
    elseif name === :bath
        return "Nbath=$(level)"
    elseif name === :inverse_rtol
        return @sprintf("rtol=%.0e", level)
    end
    return string(level)
end

function format_controls(parameters)
    return @sprintf("grid=%d×%d | Nν=%d | NΩ=%d | Nτ=%d | bath=%d", parameters.shape[1], parameters.shape[2], parameters.nfermionic,
                    parameters.nbosonic, parameters.tau_points, parameters.bath_sites)
end

function print_rule(; width::Integer=104, character::AbstractString="─")
    println(repeat(character, width))
end

function print_benchmark_header()
    print_rule(character="═")
    println("Ladder-DΓA internal convergence benchmark")
    print_rule(character="═")
    @printf("Profile       %-12s  Output       %s\n", String(PROFILE), String(OUTPUT_MODE))
    @printf("Parallelism   %-12s  Julia threads %d\n", String(PARALLEL), Base.Threads.nthreads())
    @printf("Physics       U=%g | β=%g | t=%g | t′=%g | μ=%g\n", U, BETA, T_HOP, TPRIME, CHEMICAL_POTENTIAL)
    println("Reference     ", format_controls(BASE))
    @printf("Backend       %s | equal-time=%s\n", String(TAU_ALGORITHM), String(TAU_EQUAL_TIME))
    @printf("Policy        observable ≤ %.2f%% | vertex ≤ %.2f%% | envelope=%d refinements\n", 100 * TOLERANCE, 100 * VERTEX_TOLERANCE, ENVELOPE_STEPS)
    @printf("Probes        q=%d × Ω=%d | k=%d × ν=%d | two-particle ν=%d × Ω=%d\n", length(QPOINTS), length(BOSONIC_INDICES), length(KPOINTS),
            length(FERMIONIC_INDICES), length(TWO_PARTICLE_FERMIONIC_INDICES), length(TWO_PARTICLE_BOSONIC_INDICES))
    print_rule()
end

function timed_stage(operation::Function, label::AbstractString)
    if !REPORT_TIMING
        return operation()
    end
    timed = @timed operation()
    @printf("      %-20s %7.3f s | %8.2f MiB | GC %6.3f s\n", label, timed.time, timed.bytes / 2.0^20, timed.gctime)
    return timed.value
end

function reference_for(parameters)
    key = reference_cache_key(parameters)
    if haskey(REFERENCE_CACHE, key)
        println("      DMFT reference         cached")
        return REFERENCE_CACHE[key]
    end
    reference = timed_stage("DMFT reference") do
        solve_square_hubbard_dmft(U=U, beta=BETA, chemical_potential=CHEMICAL_POTENTIAL, t=T_HOP, tprime=TPRIME, shape=parameters.shape,
                                  nfermionic=parameters.nfermionic, bath_sites=parameters.bath_sites, bath_fit_maxiter=BATH_FIT_MAXITER,
                                  bath_fit_tol=BATH_FIT_TOL, bath_fit_multistart=BATH_FIT_MULTISTART, dmft_tol=DMFT_TOL, dmft_maxiter=DMFT_MAXITER,
                                  mixing=MIXING)
    end
    REFERENCE_CACHE[key] = reference
    return reference
end

function _relative_array_residual(a, b)
    size(a) == size(b) || throw(DimensionMismatch("parallel-validation arrays differ in shape"))
    return norm(a .- b) / max(norm(a), norm(b), eps(Float64))
end

function _validate_parallel_impurity!(reference, parameters, bosonic_axis, threaded_impurity)
    VALIDATE_PARALLEL || return nothing
    PARALLEL_VALIDATED[] && return nothing
    serial_backend = TauGridFourPoint(ntime=parameters.tau_points, max_dimension=parameters.max_impurity_dimension, algorithm=TAU_ALGORITHM,
                                      parallel=:serial, equal_time=TAU_EQUAL_TIME)
    serial_impurity = timed_stage("serial cross-check") do
        impurity_two_particle(reference.dmft.impurity_result.thermal_state, reference.axis, bosonic_axis; backend=serial_backend)
    end
    residuals = Float64[
        _relative_array_residual(serial_impurity.spin_resolved[:upup].values, threaded_impurity.spin_resolved[:upup].values),
        _relative_array_residual(serial_impurity.spin_resolved[:updown].values, threaded_impurity.spin_resolved[:updown].values),
        _relative_array_residual(serial_impurity.charge.values, threaded_impurity.charge.values),
        _relative_array_residual(serial_impurity.magnetic.values, threaded_impurity.magnetic.values),
    ]
    residual = maximum(residuals)
    @printf("      threaded/serial check  %.3e | tolerance %.3e | %s\n", residual, PARALLEL_VALIDATION_TOL, status_text(residual <= PARALLEL_VALIDATION_TOL))
    residual <= PARALLEL_VALIDATION_TOL || throw(ErrorException("threaded TauGridFourPoint disagrees with the serial factored backend; refusing to continue the convergence benchmark"))
    PARALLEL_VALIDATED[] = true
    return nothing
end

function impurity_for(reference, parameters)
    key = impurity_cache_key(parameters)
    if haskey(IMPURITY_CACHE, key)
        println("      impurity four-point    cached")
        return IMPURITY_CACHE[key]
    end
    bosonic_axis = BosonicMatsubaraAxis(BETA, collect(0:Int(parameters.nbosonic)-1))
    backend = TauGridFourPoint(ntime=parameters.tau_points, max_dimension=parameters.max_impurity_dimension, algorithm=TAU_ALGORITHM,
                               parallel=PARALLEL, equal_time=TAU_EQUAL_TIME)
    impurity = timed_stage("impurity four-point") do
        impurity_two_particle(reference.dmft.impurity_result.thermal_state, reference.axis, bosonic_axis; backend=backend)
    end
    _validate_parallel_impurity!(reference, parameters, bosonic_axis, impurity)
    IMPURITY_CACHE[key] = impurity
    return impurity
end

function run_observation(parameters)
    key = cache_key(parameters)
    if haskey(RESULT_CACHE, key)
        println("      complete observation   cached")
        return RESULT_CACHE[key]
    end
    harmonic_requirement = tau_harmonic_requirement(parameters.nfermionic, parameters.nbosonic)
    resolved = tau_grid_resolved(parameters.tau_points, parameters.nfermionic, parameters.nbosonic)
    VERBOSE_OUTPUT && println("      controls: ", format_controls(parameters))
    @printf("      τ resolution            Nτ=%d | harmonic=%d | Nyquist-safe=%s\n", parameters.tau_points, harmonic_requirement, yesno(resolved))
    reference = reference_for(parameters)
    impurity = impurity_for(reference, parameters)
    result = timed_stage("ladder-DΓA") do
        solve_square_hubbard_ladder_dga(U=U, beta=BETA, chemical_potential=CHEMICAL_POTENTIAL, t=T_HOP, tprime=TPRIME, shape=parameters.shape,
                                        nfermionic=parameters.nfermionic, nbosonic=parameters.nbosonic, bath_sites=parameters.bath_sites,
                                        tau_points=parameters.tau_points, max_impurity_dimension=parameters.max_impurity_dimension,
                                        bath_fit_maxiter=BATH_FIT_MAXITER, bath_fit_tol=BATH_FIT_TOL, bath_fit_multistart=BATH_FIT_MULTISTART,
                                        dmft_tol=DMFT_TOL, dmft_maxiter=DMFT_MAXITER, mixing=MIXING, regularization=parameters.regularization,
                                        inverse_rtol=parameters.inverse_rtol, inverse_atol=parameters.inverse_atol, pseudoinverse=parameters.pseudoinverse,
                                        prepared_reference=reference, prepared_impurity=impurity, parallel=PARALLEL, tau_algorithm=TAU_ALGORITHM,
                                        tau_equal_time=TAU_EQUAL_TIME)
    end
    magnetic_inverse = inverse_diagnostic_summary(result, :magnetic)
    charge_inverse = inverse_diagnostic_summary(result, :charge)
    missing = Int(get(result.metadata, :missing_frequency_entries, 0))
    observation = Observation(select_susceptibility(result, :magnetic, QPOINTS, BOSONIC_INDICES),
                              select_susceptibility(result, :charge, QPOINTS, BOSONIC_INDICES),
                              select_self_energy(result, KPOINTS, FERMIONIC_INDICES),
                              select_two_particle(result.impurity_two_particle.magnetic, TWO_PARTICLE_FERMIONIC_INDICES, TWO_PARTICLE_BOSONIC_INDICES),
                              select_two_particle(result.impurity_two_particle.charge, TWO_PARTICLE_FERMIONIC_INDICES, TWO_PARTICLE_BOSONIC_INDICES),
                              select_two_particle(result.local_vertices[:magnetic], TWO_PARTICLE_FERMIONIC_INDICES, TWO_PARTICLE_BOSONIC_INDICES),
                              select_two_particle(result.local_vertices[:charge], TWO_PARTICLE_FERMIONIC_INDICES, TWO_PARTICLE_BOSONIC_INDICES),
                              magnetic_inverse.maximum_local_condition, charge_inverse.maximum_local_condition,
                              magnetic_inverse.maximum_kernel_condition, charge_inverse.maximum_kernel_condition,
                              magnetic_inverse.minimum_local_rank_fraction, charge_inverse.minimum_local_rank_fraction,
                              magnetic_inverse.minimum_kernel_rank_fraction, charge_inverse.minimum_kernel_rank_fraction,
                              missing, missing_frequency_fraction(parameters, missing), Bool(get(result.metadata, :frequency_box_complete, false)),
                              probe_frequency_core_complete(parameters))
    RESULT_CACHE[key] = observation
    return observation
end

function compare(previous::Observation, current::Observation)
    magnetic = relative_norm_change(previous.magnetic, current.magnetic)
    charge = relative_norm_change(previous.charge, current.charge)
    self_energy = relative_norm_change(previous.self_energy, current.self_energy)
    impurity_magnetic = relative_norm_change(previous.impurity_magnetic, current.impurity_magnetic)
    impurity_charge = relative_norm_change(previous.impurity_charge, current.impurity_charge)
    vertex_magnetic = relative_norm_change(previous.vertex_magnetic, current.vertex_magnetic)
    vertex_charge = relative_norm_change(previous.vertex_charge, current.vertex_charge)
    observable_maximum = max(magnetic, charge, self_energy)
    vertex_maximum = max(vertex_magnetic, vertex_charge)
    internal_maximum = max(impurity_magnetic, impurity_charge, vertex_magnetic, vertex_charge)
    return Comparison(magnetic, charge, self_energy, impurity_magnetic, impurity_charge, vertex_magnetic, vertex_charge,
                      observable_maximum, vertex_maximum, internal_maximum)
end

function finite_comparison(comparison::Comparison)
    values = (comparison.magnetic, comparison.charge, comparison.self_energy, comparison.impurity_magnetic, comparison.impurity_charge,
              comparison.vertex_magnetic, comparison.vertex_charge, comparison.observable_maximum, comparison.vertex_maximum, comparison.internal_maximum)
    return all(isfinite, values)
end

function observable_metric(name::Symbol, comparison::Comparison)
    return name === :bosonic ? comparison.self_energy : comparison.observable_maximum
end

vertex_metric(comparison::Comparison) = comparison.vertex_maximum

function envelope_pass(comparisons::Vector{Comparison}, metric, tolerance::Real; steps::Integer=ENVELOPE_STEPS)
    length(comparisons) >= steps || return false
    tail = @view comparisons[end-steps+1:end]
    return all(comparison -> finite_comparison(comparison) && metric(comparison) <= tolerance, tail)
end

function contraction_ok(comparisons::Vector{Comparison}, metric)
    length(comparisons) < 2 && return true
    return metric(comparisons[end]) <= metric(comparisons[end - 1])
end

function stage_amplification(comparison::Comparison)
    impurity = max(comparison.impurity_magnetic, comparison.impurity_charge, eps(Float64))
    vertex = max(comparison.vertex_magnetic, comparison.vertex_charge, eps(Float64))
    lattice = max(comparison.magnetic, comparison.charge, eps(Float64))
    return (impurity_to_vertex=vertex / impurity, vertex_to_lattice=lattice / vertex, lattice_to_self_energy=comparison.self_energy / lattice)
end

function print_inverse_diagnostics(observation::Observation)
    @printf("      conditioning           local κ M/C %.3e / %.3e | BSE κ M/C %.3e / %.3e\n", observation.local_condition_magnetic,
            observation.local_condition_charge, observation.kernel_condition_magnetic, observation.kernel_condition_charge)
    @printf("      rank / frequency       local %.3f/%.3f | BSE %.3f/%.3f | missing %.2f%% | probe core %s\n",
            observation.local_rank_fraction_magnetic, observation.local_rank_fraction_charge, observation.kernel_rank_fraction_magnetic,
            observation.kernel_rank_fraction_charge, 100 * observation.missing_frequency_fraction, yesno(observation.probe_frequency_core_complete))
end

function admissible_comparison_indices(name::Symbol, levels)
    indices = collect(2:length(levels))
    if name === :tau
        return [index for index in indices if tau_grid_resolved(parameter_set(name, levels[index - 1]).tau_points,
                                                                parameter_set(name, levels[index - 1]).nfermionic,
                                                                parameter_set(name, levels[index - 1]).nbosonic) &&
                                               tau_grid_resolved(parameter_set(name, levels[index]).tau_points,
                                                                 parameter_set(name, levels[index]).nfermionic,
                                                                 parameter_set(name, levels[index]).nbosonic)]
    elseif name === :bosonic
        return [index for index in indices if probe_frequency_core_complete(parameter_set(name, levels[index - 1])) &&
                                               probe_frequency_core_complete(parameter_set(name, levels[index]))]
    end
    return indices
end

function run_study(name::Symbol, label::AbstractString, levels; gate::Bool=true, convergence_test::Bool=true, vertex_gate::Bool=false)
    println()
    print_rule()
    println(label, convergence_test ? " convergence" : " sensitivity")
    println("Levels: ", join((format_level(name, level) for level in levels), " → "))
    print_rule()
    observations = Observation[]
    for (index, level) in enumerate(levels)
        @printf("  [%d/%d] %s\n", index, length(levels), format_level(name, level))
        observation = run_observation(parameter_set(name, level))
        push!(observations, observation)
        print_inverse_diagnostics(observation)
    end
    comparisons = Comparison[]
    for index in 2:length(observations)
        push!(comparisons, compare(observations[index - 1], observations[index]))
    end
    admissible = admissible_comparison_indices(name, levels)
    gated_comparisons = Comparison[comparisons[index - 1] for index in admissible]
    final = isempty(gated_comparisons) ? Comparison(Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf) : gated_comparisons[end]
    observable = comparison -> observable_metric(name, comparison)
    observable_pass = convergence_test ? envelope_pass(gated_comparisons, observable, TOLERANCE) : true
    effective_vertex_gate = convergence_test && vertex_gate && GATE_VERTEX_CONVERGENCE
    vertex_pass = effective_vertex_gate ? envelope_pass(gated_comparisons, vertex_metric, VERTEX_TOLERANCE) : true
    contraction_pass = !convergence_test || contraction_ok(gated_comparisons, observable)
    passed = observable_pass && vertex_pass

    println()
    println("  Relative residuals")
    @printf("  %-19s %11s %11s %11s %11s %11s\n", "refinement", "impurity", "vertex", "χ max", "Σ", "gate metric")
    for index in 2:length(observations)
        comparison = comparisons[index - 1]
        impurity_maximum = max(comparison.impurity_magnetic, comparison.impurity_charge)
        susceptibility_maximum = max(comparison.magnetic, comparison.charge)
        gate_metric = observable_metric(name, comparison)
        refinement = "$(format_level(name, levels[index - 1]))→$(format_level(name, levels[index]))"
        @printf("  %-19s %11.3e %11.3e %11.3e %11.3e %11.3e\n", refinement, impurity_maximum, comparison.vertex_maximum,
                susceptibility_maximum, comparison.self_energy, gate_metric)
        if VERBOSE_OUTPUT
            amplification = stage_amplification(comparison)
            @printf("      channels              imp M/C %.3e / %.3e | vert M/C %.3e / %.3e | χ M/C %.3e / %.3e\n",
                    comparison.impurity_magnetic, comparison.impurity_charge, comparison.vertex_magnetic, comparison.vertex_charge,
                    comparison.magnetic, comparison.charge)
            @printf("      amplification         imp→vert %.3e | vert→χ %.3e | χ→Σ %.3e\n", amplification.impurity_to_vertex,
                    amplification.vertex_to_lattice, amplification.lattice_to_self_energy)
        end
    end

    if name === :tau
        ignored = setdiff(collect(2:length(levels)), admissible)
        isempty(ignored) || println("  Note: excluded under-resolved τ refinements: ", join(("$(format_level(name, levels[i - 1]))→$(format_level(name, levels[i]))" for i in ignored), ", "))
    elseif name === :bosonic
        ignored = setdiff(collect(2:length(levels)), admissible)
        isempty(ignored) || println("  Note: excluded refinements with incomplete Schwinger-Dyson probe core: ", join(("$(format_level(name, levels[i - 1]))→$(format_level(name, levels[i]))" for i in ignored), ", "))
    end

    println()
    if convergence_test
        @printf("  Observable envelope  %-4s  last %d admissible refinements ≤ %.2f%%\n", status_text(observable_pass), ENVELOPE_STEPS, 100 * TOLERANCE)
        if effective_vertex_gate
            @printf("  Vertex envelope      %-4s  last %d admissible refinements ≤ %.2f%%\n", status_text(vertex_pass), ENVELOPE_STEPS, 100 * VERTEX_TOLERANCE)
        else
            println("  Vertex envelope      ----  diagnostic only for this study")
        end
        if REPORT_CONTRACTION
            @printf("  Contraction          %-4s  non-gating diagnostic\n", status_text(contraction_pass))
        end
    else
        println("  Gate                 ----  diagnostic sensitivity study")
    end
    @printf("  Study result         %s%s\n", status_text(passed), gate ? "" : "  (non-gating)")
    return (name=name, label=label, levels=levels, comparisons=comparisons, admissible=admissible, final=final, passed=passed, gate=gate,
            convergence_test=convergence_test, observations=observations, observable_pass=observable_pass, vertex_pass=vertex_pass,
            vertex_gate=effective_vertex_gate, contraction_pass=contraction_pass)
end

function run_inverse_sensitivity()
    levels = [1e-6, 1e-8, 1e-10, 1e-12]
    return run_study(:inverse_rtol, "pseudoinverse cutoff", levels; gate=false, convergence_test=false, vertex_gate=false)
end

function diagnose_study(result)
    final = result.final
    observation = result.observations[end]
    if result.name === :fermionic
        if final.impurity_magnetic > TOLERANCE || final.impurity_charge > TOLERANCE
            return "fermionic truncation is already visible in the impurity two-particle object"
        elseif final.vertex_magnetic > VERTEX_TOLERANCE || final.vertex_charge > VERTEX_TOLERANCE
            return "fermionic truncation is amplified primarily by local inverse-BSE vertex extraction"
        elseif final.magnetic > TOLERANCE || final.charge > TOLERANCE
            return "fermionic truncation is amplified primarily in the lattice Bethe-Salpeter stage"
        elseif final.self_energy > TOLERANCE
            return "fermionic truncation is concentrated in the Schwinger-Dyson reconstruction"
        end
    elseif result.name === :bosonic
        if !observation.probe_frequency_core_complete
            return "bosonic-box comparison reaches the boundary of the fermionic box for at least one Schwinger-Dyson probe frequency"
        elseif result.observable_pass && !result.contraction_pass
            return "safe-core Schwinger-Dyson residuals lie inside the convergence envelope; the final nonmonotonic step is a non-gating warning"
        elseif !result.observable_pass
            return "bosonic truncation remains visible in the safe-core Schwinger-Dyson self-energy"
        end
    elseif result.name === :tau
        if final.impurity_magnetic > TOLERANCE || final.impurity_charge > TOLERANCE
            return "remaining τ-grid error originates in the impurity Fourier projection"
        elseif !result.vertex_pass
            return "physical observables may be converged while the local irreducible vertex remains τ-grid sensitive"
        elseif final.vertex_magnetic > VERTEX_TOLERANCE || final.vertex_charge > VERTEX_TOLERANCE
            return "τ-grid error is small in the impurity correlator but amplified by local vertex inversion"
        end
    end
    if observation.local_rank_fraction_magnetic < 1 || observation.local_rank_fraction_charge < 1
        return "local vertex extraction is rank deficient under the configured pseudoinverse cutoff"
    elseif observation.kernel_rank_fraction_magnetic < 1 || observation.kernel_rank_fraction_charge < 1
        return "at least one lattice BSE kernel is rank deficient under the configured pseudoinverse cutoff"
    elseif !result.observable_pass && result.convergence_test
        return "observable convergence envelope is not yet satisfied"
    elseif result.vertex_gate && !result.vertex_pass
        return "local-vertex convergence envelope is not yet satisfied"
    end
    return "no isolated failure detected at the configured tolerance"
end

function print_summary(results)
    println()
    print_rule(character="═")
    println("Convergence summary")
    print_rule(character="═")
    @printf("%-28s %11s %11s %8s %8s %8s\n", "study", "obs metric", "vertex", "OBS", "VERT", "RESULT")
    for result in results
        final = result.final
        observable_value = observable_metric(result.name, final)
        vertex_label = result.vertex_gate ? status_text(result.vertex_pass) : "----"
        @printf("%-28s %11.3e %11.3e %8s %8s %8s\n", result.label, observable_value, final.vertex_maximum,
                status_text(result.observable_pass), vertex_label, result.gate ? status_text(result.passed) : "diag")
    end
    failed = [result for result in results if result.gate && !result.passed]
    println()
    if isempty(failed)
        println("Failure isolation: no gated failures detected.")
    else
        println("Failure isolation")
        for result in failed
            println("  - ", result.label, ": ", diagnose_study(result))
        end
    end
end

if !tau_grid_resolved(BASE.tau_points, BASE.nfermionic, BASE.nbosonic)
    required = tau_harmonic_requirement(BASE.nfermionic, BASE.nbosonic)
    @warn "BASE τ grid is under-resolved for the requested fermionic/bosonic boxes" tau_points=BASE.tau_points largest_sampled_phase_harmonic=required
end

print_benchmark_header()

results = Any[]
for study in STUDIES
    push!(results, run_study(study.name, study.label, study.levels; gate=study.gate, convergence_test=true, vertex_gate=study.vertex_gate))
end
if RUN_BATH_STUDY
    push!(results, run_study(:bath, "impurity bath", [1, 2]; gate=GATE_BATH_STUDY, convergence_test=false, vertex_gate=false))
else
    println()
    println("Impurity bath sensitivity: skipped (PHUNDAMENTAL_DGA_RUN_BATH_STUDY=false)")
end
if RUN_INVERSION_STUDY
    push!(results, run_inverse_sensitivity())
else
    println()
    println("Pseudoinverse cutoff sensitivity: skipped (PHUNDAMENTAL_DGA_RUN_INVERSION_STUDY=false)")
end

print_summary(results)
gated_results = filter(result -> result.gate, results)
observable_pass = !isempty(gated_results) && all(result -> result.observable_pass, gated_results)
vertex_gated_results = filter(result -> result.gate && result.vertex_gate, results)
vertex_pass = isempty(vertex_gated_results) || all(result -> result.vertex_pass, vertex_gated_results)
passed = observable_pass && vertex_pass
println()
print_rule(character="═")
println("Overall result")
print_rule(character="═")
@printf("Observables  %s\n", status_text(observable_pass))
@printf("Vertex       %s\n", status_text(vertex_pass))
@printf("Overall      %s\n", status_text(passed))
print_rule(character="═")
exit(passed ? 0 : 1)
