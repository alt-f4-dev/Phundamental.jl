#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
using .Phundamental

using LinearAlgebra
using Printf
using Statistics

#--------------------------------------------------------------------------------------------------#
# Caffarel, Michel and Werner Krauth. (1994).                                                      #
# Exact diagonalization approach to correlated fermions in infinite dimensions:                    #
# Mott transition and superconductivity. Phys. Rev. Lett. 72, 1545-1548.                           #
# DOI: 10.1103/PhysRevLett.72.1545                                                                 #
# Extended derivation: arXiv:cond-mat/9306057.                                                     #
#--------------------------------------------------------------------------------------------------#
# Georges, Antoine, Gabriel Kotliar, Werner Krauth, and Marcelo J. Rozenberg. (1996).               #
# Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite      #
# dimensions. Rev. Mod. Phys. 68, 13-125. DOI: 10.1103/RevModPhys.68.13.                            #
#--------------------------------------------------------------------------------------------------#
# The benchmark reproduces source-level conventions before comparing numerical behavior.            #
# Caffarel and Krauth use G0^-1(iω)=iω+μ-(1/2)G(iω) for the Bethe lattice. In the Phundamental       #
# BetheLattice(D) convention this corresponds to D=sqrt(2), since t^2=D^2/4=1/2. Their finite-T      #
# Fig. 1 comparison uses beta=32, U=3, and Anderson sizes ns=3 and 5; here bath_sites=ns-1.          #
# Their Eq. (5) uses an unweighted inverse-Weiss least-squares distance, equivalent at fixed impurity #
# level to an unweighted hybridization distance, so weight_power=0 is used below.                    #
#--------------------------------------------------------------------------------------------------#

function parse_integer_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Int, strip.(split(get(ENV, name, default), ',')))))
    isempty(values) && error("$name must contain at least one integer")
    return values
end

function inverse_weiss_chi2(reference::NonInteractingGreenFunction, candidate::NonInteractingGreenFunction)
    length(reference) == length(candidate) || throw(DimensionMismatch("Weiss fields must have the same number of frequencies"))
    reference_values = scalar_frequency_values(reference)
    candidate_values = scalar_frequency_values(candidate)
    return mean(abs2.(inv.(reference_values) .- inv.(candidate_values)))
end

function benchmark_mixing(prefix::AbstractString)
    mode = Symbol(lowercase(get(ENV, "$(prefix)_MIXING", "adaptive")))
    alpha = parse(Float64, get(ENV, "$(prefix)_ALPHA", "0.5"))
    if mode === :linear
        return LinearMixing(alpha), mode
    elseif mode === :adaptive
        minimum_alpha = parse(Float64, get(ENV, "$(prefix)_MIN_ALPHA", "0.05"))
        maximum_alpha = parse(Float64, get(ENV, "$(prefix)_MAX_ALPHA", "0.8"))
        recovery_window = parse(Int, get(ENV, "$(prefix)_RECOVERY_WINDOW", "3"))
        return AdaptiveLinearMixing(initial_alpha=alpha, min_alpha=minimum_alpha, max_alpha=maximum_alpha, recovery_window=recovery_window), mode
    end
    error("$(prefix)_MIXING must be linear or adaptive")
end


function benchmark_bath_symmetry(prefix::AbstractString)
    symmetry = Symbol(lowercase(get(ENV, "$(prefix)_BATH_SYMMETRY", "particle_hole")))
    symmetry in (:none, :particle_hole) || error("$(prefix)_BATH_SYMMETRY must be none or particle_hole")
    return symmetry
end

function main()
    beta = parse(Float64, get(ENV, "PHUNDAMENTAL_CK_BETA", "32"))
    interaction = parse(Float64, get(ENV, "PHUNDAMENTAL_CK_U", "3"))
    ns_values = parse_integer_list("PHUNDAMENTAL_CK_NS", "3,5")
    nfrequencies = parse(Int, get(ENV, "PHUNDAMENTAL_CK_NFREQUENCIES", "128"))
    maxiter = parse(Int, get(ENV, "PHUNDAMENTAL_CK_MAXITER", "60"))
    tolerance = parse(Float64, get(ENV, "PHUNDAMENTAL_CK_TOL", "1e-7"))
    multistart = parse(Int, get(ENV, "PHUNDAMENTAL_CK_MULTISTART", "7"))
    threading = Symbol(lowercase(get(ENV, "PHUNDAMENTAL_CK_THREADING", "auto")))
    bath_symmetry = benchmark_bath_symmetry("PHUNDAMENTAL_CK")
    mixing, mixing_mode = benchmark_mixing("PHUNDAMENTAL_CK")

    beta > 0 || error("PHUNDAMENTAL_CK_BETA must be positive")
    interaction >= 0 || error("PHUNDAMENTAL_CK_U must be nonnegative")
    all(>=(2), ns_values) || error("PHUNDAMENTAL_CK_NS values must be at least 2")
    nfrequencies > 0 || error("PHUNDAMENTAL_CK_NFREQUENCIES must be positive")
    threading in (:serial, :threads, :auto) || error("PHUNDAMENTAL_CK_THREADING must be serial, threads, or auto")
    bath_symmetry === :particle_hole && any(iseven, ns_values) && error("particle-hole-symmetric fitting requires odd PHUNDAMENTAL_CK_NS values so bath_sites=ns-1 is even")

    half_bandwidth = sqrt(2.0)
    temperature = inv(beta)
    chemical_potential = interaction / 2
    lattice = BetheLattice(half_bandwidth)
    results = NamedTuple[]
    history_rows = NamedTuple[]
    matsubara_rows = NamedTuple[]
    previous_converged_result = nothing

    println()
    println("CAFFAREL-KRAUTH 1994 FINITE-T ED-DMFT BENCHMARK")
    @printf("beta = %.6g, U = %.6g, mu = %.6g, D = sqrt(2), nfreq = %d\n", beta, interaction, chemical_potential, nfrequencies)
    @printf("threading = %s, Julia threads = %d, BLAS threads = %d, mixing = %s, bath symmetry = %s\n", string(threading), Threads.nthreads(), BLAS.get_num_threads(), string(mixing_mode), string(bath_symmetry))
    Threads.nthreads() > 1 && BLAS.get_num_threads() > 1 && println("WARNING: BLAS has more than one thread; use BLAS threads = 1 for clean Julia-thread scaling.")
    println("Reference: M. Caffarel and W. Krauth, Phys. Rev. Lett. 72, 1545 (1994), DOI: 10.1103/PhysRevLett.72.1545")
    println()
    @printf("%5s %6s %8s %12s %12s %12s %12s %12s %10s %10s %9s\n", "ns", "bath", "conv", "termination", "chi2", "bath_res", "closure", "best_it_res", "Z_est", "density", "iters")

    for ns in ns_values
        bath_sites = ns - 1
        fit_options = BathFitOptions(maxiter=400, tol=1e-9, weight_power=0.0,
                                     energy_window=(chemical_potential - 4.0, chemical_potential + 4.0), multistart=multistart,
                                     symmetry=bath_symmetry)
        impurity_solver = EDImpuritySolver(bath_sites=bath_sites, bath_fit=fit_options, backend=:sectorized, threading=threading)
        solver = DMFTSolver(impurity_solver=impurity_solver, mixing=mixing, tol=tolerance, maxiter=maxiter, verify_closure=true)
        problem = DMFTProblem(lattice, interaction; temperature=temperature, chemical_potential=chemical_potential, nfrequencies=nfrequencies)

        initial_self_energy = isnothing(previous_converged_result) ? nothing : previous_converged_result.self_energy
        timing = @timed solve(solver, problem; initial_self_energy=initial_self_energy)
        result = timing.value
        diagnostics = validate_dmft_result(result)
        chi2 = inverse_weiss_chi2(result.weiss_green_function, result.impurity_result.noninteracting_green_function)
        weight_estimate = quasiparticle_weight_estimate(result.self_energy; nfit=min(4, nfrequencies)).weight
        bath_residual = diagnostics[:bath_discretization_residual]
        particle_hole_residual = diagnostics[:bath_particle_hole_residual]
        bethe_residual = diagnostics[:bethe_self_consistency_residual]
        termination = result.metadata[:termination]
        closure_residual = result.metadata[:closure_residual]
        final_iteration_residual = result.metadata[:final_iteration_residual]
        best_iteration_residual = result.metadata[:best_iteration_residual]
        best_iteration = result.metadata[:iteration_of_best_residual]
        seed_kind = isnothing(previous_converged_result) ? :none : :self_energy_continuation

        push!(results, (; ns, bath_sites, bath_symmetry, converged=result.converged, termination, seed_kind, chi2, bath_residual, particle_hole_residual,
                        closure_residual, final_iteration_residual, best_iteration_residual, best_iteration, quasiparticle_weight=weight_estimate,
                        density=result.impurity_result.density, double_occupancy=result.impurity_result.double_occupancy,
                        iterations=result.iterations, bethe_residual, final_mixing_alpha=result.metadata[:final_mixing_alpha],
                        threading_effective=result.impurity_result.metadata[:threading_effective], julia_threads=Threads.nthreads(),
                        blas_threads=BLAS.get_num_threads(), seconds=timing.time, bytes=timing.bytes))

        for entry in result.history
            push!(history_rows, (; ns, bath_sites, iteration=entry.iteration, residual=entry.residual,
                                 weiss_residual=entry.weiss_residual, hybridization_residual=entry.hybridization_residual,
                                 bath_fit_residual=entry.bath_fit_residual, mixing_alpha=entry.mixing_alpha,
                                 density=entry.density, double_occupancy=entry.double_occupancy,
                                 bath_fit_objective=entry.bath_fit_objective, bath_fit_converged=entry.bath_fit_converged))
        end

        for i in 1:length(problem.axis)
            z = problem.axis[i]
            G = result.local_green_function[1, 1, i]
            sigma = result.self_energy[1, 1, i]
            G0 = result.weiss_green_function[1, 1, i]
            delta = result.hybridization[1, 1, i]
            push!(matsubara_rows, (; ns, bath_sites, matsubara_index=problem.axis.indices[i], omega=imag(z),
                                   green_real=real(G), green_imag=imag(G), sigma_real=real(sigma), sigma_imag=imag(sigma),
                                   weiss_real=real(G0), weiss_imag=imag(G0), delta_real=real(delta), delta_imag=imag(delta)))
        end

        @printf("%5d %6d %8s %12s %12.4e %12.4e %12.4e %12.4e %10.5f %10.6f %9d\n", ns, bath_sites, string(result.converged),
                string(termination), chi2, bath_residual, closure_residual, best_iteration_residual, weight_estimate, result.impurity_result.density, result.iterations)

        result.converged && (previous_converged_result = result)
    end

    result_directory = joinpath(@__DIR__, "results")
    mkpath(result_directory)
    output_path = joinpath(result_directory, "caffarel-krauth-1994.csv")
    open(output_path, "w") do io
        println(io, "ns,bath_sites,bath_symmetry,converged,termination,seed_kind,chi2_inverse_weiss,bath_residual,bath_particle_hole_residual,closure_residual,final_iteration_residual,best_iteration_residual,best_iteration,quasiparticle_weight,density,double_occupancy,iterations,bethe_residual,final_mixing_alpha,threading_effective,julia_threads,blas_threads,seconds,bytes")
        for row in results
            println(io, join((row.ns, row.bath_sites, row.bath_symmetry, row.converged, row.termination, row.seed_kind, row.chi2, row.bath_residual,
                              row.particle_hole_residual, row.closure_residual, row.final_iteration_residual, row.best_iteration_residual, row.best_iteration, row.quasiparticle_weight,
                              row.density, row.double_occupancy, row.iterations, row.bethe_residual, row.final_mixing_alpha,
                              row.threading_effective, row.julia_threads, row.blas_threads, row.seconds, row.bytes), ","))
        end
    end

    history_path = joinpath(result_directory, "caffarel-krauth-1994-history.csv")
    open(history_path, "w") do io
        println(io, "ns,bath_sites,iteration,self_energy_residual,weiss_residual,hybridization_residual,bath_fit_residual,mixing_alpha,density,double_occupancy,bath_fit_objective,bath_fit_converged")
        for row in history_rows
            println(io, join((row.ns, row.bath_sites, row.iteration, row.residual, row.weiss_residual, row.hybridization_residual,
                              row.bath_fit_residual, row.mixing_alpha, row.density, row.double_occupancy,
                              row.bath_fit_objective, row.bath_fit_converged), ","))
        end
    end

    matsubara_path = joinpath(result_directory, "caffarel-krauth-1994-matsubara.csv")
    open(matsubara_path, "w") do io
        println(io, "ns,bath_sites,matsubara_index,omega,green_real,green_imag,sigma_real,sigma_imag,weiss_real,weiss_imag,delta_real,delta_imag")
        for row in matsubara_rows
            println(io, join((row.ns, row.bath_sites, row.matsubara_index, row.omega, row.green_real, row.green_imag, row.sigma_real, row.sigma_imag,
                              row.weiss_real, row.weiss_imag, row.delta_real, row.delta_imag), ","))
        end
    end

    println()
    println("Published-source checks:")
    println("  Bethe normalization residuals verify Caffarel-Krauth Eq. (2) in the Phundamental D=sqrt(2) convention.")
    println("  chi2_inverse_weiss implements the source Eq. (5) distance on the converged lattice and finite-bath Weiss fields.")
    println("  quasiparticle_weight uses Im Sigma(iw) ~ (1 - 1/Z) w, the low-frequency diagnostic described with their Fig. 3.")
    println("  No digitized figure values are treated as exact numerical targets.")
    println("Summary results: ", output_path)
    println("Iteration histories: ", history_path)
    println("Matsubara curves: ", matsubara_path)
end

main()
