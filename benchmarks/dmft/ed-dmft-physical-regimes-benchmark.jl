#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
using .Phundamental

using Printf
using LinearAlgebra

#--------------------------------------------------------------------------------------------------#
# Caffarel, Michel and Werner Krauth. (1994). Phys. Rev. Lett. 72, 1545-1548.                       #
# DOI: 10.1103/PhysRevLett.72.1545                                                                 #
#--------------------------------------------------------------------------------------------------#
# Georges, Antoine, Gabriel Kotliar, Werner Krauth, and Marcelo J. Rozenberg. (1996).               #
# Rev. Mod. Phys. 68, 13-125. DOI: 10.1103/RevModPhys.68.13.                                       #
#--------------------------------------------------------------------------------------------------#
# Liebsch, Ansgar and Hiroshi Ishida. (2012). J. Phys.: Condens. Matter 24, 053201.                 #
# DOI: 10.1088/0953-8984/24/5/053201                                                               #
#--------------------------------------------------------------------------------------------------#
# This benchmark follows the literature validation logic rather than fitting to a desired phase.    #
# Nearby interaction values and bath sizes use converged DMFT solutions only as continuation seeds. #
# The final fixed point is independently closure-verified, so continuation changes the starting      #
# condition rather than the physical convergence criterion.                                          #
#--------------------------------------------------------------------------------------------------#

function parse_integer_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Int, strip.(split(get(ENV, name, default), ',')))))
    isempty(values) && error("$name must contain at least one integer")
    return values
end

function parse_float_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Float64, strip.(split(get(ENV, name, default), ',')))))
    isempty(values) && error("$name must contain at least one number")
    return values
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
    beta = parse(Float64, get(ENV, "PHUNDAMENTAL_DMFT_BETA", "32"))
    interactions = parse_float_list("PHUNDAMENTAL_DMFT_U_VALUES", "0,2,3,4.8")
    bath_sizes = parse_integer_list("PHUNDAMENTAL_DMFT_BATH_SIZES", "2,4")
    nfrequencies = parse(Int, get(ENV, "PHUNDAMENTAL_DMFT_NFREQUENCIES", "128"))
    maxiter = parse(Int, get(ENV, "PHUNDAMENTAL_DMFT_MAXITER", "320"))
    tolerance = parse(Float64, get(ENV, "PHUNDAMENTAL_DMFT_TOL", "1e-7"))
    multistart = parse(Int, get(ENV, "PHUNDAMENTAL_DMFT_MULTISTART", "7"))
    threading = Symbol(lowercase(get(ENV, "PHUNDAMENTAL_DMFT_THREADING", "auto")))
    bath_symmetry = benchmark_bath_symmetry("PHUNDAMENTAL_DMFT")
    mixing, mixing_mode = benchmark_mixing("PHUNDAMENTAL_DMFT")

    beta > 0 || error("PHUNDAMENTAL_DMFT_BETA must be positive")
    all(>=(0), interactions) || error("PHUNDAMENTAL_DMFT_U_VALUES must be nonnegative")
    all(>=(0), bath_sizes) || error("PHUNDAMENTAL_DMFT_BATH_SIZES must be nonnegative")
    threading in (:serial, :threads, :auto) || error("PHUNDAMENTAL_DMFT_THREADING must be serial, threads, or auto")
    bath_symmetry === :particle_hole && any(isodd, bath_sizes) && error("particle-hole-symmetric fitting requires even PHUNDAMENTAL_DMFT_BATH_SIZES")

    lattice = BetheLattice(sqrt(2.0))
    temperature = inv(beta)
    rows = NamedTuple[]
    history_rows = NamedTuple[]
    matsubara_rows = NamedTuple[]
    previous_by_bath = Dict{Int,Any}()

    println()
    println("SINGLE-SITE BETHE ED-DMFT PHYSICAL-REGIME BENCHMARK")
    println("References: Caffarel-Krauth 1994; Georges et al. 1996; Liebsch-Ishida 2012")
    @printf("beta = %.6g, D = sqrt(2), nfreq = %d\n", beta, nfrequencies)
    @printf("threading = %s, Julia threads = %d, BLAS threads = %d, mixing = %s, bath symmetry = %s\n", string(threading), Threads.nthreads(), BLAS.get_num_threads(), string(mixing_mode), string(bath_symmetry))
    Threads.nthreads() > 1 && BLAS.get_num_threads() > 1 && println("WARNING: BLAS has more than one thread; use BLAS threads = 1 for clean Julia-thread scaling.")
    println()
    @printf("%7s %5s %8s %12s %11s %11s %11s %11s %11s %10s %10s %9s\n", "U", "Nb", "conv", "termination", "closure", "iter_res", "best_it_res", "bath_res", "Sigma_dNb", "Z_est", "density", "iters")

    for interaction in interactions
        chemical_potential = interaction / 2
        previous_same_interaction = nothing
        previous_sigma_same_interaction = nothing

        for bath_sites in bath_sizes
            same_bath_seed = get(previous_by_bath, bath_sites, nothing)
            seed_result = if !isnothing(same_bath_seed) && same_bath_seed.converged
                same_bath_seed
            elseif !isnothing(previous_same_interaction) && previous_same_interaction.converged
                previous_same_interaction
            else
                nothing
            end
            seed_kind = isnothing(seed_result) ? :none : (seed_result === same_bath_seed ? :interaction_continuation : :bath_continuation)
            initial_self_energy = isnothing(seed_result) ? nothing : seed_result.self_energy
            initial_bath = !isnothing(same_bath_seed) && same_bath_seed.converged ? same_bath_seed.impurity_result.bath : nothing

            fit_options = BathFitOptions(maxiter=500, tol=1e-9, weight_power=1.0,
                                         energy_window=(chemical_potential - 4.0, chemical_potential + 4.0), multistart=multistart,
                                         symmetry=bath_symmetry)
            impurity_solver = EDImpuritySolver(bath_sites=bath_sites, bath_fit=fit_options, backend=:sectorized, threading=threading)
            solver = DMFTSolver(impurity_solver=impurity_solver, mixing=mixing, tol=tolerance, maxiter=maxiter, verify_closure=true)
            problem = DMFTProblem(lattice, interaction; temperature=temperature, chemical_potential=chemical_potential, nfrequencies=nfrequencies)
            timing = @timed solve(solver, problem; initial_self_energy=initial_self_energy, initial_bath=initial_bath)
            result = timing.value
            diagnostics = validate_dmft_result(result)
            sigma_change = isnothing(previous_sigma_same_interaction) ? NaN :
                           low_frequency_self_energy_residual(previous_sigma_same_interaction, result.self_energy; nfrequencies=min(8, nfrequencies))
            z_estimate = quasiparticle_weight_estimate(result.self_energy; nfit=min(4, nfrequencies)).weight
            particle_hole_residual = diagnostics[:bath_particle_hole_residual]
            termination = result.metadata[:termination]
            closure_residual = result.metadata[:closure_residual]
            final_iteration_residual = result.metadata[:final_iteration_residual]
            best_iteration_residual = result.metadata[:best_iteration_residual]
            best_iteration = result.metadata[:iteration_of_best_residual]

            push!(rows, (; interaction, bath_sites, bath_symmetry, converged=result.converged, termination, seed_kind,
                        closure_residual, final_iteration_residual, best_iteration_residual, best_iteration,
                        bath_residual=diagnostics[:bath_discretization_residual], sigma_change, particle_hole_residual,
                        causality=diagnostics[:self_energy_causality_residual], quasiparticle_weight=z_estimate,
                        density=result.impurity_result.density, double_occupancy=result.impurity_result.double_occupancy,
                        iterations=result.iterations, final_mixing_alpha=result.metadata[:final_mixing_alpha],
                        threading_effective=result.impurity_result.metadata[:threading_effective], julia_threads=Threads.nthreads(),
                        blas_threads=BLAS.get_num_threads(), seconds=timing.time, bytes=timing.bytes))

            for entry in result.history
                push!(history_rows, (; interaction, bath_sites, iteration=entry.iteration, residual=entry.residual,
                                     weiss_residual=entry.weiss_residual, hybridization_residual=entry.hybridization_residual,
                                     bath_fit_residual=entry.bath_fit_residual, mixing_alpha=entry.mixing_alpha,
                                     density=entry.density, double_occupancy=entry.double_occupancy,
                                     bath_fit_objective=entry.bath_fit_objective, bath_fit_converged=entry.bath_fit_converged))
            end

            for i in 1:length(problem.axis)
                z = problem.axis[i]
                G = result.local_green_function[1, 1, i]
                sigma = result.self_energy[1, 1, i]
                push!(matsubara_rows, (; interaction, bath_sites, matsubara_index=problem.axis.indices[i], omega=imag(z),
                                       green_real=real(G), green_imag=imag(G), sigma_real=real(sigma), sigma_imag=imag(sigma)))
            end

            @printf("%7.3f %5d %8s %12s %11.3e %11.3e %11.3e %11.3e %11.3e %10.5f %10.6f %9d\n", interaction, bath_sites,
                    string(result.converged), string(termination), closure_residual, final_iteration_residual, best_iteration_residual,
                    diagnostics[:bath_discretization_residual], sigma_change, z_estimate, result.impurity_result.density, result.iterations)

            if result.converged
                previous_by_bath[bath_sites] = result
                previous_same_interaction = result
            end
            previous_sigma_same_interaction = result.self_energy
        end
    end

    result_directory = joinpath(@__DIR__, "results")
    mkpath(result_directory)
    output_path = joinpath(result_directory, "ed-dmft-physical-regimes.csv")
    open(output_path, "w") do io
        println(io, "U,bath_sites,bath_symmetry,converged,termination,seed_kind,closure_residual,final_iteration_residual,best_iteration_residual,best_iteration,bath_residual,low_frequency_self_energy_bath_change,bath_particle_hole_residual,self_energy_causality,quasiparticle_weight,density,double_occupancy,iterations,final_mixing_alpha,threading_effective,julia_threads,blas_threads,seconds,bytes")
        for row in rows
            println(io, join((row.interaction, row.bath_sites, row.bath_symmetry, row.converged, row.termination, row.seed_kind, row.closure_residual,
                              row.final_iteration_residual, row.best_iteration_residual, row.best_iteration, row.bath_residual, row.sigma_change, row.particle_hole_residual, row.causality,
                              row.quasiparticle_weight, row.density, row.double_occupancy, row.iterations, row.final_mixing_alpha,
                              row.threading_effective, row.julia_threads, row.blas_threads, row.seconds, row.bytes), ","))
        end
    end

    history_path = joinpath(result_directory, "ed-dmft-physical-regimes-history.csv")
    open(history_path, "w") do io
        println(io, "U,bath_sites,iteration,self_energy_residual,weiss_residual,hybridization_residual,bath_fit_residual,mixing_alpha,density,double_occupancy,bath_fit_objective,bath_fit_converged")
        for row in history_rows
            println(io, join((row.interaction, row.bath_sites, row.iteration, row.residual, row.weiss_residual, row.hybridization_residual,
                              row.bath_fit_residual, row.mixing_alpha, row.density, row.double_occupancy,
                              row.bath_fit_objective, row.bath_fit_converged), ","))
        end
    end

    matsubara_path = joinpath(result_directory, "ed-dmft-physical-regimes-matsubara.csv")
    open(matsubara_path, "w") do io
        println(io, "U,bath_sites,matsubara_index,omega,green_real,green_imag,sigma_real,sigma_imag")
        for row in matsubara_rows
            println(io, join((row.interaction, row.bath_sites, row.matsubara_index, row.omega, row.green_real, row.green_imag, row.sigma_real, row.sigma_imag), ","))
        end
    end

    println()
    println("Summary results: ", output_path)
    println("Iteration histories: ", history_path)
    println("Matsubara curves: ", matsubara_path)
    println("The reported Z estimate is a low-frequency Fermi-liquid diagnostic and should not be interpreted as a quasiparticle weight when the self-energy is non-Fermi-liquid or insulating.")
end

main()
