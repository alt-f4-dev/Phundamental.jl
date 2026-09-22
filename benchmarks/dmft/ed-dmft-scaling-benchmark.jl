#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
using .Phundamental

using LinearAlgebra
using Printf

#--------------------------------------------------------------------------------------------------#
# Liebsch, Ansgar and Hiroshi Ishida. (2012).                                                      #
# Temperature and bath size in exact diagonalization dynamical mean field theory.                   #
# J. Phys.: Condens. Matter 24, 053201. DOI: 10.1088/0953-8984/24/5/053201.                         #
#--------------------------------------------------------------------------------------------------#
# Caffarel, Michel and Werner Krauth. (1994). Phys. Rev. Lett. 72, 1545-1548.                       #
# DOI: 10.1103/PhysRevLett.72.1545                                                                 #
#--------------------------------------------------------------------------------------------------#
# This benchmark separates bath fitting, the retained dense reference impurity solve, serial and    #
# threaded spin-sector impurity solves, and complete DMFT-driver costs. Base.@timed avoids a new     #
# dependency. Run threaded measurements with BLAS threads set to one to avoid nested oversubscription.#
#--------------------------------------------------------------------------------------------------#

function parse_integer_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Int, strip.(split(get(ENV, name, default), ",")))))
    isempty(values) && error("$name must contain at least one integer")
    return values
end

function benchmark_call(action::Function, repeats::Int)
    best_time = Inf
    best_bytes = typemax(Int)
    for _ in 1:repeats
        GC.gc()
        timing = @timed action()
        if timing.time < best_time
            best_time = timing.time
            best_bytes = timing.bytes
        end
    end
    return (seconds=best_time, bytes=best_bytes)
end

function benchmark_bath(bath_sites::Int)
    bath_sites >= 0 || throw(ArgumentError("bath_sites must be nonnegative"))
    bath_sites == 0 && return DiscreteBath(Float64[], ComplexF64[])
    energies = bath_sites == 1 ? [0.0] : collect(range(-1.5, 1.5; length=bath_sites))
    coupling = 0.7 / sqrt(bath_sites)
    return DiscreteBath(energies, fill(coupling, bath_sites))
end

function full_hilbert_dimension(bath_sites::Int)
    return 2^(2 * (bath_sites + 1))
end

function maximum_sector_dimension(bath_sites::Int)
    spatial_orbitals = bath_sites + 1
    center = fld(spatial_orbitals, 2)
    return binomial(spatial_orbitals, center)^2
end

bath_sizes = parse_integer_list("PHUNDAMENTAL_SCALING_BATH_SIZES", "0,1,2,3,4,5")
frequency_counts = parse_integer_list("PHUNDAMENTAL_SCALING_NFREQUENCIES", "16,32,64,128")
repeats = parse(Int, get(ENV, "PHUNDAMENTAL_SCALING_REPEATS", "1"))
dense_max_bath = parse(Int, get(ENV, "PHUNDAMENTAL_SCALING_DENSE_MAX_BATH", "3"))
dmft_max_bath = parse(Int, get(ENV, "PHUNDAMENTAL_SCALING_DMFT_MAX_BATH", "4"))
beta = parse(Float64, get(ENV, "PHUNDAMENTAL_SCALING_BETA", "8"))
all(>=(0), bath_sizes) || error("PHUNDAMENTAL_SCALING_BATH_SIZES values must be nonnegative")
all(>(0), frequency_counts) || error("PHUNDAMENTAL_SCALING_NFREQUENCIES values must be positive")
repeats > 0 || error("PHUNDAMENTAL_SCALING_REPEATS must be positive")
dense_max_bath >= 0 || error("PHUNDAMENTAL_SCALING_DENSE_MAX_BATH must be nonnegative")
dmft_max_bath >= 0 || error("PHUNDAMENTAL_SCALING_DMFT_MAX_BATH must be nonnegative")
beta > 0 || error("PHUNDAMENTAL_SCALING_BETA must be positive")

julia_threads = Threads.nthreads()
blas_threads = BLAS.get_num_threads()
threaded_available = julia_threads > 1

# Warm up each timed code path so compilation is not interpreted as algorithmic cost.
warm_axis = FermionicMatsubaraAxis(beta, 8)
warm_bath = benchmark_bath(1)
warm_problem = ImpurityProblem(1.0, warm_axis; chemical_potential=0.5, bath=warm_bath)
solve(EDImpuritySolver(bath_sites=1, backend=:dense), warm_problem)
solve(EDImpuritySolver(bath_sites=1, backend=:sectorized, threading=:serial), warm_problem)
threaded_available && solve(EDImpuritySolver(bath_sites=1, backend=:sectorized, threading=:threads), warm_problem)
warm_sigma = zero_self_energy(warm_axis)
warm_lattice_green = lattice_green(BetheLattice(2.0), warm_axis, warm_sigma)
warm_target = hybridization_from_weiss(weiss_green(warm_lattice_green, warm_sigma))
fit_bath(warm_target, 1; options=BathFitOptions(maxiter=3, tol=1e-6, energy_window=(-3.0, 3.0), multistart=1))
warm_fit = BathFitOptions(maxiter=3, tol=1e-6, energy_window=(-3.0, 3.0), multistart=1)
warm_serial_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=1, bath_fit=warm_fit, backend=:sectorized, threading=:serial),
                                mixing=LinearMixing(1.0), tol=1e-4, maxiter=1, verify_closure=false)
warm_dmft_problem = DMFTProblem(BetheLattice(2.0), 0.0; temperature=inv(beta), chemical_potential=0.0, nfrequencies=8)
solve(warm_serial_solver, warm_dmft_problem)
if threaded_available
    warm_threaded_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=1, bath_fit=warm_fit, backend=:sectorized, threading=:threads),
                                      mixing=LinearMixing(1.0), tol=1e-4, maxiter=1, verify_closure=false)
    solve(warm_threaded_solver, warm_dmft_problem)
end

rows = NamedTuple[]

println()
println("ED-DMFT COMPONENT SCALING BENCHMARK")
println("Reference context: Caffarel-Krauth 1994; Liebsch-Ishida 2012")
@printf("Julia threads = %d, BLAS threads = %d\n", julia_threads, blas_threads)
threaded_available || println("Threaded measurements are skipped because Julia was started with one thread.")
threaded_available && blas_threads > 1 && println("WARNING: BLAS has more than one thread; use OPENBLAS_NUM_THREADS=1 (or the corresponding BLAS control) for clean Julia-thread scaling.")
println()
@printf("%-22s %-9s %5s %7s %10s %10s %12s %9s %14s\n", "component", "mode", "Nb", "Nw", "Hilbert", "maxsector", "seconds", "speedup", "bytes")

for nfrequencies in frequency_counts
    axis = FermionicMatsubaraAxis(beta, nfrequencies)
    sigma0 = zero_self_energy(axis)
    lattice_green_function = lattice_green(BetheLattice(2.0), axis, sigma0)
    target = hybridization_from_weiss(weiss_green(lattice_green_function, sigma0))
    for bath_sites in filter(>(0), bath_sizes)
        options = BathFitOptions(maxiter=300, tol=1e-9, weight_power=1.0, energy_window=(-4.0, 4.0), multistart=5)
        measurement = benchmark_call(() -> fit_bath(target, bath_sites; options=options), repeats)
        push!(rows, (; component="bath_fit", mode="serial", bath_sites, nfrequencies, hilbert_dimension=0, max_sector_dimension=0,
                      julia_threads, blas_threads, speedup=NaN, measurement...))
        @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9s %14d\n", "bath_fit", "serial", bath_sites, nfrequencies, 0, 0,
                measurement.seconds, "-", measurement.bytes)
    end
end

fixed_nfrequencies = maximum(frequency_counts)
axis = FermionicMatsubaraAxis(beta, fixed_nfrequencies)
for bath_sites in bath_sizes
    bath = benchmark_bath(bath_sites)
    problem = ImpurityProblem(2.0, axis; chemical_potential=1.0, bath=bath)
    hilbert_dimension = full_hilbert_dimension(bath_sites)
    sector_dimension = maximum_sector_dimension(bath_sites)

    if bath_sites <= dense_max_bath
        dense_measurement = benchmark_call(() -> solve(EDImpuritySolver(bath_sites=bath_sites, backend=:dense), problem), repeats)
        push!(rows, (; component="dense_impurity", mode="serial", bath_sites, nfrequencies=fixed_nfrequencies, hilbert_dimension,
                      max_sector_dimension=hilbert_dimension, julia_threads, blas_threads, speedup=NaN, dense_measurement...))
        @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9s %14d\n", "dense_impurity", "serial", bath_sites, fixed_nfrequencies,
                hilbert_dimension, hilbert_dimension, dense_measurement.seconds, "-", dense_measurement.bytes)
    end

    serial_measurement = benchmark_call(() -> solve(EDImpuritySolver(bath_sites=bath_sites, backend=:sectorized, threading=:serial), problem), repeats)
    push!(rows, (; component="sectorized_impurity", mode="serial", bath_sites, nfrequencies=fixed_nfrequencies, hilbert_dimension,
                  max_sector_dimension=sector_dimension, julia_threads, blas_threads, speedup=1.0, serial_measurement...))
    @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9.3f %14d\n", "sectorized_impurity", "serial", bath_sites,
            fixed_nfrequencies, hilbert_dimension, sector_dimension, serial_measurement.seconds, 1.0, serial_measurement.bytes)

    if threaded_available
        threaded_measurement = benchmark_call(() -> solve(EDImpuritySolver(bath_sites=bath_sites, backend=:sectorized, threading=:threads), problem), repeats)
        speedup = serial_measurement.seconds / threaded_measurement.seconds
        push!(rows, (; component="sectorized_impurity", mode="threads", bath_sites, nfrequencies=fixed_nfrequencies, hilbert_dimension,
                      max_sector_dimension=sector_dimension, julia_threads, blas_threads, speedup, threaded_measurement...))
        @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9.3f %14d\n", "sectorized_impurity", "threads", bath_sites,
                fixed_nfrequencies, hilbert_dimension, sector_dimension, threaded_measurement.seconds, speedup, threaded_measurement.bytes)
    end
end

for bath_sites in filter(<=(dmft_max_bath), bath_sizes)
    fit_options = BathFitOptions(maxiter=200, tol=1e-8, energy_window=(-3.0, 3.0), multistart=3)
    problem = DMFTProblem(BetheLattice(2.0), 1.0; temperature=inv(beta), chemical_potential=0.5, nfrequencies=fixed_nfrequencies)
    hilbert_dimension = full_hilbert_dimension(bath_sites)
    sector_dimension = maximum_sector_dimension(bath_sites)

    serial_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=bath_sites, bath_fit=fit_options, backend=:sectorized, threading=:serial),
                               mixing=LinearMixing(0.5), tol=1e-7, maxiter=2, verify_closure=false)
    serial_measurement = benchmark_call(() -> solve(serial_solver, problem), repeats)
    push!(rows, (; component="dmft_driver", mode="serial", bath_sites, nfrequencies=fixed_nfrequencies, hilbert_dimension,
                  max_sector_dimension=sector_dimension, julia_threads, blas_threads, speedup=1.0, serial_measurement...))
    @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9.3f %14d\n", "dmft_driver", "serial", bath_sites, fixed_nfrequencies,
            hilbert_dimension, sector_dimension, serial_measurement.seconds, 1.0, serial_measurement.bytes)

    if threaded_available
        threaded_solver = DMFTSolver(impurity_solver=EDImpuritySolver(bath_sites=bath_sites, bath_fit=fit_options, backend=:sectorized, threading=:threads),
                                     mixing=LinearMixing(0.5), tol=1e-7, maxiter=2, verify_closure=false)
        threaded_measurement = benchmark_call(() -> solve(threaded_solver, problem), repeats)
        speedup = serial_measurement.seconds / threaded_measurement.seconds
        push!(rows, (; component="dmft_driver", mode="threads", bath_sites, nfrequencies=fixed_nfrequencies, hilbert_dimension,
                      max_sector_dimension=sector_dimension, julia_threads, blas_threads, speedup, threaded_measurement...))
        @printf("%-22s %-9s %5d %7d %10d %10d %12.5f %9.3f %14d\n", "dmft_driver", "threads", bath_sites, fixed_nfrequencies,
                hilbert_dimension, sector_dimension, threaded_measurement.seconds, speedup, threaded_measurement.bytes)
    end
end

result_directory = joinpath(@__DIR__, "results")
mkpath(result_directory)
output_path = joinpath(result_directory, "ed-dmft-scaling.csv")
open(output_path, "w") do io
    println(io, "component,mode,bath_sites,nfrequencies,hilbert_dimension,max_sector_dimension,julia_threads,blas_threads,seconds,speedup,bytes")
    for row in rows
        println(io, join((row.component, row.mode, row.bath_sites, row.nfrequencies, row.hilbert_dimension, row.max_sector_dimension,
                          row.julia_threads, row.blas_threads, row.seconds, row.speedup, row.bytes), ","))
    end
end

println()
println("Results: ", output_path)
println("Dense-reference timings are intentionally capped by PHUNDAMENTAL_SCALING_DENSE_MAX_BATH; compare serial and threaded sectorized rows at fixed Nb/Nw.")
println("For clean explicit Julia-thread scaling, use BLAS threads = 1 and launch Julia with more than one thread.")
