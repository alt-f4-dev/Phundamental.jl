#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
using .Phundamental

using Printf
using LinearAlgebra

#--------------------------------------------------------------------------------------------------#
# Liebsch, Ansgar and Hiroshi Ishida. (2012).                                                      #
# Temperature and bath size in exact diagonalization dynamical mean field theory.                   #
# J. Phys.: Condens. Matter 24, 053201. DOI: 10.1088/0953-8984/24/5/053201.                         #
#--------------------------------------------------------------------------------------------------#
# Their validation strategy emphasizes (i) self-energy convergence with bath size, (ii) quality of  #
# the finite-bath representation of the Weiss/bath Green function, and (iii) comparison with an      #
# independent impurity solver. Their bath-discretization analysis tests W_n = 1/omega_n^N for         #
# N = 0, 1, 2 and reports W_n = 1/omega_n as a reliable default.                                     #
#--------------------------------------------------------------------------------------------------#

function parse_integer_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Int, strip.(split(get(ENV, name, default), ",")))))
    isempty(values) && error("$name must contain at least one integer")
    return values
end

function parse_float_list(name::AbstractString, default::AbstractString)
    values = sort!(unique(parse.(Float64, strip.(split(get(ENV, name, default), ",")))))
    isempty(values) && error("$name must contain at least one number")
    return values
end

function weiss_from_hybridization(axis::FermionicMatsubaraAxis, delta::HybridizationFunction)
    values = ComplexF64[inv(axis[i] - delta[1, 1, i]) for i in 1:length(axis)]
    return NonInteractingGreenFunction(axis, values; labels=[:impurity])
end

betas = parse_float_list("PHUNDAMENTAL_BATH_BETAS", "4,8,16,32")
bath_sizes = parse_integer_list("PHUNDAMENTAL_BATH_SIZES", "1,2,3,4")
weight_powers = parse_float_list("PHUNDAMENTAL_BATH_WEIGHT_POWERS", "0,1,2")
nfrequencies = parse(Int, get(ENV, "PHUNDAMENTAL_BATH_NFREQUENCIES", "128"))
multistart = parse(Int, get(ENV, "PHUNDAMENTAL_BATH_MULTISTART", "7"))

all(>(0), betas) || error("PHUNDAMENTAL_BATH_BETAS must contain positive values")
all(>=(1), bath_sizes) || error("PHUNDAMENTAL_BATH_SIZES must contain positive integers")
all(>=(0), weight_powers) || error("PHUNDAMENTAL_BATH_WEIGHT_POWERS must contain nonnegative values")

lattice = BetheLattice(2.0)
rows = NamedTuple[]

println()
println("FINITE-BATH DISCRETIZATION BENCHMARK")
println("Reference: A. Liebsch and H. Ishida, J. Phys.: Condens. Matter 24, 053201 (2012), DOI: 10.1088/0953-8984/24/5/053201")
println()
@printf("%7s %5s %7s %13s %13s %13s %10s %9s %18s\n", "beta", "Nb", "weight", "Delta residual", "G0 residual", "objective", "spread", "nconv", "termination")

for beta in betas
    axis = FermionicMatsubaraAxis(beta, nfrequencies)
    sigma0 = zero_self_energy(axis)
    lattice_green_function = lattice_green(lattice, axis, sigma0)
    target_weiss = weiss_green(lattice_green_function, sigma0)
    target_delta = hybridization_from_weiss(target_weiss)

    for weight_power in weight_powers, bath_sites in bath_sizes
        options = BathFitOptions(maxiter=500, tol=1e-10, weight_power=weight_power, energy_window=(-4.0, 4.0), multistart=multistart)
        timing = @timed fit_bath(target_delta, bath_sites; options=options)
        fit = timing.value
        delta_residual = bath_discretization_residual(target_delta, fit.bath)
        fitted_delta = hybridization_function(fit.bath, axis)
        fitted_weiss = weiss_from_hybridization(axis, fitted_delta)
        weiss_residual = norm(fitted_weiss.values - target_weiss.values) / max(norm(target_weiss.values), eps(Float64))
        spread = fit.metadata[:objective_spread]
        converged_starts = count(identity, fit.metadata[:start_converged])
        selected_start = fit.metadata[:selected_start]
        selected_termination = fit.metadata[:start_termination][selected_start]
        push!(rows, (; beta, bath_sites, weight_power, delta_residual, weiss_residual, objective=fit.objective,
                    spread, converged=fit.converged, converged_starts, selected_termination, seconds=timing.time, bytes=timing.bytes))
        @printf("%7.2f %5d %7.2f %13.4e %13.4e %13.4e %10.3e %9d %18s\n", beta, bath_sites, weight_power,
                delta_residual, weiss_residual, fit.objective, spread, converged_starts, string(selected_termination))
    end
end

result_directory = joinpath(@__DIR__, "results")
mkpath(result_directory)
output_path = joinpath(result_directory, "bath-discretization.csv")
open(output_path, "w") do io
    println(io, "beta,bath_sites,weight_power,delta_residual,weiss_residual,objective,objective_spread,converged,converged_starts,selected_termination,seconds,bytes")
    for row in rows
        println(io, join((row.beta, row.bath_sites, row.weight_power, row.delta_residual, row.weiss_residual, row.objective,
                          row.spread, row.converged, row.converged_starts, row.selected_termination, row.seconds, row.bytes), ","))
    end
end

println()
println("Results: ", output_path)
println("Interpretation should be based on convergence trends with bath size and beta; the script does not impose a fitted phase or expected residual threshold.")
