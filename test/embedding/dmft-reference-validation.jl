#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module DMFTReferenceValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

# Reference:
# M. Caffarel and W. Krauth, "Exact diagonalization approach to correlated fermions in infinite dimensions: Mott transition and superconductivity," Phys. Rev. Lett. 72, 1545-1548 (1994), DOI: 10.1103/PhysRevLett.72.1545.
# The extended derivation is arXiv:cond-mat/9306057. Their Bethe normalization gives G0^-1(iω)=iω+μ-(1/2)G(iω), corresponding here to half-bandwidth D=sqrt(2).

@testset "Caffarel-Krauth Bethe self-consistency normalization" begin
    beta = 12.0
    U = 2.0
    mu = U / 2
    axis = P.FermionicMatsubaraAxis(beta, 64)
    lattice = P.BetheLattice(sqrt(2.0))
    sigma = P.atomic_hubbard_self_energy(axis, U)
    lattice_green_function = P.lattice_green(lattice, axis, sigma; chemical_potential=mu)
    weiss = P.weiss_green(lattice_green_function, sigma)
    hybridization = P.hybridization_from_weiss(weiss; chemical_potential=mu)

    @test P.bethe_self_consistency_residual(lattice, lattice_green_function, weiss; chemical_potential=mu) <= 2e-12
    @test maximum(abs.(P.scalar_frequency_values(hybridization) .- 0.5 .* P.scalar_frequency_values(lattice_green_function))) <= 2e-12
end

@testset "Bethe noninteracting analytic closure" begin
    axis = P.FermionicMatsubaraAxis(10.0, 64)
    lattice = P.BetheLattice(sqrt(2.0))
    sigma = P.zero_self_energy(axis)
    lattice_green_function = P.lattice_green(lattice, axis, sigma)
    hopping_squared = lattice.half_bandwidth^2 / 4

    quadratic_residual = maximum(abs.(hopping_squared .* P.scalar_frequency_values(lattice_green_function).^2 .-
                                      axis.values .* P.scalar_frequency_values(lattice_green_function) .+ 1))
    @test quadratic_residual <= 2e-12
    @test P.causality_residual(lattice_green_function) <= 2e-12
end

end
