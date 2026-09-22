#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module FiniteTemperatureGreensValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

@testset "Finite-temperature Greens API" begin
    beta = 5.0
    axis = P.FermionicMatsubaraAxis(beta, 48)
    specification = P.AndersonImpurityModel(0.0; chemical_potential=0.3, impurity_energy=0.1)
    model = P.build_model(specification)
    green = P.matsubara_green(model, P.ExactGibbs(); axis=axis, modes=[1])
    expected = ComplexF64[inv(z + 0.3 - 0.1) for z in axis]

    @test P.scalar_frequency_values(green) ≈ expected atol=2e-12 rtol=2e-12
    @test P.causality_residual(green) <= 1e-12
    @test P.zeroth_moment_residual(green; ntail=8) < 5e-3

    green0 = P.noninteracting_green(0.1, axis; chemical_potential=0.3)
    sigma = P.self_energy(green0, green)
    @test maximum(abs.(sigma.values)) <= 5e-12
    @test P.dyson_green(green0, sigma).values ≈ green.values atol=5e-12 rtol=5e-12
end

@testset "Nambu indexing and symmetry" begin
    beta = 4.0
    axis = P.FermionicMatsubaraAxis(beta, 24; include_negative=true)
    model = P.build_model(P.AndersonImpurityModel(0.0; chemical_potential=0.0, impurity_energy=0.35))
    green = P.nambu_green(model, P.ExactGibbs(); axis=axis, modes=[1])

    @test P.matrix_dimension(green) == 2
    @test P.anomalous_norm(green) <= 2e-12
    @test P.nambu_symmetry_residual(green) <= 5e-12
    @test P.matsubara_conjugation_residual(green) <= 5e-12
end

end
