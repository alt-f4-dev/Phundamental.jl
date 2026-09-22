#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental
if !isdefined(Main, :Phundamental)
    include(joinpath(@__DIR__, "..", "..", "src", "Phundamental.jl"))
end

module GreensIdentitiesValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

# References:
# A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, Rev. Mod. Phys. 68, 13-125 (1996), DOI: 10.1103/RevModPhys.68.13.
# A. Liebsch and H. Ishida, J. Phys.: Condens. Matter 24, 053201 (2012), DOI: 10.1088/0953-8984/24/5/053201.

@testset "Atomic Hubbard Green function and self-energy" begin
    U = 2.4
    beta = 8.0
    axis = P.FermionicMatsubaraAxis(beta, 64; include_negative=true)
    model = P.build_model(P.AndersonImpurityModel(U; chemical_potential=U / 2, impurity_energy=0.0))
    green = P.matsubara_green(model, P.ExactGibbs(); axis=axis, modes=[1])
    expected_green = P.atomic_hubbard_green(axis, U)
    green0 = P.noninteracting_green(0.0, axis; chemical_potential=U / 2)
    sigma = P.self_energy(green0, green)
    expected_sigma = P.atomic_hubbard_self_energy(axis, U)

    @test green.values ≈ expected_green.values atol=2e-12 rtol=2e-12
    @test sigma.values ≈ expected_sigma.values atol=2e-11 rtol=2e-11
    @test P.low_frequency_self_energy_residual(expected_sigma, sigma; nfrequencies=8) <= 2e-11
    @test P.causality_residual(green) <= 2e-12
    @test P.causality_residual(sigma) <= 2e-11
    @test P.matsubara_conjugation_residual(green) <= 2e-12
    @test P.matsubara_conjugation_residual(sigma) <= 2e-11
    @test P.particle_hole_symmetry_residual(green) <= 2e-12
    @test P.particle_hole_symmetry_residual(sigma; center=U / 2) <= 2e-11
end

@testset "High-frequency moment sum rules" begin
    U = 2.4
    axis = P.FermionicMatsubaraAxis(8.0, 512)
    atomic_green = P.atomic_hubbard_green(axis, U)
    moments = P.high_frequency_moments(atomic_green; order=2, ntail=128)

    @test abs(moments[1][1, 1] - 1) < 2e-8
    @test abs(moments[2][1, 1]) < 1e-5
    @test abs(moments[3][1, 1] - U^2 / 4) < 2e-4

    noninteracting = P.noninteracting_green(0.1, axis; chemical_potential=0.3)
    noninteracting_green = P.GreenFunction(axis, noninteracting.values; labels=[:test])
    noninteracting_moments = P.high_frequency_moments(noninteracting_green; order=2, ntail=128)
    @test abs(noninteracting_moments[1][1, 1] - 1) < 2e-8
    @test abs(noninteracting_moments[2][1, 1] - (0.1 - 0.3)) < 2e-6
end

end
