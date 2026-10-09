using Test
using LinearAlgebra: I
using Phundamental

@testset "two-particle Bethe-Salpeter API" begin
    faxis = FermionicMatsubaraAxis(4.0, [-1, 0])
    baxis = BosonicMatsubaraAxis(4.0, [0])
    χ0values = zeros(ComplexF64, 2, 2, 1)
    χ0values[:, :, 1] .= Matrix{ComplexF64}(I, 2, 2)
    χvalues = zeros(ComplexF64, 2, 2, 1)
    χvalues[:, :, 1] .= 2 .* Matrix{ComplexF64}(I, 2, 2)
    bubble = GeneralizedSusceptibility(faxis, baxis, χ0values; channel=:spin)
    χ = GeneralizedSusceptibility(faxis, baxis, χvalues; channel=:spin)
    Γ = irreducible_vertex(bubble, χ)
    reconstructed = bethe_salpeter(bubble, Γ)
    @test reconstructed.values ≈ χ.values
    @test physical_susceptibility(reconstructed; beta_normalization=16.0)[1] ≈ 0.25
end
