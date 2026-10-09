using Test
using Phundamental

@testset "momentum Green function and lattice bubble" begin
    beta = 4.0
    fermionic = FermionicMatsubaraAxis(beta, collect(-2:2))
    bosonic = BosonicMatsubaraAxis(beta, [0])
    sigma = zero_self_energy(fermionic, 1; labels=[:local])
    dispersion = [-1.0, 1.0]
    green = momentum_green(dispersion, fermionic, sigma; kpoints=[0.0, π])
    shifts = periodic_momentum_shift_maps((2,))
    bubbles, diagnostics = lattice_bubble(green, shifts.maps, bosonic; missing_frequency=:error)

    @test diagnostics.frequency_box_complete
    @test length(bubbles) == 2
    for (iq, mapping) in enumerate(shifts.maps), iν in eachindex(fermionic.indices)
        expected = -beta * sum(green.values[ik, iν] * green.values[mapping[ik], iν] for ik in 1:2) / 2
        @test bubbles[iq].values[iν, iν, 1] ≈ expected
    end

    zero_vertex = IrreducibleVertex(fermionic, bosonic, zeros(ComplexF64, length(fermionic), length(fermionic), 1); channel=:density)
    reconstructed = bethe_salpeter(bubbles[1], zero_vertex)
    @test reconstructed.values ≈ bubbles[1].values
end
