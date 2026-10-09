using Test
using Phundamental

const FRC = Phundamental
const ORC = Phundamental.Observables

@testset "Dynamical response capability contract" begin
    axis = [-1.0, 0.0, 1.0]
    green = ComplexF64[-1.0 - im * pi, 0.2 - 2im * pi, 0.5 - im * pi]
    local_spectrum = FRC.spectral_function_from_retarded_green(axis, green; convention=FRC.SpectrumConvention(spectral_axis=:energy))

    @test local_spectrum.intensity[1, :] ≈ [1.0, 2.0, 1.0]
    @test FRC.response_kind(local_spectrum) == :spectral_function
    @test FRC.frequency_domain(local_spectrum) == :real
    @test FRC.response_statistics(local_spectrum) == :fermionic
    @test FRC.momentum_scope(local_spectrum) == :local
    @test FRC.response_order(local_spectrum) == 2
    @test FRC.is_dynamical_response(local_spectrum)
    @test !FRC.requires_analytic_continuation(local_spectrum)
    @test FRC.supports(local_spectrum, :dynamical_response)
    @test FRC.supports(local_spectrum, :response_statistics)

    qgrid = [0.0, pi]
    green_q = [transpose(green); transpose(2 .* green)]
    momentum_spectrum = FRC.spectral_function_from_retarded_green(qgrid, axis, green_q; convention=FRC.SpectrumConvention(spectral_axis=:energy))
    @test size(momentum_spectrum.intensity) == (2, 3)
    @test FRC.momentum_scope(momentum_spectrum) == :momentum_resolved
    @test momentum_spectrum.intensity[2, :] ≈ [2.0, 4.0, 2.0]

    faxis = FRC.FermionicMatsubaraAxis(4.0, [0, 1])
    matsubara_green = FRC.GreenFunction(faxis, ComplexF64[1 / (1im * pi / 4), 1 / (3im * pi / 4)])
    @test FRC.response_kind(matsubara_green) == :green_function
    @test FRC.frequency_domain(matsubara_green) == :matsubara
    @test FRC.response_statistics(matsubara_green) == :fermionic
    @test FRC.requires_analytic_continuation(matsubara_green)
    @test :requires_analytic_continuation in FRC.response_capabilities(matsubara_green)
end

@testset "Explicit retarded susceptibility conversions" begin
    qgrid = [0.0]
    axis = [-1.5, -0.5, 0.5, 1.5]
    imag_chi = [0.6, 0.2, -0.2, -0.6]
    values = reshape(ComplexF64[im * value for value in imag_chi], 1, :)
    metadata = Dict{Symbol,Any}(:response_kind => :susceptibility, :frequency_domain => :real, :statistics => :bosonic,
                                :momentum_scope => :momentum_resolved, :response_order => 2, :channel => :magnetic,
                                :operator_channels => (:magnetic,), :hermitian_channel => true)
    susceptibility = FRC.SusceptibilityResult(qgrid, axis, values, metadata)
    convention = FRC.SpectrumConvention(spectral_axis=:energy, hbar=1.0, kB=1.0)

    rho = FRC.commutator_spectral_density(susceptibility; convention=convention)
    @test rho.intensity[1, :] ≈ -imag_chi ./ pi
    @test FRC.response_kind(rho) == :commutator_spectral_density
    @test FRC.operator_channels(rho) == (:magnetic,)

    temperature = 1.5
    correlation = FRC.correlation_spectrum_from_retarded_susceptibility(susceptibility; temperature=temperature, convention=convention)
    expected = [-imag_chi[index] / pi / (1 - exp(-axis[index] / temperature)) for index in eachindex(axis)]
    @test correlation.intensity[1, :] ≈ expected
    @test FRC.response_kind(correlation) == :correlation_spectrum
    @test correlation.metadata[:fluctuation_dissipation_applied]

    zero_axis = [-1.0, 0.0, 1.0]
    zero_values = reshape(ComplexF64[0.2im, 0.0im, -0.2im], 1, :)
    zero_susceptibility = FRC.SusceptibilityResult(qgrid, zero_axis, zero_values, copy(metadata))
    @test_throws ArgumentError FRC.correlation_spectrum_from_retarded_susceptibility(zero_susceptibility; temperature=1.0, convention=convention)
    zero_correlation = FRC.correlation_spectrum_from_retarded_susceptibility(zero_susceptibility; temperature=1.0, convention=convention,
                                                                              zero_frequency_value=0.4)
    @test zero_correlation.intensity[1, 2] == 0.4

    bosonic_axis = FRC.BosonicMatsubaraAxis(4.0, [0, 1])
    matsubara = FRC.SusceptibilityResult(qgrid, bosonic_axis, zeros(ComplexF64, 1, 2), Dict{Symbol,Any}(:channel => :magnetic,
                                                                                                       :hermitian_channel => true))
    @test FRC.frequency_domain(matsubara) == :matsubara
    @test FRC.response_statistics(matsubara) == :bosonic
    @test FRC.requires_analytic_continuation(matsubara)
    @test_throws ArgumentError FRC.commutator_spectral_density(matsubara; convention=convention)
    @test_throws ArgumentError FRC.correlation_spectrum_from_retarded_susceptibility(matsubara; temperature=1.0, convention=convention)
end
