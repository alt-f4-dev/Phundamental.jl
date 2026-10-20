using Test
using LinearAlgebra

@testset "High-level harmonic phonon forward API" begin
    A = 1.0
    M = 2.0
    m = 1.0
    k = 4.0

    lattice = BravaisLattice(reshape([A], 1, 1))
    basis = BasisSite[BasisSite(:M, :M, [0.0]), BasisSite(:m, :m, [0.5])]
    crystal = LatticeCrystalStructure(lattice, basis)
    species = Dict(:M => AtomicSpecies(:M; mass=M, nuclear_scattering_length=1.0),
                   :m => AtomicSpecies(:m; mass=m, nuclear_scattering_length=1.0))
    material = Material("Diatomic phonon forward reference", crystal; species=species)
    interactions = HarmonicBondInteraction[HarmonicBondInteraction(1, 2, [0]; longitudinal=k),
                                             HarmonicBondInteraction(1, 2, [-1]; longitudinal=k)]
    model = harmonic_model(material, interactions)

    Q = [0.25, 0.75, 1.25, 1.75]
    axis = collect(range(-4.5, 4.5; length=6001))
    temperature = 1.0
    convention = SpectrumConvention(spectral_axis=:omega, hbar=1.0, kB=1.0)
    broadening = GaussianBroadening(0.03)
    solver = HarmonicPhononSolver()
    probe = NuclearNeutronProbe()
    resolution = GaussianResolution(0.05)

    mode_result = solve(solver, model, ReciprocalWaveVector([0.25]))
    physical_modes = physical_displacement_modes(mode_result)
    mass_metric = Diagonal(Float64.(model.parameters[:masses]))
    identity_modes = Matrix{ComplexF64}(I, size(physical_modes, 2), size(physical_modes, 2))

    @test size(physical_modes) == size(mode_result.modes)
    @test isapprox(physical_modes, mode_result.metadata[:physical_displacement_modes]; rtol=1e-12, atol=1e-12)
    @test isapprox(adjoint(mode_result.modes) * mode_result.modes, identity_modes; rtol=1e-12, atol=1e-12)
    @test isapprox(adjoint(physical_modes) * mass_metric * physical_modes, identity_modes; rtol=1e-12, atol=1e-12)

    request = PhononStructureFactorRequest(Q, axis; temperature=temperature, broadening=broadening, convention=convention)
    calculation = forward(ForwardProblem(model; solver=solver, observable=request, probe=probe, resolution=resolution))
    dispersion_physical_modes = physical_displacement_modes(calculation.solution)

    @test size(dispersion_physical_modes) == size(calculation.solution.modes)
    for iq in axes(calculation.solution.modes, 3)
        indexed_modes = physical_displacement_modes(calculation.solution, iq)
        @test isapprox(indexed_modes, view(dispersion_physical_modes, :, :, iq); rtol=1e-12, atol=1e-12)
        @test isapprox(adjoint(indexed_modes) * mass_metric * indexed_modes, identity_modes; rtol=1e-12, atol=1e-12)
    end

    @test calculation.solution isa PhononDispersionResult
    @test calculation.observable_result isa PhononStructureFactorResult
    @test calculation.scattering_result isa Phundamental.Observables.ScatteringResult
    @test size(calculation.solution.frequencies) == (length(Q), 2)
    @test size(calculation.observable_result.intensity) == (length(Q), length(axis), 2, 2)
    @test !calculation.observable_result.metadata[:coherent_scattering_lengths_applied]
    @test !calculation.observable_result.metadata[:neutron_Q_projection_applied]
    @test calculation.scattering_result.metadata[:coherent_scattering_lengths_applied]
    @test calculation.scattering_result.metadata[:neutron_Q_projection_applied]

    manual_scattering = one_phonon_neutron_intensity(model, Q, axis; solver=solver, probe=probe, broadening=broadening,
                                                       temperature=temperature, convention=convention)
    manual_result = convolve_resolution(resolution, manual_scattering)

    @test Phundamental.Observables.validate_observable_equivalence(calculation.scattering_result, manual_scattering; rtol=1e-10, atol=1e-12)
    @test Phundamental.Observables.validate_observable_equivalence(calculation.result, manual_result; rtol=1e-10, atol=1e-12)

    expected_qprime = [[-0.25], [0.25], [-0.25], [0.25]]
    actual_qprime = calculation.observable_result.metadata[:eigenvector_qprime_reciprocal]
    @test all(isapprox(actual, expected; atol=1e-12, rtol=0.0) for (actual, expected) in zip(actual_qprime, expected_qprime))

    single_request = PhononStructureFactorRequest(0.25, axis; temperature=0.0, broadening=broadening, convention=convention)
    single_calculation = forward(ForwardProblem(model; solver=solver, observable=single_request, probe=probe))
    single_manual = one_phonon_neutron_intensity(model, 0.25, axis; solver=solver, probe=probe, broadening=broadening,
                                                   temperature=0.0, convention=convention)
    @test Phundamental.Observables.validate_observable_equivalence(single_calculation.scattering_result, single_manual;
                                                                      rtol=1e-10, atol=1e-12)

    reference_lines = one_phonon_neutron_lines(model, 0.25; solver=solver, probe=probe, temperature=0.0, convention=convention)
    @test isapprox(reference_lines.centers, [1.2360679774997898, 3.23606797749979]; rtol=1e-12, atol=1e-12)
    @test isapprox(real.(reference_lines.weights), [1.268218467143, 0.087435304654]; rtol=1e-11, atol=1e-12)

    @testset "F7 phonon spectrum-to-experiment portability" begin
        prediction = portable_prediction(calculation; stage=:instrument, coordinate_frame=:reciprocal,
                                         coordinate_unit="rlu", spectral_unit="rad/ps", intensity_unit="arb",
                                         normalization="one_phonon", spectral_kind=:omega)
        @test prediction.stage === :instrument
        @test prediction.grid.spectral_kind === :omega
        @test prediction.grid.coordinate_frame === :reciprocal
        @test prediction.intensity ≈ manual_result.intensity
        measured = experimental_dataset(prediction.grid, Float64.(real.(prediction.intensity));
                                        uncertainty=fill(0.1, size(prediction.intensity)),
                                        intensity_unit="arb", normalization="one_phonon")
        @test isapprox(compare_measurement(prediction, measured).chi_squared, 0.0; atol=1e-10)
        @test compare_measurement(import_result(export_result(prediction)), import_result(export_result(measured))).chi_squared ≈ 0.0 atol=1e-10
        @test_throws ArgumentError portable_prediction(calculation; stage=:instrument, coordinate_frame=:reciprocal,
            coordinate_unit="rlu", spectral_unit="rad/ps", intensity_unit="arb", normalization="one_phonon", spectral_kind=:energy)
        equivalent = validate_forward_equivalence(calculation, calculation; compare=(:intrinsic, :probe, :instrument))
        @test equivalent.valid
        intrinsic = calculation.observable_result
        copied_intrinsic = Phundamental.Observables.PhononStructureFactorResult(intrinsic.model, copy(intrinsic.q),
            copy(intrinsic.axis), copy(intrinsic.intensity), copy(intrinsic.metadata))
        @test Phundamental.Observables.validate_observable_equivalence(intrinsic, copied_intrinsic)
        @test !Phundamental.Observables.validate_observable_equivalence(intrinsic,
            Phundamental.Observables.PhononStructureFactorResult(intrinsic.model, copy(intrinsic.q),
                copy(intrinsic.axis), 1.01 .* intrinsic.intensity, copy(intrinsic.metadata)))
        @test !validate_forward_equivalence(calculation, calculation; compare=(:spectrum,)).valid
        @test_throws ArgumentError validate_forward_equivalence(calculation, calculation; compare=())
    end

end
