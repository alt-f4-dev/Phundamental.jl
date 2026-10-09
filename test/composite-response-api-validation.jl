using Test
using LinearAlgebra
using Phundamental

const FCR = Phundamental

function _composite_response_fixture()
    lattice = FCR.BravaisLattice(reshape([2.0], 1, 1))
    properties = FCR.SiteProperties(spin=0.5, g_tensor=2.0)
    basis = FCR.BasisSite[
        FCR.BasisSite(:M1, :M, [0.0]; properties=properties),
        FCR.BasisSite(:M2, :M, [0.5]; properties=properties),
    ]
    crystal = FCR.LatticeCrystalStructure(lattice, basis)
    cluster = FCR.Supercell(crystal, (1,); periodic=false)
    material = FCR.Material("composite response fixture", crystal; species=Dict{Symbol,Any}(:M => FCR.AtomicSpecies(:M; mass=1.0)),
                            basis_properties=[properties, properties])
    specification = FCR.LongitudinalSpinBosonModel(0.43, 0.67, 0.91, 0.23; field_z=0.08, phonon_cutoff=1, shell=1)
    return FCR.build_model(material, cluster, specification)
end

@testset "Composite and spin-phonon response API" begin
    model = _composite_response_fixture()
    solver = FCR.ExactDiagonalization()
    solution = FCR.solve(solver, model)
    positions = model.parameters[:positions]
    qgrid = [[0.0], [π]]
    axis = collect(range(-3.0, 3.0; length=401))
    convention = FCR.SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=1.0)
    broadening = FCR.GaussianBroadening(0.025)

    spin_z = FCR.registered_momentum_field(model, [:spin_z_1, :spin_z_2]; normalization=:sqrtN)
    displacement = FCR.registered_momentum_field(model, [:boson_displacement_1, :boson_displacement_2]; normalization=:sqrtN)
    boson_number = FCR.registered_momentum_field(model, [:boson_number_1, :boson_number_2]; normalization=:sqrtN)
    @test spin_z.positions == positions
    @test_throws DimensionMismatch FCR.registered_momentum_field(model, [:spin_z_1]; normalization=:sqrtN)
    @test_throws KeyError FCR.registered_momentum_field(model, [:spin_z_1, :missing_observable]; normalization=:sqrtN)

    fields = (spin_z=spin_z, displacement=displacement, boson_number=boson_number)
    composite = FCR.composite_dynamical_correlation(model, solution, fields, qgrid, axis; convention=convention, broadening=broadening)
    @test composite.channels == (:spin_z, :displacement, :boson_number)
    @test size(composite.intensity) == (length(qgrid), length(axis), 3, 3)
    @test composite.metadata[:definition] == :S_AB_matrix
    @test composite.metadata[:operator_order] == :row_channel_then_column_channel
    @test FCR.response_kind(composite) == :correlation_spectrum
    @test FCR.frequency_domain(composite) == :real
    @test FCR.momentum_scope(composite) == :momentum_resolved
    @test FCR.operator_channels(composite) == (:spin_z, :displacement, :boson_number)

    field_values = values(fields)
    for a in eachindex(field_values), b in eachindex(field_values)
        pair = FCR.dynamical_correlation(model, solution, field_values[a], field_values[b], qgrid, axis;
                                         convention=convention, broadening=broadening)
        @test isapprox(composite.intensity[:, :, a, b], pair.intensity; rtol=1e-12, atol=1e-12)
    end

    request = FCR.CompositeDynamicalCorrelationRequest(fields, qgrid, axis; convention=convention, broadening=broadening,
                                                        metadata=Dict(:purpose => :arbitrary_composite_validation))
    calculation = FCR.forward(FCR.ForwardProblem(model; solver=solver, observable=request))
    @test FCR.Observables.validate_observable_equivalence(calculation.observable_result, composite; rtol=1e-12, atol=1e-12)
    @test calculation.observable_result.metadata[:observable_request] == :composite_dynamical_correlation
    @test calculation.observable_result.metadata[:request_metadata][:purpose] == :arbitrary_composite_validation

    resolution = FCR.GaussianResolution(0.04)
    resolved = FCR.forward(FCR.ForwardProblem(model; solver=solver, observable=request, resolution=resolution))
    manual_resolved = FCR.Observables.convolve_resolution(resolution, composite)
    @test FCR.Observables.validate_observable_equivalence(resolved.result, manual_resolved; rtol=1e-12, atol=1e-12)
    @test resolved.result.metadata[:resolution_applied]

    spin_field = FCR.SpinTensorField(
        xkeys=[:spin_x_1, :spin_x_2],
        ykeys=[:spin_y_1, :spin_y_2],
        zkeys=[:spin_z_1, :spin_z_2],
        positions=positions,
        normalization=:sqrtN,
    )
    spin_phonon = FCR.spin_phonon_dynamical_correlation(model, solution, spin_field, displacement, qgrid, axis;
                                                        spin_components=(:z,), convention=convention, broadening=broadening)
    expected_spin_phonon = FCR.composite_dynamical_correlation(model, solution, (spin_z=spin_z, phonon_displacement=displacement), qgrid, axis;
                                                               convention=convention, broadening=broadening)
    @test spin_phonon.channels == (:spin_z, :phonon_displacement)
    @test FCR.Observables.validate_observable_equivalence(spin_phonon, expected_spin_phonon; rtol=1e-12, atol=1e-12)
    @test spin_phonon.metadata[:response_family] == :spin_phonon
    @test FCR.response_statistics(spin_phonon) == :bosonic

    spin_phonon_request = FCR.SpinPhononCorrelationRequest(qgrid, axis; spin_components=(:z,), convention=convention,
                                                           broadening=broadening, metadata=Dict(:channel_set => :longitudinal))
    spin_phonon_forward = FCR.forward(FCR.ForwardProblem(model; solver=solver, observable=spin_phonon_request))
    @test FCR.Observables.validate_observable_equivalence(spin_phonon_forward.observable_result, spin_phonon; rtol=1e-12, atol=1e-12)
    @test spin_phonon_forward.observable_result.metadata[:observable_request] == :spin_phonon_correlation
    @test spin_phonon_forward.observable_result.metadata[:displacement_prefix] == :boson_displacement
    @test spin_phonon_forward.observable_result.metadata[:request_metadata][:channel_set] == :longitudinal

    full_request = FCR.SpinPhononCorrelationRequest(qgrid, axis; convention=convention, broadening=broadening)
    full_response = FCR.forward(FCR.ForwardProblem(model; solver=solver, observable=full_request)).observable_result
    @test full_response.channels == (:spin_x, :spin_y, :spin_z, :phonon_displacement)
    @test size(full_response.intensity) == (length(qgrid), length(axis), 4, 4)
    @test size(FCR.Observables.integrated_spectral_weight(full_response)) == (length(qgrid), 4, 4)

    holstein_model = FCR.build_model(model.parameters[:material], model.parameters[:cluster],
                                     FCR.HolsteinModel(0.2, 0.0, 1.0, 0.1; phonon_cutoff=0, shell=1))
    holstein_request = FCR.SpinPhononCorrelationRequest(qgrid, axis; spin_components=:z, convention=convention, broadening=broadening)
    holstein_response = FCR.forward(FCR.ForwardProblem(holstein_model; solver=solver, observable=holstein_request)).observable_result
    @test holstein_response.channels == (:spin_z, :phonon_displacement)
    @test holstein_response.metadata[:displacement_prefix] == :phonon_displacement
end

@testset "Quadratic-mode composite response" begin
    representation = FCR.Representation(:composite_response_boson, FCR.BosonFockSpace(1), FCR.BosonAlgebra(1), FCR.BosonOccupationBasis([1]);
                                        ordering=[1], reference_state=:boson_vacuum)
    hamiltonian = 1.4 * FCR.nb(1) + 0.2 * FCR.b(1)' * FCR.b(1)' + 0.2 * FCR.b(1) * FCR.b(1)
    model = FCR.ManyBodyModel(representation, hamiltonian)
    solver = FCR.QuadraticBosonSolver(tol=1e-12)
    solution = FCR.solve(solver, model)
    field_x = FCR.MomentumField([FCR.b(1) + FCR.b(1)'], [0.0]; normalization=:none)
    field_p = FCR.MomentumField([FCR.b(1)' - FCR.b(1)], [0.0]; normalization=:none)
    fields = (position=field_x, momentum=field_p)
    qgrid = [0.0]
    axis = collect(range(-2.0, 2.0; length=401))
    convention = FCR.SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=1.0, kB=1.0)
    broadening = FCR.GaussianBroadening(0.02)

    composite = FCR.composite_dynamical_correlation(model, solution, fields, qgrid, axis; convention=convention, broadening=broadening)
    @test composite.channels == (:position, :momentum)
    @test composite.metadata[:mode_native]
    for a in 1:2, b in 1:2
        pair = FCR.dynamical_correlation(model, solution, values(fields)[a], values(fields)[b], qgrid, axis;
                                         convention=convention, broadening=broadening)
        @test isapprox(composite.intensity[:, :, a, b], pair.intensity; rtol=1e-12, atol=1e-12)
    end

    request = FCR.CompositeDynamicalCorrelationRequest(fields, qgrid, axis; convention=convention, broadening=broadening)
    forward_result = FCR.forward(FCR.ForwardProblem(model; solver=solver, observable=request)).observable_result
    @test FCR.Observables.validate_observable_equivalence(forward_result, composite; rtol=1e-12, atol=1e-12)
end
