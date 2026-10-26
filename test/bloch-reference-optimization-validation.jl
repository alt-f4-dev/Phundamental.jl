using Test
using LinearAlgebra
using Phundamental

function _bloch_optimization_fixture(S; exchange=-0.25 .* Matrix{Float64}(I, 3, 3), field=(0.0, 0.0, 2.0), anisotropy=nothing)
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    properties = SiteProperties(spin=S, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0]; properties)])
    cluster = Supercell(crystal, (2,); periodic=true)
    material = Material("Bloch-reference fixture", crystal; species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    return build_model(material, cluster, TensorSpinModel(exchange; field, anisotropy, shell=1))
end

@testset "Periodic Bloch tensors and magnetic reference minimization" begin
    z = MagneticReferenceState([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    Q = [[0.0], [π / 3]]
    energy = collect(range(-5.0, 5.0; length=501))
    dipole_model = _bloch_optimization_fixture(0.5)
    solver = SpinWaveSolver(reference=z; formalism=:dipole)
    bands = spinwave_dispersion(solver, dipole_model, Q)
    @test bands isa SpinWaveDispersionResult
    @test size(bands.frequencies) == (2, 2)
    @test all(>(0), bands.frequencies)
    @test bands.metadata[:scattering_tensors_from_bloch_modes] === :elastic_basis_and_one_magnon

    request = SpinTensorStructureFactorRequest(Q, energy; metadata=Dict(:spinwave_momentum_scope => :magnetic_supercell_bloch))
    field = Phundamental.Observables._spin_tensor_request_field(request, dipole_model)
    tensor = spin_tensor_structure_factor(dipole_model, bands, field, Q, energy; channels=(:one_magnon,))
    @test size(tensor.intensity) == (2, length(energy), 3, 3)
    @test tensor.metadata[:momentum_scope] === :magnetic_supercell_bloch
    @test maximum(abs, tensor.intensity) > 0
    @test all(isfinite, tensor.intensity)
    for iq in eachindex(Q), ix in eachindex(energy)
        @test isapprox(tensor.intensity[iq, ix, 1, 2], conj(tensor.intensity[iq, ix, 2, 1]); atol=1e-9)
    end
    @test_throws ArgumentError spin_tensor_structure_factor(dipole_model, bands, field, [[π / 5]], energy)
    # At Q=0 the position-phase Bloch gauge must reproduce the finite-cell one-magnon tensor.
    zero_Q = [[0.0]]
    zero_field = Phundamental.Observables._spin_tensor_request_field(SpinTensorStructureFactorRequest(zero_Q, energy), dipole_model)
    bloch_zero = spin_tensor_structure_factor(dipole_model, spinwave_dispersion(solver, dipole_model, zero_Q), zero_field,
        zero_Q, energy; channels=(:one_magnon,))
    finite_zero = spin_tensor_structure_factor(dipole_model, solve(solver, dipole_model), zero_field, zero_Q, energy;
        channels=(:one_magnon,))
    @test isapprox(bloch_zero.intensity, finite_zero.intensity; atol=1e-8)

    thermal = spin_tensor_structure_factor(dipole_model, bands, field, Q, energy; temperature=0.8, channels=(:one_magnon,))
    @test all(isfinite, thermal.intensity)
    @test maximum(abs, thermal.intensity[1, 1:245, :, :]) > 0

    calculated = forward(ForwardProblem(dipole_model; solver, observable=request))
    @test calculated.solution isa SpinWaveDispersionResult
    @test calculated.observable_result.metadata[:momentum_scope] === :magnetic_supercell_bloch
    @test isapprox(calculated.observable_result.intensity,
        spin_tensor_structure_factor(dipole_model, bands, field, Q, energy).intensity; atol=1e-10)

    # The default typed route continues to return the existing finite-cell result.
    finite_request = SpinTensorStructureFactorRequest(Q, energy)
    finite = forward(ForwardProblem(dipole_model; solver, observable=finite_request))
    @test finite.solution isa SpinWaveResult
    @test finite.observable_result.metadata[:momentum_scope] === :finite_cell_form_factor

    optimized = optimize_spin_reference(dipole_model; formalism=:dipole, initial=MagneticReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), starts=2)
    @test optimized.converged
    @test optimized.residual < 1e-8
    @test isapprox(optimized.reference.directions[3, :], ones(2); atol=1e-5)
    @test !optimized.global_minimum_certified
    @test optimized.harmonic_stability === :not_checked
    @test isapprox(optimized.energy, solve(solver, dipole_model).classical_energy; atol=1e-8)

    spinone = _bloch_optimization_fixture(1.0; exchange=zeros(3, 3))
    initial = SUNReferenceState([ComplexF64[sqrt(0.5), sqrt(0.5), 0], ComplexF64[sqrt(0.5), sqrt(0.5), 0]])
    optimized_sun = optimize_spin_reference(spinone; formalism=:SUN, initial, starts=2)
    @test optimized_sun.converged
    @test optimized_sun.residual < 1e-8
    @test isapprox(optimized_sun.energy, -4.0; atol=1e-6)
    @test optimized_sun.reference isa SUNReferenceState
    @test isapprox(solve(SpinWaveSolver(reference=optimized_sun.reference; formalism=:SUN), spinone).classical_energy,
        optimized_sun.energy; atol=1e-7)

    sun_solver = SpinWaveSolver(reference=SUNReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]); formalism=:SUN)
    sun_bands = spinwave_dispersion(sun_solver, spinone, Q)
    sun_request = SpinTensorStructureFactorRequest(Q, energy; metadata=Dict(:spinwave_momentum_scope => :magnetic_supercell_bloch))
    sun_calculated = forward(ForwardProblem(spinone; solver=sun_solver, observable=sun_request))
    @test sun_calculated.solution isa SpinWaveDispersionResult
    @test size(sun_calculated.observable_result.intensity) == (2, length(energy), 3, 3)
    @test sun_calculated.observable_result.metadata[:formalism] === :SUN
    @test all(isfinite, sun_calculated.observable_result.intensity)
    @test isapprox(sun_calculated.solution.frequencies, sun_bands.frequencies; atol=1e-10)
    tangent_gauge = ComplexF64[1 0 0; 0 0 1; 0 1 0]
    sun_gauge = SUNReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        tangent_bases=[tangent_gauge, tangent_gauge])
    gauge_solver = SpinWaveSolver(reference=sun_gauge; formalism=:SUN)
    gauge_result = forward(ForwardProblem(spinone; solver=gauge_solver, observable=sun_request))
    @test isapprox(gauge_result.solution.frequencies, sun_bands.frequencies; atol=1e-10)
    @test isapprox(gauge_result.observable_result.intensity, sun_calculated.observable_result.intensity; atol=1e-8)

    # The spin-half SU(2) and SU(N) representations give the same periodic one-magnon tensor.
    half_sun = SpinWaveSolver(reference=z; formalism=:SUN)
    half_bands = spinwave_dispersion(half_sun, dipole_model, Q)
    half_tensor = spin_tensor_structure_factor(dipole_model, half_bands, field, Q, energy; channels=(:one_magnon,))
    @test isapprox(half_bands.frequencies, bands.frequencies; atol=1e-9)
    @test isapprox(half_tensor.intensity, tensor.intensity; atol=1e-8)

    @test_throws ArgumentError optimize_spin_reference(dipole_model; formalism=:other)
    @test_throws ArgumentError optimize_spin_reference(dipole_model; starts=0)
end
