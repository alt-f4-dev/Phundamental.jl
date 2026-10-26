using Test
using LinearAlgebra
using Phundamental

function _sun_test_model(S; exchange=zeros(3, 3), field=(0.0, 0.0, 2.0), anisotropy=nothing, periodic=false)
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    properties = SiteProperties(spin=S, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0]; properties)])
    cluster = Supercell(crystal, (2,); periodic)
    material = Material("SU(N) spin-wave fixture", crystal;
        species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    model = build_model(material, cluster, TensorSpinModel(exchange; field, anisotropy, shell=1))
    return model
end

@testset "Site SU(N) generalized multi-flavor spin waves" begin
    z = MagneticReferenceState([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    @test SpinWaveSolver(reference=z).formalism === :dipole
    @test_throws ArgumentError SpinWaveSolver(reference=z; formalism=:unknown)

    # SU(2) = one tangent boson per spin-half site; compare all harmonic branches.
    ferro = _sun_test_model(0.5; exchange=-Matrix{Float64}(I, 3, 3), field=(0.0, 0.0, 1.0))
    dipole = solve(SpinWaveSolver(reference=z), ferro)
    sun = solve(SpinWaveSolver(reference=z; formalism=:SUN), ferro)
    @test solver_compatibility(SpinWaveSolver(reference=z; formalism=:SUN), ferro).compatible
    @test isapprox(sun.frequencies, dipole.frequencies; atol=1e-10)
    @test isapprox(sun.classical_energy, dipole.classical_energy; atol=1e-10)
    @test validate_spinwave_result(sun)[:passed]
    @test sun.metadata[:local_flavors] == [1, 1]
    @test sun.metadata[:formalism] === :SUN

    # S=1 has two independent local excitation flavors, with exact onsite energies.
    Kz = Matrix(Diagonal([0.0, 0.0, -0.3]))
    spinone = _sun_test_model(1.0; anisotropy=Kz)
    reference = SUNReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = solve(SpinWaveSolver(reference=reference; formalism=:SUN), spinone)
    @test isapprox(result.frequencies, [2.3, 2.3, 4.0, 4.0]; atol=1e-10)
    @test size(result.A) == (4, 4)
    @test norm(result.B) < 1e-12
    @test result.metadata[:local_flavors] == [2, 2]
    @test validate_spinwave_result(result)[:passed]
    @test isapprox(result.classical_energy, -4.6; atol=1e-10)
    @test isapprox(solve(SpinWaveSolver(reference=z), spinone).frequencies, [2.3, 2.3]; atol=1e-10)

    # Quadrupole Q=Sx²-Sy² reaches the local |m=-1> flavor, invisible to linear Sx.
    x, y, Sz = Phundamental.Solvers._sun_spin_matrices(1.0)
    quadrupole = x * x - y * y
    weights = sun_mode_matrix_elements(result, [quadrupole, zeros(ComplexF64, 3, 3)])
    @test isapprox(sum(abs2, weights), 1.0; atol=1e-10)
    @test all(isapprox(weights[ν], 0.0; atol=1e-10) for ν in eachindex(weights) if result.frequencies[ν] < 3.0)
    entries = sun_operator_matrix_elements(result, 1, quadrupole)
    @test isapprox(entries.expectation, 0.0; atol=1e-10)
    @test isapprox(norm(entries.annihilation), 1.0; atol=1e-10)

    # The large-D spin-1 |m=0> paramagnet has zero dipole and two finite-gap branches.
    planar = _sun_test_model(1.0; field=(0.0, 0.0, 0.0), anisotropy=Matrix(Diagonal([0.0, 0.0, 1.0])))
    singletlike = SUNReferenceState([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    paramagnon = solve(SpinWaveSolver(reference=singletlike; formalism=:SUN), planar)
    @test isapprox(paramagnon.frequencies, fill(1.0, 4); atol=1e-10)
    @test norm(paramagnon.metadata[:reference_spin_expectations]) < 1e-12
    @test validate_spinwave_result(paramagnon)[:passed]

    # A unitary rotation within the local tangent space changes coordinates, not the spectrum.
    gauge = ComplexF64[1 0 0; 0 0 1; 0 1 0]
    rephased = SUNReferenceState([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        tangent_bases=[gauge, gauge])
    rotated = solve(SpinWaveSolver(reference=rephased; formalism=:SUN), spinone)
    @test isapprox(rotated.frequencies, result.frequencies; atol=1e-10)

    # The existing typed forward and tensor scattering routes accept a site SU(N) solution.
    request = SpinTensorStructureFactorRequest([[0.0], [π / 2]], collect(range(-5.0, 5.0; length=251)))
    solver = SpinWaveSolver(reference=reference; formalism=:SUN)
    @test solver_compatibility(solver, spinone, request).compatible
    calculated = forward(ForwardProblem(spinone; solver, observable=request))
    @test calculated.solution isa SpinWaveResult
    @test size(calculated.observable_result.intensity) == (2, 251, 3, 3)
    @test calculated.observable_result.metadata[:formalism] === :SUN
    @test maximum(abs, calculated.observable_result.intensity) > 0
    rotated_tensor = forward(ForwardProblem(spinone;
        solver=SpinWaveSolver(reference=rephased; formalism=:SUN), observable=request)).observable_result
    @test isapprox(calculated.observable_result.intensity, rotated_tensor.intensity; atol=1e-8)
    @test_throws ArgumentError spin_tensor_structure_factor(spinone, result,
        Phundamental.Observables._spin_tensor_request_field(request, spinone), [[0.0]], request.axis;
        channels=(:two_magnon,))

    # Invalid coherent-state normalization, dimensions, and stationarity fail closed.
    @test_throws ArgumentError SUNReferenceState([[2.0, 0.0, 0.0]])
    @test_throws ArgumentError SpinWaveSolver(reference=reference; formalism=:dipole)
    nonstationary = SUNReferenceState([[sqrt(0.5), sqrt(0.5), 0.0], [1.0, 0.0, 0.0]])
    @test !solver_compatibility(SpinWaveSolver(reference=nonstationary; formalism=:SUN), spinone).compatible
    @test_throws ArgumentError solve(SpinWaveSolver(reference=nonstationary; formalism=:SUN), spinone)
    @test !solver_compatibility(SpinWaveSolver(reference=singletlike; formalism=:SUN), spinone).compatible
    @test_throws ArgumentError solve(SpinWaveSolver(reference=singletlike; formalism=:SUN), spinone)
    @test_throws DimensionMismatch solve(SpinWaveSolver(reference=reference; formalism=:SUN), ferro)

    # For S=5/2 the five local tangent flavors reproduce the Zeeman multiplet.
    fivehalf = _sun_test_model(5//2)
    highspin = SUNReferenceState([ComplexF64[1, 0, 0, 0, 0, 0], ComplexF64[1, 0, 0, 0, 0, 0]])
    highresult = solve(SpinWaveSolver(reference=highspin; formalism=:SUN), fivehalf)
    @test isapprox(highresult.frequencies, repeat([2.0, 4.0, 6.0, 8.0, 10.0]; inner=2); atol=1e-10)
    @test highresult.metadata[:local_flavors] == [5, 5]

    # Periodic bands retain magnetic-cell bond displacements and the enlarged flavor count.
    periodic = _sun_test_model(1.0; exchange=-Matrix{Float64}(I, 3, 3), periodic=true)
    bands = spinwave_dispersion(solver, periodic, [[0.0], [π / 3]])
    @test bands isa SpinWaveDispersionResult
    @test size(bands.frequencies) == (2, 4)
    @test minimum(bands.frequencies) > 0
    @test bands.metadata[:formalism] === :SUN
end
