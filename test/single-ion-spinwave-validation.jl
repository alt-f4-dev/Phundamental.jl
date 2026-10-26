using Test
using LinearAlgebra
using Phundamental

function _single_ion_fixture(spin; anisotropy=nothing, field=(0.0, 0.0, 2.0), periodic=false, hbar=1.0)
    lattice = BravaisLattice(reshape([1.0], 1, 1))
    properties = SiteProperties(spin=spin, g_tensor=2.0)
    crystal = LatticeCrystalStructure(lattice, BasisSite[BasisSite(:Cu, :Cu, [0.0]; properties)])
    cluster = Supercell(crystal, (2,); periodic)
    material = Material("single-ion spin-wave fixture", crystal;
        species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)), basis_properties=[properties])
    model = build_model(material, cluster, TensorSpinModel(zeros(3, 3); field, anisotropy, shell=1, hbar))
    return model, material, cluster
end

@testset "Single-ion anisotropy in noncollinear spin-wave API" begin
    reference = MagneticReferenceState([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    solver = SpinWaveSolver(reference=reference)

    # D(Sz)^2 shifts a spin-one excitation by -(2S-1)D, with the original field gap retained.
    Kz = Diagonal([0.0, 0.0, -0.4]) |> Matrix
    spinone, _, _ = _single_ion_fixture(1.0; anisotropy=Kz)
    @test solver_compatibility(solver, spinone).compatible
    axial = solve(solver, spinone)
    @test isapprox(axial.frequencies, [2.4, 2.4]; atol=1e-10)
    @test isapprox(real.(diag(axial.A)), [2.4, 2.4]; atol=1e-10)
    @test norm(axial.B) < 1e-12
    @test axial.metadata[:onsite_anisotropy_present]
    @test axial.metadata[:interaction_scope] == :bilinear_exchange_and_single_ion_quadratic
    @test validate_spinwave_result(axial)[:passed]

    # Spin one-half: every real symmetric quadratic single-ion tensor is a scalar operator.
    K = [0.5 0.2 0.3; 0.2 -0.4 0.0; 0.3 0.0 -0.2]
    half, _, _ = _single_ion_fixture(0.5; anisotropy=K)
    halfresult = solve(solver, half)
    @test isapprox(halfresult.frequencies, [2.0, 2.0]; atol=1e-10)
    @test norm(halfresult.B) < 1e-12
    @test isapprox(halfresult.classical_energy, -2.0 + tr(K) / 2; atol=1e-10)
    @test maximum(halfresult.metadata[:stationarity_residuals]) < 1e-12

    # Exact finite-spin vacuum, one-boson, and two-boson matrix elements are retained.
    Kxy = [0.5 0.2 0.0; 0.2 -0.4 0.0; 0.0 0.0 -0.2]
    nonaxial, _, _ = _single_ion_fixture(1.0; anisotropy=Kxy, field=(0.0, 0.0, 3.0))
    result = solve(solver, nonaxial)
    Aexpected = 3.0 + ((Kxy[1, 1] + Kxy[2, 2]) / 2 - Kxy[3, 3])
    Bexpected = sqrt(0.5) * complex(Kxy[1, 1] - Kxy[2, 2], 2 * Kxy[1, 2])
    @test isapprox(result.A[1, 1], Aexpected; atol=1e-10)
    @test isapprox(result.B[1, 1], Bexpected; atol=1e-10)
    @test isapprox(result.frequencies, fill(sqrt(Aexpected^2 - abs2(Bexpected)), 2); atol=1e-9)
    @test validate_spinwave_result(result)[:passed]

    # A local transverse-frame gauge rotation must not change physical frequencies.
    gauge = MagneticReferenceState([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        transverse_axes=[0.0 0.0; 1.0 1.0; 0.0 0.0])
    rotated = solve(SpinWaveSolver(reference=gauge), nonaxial)
    @test isapprox(rotated.frequencies, result.frequencies; atol=1e-10)

    # An isotropic onsite tensor contributes the exact S(S+1) constant only.
    isotropic, _, _ = _single_ion_fixture(1.0; anisotropy=0.3 * Matrix{Float64}(I, 3, 3))
    iso_result = solve(solver, isotropic)
    @test isapprox(iso_result.frequencies, [2.0, 2.0]; atol=1e-10)
    @test isapprox(iso_result.classical_energy, -4.0 + 2 * 1.0 * 2.0 * 0.3; atol=1e-10)

    # Site-resolved anisotropy and both nondefault units and periodic dispersion.
    siteK = [Kz, zeros(3, 3)]
    site_model, _, _ = _single_ion_fixture(1.0; anisotropy=siteK, hbar=2.0)
    callable_model, _, _ = _single_ion_fixture(1.0; anisotropy=(i, site) -> i == 1 ? Kz : zeros(3, 3))
    @test isapprox(solve(solver, callable_model).frequencies, [2.0, 2.4]; atol=1e-10)
    @test isapprox(solve(solver, site_model).frequencies, [2.0, 2.4]; atol=1e-10)
    periodic_model, _, _ = _single_ion_fixture(1.0; anisotropy=siteK, periodic=true)
    dispersion = spinwave_dispersion(solver, periodic_model, [[0.0], [π / 3]])
    @test isapprox(dispersion.frequencies[1, :], [2.0, 2.4]; atol=1e-10)
    @test isapprox(dispersion.frequencies[2, :], [2.0, 2.4]; atol=1e-10)
    @test dispersion.metadata[:onsite_anisotropy_present]

    # Noncollinear DM exchange and axis-dependent single-ion terms coexist in one solver.
    n1, n2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    exchange = Matrix{Float64}(I, 3, 3)
    exchange[1, 2] = 0.2
    exchange[2, 1] = -0.2
    fields = [exchange * n2 + 2n1, transpose(exchange) * n1 + 2n2]
    K1, K2 = Diagonal([-0.25, 0.0, 0.0]) |> Matrix, Diagonal([0.0, -0.3, 0.0]) |> Matrix
    _, material1, cluster1 = _single_ion_fixture(1.0)
    coupled = build_model(material1, cluster1, TensorSpinModel(exchange; field=fields, anisotropy=[K1, K2], shell=1))
    canted_reference = MagneticReferenceState([n1, n2])
    canted_solver = SpinWaveSolver(reference=canted_reference)
    @test solver_compatibility(canted_solver, coupled).compatible
    canted_result = solve(canted_solver, coupled)
    @test all(>(0), canted_result.frequencies)
    @test validate_spinwave_result(canted_result)[:passed]
    @test maximum(canted_result.metadata[:stationarity_residuals]) < 1e-10

    # A nonstationary anisotropy tensor must not be silently projected onto the reference.
    Kxz = [0.0 0.0 0.2; 0.0 0.0 0.0; 0.2 0.0 0.0]
    unstable, _, _ = _single_ion_fixture(1.0; anisotropy=Kxz)
    @test !solver_compatibility(solver, unstable).compatible
    @test_throws ArgumentError solve(solver, unstable)

    # Same-site ordered ladder products must resolve their SU(2) commutator as a field.
    bare, _, _ = _single_ion_fixture(0.5)
    ordered = ManyBodyModel(bare.representation, bare.hamiltonian + 0.2 * (Sp(1) * Sm(1));
        parameters=bare.parameters, observables=bare.observables)
    ordered_result = solve(solver, ordered)
    @test isapprox(ordered_result.frequencies, [1.8, 2.0]; atol=1e-10)

    # Non-Hermitian ordered single-site terms and nonsymmetric anisotropy inputs fail closed.
    bad = ManyBodyModel(bare.representation, bare.hamiltonian + Sx(1) * Sy(1);
        parameters=bare.parameters, observables=bare.observables)
    @test !solver_compatibility(solver, bad).compatible
    @test_throws ArgumentError solve(solver, bad)
    nonsymmetric = [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    _, material, cluster = _single_ion_fixture(1.0)
    @test_throws ArgumentError build_model(material, cluster, TensorSpinModel(zeros(3, 3); anisotropy=nonsymmetric, shell=1))
    @test_throws DimensionMismatch build_model(material, cluster, TensorSpinModel(zeros(3, 3); anisotropy=[Kz], shell=1))

    # Existing zero-anisotropy models retain their prior result classification.
    unmodified, _, _ = _single_ion_fixture(1.0)
    original = solve(solver, unmodified)
    @test original.metadata[:interaction_scope] == :bilinear_distinct_spin_sites
    @test !original.metadata[:onsite_anisotropy_present]
    @test isapprox(original.frequencies, [2.0, 2.0]; atol=1e-10)
end
