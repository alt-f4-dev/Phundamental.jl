using Test
using LinearAlgebra

@testset "High-level material-to-measurement API" begin
    lattice = BravaisLattice(Matrix{Float64}(I, 3, 3))
    cu_properties = SiteProperties(spin=0.5, g_tensor=2.0)
    parent_crystal = LatticeCrystalStructure(
        lattice,
        BasisSite[BasisSite(:Cu, :Cu, [0.0, 0.0, 0.0]; properties=cu_properties)],
    )
    parent_cluster = Supercell(parent_crystal, (1, 1, 1); periodic=true)
    material = Material(
        "periodic Cu reference",
        parent_crystal;
        species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)),
        basis_properties=[cu_properties],
    )

    parent_bonds = neighbor_bonds(parent_cluster; cutoff=1.01, max_shell=1)
    automatic_parent_shell = bonds_in_shell(parent_cluster, 1)
    @test length(automatic_parent_shell) == length(parent_bonds)
    @test isapprox(sort([bond.distance for bond in automatic_parent_shell]), sort([bond.distance for bond in parent_bonds]); atol=1e-12)

    dimer_bond = first(parent_bonds)
    cluster = bond_cluster(parent_cluster, dimer_bond)

    @test nsites(cluster) == 2
    @test !any(cluster.periodic)
    @test isapprox(site_position(cluster, 2) - site_position(cluster, 1), dimer_bond.displacement; atol=1e-12)
    @test Phundamental.Models.site_spin(material, cluster, 1) == 0.5
    @test Phundamental.Models.site_spin(material, cluster, 2) == 0.5

    J = 1.7
    model = build_model(material, cluster, HeisenbergModel(J; shell=1))
    legacy_model = build_model(material, cluster, HeisenbergModel(J; neighbor_cutoff=1.01, max_shell=1))

    @test length(model.parameters[:bonds]) == 1
    @test model.parameters[:bonds][1].shell == 1

    solver = ExactDiagonalization()
    solution = solve(solver, model)
    legacy_solution = solve(solver, legacy_model)
    @test isapprox(solution.energies, legacy_solution.energies; rtol=1e-12, atol=1e-12)

    positions = model.parameters[:positions]
    dimer_vector = positions[2] - positions[1]
    Q = (π / dot(dimer_vector, dimer_vector)) .* dimer_vector
    axis = collect(range(J - 0.8, J + 0.8; length=1601))
    convention = SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=1.0)
    broadening = LorentzianBroadening(0.03)

    request = SpinTensorStructureFactorRequest(Q, axis; temperature=0.0, broadening=broadening, convention=convention)
    probe = MagneticNeutronProbe(material, cluster; site=1)
    resolution = GaussianResolution(0.06)

    problem = ForwardProblem(model; observable=request, probe=probe, resolution=resolution)
    calculation = forward(problem)

    field = SpinTensorField(
        xkeys=[:spin_x_1, :spin_x_2],
        ykeys=[:spin_y_1, :spin_y_2],
        zkeys=[:spin_z_1, :spin_z_2],
        positions=positions,
        normalization=:sqrtN,
    )
    manual_intrinsic = spin_tensor_structure_factor(
        model,
        solution,
        field,
        [Q],
        axis;
        temperature=0.0,
        convention=convention,
        broadening=broadening,
        weight_tol=0.0,
    )
    manual_probe = MagneticNeutronProbe(form_factor=Qvec -> 1.0, g_tensor=2.0 .* Matrix{Float64}(I, 3, 3), prefactor=1.0)
    manual_scattering = scattering_intensity(manual_probe, manual_intrinsic)
    manual_result = convolve_resolution(resolution, manual_scattering)

    @test Phundamental.Observables.validate_observable_equivalence(calculation.observable_result, manual_intrinsic; rtol=1e-10, atol=1e-12)
    @test Phundamental.Observables.validate_observable_equivalence(calculation.scattering_result, manual_scattering; rtol=1e-10, atol=1e-12)
    @test Phundamental.Observables.validate_observable_equivalence(calculation.result, manual_result; rtol=1e-10, atol=1e-12)

    transformed = forward(ForwardProblem(model; transformation=JordanWigner(1:2), observable=request, probe=probe, resolution=resolution))
    @test Phundamental.Observables.validate_observable_equivalence(calculation.result, transformed.result; rtol=1e-10, atol=1e-12)
    @test validate_forward_equivalence(calculation, transformed).valid

    @testset "F7 magnetic model-to-measurement and policy equivalence" begin
        policy = ForwardExecutionPolicy(include_default_exact_routes=false)
        route = ForwardRoute(ExactDiagonalization(); label=:f7_independent_ed)
        plan = plan_forward(model, request; policy=policy, routes=(route,), probe=probe, resolution=resolution)
        @test planning_report(plan).selected_label === :f7_independent_ed
        @test planning_report(plan).selected_certified
        @test isempty(planning_report(plan).selected_approximations)
        planned = forward(plan)
        agreement = validate_forward_equivalence(calculation, planned)
        @test agreement.valid
        @test all(Base.values(agreement.checks))
        @test planned.result.metadata[:forward_planned]

        predicted = portable_prediction(calculation; stage=:instrument, coordinate_frame=:cartesian,
                                        coordinate_unit="1/Å", spectral_unit="meV", intensity_unit="arb/meV",
                                        normalization="per_spin", provenance=Dict("origin" => "F7 magnetic reference"))
        @test predicted.stage === :instrument
        @test haskey(predicted.provenance, "forward")
        @test import_result(export_result(predicted)).intensity ≈ predicted.intensity
        values = Float64.(real.(predicted.intensity))
        mask = trues(size(values))
        mask[1, 2] = false
        observed = values .+ 0.2
        observed[1, 2] = NaN
        errors = fill(0.5, size(values))
        errors[1, 2] = NaN
        measured = experimental_dataset(predicted.grid, observed; valid=mask, uncertainty=errors,
                                        intensity_unit="arb/meV", normalization="per_spin")
        comparison = compare_measurement(predicted, measured)
        @test comparison.nvalid == count(mask)
        @test isapprox(comparison.chi_squared, comparison.nvalid * (0.2 / 0.5)^2; atol=1e-8, rtol=1e-8)
        @test compare_measurement(import_result(export_result(predicted)), import_result(export_result(measured))).chi_squared ≈ comparison.chi_squared
        @test_throws ArgumentError compare_measurement(predicted, experimental_dataset(predicted.grid, observed;
            valid=mask, uncertainty=errors, intensity_unit="incompatible", normalization="per_spin"))
        @test_throws ArgumentError validate_forward_equivalence(calculation, planned; atol=-1)
        @test_throws ArgumentError validate_forward_equivalence(calculation, planned; compare=(:instrument, :instrument))
        @test_throws ArgumentError validate_forward_equivalence(calculation, planned; compare=(:unknown,))
    end

end
