using Test
using LinearAlgebra
using Phundamental

const MSA = Phundamental

function _milestone_a_family(; equality=false)
    lattice = MSA.BravaisLattice([4.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 6.0])
    sites = [MSA.BasisSite(:A1, :A, [0.0, 0.0, 0.0]), MSA.BasisSite(:A2, :A, [0.5, 0.0, 0.0])]
    material = MSA.Material("half-translation spin chain reference", MSA.LatticeCrystalStructure(lattice, sites);
        species=Dict(:A => MSA.AtomicSpecies(:A; mass=1.0)),
        basis_properties=[MSA.SiteProperties(spin=1//2), MSA.SiteProperties(spin=1//2)])
    identity = MSA.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    translation = MSA.SpaceGroupOperation(Matrix{Int}(I, 3, 3), [0.5, 0.0, 0.0])
    dataset = MSA.CrystallographicDataset(spacegroup_number=1, international_symbol="P1 + half translation",
        operations=[identity, translation], crystallographic_orbits=[1, 1])
    constraints = equality ? (MSA.EqualParameterConstraint(:field, :exchange),) : ()
    result = MSA.generate_hamiltonian_family(material, MSA.SpinAlgebra(2); dataset=dataset,
        candidates=MSA.AbstractOperatorExpr[MSA.Sz(1) + MSA.Sz(2), MSA.Sz(1) * MSA.Sz(2)],
        basis_labels=[:field, :exchange], max_body_order=2, parameter_constraints=constraints)
    family = MSA.structural_family(result)
    representation = MSA.Representation(:milestone_a_spin, MSA.SpinHilbertSpace([1//2, 1//2]),
        MSA.SpinAlgebra(2), MSA.SpinProductBasis([1, 2]))
    return (; material, result, family, representation)
end

@testset "Milestone A model context, external structural coordinates and source integration" begin
    @testset "Crystallographic metadata propagates without caller reconstruction" begin
        fixture = _milestone_a_family()
        selection = MSA.structural_selection(fixture.family, [:field, :exchange])
        model = MSA.instantiate_model(selection, fixture.representation, Dict(:field => 0.6, :exchange => 0.8))
        context = MSA.model_context(model)
        @test context isa MSA.PhysicalModelContext
        @test context.material === fixture.material
        @test context.dataset === fixture.result.dataset
        @test MSA.observable_context(model) isa MSA.ObservableContext
        @test MSA.observable_context(model).convention === :spin_cartesian_site
        @test length(MSA.observable_context(model).registry) == 6
        @test MSA.geometry_context(model) isa MSA.GeometryContext
        @test MSA.geometry_context(model).labels == [:A1, :A2]
        @test MSA.geometry_context(model).positions == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        @test length(model.observables) == 6
        @test model.observables[:spin_x_1] == MSA.Sx(1)
        @test model.parameters[:positions] == MSA.geometry_context(model).positions
        @test model.parameters[:material] === fixture.material
        @test MSA.admissible(model)
        @test_throws ArgumentError MSA.instantiate_model(selection, fixture.representation, [0.6, 0.8];
            metadata_parameters=Dict(:positions => [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]]))
        @test_throws ArgumentError MSA.instantiate_model(selection, fixture.representation, [0.6, 0.8];
            observables=Dict(:spin_x_1 => MSA.Sx(2)))
        source = MSA.instantiate_model(fixture.result, fixture.representation, [0.6, 0.8])
        @test MSA.geometry_context(source).positions == MSA.geometry_context(model).positions
        @test length(source.observables) == 6
        legacy = MSA.ManyBodyModel(fixture.representation, MSA.Sz(1))
        @test isnothing(MSA.model_context(legacy))
        @test isnothing(MSA.geometry_context(legacy))
    end

    @testset "Other site-indexed algebras and uncertified custom mappings" begin
        fixture = _milestone_a_family()
        for algebra in (MSA.FermionAlgebra(2), MSA.BosonAlgebra(2))
            number = algebra isa MSA.FermionAlgebra ? MSA.nf : MSA.nb
            source = MSA.generate_hamiltonian_family(fixture.material, algebra;
                dataset=fixture.result.dataset, candidates=MSA.AbstractOperatorExpr[number(1) + number(2)],
                basis_labels=[:density])
            context = MSA.generation_context(source, algebra)
            @test context.geometry isa MSA.GeometryContext
            @test length(context.geometry.positions) == 2
            registry = MSA.default_observable_context(source,
                algebra isa MSA.FermionAlgebra ? MSA.Representation(:fermion_sites, MSA.FermionFockSpace(2), algebra,
                    MSA.FermionOccupationBasis([1, 2])) : MSA.Representation(:boson_sites, MSA.BosonFockSpace(2), algebra,
                    MSA.BosonOccupationBasis([1, 2])))
            @test isempty(registry.observables)
            @test registry.metadata_parameters[:observable_context].convention === :explicit_or_unregistered
        end
        custom_action = (expression, operation, crystal, algebra) ->
            MSA.Crystallography.standard_symmetry_action(expression, operation, crystal, algebra)
        custom = MSA.generate_hamiltonian_family(fixture.material, MSA.SpinAlgebra(2);
            dataset=fixture.result.dataset, symmetry_action=custom_action,
            candidates=MSA.AbstractOperatorExpr[MSA.Sz(1) + MSA.Sz(2)], basis_labels=[:field])
        context = MSA.generation_context(custom, MSA.SpinAlgebra(2))
        @test isnothing(context.geometry)
        @test isempty(MSA.default_observable_context(custom, fixture.representation).observables)
    end

    @testset "Material-to-neutron-forward agrees with independently registered reference" begin
        fixture = _milestone_a_family()
        selection = MSA.structural_selection(fixture.family, [:field, :exchange])
        grid = collect(range(-2.0, 2.0; length=121))
        request = MSA.SpinTensorStructureFactorRequest([0.3, 0.0, 0.0], grid;
            temperature=0.0, broadening=MSA.LorentzianBroadening(0.08))
        probe = MSA.MagneticNeutronProbe()
        resolution = MSA.GaussianResolution(0.05)
        prepared = MSA.prepare_forward(selection, fixture.representation; observable=request,
            probe=probe, resolution=resolution)
        result = MSA.forward_parameters(prepared, Dict(:field => 0.6, :exchange => 0.8))
        reference_operators = Dict{Symbol,Any}()
        for i in 1:2
            reference_operators[Symbol("spin_x_", i)] = MSA.Sx(i)
            reference_operators[Symbol("spin_y_", i)] = MSA.Sy(i)
            reference_operators[Symbol("spin_z_", i)] = MSA.Sz(i)
        end
        reference_h = 0.6 * (MSA.Sz(1) + MSA.Sz(2)) + 0.8 * MSA.Sz(1) * MSA.Sz(2)
        reference = MSA.ManyBodyModel(fixture.representation, reference_h; observables=reference_operators,
            parameters=Dict(:positions => [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        manual = MSA.forward(reference; observable=request, probe=probe, resolution=resolution)
        @test MSA.validate_forward_equivalence(result, manual).valid
        @test isfinite(sum(result.result.intensity))
    end

    @testset "External structural localization recovers independent coordinates" begin
        fixture = _milestone_a_family()
        operator = 0.6 * (MSA.Sz(1) + MSA.Sz(2)) + 0.8 * MSA.Sz(1) * MSA.Sz(2)
        manual = MSA.ManyBodyModel(fixture.representation, operator)
        located = MSA.locate_in_family(fixture.family, manual)
        @test located.in_span && located.valid_parameters
        @test located.selection isa MSA.StructuralSelection
        @test located.active_units == [:field, :exchange]
        @test located.parent_coefficients ≈ [0.6, 0.8] atol=1e-10
        @test located.selected_coefficients ≈ [0.6, 0.8] atol=1e-10
        @test located.residual < 1e-10
        without_schema = MSA.structural_family(fixture.result.hamiltonian_space)
        @test MSA.locate_in_family(without_schema, operator).selection isa MSA.StructuralSelection
        @test MSA.describe(located)["selection_id"] == MSA.structural_id(located.selection)
        shifted = operator + 1.75 * MSA.IdentityOperator()
        strict = MSA.locate_in_family(fixture.family, shifted)
        @test !strict.in_span && isnothing(strict.selection)
        projected = MSA.locate_in_family(fixture.family, shifted; project_identity=true)
        @test projected.in_span && projected.valid_parameters
        @test projected.active_units == located.active_units
        @test projected.parent_coefficients ≈ located.parent_coefficients
        @test projected.identity_offset ≈ 1.75
        @test projected.identity_projected
        @test MSA.describe(projected)["identity_offset"] ≈ 1.75
        @test MSA.describe(projected)["identity_projected"] === true
        @test_throws ArgumentError MSA.locate_in_family(fixture.family, operator + im * MSA.IdentityOperator(); project_identity=true)
        weak = MSA.locate_in_family(fixture.family,
            0.6 * (MSA.Sz(1) + MSA.Sz(2)) + 1e-7 * MSA.Sz(1) * MSA.Sz(2);
            relative_zero_tolerance=1e-5)
        @test weak.active_units == [:field]
        @test weak.parent_coefficients[2] ≈ 1e-7 atol=1e-10
        selected = MSA.structural_selection(fixture.family, [:field])
        roundtrip = MSA.instantiate_model(selected, fixture.representation, [0.6])
        recovered = MSA.locate_in_family(fixture.family, roundtrip)
        @test recovered.selection.unit_ids == (:field,)
        @test recovered.selected_coefficients ≈ [0.6]
        outside = MSA.locate_in_family(fixture.family, MSA.Sx(1))
        @test !outside.in_span
        @test isnothing(outside.selection)
        @test outside.residual > 0
        zero = MSA.locate_in_family(fixture.family, 0.0 * MSA.Sz(1))
        @test zero.in_span && isnothing(zero.selection)
        @test isempty(zero.active_units)
        wrong = MSA.Representation(:wrong, MSA.FermionFockSpace(2), MSA.FermionAlgebra(2), MSA.FermionOccupationBasis([1, 2]))
        @test_throws ArgumentError MSA.locate_in_family(fixture.family, MSA.ManyBodyModel(wrong, MSA.nf(1)))
    end

    @testset "Parameter equality must not be ignored during localization" begin
        fixture = _milestone_a_family(equality=true)
        admissible = MSA.locate_in_family(fixture.family, 0.7 * (MSA.Sz(1) + MSA.Sz(2)) + 0.7 * MSA.Sz(1) * MSA.Sz(2))
        @test admissible.in_span && admissible.valid_parameters
        incompatible = MSA.locate_in_family(fixture.family, 0.7 * (MSA.Sz(1) + MSA.Sz(2)) + 0.2 * MSA.Sz(1) * MSA.Sz(2))
        @test incompatible.in_span && !incompatible.valid_parameters
        @test isnothing(incompatible.selection)
        @test !isempty(incompatible.reasons)
    end
end
