using Test
using LinearAlgebra
using Phundamental

const B12 = Phundamental

function _batch12_material(; spins=(1//2, 1//2))
    lattice = B12.BravaisLattice(Matrix{Float64}(I, 3, 3))
    crystal = B12.LatticeCrystalStructure(lattice, [B12.BasisSite(:A1, :A, [0.0, 0.0, 0.0]),
                                                           B12.BasisSite(:B1, :B, [0.4, 0.0, 0.0])])
    return B12.Material("batch12 spin reference", crystal;
                        species=Dict(:A => B12.AtomicSpecies(:A; mass=1.0), :B => B12.AtomicSpecies(:B; mass=1.0)),
                        basis_properties=[B12.SiteProperties(spin=s) for s in spins])
end

function _batch12_dataset()
    return B12.CrystallographicDataset(spacegroup_number=1, international_symbol="P1",
        operations=[B12.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))], crystallographic_orbits=[1, 2])
end

function _batch12_spin_family(; shared=false)
    constraints = shared ? (B12.EqualParameterConstraint(:h1, :h2),) : ()
    result = B12.generate_hamiltonian_family(_batch12_material(), B12.SpinAlgebra(2);
        dataset=_batch12_dataset(), candidates=B12.AbstractOperatorExpr[B12.Sz(1), B12.Sz(2), B12.Sz(1) * B12.Sz(2)],
        max_body_order=2, basis_labels=[:h1, :h2, :J12], parameter_constraints=constraints)
    return result, B12.structural_family(result)
end

function _batch12_spin_observables()
    registry = Dict{Symbol,Any}()
    for site in 1:2
        registry[Symbol("spin_x_", site)] = B12.Sx(site)
        registry[Symbol("spin_y_", site)] = B12.Sy(site)
        registry[Symbol("spin_z_", site)] = B12.Sz(site)
    end
    return registry
end

@testset "Batch 12 structural forward closure and neutral reconstruction" begin
    @testset "Selection instantiation preserves physical and parameter boundaries" begin
        result, family = _batch12_spin_family()
        selection = B12.structural_selection(family, [:J12, :h1])
        @test family.metadata[:generation_result] === result
        representation = B12.Representation(:batch12_spin, B12.SpinHilbertSpace([1//2, 1//2]), B12.SpinAlgebra(2), B12.SpinProductBasis([1, 2]))
        model = B12.instantiate_model(selection, representation, Dict(:J12 => 0.3, :h1 => 0.7))
        @test B12.admissible(model)
        @test B12.parameter_names(model.parameterization) == [:h1, :J12]
        @test B12.embed_parameters(selection, model.parameterization.values) == [0.7, 0.0, 0.3]
        @test model.model_space === selection.hamiltonian_space
        @test last(model.provenance).selection_id == B12.structural_id(selection)
        @test last(model.provenance).generation_result === result
        @test B12.structural_id(family) == B12.structural_id(B12.structural_family(result))
        @test length(B12.structural_id(selection)) > 30
        @test B12.describe(result)["space"]["dimension"] == 3
        @test B12.describe(family)["id"] == B12.structural_id(family)
        @test B12.describe(family)["material"] == result.material.name
        @test B12.describe(family)["units"][1]["family_id"] == B12.structural_id(family)
        @test B12.describe(selection)["selected_space"]["dimension"] == 2
        @test B12.describe(model)["admissible"]
        @test B12.describe(model.parameterization.space)["names"] == ["h1", "J12"]
        @test B12.describe(selection)["selected_space"]["completeness_scope"] == "selected_structural_subspace"
        wrong_spin = B12.Representation(:wrong_spins, B12.SpinHilbertSpace([1, 1]), B12.SpinAlgebra(2), B12.SpinProductBasis([1, 2]))
        @test_throws ArgumentError B12.validate_structural_representation(selection, wrong_spin)
        wrong_rep = B12.Representation(:wrong, B12.FermionFockSpace(2), B12.FermionAlgebra(2), B12.FermionOccupationBasis([1, 2]))
        @test_throws ArgumentError B12.instantiate_model(selection, wrong_rep, [0.7, 0.3])
        @test_throws DimensionMismatch B12.instantiate_model(selection, representation, [0.7])
        @test_throws ArgumentError B12.instantiate_model(selection, representation, Dict(:h1 => 0.7, :h2 => 0.0))
    end

    @testset "Stable neutral selection records cannot authorize unknown structures" begin
        _, family = _batch12_spin_family()
        selection = B12.structural_selection(family, [:h1, :J12])
        record = B12.describe(selection)
        @test B12.structural_id(B12.reconstruct_selection(family, record)) == B12.structural_id(selection)
        @test B12.structural_id(B12.reconstruct_selection(family, Dict("schema_version" => "1.0", "family_id" => record["family_id"],
                  "selection_id" => record["id"]))) == record["id"]
        bad_id = merge(copy(record), Dict("id" => "selection-v1-tampered"))
        @test_throws ArgumentError B12.reconstruct_selection(family, bad_id)
        bad_family = merge(copy(record), Dict("family_id" => "family-v1-tampered"))
        @test_throws ArgumentError B12.reconstruct_selection(family, bad_family)
        bad_version = merge(copy(record), Dict("schema_version" => "99"))
        @test_throws ArgumentError B12.reconstruct_selection(family, bad_version)
        bad_unit = merge(copy(record), Dict("active_units" => ["missing_unit"]))
        @test_throws ArgumentError B12.reconstruct_selection(family, bad_unit)
        representation = B12.Representation(:spin, B12.SpinHilbertSpace([1//2, 1//2]), B12.SpinAlgebra(2), B12.SpinProductBasis([1, 2]))
        proposed = merge(copy(record), Dict("parameters" => Dict("h1" => 0.7, "J12" => 0.3)))
        @test B12.admissible(B12.reconstruct_model(family, representation, proposed))
        @test_throws ArgumentError B12.reconstruct_model(family, representation,
            merge(copy(proposed), Dict("parameters" => Dict("h1" => 0.7, "h2" => 0.3))))
    end

    @testset "Cross-unit parameter equality remains binding after selection" begin
        _, family = _batch12_spin_family(shared=true)
        selection = B12.structural_selection(family, [:h1, :J12])
        representation = B12.Representation(:spin, B12.SpinHilbertSpace([1//2, 1//2]), B12.SpinAlgebra(2), B12.SpinProductBasis([1, 2]))
        @test_throws ArgumentError B12.instantiate_model(selection, representation, [0.8, 0.3])
        model = B12.instantiate_model(selection, representation, [0.0, 0.3])
        @test B12.parameter_vector(model) == [0.0, 0.3]
        @test B12.embed_parameters(selection, B12.parameter_vector(model)) == [0.0, 0.0, 0.3]
    end

    @testset "Material generation to spin scattering agrees with a manual model" begin
        _, family = _batch12_spin_family()
        selection = B12.structural_selection(family, [:J12, :h1])
        representation = B12.Representation(:spin, B12.SpinHilbertSpace([1//2, 1//2]), B12.SpinAlgebra(2), B12.SpinProductBasis([1, 2]))
        values = Dict(:h1 => 0.7, :J12 => 0.3)
        positions = [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]
        observables = _batch12_spin_observables()
        axis = collect(range(-2.0, 2.0; length=201))
        request = B12.SpinTensorStructureFactorRequest([1.0, 0.0, 0.0], axis; temperature=0.0,
            broadening=B12.LorentzianBroadening(0.06))
        probe = B12.MagneticNeutronProbe()
        resolution = B12.GaussianResolution(0.05)
        prepared = B12.prepare_forward(selection, representation; observable=request, probe=probe, resolution=resolution,
                                       observables=observables, metadata_parameters=Dict(:positions => positions))
        @test B12.describe(prepared)["prepared_cache"] === false
        output = B12.forward_parameters(prepared, values)
        manual = B12.ManyBodyModel(representation, 0.7 * B12.Sz(1) + 0.3 * B12.Sz(1) * B12.Sz(2);
                                   observables=observables, parameters=Dict(:positions => positions))
        reference = B12.forward(manual; observable=request, probe=probe, resolution=resolution)
        @test isapprox(output.solution.energies, reference.solution.energies; atol=1e-12)
        @test B12.Observables.validate_observable_equivalence(output.observable_result, reference.observable_result; atol=1e-10)
        @test B12.Observables.validate_observable_equivalence(output.scattering_result, reference.scattering_result; atol=1e-10)
        @test B12.Observables.validate_observable_equivalence(output.result, reference.result; atol=1e-10)
        @test B12.describe(output)["selection_id"] == B12.structural_id(selection)
        @test B12.describe(output)["probe_response"] !== nothing
        @test length(B12.forward_batch(prepared, [values, Dict(:h1 => 0.3, :J12 => 0.5)])) == 2
        @test_throws ArgumentError B12.forward_batch(prepared, [])
        @test_throws ArgumentError B12.forward_parameters(prepared, Dict(:h1 => 0.7, :h2 => 0.3))
        @test isapprox(B12.forward(selection, representation, [0.7, 0.3]).solution.energies, output.solution.energies; atol=1e-12)
    end

    @testset "Geometric range uses the closest Cartesian periodic image for skew cells" begin
        A = [1.0 0.9 0.0; 0.0 0.1 0.0; 0.0 0.0 1.0]
        delta = [0.49, 0.49, 0.0]
        expected = minimum(norm(A * (delta .- [i, j, 0])) for i in -2:2 for j in -2:2)
        computed = B12.Models._minimum_periodic_cartesian_distance(A, delta)
        @test isapprox(computed, expected; atol=1e-12)
        @test computed < norm(A * delta)
        lattice = B12.BravaisLattice(A)
        crystal = B12.LatticeCrystalStructure(lattice, [B12.BasisSite(:A1, :A, [0.0, 0.0, 0.0]),
                                                               B12.BasisSite(:B1, :B, delta)])
        material = B12.Material("skew periodic reference", crystal;
            species=Dict(:A => B12.AtomicSpecies(:A; mass=1.0), :B => B12.AtomicSpecies(:B; mass=1.0)),
            basis_properties=[B12.SiteProperties(spin=1//2), B12.SiteProperties(spin=1//2)])
        spec = B12.material_generative_specification(material, B12.SpinAlgebra(2); dataset=_batch12_dataset(),
            max_body_order=2, max_range=0.1)
        @test B12.admissibility_report(spec, B12.Sz(1) * B12.Sz(2)).admissible
        restricted = B12.material_generative_specification(material, B12.SpinAlgebra(2); dataset=_batch12_dataset(),
            max_body_order=2, max_range=0.05)
        @test !B12.admissibility_report(restricted, B12.Sz(1) * B12.Sz(2)).admissible
    end

    @testset "Finite bosonic forward prediction preserves truncation boundary" begin
        lattice = B12.BravaisLattice(Matrix{Float64}(I, 3, 3))
        crystal = B12.LatticeCrystalStructure(lattice, [B12.BasisSite(:X1, :X, [0.0, 0.0, 0.0])])
        material = B12.Material("batch12 boson reference", crystal;
            species=Dict(:X => B12.AtomicSpecies(:X; mass=1.0)))
        dataset = B12.CrystallographicDataset(spacegroup_number=1, international_symbol="P1",
            operations=[B12.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))], crystallographic_orbits=[1])
        algebra = B12.BosonAlgebra(1)
        result = B12.generate_hamiltonian_family(material, algebra; dataset=dataset, max_body_order=1,
            candidates=B12.AbstractOperatorExpr[B12.nb(1)], basis_labels=[:omega])
        selection = B12.structural_selection(B12.structural_family(result), [:omega])
        for cutoff in (4, 6)
            representation = B12.Representation(:boson, B12.BosonFockSpace(1), algebra, B12.BosonOccupationBasis([1]);
                truncation=B12.NumericalTruncation(cutoff))
            result_forward = B12.forward(selection, representation, [1.3])
            manual = B12.ManyBodyModel(representation, 1.3 * B12.nb(1))
            reference = B12.forward(manual)
            @test isapprox(result_forward.solution.energies, reference.solution.energies; atol=1e-12)
            @test isapprox(result_forward.solution.energies[1:3], [0.0, 1.3, 2.6]; atol=1e-12)
            @test result_forward.source_model.representation.truncation === representation.truncation
        end
    end

    @testset "Fermionic generated submodel matches an independent reference" begin
        material = _batch12_material()
        algebra = B12.FermionAlgebra(2)
        result = B12.generate_hamiltonian_family(material, algebra; dataset=_batch12_dataset(), max_body_order=1,
            candidates=B12.AbstractOperatorExpr[B12.nf(1), B12.nf(2)], basis_labels=[:eps1, :eps2])
        family = B12.structural_family(result)
        selection = B12.structural_selection(family, [:eps1])
        representation = B12.Representation(:fermions, B12.FermionFockSpace(2), algebra, B12.FermionOccupationBasis([1, 2]))
        prediction = B12.forward(selection, representation, [0.8])
        reference = B12.forward(B12.ManyBodyModel(representation, 0.8 * B12.nf(1)))
        @test isapprox(prediction.solution.energies, reference.solution.energies; atol=1e-12)
        @test isnothing(prediction.result)
    end
end
