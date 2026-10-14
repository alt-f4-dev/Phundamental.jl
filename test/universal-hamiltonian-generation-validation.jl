using Test
using LinearAlgebra
using Phundamental

const B10 = Phundamental

function _batch10_identity_dataset(nsites::Integer)
    operation = B10.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    return B10.CrystallographicDataset(
        spacegroup_number=1,
        international_symbol="P1",
        operations=[operation],
        crystallographic_orbits=collect(1:Int(nsites)),
    )
end

function _batch10_two_site_material()
    lattice = B10.BravaisLattice(Matrix{Float64}(I, 3, 3))
    crystal = B10.LatticeCrystalStructure(
        lattice,
        B10.BasisSite[
            B10.BasisSite(:A1, :A, [0.0, 0.0, 0.0]),
            B10.BasisSite(:B1, :B, [0.4, 0.0, 0.0]),
        ],
    )
    species = Dict(
        :A => B10.AtomicSpecies(:A; mass=1.0),
        :B => B10.AtomicSpecies(:B; mass=2.0),
    )
    properties = [B10.SiteProperties(spin=1//2), B10.SiteProperties(spin=1//2)]
    return B10.Material("batch10 two-site material", crystal; species=species, basis_properties=properties)
end

@testset "Batch 10 material-aware universal Hamiltonian generation" begin
    @testset "Material geometry becomes an explicit crystallographic specification" begin
        material = _batch10_two_site_material()
        dataset = _batch10_identity_dataset(2)
        algebra = B10.SpinAlgebra(2)
        crystal = B10.material_crystallographic_structure(material)
        specification = B10.material_generative_specification(material, algebra; dataset=dataset, max_body_order=1)
        discovered = B10.material_generative_specification(material, algebra; max_body_order=1)

        @test crystal isa B10.CrystalStructure
        @test crystal.labels == [:A1, :B1]
        @test crystal.species == [1, 2]
        @test specification.space_group === dataset
        @test discovered.space_group isa B10.CrystallographicDataset
        @test get(discovered.space_group.metadata, :backend, nothing) == :spglib
        @test specification.algebra === algebra
        @test specification.max_body_order == 1
        @test specification.metadata[:material_name] == material.name
        @test specification.metadata[:generation_domain] == :primitive_cell
        @test specification.metadata[:operator_site_mapping] == :one_index_per_crystallographic_site
        @test length(specification.local_spaces) == 2
        @test all(space -> space isa B10.SpinHilbertSpace && space.spins == [0.5], specification.local_spaces)
    end

    @testset "Universal result closes specification, model space, structure, and parameter schema" begin
        material = _batch10_two_site_material()
        dataset = _batch10_identity_dataset(2)
        algebra = B10.SpinAlgebra(2)
        candidates = B10.AbstractOperatorExpr[B10.Sz(1), B10.Sz(2)]
        shared = B10.EqualParameterConstraint(:h1, :h2)
        result = B10.generate_hamiltonian_family(
            material,
            algebra;
            dataset=dataset,
            candidates=candidates,
            max_body_order=1,
            basis_labels=[:h1, :h2],
            parameter_values=[0.4, 0.4],
            parameter_constraints=(shared,),
        )

        @test result isa B10.HamiltonianGenerationResult
        @test result.material === material
        @test isnothing(result.cluster)
        @test result.dataset === dataset
        @test B10.model_space_dimension(result.hamiltonian_space) == 2
        @test length(result.model_structure.terms) == 2
        @test B10.parameter_names(result.parameter_space) == [:h1, :h2]
        @test B10.parameter_dof(result.parameter_space) == 1
        @test B10.parameter_constraint_report(result.parameter_space, [0.4, 0.4]).valid
        @test result.metadata[:source] == :material_primitive_cell
        @test result.metadata[:model_space_dimension] == 2

        representation = B10.Representation(:batch10_spin, B10.SpinHilbertSpace([1//2, 1//2]), algebra, B10.SpinProductBasis([1, 2]))
        model = B10.instantiate_model(result, representation)
        @test B10.parameter_vector(model) == [0.4, 0.4]
        @test B10.admissible(model)
        @test last(model.provenance).generator == :batch10_material_crystallographic
        @test last(model.provenance).spacegroup_number == 1
    end

    @testset "Material geometry supplies physical range when standard site indexing is valid" begin
        material = _batch10_two_site_material()
        dataset = _batch10_identity_dataset(2)
        candidates = B10.AbstractOperatorExpr[B10.Sz(1), B10.Sz(1) * B10.Sz(2)]
        result = B10.generate_hamiltonian_family(
            material,
            B10.SpinAlgebra(2);
            dataset=dataset,
            candidates=candidates,
            max_body_order=2,
            max_range=0.25,
            parameter_values=[0.5],
        )

        @test result.specification.metadata[:range_evaluator_source] == :crystallographic_geometry
        @test B10.model_space_dimension(result.hamiltonian_space) == 1
        @test result.hamiltonian_space.basis.metadata[:rejected_admissibility] == 1
        @test B10.admissibility_report(result.specification, B10.Sz(1)).admissible
        @test !B10.admissibility_report(result.specification, B10.Sz(1) * B10.Sz(2)).admissible
    end

    @testset "Periodic supercells define translated finite structural domains" begin
        lattice = B10.BravaisLattice(Matrix{Float64}(I, 3, 3))
        crystal = B10.LatticeCrystalStructure(lattice, [B10.BasisSite(:X, :X, [0.0, 0.0, 0.0])])
        material = B10.Material(
            "batch10 periodic material",
            crystal;
            species=Dict(:X => B10.AtomicSpecies(:X; mass=1.0)),
            basis_properties=[B10.SiteProperties(spin=1//2)],
        )
        cluster = B10.Supercell(crystal, [2, 1, 1]; periodic=true)
        dataset = _batch10_identity_dataset(2)
        converted = B10.material_crystallographic_structure(material, cluster)
        result = B10.generate_hamiltonian_family(
            material,
            cluster,
            B10.SpinAlgebra(2);
            dataset=dataset,
            candidates=B10.AbstractOperatorExpr[B10.Sz(1) * B10.Sz(2)],
            max_body_order=2,
            max_range=1.01,
            parameter_values=[0.7],
        )

        @test size(converted.lattice) == (3, 3)
        @test converted.labels == [:X_1, :X_2]
        @test converted.positions[:, 1] == [0.0, 0.0, 0.0]
        @test converted.positions[:, 2] == [0.5, 0.0, 0.0]
        @test result.cluster === cluster
        @test result.metadata[:source] == :material_periodic_supercell
        @test B10.model_space_dimension(result.hamiltonian_space) == 1

        open_cluster = B10.Supercell(crystal, [2, 1, 1]; periodic=false)
        @test_throws ArgumentError B10.material_crystallographic_structure(material, open_cluster)
    end

    @testset "Standard symmetry action preserves local Hilbert-space classes" begin
        lattice = B10.BravaisLattice(Matrix{Float64}(I, 3, 3))
        crystal = B10.LatticeCrystalStructure(
            lattice,
            B10.BasisSite[
                B10.BasisSite(:X1, :X, [0.0, 0.0, 0.0]),
                B10.BasisSite(:X2, :X, [0.5, 0.0, 0.0]),
            ],
        )
        material = B10.Material(
            "batch10 inequivalent local spaces",
            crystal;
            species=Dict(:X => B10.AtomicSpecies(:X; mass=1.0)),
            basis_properties=[B10.SiteProperties(spin=1//2), B10.SiteProperties(spin=1)],
        )
        swap = B10.SpaceGroupOperation(Matrix{Int}(I, 3, 3), [0.5, 0.0, 0.0])
        dataset = B10.CrystallographicDataset(spacegroup_number=1, international_symbol="P1", operations=[swap])

        @test_throws ArgumentError B10.material_generative_specification(material, B10.SpinAlgebra(2); dataset=dataset, max_body_order=1)
    end

    @testset "Automatic standard symmetry action rejects ambiguous internal indexing" begin
        material = _batch10_two_site_material()
        dataset = _batch10_identity_dataset(2)
        @test_throws ArgumentError B10.material_generative_specification(material, B10.FermionAlgebra(4); dataset=dataset, max_body_order=1)
        @test_throws ArgumentError B10.material_generative_specification(material, B10.CoordinateMomentumAlgebra(2); dataset=dataset, max_body_order=1)

        custom_action = (expression, operation, structure, algebra) -> expression
        explicit_spaces = (B10.FermionFockSpace(2), B10.FermionFockSpace(2))
        custom = B10.material_generative_specification(
            material,
            B10.FermionAlgebra(4);
            dataset=dataset,
            symmetry_action=custom_action,
            local_spaces=explicit_spaces,
            max_body_order=1,
        )
        @test custom.local_spaces == explicit_spaces
        @test custom.metadata[:operator_site_mapping] == :custom
    end

    @testset "Lower-level specification entry remains available" begin
        material = _batch10_two_site_material()
        dataset = _batch10_identity_dataset(2)
        algebra = B10.SpinAlgebra(2)
        crystal = B10.material_crystallographic_structure(material)
        specification = B10.material_generative_specification(material, algebra; dataset=dataset, max_body_order=1)
        result = B10.generate_hamiltonian_family(
            specification,
            crystal;
            candidates=B10.AbstractOperatorExpr[B10.Sx(1), B10.Sy(2)],
            parameter_values=[0.2, -0.1],
        )

        @test isnothing(result.material)
        @test result.metadata[:source] == :generative_specification
        @test B10.parameter_vector(B10.ParameterPoint(result.parameter_space)) == [0.2, -0.1]
    end
end
