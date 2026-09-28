using Test
using LinearAlgebra

function _bond_selection_fixture()
    lattice = BravaisLattice(reshape([2.0], 1, 1))
    properties = SiteProperties(spin=0.5, g_tensor=2.0)
    basis = BasisSite[
        BasisSite(:Cu1, :Cu, [0.0]; properties=properties),
        BasisSite(:Cu2, :Cu, [0.5]; properties=properties),
    ]
    crystal = LatticeCrystalStructure(lattice, basis)
    cluster = Supercell(crystal, (1,); periodic=false)
    material = Material(
        "geometric bond-selection fixture",
        crystal;
        species=Dict{Symbol,Any}(:Cu => lookup_species(:Cu)),
        basis_properties=[properties, properties],
    )
    return material, cluster
end

function _assert_bond_models_equivalent(material, cluster, shell_spec, cutoff_spec)
    shell_model = build_model(material, cluster, shell_spec)
    cutoff_model = build_model(material, cluster, cutoff_spec)

    shell_bonds = shell_model.parameters[:bonds]
    cutoff_bonds = cutoff_model.parameters[:bonds]
    @test length(shell_bonds) == length(cutoff_bonds) == 1
    @test shell_bonds[1].i == cutoff_bonds[1].i
    @test shell_bonds[1].j == cutoff_bonds[1].j
    @test shell_bonds[1].shell == cutoff_bonds[1].shell == 1
    @test isapprox(shell_bonds[1].distance, cutoff_bonds[1].distance; atol=1e-12, rtol=0.0)
    @test isapprox(shell_bonds[1].displacement, cutoff_bonds[1].displacement; atol=1e-12, rtol=0.0)

    shell_solution = solve(ExactDiagonalization(), shell_model)
    cutoff_solution = solve(ExactDiagonalization(), cutoff_model)
    @test isapprox(shell_solution.energies, cutoff_solution.energies; atol=1e-12, rtol=1e-12)
end

@testset "Shared geometric bond-selection API" begin
    material, cluster = _bond_selection_fixture()
    cutoff = 1.0 + 1e-8

    @testset "Longitudinal XXZ phonon alias" begin
        @test LongitudinalXXZPhononModel === LongitudinalSpinBosonModel

        canonical_spec = LongitudinalSpinBosonModel(0.7, 1.1, 0.9, 0.2; field_z=0.05, phonon_cutoff=1, shell=1)
        alias_spec = LongitudinalXXZPhononModel(0.7, 1.1, 0.9, 0.2; field_z=0.05, phonon_cutoff=1, shell=1)
        @test typeof(alias_spec) === typeof(canonical_spec)
        @test alias_spec isa LongitudinalSpinBosonModel

        canonical_model = build_model(material, cluster, canonical_spec)
        alias_model = build_model(material, cluster, alias_spec)
        canonical_solution = solve(ExactDiagonalization(), canonical_model)
        alias_solution = solve(ExactDiagonalization(), alias_model)

        @test canonical_model.provenance == alias_model.provenance
        @test isapprox(alias_solution.energies, canonical_solution.energies; atol=1e-12, rtol=1e-12)
    end

    cases = [
        (
            :heisenberg,
            HeisenbergModel(1.2; field=(0.03, -0.02, 0.05), shell=1),
            HeisenbergModel(1.2; field=(0.03, -0.02, 0.05), neighbor_cutoff=cutoff, max_shell=1),
        ),
        (
            :xxz,
            XXZModel(0.8, 1.1; field_z=0.07, shell=1),
            XXZModel(0.8, 1.1; field_z=0.07, neighbor_cutoff=cutoff, max_shell=1),
        ),
        (
            :hubbard,
            HubbardModel(0.5, 1.0; chemical_potential=0.1, shell=1),
            HubbardModel(0.5, 1.0; chemical_potential=0.1, neighbor_cutoff=cutoff, max_shell=1),
        ),
        (
            :holstein,
            HolsteinModel(0.4, 1.2, 0.8, 0.15; chemical_potential=0.1, phonon_cutoff=1, shell=1),
            HolsteinModel(0.4, 1.2, 0.8, 0.15; chemical_potential=0.1, phonon_cutoff=1, neighbor_cutoff=cutoff, max_shell=1),
        ),
        (
            :longitudinal_spin_boson,
            LongitudinalSpinBosonModel(0.7, 1.1, 0.9, 0.2; field_z=0.05, phonon_cutoff=1, shell=1),
            LongitudinalSpinBosonModel(0.7, 1.1, 0.9, 0.2; field_z=0.05, phonon_cutoff=1, neighbor_cutoff=cutoff, max_shell=1),
        ),
    ]

    for (name, shell_spec, cutoff_spec) in cases
        @testset "$(name) shell/cutoff equivalence" begin
            @test shell_spec.neighbor_cutoff === nothing
            @test shell_spec.max_shell == 1
            @test cutoff_spec.neighbor_cutoff == cutoff
            @test cutoff_spec.max_shell == 1
            _assert_bond_models_equivalent(material, cluster, shell_spec, cutoff_spec)
        end
    end

    constructors = [
        kwargs -> HeisenbergModel(1.0; kwargs...),
        kwargs -> XXZModel(0.8, 1.1; kwargs...),
        kwargs -> HubbardModel(0.5, 1.0; kwargs...),
        kwargs -> HolsteinModel(0.4, 1.0, 0.8, 0.1; phonon_cutoff=1, kwargs...),
        kwargs -> LongitudinalSpinBosonModel(0.7, 1.0, 0.9, 0.1; phonon_cutoff=1, kwargs...),
    ]

    @testset "uniform constructor validation" begin
        for make_spec in constructors
            @test_throws ArgumentError make_spec((;))
            @test_throws ArgumentError make_spec((; shell=0))
            @test_throws ArgumentError make_spec((; max_shell=0, neighbor_cutoff=cutoff))
            @test_throws ArgumentError make_spec((; neighbor_cutoff=0.0))
            @test_throws ArgumentError make_spec((; shell=1, neighbor_cutoff=cutoff))
            @test_throws ArgumentError make_spec((; shell=1, max_shell=1))
        end
    end
end
