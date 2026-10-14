using Test
using LinearAlgebra
using Phundamental

const B11 = Phundamental

function _batch11_space(n::Integer=3; labels=[Symbol("h", i) for i in 1:n])
    algebra = B11.SpinAlgebra(n)
    local_spaces = Tuple(B11.SpinHilbertSpace([1//2]) for _ in 1:n)
    specification = B11.GenerativeSpecification(:C1, Tuple(1:n), local_spaces, algebra, identity; max_body_order=1)
    basis = B11.HamiltonianBasis(B11.AbstractOperatorExpr[B11.Sz(i) for i in 1:n]; labels=labels,
                                  metadata=Dict(:completeness_scope => :controlled_test_basis, :linear_tolerance => 1e-10))
    return B11.generate_hamiltonian_space(specification, _ -> basis)
end

function _batch11_two_site_material()
    lattice = B11.BravaisLattice(Matrix{Float64}(I, 3, 3))
    crystal = B11.LatticeCrystalStructure(lattice,
        B11.BasisSite[B11.BasisSite(:A1, :A, [0.0, 0.0, 0.0]), B11.BasisSite(:A2, :A, [0.5, 0.0, 0.0])])
    return B11.Material("Batch 11 symmetric pair", crystal;
        species=Dict(:A => B11.AtomicSpecies(:A; mass=1.0)),
        basis_properties=[B11.SiteProperties(spin=1//2), B11.SiteProperties(spin=1//2)])
end

@testset "Batch 11 crystallography-constrained structural search" begin
    @testset "Canonical invariant directions preserve complex ratios and quotient real rescaling" begin
        algebra = B11.SpinAlgebra(2)
        x = B11.Sz(1) + 0.5 * B11.Sz(2)
        reference = B11.invariant_direction_signature(x, algebra)
        @test reference == B11.invariant_direction_signature(2 * x, algebra)
        @test reference == B11.invariant_direction_signature(-4 * x, algebra)
        @test reference != B11.invariant_direction_signature(B11.Sz(1) + 0.25 * B11.Sz(2), algebra)
        @test_throws ArgumentError B11.invariant_direction_signature(0 * B11.Sz(1), algebra)
    end

    @testset "Parent family partitions and canonical identity" begin
        space = _batch11_space()
        family = B11.structural_family(space)
        @test length(family.units) == 3
        @test B11.structural_signature(family) == family.signature
        @test all(unit -> length(unit.basis_indices) == 1, family.units)
        @test_throws ArgumentError B11.structural_family(space; require_crystallographic=true)
        @test_throws ArgumentError B11.structural_family(space; groups=[:combined => [:h1, :h2]])
        @test_throws ArgumentError B11.structural_family(space; groups=[:u => [:h1, :h2], :v => [:h2, :h3]])
        @test_throws ArgumentError B11.structural_family(space; groups=[:u => :h1, :u => [:h2, :h3]])
        @test_throws ArgumentError B11.structural_family(space; groups=[:u => [:h1, :unknown], :v => [:h2, :h3]])
        grouped = B11.structural_family(space; groups=[:pair => [:h1, :h2], :field => :h3])
        @test length(grouped.units) == 2
        @test grouped.units[1].basis_indices == (1, 2)
        @test B11.structural_selection(grouped, :pair).basis_indices == (1, 2)
    end

    @testset "Selection, transitions, neighborhoods and replay" begin
        family = B11.structural_family(_batch11_space())
        a = B11.structural_selection(family, [:h1, :h2])
        b = B11.structural_selection(family, [:h2, :h1])
        @test a.signature == b.signature
        @test a.basis_indices == (1, 2)
        @test B11.structural_signature(a) == a.signature
        @test a.hamiltonian_space.specification === family.parent.specification
        @test a.hamiltonian_space.basis.metadata[:completeness_scope] == :selected_structural_subspace
        @test a.hamiltonian_space.basis.metadata[:parent_completeness_scope] == :controlled_test_basis
        @test B11.admissible(a.hamiltonian_space)
        @test_throws ArgumentError B11.structural_selection(family, ())
        @test_throws ArgumentError B11.structural_selection(family, [:h1, :h1])
        @test_throws ArgumentError B11.structural_selection(family, :unknown)

        c = B11.structural_transition(a; remove_units=:h2, add_units=:h3)
        move = B11.structural_move(a, c)
        @test move.kind == :replace
        @test B11.apply_structural_move(a, move).signature == c.signature
        @test_throws ArgumentError B11.apply_structural_move(c, move)
        @test_throws ArgumentError B11.structural_transition(a)
        @test_throws ArgumentError B11.structural_transition(a; add_units=:h1)
        @test_throws ArgumentError B11.structural_transition(a; remove_units=:h3)
        @test_throws ArgumentError B11.structural_transition(a; add_units=:h3, remove_units=:h3)
        @test_throws ArgumentError B11.structural_transition(B11.structural_selection(family, :h1); remove_units=:h1)
        @test B11.structural_move(a, B11.structural_transition(a; add_units=:h3)).kind == :add
        @test B11.structural_move(a, B11.structural_transition(a; remove_units=:h2)).kind == :remove

        neighborhood = B11.structural_neighbors(a)
        @test length(neighborhood.neighbors) == 5
        @test count(move -> move.kind == :replace, neighborhood.moves) == 2
        @test count(move -> move.kind == :add, neighborhood.moves) == 1
        @test count(move -> move.kind == :remove, neighborhood.moves) == 2
        @test all(pair -> B11.apply_structural_move(a, pair[2]).signature == pair[1].signature,
                  zip(neighborhood.neighbors, neighborhood.moves))
        @test_throws ArgumentError B11.structural_neighbors(a; required_units=:h3)
        @test_throws ArgumentError B11.structural_neighbors(a; max_neighbors=2)
        constrained = B11.structural_neighbors(a; required_units=:h1, forbidden_units=:h3)
        @test length(constrained.neighbors) == 1
        @test constrained.neighbors[1].unit_ids == (:h1,)
    end

    @testset "Finite enumeration, cardinality guards and Hasse hierarchy" begin
        family = B11.structural_family(_batch11_space())
        selections = B11.enumerate_structures(family)
        @test length(selections) == 7
        @test length(unique(map(item -> item.signature, selections))) == 7
        @test_throws ArgumentError B11.enumerate_structures(family; max_structures=6)
        @test length(B11.enumerate_structures(family; required_units=:h1)) == 4
        @test length(B11.enumerate_structures(family; forbidden_units=:h3)) == 3
        @test length(B11.enumerate_structures(family; min_units=2, max_units=2)) == 3
        @test isempty(B11.enumerate_structures(family; required_units=:h1, forbidden_units=[:h2, :h3], min_units=2))
        hierarchy = B11.model_space_hierarchy(family)
        @test length(hierarchy.selections) == 7
        @test length(hierarchy.edges) == 9
        singleton = B11.structural_selection(family, :h1)
        pair = B11.structural_selection(family, [:h1, :h2])
        full = B11.structural_selection(family, [:h1, :h2, :h3])
        @test isempty(B11.parent_spaces(hierarchy, singleton))
        @test length(B11.child_spaces(hierarchy, singleton)) == 2
        @test length(B11.parent_spaces(hierarchy, pair)) == 2
        @test length(B11.child_spaces(hierarchy, pair)) == 1
        @test length(B11.parent_spaces(hierarchy, full)) == 3
        @test isempty(B11.child_spaces(hierarchy, full))
        @test B11.is_subspace(singleton, pair)
        @test !B11.is_subspace(pair, singleton)
        @test B11.is_subspace(singleton.hamiltonian_space, pair.hamiltonian_space)
        @test !B11.is_subspace(pair.hamiltonian_space, singleton.hamiltonian_space)
    end

    @testset "Nested coordinate maps and parent-parameter compatibility" begin
        space = _batch11_space()
        schema = B11.ParameterSpace(space; values=[0.0, 0.0, 0.4], bounds=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                                    units=[:meV, :meV, :meV], constraints=(B11.EqualParameterConstraint(:h1, :h2),))
        family = B11.structural_family(space; parameter_space=schema)
        small = B11.structural_selection(family, :h1)
        large = B11.structural_selection(family, [:h1, :h3])
        map = B11.structural_coordinate_map(small, large)
        @test B11.embed_coordinates(map, [0.3]) == [0.3, 0.0]
        @test B11.project_coordinates(map, [0.3, 0.2]) == [0.3]
        @test_throws ArgumentError B11.project_coordinates(map, [0.3, 0.2]; strict=true)
        @test_throws ArgumentError B11.structural_coordinate_map(large, small)
        @test B11.embed_parameters(small, [0.0]) == [0.0, 0.0, 0.0]
        @test_throws ArgumentError B11.embed_parameters(small, [0.5])
        @test_throws ArgumentError B11.project_parameters(small, [0.0, 0.0, 0.3])
        @test B11.project_parameters(small, [0.0, 0.0, 0.3]; strict=false) == [0.0]
        child = B11.parameter_subspace(small, [0.0])
        @test B11.parameter_names(child) == [:h1]
        @test B11.parameter_binding(child, :h1).basis_index == 1
        @test any(c -> c isa B11.FixedParameterConstraint && c.parameter == :h1 && c.value == 0,
                  B11.parameter_constraints(child; include_implicit=false))
        @test !B11.valid_parameter_point(child, [0.2])
        @test B11.parameter_units(child) == [:meV]
    end

    @testset "Linear and fixed parent constraints survive structural restriction" begin
        space = _batch11_space()
        schema = B11.ParameterSpace(space; values=[0.0, 0.0, 0.0], constraints=(
            B11.LinearParameterConstraint(Dict(:h1 => 2.0, :h2 => -1.0), 0.0; name=:relation),
            B11.FixedParameterConstraint(:h3, 0.0)))
        selection = B11.structural_selection(B11.structural_family(space; parameter_space=schema), :h1)
        child = B11.parameter_subspace(selection, [0.0])
        @test any(c -> c isa B11.LinearParameterConstraint && c.coefficients == Dict(:h1 => 2.0), child.constraints)
        @test !B11.valid_parameter_point(child, [0.5])
        invalid = B11.ParameterSpace(space; values=[0.0, 0.0, 1.0], fixed=[false, false, true])
        @test_throws ArgumentError B11.parameter_subspace(B11.structural_selection(B11.structural_family(space; parameter_space=invalid), :h1), [0.0])
    end

    @testset "Batch 10 crystallographic fixed-space directions are indivisible" begin
        material = _batch11_two_site_material()
        identity_op = B11.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        swap_op = B11.SpaceGroupOperation(Matrix{Int}(I, 3, 3), [0.5, 0.0, 0.0])
        dataset = B11.CrystallographicDataset(spacegroup_number=1, international_symbol="P1",
                                              operations=[identity_op, swap_op], crystallographic_orbits=[1, 1])
        result = B11.generate_hamiltonian_family(material, B11.SpinAlgebra(2); dataset=dataset, max_body_order=1,
                                                  candidates=B11.AbstractOperatorExpr[B11.Sz(1), B11.Sz(2)],
                                                  parameter_values=[0.0])
        family = B11.structural_family(result)
        @test length(family.units) == 1
        @test family.units[1].metadata[:symmetry_complete]
        @test length(B11.structure(family.parent).terms) == 2
        @test_throws ArgumentError B11.structural_selection(family, ())
        selected = B11.structural_selection(family, family.units[1].id)
        @test length(B11.structure(selected).terms) == 2
        @test selected.hamiltonian_space.basis.metadata[:parent_completeness_scope] == :supplied_candidate_space
        @test length(B11.enumerate_structures(family)) == 1
    end
end
