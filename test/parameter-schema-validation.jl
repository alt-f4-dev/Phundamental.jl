using Test
using Phundamental

const B9 = Phundamental

@testset "Batch 9 constraint-bearing parameter schemas" begin
    @testset "Structural bindings attach named couplings to Hamiltonian basis directions" begin
        state_space = B9.SpinHilbertSpace([1//2, 1//2])
        algebra = B9.SpinAlgebra(2)
        specification = B9.GenerativeSpecification(:C1, (:site_1, :site_2), (state_space,), algebra, identity; max_body_order=2)
        basis = B9.HamiltonianBasis(B9.AbstractOperatorExpr[B9.Sz(1) * B9.Sz(2), B9.Sx(1)]; labels=[:Jz, :hx])
        space = B9.generate_hamiltonian_space(specification, _ -> basis)
        schema = B9.ParameterSpace(space; values=[1.0, 0.25], bounds=[(-2.0, 2.0), (-1.0, 1.0)], units=[:meV, :meV])

        bindings = B9.parameter_bindings(schema)
        @test length(bindings) == 2
        @test B9.parameter_binding(schema, :Jz).basis_index == 1
        @test B9.parameter_binding(schema, :Jz).basis_label == :Jz
        @test B9.parameter_binding(schema, :Jz).structural_signature == B9.structural_signature(B9.structure(space.basis[1], algebra))
        @test B9.parameter_dof(schema) == 2
        A, b = B9.parameter_constraint_matrix(schema)
        @test size(A) == (0, 2)
        @test isempty(b)
    end

    @testset "Fixed parameters and degenerate bounds are enforced" begin
        state_space = B9.SpinHilbertSpace([1//2, 1//2])
        algebra = B9.SpinAlgebra(2)
        specification = B9.GenerativeSpecification(:C1, (:site_1, :site_2), (state_space,), algebra, identity; max_body_order=1)
        basis = B9.HamiltonianBasis(B9.AbstractOperatorExpr[B9.Sz(1), B9.Sz(2)]; labels=[:h1, :h2])
        space = B9.generate_hamiltonian_space(specification, _ -> basis)
        fixed_schema = B9.ParameterSpace(space; values=[1.0, 0.25], fixed=[true, false])
        bounded_schema = B9.ParameterSpace(space; values=[1.0, 0.25], bounds=[(1.0, 1.0), (-Inf, Inf)])

        @test B9.parameter_dof(fixed_schema) == 1
        @test B9.parameter_dof(bounded_schema) == 1
        @test B9.valid_parameter_point(fixed_schema, [1.0, -0.5])
        @test !B9.valid_parameter_point(fixed_schema, [0.9, -0.5])
        @test_throws ArgumentError B9.ParameterPoint(fixed_schema, [0.9, -0.5])
        @test any(constraint -> constraint isa B9.FixedParameterConstraint, B9.parameter_constraints(fixed_schema))
    end

    @testset "Shared and linear coupling constraints are inspectable" begin
        state_space = B9.SpinHilbertSpace([1//2])
        algebra = B9.SpinAlgebra(1)
        specification = B9.GenerativeSpecification(:C1, (:site_1,), (state_space,), algebra, identity; max_body_order=1)
        basis = B9.HamiltonianBasis(B9.AbstractOperatorExpr[B9.Sz(1), B9.Sx(1), B9.Sy(1)]; labels=[:hz, :hx, :hy])
        space = B9.generate_hamiltonian_space(specification, _ -> basis)
        shared = B9.EqualParameterConstraint(:hx, :hy)
        schema = B9.ParameterSpace(space; values=[1.0, 0.5, 0.5], constraints=(shared,), units=[:meV, :meV, :meV])

        @test B9.parameter_dof(schema) == 2
        @test B9.parameter_constraint_report(schema, [1.0, 0.5, 0.5]).valid
        invalid_report = B9.parameter_constraint_report(schema, [1.0, 0.5, 0.25])
        @test !invalid_report.valid
        @test any(check -> check.name == :equal_hx_hy && !check.satisfied, invalid_report.checks)
        @test_throws ArgumentError B9.ParameterPoint(schema, [1.0, 0.5, 0.25])

        linear = B9.LinearParameterConstraint(Dict(:hx => 2.0, :hy => -1.0), 0.0; name=:anisotropy_relation)
        linear_schema = B9.ParameterSpace(space; values=[1.0, 0.5, 1.0], constraints=(linear,))
        A, b = B9.parameter_constraint_matrix(linear_schema)
        names = B9.parameter_names(linear_schema)
        hx = findfirst(==(:hx), names)
        hy = findfirst(==(:hy), names)

        @test B9.parameter_dof(linear_schema) == 2
        @test A[1, hx] == 2.0
        @test A[1, hy] == -1.0
        @test b == [0.0]
        @test B9.parameter_constraint_report(linear_schema, [1.0, 0.5, 1.0]).valid
        @test !B9.parameter_constraint_report(linear_schema, [1.0, 0.5, 0.9]).valid
    end

    @testset "Schema construction rejects invalid references and incompatible equality units" begin
        specifications = B9.ParameterSpec[
            B9.ParameterSpec(:a, 1.0; unit=:meV),
            B9.ParameterSpec(:b, 1.0; unit=:K),
        ]

        @test_throws ArgumentError B9.ParameterSpace(specifications; constraints=(B9.EqualParameterConstraint(:a, :missing),))
        @test_throws ArgumentError B9.ParameterSpace(specifications; constraints=(B9.EqualParameterConstraint(:a, :b),))
        @test_throws ArgumentError B9.ParameterSpace(
            B9.ParameterSpec[B9.ParameterSpec(:a, 1.0), B9.ParameterSpec(:b, 2.0)];
            constraints=(B9.EqualParameterConstraint(:a, :b),),
        )
    end

    @testset "Model parameter schema cannot be bypassed by a foreign ParameterPoint" begin
        state_space = B9.SpinHilbertSpace([1//2])
        algebra = B9.SpinAlgebra(1)
        representation = B9.Representation(:batch9_spin, state_space, algebra, B9.SpinProductBasis([1]))
        specification = B9.GenerativeSpecification(:C1, (:site_1,), (state_space,), algebra, identity; max_body_order=1)
        basis = B9.HamiltonianBasis(B9.AbstractOperatorExpr[B9.Sz(1), B9.Sx(1)]; labels=[:hz, :hx])
        space = B9.generate_hamiltonian_space(specification, _ -> basis)
        constrained_schema = B9.ParameterSpace(space; values=[0.5, 0.5], constraints=(B9.EqualParameterConstraint(:hz, :hx),))
        model = B9.instantiate_model(space, representation, B9.ParameterPoint(constrained_schema))
        foreign_schema = B9.ParameterSpace(space; values=[0.5, 0.25])
        violating_point = B9.ParameterPoint(foreign_schema)
        valid_point = B9.ParameterPoint(B9.ParameterSpace(space; values=[0.25, 0.25]))

        @test B9.parameter_constraint_report(model).valid
        @test_throws ArgumentError B9.with_parameters(model, violating_point)
        moved = B9.with_parameters(model, valid_point)
        @test B9.parameter_vector(moved) == [0.25, 0.25]
        @test B9.parameter_constraint_report(moved).valid
    end

    @testset "Exact transformations preserve coupling constraints and regenerate structural bindings" begin
        state_space = B9.SpinHilbertSpace([1//2])
        algebra = B9.SpinAlgebra(1)
        representation = B9.Representation(:batch9_transform_source, state_space, algebra, B9.SpinProductBasis([1]))
        specification = B9.GenerativeSpecification(:C1, (:site_1,), (state_space,), algebra, identity; max_body_order=1)
        basis = B9.HamiltonianBasis(B9.AbstractOperatorExpr[B9.Sz(1), B9.Sx(1)]; labels=[:hz, :hx])
        space = B9.generate_hamiltonian_space(specification, _ -> basis)
        schema = B9.ParameterSpace(space; values=[0.5, 0.5], constraints=(B9.EqualParameterConstraint(:hz, :hx),))
        source = B9.instantiate_model(space, representation, B9.ParameterPoint(schema))
        target = B9.transform(B9.JordanWigner([1]), source).model
        source_schema = B9.parameter_space(source)
        target_schema = B9.parameter_space(target)

        @test B9.parameter_names(source_schema) == B9.parameter_names(target_schema)
        @test length(B9.parameter_constraints(target_schema; include_implicit=false)) == 1
        @test only(B9.parameter_constraints(target_schema; include_implicit=false)) isa B9.EqualParameterConstraint
        @test B9.parameter_constraint_report(target).valid
        @test B9.parameter_binding(source_schema, :hz).structural_signature != B9.parameter_binding(target_schema, :hz).structural_signature
        @test B9.parameter_binding(target_schema, :hz).basis_label == :hz
    end

    @testset "Pre-Batch-9 positional ParameterSpace construction remains accepted" begin
        metadata = Dict{Symbol,Any}(:legacy => true)
        specifications = B9.ParameterSpec[B9.ParameterSpec(:g, 0.5)]
        schema = B9.ParameterSpace(specifications, nothing, metadata)

        @test schema.metadata === metadata
        @test isempty(schema.constraints)
        @test isempty(schema.bindings)
        @test B9.ParameterPoint(schema)[:g] == 0.5
    end
end
