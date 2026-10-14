using Test
using Phundamental

const FMS = Phundamental

@testset "Physical model-space admissibility API" begin
    spin_space = FMS.SpinHilbertSpace([1//2, 1//2])
    spin_algebra = FMS.SpinAlgebra(2)
    representation = FMS.Representation(:batch7_spin, spin_space, spin_algebra, FMS.SpinProductBasis([1, 2]))

    @testset "Structured reports preserve rejection reasons" begin
        onsite_only = FMS.PredicateAdmissibility(
            :onsite_only,
            operator -> FMS.support_size(operator) <= 1;
            description="candidate must act on at most one spin site",
        )
        specification = FMS.GenerativeSpecification(
            :C1,
            (:site_1, :site_2),
            (spin_space,),
            spin_algebra,
            identity;
            admissibility_conditions=(onsite_only,),
            max_body_order=2,
        )

        accepted = FMS.admissibility_report(specification, FMS.Sz(1))
        rejected = FMS.admissibility_report(specification, FMS.Sz(1) * FMS.Sz(2))

        @test accepted.complete
        @test accepted.admissible
        @test FMS.admissible(accepted)
        @test rejected.complete
        @test !rejected.admissible
        @test !FMS.admissible(specification, FMS.Sz(1) * FMS.Sz(2))
        @test "candidate must act on at most one spin site" in FMS.admissibility_reasons(rejected)
        @test any(check -> check.name == :onsite_only && check.passed === false, rejected.checks)
    end

    @testset "Algebra-aware body order is reported explicitly" begin
        specification = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity; max_body_order=1)
        report = FMS.admissibility_report(specification, FMS.Sz(1) * FMS.Sz(2))
        order_check = only(filter(check -> check.name == :max_body_order, report.checks))

        @test report.complete
        @test !report.admissible
        @test order_check.passed === false
        @test order_check.metadata[:body_order] == 2
        @test order_check.metadata[:max_body_order] == 1
    end

    @testset "Operator algebra and index bounds are physical checks" begin
        specification = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity)
        wrong_family = FMS.admissibility_report(specification, FMS.c(1))
        out_of_bounds = FMS.admissibility_report(specification, FMS.Sz(3))

        @test wrong_family.complete
        @test !wrong_family.admissible
        @test out_of_bounds.complete
        @test !out_of_bounds.admissible
        @test only(wrong_family.checks).name == :operator_algebra
        @test only(out_of_bounds.checks).name == :operator_algebra
    end

    @testset "Physical range is explicit and never inferred from indices" begin
        range_evaluator = operator -> FMS.support_size(operator) <= 1 ? 0.0 : 2.0
        specification = FMS.GenerativeSpecification(
            :C1,
            (:site_1, :site_2),
            (spin_space,),
            spin_algebra,
            identity;
            max_range=1.0,
            max_body_order=2,
            range_evaluator=range_evaluator,
        )

        local_report = FMS.admissibility_report(specification, FMS.Sz(1))
        distant_report = FMS.admissibility_report(specification, FMS.Sz(1) * FMS.Sz(2))
        range_check = only(filter(check -> check.name == :max_range, distant_report.checks))

        @test local_report.complete
        @test local_report.admissible
        @test distant_report.complete
        @test !distant_report.admissible
        @test range_check.metadata[:interaction_range] == 2.0
        @test range_check.metadata[:max_range] == 1.0

        unresolved = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity; max_range=1.0)
        unresolved_report = FMS.admissibility_report(unresolved, FMS.Sz(1))
        unresolved_check = only(filter(check -> check.name == :max_range, unresolved_report.checks))

        @test !unresolved_report.complete
        @test !unresolved_report.admissible
        @test ismissing(unresolved_check.passed)
        @test_throws ArgumentError FMS.generate_hamiltonian_space(
            unresolved,
            _ -> FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1)]),
        )
    end

    @testset "Generated spaces are audited but direct spaces remain inspectable" begin
        range_evaluator = operator -> FMS.support_size(operator) <= 1 ? 0.0 : 2.0
        specification = FMS.GenerativeSpecification(
            :C1,
            (:site_1, :site_2),
            (spin_space,),
            spin_algebra,
            identity;
            max_range=1.0,
            max_body_order=2,
            range_evaluator=range_evaluator,
        )
        mixed_basis = FMS.HamiltonianBasis(
            FMS.AbstractOperatorExpr[FMS.Sz(1), FMS.Sz(1) * FMS.Sz(2)];
            labels=[:local, :distant],
        )
        direct_space = FMS.HamiltonianSpace(specification, mixed_basis)
        space_report = FMS.admissibility_report(direct_space)

        @test space_report.complete
        @test !space_report.admissible
        @test !FMS.admissible(direct_space)
        @test any(check -> check.name == :distant && check.passed === false, space_report.checks)
        @test_throws ArgumentError FMS.generate_hamiltonian_space(specification, _ -> mixed_basis)

        generated = FMS.generate_hamiltonian_space(
            specification,
            _ -> FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1)]; labels=[:local]),
        )
        @test FMS.admissible(generated)
    end

    @testset "Instantiated and reparameterized models remain inside the declared space" begin
        one_direction = FMS.PredicateAdmissibility(
            :one_direction,
            operator -> !(operator isa FMS.OperatorSum);
            description="selected Hamiltonian must contain one active basis direction",
        )
        one_site_space = FMS.SpinHilbertSpace([1//2])
        one_site_algebra = FMS.SpinAlgebra(1)
        one_site_representation = FMS.Representation(
            :batch7_one_site,
            one_site_space,
            one_site_algebra,
            FMS.SpinProductBasis([1]),
        )
        specification = FMS.GenerativeSpecification(
            :C1,
            (:site_1,),
            (one_site_space,),
            one_site_algebra,
            identity;
            admissibility_conditions=(one_direction,),
            max_body_order=1,
        )
        basis = FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1), FMS.Sx(1)]; labels=[:hz, :hx])
        space = FMS.generate_hamiltonian_space(specification, _ -> basis)
        parameters = FMS.ParameterSpace(space; values=[1.0, 0.0])
        model = FMS.instantiate_model(space, one_site_representation, FMS.ParameterPoint(parameters))

        @test FMS.admissibility_report(model).complete
        @test FMS.admissible(model)
        @test_throws ArgumentError FMS.with_parameters(model, FMS.parameter_point(parameters; hz=1.0, hx=1.0))

        unconstrained = FMS.ManyBodyModel(one_site_representation, FMS.Sz(1))
        @test_throws ArgumentError FMS.admissibility_report(unconstrained)
    end

    @testset "Physical admissibility is independent of solver compatibility" begin
        specification = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity; max_body_order=2)
        basis = FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1) * FMS.Sz(2)]; labels=[:Jz])
        space = FMS.generate_hamiltonian_space(specification, _ -> basis)
        model = FMS.instantiate_model(space, representation, [1.0])
        solver_report = FMS.solver_compatibility(FMS.QuadraticFermionSolver(), model)

        @test FMS.admissible(model)
        @test !solver_report.compatible
        @test FMS.admissibility_report(model).metadata[:scope] == :many_body_model
    end

    @testset "HamiltonianSpace distinguishes specification admissibility from certified basis membership" begin
        specification = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity; max_body_order=2)
        basis = FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1)]; labels=[:hz])
        space = FMS.generate_hamiltonian_space(specification, _ -> basis)
        in_span = FMS.admissibility_report(space, 2.0 * FMS.Sz(1))
        outside_span = FMS.admissibility_report(space, FMS.Sx(1))

        @test space isa FMS.AbstractModelSpace
        @test in_span.complete
        @test in_span.admissible
        @test only(filter(check -> check.name == :basis_span, in_span.checks)).passed === true
        @test outside_span.complete
        @test !outside_span.admissible
        @test only(filter(check -> check.name == :basis_span, outside_span.checks)).passed === false
        @test FMS.admissibility_report(specification, FMS.Sx(1)).admissible

        summed_basis = FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1) + FMS.Sz(2)]; labels=[:uniform_z])
        summed_space = FMS.generate_hamiltonian_space(specification, _ -> summed_basis)
        summed_model = FMS.instantiate_model(summed_space, representation, [0.5])
        @test FMS.admissibility_report(summed_model).complete
        @test FMS.admissible(summed_model)
    end

    @testset "Representation-specific range evaluators are not silently transported" begin
        range_evaluator = _ -> 0.0
        specification = FMS.GenerativeSpecification(
            :C1,
            (:site_1, :site_2),
            (spin_space,),
            spin_algebra,
            identity;
            max_range=1.0,
            max_body_order=2,
            range_evaluator=range_evaluator,
        )
        basis = FMS.HamiltonianBasis(FMS.AbstractOperatorExpr[FMS.Sz(1)]; labels=[:hz])
        space = FMS.generate_hamiltonian_space(specification, _ -> basis)
        model = FMS.instantiate_model(space, representation, [0.5])
        transformed = FMS.transform(FMS.JordanWigner([1, 2]), model).model
        transformed_report = FMS.admissibility_report(transformed)

        @test transformed.specification.max_range == 1.0
        @test isnothing(transformed.specification.range_evaluator)
        @test transformed.specification.metadata[:range_evaluator_transport] == :not_assumed
        @test !transformed_report.complete
        @test !transformed_report.admissible
        @test any(check -> check.name == :max_range && ismissing(check.passed), transformed_report.checks)
    end

    @testset "Pre-Batch-7 positional specification construction remains accepted" begin
        metadata = Dict{Symbol,Any}(:legacy => true)
        specification = FMS.GenerativeSpecification(:C1, (:site_1, :site_2), (spin_space,), spin_algebra, identity, (), nothing, 2, metadata)

        @test isnothing(specification.range_evaluator)
        @test specification.metadata === metadata
        @test FMS.admissible(specification, FMS.Sz(1) * FMS.Sz(2))
    end
end
