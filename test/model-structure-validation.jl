using Test
using Phundamental

const B8 = Phundamental

@testset "Batch 8 canonical Hamiltonian structures" begin
    @testset "Coefficient-independent canonical identity" begin
        representation = B8.Representation(
            :batch8_spin,
            B8.SpinHilbertSpace([1//2, 1//2]),
            B8.SpinAlgebra(2),
            B8.SpinProductBasis([1, 2]),
        )
        first_model = B8.ManyBodyModel(representation, 2.0 * B8.Sz(1) * B8.Sz(2) + 0.5 * B8.Sx(1))
        reordered_model = B8.ManyBodyModel(representation, -3.0 * B8.Sx(1) + 7.0 * B8.Sz(1) * B8.Sz(2))
        reduced_model = B8.ManyBodyModel(representation, 4.0 * B8.Sz(1) * B8.Sz(2))

        first_structure = B8.structure(first_model)
        reordered_structure = B8.structure(reordered_model)

        @test first_structure isa B8.ModelStructure
        @test B8.structurally_equivalent(first_model, reordered_model)
        @test B8.structural_signature(first_model) == B8.structural_signature(reordered_model)
        @test !B8.structurally_equivalent(first_model, reduced_model)
        @test length(first_structure.terms) == 2
        @test length(B8.interaction_graph(first_structure).vertices) == 2
        @test length(B8.interaction_graph(first_structure).edges) == 2
    end

    @testset "Representation names and numerical truncations do not define physical structural identity" begin
        source_representation = B8.Representation(
            :source_name,
            B8.SpinHilbertSpace([1//2]),
            B8.SpinAlgebra(1),
            B8.SpinProductBasis([1]),
        )
        renamed_representation = B8.Representation(
            :renamed_only,
            B8.SpinHilbertSpace([1//2]),
            B8.SpinAlgebra(1),
            B8.SpinProductBasis([1]),
        )
        source = B8.ManyBodyModel(source_representation, B8.Sz(1))
        renamed = B8.ManyBodyModel(renamed_representation, 9.0 * B8.Sz(1))

        @test B8.representation_signature(source_representation) == B8.representation_signature(renamed_representation)
        @test B8.structurally_equivalent(source, renamed)
    end

    @testset "Exact algebraic canonicalization feeds structural identity" begin
        representation = B8.Representation(
            :fermionic_identity,
            B8.FermionFockSpace(1),
            B8.FermionAlgebra(1),
            B8.FermionOccupationBasis([1]),
        )
        left = B8.ManyBodyModel(representation, B8.c(1) * B8.c(1)')
        right = B8.ManyBodyModel(representation, B8.IdentityOperator() - B8.nf(1))

        @test B8.structurally_equivalent(left, right)
    end

    @testset "Canonical linear span closes Batch 7 structural membership gap" begin
        state_space = B8.FermionFockSpace(1)
        algebra = B8.FermionAlgebra(1)
        specification = B8.GenerativeSpecification(:C1, (:mode_1,), (state_space,), algebra, identity; max_body_order=1)
        basis = B8.HamiltonianBasis(B8.AbstractOperatorExpr[B8.IdentityOperator(), B8.nf(1)]; labels=[:constant, :density])
        space = B8.generate_hamiltonian_space(specification, _ -> basis)

        equivalent_candidate = B8.c(1) * B8.c(1)'
        report = B8.admissibility_report(space, equivalent_candidate)

        @test report.complete
        @test report.admissible
        @test only(filter(check -> check.name == :basis_span, report.checks)).metadata[:certification] == :canonical_linear_span
    end

    @testset "HamiltonianSpace exposes family structure independently of parameter values" begin
        state_space = B8.SpinHilbertSpace([1//2, 1//2])
        algebra = B8.SpinAlgebra(2)
        specification = B8.GenerativeSpecification(:C1, (:site_1, :site_2), (state_space,), algebra, identity; max_body_order=2)
        basis = B8.HamiltonianBasis(B8.AbstractOperatorExpr[B8.Sz(1) * B8.Sz(2), B8.Sx(1)]; labels=[:Jz, :hx])
        space = B8.generate_hamiltonian_space(specification, _ -> basis)
        family_structure = B8.structure(space)

        @test family_structure.metadata[:scope] == :hamiltonian_space
        @test length(family_structure.terms) == 2
        @test length(B8.structural_signature(space).terms) == 2
    end

    @testset "Exact transformation relation is not structural identity" begin
        source_representation = B8.Representation(
            :spin_source,
            B8.SpinHilbertSpace([1//2]),
            B8.SpinAlgebra(1),
            B8.SpinProductBasis([1]),
        )
        source = B8.ManyBodyModel(source_representation, 0.8 * B8.Sz(1) + 0.3 * B8.Sx(1))
        transformation = B8.JordanWigner([1])
        target = B8.transform(transformation, source).model

        @test !B8.structurally_equivalent(source, target)
        @test B8.structurally_related(source, target, transformation)
    end

    @testset "Unrelated representations remain structurally distinct" begin
        spin_representation = B8.Representation(
            :spin,
            B8.SpinHilbertSpace([1//2]),
            B8.SpinAlgebra(1),
            B8.SpinProductBasis([1]),
        )
        oscillator_representation = B8.Representation(
            :oscillator,
            B8.CoordinateHilbertSpace(1),
            B8.CoordinateMomentumAlgebra(1),
            B8.CoordinateBasis([1]),
        )
        spin_model = B8.ManyBodyModel(spin_representation, B8.Sz(1))
        oscillator_model = B8.ManyBodyModel(oscillator_representation, B8.pop(1) * B8.pop(1) / 2 + B8.xop(1) * B8.xop(1) / 2)

        @test !B8.structurally_equivalent(spin_model, oscillator_model)
    end
end
