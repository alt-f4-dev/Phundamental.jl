using Test
using LinearAlgebra
using Phundamental

const PHI = Phundamental

function _physical_api_material(; species=(:Cu, :V))
    lattice = PHI.BravaisLattice([4.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 7.5])
    planar = ([0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75])
    sites = [PHI.BasisSite(Symbol("site_", i), i <= 4 ? species[1] : species[2],
        [planar[mod1(i, 4)][1], planar[mod1(i, 4)][2], i <= 4 ? 0.25 : 0.75]) for i in 1:8]
    db = Dict(symbol => PHI.AtomicSpecies(symbol; mass=1.0) for symbol in species)
    material = PHI.Material("interaction API validation", PHI.LatticeCrystalStructure(lattice, sites);
        species=db, basis_properties=[PHI.SiteProperties(spin=1//2) for _ in 1:8])
    rotation = [0 -1 0; 1 0 0; 0 0 1]
    ops = [PHI.SpaceGroupOperation(rotation^n, zeros(3)) for n in 0:3]
    dataset = PHI.CrystallographicDataset(spacegroup_number=75, international_symbol="P4", operations=ops,
        crystallographic_orbits=vcat(fill(1, 4), fill(2, 4)))
    return material, dataset
end

function _physical_api_identity(nsites::Int)
    return PHI.CrystallographicDataset(spacegroup_number=1, international_symbol="P1",
        operations=[PHI.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))], crystallographic_orbits=collect(1:nsites))
end

function _physical_api_small_material()
    lattice = PHI.BravaisLattice(Matrix{Float64}(I, 3, 3))
    sites = [PHI.BasisSite(Symbol("site_", i), Symbol("X", i), [0.4 * (i - 1), 0.0, 0.0]) for i in 1:2]
    return PHI.Material("two-site multi-algebra reference", PHI.LatticeCrystalStructure(lattice, sites);
        species=Dict(site.species => PHI.AtomicSpecies(site.species; mass=1.0) for site in sites),
        basis_properties=[PHI.SiteProperties(spin=1//2) for _ in sites])
end

@testset "Generic high-level physical interaction specification and certified coefficient binding" begin
    @testset "Nontrivial C4 frustrated bilayer: physical bond sums remain exact" begin
        material, dataset = _physical_api_material()
        bonds = [
            :Cu_nn => [(1, 2), (2, 3), (3, 4), (4, 1)],
            :Cu_nnn => [(1, 3), (2, 4)],
            :V_nn => [(5, 6), (6, 7), (7, 8), (8, 5)],
            :V_nnn => [(5, 7), (6, 8)],
            :rung => [(1, 5), (2, 6), (3, 7), (4, 8)],
            :cross_cw => [(1, 6), (2, 7), (3, 8), (4, 5)],
            :cross_ccw => [(1, 8), (2, 5), (3, 6), (4, 7)],
        ]
        interactions = PHI.InteractionSpecification(
            [PHI.ExchangeOrbit(name, supports) for (name, supports) in bonds]...,
            PHI.FieldOrbit(:field_Cu, 1:4), PHI.FieldOrbit(:field_V, 5:8))
        generation = PHI.generate_hamiltonian_family(material, PHI.SpinAlgebra(8); interactions=interactions,
            dataset=dataset, max_body_order=2, max_range=4.40, parameter_units=fill(:meV, 16))
        @test generation isa PHI.InteractionGenerationResult
        @test length(generation.generation.hamiltonian_space) == 16
        @test length(generation.family.units) == 9
        @test size(generation.coefficient_map.transformation) == (16, 16)
        @test maximum(generation.coefficient_map.residuals) < 1e-9
        expected_multiplicities = vcat(Float64[length(pairs) for (_, pairs) in bonds for _ in 1:2], [4.0, 4.0])
        @test generation.coefficient_map.transformation ≈ Diagonal(expected_multiplicities)
        J = Dict(:Cu_nn => (3.5, 4.2), :Cu_nnn => (1.4, 1.2), :V_nn => (3.1, 3.6), :V_nnn => (1.1, 1.5),
            :rung => (2.2, 2.0), :cross_cw => (0.65, 0.55), :cross_ccw => (0.40, 0.38))
        fields = Dict(:field_Cu => 0.25, :field_V => -0.15)
        couplings = PHI.PhysicalCouplings(exchange=J, fields=fields, unit=:meV)
        theta = PHI.physical_parameters(generation, couplings)
        selection = PHI.structural_selection(generation.family, [unit.id for unit in generation.family.units])
        representation = PHI.Representation(:bilayer_exact, PHI.SpinHilbertSpace(fill(1//2, 8)), PHI.SpinAlgebra(8),
            PHI.SpinProductBasis(collect(1:8)))
        model = PHI.instantiate_model(selection, representation, theta)
        manual_terms = PHI.AbstractOperatorExpr[]
        for (name, pairs) in bonds
            xy, zz = J[name]
            for (i, j) in pairs
                push!(manual_terms, xy * (PHI.Sx(i) * PHI.Sx(j) + PHI.Sy(i) * PHI.Sy(j)) + zz * PHI.Sz(i) * PHI.Sz(j))
            end
        end
        append!(manual_terms, [-fields[:field_Cu] * PHI.Sz(i) for i in 1:4])
        append!(manual_terms, [-fields[:field_V] * PHI.Sz(i) for i in 5:8])
        reference = PHI.ManyBodyModel(representation, reduce(+, manual_terms))
        generated = PHI.forward(model)
        benchmark = PHI.forward(reference)
        @test isapprox(sort(generated.solution.energies), sort(benchmark.solution.energies); atol=1e-8)
        @test isapprox(minimum(generated.solution.energies), -13.20109679; atol=1e-6)
        @test PHI.structural_family(generation) === generation.family
        @test PHI.describe(generation)["symmetry_policy"] == "require"
        problem = PHI.spin_scattering_problem(material, representation, [[π/2, 0.0, 0.0]], range(0.0, 10.0; length=101);
            resolution=PHI.GaussianResolution(0.16))
        prepared = PHI.prepare_forward(selection, problem; solver=PHI.ExactDiagonalization())
        scattering = PHI.forward_parameters(prepared, theta; parameter_domain=:parent)
        @test size(scattering.result.intensity) == (1, 101)
        @test isfinite(sum(scattering.result.intensity))
        @test length(problem.observables) == 24
        @test length(problem.positions) == 8
        @test PHI.validate_forward_equivalence(scattering, scattering).valid
        @test all(values(PHI.validate_forward_equivalence(scattering, scattering).checks))
    end

    @testset "Fermionic hopping, density, pairing and bosonic hopping are algebra neutral" begin
        material = _physical_api_small_material()
        dataset = _physical_api_identity(2)
        for (algebra, hopping) in ((PHI.FermionAlgebra(2), :fermion), (PHI.BosonAlgebra(2), :boson))
            interactions = PHI.InteractionSpecification(PHI.HoppingOrbit(:t, [(1, 2)]; statistics=hopping),
                PHI.DensityInteractionOrbit(:V, [(1, 2)]; statistics=hopping))
            result = PHI.generate_interaction_family(material, algebra, interactions; dataset=dataset, max_body_order=2)
            @test length(result.generation.hamiltonian_space) == 2
            @test size(result.coefficient_map.transformation) == (2, 2)
            @test PHI.physical_parameters(result, Dict(:t => 1.1, :V => 2.3))[:t] ≈ 1.1
        end
        pairing = PHI.InteractionSpecification(PHI.PairingOrbit(:Delta, [(1, 2)]))
        result = PHI.generate_interaction_family(material, PHI.FermionAlgebra(2), pairing; dataset=dataset, max_body_order=2)
        @test length(result.family.units) == 1
    end



    @testset "Parent-domain structural projection is explicit and constraint-aware" begin
        material = _physical_api_small_material()
        algebra = PHI.FermionAlgebra(2)
        interactions = PHI.InteractionSpecification(
            PHI.OperatorOrbit(:eps1, [1]; operator=i -> PHI.nf(i)),
            PHI.OperatorOrbit(:eps2, [2]; operator=i -> PHI.nf(i)))
        result = PHI.generate_interaction_family(material, algebra, interactions; dataset=_physical_api_identity(2))
        selection = PHI.structural_selection(result.family, [:eps1])
        representation = PHI.Representation(:two_site_fermion, PHI.FermionFockSpace(2), algebra, PHI.FermionOccupationBasis([1, 2]))
        prepared = PHI.prepare_forward(selection, representation)
        parent = PHI.physical_parameters(result, Dict(:eps1 => 0.8, :eps2 => 0.3))
        @test_throws ArgumentError PHI.forward_parameters(prepared, parent; parameter_domain=:parent)
        response = PHI.forward_parameters(prepared, parent; parameter_domain=:parent, strict=false)
        reference = PHI.forward(PHI.ManyBodyModel(representation, 0.8 * PHI.nf(1)))
        @test PHI.validate_forward_equivalence(response, reference; compare=(:spectrum,)).valid
        different = PHI.forward(PHI.ManyBodyModel(representation, 1.2 * PHI.nf(1)))
        @test !PHI.validate_forward_equivalence(response, different; compare=(:spectrum,)).valid
        @test_throws ArgumentError PHI.forward_parameters(prepared, parent; parameter_domain=:unrecognized)
    end
    @testset "Projection must be explicit for symmetry-breaking input" begin
        material, dataset = _physical_api_material()
        algebra = PHI.SpinAlgebra(8)
        interactions = PHI.InteractionSpecification(PHI.OperatorOrbit(:local_spin, [1]; operator=i -> PHI.Sz(i)))
        @test_throws ArgumentError PHI.generate_interaction_family(material, algebra, interactions; dataset=dataset)
        projected = PHI.generate_interaction_family(material, algebra, interactions; dataset=dataset, symmetry_policy=:project)
        @test projected.coefficient_map.symmetry_policy === :project
        @test PHI.physical_parameters(projected, Dict(:local_spin => 2.0))[:local_spin] ≈ 2.0
    end

    @testset "Custom three-body interactions retain a general operator builder" begin
        lattice = PHI.BravaisLattice(Matrix{Float64}(I, 3, 3))
        sites = [PHI.BasisSite(Symbol("S", i), :X, [0.2 * (i - 1), 0.0, 0.0]) for i in 1:3]
        material = PHI.Material("custom three-body magnet", PHI.LatticeCrystalStructure(lattice, sites);
            species=Dict(:X => PHI.AtomicSpecies(:X; mass=1.0)),
            basis_properties=[PHI.SiteProperties(spin=1//2) for _ in sites])
        interactions = PHI.InteractionSpecification(PHI.OperatorOrbit(:three_body, [(1, 2, 3)];
            operator=(i, j, k) -> PHI.Sz(i) * PHI.Sz(j) * PHI.Sz(k)))
        generated = PHI.generate_interaction_family(material, PHI.SpinAlgebra(3), interactions;
            dataset=_physical_api_identity(3), max_body_order=3)
        @test length(generated.generation.hamiltonian_space) == 1
        @test PHI.physical_parameters(generated, Dict(:three_body => 2.0))[:three_body] ≈ 2.0
    end
    @testset "Symmetry, Hermiticity, redundancy, labels, and units fail explicitly" begin
        material = _physical_api_small_material()
        identity = _physical_api_identity(2)
        algebra = PHI.FermionAlgebra(2)
        @test_throws ArgumentError PHI.InteractionSpecification(PHI.FieldOrbit(:h, [1]), PHI.FieldOrbit(:h, [2]))
        @test_throws ArgumentError PHI.OperatorOrbit(:bad, Int[]; operator=i -> PHI.nf(i))
        @test_throws DimensionMismatch PHI.OperatorOrbit(:bad, [1, 2]; operator=i -> PHI.nf(i), weights=[1.0])
        invalid = PHI.InteractionSpecification(PHI.OperatorOrbit(:bad, [(1, 2)]; operator=(i, j) -> PHI.c(i)' * PHI.c(j)))
        @test_throws ArgumentError PHI.generate_interaction_family(material, algebra, invalid; dataset=identity)
        duplicate = PHI.InteractionSpecification(PHI.OperatorOrbit(:a, [1]; operator=i -> PHI.nf(i)),
            PHI.OperatorOrbit(:b, [1]; operator=i -> PHI.nf(i)))
        @test_throws Exception PHI.generate_interaction_family(material, algebra, duplicate; dataset=identity)
        terms = PHI.InteractionSpecification(PHI.OperatorOrbit(:onsite, [1]; operator=i -> PHI.nf(i)))
        result = PHI.generate_interaction_family(material, algebra, terms; dataset=identity)
        @test_throws ArgumentError PHI.physical_parameters(result, Dict(:wrong => 1.0))
        @test_throws ArgumentError PHI.physical_parameters(result, PHI.PhysicalCouplings(Dict(:onsite => 1.0); unit=:meV))
        @test_throws ArgumentError PHI.generate_hamiltonian_family(material, algebra; interactions=terms, candidates=[PHI.nf(1)], dataset=identity)
    end
end
