using Test
using LinearAlgebra
using Phundamental
#include(joinpath(@__DIR__, "..", "src/Phundamental.jl"))

const F = Phundamental
const C = F.PhundamentalCore

@testset "Generalized Phundamental API" begin
    @testset "Generated Hamiltonian space and typed parameters" begin
        spin_space = C.SpinHilbertSpace([1//2])
        spin_algebra = C.SpinAlgebra(1)
        spin_representation = C.Representation(:spin_z, spin_space, spin_algebra, C.SpinProductBasis([1]))
        specification = F.GenerativeSpecification(:C1, (:site_1,), (spin_space,), spin_algebra, identity; max_body_order=1)
        basis = F.HamiltonianBasis(C.AbstractOperatorExpr[C.Sz(1), C.Sx(1)]; labels=[:hz, :hx])
        model_space = F.HamiltonianSpace(specification, basis)
        parameter_space = F.ParameterSpace(model_space; values=[1.0, 0.25], bounds=[(-2.0, 2.0), (-1.0, 1.0)])
        point = F.ParameterPoint(parameter_space)
        model = F.instantiate_model(model_space, spin_representation, point)
        @test F.model_space_dimension(model_space) == 2
        @test F.parameter_names(model) == [:hz, :hx]
        @test F.parameter_vector(model) == [1.0, 0.25]
        moved = F.with_parameters(model, F.parameter_point(parameter_space; hz=0.5, hx=-0.125))
        @test F.parameter_vector(moved) == [0.5, -0.125]
    end

    @testset "Factor-qualified composite fermions" begin
        state_space = C.CompositeStateSpace(C.FermionFockSpace(1), C.FermionFockSpace(1))
        algebra = C.CompositeAlgebra(C.FermionAlgebra(1), C.FermionAlgebra(1))
        basis = C.CompositeBasis(C.FermionOccupationBasis([1]), C.FermionOccupationBasis([1]))
        representation = C.Representation(:two_fermion_factors, state_space, algebra, basis)
        hamiltonian = C.atfactor(1, C.nf(1)) + 2 * C.atfactor(2, C.nf(1))
        model = C.ManyBodyModel(representation, hamiltonian)
        matrix_result = F.Solvers.hamiltonian_matrix(model; sparse=false)
        @test sort(real.(diag(matrix_result.matrix))) == [0.0, 1.0, 2.0, 3.0]
        cross_car = F.Algebra.anticommutator(algebra, C.atfactor(1, C.c(1)), C.atfactor(2, C.c(1)'))
        @test cross_car isa C.ScaledOperator
        @test iszero(cross_car.coefficient)
    end

    @testset "Generic symmetry-sector descriptors" begin
        sector = F.QuantumNumberSector(:nup => 2, :ndown => 1; metadata=Dict(:symmetry => :particle_number))
        @test F.has_quantum_number(sector, :nup)
        @test F.sector_value(sector, :ndown) == 1
        @test length(F.quantum_numbers(sector)) == 2
        decomposition = F.SectorDecomposition(symmetry=sector)
        @test decomposition.symmetry === sector
    end

    @testset "Approximation objects" begin
        spin_space = C.SpinHilbertSpace([1//2])
        spin_algebra = C.SpinAlgebra(1)
        representation = C.Representation(:spin_projection_test, spin_space, spin_algebra, C.SpinProductBasis([1]))
        specification = F.GenerativeSpecification(:C1, (:site_1,), (spin_space,), spin_algebra, identity)
        basis = F.HamiltonianBasis(C.AbstractOperatorExpr[C.Sz(1)]; labels=[:hz])
        target_space = F.HamiltonianSpace(specification, basis)
        model = F.instantiate_model(target_space, representation, [0.7])
        approximation = F.ModelSpaceProjection(target_space, (H, space) -> (H, 0 * C.IdentityOperator()))
        result = F.apply_approximation(approximation, model)
        @test result.validation[:applicable]
        @test result.validation[:admissible]
    end

    @testset "Typed forward problem and result capabilities" begin
        state_space = C.SpinHilbertSpace([1//2])
        algebra = C.SpinAlgebra(1)
        representation = C.Representation(:forward_spin, state_space, algebra, C.SpinProductBasis([1]))
        model = C.ManyBodyModel(representation, C.Sz(1))
        request = F.ObservableRequest(:ground_energy, (m, solution) -> F.Solvers.groundenergy(solution))
        problem = F.ForwardProblem(model; solver=F.ExactDiagonalization(), observable=request)
        result = F.forward(problem)
        @test result.observable_result ≈ -0.5
        @test F.supports(result.solution, :energies)
        @test F.result_energies(result.solution) == result.solution.energies
        @test :model in F.result_capabilities(result.solution)
    end


    @testset "Top-level exports and canonical creation syntax" begin
        @test F.SpinHilbertSpace === C.SpinHilbertSpace
        @test F.computational_basis === F.Solvers.computational_basis
        @test F.hamiltonian_matrix === F.Solvers.hamiltonian_matrix
        @test (F.c(1)' isa F.FermionCreate)
        @test (F.b(1)' isa F.BosonCreate)
        @test isdefined(C, :cd)
        @test isdefined(C, :bd)
    end

    @testset "Algebra-aware many-body rank" begin
        fermions = F.FermionAlgebra(4)
        one_body = F.c(1)' * F.c(2)
        two_body = F.c(1)' * F.c(2)' * F.c(3) * F.c(4)
        @test F.operator_degree(one_body) == 2
        @test F.body_order(fermions, one_body) == 1
        @test F.operator_degree(two_body) == 4
        @test F.body_order(fermions, two_body) == 2
        bosons = F.BosonAlgebra(2)
        @test F.body_order(bosons, F.b(1)' * F.b(2)) == 1
        @test F.body_order(bosons, F.b(1)' * F.b(1)' * F.b(1) * F.b(1)) == 2
        spins = F.SpinAlgebra(2)
        @test F.body_order(spins, F.Sz(1) * F.Sz(1)) == 1
        @test F.body_order(spins, F.Sz(1) * F.Sz(2)) == 2
    end

    @testset "Transformation numerical closure" begin
        boson_representation = F.Representation(:boson_numeric_test, F.BosonFockSpace(1), F.BosonAlgebra(1), F.BosonOccupationBasis([1]))
        boson_model = F.ManyBodyModel(boson_representation, F.nb(1) + 0.1 * (F.b(1) + F.b(1)'))
        bare_capabilities = F.numerical_capabilities(boson_model)
        @test bare_capabilities.requires_truncation
        @test !bare_capabilities.numerically_realizable

        truncated = F.with_truncation(boson_model, F.NumericalTruncation(4))
        truncated_capabilities = F.numerical_capabilities(truncated)
        @test truncated_capabilities.numerically_realizable
        @test truncated_capabilities.finite_basis
        @test F.basis_dimension(F.computational_basis(truncated)) == 5
        @test size(F.hamiltonian_matrix(truncated; sparse=false).matrix) == (5, 5)

        displacement = F.BosonicDisplacement([0.2])
        transformed = F.transform(displacement, truncated).model
        transported_capabilities = F.numerical_capabilities(transformed)
        @test transported_capabilities.transported_truncation
        @test transported_capabilities.requires_truncation
        @test !transported_capabilities.numerically_realizable
        @test_throws ArgumentError F.computational_basis(transformed)

        reselected = F.with_truncation(transformed, F.NumericalTruncation(4))
        @test F.numerical_capabilities(reselected).numerically_realizable
        @test size(F.hamiltonian_matrix(reselected; sparse=false).matrix) == (5, 5)

        displacement_capabilities = F.transformation_capabilities(displacement, boson_model)
        @test displacement_capabilities.applicable
        @test displacement_capabilities.numerical.requires_truncation

        spin_representation = F.Representation(:spin_numeric_test, F.SpinHilbertSpace([1//2]), F.SpinAlgebra(1), F.SpinProductBasis([1]))
        spin_model = F.ManyBodyModel(spin_representation, 0.31 * F.Sz(1) + 0.2 * F.Sx(1))
        dyson_model = F.transform(F.DysonMaleev(), spin_model).model
        dyson_capabilities = F.numerical_capabilities(dyson_model)
        @test dyson_capabilities.biorthogonal
        @test :exact_diagonalization in dyson_capabilities.compatible_solvers
        @test !(:lanczos in dyson_capabilities.compatible_solvers)
        @test_throws ArgumentError F.solve(F.LanczosSolver(nev=1), dyson_model)
        dyson_result = F.solve(F.ExactDiagonalization(), dyson_model)
        @test F.isbiorthogonal(dyson_result)
        @test length(dyson_result.energies) == 2

        spin_one_representation = F.Representation(:spin_constraint_transport_test, F.SpinHilbertSpace([1]), F.SpinAlgebra(1), F.SpinProductBasis([1]))
        spin_one_model = F.ManyBodyModel(spin_one_representation, F.Sz(1))
        hp_model = F.transform(F.HolsteinPrimakoff(), spin_one_model).model
        fourier = F.FourierTransformation(reshape([1.0], 1, 1); statistics=:boson)
        chained = F.transform(fourier, hp_model).model
        chained_capabilities = F.numerical_capabilities(chained)
        @test chained_capabilities.unresolved_constraints
        @test !chained_capabilities.numerically_realizable
        @test_throws ArgumentError F.computational_basis(chained)
    end

    @testset "Reciprocal-space decomposition" begin
        B = 2π .* Matrix{Float64}(I, 3, 3)
        Q = F.ReciprocalWaveVector([1.2, -0.7, 0.49])
        decomposition = F.decompose_reciprocal_vector(B, Q)
        @test decomposition.G_indices == [1, -1, 0]
        @test decomposition.q_reciprocal.coordinates ≈ [0.2, 0.3, 0.49]
        @test decomposition.reconstruction_error <= 1e-12

        Bskew = [1.0 0.45 0.10; 0.0 1.15 0.25; 0.0 0.0 0.90]
        Qskew = F.ReciprocalWaveVector([1.31, -0.62, 0.74])
        skew = F.decompose_reciprocal_vector(Bskew, Qskew)
        qnorm = norm(skew.q_cartesian.coordinates)
        for offset in Iterators.product((-1:1 for _ in 1:3)...)
            candidate_G = skew.G_indices .+ collect(offset)
            candidate_q = Bskew * (Qskew.coordinates .- candidate_G)
            @test qnorm <= norm(candidate_q) + 1e-12
        end
    end


    @testset "Streamlined harmonic phonon workflow" begin
        @test F.LatticeCrystalStructure === F.Models.CrystalStructure
        @test F.CrystalStructure === F.Crystallography.CrystalStructure

        lattice = F.BravaisLattice(reshape([1.0], 1, 1))
        crystal = F.LatticeCrystalStructure(lattice, [F.BasisSite(:A, :A, [0.0]), F.BasisSite(:B, :B, [0.5])])
        material = F.Material("streamlined phonon validation", crystal; species=Dict(:A => F.AtomicSpecies(:A; mass=1.0, nuclear_scattering_length=1.0), 
                                                                                     :B => F.AtomicSpecies(:B; mass=2.0, nuclear_scattering_length=1.5)))
        interactions = [F.HarmonicBondInteraction(1, 2, [0]; longitudinal=4.0), F.HarmonicBondInteraction(1, 2, [-1]; longitudinal=4.0)]

        model = F.harmonic_model(material, interactions)
        ifcs = F.force_constants(crystal, interactions)
        explicit_cluster = F.Supercell(crystal, [1]; periodic=true)
        explicit_model = F.build_model(material, explicit_cluster, F.HarmonicPhononModel(ifcs))
        q = F.ReciprocalWaveVector([0.25])
        @test F.dynamical_matrix(model, q) ≈ F.dynamical_matrix(explicit_model, q)

        convenience_mode = F.solve(model, 0.25)
        explicit_mode = F.solve(F.HarmonicPhononSolver(), model, q)
        @test convenience_mode.frequencies ≈ explicit_mode.frequencies

        qvalues = [0.0, 0.25, 0.5]
        convenience_bands = F.phonon_dispersion(model, qvalues)
        explicit_bands = F.phonon_dispersion(F.HarmonicPhononSolver(), model, [F.ReciprocalWaveVector([value]) for value in qvalues])
        @test convenience_bands.frequencies ≈ explicit_bands.frequencies

        zero_point = F.phonon_zero_point_energy(model; qmesh=(8,))
        @test zero_point ≈ F.groundenergy(model; qmesh=(8,))
        @test zero_point >= 0

        axis = collect(range(0.0, 5.0; length=101))
        broadening = F.GaussianBroadening(0.05)
        full_cut = F.one_phonon_neutron_intensity(model, [0.25, 1.25], axis; broadening=broadening)
        first_point = F.one_phonon_neutron_intensity(model, 0.25, axis; broadening=broadening)
        second_point = F.one_phonon_neutron_intensity(model, 1.25, axis; broadening=broadening)
        @test size(full_cut.intensity) == (2, length(axis))
        @test full_cut.intensity[1, :] ≈ first_point.intensity[1, :]
        @test full_cut.intensity[2, :] ≈ second_point.intensity[1, :]
        @test full_cut.metadata[:vectorized_full_Q]
    end

    @testset "Representation graph" begin
        source = C.Representation(:particles, C.FermionFockSpace(1), C.FermionAlgebra(1), C.FermionOccupationBasis([1]))
        target = C.Representation(:holes, C.FermionFockSpace(1), C.FermionAlgebra(1), C.FermionOccupationBasis([1]))
        graph = F.RepresentationGraph()
        F.add_representation!(graph, source)
        F.add_representation!(graph, target)
        F.add_transformation!(graph, :particles, :holes, F.ParticleHoleTransformation())
        path = F.transformation_path(graph, :particles, :holes; exact_only=true)
        @test length(path) == 1
        @test Set(F.representation_class(graph, :particles)) == Set([:particles, :holes])
    end
end

@testset "Crystallographic invariant-generation API" begin
    structure = F.CrystalStructure(Matrix{Float64}(I, 3, 3), [[0.0, 0.0, 0.0]], [1])
    identity_operation = F.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    dataset = F.CrystallographicDataset(spacegroup_number=1, international_symbol="P1", operations=[identity_operation], wyckoffs=["a"], equivalent_atoms=[1], crystallographic_orbits=[1])
    spin_space = F.SpinHilbertSpace([1//2])
    spin_algebra = F.SpinAlgebra(1)
    specification = F.GenerativeSpecification(dataset, (1,), (spin_space,), spin_algebra, F.standard_symmetry_action; max_body_order=1, metadata=Dict(:crystal_structure => structure))
    candidates = F.AbstractOperatorExpr[F.Sx(1), F.Sy(1), F.Sz(1)]
    basis = F.generate_invariant_basis(specification, structure, candidates)
    discovered = F.discover_symmetry(structure)
    @test F.model_space_dimension(basis) == 3
    @test F.site_permutation(structure, identity_operation) == [1]
    @test length(F.enumerate_operator_candidates(spin_algebra; max_body_order=1)) == 3
    @test discovered.spacegroup_number == 221
    @test length(discovered.operations) == 48

    @testset "site_tolerance propagation" begin
        near_identity = F.SpaceGroupOperation(Matrix{Int}(I, 3, 3), [5e-6, 0.0, 0.0])
        near_dataset = F.CrystallographicDataset(spacegroup_number=1, international_symbol="P1", operations=[near_identity])
        near_specification = F.GenerativeSpecification(near_dataset, (1,), (spin_space,), spin_algebra, F.standard_symmetry_action; max_body_order=1)
        loose = F.CrystallographyOptions(site_tolerance=1e-5)
        tight = F.CrystallographyOptions(site_tolerance=1e-7)
        @test F.site_permutation(structure, near_identity; atol=loose.site_tolerance) == [1]
        @test (F.reynolds_project(F.Sz(1), near_specification, structure, near_dataset; options=loose) isa F.SpinZ)
        stored_specification = F.GenerativeSpecification(near_dataset, (1,), (spin_space,), spin_algebra, F.standard_symmetry_action; max_body_order=1, metadata=Dict(:crystallography_options => loose))
        stored_generator = F.CrystallographicGenerator(structure, F.AbstractOperatorExpr[F.Sz(1)])
        @test F.model_space_dimension(F.generate_hamiltonian_space(stored_specification, stored_generator)) == 1
        @test_throws ArgumentError F.reynolds_project(F.Sz(1), near_specification, structure, near_dataset; options=tight)
    end

    @testset "manual max_body_order filtering" begin
        two_site_structure = F.CrystalStructure(Matrix{Float64}(I, 3, 3), [[0.0, 0.0, 0.0], [0.271, 0.183, 0.347]], [1, 2])
        two_site_dataset = F.CrystallographicDataset(spacegroup_number=1, international_symbol="P1", operations=[identity_operation])
        two_site_algebra = F.SpinAlgebra(2)
        two_site_spaces = (F.SpinHilbertSpace([1//2]), F.SpinHilbertSpace([1//2]))
        one_body_specification = F.GenerativeSpecification(two_site_dataset, (1, 2), two_site_spaces, two_site_algebra, F.standard_symmetry_action; max_body_order=1)
        manual_candidates = F.AbstractOperatorExpr[F.Sz(1) * F.Sz(1), F.Sz(1) * F.Sz(2)]
        filtered_basis = F.generate_invariant_basis(one_body_specification, two_site_structure, manual_candidates)
        @test F.model_space_dimension(filtered_basis) == 1
        @test filtered_basis.metadata[:rejected_body_order] == 1
        @test F.body_order(two_site_algebra, only(filtered_basis.operators)) == 1
    end

    @testset "automatic particle-rank enumeration" begin
        fermion_algebra = F.FermionAlgebra(2)
        fermion_candidates = F.enumerate_operator_candidates(fermion_algebra; max_body_order=1)
        @test all(F.body_order(fermion_algebra, candidate) <= 1 for candidate in fermion_candidates)
        @test maximum(F.operator_degree(candidate) for candidate in fermion_candidates) == 2
    end

    @testset "Dipolar interaction constructors" begin
        default_interaction = F.DipolarInteraction()
        integer_interaction = F.DipolarInteraction(2)
        float32_interaction = F.DipolarInteraction(2.0f0)
        typed_interaction = F.DipolarInteraction{Float32}(2)

        @test default_interaction isa F.DipolarInteraction{Float64}
        @test default_interaction.strength === 1.0
        @test integer_interaction isa F.DipolarInteraction{Float64}
        @test integer_interaction.strength === 2.0
        @test float32_interaction isa F.DipolarInteraction{Float32}
        @test float32_interaction.strength === 2.0f0
        @test typed_interaction isa F.DipolarInteraction{Float32}
        @test typed_interaction.strength === 2.0f0
    end

end
