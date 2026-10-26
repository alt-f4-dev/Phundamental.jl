using Test
using LinearAlgebra
using Phundamental

const MSD = Phundamental

function _msd_spin_fixture()
    lattice = MSD.BravaisLattice(Matrix{Float64}(I, 3, 3))
    sites = [MSD.BasisSite(:A, :A, [0.0, 0.0, 0.0]), MSD.BasisSite(:B, :B, [0.4, 0.0, 0.0])]
    material = MSD.Material("structural NLCE two-spin test", MSD.LatticeCrystalStructure(lattice, sites);
        species=Dict(:A => MSD.AtomicSpecies(:A; mass=1.0), :B => MSD.AtomicSpecies(:B; mass=1.0)),
        basis_properties=[MSD.SiteProperties(spin=1//2), MSD.SiteProperties(spin=1//2)])
    reference = MSD.Supercell(material.crystal, (1, 1, 1); periodic=true)
    op = MSD.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    dataset = MSD.CrystallographicDataset(spacegroup_number=1, operations=[op], crystallographic_orbits=[1, 2])
    result = MSD.generate_hamiltonian_family(material, reference, MSD.SpinAlgebra(2); dataset=dataset,
        candidates=MSD.AbstractOperatorExpr[MSD.Sz(1), MSD.Sz(2), MSD.Sz(1) * MSD.Sz(2)],
        basis_labels=[:h1, :h2, :exchange], max_body_order=2)
    family = MSD.structural_family(result)
    selection = MSD.structural_selection(family, [:h1, :h2, :exchange])
    representation = MSD.Representation(:spin, MSD.SpinHilbertSpace([1//2, 1//2]), MSD.SpinAlgebra(2), MSD.SpinProductBasis([1, 2]))
    model = MSD.instantiate_model(selection, representation, Dict(:h1 => 0.25, :h2 => -0.3, :exchange => 0.8))
    bond = MSD.Bond{Float64}(1, 2, [0.4, 0.0, 0.0], 0.4, 1)
    return (; material, reference, selection, model, bond)
end

@testset "Milestone D extended forward physics and certified limits" begin
    @testset "Induced structural spin clusters and finite-reference NLCE" begin
        f = _msd_spin_fixture()
        clusters = MSD.connected_clusters(f.reference, [f.bond]; max_order=2, allow_periodic_reference=true)
        @test length(clusters) == 3
        singles = [cluster for cluster in clusters if cluster.order == 1]
        @test length(singles) == 2
        for cluster in singles
            submodel = MSD.regenerate_induced_spin_cluster(f.model, f.reference, cluster.supercell; bond_terms=[f.bond => 0.8 * MSD.Sz(1) * MSD.Sz(2)])
            @test submodel.representation.algebra.nsites == 1
            reference_energy = cluster.parent_sites[1] == 1 ? -abs(0.25) / 2 : -abs(-0.3) / 2
            @test isapprox(MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), submodel)), reference_energy; atol=1e-10)
            @test length(submodel.observables) == 3
        end
        pair = only(cluster for cluster in clusters if cluster.order == 2)
        reconstructed = MSD.regenerate_induced_spin_cluster(f.model, f.reference, pair.supercell; bond_terms=[f.bond => 0.8 * MSD.Sz(1) * MSD.Sz(2)])
        @test isapprox(MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), reconstructed)),
                       MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), f.model)); atol=1e-10)
        context = MSD.StructuralNLCEForwardContext(f.selection, 2; bonds=[f.bond],
            bond_terms=[f.bond => 0.8 * MSD.Sz(1) * MSD.Sz(2)], allow_periodic_reference=true)
        request = MSD.ObservableRequest(:ground_energy, (_, sol) -> MSD.groundenergy(sol))
        @test MSD.solver_compatibility(MSD.NLCESolver(), f.model, request, context).compatible
        calculation = MSD.forward(MSD.ForwardProblem(f.model; solver=MSD.NLCESolver(), method_context=context, observable=request))
        @test calculation.solution isa MSD.NLCEResult
        @test isapprox(calculation.observable_result,
                       MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), f.model)) / 2; atol=1e-10)
        @test calculation.solution.metadata[:structural_selection_id] == MSD.structural_id(f.selection)
        @test_throws ArgumentError MSD.StructuralNLCEForwardContext(f.selection, 2)
    end

    @testset "F1: periodic-image aliasing is rejected or reconstructed by physical bond" begin
        f = _msd_spin_fixture()
        intrabond = f.bond
        wrapped = MSD.Bond{Float64}(1, 2, [-0.6, 0.0, 0.0], 0.6, 2)
        pair = only(c for c in MSD.connected_clusters(f.reference, [intrabond]; max_order=2, allow_periodic_reference=true) if c.order == 2)
        request = MSD.ObservableRequest(:ground_energy, (_, solution) -> MSD.groundenergy(solution))
        terms = [intrabond => 0.3 * MSD.Sz(1) * MSD.Sz(2), wrapped => 0.5 * MSD.Sz(1) * MSD.Sz(2)]
        ambiguous = MSD.StructuralNLCEForwardContext(f.selection, 2; bonds=[intrabond, wrapped], allow_periodic_reference=true)
        @test !MSD.solver_compatibility(MSD.NLCESolver(), f.model, request, ambiguous).compatible
        @test_throws ArgumentError MSD.regenerate_induced_spin_cluster(f.model, f.reference, pair.supercell)
        @test_throws ArgumentError MSD.regenerate_induced_spin_cluster(f.model, f.reference, pair.supercell;
            bond_terms=[intrabond => 0.3 * MSD.Sz(1) * MSD.Sz(2), wrapped => 0.4 * MSD.Sz(1) * MSD.Sz(2)])
        @test_throws ArgumentError MSD.regenerate_induced_spin_cluster(f.model, f.reference, pair.supercell;
            bond_terms=[intrabond => 0.8 * MSD.Sz(1) * MSD.Sz(2), intrabond => 0.1 * MSD.Sz(1) * MSD.Sz(2)])
        open_model = MSD.regenerate_induced_spin_cluster(f.model, f.reference, pair.supercell; bond_terms=terms)
        expected_h = 0.25 * MSD.Sz(1) - 0.3 * MSD.Sz(2) + 0.3 * MSD.Sz(1) * MSD.Sz(2)
        independent = MSD.ManyBodyModel(open_model.representation, expected_h)
        @test isapprox(MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), open_model)),
            MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), independent)); atol=1e-10)
        @test !isapprox(MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), open_model)),
            MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), f.model)); atol=1e-10)
        context = MSD.StructuralNLCEForwardContext(f.selection, 2; bonds=[intrabond, wrapped], bond_terms=terms,
            allow_periodic_reference=true)
        # No translated interior representative of the wrapped bond exists in a one-cell reference.
        @test !MSD.solver_compatibility(MSD.NLCESolver(), f.model, request, context).compatible
        @test_throws ArgumentError MSD.forward(MSD.ForwardProblem(f.model; solver=MSD.NLCESolver(),
            method_context=context, observable=request))
    end

    @testset "F1: enlarged reference recovers distinct open intrabond and interbond energies" begin
        lattice = MSD.BravaisLattice(Matrix{Float64}(I, 3, 3))
        sites = [MSD.BasisSite(:A, :A, [0.0, 0.0, 0.0]), MSD.BasisSite(:B, :B, [0.4, 0.0, 0.0])]
        material = MSD.Material("F1 periodic AB chain", MSD.LatticeCrystalStructure(lattice, sites);
            species=Dict(:A => MSD.AtomicSpecies(:A; mass=1.0), :B => MSD.AtomicSpecies(:B; mass=1.0)),
            basis_properties=[MSD.SiteProperties(spin=1//2), MSD.SiteProperties(spin=1//2)])
        J1, J2 = 0.3, 0.5
        for ncells in (3, 4)
            reference = MSD.Supercell(material.crystal, (ncells, 1, 1); periodic=true)
            links = MSD.neighbor_bonds(reference; cutoff=0.65)
            @test count(bond -> isapprox(bond.distance, 0.4; atol=1e-9), links) == ncells
            @test count(bond -> isapprox(bond.distance, 0.6; atol=1e-9), links) == ncells
            intra = reduce(+, (MSD.Sz(b.i) * MSD.Sz(b.j) for b in links if isapprox(b.distance, 0.4; atol=1e-9)))
            inter = reduce(+, (MSD.Sz(b.i) * MSD.Sz(b.j) for b in links if isapprox(b.distance, 0.6; atol=1e-9)))
            algebra = MSD.SpinAlgebra(2 * ncells)
            operation = MSD.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
            dataset = MSD.CrystallographicDataset(spacegroup_number=1, operations=[operation],
                crystallographic_orbits=collect(1:(2 * ncells)))
            generated = MSD.generate_hamiltonian_family(material, reference, algebra; dataset=dataset,
                candidates=MSD.AbstractOperatorExpr[intra, inter], basis_labels=[:intra, :inter], max_body_order=2)
            family = MSD.structural_family(generated)
            selection = MSD.structural_selection(family, [:intra, :inter])
            representation = MSD.Representation(:spin, MSD.SpinHilbertSpace(fill(1//2, 2 * ncells)), algebra,
                MSD.SpinProductBasis(collect(1:(2 * ncells))))
            exact_source = J1 * intra + J2 * inter
            coordinates = MSD.locate_in_family(family, exact_source)
            @test coordinates.in_span && coordinates.valid_parameters
            source = MSD.instantiate_model(selection, representation, coordinates.selected_coefficients)
            bond_terms = [bond => (isapprox(bond.distance, 0.4; atol=1e-9) ? J1 : J2) * MSD.Sz(bond.i) * MSD.Sz(bond.j) for bond in links]
            request = MSD.ObservableRequest(:ground_energy, (_, sol) -> MSD.groundenergy(sol))
            context = MSD.StructuralNLCEForwardContext(selection, 2; bonds=links, bond_terms=bond_terms, allow_periodic_reference=true)
            @test MSD.solver_compatibility(MSD.NLCESolver(), source, request, context).compatible
            result = MSD.forward(MSD.ForwardProblem(source; solver=MSD.NLCESolver(), method_context=context, observable=request))
            pair_clusters = [c for c in result.solution.clusters if c.order == 2]
            @test length(pair_clusters) == 2 * ncells - 1
            for cluster in pair_clusters
                @test length(cluster.bonds) == 1
                bond = only(cluster.bonds)
                coupling = isapprox(bond.distance, 0.4; atol=1e-9) ? J1 : J2
                independent = MSD.ManyBodyModel(MSD.Representation(:spin, MSD.SpinHilbertSpace(fill(1//2, 2)),
                    MSD.SpinAlgebra(2), MSD.SpinProductBasis([1, 2])), coupling * MSD.Sz(1) * MSD.Sz(2))
                expected = MSD.groundenergy(MSD.solve(MSD.ExactDiagonalization(), independent))
                @test isapprox(result.solution.raw_values[cluster.key], expected; atol=1e-10)
            end
            expected_bulk = -(J1 + J2) / 8
            @test isapprox(result.observable_result, expected_bulk; atol=1e-10)
            @test result.solution.metadata[:translation_embedding_corrected]
            @test result.solution.metadata[:infinite_lattice_claim]
        end
    end

    @testset "Bravais geometry and periodic embedding agree with exact tight-binding dispersion" begin
        lattice = MSD.BravaisLattice(reshape([2.0], 1, 1))
        crystal = MSD.LatticeCrystalStructure(lattice, [MSD.BasisSite(:H, :H, [0.0])])
        material = MSD.Material("one-orbital chain", crystal; species=Dict(:H => MSD.lookup_species(:H)))
        cluster = MSD.Supercell(crystal, (2,); periodic=false)
        model = MSD.build_model(material, cluster, MSD.HubbardModel(0.7, 2.0; shell=1))
        hopping = MSD.BravaisHopping((1,), -0.7)
        embedding = MSD.bravais_momentum_lattice(model, (16,), [hopping])
        @test length(embedding.energies) == 16
        @test maximum(abs.(embedding.energies .+ 1.4 .* cos.(first.(embedding.kpoints)))) < 1e-12
        @test embedding.qmaps[1] == collect(1:16)
        context = MSD.DMFTForwardContext(model, (16,), [hopping]; temperature=0.5, nfrequencies=8)
        @test context.lattice.energies ≈ embedding.energies
        @test_throws ArgumentError MSD.bravais_momentum_lattice(model, (16,), [MSD.BravaisHopping((1,), 0.7)])
        @test_throws ArgumentError MSD.bravais_momentum_lattice(material, (8,), [hopping, MSD.BravaisHopping((-1,), -0.7)])
    end

    @testset "Approximate rational continuation is explicit and diagnostically checked" begin
        axis = MSD.FermionicMatsubaraAxis(2.0, 24)
        values = ComplexF64[inv(z - 1.2) for z in axis.values]
        energies = collect(range(-2.0, 3.0; length=101))
        method = MSD.RationalContinuation(numerator_order=0, denominator_order=1, eta=0.15)
        output = MSD.continue_matsubara(axis, values, energies; method=method)
        reference = ComplexF64[inv(w + 0.15im - 1.2) for w in energies]
        @test maximum(abs.(output.retarded .- reference)) < 1e-8
        @test maximum(abs.(output.spectral_density .+ imag.(reference) ./ π)) < 1e-8
        @test output.metadata[:approximate] === true
        @test output.metadata[:causality_certified] === false
        @test_throws ArgumentError MSD.continue_matsubara(axis, values, energies;
            method=MSD.RationalContinuation(numerator_order=25, denominator_order=25))
    end

    @testset "CPT typed requests reject incompatible spin models" begin
        request = MSD.CPTSpectralFunctionRequest([0.0, π], [-2.0, -1.0, 0.0, 1.0, 2.0];
            eta=0.1, t_intercluster=-0.7, spin=:up)
        @test MSD.request_requirements(request).request_kind == :cluster_perturbation_spectral_function
        @test !MSD.solver_compatibility(MSD.OptimizedLanczos(), _msd_spin_fixture().model, request).compatible
        @test_throws ArgumentError MSD.CPTSpectralFunctionRequest([0.0], [-1.0, 1.0]; eta=0.0, t_intercluster=-0.7)
    end

    @testset "Typed multiphonon requests retain explicit physical and dimensional requirements" begin
        request = MSD.HarmonicMultiphononSliceRequest(MSD.NuclearNeutronProbe(); fixed_axis=:x, fixed_value=0.25,
            horizontal_axis=:y, horizontal_values=[0.0, 0.5], vertical_axis=:z, vertical_values=[0.0, 0.5],
            qmesh_shape=(2, 2, 2), energy_window=(0.0, 5.0), order=1)
        @test MSD.request_requirements(request).request_kind == :harmonic_multiphonon_slice
        @test request.qmesh_shape == (2, 2, 2)
        @test !MSD.solver_compatibility(MSD.HarmonicPhononSolver(), _msd_spin_fixture().model, request).compatible
        @test_throws ArgumentError MSD.HarmonicMultiphononSliceRequest(MSD.NuclearNeutronProbe();
            fixed_axis=:x, fixed_value=0.0, horizontal_axis=:x, horizontal_values=[0.0],
            vertical_axis=:z, vertical_values=[0.0], qmesh_shape=(2, 2, 2), energy_window=(0.0, 1.0))
        @test_throws ArgumentError MSD.HarmonicMultiphononSliceRequest(MSD.NuclearNeutronProbe();
            fixed_axis=:x, fixed_value=0.0, horizontal_axis=:y, horizontal_values=[0.0],
            vertical_axis=:z, vertical_values=[0.0], qmesh_shape=(2, 2, 2), energy_window=(0.0, 1.0), order=:all)
    end

    @testset "Static and thermodynamic typed requests use the established numerical routines" begin
        f = _msd_spin_fixture()
        static = MSD.StaticStructureFactorRequest(MSD.MomentumField([:spin_z_1, :spin_z_2], f.model.parameters[:positions]), [zeros(3)];
                                                  temperature=0.0)
        @test !MSD.solver_compatibility(MSD.OptimizedLanczos(), f.model, static).compatible
        direct_solution = MSD.solve(MSD.ExactDiagonalization(), f.model)
        reference = MSD.static_structure_factor(f.model, direct_solution, static.field, static.qgrid)
        actual = MSD.forward(MSD.ForwardProblem(f.model; observable=static))
        @test actual.observable_result.intensity ≈ reference.intensity
        curve_request = MSD.ThermodynamicCurveRequest([0.2, 0.5, 1.0])
        solver = MSD.ThermodynamicForwardSolver(MSD.ExactGibbs())
        @test MSD.solver_compatibility(solver, f.model, curve_request).compatible
        @test MSD.solver_compatibility(solver, f.model, curve_request).certified
        calculation = MSD.forward(MSD.ForwardProblem(f.model; solver=solver, observable=curve_request))
        low_level = MSD.thermal_curve(f.model, MSD.ExactGibbs(), curve_request.temperatures)
        @test calculation.observable_result.internal_energy ≈ low_level.internal_energy
        @test calculation.observable_result.heat_capacity ≈ low_level.heat_capacity
        report = MSD.planning_report(MSD.plan_forward(f.model, curve_request))
        @test report.selected_solver == :thermal_curve
        stochastic = MSD.ThermodynamicForwardSolver(MSD.ThermalLanczos(samples=4, krylov_dim=4))
        @test_throws MSD.ForwardPlanningError MSD.plan_forward(f.model, curve_request; routes=(stochastic,),
            policy=MSD.ForwardExecutionPolicy(include_default_exact_routes=false))
    end

    @testset "F3 prepared structural contexts preserve NLCE physics and policy" begin
        fixture = _msd_spin_fixture()
        request = MSD.ObservableRequest(:ground_energy, (_, solution) -> MSD.groundenergy(solution))
        context = MSD.StructuralNLCEForwardContext(fixture.selection, 2;
            bonds=[fixture.bond], bond_terms=[fixture.bond => 0.8 * MSD.Sz(1) * MSD.Sz(2)],
            allow_periodic_reference=true)
        representation = fixture.model.representation
        first_point = Dict(:h1 => 0.25, :h2 => -0.3, :exchange => 0.8)
        second_point = Dict(:h1 => -0.12, :h2 => -0.3, :exchange => 0.8)
        prepared = MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=context, observable=request)
        @test MSD.describe(prepared)["method_context"] == "StructuralNLCEForwardContext"
        direct = MSD.forward(MSD.ForwardProblem(fixture.model;
            solver=MSD.NLCESolver(), method_context=context, observable=request))
        repeated = MSD.forward_parameters(prepared, first_point)
        @test isapprox(repeated.observable_result, direct.observable_result; atol=1e-12)
        @test repeated.solution.metadata[:structural_selection_id] == MSD.structural_id(fixture.selection)
        batch = MSD.forward_batch(prepared, [first_point, second_point])
        @test length(batch) == 2
        @test isapprox(batch[1].observable_result, repeated.observable_result; atol=1e-12)
        @test batch[2].solution.metadata[:structural_selection_id] == MSD.structural_id(fixture.selection)
        @test_throws ArgumentError MSD.forward_parameters(prepared,
            Dict(:h1 => 0.25, :h2 => -0.3, :exchange => 0.9))
        @test_throws ArgumentError MSD.prepare_forward(fixture.selection, representation; method_context=context)
        @test_throws ArgumentError MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=123, observable=request)
        policy = MSD.ForwardExecutionPolicy(allowed_approximations=(:finite_order_nlce,),
            require_certified=false, include_default_exact_routes=false)
        planned = MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=context, observable=request, policy=policy)
        routed = MSD.forward_parameters(planned, first_point)
        @test isapprox(routed.observable_result, direct.observable_result; atol=1e-12)
        @test routed.solution.metadata[:forward_planned]
        @test routed.solution.metadata[:planned_solver] === :nlce
        # F7: scalar NLCE estimates are comparable even without complete eigenenergy arrays.
        scalar_equivalence = MSD.validate_forward_equivalence(repeated, routed;
            compare=(:intrinsic, :instrument))
        @test scalar_equivalence.valid
        @test !MSD.validate_forward_equivalence(repeated, routed; compare=(:spectrum,)).valid
        alternative = MSD.prepare_forward(fixture.selection, representation; observable=request, policy=policy,
            routes=(MSD.ForwardRoute(MSD.NLCESolver(); method_context=context),))
        @test isapprox(MSD.forward_parameters(alternative, first_point).observable_result, direct.observable_result; atol=1e-12)
        default_candidates = MSD.ForwardExecutionPolicy(allowed_approximations=(:finite_order_nlce,), require_certified=false)
        @test_throws ArgumentError MSD.forward_parameters(MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=context, observable=request, policy=default_candidates), first_point)
        @test_throws MSD.ForwardPlanningError MSD.forward_parameters(MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=context, observable=request,
            policy=MSD.ForwardExecutionPolicy(require_certified=false, include_default_exact_routes=false)), first_point)
        @test_throws ArgumentError MSD.prepare_forward(fixture.selection, representation; solver=MSD.NLCESolver(),
            method_context=context, observable=request, policy=policy, routes=(MSD.ForwardRoute(MSD.NLCESolver()),))
        @test_throws ArgumentError MSD.forward_parameters(MSD.prepare_forward(fixture.selection, representation;
            solver=MSD.NLCESolver(), method_context=MSD.StructuralNLCEForwardContext(fixture.selection, 2;
                bonds=[fixture.bond], allow_periodic_reference=true), observable=request), first_point)
    end

    @testset "F3 generated single-band Hubbard embedding certifies the canonical Hamiltonian" begin
        lattice = MSD.BravaisLattice(Diagonal([1.0, 7.0, 8.0]) |> Matrix)
        crystal = MSD.LatticeCrystalStructure(lattice, [MSD.BasisSite(:A, :H, [0.0, 0.0, 0.0])])
        material = MSD.Material("F3 single-orbital nearest-shell Hubbard reference", crystal;
            species=Dict(:H => MSD.lookup_species(:H)))
        reference = MSD.Supercell(crystal, (4, 1, 1); periodic=true)
        algebra = MSD.FermionAlgebra(2 * MSD.nsites(reference))
        operation = MSD.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        dataset = MSD.CrystallographicDataset(spacegroup_number=1, operations=[operation],
            crystallographic_orbits=collect(1:MSD.nsites(reference)))
        template(t, U, mu=0.0) = MSD.build_model(material, reference,
            MSD.HubbardModel(t, U; shell=1, chemical_potential=mu)).hamiltonian
        generators = MSD.AbstractOperatorExpr[template(1.0, 0.0), template(0.0, 1.0), template(0.0, 0.0, 1.0)]
        generation = MSD.generate_hamiltonian_family(material, reference, algebra;
            dataset=dataset, symmetry_action=MSD.SpinfulFermionAction(),
            candidates=generators, basis_labels=[:hop, :interaction, :chemical], max_body_order=4, max_range=1.01)
        family = MSD.structural_family(generation)
        physical = template(0.7, 1.6, 0.2)
        location = MSD.locate_in_family(family, physical)
        @test location.in_span && location.valid_parameters
        selection = MSD.structural_selection(family, [:hop, :interaction, :chemical])
        representation = MSD.Representation(:f3_spinful, MSD.FermionFockSpace(algebra.nmodes), algebra,
            MSD.FermionOccupationBasis(collect(1:algebra.nmodes)))
        source = MSD.instantiate_model(selection, representation, location.selected_coefficients)
        @test MSD.geometry_context(source).component_layout.norbitals == 1
        certified = MSD._single_band_hubbard_forward_parameters(source)
        @test certified.certified_from === :structural_generation
        @test isapprox(certified.hopping, 0.7; atol=1e-9)
        @test isapprox(certified.interaction, 1.6; atol=1e-9)
        @test isapprox(certified.chemical_potential, 0.2; atol=1e-9)
        hopping = MSD.BravaisHopping((1, 0, 0), -0.7)
        mesh = MSD.bravais_momentum_lattice(source, (8, 1, 1), [hopping])
        @test maximum(abs.(mesh.energies .+ 1.4 .* cos.(first.(mesh.kpoints)))) < 1e-9
        @test mesh.qmaps[1] == collect(1:8)
        embedded = MSD.DMFTForwardContext(source, (8, 1, 1), [hopping]; temperature=0.5, nfrequencies=8)
        @test embedded.lattice.energies ≈ mesh.energies
        request = MSD.LocalMatsubaraGreenFunctionRequest(metadata=Dict(:temperature => 0.5))
        @test MSD.solver_compatibility(MSD.DMFTSolver(), source, request, embedded).compatible
        @test_throws ArgumentError MSD.bravais_momentum_lattice(source, (8, 1, 1), [MSD.BravaisHopping((0, 1, 0), -0.7)])
        @test_throws ArgumentError MSD.bravais_momentum_lattice(source, (8, 1, 1), [MSD.BravaisHopping((1, 0, 0), -0.6)])
        @test_throws ArgumentError MSD.bravais_momentum_lattice(source, (8, 1, 1), [hopping, MSD.BravaisHopping((0, 1, 0), -0.7)])
        @test_throws ArgumentError MSD.bravais_momentum_lattice(source, (8, 1, 1), [hopping]; onsite_energy=0.1)
    end
end
