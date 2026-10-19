using Test
using LinearAlgebra
using Phundamental

const MSB = Phundamental

function _msb_zero(operator, algebra; atol=1e-9)
    terms = MSB.PhundamentalCore._numeric_linear_terms(algebra, MSB.Algebra.canonicalize(algebra, operator))
    isnothing(terms) && return false
    return maximum((abs(value) for value in values(first(terms))); init=0.0) <= atol
end

function _msb_crystal(; sites=1)
    lattice = Matrix{Float64}(I, 3, 3)
    positions = sites == 1 ? reshape([0.0, 0.0, 0.0], 3, 1) : [0.25 0.75; 0.0 0.0; 0.0 0.0]
    return MSB.Crystallography.CrystalStructure(lattice, positions, fill(1, sites))
end

function _msb_dataset(operations, sites)
    return MSB.CrystallographicDataset(spacegroup_number=1, operations=operations, crystallographic_orbits=fill(1, sites))
end

@testset "Milestone B physically correct internal symmetry and specification transport" begin
    @testset "Magnetic inversion, Moriya mirror constraints, and time reversal" begin
        crystal = _msb_crystal(sites=2)
        identity = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        inversion = MSB.SpaceGroupOperation(-Matrix{Int}(I, 3, 3), [1.0, 0.0, 0.0])
        mirror = MSB.SpaceGroupOperation(Diagonal([1, -1, 1]) |> Matrix, zeros(3))
        algebra = MSB.SpinAlgebra(2)
        local_spaces = (MSB.SpinHilbertSpace([1//2]), MSB.SpinHilbertSpace([1//2]))
        action = MSB.MagneticSymmetryAction()
        inversion_dataset = _msb_dataset([identity, inversion], 2)
        mirror_dataset = _msb_dataset([identity, mirror], 2)
        @test MSB.validate_internal_symmetry(action, crystal, inversion_dataset, algebra)
        @test MSB.validate_internal_symmetry(action, crystal, mirror_dataset, algebra)
        spec_inv = MSB.GenerativeSpecification(inversion_dataset, (1, 1), local_spaces, algebra, action)
        spec_mirror = MSB.GenerativeSpecification(mirror_dataset, (1, 1), local_spaces, algebra, action)
        D_z = MSB.Sx(1) * MSB.Sy(2) - MSB.Sy(1) * MSB.Sx(2)
        D_y = MSB.Sz(1) * MSB.Sx(2) - MSB.Sx(1) * MSB.Sz(2)
        heisenberg = MSB.Sx(1) * MSB.Sx(2) + MSB.Sy(1) * MSB.Sy(2) + MSB.Sz(1) * MSB.Sz(2)
        @test _msb_zero(MSB.Crystallography.reynolds_project(D_z, spec_inv, crystal, inversion_dataset), algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(D_y, spec_inv, crystal, inversion_dataset), algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(D_z, spec_mirror, crystal, mirror_dataset), algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(D_y, spec_mirror, crystal, mirror_dataset) - D_y, algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(heisenberg, spec_inv, crystal, inversion_dataset) - heisenberg, algebra)
        antiunitary = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3); time_reversal=true)
        tr_dataset = _msb_dataset([identity, antiunitary], 2)
        spec_tr = MSB.GenerativeSpecification(tr_dataset, (1, 1), local_spaces, algebra, action)
        @test _msb_zero(MSB.Crystallography.reynolds_project(MSB.Sz(1), spec_tr, crystal, tr_dataset), algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(D_y, spec_tr, crystal, tr_dataset) - D_y, algebra)
        @test _msb_zero(MSB.Crystallography.reynolds_project(im * D_y, spec_tr, crystal, tr_dataset), algebra)
    end

    @testset "Spinor, orbital, and Cartesian phonon component actions" begin
        crystal = _msb_crystal()
        identity = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        rotation = MSB.SpaceGroupOperation([0 -1 0; 1 0 0; 0 0 1], zeros(3))
        dataset = _msb_dataset([identity, rotation], 1)
        spinor = MSB.SpinfulFermionAction()
        algebra = MSB.FermionAlgebra(2)
        @test MSB.validate_internal_symmetry(spinor, crystal, dataset, algebra)
        @test _msb_zero(spinor(MSB.nf(1) + MSB.nf(2), rotation, crystal, algebra) - (MSB.nf(1) + MSB.nf(2)), algebra)
        spin_x = (MSB.c(1)' * MSB.c(2) + MSB.c(2)' * MSB.c(1)) / 2
        spin_y = (MSB.c(1)' * MSB.c(2) - MSB.c(2)' * MSB.c(1)) / (2im)
        @test _msb_zero(spinor(spin_x, rotation, crystal, algebra) - spin_y, algebra)
        @test _msb_zero(spinor(spin_y, rotation, crystal, algebra) + spin_x, algebra)
        @test _msb_zero(spinor(MSB.nf(1) - MSB.nf(2), rotation, crystal, algebra) - (MSB.nf(1) - MSB.nf(2)), algebra)
        half_turn = MSB.SpaceGroupOperation(Diagonal([1, -1, -1]) |> Matrix, zeros(3))
        @test _msb_zero(spinor(MSB.nf(1), half_turn, crystal, algebra) - MSB.nf(2), algebra)
        antiunitary_spinor = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3); time_reversal=true)
        @test _msb_zero(spinor(MSB.nf(1) - MSB.nf(2), antiunitary_spinor, crystal, algebra) +
            MSB.nf(1) - MSB.nf(2), algebra)
        transformed_up = spinor(MSB.c(1), rotation, crystal, algebra)
        transformed_down = spinor(MSB.c(2), rotation, crystal, algebra)
        @test _msb_zero(spinor(MSB.c(1)', rotation, crystal, algebra) - adjoint(transformed_up), algebra)
        fermion_spec = MSB.GenerativeSpecification(dataset, (1,), (MSB.FermionFockSpace(2),), algebra, spinor)
        @test_throws ArgumentError MSB.Crystallography.generate_invariant_basis(
            fermion_spec, crystal, MSB.AbstractOperatorExpr[MSB.c(1)])
        @test _msb_zero(MSB.Algebra.anticommutator(algebra, transformed_up, adjoint(transformed_up)) - MSB.IdentityOperator(), algebra)
        @test _msb_zero(MSB.Algebra.anticommutator(algebra, transformed_up, adjoint(transformed_down)), algebra)
        @test_throws DimensionMismatch spinor(MSB.nf(1), rotation, crystal, MSB.FermionAlgebra(4))
        orbital_rotation = (op, _) -> Float64.(op.rotation[1:2, 1:2])
        orbitals = MSB.SpinfulFermionAction(norbitals=2, orbital_representation=orbital_rotation)
        @test MSB.validate_internal_symmetry(orbitals, crystal, dataset, MSB.FermionAlgebra(4))
        @test _msb_zero(orbitals(MSB.nf(1), rotation, crystal, MSB.FermionAlgebra(4)) - MSB.nf(3), MSB.FermionAlgebra(4))
        @test_throws ArgumentError MSB.validate_internal_symmetry(
            MSB.SpinfulFermionAction(norbitals=2, orbital_representation=[2.0 0.0; 0.0 1.0]), crystal, dataset, MSB.FermionAlgebra(4))
        antiunitary = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3); time_reversal=true)
        @test_throws ArgumentError orbitals(MSB.nf(1), antiunitary, crystal, MSB.FermionAlgebra(4))
        vector_action = MSB.VectorDisplacementAction()
        coords = MSB.CoordinateMomentumAlgebra(3)
        @test MSB.validate_internal_symmetry(vector_action, crystal, dataset, coords)
        @test _msb_zero(vector_action(MSB.xop(1), rotation, crystal, coords) - MSB.xop(2), coords)
        @test _msb_zero(vector_action(MSB.xop(2), rotation, crystal, coords) + MSB.xop(1), coords)
        @test _msb_zero(vector_action(MSB.pop(2), rotation, crystal, coords) + MSB.pop(1), coords)
        @test _msb_zero(vector_action(MSB.pop(1), antiunitary, crystal, coords) + MSB.pop(1), coords)
        @test _msb_zero(vector_action(MSB.xop(1), antiunitary, crystal, coords) - MSB.xop(1), coords)
    end

    @testset "Material-aware internal layouts and acoustic equality constraints" begin
        lattice = MSB.BravaisLattice(Matrix{Float64}(I, 3, 3))
        sites = [MSB.BasisSite(:A, :A, [0.25, 0.0, 0.0]), MSB.BasisSite(:B, :B, [0.75, 0.0, 0.0])]
        material = MSB.Material("two-site harmonic force-constant reference", MSB.LatticeCrystalStructure(lattice, sites);
            species=Dict(:A => MSB.AtomicSpecies(:A; mass=1.0), :B => MSB.AtomicSpecies(:B; mass=1.0)),
            basis_properties=[MSB.SiteProperties(spin=1//2) for _ in sites])
        identity = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        dataset = _msb_dataset([identity], 2)
        fermion_spec = MSB.material_generative_specification(material, MSB.FermionAlgebra(4);
            dataset=dataset, symmetry_action=MSB.SpinfulFermionAction())
        @test all(space -> space isa MSB.FermionFockSpace && space.nmodes == 2, fermion_spec.local_spaces)
        @test_throws DimensionMismatch MSB.material_generative_specification(material, MSB.FermionAlgebra(4);
            dataset=dataset, symmetry_action=MSB.SpinfulFermionAction(),
            local_spaces=(MSB.FermionFockSpace(1), MSB.FermionFockSpace(1)))
        phonon_spec = MSB.material_generative_specification(material, MSB.CoordinateMomentumAlgebra(6);
            dataset=dataset, symmetry_action=MSB.VectorDisplacementAction())
        @test all(space -> space isa MSB.CoordinateHilbertSpace && space.ndof == 3, phonon_spec.local_spaces)
        @test_throws DimensionMismatch MSB.material_generative_specification(material, MSB.CoordinateMomentumAlgebra(6);
            dataset=dataset, symmetry_action=MSB.VectorDisplacementAction(),
            local_spaces=(MSB.CoordinateHilbertSpace(2), MSB.CoordinateHilbertSpace(2)))
        @test_throws DimensionMismatch MSB.material_generative_specification(material, MSB.CoordinateMomentumAlgebra(3);
            dataset=dataset, symmetry_action=MSB.VectorDisplacementAction())
        result = MSB.generate_hamiltonian_family(material, MSB.CoordinateMomentumAlgebra(6);
            dataset=dataset, symmetry_action=MSB.VectorDisplacementAction(), acoustic_sum_rule=true,
            candidates=MSB.AbstractOperatorExpr[MSB.xop(1) * MSB.xop(1), MSB.xop(4) * MSB.xop(4), MSB.xop(1) * MSB.xop(4)],
            basis_labels=[:k1, :k2, :cross], max_body_order=2)
        @test result.metadata[:acoustic_sum_rule]
        @test length(MSB.acoustic_sum_rule_constraints(result.hamiltonian_space; nsites=2)) == 2
        @test length(result.parameter_space.constraints) == 2
        @test MSB.parameter_constraint_report(result.parameter_space, [1.0, 1.0, -2.0]).valid
        @test !MSB.parameter_constraint_report(result.parameter_space, [1.0, 1.0, 0.0]).valid
    end

    @testset "F2 certified multicomponent geometry and registered momentum observables" begin
        lattice = MSB.BravaisLattice(Diagonal([4.0, 4.0, 6.0]) |> Matrix)
        sites = [MSB.BasisSite(:A, :A, [0.25, 0.0, 0.0]), MSB.BasisSite(:B, :B, [0.75, 0.0, 0.0])]
        material = MSB.Material("component-resolved two-site reference", MSB.LatticeCrystalStructure(lattice, sites);
            species=Dict(:A => MSB.AtomicSpecies(:A; mass=2.0), :B => MSB.AtomicSpecies(:B; mass=5.0)))
        identity = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        dataset = _msb_dataset([identity], 2)
        fermions = MSB.FermionAlgebra(8)
        generated_f = MSB.generate_hamiltonian_family(material, fermions; dataset=dataset,
            symmetry_action=MSB.SpinfulFermionAction(norbitals=2),
            candidates=MSB.AbstractOperatorExpr[MSB.nf(1) + MSB.nf(5)], basis_labels=[:density])
        f_geometry = MSB.generation_context(generated_f, fermions).geometry
        @test f_geometry isa MSB.GeometryContext
        @test f_geometry.indexing === :site_component_layout
        @test f_geometry.component_layout.kind === :spinful_fermion
        @test f_geometry.site_masses == [2.0, 5.0]
        @test f_geometry.positions == [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        @test [MSB.component_site(f_geometry, i) for i in 1:8] == [1, 1, 1, 1, 2, 2, 2, 2]
        @test MSB.component_index(f_geometry, 2, 2, 1) == 7
        @test MSB.component_index(f_geometry, 1, 2, 2) == 4
        @test MSB.component_coordinates(f_geometry, 8) == (site=2, orbital=2, spin=2)
        @test_throws BoundsError MSB.component_index(f_geometry, 1, 3, 1)
        @test_throws BoundsError MSB.component_site(f_geometry, 9)
        @test_throws ArgumentError MSB.component_index(f_geometry, 1, :x)
        representation_f = MSB.Representation(:f2_fermion, MSB.FermionFockSpace(8), fermions,
            MSB.FermionOccupationBasis(collect(1:8)))
        model_f = MSB.instantiate_model(generated_f, representation_f, [0.8])
        @test MSB.observable_context(model_f).convention === :fermion_site_orbital_spin
        @test MSB.model_context(model_f).material === material
        selection_f = MSB.structural_selection(MSB.structural_family(generated_f), [:density])
        selected_f = MSB.instantiate_model(selection_f, representation_f, [0.8])
        @test MSB.observable_context(selected_f).convention === :fermion_site_orbital_spin
        @test MSB.geometry_context(selected_f).positions == f_geometry.positions
        @test MSB.geometry_context(model_f).positions == f_geometry.positions
        @test length(model_f.observables) == 12
        @test model_f.parameters[:positions] == f_geometry.positions
        density_site2 = MSB.nf(5) + MSB.nf(6) + MSB.nf(7) + MSB.nf(8)
        @test _msb_zero(model_f.observables[:fermion_density_2] - density_site2, fermions)
        @test _msb_zero(model_f.observables[:fermion_density_2_2] - (MSB.nf(7) + MSB.nf(8)), fermions)
        sy_site1 = (MSB.c(1)' * MSB.c(2) - MSB.c(2)' * MSB.c(1) +
                    MSB.c(3)' * MSB.c(4) - MSB.c(4)' * MSB.c(3)) / (2im)
        @test _msb_zero(model_f.observables[:fermion_spin_y_1] - sy_site1, fermions)
        field_f = MSB.registered_momentum_field(model_f, [:fermion_density_1, :fermion_density_2])
        @test field_f.positions == f_geometry.positions
        q = [π / 2, 0.0, 0.0]
        density_site1 = MSB.nf(1) + MSB.nf(2) + MSB.nf(3) + MSB.nf(4)
        expected_f = (exp(-im * dot(q, f_geometry.positions[1])) * density_site1 +
                      exp(-im * dot(q, f_geometry.positions[2])) * density_site2) / sqrt(2)
        @test _msb_zero(MSB.Observables.momentum_operator(model_f, field_f, q) - expected_f, fermions)
        @test_throws ArgumentError MSB.instantiate_model(generated_f, representation_f, [0.8];
            observables=Dict(:fermion_density_1 => MSB.nf(5)))
        @test_throws ArgumentError MSB.instantiate_model(generated_f, representation_f, [0.8];
            metadata_parameters=Dict(:positions => [[1.0, 0.0, 0.0], [3.1, 0.0, 0.0]]))

        coordinates = MSB.CoordinateMomentumAlgebra(6)
        generated_x = MSB.generate_hamiltonian_family(material, coordinates; dataset=dataset,
            symmetry_action=MSB.VectorDisplacementAction(),
            candidates=MSB.AbstractOperatorExpr[MSB.xop(1) * MSB.xop(1) + MSB.xop(4) * MSB.xop(4)],
            basis_labels=[:spring], max_body_order=2)
        x_geometry = MSB.generation_context(generated_x, coordinates).geometry
        @test x_geometry.indexing === :site_component_layout
        @test x_geometry.component_layout.kind === :cartesian_vector
        @test x_geometry.site_masses == [2.0, 5.0]
        @test [MSB.component_site(x_geometry, i) for i in 1:6] == [1, 1, 1, 2, 2, 2]
        @test MSB.component_coordinates(x_geometry, 5) == (site=2, axis=:y)
        @test MSB.component_index(x_geometry, 2, :z) == 6
        @test_throws ArgumentError MSB.component_index(x_geometry, 1, 1, 2)
        @test_throws ArgumentError MSB.component_index(x_geometry, 1, :w)
        representation_x = MSB.Representation(:f2_vector, MSB.CoordinateHilbertSpace(6), coordinates,
            MSB.CoordinateBasis(collect(1:6)); ordering=collect(1:6))
        model_x = MSB.instantiate_model(generated_x, representation_x, [0.5])
        @test MSB.observable_context(model_x).convention === :phonon_cartesian_site
        selection_x = MSB.structural_selection(MSB.structural_family(generated_x), [:spring])
        selected_x = MSB.instantiate_model(selection_x, representation_x, [0.5])
        @test MSB.observable_context(selected_x).convention === :phonon_cartesian_site
        @test model_x.parameters[:masses] == [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
        @test length(model_x.observables) == 12
        @test model_x.observables[:phonon_displacement_z_2] == MSB.xop(6)
        @test model_x.observables[:phonon_momentum_y_2] == MSB.pop(5)
        field_x = MSB.registered_momentum_field(model_x, [:phonon_displacement_x_1, :phonon_displacement_x_2])
        expected_x = (exp(-im * dot(q, x_geometry.positions[1])) * MSB.xop(1) +
                      exp(-im * dot(q, x_geometry.positions[2])) * MSB.xop(4)) / sqrt(2)
        @test _msb_zero(MSB.Observables.momentum_operator(model_x, field_x, q) - expected_x, coordinates)
        @test_throws ArgumentError MSB.instantiate_model(generated_x, representation_x, [0.5];
            metadata_parameters=Dict(:masses => fill(1.0, 6)))
        @test_throws ArgumentError MSB.instantiate_model(generated_x, representation_x, [0.5];
            observables=Dict(:phonon_displacement_x_1 => MSB.xop(4)))
        @test_throws DimensionMismatch MSB.GeometryContext(x_geometry.positions, x_geometry.direct, x_geometry.reciprocal,
            x_geometry.labels; indexing=:site_component_layout, component_layout=MSB.ComponentIndexLayout(:cartesian_vector, 3))
        @test_throws ArgumentError MSB.GeometryContext(x_geometry.positions, x_geometry.direct, x_geometry.reciprocal,
            x_geometry.labels; indexing=:one_index_per_site, component_layout=x_geometry.component_layout)
    end

    @testset "Representation transforms do not reuse source-only physical constraints" begin
        algebra = MSB.SpinAlgebra(2)
        source = MSB.Representation(:milestone_b_spin, MSB.SpinHilbertSpace([1//2, 1//2]), algebra, MSB.SpinProductBasis([1, 2]))
        id = MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
        dataset = _msb_dataset([id], 2)
        specification = MSB.GenerativeSpecification(dataset, (1, 2), (MSB.SpinHilbertSpace([1//2]), MSB.SpinHilbertSpace([1//2])),
            algebra, MSB.Crystallography.standard_symmetry_action; max_body_order=1, max_range=1.0, range_evaluator=op -> 0.0)
        space = MSB.HamiltonianSpace(specification, MSB.HamiltonianBasis(MSB.AbstractOperatorExpr[MSB.Sz(1) + MSB.Sz(2)]))
        model = MSB.ManyBodyModel(source, MSB.Sz(1) + MSB.Sz(2); model_space=space)
        transformed = MSB.transform(MSB.JordanWigner([1, 2]), model).model
        @test transformed.specification.algebra isa MSB.FermionAlgebra
        @test isnothing(transformed.specification.symmetry_action)
        @test isnothing(transformed.specification.local_spaces)
        @test isnothing(transformed.specification.max_body_order)
        @test isnothing(transformed.specification.range_evaluator)
        @test transformed.specification.metadata[:source_max_body_order] == 1
        @test transformed.specification.metadata[:symmetry_action_transport] === :not_transported
        @test !MSB.admissibility_report(transformed).complete
        @test_throws ArgumentError MSB.Crystallography.reynolds_project(
            transformed.hamiltonian, transformed.specification, _msb_crystal(sites=2), dataset)
    end
end


@testset "F4 discovered crystallographic group certification" begin
    I3 = Matrix{Int}(I, 3, 3)
    lattice = MSB.BravaisLattice(Matrix{Float64}(I, 3, 3))
    sites = [MSB.BasisSite(:Cs, :Cs, [0.0, 0.0, 0.0]), MSB.BasisSite(:Cl, :Cl, [0.5, 0.5, 0.5])]
    crystal = MSB.LatticeCrystalStructure(lattice, sites)
    species = Dict(:Cs => MSB.AtomicSpecies(:Cs; mass=1.0), :Cl => MSB.AtomicSpecies(:Cl; mass=1.0))
    material = MSB.Material("cubic B2 symmetry reference", crystal; species=species,
        basis_properties=[MSB.SiteProperties(spin=1//2), MSB.SiteProperties(spin=1//2)])
    source = MSB.material_crystallographic_structure(material)
    dataset = MSB.discover_symmetry(source)
    @test dataset.spacegroup_number == 221
    @test length(dataset.operations) == 48
    @test MSB.validate_crystallographic_group(source, dataset)
    algebra = MSB.SpinAlgebra(2)
    Jx = MSB.Sx(1) * MSB.Sx(2)
    Jy = MSB.Sy(1) * MSB.Sy(2)
    Jz = MSB.Sz(1) * MSB.Sz(2)
    exchange = Jx + Jy + Jz
    generated = MSB.generate_hamiltonian_family(material, algebra; certify_group=true,
        candidates=MSB.AbstractOperatorExpr[Jx, Jy, Jz], max_body_order=2)
    @test generated.specification.metadata[:crystallographic_group_certified]
    @test length(generated.hamiltonian_space) == 1
    @test _msb_zero(MSB.Crystallography.reynolds_project(Jx, generated.specification, source, dataset) - exchange / 3, algebra)
    @test _msb_zero(MSB.Crystallography.reynolds_project(exchange, generated.specification, source, dataset) - exchange, algebra)

    C4 = [0 -1 0; 1 0 0; 0 0 1]
    screw_positions = Vector{Float64}[]
    fractional = [0.17, 0.09, 0.13]
    for _ in 1:4
        push!(screw_positions, copy(fractional))
        fractional = mod.(C4 * fractional + [0.0, 0.0, 0.25], 1.0)
    end
    screw_sites = [MSB.BasisSite(Symbol("P", i), :P, screw_positions[i]) for i in 1:4]
    screw_crystal = MSB.LatticeCrystalStructure(lattice, screw_sites)
    screw_material = MSB.Material("fourfold screw reference", screw_crystal;
        species=Dict(:P => MSB.AtomicSpecies(:P; mass=1.0)),
        basis_properties=[MSB.SiteProperties(spin=1//2) for _ in 1:4])
    screw_source = MSB.material_crystallographic_structure(screw_material)
    screw_dataset = MSB.discover_symmetry(screw_source)
    @test screw_dataset.spacegroup_number == 76
    @test length(screw_dataset.operations) == 4
    @test MSB.validate_crystallographic_group(screw_source, screw_dataset)
    @test any(op -> op.rotation == C4 && isapprox(mod(op.translation[3], 1.0), 0.25; atol=1e-6), screw_dataset.operations)
    screw_algebra = MSB.SpinAlgebra(4)
    field = MSB.AbstractOperatorExpr[MSB.Sz(i) for i in 1:4]
    screw_result = MSB.generate_hamiltonian_family(screw_material, screw_algebra; certify_group=true,
        candidates=field, max_body_order=1)
    @test length(screw_result.hamiltonian_space) == 1
    @test _msb_zero(MSB.Crystallography.reynolds_project(MSB.Sz(1), screw_result.specification,
        screw_source, screw_dataset) - (field[1] + field[2] + field[3] + field[4]) / 4, screw_algebra)

    identity = MSB.SpaceGroupOperation(I3, zeros(3))
    screw = MSB.SpaceGroupOperation(C4, [0.0, 0.0, 0.25])
    incomplete = _msb_dataset([identity, screw], 4)
    @test_throws ArgumentError MSB.validate_crystallographic_group(screw_source, incomplete)
    @test_throws ArgumentError MSB.material_generative_specification(screw_material, screw_algebra;
        dataset=incomplete, certify_group=true)
    bad_metric = MSB.SpaceGroupOperation(Diagonal([2, 1, 1]) |> Matrix, zeros(3))
    @test_throws ArgumentError MSB.validate_crystallographic_group(screw_source, _msb_dataset([identity, bad_metric], 4))
    time_reversal = MSB.SpaceGroupOperation(I3, zeros(3); time_reversal=true)
    @test MSB.validate_crystallographic_group(source, _msb_dataset([identity, time_reversal], 2))
    @test_throws ArgumentError MSB.validate_crystallographic_group(source, _msb_dataset([identity, identity], 2))
    @test_throws ArgumentError MSB.validate_crystallographic_group(source, _msb_dataset([time_reversal], 2))
end

@testset "F4 finite-reference rigid-rotation constraints" begin
    lattice = MSB.BravaisLattice(Matrix(Diagonal([4.0, 4.0, 4.0])))
    sites = [MSB.BasisSite(:A, :A, [0.25, 0.0, 0.0]), MSB.BasisSite(:B, :B, [0.75, 0.0, 0.0])]
    material = MSB.Material("finite Hessian rotation reference", MSB.LatticeCrystalStructure(lattice, sites);
        species=Dict(:A => MSB.AtomicSpecies(:A; mass=2.0), :B => MSB.AtomicSpecies(:B; mass=2.0)))
    dataset = _msb_dataset([MSB.SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))], 2)
    algebra = MSB.CoordinateMomentumAlgebra(6)
    axial = MSB.xop(1) * MSB.xop(1) + MSB.xop(4) * MSB.xop(4) - 2 * MSB.xop(1) * MSB.xop(4)
    transverse = MSB.xop(2) * MSB.xop(2) + MSB.xop(5) * MSB.xop(5) - 2 * MSB.xop(2) * MSB.xop(5)
    coordinates = [1.0 3.0; 0.0 0.0; 0.0 0.0]
    candidates = MSB.AbstractOperatorExpr[axial, transverse]
    base = MSB.generate_hamiltonian_family(material, algebra; dataset=dataset,
        symmetry_action=MSB.VectorDisplacementAction(), candidates=candidates, basis_labels=[:axial, :transverse],
        max_body_order=2)
    @test isempty(MSB.acoustic_sum_rule_constraints(base.hamiltonian_space; nsites=2))
    rotational = MSB.rotational_sum_rule_constraints(base.hamiltonian_space; nsites=2, positions=coordinates)
    @test length(rotational) == 1
    @test rotational[1].metadata[:physical_rule] === :finite_rigid_rotation
    @test_throws ArgumentError MSB.rotational_sum_rule_constraints(base.hamiltonian_space; nsites=2,
        positions=coordinates, boundary=:periodic)
    @test_throws DimensionMismatch MSB.rotational_sum_rule_constraints(base.hamiltonian_space; nsites=2,
        positions=zeros(3, 3))
    @test_throws ArgumentError MSB.generate_hamiltonian_family(material, algebra; dataset=dataset,
        symmetry_action=MSB.VectorDisplacementAction(), candidates=candidates, rotational_sum_rule=true,
        acoustic_sum_rule=true, rotational_positions=coordinates)
    @test_throws ArgumentError MSB.generate_hamiltonian_family(material, algebra; dataset=dataset,
        symmetry_action=MSB.VectorDisplacementAction(), candidates=candidates, rotational_sum_rule=true,
        rotational_scope=:finite_site_hessian, rotational_positions=coordinates)
    result = MSB.generate_hamiltonian_family(material, algebra; dataset=dataset,
        symmetry_action=MSB.VectorDisplacementAction(), candidates=candidates, basis_labels=[:axial, :transverse],
        max_body_order=2, acoustic_sum_rule=true, rotational_sum_rule=true,
        rotational_scope=:finite_site_hessian, rotational_positions=coordinates)
    @test result.metadata[:rotational_sum_rule]
    @test result.metadata[:rotational_constraint_count] == 1
    @test result.metadata[:rotational_scope] === :finite_site_hessian
    @test_throws ArgumentError MSB.generate_hamiltonian_family(material, algebra; dataset=dataset,
        symmetry_action=MSB.VectorDisplacementAction(), candidates=candidates, basis_labels=[:axial, :transverse],
        acoustic_sum_rule=true, rotational_sum_rule=true, rotational_scope=:finite_site_hessian,
        rotational_positions=coordinates, parameter_values=[1.0, 1.0])
    @test MSB.parameter_constraint_report(result.parameter_space, [1.0, 0.0]).valid
    @test !MSB.parameter_constraint_report(result.parameter_space, [1.0, 1.0]).valid
    translated = coordinates .+ [4.0, -2.0, 1.0]
    shifted = MSB.rotational_sum_rule_constraints(base.hamiltonian_space; nsites=2, positions=translated)
    @test length(shifted) == length(rotational)
    @test shifted[1].coefficients == rotational[1].coefficients
end
