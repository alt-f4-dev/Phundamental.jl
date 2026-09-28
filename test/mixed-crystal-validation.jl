using Test
using LinearAlgebra
using Random
using Phundamental

const F = Phundamental

@testset "Symmetry orbit expansion and mixed crystal disorder" begin
    @testset "Database symmetry-orbit expansion" begin
        inversion_operations = F.space_group_operations(2)
        general_orbit = F.expand_symmetry_orbit([0.13, 0.21, 0.37], inversion_operations)
        special_orbit = F.expand_symmetry_orbit([0.0, 0.0, 0.0], inversion_operations)

        @test length(inversion_operations) == 2
        @test length(general_orbit) == 2
        @test length(special_orbit) == 1
        @test sum(length, special_orbit.operation_indices) == 2
        @test_throws ArgumentError F.space_group_operations(2; hall_number=1, choice=1)
    end

    @testset "LBCO P4_2/ncm origin-choice-2 multiplicities" begin
        operations = F.space_group_operations(138; choice=2)
        laba = F.expand_symmetry_orbit([0.0053, 0.0053, 0.36059], operations)
        cu = F.expand_symmetry_orbit([0.0, 0.0, 0.0], operations)
        o1 = F.expand_symmetry_orbit([0.25, 0.25, 0.0074], operations)
        o1p = F.expand_symmetry_orbit([0.75, 0.25, 0.0], operations)
        o2 = F.expand_symmetry_orbit([-0.0163, -0.0163, 0.1821], operations)

        @test length(laba) == 8
        @test length(cu) == 4
        @test length(o1) == 4
        @test length(o1p) == 4
        @test length(o2) == 8
    end

    @testset "Mixed occupancy and exact disordered supercell" begin
        lattice = F.BravaisLattice(reshape([1.0], 1, 1))
        occupancy = F.MixedOccupancy(:La => 0.75, :Ba => 0.25)
        mixed_site = F.MixedBasisSite(:A, occupancy, [0.0]; properties=(sublattice=:A,))
        crystal = F.MixedCrystalStructure(lattice, [mixed_site])

        cluster = F.disordered_supercell(crystal, (4,); composition=:exact, rng=MersenneTwister(17), periodic=true)
        realized_species = [site.species for site in cluster.sites]

        @test cluster isa F.Supercell
        @test F.nsites(cluster) == 4
        @test cluster.repetitions == [1]
        @test all(cluster.periodic)
        @test F.direct_matrix(cluster.crystal.lattice) ≈ reshape([4.0], 1, 1)
        @test count(==(:La), realized_species) == 3
        @test count(==(:Ba), realized_species) == 1
        @test all(site.properties.sublattice === :A for site in cluster.crystal.basis)
        @test F.occupancy(mixed_site) === occupancy

        species = Dict(
            :La => F.AtomicSpecies(:La; mass=1.0, nuclear_scattering_length=2.0),
            :Ba => F.AtomicSpecies(:Ba; mass=3.0, nuclear_scattering_length=4.0),
        )
        material = F.Material("explicit mixed-site realization", cluster; species=species)
        expected_masses = [species[site.species].mass for site in cluster.sites]
        expected_scattering = [species[site.species].nuclear_scattering_length for site in cluster.sites]

        @test F.mean_mass(occupancy, species) ≈ 1.5
        @test F.mean_scattering_length(occupancy, species) ≈ 2.5 + 0.0im
        @test [F.site_mass(material, cluster, i) for i in 1:F.nsites(cluster)] == expected_masses
        @test [F.site_scattering_length(material, cluster, i) for i in 1:F.nsites(cluster)] == expected_scattering

        phonon = F.build_model(material, cluster, F.HarmonicPhononModel(zeros(Float64, F.nsites(cluster), F.nsites(cluster))))
        @test phonon.parameters[:masses] == expected_masses
    end

    @testset "Stochastic disorder is reproducible for an explicit RNG seed" begin
        lattice = F.BravaisLattice(reshape([1.0], 1, 1))
        occupancy = F.MixedOccupancy(:A => 0.4, :B => 0.6)
        crystal = F.MixedCrystalStructure(lattice, [F.MixedBasisSite(:mixed, occupancy, [0.0])])

        first_cluster = F.disordered_supercell(crystal, (32,); rng=MersenneTwister(20260924))
        second_cluster = F.disordered_supercell(crystal, (32,); rng=MersenneTwister(20260924))
        first_species = [site.species for site in first_cluster.sites]
        second_species = [site.species for site in second_cluster.sites]

        @test first_species == second_species
    end
end

@testset "Crystal basis expansion hierarchy" begin
    irreps = (
        LaBa=[0.0053, 0.0053, 0.36059],
        Cu=[0.0, 0.0, 0.0],
        O1=[0.25, 0.25, 0.0074],
        O1p=[0.75, 0.25, 0.0],
        O2=[-0.0163, -0.0163, 0.1821],
    )
    mixed_occ = F.MixedOccupancy(:La => 0.9375, :Ba => 0.0625)
    occs = (LaBa=mixed_occ, Cu=:Cu, O1=:O, O1p=:O, O2=:O)
    site_props = (
        LaBa=(orbit=:LaBa,),
        Cu=(orbit=:Cu,),
        O1=(orbit=:O1,),
        O1p=(orbit=:O1p,),
        O2=(orbit=:O2,),
    )

    operations = F.space_group_operations(138; choice=2)
    mid_basis = F.expand_crystal_basis(operations, irreps, occs, site_props)
    high_basis = F.expand_crystal_basis(138, irreps, occs, site_props; choice=2)

    @test length(mid_basis) == 28
    @test length(high_basis) == 28
    @test [site.label for site in mid_basis] == [site.label for site in high_basis]
    @test [site.fractional for site in mid_basis] == [site.fractional for site in high_basis]
    @test count(site -> site isa F.MixedBasisSite, high_basis) == 8
    @test count(site -> site isa F.BasisSite && site.species === :Cu, high_basis) == 4
    @test count(site -> site isa F.BasisSite && site.species === :O, high_basis) == 16
    @test all(site -> site.properties.orbit === :LaBa, high_basis[1:8])

    no_properties = F.expand_crystal_basis(138, irreps, occs; choice=2)
    @test all(site -> site.properties == NamedTuple(), no_properties)

    partial_properties = (LaBa=(orbit=:LaBa,),)
    partial_basis = F.expand_crystal_basis(138, irreps, occs, partial_properties; choice=2)
    @test all(site -> site.properties == (orbit=:LaBa,), partial_basis[1:8])
    @test all(site -> site.properties == NamedTuple(), partial_basis[9:end])

    irreps_dict = Dict{Symbol,Any}(pairs(irreps))
    occs_dict = Dict{Symbol,Any}(pairs(occs))
    props_dict = Dict{Symbol,Any}(pairs(site_props))
    dict_basis = F.expand_crystal_basis(138, irreps_dict, occs_dict, props_dict; choice=2)

    @test length(dict_basis) == 28
    @test Set(site.label for site in dict_basis) == Set(site.label for site in high_basis)

    bad_occs = (LaBa=mixed_occ, Cu=:Cu)
    @test_throws ArgumentError F.expand_crystal_basis(138, irreps, bad_occs; choice=2)
    @test_throws ArgumentError F.expand_crystal_basis(138, irreps, occs, (Ghost=(x=1,),); choice=2)
end

@testset "Crystal basis convenience overloads" begin
    operations = F.space_group_operations(2)
    cu_irrep = [0.13, 0.21, 0.37]
    o1_irrep = [0.19, 0.07, 0.28]
    o2_irrep = [0.31, 0.16, 0.09]
    cu_properties = (spin=0.5, orbit=:Cu)

    single_basis = F.expand_crystal_basis(operations, :Cu, cu_irrep; properties=cu_properties)
    single_reference = F.expand_crystal_basis(operations, (Cu=cu_irrep,), (Cu=:Cu,), (Cu=cu_properties,))
    single_number_basis = F.expand_crystal_basis(2, :Cu, cu_irrep; properties=cu_properties)

    @test [site.label for site in single_basis] == [site.label for site in single_reference]
    @test [site.fractional for site in single_basis] == [site.fractional for site in single_reference]
    @test [site.label for site in single_number_basis] == [site.label for site in single_reference]
    @test all(site -> site.species === :Cu, single_basis)
    @test all(site -> site.properties == cu_properties, single_basis)

    explicit_species_basis = F.expand_crystal_basis(operations, :Cu1, cu_irrep; species=:Cu, properties=cu_properties)
    @test all(site -> site.species === :Cu, explicit_species_basis)
    @test all(startswith(string(site.label), "Cu1_") for site in explicit_species_basis)

    site_labels = (:Cu1, :O1, :O2)
    irreps = (cu_irrep, o1_irrep, o2_irrep)
    species = (:Cu, :O, :O)
    properties = ((orbit=:Cu1,), (orbit=:O1,), (orbit=:O2,))
    multi_basis = F.expand_crystal_basis(operations, site_labels, irreps; species=species, properties=properties)
    multi_number_basis = F.expand_crystal_basis(2, site_labels, irreps; species=species, properties=properties)
    multi_reference = F.expand_crystal_basis(operations,
                                             (Cu1=cu_irrep, O1=o1_irrep, O2=o2_irrep),
                                             (Cu1=:Cu, O1=:O, O2=:O),
                                             (Cu1=properties[1], O1=properties[2], O2=properties[3]))

    @test [site.label for site in multi_basis] == [site.label for site in multi_reference]
    @test [site.species for site in multi_basis] == [site.species for site in multi_reference]
    @test [site.fractional for site in multi_basis] == [site.fractional for site in multi_reference]
    @test [site.properties for site in multi_basis] == [site.properties for site in multi_reference]
    @test [site.label for site in multi_number_basis] == [site.label for site in multi_reference]

    default_species_basis = F.expand_crystal_basis(operations, (:Cu, :O), (cu_irrep, o1_irrep))
    @test Set(site.species for site in default_species_basis) == Set((:Cu, :O))
    @test all(site -> site.properties == NamedTuple(), default_species_basis)

    mixed_occupancy = F.MixedOccupancy(:La => 0.75, :Ba => 0.25)
    mixed_basis = F.expand_crystal_basis(operations, (:A, :Cu1), (o2_irrep, cu_irrep); species=(mixed_occupancy, :Cu))
    @test count(site -> site isa F.MixedBasisSite, mixed_basis) == length(F.expand_symmetry_orbit(o2_irrep, operations))

    @test_throws ArgumentError F.expand_crystal_basis(operations, (:O, :O), (o1_irrep, o2_irrep))
    @test_throws DimensionMismatch F.expand_crystal_basis(operations, (:Cu1, :O1), (cu_irrep,); species=(:Cu, :O))
    @test_throws DimensionMismatch F.expand_crystal_basis(operations, (:Cu1, :O1), (cu_irrep, o1_irrep); species=(:Cu,))
    @test_throws DimensionMismatch F.expand_crystal_basis(operations, (:Cu1, :O1), (cu_irrep, o1_irrep); properties=((orbit=:Cu1,),))
    @test_throws ArgumentError F.expand_crystal_basis(operations, (:Cu1, :O1), (cu_irrep, o1_irrep); species=:Cu)
end
