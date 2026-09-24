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
