using Test
using Phundamental
using LinearAlgebra: I, inv, Diagonal

@testset "momentum Green function and lattice bubble" begin
    beta = 4.0
    fermionic = FermionicMatsubaraAxis(beta, collect(-2:2))
    bosonic = BosonicMatsubaraAxis(beta, [0])
    sigma = zero_self_energy(fermionic, 1; labels=[:local])
    dispersion = [-1.0, 1.0]
    green = momentum_green(dispersion, fermionic, sigma; kpoints=[0.0, π])
    shifts = periodic_momentum_shift_maps((2,))
    bubbles, diagnostics = lattice_bubble(green, shifts.maps, bosonic; missing_frequency=:error)

    @test diagnostics.frequency_box_complete
    @test length(bubbles) == 2
    for (iq, mapping) in enumerate(shifts.maps), iν in eachindex(fermionic.indices)
        expected = -beta * sum(green.values[ik, iν] * green.values[mapping[ik], iν] for ik in 1:2) / 2
        @test bubbles[iq].values[iν, iν, 1] ≈ expected
    end

    zero_vertex = IrreducibleVertex(fermionic, bosonic, zeros(ComplexF64, length(fermionic), length(fermionic), 1); channel=:density)
    reconstructed = bethe_salpeter(bubbles[1], zero_vertex)
    @test reconstructed.values ≈ bubbles[1].values
end


@testset "F6 matrix orbital Dyson propagation and orbital-pair bubbles" begin

    beta = 5.0
    axis = FermionicMatsubaraAxis(beta, collect(-2:2))
    bosonic = BosonicMatsubaraAxis(beta, [0])
    onsite = ComplexF64[0.2 0.35im; -0.35im -0.3]
    hopping = ComplexF64[-0.6 0.12; 0.04 -0.25]
    weights = [1.0, 2.0, 3.0, 4.0]
    nk = length(weights)
    ks = [2π * (j - 1) / nk for j in 1:nk]
    matrices = cat([onsite + hopping * cis(k) + hopping' * cis(-k) for k in ks]...; dims=3)
    lattice = MatrixMomentumLattice(matrices; kpoints=ks, weights=weights,
        qmaps=periodic_momentum_shift_maps((nk,)).maps, labels=[:a, :b])
    @test lattice.weights ≈ weights ./ sum(weights)
    @test lattice.labels == [:a, :b]
    @test lattice.qmaps[1] == collect(1:nk)
    @test_throws ArgumentError MatrixMomentumLattice(matrices .+ 0.1im)
    @test_throws DimensionMismatch MatrixMomentumLattice(matrices; labels=[:only_one])
    @test_throws ArgumentError MatrixMomentumLattice(matrices; qmaps=[[1, 1, 2, 3]])

    sigma_values = zeros(ComplexF64, 2, 2, length(axis))
    for iν in eachindex(axis.indices)
        sigma_values[:, :, iν] .= ComplexF64[0.1 + 0.05im 0.03im; -0.02im -0.07 + 0.02im]
    end
    sigma = SelfEnergy(axis, sigma_values; labels=[:a, :b])
    chemical_potential = 0.17
    green = momentum_green(lattice, axis, sigma; chemical_potential=chemical_potential)
    @test green isa MatrixMomentumResolvedGreenFunction
    @test size(green.values) == (2, 2, nk, length(axis))
    @test green.labels == [:a, :b]
    direct_green = momentum_green(matrices, axis, sigma; kpoints=ks, labels=[:a, :b],
        weights=weights, chemical_potential=chemical_potential)
    @test direct_green.values ≈ green.values
    mislabeled = SelfEnergy(axis, sigma_values; labels=[:incorrect, :b])
    @test_throws ArgumentError momentum_green(lattice, axis, mislabeled)
    high_axis = FermionicMatsubaraAxis(beta, [999])
    high_green = momentum_green(lattice, high_axis, zero_self_energy(high_axis, 2; labels=[:a, :b]))
    @test maximum(abs.(high_axis[1] .* high_green.values[:, :, 1, 1] .- Matrix{ComplexF64}(I, 2, 2))) < 2e-3
    for ik in 1:nk, iν in eachindex(axis.indices)
        expected = inv((axis[iν] + chemical_potential) * Matrix{ComplexF64}(I, 2, 2) - matrices[:, :, ik] - sigma_values[:, :, iν])
        @test green.values[:, :, ik, iν] ≈ expected atol=2e-13 rtol=2e-13
    end
    local_green = lattice_green(lattice, axis, sigma; chemical_potential=chemical_potential)
    @test matrix_dimension(local_green) == 2
    @test component_labels(local_green) == [:a, :b]
    for iν in eachindex(axis.indices)
        independent_sum = sum(lattice.weights[ik] * green.values[:, :, ik, iν] for ik in 1:nk)
        @test local_green.values[:, :, iν] ≈ independent_sum atol=2e-13
    end

    bubble, diagnostics = orbital_lattice_bubble(green, lattice.qmaps, bosonic)
    @test diagnostics.frequency_box_complete
    @test length(bubble) == nk
    @test size(bubble[1].values) == (2, 2, 2, 2, length(axis), 1)
    for (iq, map) in enumerate(lattice.qmaps), iν in eachindex(axis.indices), a in 1:2, b in 1:2, c in 1:2, d in 1:2
        independent = -beta * sum(lattice.weights[ik] * green.values[a, c, map[ik], iν] * green.values[d, b, ik, iν] for ik in 1:nk)
        @test bubble[iq].values[a, b, c, d, iν, 1] ≈ independent atol=2e-13
    end
    shifted_bosonic = BosonicMatsubaraAxis(beta, [0, 1])
    @test_throws ArgumentError orbital_lattice_bubble(green, lattice.qmaps, shifted_bosonic)
    truncated, status = orbital_lattice_bubble(green, lattice.qmaps, shifted_bosonic; missing_frequency=:zero)
    @test !status.frequency_box_complete
    @test status.missing_frequency_entries == nk
    @test all(iszero, truncated[1].values[:, :, :, :, end, 2])
    @test_throws DimensionMismatch momentum_green(zeros(ComplexF64, 3, 3, nk), axis, sigma)
    @test_throws ArgumentError DMFTProblem(lattice, 2.0; temperature=0.5)
    @test_throws ArgumentError DMFTForwardContext(lattice; temperature=0.5)
    @test_throws ArgumentError LadderDGAForwardContext(lattice, bosonic)

    # Scalar methods retain their independent, unchanged data layouts.
    scalar_sigma = zero_self_energy(axis, 1)
    scalar_green = momentum_green([-0.5, 0.5], axis, scalar_sigma)
    @test scalar_green isa MomentumResolvedGreenFunction
    @test size(scalar_green.values) == (2, length(axis))
end

@testset "F6 orbital Bravais gauge, analytic dispersion, and rejection" begin

    atomic_lattice = BravaisLattice(reshape([2.0], 1, 1))
    crystal = LatticeCrystalStructure(atomic_lattice, [BasisSite(:H, :H, [0.0])])
    material = Material("F6 explicit orbital chain", crystal; species=Dict(:H => lookup_species(:H)))
    onsite = ComplexF64[0.4 0.25; 0.25 -0.1]
    hopping = ComplexF64[-0.6 0.2im; 0.1 -0.3]
    centers = [(0.0,), (0.25,)]
    hop = OrbitalBravaisHopping((1,), hopping)
    lattice = matrix_bravais_momentum_lattice(material, (8,), [hop]; onsite=onsite,
        orbital_positions=centers, labels=[:one, :two])
    @test size(lattice.hamiltonians) == (2, 2, 8)
    @test lattice.qmaps[1] == collect(1:8)
    @test lattice.labels == [:one, :two]
    @test lattice.gauge === :embedded
    embedded_axis = FermionicMatsubaraAxis(5.0, [-1, 0, 1])
    embedded_green = momentum_green(lattice, embedded_axis, zero_self_energy(embedded_axis, 2; labels=[:one, :two]))
    @test_throws ArgumentError orbital_lattice_bubble(embedded_green, lattice.qmaps, BosonicMatsubaraAxis(5.0, [0]))
    for (ik, tuple_k) in enumerate(lattice.kpoints)
        k = only(tuple_k)
        reference = ComplexF64[onsite[a, b] * cis(k * (centers[b][1] - centers[a][1])) for a in 1:2, b in 1:2]
        for a in 1:2, b in 1:2
            value = hopping[a, b] * cis(k * (1 + centers[b][1] - centers[a][1]))
            reference[a, b] += value
            reference[b, a] += conj(value)
        end
        @test lattice.hamiltonians[:, :, ik] ≈ reference atol=2e-13
        @test lattice.hamiltonians[:, :, ik] ≈ lattice.hamiltonians[:, :, ik]' atol=2e-13
    end
    @test_throws ArgumentError matrix_bravais_momentum_lattice(material, (8,), [hop, OrbitalBravaisHopping((-1,), hopping')]; onsite=onsite)
    @test_throws ArgumentError matrix_bravais_momentum_lattice(material, (8,), [hop, hop]; onsite=onsite)
    @test_throws DimensionMismatch matrix_bravais_momentum_lattice(material, (8,), [hop]; onsite=onsite, orbital_positions=[(0.0,)])
    @test_throws ArgumentError matrix_bravais_momentum_lattice(material, (8,), [hop]; onsite=ComplexF64[0 1; 0 0])
    @test_throws ArgumentError OrbitalBravaisHopping((0,), hopping)
    @test_throws ArgumentError OrbitalBravaisHopping{1}((0,), hopping)
    @test_throws DimensionMismatch OrbitalBravaisHopping{1}((1,), zeros(ComplexF64, 2, 3))
    @test_throws ArgumentError OrbitalBravaisHopping{1}((1,), fill(ComplexF64(NaN), 2, 2))
    @test_throws ArgumentError BravaisHopping((0,), 1.0 + 0.0im)
    @test_throws ArgumentError BravaisHopping{1}((0,), 1.0 + 0.0im)
end

@testset "F6 generated multiorbital reference requires exact quadratic structural reconstruction" begin
    lattice = BravaisLattice(Matrix(Diagonal([1.0, 7.0, 8.0])))
    crystal = LatticeCrystalStructure(lattice, [BasisSite(:A, :H, [0.0, 0.0, 0.0])])
    material = Material("F6 two-orbital periodic reference", crystal; species=Dict(:H => lookup_species(:H)))
    reference = Supercell(crystal, (3, 1, 1); periodic=true)
    algebra = FermionAlgebra(4 * nsites(reference))
    onsite = Matrix(Diagonal(ComplexF64[-0.2, -0.2, 0.3, 0.3]))
    onsite[1, 3] = onsite[3, 1] = 0.11
    onsite[2, 4] = onsite[4, 2] = 0.11
    amplitude = Matrix(Diagonal(ComplexF64[-0.6, -0.6, -0.3, -0.3]))
    amplitude[1, 3] = 0.08
    amplitude[2, 4] = 0.08
    hop = OrbitalBravaisHopping((1, 0, 0), amplitude)
    terms = AbstractOperatorExpr[]
    for cell in 0:2
        current = site_index(reference, [cell, 0, 0], 1)
        shifted = site_index(reference, [mod(cell + 1, 3), 0, 0], 1)
        for a in 1:4, b in 1:4
            if !iszero(onsite[a, b])
                push!(terms, onsite[a, b] * c(4 * (current - 1) + a)' * c(4 * (current - 1) + b))
            end
            if !iszero(amplitude[a, b])
                pair = c(4 * (current - 1) + a)' * c(4 * (shifted - 1) + b)
                push!(terms, amplitude[a, b] * pair + conj(amplitude[a, b]) * pair')
            end
        end
    end
    operator = OperatorSum(terms)
    identity_operation = SpaceGroupOperation(Matrix{Int}(I, 3, 3), zeros(3))
    dataset = CrystallographicDataset(spacegroup_number=1, operations=[identity_operation],
        crystallographic_orbits=collect(1:nsites(reference)))
    result = generate_hamiltonian_family(material, reference, algebra; dataset=dataset,
        symmetry_action=SpinfulFermionAction(norbitals=2), candidates=AbstractOperatorExpr[operator],
        basis_labels=[:quadratic], max_body_order=2, max_range=1.01)
    family = structural_family(result)
    recovered = locate_in_family(family, operator)
    @test recovered.in_span
    selection = structural_selection(family, [:quadratic])
    representation = Representation(:f6_spinful_orbitals, FermionFockSpace(algebra.nmodes), algebra,
        FermionOccupationBasis(collect(1:algebra.nmodes)))
    model = instantiate_model(selection, representation, recovered.selected_coefficients)
    @test geometry_context(model).component_layout.norbitals == 2
    mesh = matrix_bravais_momentum_lattice(model, (8, 1, 1), [hop]; onsite=onsite,
        labels=[:o1up, :o1down, :o2up, :o2down])
    @test size(mesh.hamiltonians) == (4, 4, 8)
    @test mesh.hamiltonians[:, :, 1] ≈ onsite + amplitude + amplitude'
    @test_throws ArgumentError matrix_bravais_momentum_lattice(model, (8, 1, 1), [hop];
        onsite=onsite + 0.01 * Matrix{ComplexF64}(I, 4, 4))
    @test_throws ArgumentError matrix_bravais_momentum_lattice(model, (8, 1, 1),
        [OrbitalBravaisHopping((2, 0, 0), amplitude)]; onsite=onsite)
    @test_throws ArgumentError DMFTForwardContext(mesh; temperature=0.5)
end
