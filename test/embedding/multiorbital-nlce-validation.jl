using Test
using LinearAlgebra
using Phundamental

const NLF = Phundamental

function _fermion_nlce_fixture(; nsites::Int=3, hopping::Float64=0.4, interaction::Float64=1.7,
                                chemical_potential::Float64=0.25, periodic::Bool=false)
    lattice = NLF.BravaisLattice(reshape([1.0], 1, 1))
    crystal = NLF.LatticeCrystalStructure(lattice, [NLF.BasisSite(:C, :C, [0.0])])
    material = NLF.Material("fermionic NLCE chain", crystal; species=Dict(:C => NLF.AtomicSpecies(:C; mass=1.0)))
    reference = NLF.Supercell(crystal, (nsites,); periodic=periodic)
    spec = NLF.HubbardModel(hopping, interaction; neighbor_cutoff=1.01, chemical_potential=chemical_potential)
    source = NLF.build_model(material, reference, spec)
    bonds = source.parameters[:bonds]
    site_modes = [[2i - 1, 2i] for i in 1:nsites]
    onsite = Dict(i => interaction * NLF.nf(2i - 1) * NLF.nf(2i) -
                    chemical_potential * (NLF.nf(2i - 1) + NLF.nf(2i)) for i in 1:nsites)
    contributions = Pair[]
    for bond in bonds
        i, j = bond.i, bond.j
        hopping_term = reduce(+, ((-hopping) * (NLF.c(2i - 2 + spin)' * NLF.c(2j - 2 + spin) +
                                     NLF.c(2j - 2 + spin)' * NLF.c(2i - 2 + spin)) for spin in 1:2))
        iszero(hopping) || push!(contributions, bond => hopping_term)
    end
    return (; source, reference, material, spec, bonds, site_modes, onsite, contributions)
end

function _nlce_certified(fixture)
    return NLF.certify_fermionic_clusters(fixture.source, fixture.reference;
        site_modes=fixture.site_modes, onsite_terms=fixture.onsite,
        bond_terms=fixture.contributions, bonds=fixture.bonds)
end

@testset "Multiorbital fermionic NLCE: certified open-cluster Hamiltonians" begin
    fixture = _fermion_nlce_fixture()
    decomposition = _nlce_certified(fixture)
    @test length(decomposition.site_modes) == 3
    clusters = NLF.connected_clusters(fixture.reference, fixture.bonds; max_order=3)
    @test !isempty(clusters)
    for cluster in clusters
        induced = NLF.regenerate_induced_fermion_cluster(decomposition, cluster.supercell)
        direct = NLF.build_model(fixture.material, cluster.supercell, fixture.spec; bonds=cluster.bonds)
        actual = NLF.matrix(induced.hamiltonian, induced.representation; sparse=false)
        expected = NLF.matrix(direct.hamiltonian, direct.representation; sparse=false)
        @test actual ≈ expected atol=1e-12
        @test actual ≈ actual' atol=1e-12
        @test induced.representation.algebra.nmodes == 2 * cluster.order
        @test length(induced.parameters[:parent_mode_map]) == 2 * cluster.order
    end
    @test_throws ArgumentError NLF.certify_fermionic_clusters(fixture.source, fixture.reference;
        site_modes=[[1, 2], [3, 4], [5, 5]], onsite_terms=fixture.onsite,
        bond_terms=fixture.contributions, bonds=fixture.bonds)
    @test_throws ArgumentError NLF.certify_fermionic_clusters(fixture.source, fixture.reference;
        site_modes=fixture.site_modes, onsite_terms=fixture.onsite,
        bond_terms=Pair[], bonds=fixture.bonds)
    wrong_onsite = copy(fixture.onsite)
    wrong_onsite[1] = wrong_onsite[1] + NLF.nf(3)
    @test_throws ArgumentError NLF.certify_fermionic_clusters(fixture.source, fixture.reference;
        site_modes=fixture.site_modes, onsite_terms=wrong_onsite,
        bond_terms=fixture.contributions, bonds=fixture.bonds)
    periodic_fixture = _fermion_nlce_fixture(; periodic=true)
    @test_throws ArgumentError _nlce_certified(periodic_fixture)
end

@testset "Multiorbital fermionic NLCE: thermal inclusion-exclusion and atomic limit" begin
    fixture = _fermion_nlce_fixture(; hopping=0.0)
    decomposition = _nlce_certified(fixture)
    beta = 2.0
    evaluator = (model, cluster, index) -> begin
        matrix = NLF.matrix(model.hamiltonian, model.representation; sparse=false)
        levels = eigvals(Hermitian(matrix))
        return -log(sum(exp.(-beta .* levels))) / beta
    end
    problem = NLF.fermionic_nlce_problem(decomposition, evaluator; max_order=3)
    @test_throws ArgumentError NLF.solve(NLF.NLCESolver(), NLF.fermionic_nlce_problem(decomposition, evaluator; max_order=3, max_fock_dimension=8))
    result = NLF.solve(NLF.NLCESolver(), problem)
    single = NLF.build_model(fixture.material, NLF.Supercell(fixture.reference.crystal, (1,); periodic=false), fixture.spec)
    levels = eigvals(Hermitian(NLF.matrix(single.hamiltonian, single.representation; sparse=false)))
    expected = -log(sum(exp.(-beta .* levels))) / beta
    onsite_partition = 1 + 2exp(beta * 0.25) + exp(beta * (2 * 0.25 - 1.7))
    @test expected ≈ -log(onsite_partition) / beta atol=1e-12
    @test length(result.partial_sums) == 3
    @test all(isapprox(value, expected; atol=1e-10) for value in result.partial_sums)
    @test all(isfinite, result.partial_sums)
end

@testset "Multiorbital fermionic NLCE: high-level thermal forward integration" begin
    fixture = _fermion_nlce_fixture(; hopping=0.0)
    decomposition = _nlce_certified(fixture)
    request = NLF.ThermodynamicCurveRequest([0.5, 1.0]; kB=1.0)
    context = NLF.FermionicNLCEForwardContext(decomposition, 3;
        inner_solver=NLF.ThermodynamicForwardSolver(NLF.ExactGibbs()))
    @test NLF.solver_compatibility(NLF.NLCESolver(), fixture.source, request, context).compatible
    output = NLF.forward(NLF.ForwardProblem(fixture.source; solver=NLF.NLCESolver(),
                                            method_context=context, observable=request))
    single = NLF.build_model(fixture.material, NLF.Supercell(fixture.reference.crystal, (1,); periodic=false), fixture.spec)
    direct = NLF.thermal_curve(single, NLF.ExactGibbs(), [0.5, 1.0])
    @test output.observable_result[:, 1] ≈ direct.free_energy atol=1e-10
    @test output.observable_result[:, 2] ≈ direct.internal_energy atol=1e-10
    @test output.observable_result[:, 3] ≈ direct.entropy atol=1e-10
    @test output.observable_result[:, 4] ≈ direct.heat_capacity atol=1e-10
end

@testset "Multiorbital fermionic NLCE: interacting four-mode sites and complex hopping" begin
    lattice = NLF.BravaisLattice(reshape([1.0], 1, 1))
    crystal = NLF.LatticeCrystalStructure(lattice, [NLF.BasisSite(:M, :M, [0.0])])
    material = NLF.Material("two-orbital chain", crystal; species=Dict(:M => NLF.AtomicSpecies(:M; mass=1.0)))
    reference = NLF.Supercell(crystal, (2,); periodic=false)
    backbone = NLF.build_model(material, reference, NLF.HubbardModel(0.2, 0.0; neighbor_cutoff=1.01))
    bonds = backbone.parameters[:bonds]
    site_modes = [[4i-3, 4i-2, 4i-1, 4i] for i in 1:2]
    onsite = Dict{Int,NLF.AbstractOperatorExpr}()
    for site in 1:2
        a,b,c,d = site_modes[site]
        flip = NLF.c(a)' * NLF.c(d)' * NLF.c(b) * NLF.c(c)
        pair = NLF.c(a)' * NLF.c(b)' * NLF.c(d) * NLF.c(c)
        onsite[site] = 1.4 * (NLF.nf(a) * NLF.nf(b) + NLF.nf(c) * NLF.nf(d)) +
                       0.8 * (NLF.nf(a) * NLF.nf(c) + NLF.nf(b) * NLF.nf(d)) +
                       0.15 * (flip + flip') + 0.11 * (pair + pair')
    end
    hopping = Pair[]
    for bond in bonds
        term = 0 * NLF.IdentityOperator()
        for mode_index in 1:4
            left = site_modes[bond.i][mode_index]
            right = site_modes[bond.j][mode_index]
            amplitude = 0.19 + 0.07im
            term += amplitude * (NLF.c(left)' * NLF.c(right)) +
                    conj(amplitude) * (NLF.c(right)' * NLF.c(left))
        end
        push!(hopping, bond => term)
    end
    terms = NLF.AbstractOperatorExpr[values(onsite)..., last.(hopping)...]
    H = reduce(+, terms)
    rep = NLF.Representation(:two_orbital_nlce, NLF.FermionFockSpace(8), NLF.FermionAlgebra(8),
                             NLF.FermionOccupationBasis(collect(1:8)); ordering=collect(1:8))
    model = NLF.ManyBodyModel(rep, H; parameters=Dict{Symbol,Any}(:material => material, :cluster => reference))
    decomposition = NLF.certify_fermionic_clusters(model, reference;
        site_modes=site_modes, onsite_terms=onsite, bond_terms=hopping, bonds=bonds)
    request = NLF.ThermodynamicCurveRequest([0.75]; kB=1.0)
    context = NLF.FermionicNLCEForwardContext(decomposition, 1;
        inner_solver=NLF.ThermodynamicForwardSolver(NLF.ExactGibbs()))
    @test NLF.solver_compatibility(NLF.NLCESolver(), model, request, context).compatible
    @test !NLF.solver_compatibility(NLF.NLCESolver(), model, request, NLF.NLCEForwardContext(1)).compatible
    result = NLF.forward(NLF.ForwardProblem(model; solver=NLF.NLCESolver(), method_context=context, observable=request))
    @test all(isfinite, result.observable_result)
    clusters = NLF.connected_clusters(reference, bonds; max_order=2)
    for cluster in clusters
        generated = NLF.regenerate_induced_fermion_cluster(decomposition, cluster.supercell)
        actual = NLF.matrix(generated.hamiltonian, generated.representation; sparse=false)
        expected = if cluster.order == 2
            NLF.matrix(model.hamiltonian, rep; sparse=false)
        else
            local_rep = NLF.Representation(:one_site, NLF.FermionFockSpace(4), NLF.FermionAlgebra(4),
                                           NLF.FermionOccupationBasis(collect(1:4)))
            first_site = cluster.parent_sites[1]
            a,b,c,d = 1:4
            flip = NLF.c(a)' * NLF.c(d)' * NLF.c(b) * NLF.c(c)
            pair = NLF.c(a)' * NLF.c(b)' * NLF.c(d) * NLF.c(c)
            local_expr = 1.4 * (NLF.nf(a) * NLF.nf(b) + NLF.nf(c) * NLF.nf(d)) +
                         0.8 * (NLF.nf(a) * NLF.nf(c) + NLF.nf(b) * NLF.nf(d)) +
                         0.15 * (flip + flip') + 0.11 * (pair + pair')
            NLF.matrix(local_expr, local_rep; sparse=false)
        end
        @test actual ≈ expected atol=1e-12
        @test actual ≈ actual' atol=1e-12
    end
end
