#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalEDValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

# Independent full-Fock reference: no calls into the new sector Hamiltonian builder.
function _apply(mask::Int, mode::Int, create::Bool)
    bit = 1 << (mode - 1)
    occupied = !iszero(mask & bit)
    occupied == create && return nothing
    sign = isodd(count_ones(mask & (bit - 1))) ? -1.0 : 1.0
    return (create ? mask | bit : mask & ~bit), sign
end

function _full_reference(specification::P.MultiOrbitalAndersonHamiltonian, axis::P.FermionicMatsubaraAxis)
    h = specification.one_body
    U = specification.interaction.tensor
    nimp = specification.impurity_dimension
    nmodes = size(h, 1)
    dim = 1 << nmodes
    H = zeros(ComplexF64, dim, dim)
    C = [zeros(ComplexF64, dim, dim) for _ in 1:nimp]
    for ket in 0:dim-1
        for a in 1:nimp
            transition = _apply(ket, a, false)
            isnothing(transition) && continue
            output, sign = transition
            C[a][output+1, ket+1] = sign
        end
        for a in 1:nmodes, b in 1:nmodes
            state = _apply(ket, b, false)
            isnothing(state) && continue
            middle, factor = state
            state = _apply(middle, a, true)
            isnothing(state) && continue
            output, sign = state
            H[output+1, ket+1] += h[a, b] * factor * sign
        end
        for a in 1:nimp, b in 1:nimp, c in 1:nimp, d in 1:nimp
            state = _apply(ket, c, false)
            isnothing(state) && continue
            middle, factor = state
            state = _apply(middle, d, false)
            isnothing(state) && continue
            middle, sign = state
            factor *= sign
            state = _apply(middle, b, true)
            isnothing(state) && continue
            middle, sign = state
            factor *= sign
            state = _apply(middle, a, true)
            isnothing(state) && continue
            output, sign = state
            H[output+1, ket+1] += U[a, b, c, d] * factor * sign / 4
        end
    end
    @test H ≈ H' atol=1e-13
    solution = eigen(Hermitian(H))
    beta = axis.beta
    shifted = exp.(-beta .* (solution.values .- minimum(solution.values)))
    probability = shifted ./ sum(shifted)
    D = [solution.vectors' * C[a] * solution.vectors for a in 1:nimp]
    values = zeros(ComplexF64, nimp, nimp, length(axis))
    density = zeros(ComplexF64, nimp, nimp)
    moment0 = zeros(ComplexF64, nimp, nimp)
    for src in 1:dim, tgt in 1:dim, b in 1:nimp, a in 1:nimp
        amplitude = D[a][tgt, src] * conj(D[b][tgt, src])
        iszero(amplitude) && continue
        density[a, b] += probability[src] * amplitude
        spectral_weight = (probability[src] + probability[tgt]) * amplitude
        moment0[a, b] += spectral_weight
        for i in eachindex(axis.values)
            values[a, b, i] += spectral_weight / (axis[i] + solution.values[tgt] - solution.values[src])
        end
    end
    return (; energies=solution.values, green=values, density, moment0, H)
end

function _empty_bath(labels)
    return P.MatrixDiscreteBath(Float64[], zeros(ComplexF64, length(labels), 0); labels=labels)
end

@testset "II.A.2: interacting Kanamori atom and independent full-Fock ED" begin
    axis = P.FermionicMatsubaraAxis(7.0, 12; include_negative=true)
    interaction = P.kanamori_interaction(2; U=4.0, J=0.6)
    bath = _empty_bath(interaction.labels)
    h = ComplexF64[0.12 0 0.17im 0;
                   0 0.12 0 -0.17im;
                   -0.17im 0 0.24 0;
                   0 0.17im 0 0.24]
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=0.7, bath=bath)
    reference = _full_reference(P.anderson_hamiltonian(problem), axis)
    serial = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:sectorized), problem)
    dense = P.solve(P.MultiOrbitalEDSolver(bath_sites=0, backend=:dense), problem)
    @test serial isa P.MultiOrbitalEDResult
    @test serial.metadata[:full_hilbert_dimension] == 16
    @test serial.metadata[:sector_count] == 5
    @test serial.metadata[:max_sector_dimension] == 6
    @test dense.metadata[:sector_count] == 1
    @test sort(vcat([s.energies for s in serial.thermal_state.sectors]...)) ≈ reference.energies atol=1e-12
    @test serial.green_function.values ≈ reference.green atol=5e-12 rtol=5e-12
    @test dense.green_function.values ≈ reference.green atol=5e-12 rtol=5e-12
    @test serial.density_matrix ≈ reference.density atol=5e-12
    @test dense.density_matrix ≈ reference.density atol=5e-12
    @test serial.occupations ≈ real.(diag(reference.density)) atol=5e-12
    @test P.multiorbital_impurity_green(serial.thermal_state, axis).values ≈ serial.green_function.values atol=5e-12
    @test P.matsubara_green(serial.thermal_state, axis).values ≈ serial.green_function.values atol=5e-12
    @test P.matsubara_green(serial.thermal_state, axis; modes=[2, 4]).values ≈ serial.green_function.values[[2, 4], [2, 4], :] atol=5e-12
    @test reference.moment0 ≈ Matrix{ComplexF64}(I, 4, 4) atol=5e-12
    @test serial.density_matrix ≈ serial.density_matrix' atol=1e-12
    @test minimum(eigvals(Hermitian(serial.density_matrix))) >= -1e-12
    @test maximum(real.(serial.occupations)) <= 1 + 1e-12
    # Exactly degenerate, zero-field two-electron Kanamori multiplets.
    atomic_problem = P.MultiOrbitalImpurityProblem(zeros(ComplexF64, 4, 4), interaction, axis; bath=bath)
    atomic_state = P.multiorbital_ed_state(P.anderson_hamiltonian(atomic_problem), P.MultiOrbitalEDSolver(bath_sites=0), 1 / axis.beta)
    expected = sort([4.0 - 3*0.6, 4.0 - 3*0.6, 4.0 - 3*0.6, 4.0 - 0.6, 4.0 - 0.6, 4.0 + 0.6])
    @test atomic_state.sectors[3].energies ≈ expected atol=1e-12
end

@testset "II.A.2: fully complex off-diagonal bath and interacting matrix Green functions" begin
    axis = P.FermionicMatsubaraAxis(5.0, 8; include_negative=true)
    interaction = P.kanamori_interaction(2; U=1.6, J=0.22)
    bath = P.MatrixDiscreteBath([0.3], reshape(ComplexF64[0.35+0.08im, 0.13-0.18im, -0.17+0.04im, 0.19+0.16im], 4, 1); labels=interaction.labels)
    h = ComplexF64[0.2 0.11im 0 0;
                   -0.11im 0.23 0 0;
                   0 0 0.38 0.06;
                   0 0 0.06 0.41]
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=0.45, bath=bath)
    reference = _full_reference(P.anderson_hamiltonian(problem), axis)
    result = P.solve(P.MultiOrbitalEDSolver(), problem)
    @test result.green_function.values ≈ reference.green atol=1e-11 rtol=1e-11
    @test result.density_matrix ≈ reference.density atol=1e-11
    @test any(abs.(result.green_function.values[1, 2, :]) .> 1e-8)
    @test P.dyson_green(result.noninteracting_green_function, result.self_energy).values ≈ result.green_function.values atol=1e-11
    for i in 1:length(axis)
        j = findfirst(z -> isapprox(z, -axis[i]; atol=1e-14), axis.values)
        @test !isnothing(j)
        @test result.green_function[i] ≈ result.green_function[j]' atol=1e-11
        if imag(axis[i]) > 0
            @test minimum(eigvals(Hermitian((result.green_function[i]' - result.green_function[i]) / (2im)))) >= -1e-11
        end
    end
end

@testset "II.A.2: noninteracting resolvent and self-energy" begin
    axis = P.FermionicMatsubaraAxis(4.0, 10)
    labels = [:spin_up, :spin_down]
    interaction = P.SpinOrbitalInteraction(zeros(ComplexF64, 2, 2, 2, 2); labels=labels)
    h = ComplexF64[0.2 0.12+0.03im; 0.12-0.03im -0.1]
    bath = P.MatrixDiscreteBath([-0.4, 0.7], ComplexF64[0.3+0.1im 0.2; 0.1im 0.25-0.12im]; labels=labels)
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=0.15, bath=bath)
    result = P.solve(P.MultiOrbitalEDSolver(), problem)
    @test result.green_function.values ≈ result.noninteracting_green_function.values atol=2e-12
    @test maximum(abs, result.self_energy.values) <= 2e-11
    @test result.occupations ≈ real.(diag(result.density_matrix)) atol=1e-13
    @test all(x -> -1e-12 <= x <= 1+1e-12, result.occupations)
end

@testset "II.A.2: fitted matrix bath and finite-size acceptance guards" begin
    axis = P.FermionicMatsubaraAxis(6.0, 16)
    labels = [:spin_up, :spin_down]
    interaction = P.SpinOrbitalInteraction(zeros(ComplexF64, 2, 2, 2, 2); labels=labels)
    h = ComplexF64[0.1 0; 0 0.2]
    bath = P.MatrixDiscreteBath([0.25], reshape(ComplexF64[0.36+0.07im, -0.11+0.18im], 2, 1); labels=labels)
    target = P.hybridization_function(bath, axis; chemical_potential=0.0)
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; target_hybridization=target)
    solver = P.MultiOrbitalEDSolver(bath_sites=1, bath_fit=P.BathFitOptions(multistart=1, maxiter=40, tol=1e-9))
    result = P.solve(solver, problem; initial_bath=bath)
    @test result.metadata[:initial_bath_used]
    @test result.metadata[:bath_fit_objective] <= 1e-15
    @test result.green_function.values ≈ result.noninteracting_green_function.values atol=2e-11
    @test_throws ArgumentError P.MultiOrbitalEDSolver(backend=:unsupported)
    @test_throws ArgumentError P.solve(P.MultiOrbitalEDSolver(max_full_dimension=4), problem; initial_bath=bath)
    @test_throws ArgumentError P.solve(P.MultiOrbitalEDSolver(max_sector_dimension=1), problem; initial_bath=bath)
    @test_throws ArgumentError P.solve(P.MultiOrbitalEDSolver(max_eigensystem_bytes=100), problem; initial_bath=bath)
    @test_throws ArgumentError P.multiorbital_impurity_green(result.thermal_state, P.FermionicMatsubaraAxis(7.0, 10))
    @test_throws ArgumentError P.solve(solver, P.MultiOrbitalImpurityProblem(h, interaction, axis; bath=bath); initial_bath=bath)
end

end # module MultiorbitalEDValidation
