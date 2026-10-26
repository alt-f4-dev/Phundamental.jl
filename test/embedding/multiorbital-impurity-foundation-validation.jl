#!/usr/bin/env julia

using Test
using LinearAlgebra
using Phundamental

module MultiorbitalImpurityFoundationValidation

using Test
using LinearAlgebra

const P = Main.Phundamental

function _apply_fermion(ket::Int, orbital::Int, creation::Bool)
    bit = 1 << (orbital - 1)
    occupied = (ket & bit) != 0
    occupied == creation && return nothing
    preceding = count_ones(ket & (bit - 1))
    sign = isodd(preceding) ? -1.0 : 1.0
    return (creation ? ket | bit : ket & ~bit), sign
end

function _two_particle_interaction_matrix(tensor)
    n = size(tensor, 1)
    states = [ket for ket in 0:(1 << n)-1 if count_ones(ket) == 2]
    lookup = Dict(ket => i for (i, ket) in enumerate(states))
    matrix = zeros(ComplexF64, length(states), length(states))
    for (column, ket) in enumerate(states), d in 1:n, c in 1:n, b in 1:n, a in 1:n
        result = _apply_fermion(ket, c, false)
        isnothing(result) && continue
        state, factor = result
        result = _apply_fermion(state, d, false)
        isnothing(result) && continue
        state, sign = result
        factor *= sign
        result = _apply_fermion(state, b, true)
        isnothing(result) && continue
        state, sign = result
        factor *= sign
        result = _apply_fermion(state, a, true)
        isnothing(result) && continue
        state, sign = result
        matrix[lookup[state], column] += tensor[a, b, c, d] * factor * sign / 4
    end
    return matrix
end

@testset "II.A.1: antisymmetrized spin-orbital interactions" begin
    U, J = 4.0, 0.6
    interaction = P.kanamori_interaction(2; U=U, J=J)
    tensor = interaction.tensor
    @test size(tensor) == (4, 4, 4, 4)
    @test tensor[1, 2, 1, 2] ≈ U
    @test tensor[1, 3, 1, 3] ≈ U - 3J
    @test tensor[1, 4, 1, 4] ≈ U - 2J
    @test tensor[1, 4, 3, 2] ≈ J
    @test tensor[1, 2, 3, 4] ≈ J
    @test maximum(abs.(tensor .+ permutedims(tensor, (2, 1, 3, 4)))) <= 1e-14
    @test maximum(abs.(tensor .+ permutedims(tensor, (1, 2, 4, 3)))) <= 1e-14
    @test maximum(abs.(tensor .- conj.(permutedims(tensor, (3, 4, 1, 2))))) <= 1e-14
    h2 = _two_particle_interaction_matrix(tensor)
    @test h2 ≈ h2' atol=1e-13
    expected = sort([U-3J, U-3J, U-3J, U-J, U-J, U+J])
    @test eigvals(Hermitian(h2)) ≈ expected atol=1e-12
    malformed = copy(tensor)
    malformed[1, 2, 1, 2] += 0.1
    @test_throws ArgumentError P.SpinOrbitalInteraction(malformed)
end

@testset "II.A.1: causal matrix bath and Anderson resolvent" begin
    axis = P.FermionicMatsubaraAxis(7.0, 24; include_negative=true)
    interaction = P.kanamori_interaction(2; U=2.0, J=0.2)
    names = interaction.labels
    V = ComplexF64[0.4+0.1im 0.2+0.0im 0.0+0.0im;
                   0.0+0.2im 0.3-0.1im 0.0+0.0im;
                   0.2-0.1im 0.1+0.2im 0.3+0.0im;
                   0.0+0.0im 0.1+0.0im 0.4+0.2im]
    bath = P.MatrixDiscreteBath([-0.8, 0.4, 1.2], V; labels=names)
    mu = 0.3
    delta = P.hybridization_function(bath, axis; chemical_potential=mu)
    @test size(delta.values) == (4, 4, length(axis))
    @test P.hybridization_moment(bath) ≈ V * V'
    for i in 1:length(axis)
        direct = V * Diagonal(inv.(axis[i] .+ mu .- bath.energies)) * V'
        @test delta[i] ≈ direct atol=1e-14 rtol=1e-14
        @test delta[1, 2, i] ≈ direct[1, 2] atol=1e-14 rtol=1e-14
        if imag(axis[i]) > 0
            spectral = Hermitian((delta[i]' - delta[i]) / (2im))
            @test minimum(eigvals(spectral)) >= -1e-12
        end
    end
    h = ComplexF64[0.1 0.0 0.0 0.0;
                   0.0 0.2 0.0 0.0;
                   0.0 0.0 0.4 0.15im;
                   0.0 0.0 -0.15im 0.3]
    problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=mu, bath=bath)
    specification = P.anderson_hamiltonian(problem)
    @test specification isa P.MultiOrbitalAndersonHamiltonian
    @test specification.impurity_dimension == 4
    @test specification.interaction === interaction
    onebody = P.impurity_one_body_matrix(problem)
    @test specification.one_body ≈ onebody
    @test size(onebody) == (7, 7)
    @test onebody ≈ onebody'
    G = P.noninteracting_impurity_green(problem)
    for i in 1:length(axis)
        exact = inv(axis[i] * Matrix{ComplexF64}(I, 7, 7) - onebody)[1:4, 1:4]
        @test G[i] ≈ exact atol=1e-12 rtol=1e-12
        @test G[3, 4, i] ≈ exact[3, 4] atol=1e-12 rtol=1e-12
    end
    target_problem = P.MultiOrbitalImpurityProblem(h, interaction, axis; chemical_potential=mu, target_hybridization=delta)
    @test P.noninteracting_impurity_green(target_problem).values ≈ G.values atol=1e-12
    @test_throws ArgumentError P.MultiOrbitalImpurityProblem(h, interaction, axis; bath=bath, target_hybridization=delta)
    @test_throws DimensionMismatch P.MatrixDiscreteBath([-1.0], ones(4, 2); labels=names)
    @test_throws ArgumentError P.MatrixDiscreteBath([-1.0], fill(Inf, 4, 1); labels=names)
end

@testset "II.A.1: analytic matrix bath fit Jacobian and warm start" begin
    axis = P.FermionicMatsubaraAxis(10.0, 16)
    names = [:a, :b]
    V = ComplexF64[0.4+0.1im 0.1+0.2im;
                   0.15-0.2im 0.35+0.05im]
    exact_bath = P.MatrixDiscreteBath([-0.75, 1.1], V; labels=names)
    target = P.hybridization_function(exact_bath, axis; chemical_potential=0.2)
    options = P.BathFitOptions(maxiter=80, tol=1e-10, multistart=2, energy_window=(-2.0, 2.0), weight_power=1.0)
    weights = P.Embedding._matrix_bath_weights(target, options)
    parameters = P.Embedding._matrix_bath_encode(exact_bath)
    residual, jacobian = P.Embedding._matrix_bath_residual_jacobian(parameters, target, 0.2, weights)
    @test maximum(abs, residual) <= 2e-15
    for column in eachindex(parameters)
        shift = 1e-6
        plus = copy(parameters)
        minus = copy(parameters)
        plus[column] += shift
        minus[column] -= shift
        residual_plus, _ = P.Embedding._matrix_bath_residual_jacobian(plus, target, 0.2, weights)
        residual_minus, _ = P.Embedding._matrix_bath_residual_jacobian(minus, target, 0.2, weights)
        @test (residual_plus - residual_minus) / (2 * shift) ≈ jacobian[:, column] atol=1e-7 rtol=1e-6
    end
    fitted = P.fit_bath(target, 2; chemical_potential=0.2, options=options, initial_bath=exact_bath)
    @test fitted isa P.MatrixBathFitResult
    @test fitted.metadata[:warm_start_used]
    @test fitted.objective <= 1e-18
    @test P.hybridization_function(fitted.bath, axis; chemical_potential=0.2).values ≈ target.values atol=1e-9 rtol=1e-9
    rank_one = P.MatrixDiscreteBath([0.3], reshape(ComplexF64[0.35+0.1im, 0.25-0.1im], 2, 1); labels=names)
    rank_one_target = P.hybridization_function(rank_one, axis)
    free_fit = P.fit_matrix_bath(rank_one_target, 1; options=options)
    zero_objective = sum(weights[i] * sum(abs2, view(rank_one_target.values, :, :, i)) for i in 1:length(axis))
    @test free_fit.objective < 1e-3 * zero_objective
    @test maximum(abs, P.hybridization_function(free_fit.bath, axis).values - rank_one_target.values) < 5e-3
    @test_throws ArgumentError P.fit_matrix_bath(target, 2; options=P.BathFitOptions(symmetry=:particle_hole))
    empty_fit = P.fit_bath(target, 0)
    @test length(empty_fit.bath) == 0
    @test isfinite(empty_fit.objective)
end

end
