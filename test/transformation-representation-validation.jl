#!/usr/bin/env julia
module TransformationRepresentationValidation

using Test
using LinearAlgebra
using Printf
using Phundamental
#include(joinpath(@__DIR__, "..", "src/Phundamental.jl"))
#using .Phundamental

const F = Phundamental
const TF = Phundamental.Transformations
const ATOL = 2.0e-10
const RTOL = 2.0e-10
const RESULTS = NamedTuple[]

# =============================================================================
# Independent finite-dimensional reference algebra
# =============================================================================

function kron_all(operators)
    result = ones(ComplexF64, 1, 1)
    for operator in operators
        result = kron(result, ComplexF64.(operator))
    end
    return Matrix{ComplexF64}(result)
end

identity_matrix(dimension::Integer) = Matrix{ComplexF64}(I, Int(dimension), Int(dimension))

function embed_local(local_op::AbstractMatrix, dimensions::AbstractVector{<:Integer}, position::Integer)
    operators = Matrix{ComplexF64}[]
    for i in eachindex(dimensions)
        push!(operators, i == position ? ComplexF64.(local_op) : identity_matrix(dimensions[i]))
    end
    return kron_all(operators)
end

function local_spin_matrices(S::Real; hbar::Real=1.0)
    dimension = Int(round(2 * S + 1))
    Sz = zeros(ComplexF64, dimension, dimension)
    Sp = zeros(ComplexF64, dimension, dimension)
    for level in 0:(dimension - 1)
        m = -S + level
        Sz[level + 1, level + 1] = hbar * m
        if level < dimension - 1
            Sp[level + 2, level + 1] = hbar * sqrt(S * (S + 1) - m * (m + 1))
        end
    end
    Sm = adjoint(Sp)
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2im)
    return (x=Sx, y=Sy, z=Sz, plus=Sp, minus=Sm)
end

function reference_spin_operators(spins::AbstractVector{<:Real}; ordering=collect(1:length(spins)), hbar::Real=1.0)
    dimensions = [Int(round(2 * spins[label] + 1)) for label in ordering]
    x = Vector{Matrix{ComplexF64}}(undef, length(spins))
    y = similar(x)
    z = similar(x)
    plus = similar(x)
    minus = similar(x)
    for site in eachindex(spins)
        position = findfirst(==(site), ordering)
        local_spin = local_spin_matrices(spins[site]; hbar=hbar)
        x[site] = embed_local(local_spin.x, dimensions, position)
        y[site] = embed_local(local_spin.y, dimensions, position)
        z[site] = embed_local(local_spin.z, dimensions, position)
        plus[site] = embed_local(local_spin.plus, dimensions, position)
        minus[site] = embed_local(local_spin.minus, dimensions, position)
    end
    return (x=x, y=y, z=z, plus=plus, minus=minus, dimension=prod(dimensions))
end

function reference_fermion_operators(nmodes::Integer; ordering=collect(1:Int(nmodes)))
    nmodes = Int(nmodes)
    annihilation_local = ComplexF64[0 1; 0 0]
    parity_local = ComplexF64[1 0; 0 -1]
    identity_local = identity_matrix(2)
    annihilation = Vector{Matrix{ComplexF64}}(undef, nmodes)
    for mode in 1:nmodes
        position = findfirst(==(mode), ordering)
        operators = Matrix{ComplexF64}[]
        for p in 1:nmodes
            if p < position
                push!(operators, parity_local)
            elseif p == position
                push!(operators, annihilation_local)
            else
                push!(operators, identity_local)
            end
        end
        annihilation[mode] = kron_all(operators)
    end
    creation = [adjoint(operator) for operator in annihilation]
    number = [creation[i] * annihilation[i] for i in 1:nmodes]
    return (annihilation=annihilation, creation=creation, number=number, dimension=2^nmodes)
end

function local_boson_annihilation(cutoff::Integer)
    cutoff = Int(cutoff)
    operator = zeros(ComplexF64, cutoff + 1, cutoff + 1)
    for n in 1:cutoff
        operator[n, n + 1] = sqrt(n)
    end
    return operator
end

function reference_boson_operators(cutoffs::AbstractVector{<:Integer}; ordering=collect(1:length(cutoffs)))
    dimensions = [Int(cutoffs[label]) + 1 for label in ordering]
    annihilation = Vector{Matrix{ComplexF64}}(undef, length(cutoffs))
    for mode in eachindex(cutoffs)
        position = findfirst(==(mode), ordering)
        local_boson = local_boson_annihilation(cutoffs[mode])
        annihilation[mode] = embed_local(local_boson, dimensions, position)
    end
    creation = [adjoint(operator) for operator in annihilation]
    number = [creation[i] * annihilation[i] for i in eachindex(annihilation)]
    return (annihilation=annihilation, creation=creation, number=number, dimension=prod(dimensions))
end

function reference_displacement(annihilation::AbstractMatrix, alpha::Number)
    return exp(ComplexF64(alpha) * adjoint(annihilation) - conj(ComplexF64(alpha)) * annihilation)
end

function reference_quadratic(coefficient_matrix::AbstractMatrix, annihilation)
    dimension = size(first(annihilation), 1)
    result = zeros(ComplexF64, dimension, dimension)
    for i in axes(coefficient_matrix, 1), j in axes(coefficient_matrix, 2)
        result .+= coefficient_matrix[i, j] .* (adjoint(annihilation[i]) * annihilation[j])
    end
    return result
end

function project_matrix(matrix::AbstractMatrix, indices::AbstractVector{<:Integer})
    return Matrix{ComplexF64}(matrix[indices, indices])
end

function reverse_basis_permutation(dimension::Integer)
    dimension = Int(dimension)
    permutation = zeros(ComplexF64, dimension, dimension)
    for source in 1:dimension
        permutation[dimension - source + 1, source] = 1
    end
    return permutation
end

# =============================================================================
# Symbolic helpers written independently of Transformations.transform
# =============================================================================

function symbolic_sum(terms)
    expressions = F.AbstractOperatorExpr[term for term in terms]
    isempty(expressions) && return 0.0 * F.IdentityOperator()
    length(expressions) == 1 && return only(expressions)
    return F.OperatorSum(expressions)
end

function symbolic_quadratic(coefficient_matrix::AbstractMatrix, statistics::Symbol)
    terms = F.AbstractOperatorExpr[]
    for i in axes(coefficient_matrix, 1), j in axes(coefficient_matrix, 2)
        coefficient = coefficient_matrix[i, j]
        abs(coefficient) <= 10eps(Float64) && continue
        annihilate = statistics === :fermion ? F.c(j) : F.b(j)
        create = statistics === :fermion ? F.c(i)' : F.b(i)'
        push!(terms, coefficient * create * annihilate)
    end
    return symbolic_sum(terms)
end

function symbolic_bogoliubov_annihilator(U::AbstractMatrix, V::AbstractMatrix, mode::Integer, statistics::Symbol)
    terms = F.AbstractOperatorExpr[]
    for k in axes(U, 1)
        annihilate = statistics === :fermion ? F.c(k) : F.b(k)
        create = statistics === :fermion ? F.c(k)' : F.b(k)'
        push!(terms, conj(U[k, mode]) * annihilate)
        sign = statistics === :fermion ? one(V[k, mode]) : -one(V[k, mode])
        push!(terms, sign * V[k, mode] * create)
    end
    return symbolic_sum(terms)
end

function numeric_bogoliubov_annihilator(U::AbstractMatrix, V::AbstractMatrix, annihilation, mode::Integer, statistics::Symbol)
    result = zeros(ComplexF64, size(first(annihilation))...)
    for k in axes(U, 1)
        result .+= conj(U[k, mode]) .* annihilation[k]
        sign = statistics === :fermion ? 1 : -1
        result .+= sign .* V[k, mode] .* adjoint(annihilation[k])
    end
    return result
end

# =============================================================================
# Representation constructors used by the direct symbolic -> numeric route
# =============================================================================

function spin_representation(spins; name::Symbol=:direct_spin)
    nsites = length(spins)
    return F.Representation(name, F.SpinHilbertSpace(collect(spins)), F.SpinAlgebra(nsites), 
                            F.SpinProductBasis(collect(1:nsites)); ordering=collect(1:nsites))
end

function fermion_representation(nmodes::Integer; name::Symbol=:direct_fermion)
    nmodes = Int(nmodes)
    return F.Representation(name, F.FermionFockSpace(nmodes), F.FermionAlgebra(nmodes), 
                            F.FermionOccupationBasis(collect(1:nmodes)); ordering=collect(1:nmodes))
end

function boson_representation(nmodes::Integer; name::Symbol=:direct_boson)
    nmodes = Int(nmodes)
    return F.Representation(name, F.BosonFockSpace(nmodes), F.BosonAlgebra(nmodes),
                            F.BosonOccupationBasis(collect(1:nmodes)); ordering=collect(1:nmodes))
end

function majorana_representation(nmodes::Integer; name::Symbol=:direct_majorana)
    nmodes = Int(nmodes)
    return F.Representation(name, F.FermionFockSpace(nmodes), F.MajoranaAlgebra(2 * nmodes),
                            F.FermionOccupationBasis(collect(1:nmodes)); ordering=collect(1:nmodes))
end

function coordinate_representation(ndof::Integer; name::Symbol=:direct_coordinate)
    ndof = Int(ndof)
    return F.Representation(name, F.CoordinateHilbertSpace(ndof), 
                            F.CoordinateMomentumAlgebra(ndof),
                            F.CoordinateBasis(collect(1:ndof)); ordering=collect(1:ndof))
end

function fermion_boson_representation(nfermions::Integer, nbosons::Integer; name::Symbol=:direct_fermion_boson)
    nfermions = Int(nfermions)
    nbosons = Int(nbosons)
    return F.Representation(
        name,
        F.CompositeStateSpace(F.FermionFockSpace(nfermions), F.BosonFockSpace(nbosons)),
        F.CompositeAlgebra(F.FermionAlgebra(nfermions), F.BosonAlgebra(nbosons)),
        F.CompositeBasis(F.FermionOccupationBasis(collect(1:nfermions)), F.BosonOccupationBasis(collect(1:nbosons))),
    )
end

function spin_boson_representation(spins, nbosons::Integer; name::Symbol=:direct_spin_boson)
    nsites = length(spins)
    nbosons = Int(nbosons)
    return F.Representation(
        name,
        F.CompositeStateSpace(F.SpinHilbertSpace(collect(spins)), F.BosonFockSpace(nbosons)),
        F.CompositeAlgebra(F.SpinAlgebra(nsites), F.BosonAlgebra(nbosons)),
        F.CompositeBasis(F.SpinProductBasis(collect(1:nsites)), F.BosonOccupationBasis(collect(1:nbosons))),
    )
end

function holstein_primakoff_representation(S::Real; name::Symbol=:direct_holstein_primakoff)
    cutoff = Int(round(2 * S))
    ambient = F.BosonFockSpace(1)
    constraints = ((kind=:occupation_bounds, mode=1, minimum=0, maximum=cutoff),)
    physical = F.PhysicalSubspace(ambient; constraints=constraints, dimension=cutoff + 1)
    return F.Representation(
        name,
        ambient,
        F.BosonAlgebra(1),
        F.BosonOccupationBasis([1]);
        physical_subspace=physical,
        constraints=constraints,
        reference_state=:maximally_polarized_spin_vacuum,
        ordering=[1],
    )
end

function dyson_maleev_representation(S::Real; name::Symbol=:direct_dyson_maleev)
    cutoff = Int(round(2 * S))
    bosons = F.BosonFockSpace(1)
    ambient = F.BiorthogonalSpace(bosons)
    constraints = ((kind=:occupation_bounds, mode=1, minimum=0, maximum=cutoff),)
    physical = F.PhysicalSubspace(ambient; constraints=constraints, dimension=cutoff + 1)
    basis = F.BosonOccupationBasis([1])
    return F.Representation(
        name,
        ambient,
        F.BosonAlgebra(1),
        F.BiorthogonalBasis(basis, basis);
        physical_subspace=physical,
        constraints=constraints,
        reference_state=:maximally_polarized_spin_vacuum,
        ordering=[1],
    )
end

function schwinger_boson_representation(S::Real; name::Symbol=:direct_schwinger_boson)
    total = Int(round(2 * S))
    ambient = F.BosonFockSpace(2)
    constraints = ((kind=:schwinger_occupancy, site=1, modes=(1, 2), total=total),)
    physical = F.PhysicalSubspace(ambient; constraints=constraints, dimension=total + 1)
    gauge = F.GaugeStructure(:U1, (F.nb(1) + F.nb(2) - total * F.IdentityOperator(),); description="validation Schwinger constraint")
    return F.Representation(
        name,
        ambient,
        F.BosonAlgebra(2),
        F.BosonOccupationBasis([1, 2]);
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=[1, 2],
    )
end

function abriko_representation(; name::Symbol=:direct_abrikosov)
    ambient = F.FermionFockSpace(2)
    constraints = ((kind=:single_occupancy, site=1, modes=(1, 2), total=1),)
    physical = F.PhysicalSubspace(ambient; constraints=constraints, dimension=2)
    gauge = F.GaugeStructure(:U1, (F.nf(1) + F.nf(2) - F.IdentityOperator(),); description="validation single-occupancy constraint")
    return F.Representation(
        name,
        ambient,
        F.FermionAlgebra(2),
        F.FermionOccupationBasis([1, 2]);
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=[1, 2],
    )
end

# =============================================================================
# Validation helpers
# =============================================================================

function dense_hamiltonian(model)
    realization = F.hamiltonian_matrix(model; sparse=false, closure_tol=1.0e-11)
    return Matrix{ComplexF64}(realization.matrix), realization.diagnostics
end

function sorted_eigenvalues(matrix::AbstractMatrix)
    values = ComplexF64.(eigvals(Matrix{ComplexF64}(matrix)))
    sort!(values; by=value -> (real(value), imag(value)))
    return values
end

function sorted_solver_energies(model)
    solution = F.solve(F.ExactDiagonalization(closure_tol=1.0e-11), model)
    values = ComplexF64.(solution.energies)
    sort!(values; by=value -> (real(value), imag(value)))
    return values, solution
end

matrix_error(actual, reference) = maximum(abs, actual .- reference)
spectrum_error(actual, reference) = maximum(abs, sorted_eigenvalues(actual) .- sorted_eigenvalues(reference))

function validate_numeric_model(label::AbstractString, model, reference; expect_biorthogonal::Bool=false)
    capabilities = F.numerical_capabilities(model)
    @test capabilities.numerically_realizable
    actual, diagnostics = dense_hamiltonian(model)
    @test size(actual) == size(reference)
    @test isapprox(actual, reference; atol=ATOL, rtol=RTOL)
    energies, solution = sorted_solver_energies(model)
    reference_energies = sorted_eigenvalues(reference)
    @test isapprox(energies, reference_energies; atol=ATOL, rtol=RTOL)
    @test F.isbiorthogonal(solution) == expect_biorthogonal
    return (
        label=String(label),
        matrix_error=matrix_error(actual, reference),
        spectrum_error=maximum(abs, energies .- reference_energies),
        truncation_leakage=get(diagnostics, :truncation_leakage, 0.0),
    )
end

function validate_case!(name::Symbol, source_result, target_result, transformed_result; note::AbstractString="")
    push!(RESULTS, (
        case=name,
        source_matrix_error=isnothing(source_result) ? missing : source_result.matrix_error,
        source_spectrum_error=isnothing(source_result) ? missing : source_result.spectrum_error,
        target_matrix_error=target_result.matrix_error,
        transformed_matrix_error=transformed_result.matrix_error,
        transformed_spectrum_error=transformed_result.spectrum_error,
        truncation_leakage=transformed_result.truncation_leakage,
        note=String(note),
    ))
end

function transform_with_reselected_truncation(T, source_model, truncation; expected_name::Symbol)
    transformed = F.transform(T, source_model).model
    @test transformed.representation.name == expected_name
    capabilities = F.numerical_capabilities(transformed)
    @test capabilities.transported_truncation
    @test capabilities.requires_truncation
    @test !capabilities.numerically_realizable
    @test_throws ArgumentError F.computational_basis(transformed)
    reselected = F.with_truncation(transformed, truncation)
    @test F.numerical_capabilities(reselected).numerically_realizable
    return reselected
end

function print_summary()
    println("\nTRANSFORMATION REPRESENTATION NUMERICS VALIDATION")
    println(repeat("-", 133))
    @printf("%-28s %12s %12s %12s %12s %12s %12s  %s\n", "case", "src|dH|", "src|dE|", "tgt|dH|", "xfm|dH|", "xfm|dE|", "leakage", "note")
    println(repeat("-", 133))
    format_error(value) = ismissing(value) ? "n/a" : @sprintf("%.3e", value)
    for result in RESULTS
        @printf(
            "%-28s %12s %12s %12s %12s %12s %12s  %s\n",
            String(result.case),
            format_error(result.source_matrix_error),
            format_error(result.source_spectrum_error),
            format_error(result.target_matrix_error),
            format_error(result.transformed_matrix_error),
            format_error(result.transformed_spectrum_error),
            format_error(result.truncation_leakage),
            result.note,
        )
    end
    println(repeat("-", 133))
end

# =============================================================================
# Registry coverage and transformation-by-transformation validation
# =============================================================================

@testset "Transformation representation numerics" begin
    expected_registry = Set([
        :fourier,
        :bogoliubov,
        :particle_hole,
        :majorana,
        :jordan_wigner,
        :holstein_primakoff,
        :dyson_maleev,
        :schwinger_boson,
        :abrikosov_fermion,
        :slave_particle,
        :coordinate_ladder,
        :bosonic_displacement,
        :lang_firsov,
        :spin_polaron,
    ])
    @test Set(TF.available_transformations()) == expected_registry

    @testset "Particle-hole" begin
        epsilon = 0.73
        drive = 0.21
        fermion = reference_fermion_operators(1)
        C = fermion.annihilation[1]
        N = fermion.number[1]
        identity = identity_matrix(2)
        source_reference = epsilon * N + drive * (C + adjoint(C))
        target_reference = epsilon * (identity - N) + drive * (C + adjoint(C))

        source_symbolic = epsilon * F.nf(1) + drive * (F.c(1) + F.c(1)')
        source_model = F.ManyBodyModel(fermion_representation(1; name=:particle_source), source_symbolic)
        source_result = validate_numeric_model("particle-hole/source", source_model, source_reference)

        target_symbolic = epsilon * (F.IdentityOperator() - F.nf(1)) + drive * (F.c(1)' + F.c(1))
        target_model = F.ManyBodyModel(fermion_representation(1; name=:hole_direct), target_symbolic)
        target_result = validate_numeric_model("particle-hole/target-direct", target_model, target_reference)

        transformation = F.ParticleHoleTransformation()
        capabilities = F.transformation_capabilities(transformation, source_model)
        @test capabilities.applicable
        @test capabilities.numerical.numerically_realizable
        transformed = F.transform(transformation, source_model).model
        @test transformed.representation.name == :hole
        transformed_result = validate_numeric_model("particle-hole/transformed", transformed, target_reference)
        @test isapprox(sorted_eigenvalues(source_reference), sorted_eigenvalues(target_reference); atol=ATOL, rtol=RTOL)
        validate_case!(:particle_hole, source_result, target_result, transformed_result)
    end

    @testset "Majorana" begin
        epsilon = 0.61
        drive = 0.17
        fermion = reference_fermion_operators(1)
        C = fermion.annihilation[1]
        N = fermion.number[1]
        identity = identity_matrix(2)
        gamma1 = C + adjoint(C)
        gamma2 = -im * (C - adjoint(C))
        source_reference = epsilon * (N - 0.5 * identity) + drive * (C + adjoint(C))
        target_reference = (im * epsilon / 2) * gamma1 * gamma2 + drive * gamma1

        source_symbolic = epsilon * (F.nf(1) - 0.5 * F.IdentityOperator()) + drive * (F.c(1) + F.c(1)')
        source_model = F.ManyBodyModel(fermion_representation(1; name=:majorana_source), source_symbolic)
        source_result = validate_numeric_model("majorana/source", source_model, source_reference)

        target_symbolic = (im * epsilon / 2) * F.γ(1) * F.γ(2) + drive * F.γ(1)
        target_model = F.ManyBodyModel(majorana_representation(1), target_symbolic)
        target_result = validate_numeric_model("majorana/target-direct", target_model, target_reference)

        transformed = F.transform(F.MajoranaTransformation(), source_model).model
        @test transformed.representation.name == :majorana
        transformed_result = validate_numeric_model("majorana/transformed", transformed, target_reference)
        @test isapprox(source_reference, target_reference; atol=ATOL, rtol=RTOL)
        validate_case!(:majorana, source_result, target_result, transformed_result)
    end

    @testset "Jordan-Wigner" begin
        h1 = 0.41
        h2 = -0.27
        coupling = 0.33
        spins = reference_spin_operators([0.5, 0.5])
        source_reference = h1 * spins.z[1] + h2 * spins.z[2] + coupling * spins.x[1] * spins.x[2]
        source_symbolic = h1 * F.Sz(1) + h2 * F.Sz(2) + coupling * F.Sx(1) * F.Sx(2)
        source_model = F.ManyBodyModel(spin_representation([1//2, 1//2]; name=:jw_source), source_symbolic)
        source_result = validate_numeric_model("jordan-wigner/source", source_model, source_reference)

        fermion = reference_fermion_operators(2)
        C1, C2 = fermion.annihilation
        N1, N2 = fermion.number
        identity = identity_matrix(4)
        target_reference = h1 * (N1 - 0.5 * identity) + h2 * (N2 - 0.5 * identity) +
                           (coupling / 4) * (adjoint(C1) - C1) * (adjoint(C2) + C2)
        target_symbolic = h1 * (F.nf(1) - 0.5 * F.IdentityOperator()) + h2 * (F.nf(2) - 0.5 * F.IdentityOperator()) +
                          (coupling / 4) * (F.c(1)' - F.c(1)) * (F.c(2)' + F.c(2))
        target_model = F.ManyBodyModel(fermion_representation(2; name=:jw_target_direct), target_symbolic)
        target_result = validate_numeric_model("jordan-wigner/target-direct", target_model, target_reference)

        transformed = F.transform(F.JordanWigner([1, 2]), source_model).model
        @test transformed.representation.name == :jordan_wigner_fermion
        transformed_result = validate_numeric_model("jordan-wigner/transformed", transformed, target_reference)
        @test isapprox(source_reference, target_reference; atol=ATOL, rtol=RTOL)
        validate_case!(:jordan_wigner, source_result, target_result, transformed_result)
    end

    @testset "Holstein-Primakoff" begin
        S = 1.0
        hz = 0.43
        hx = 0.29
        anisotropy = 0.11
        spin = reference_spin_operators([S])
        source_reference = hz * spin.z[1] + hx * spin.x[1] + anisotropy * spin.z[1] * spin.z[1]
        source_symbolic = hz * F.Sz(1) + hx * F.Sx(1) + anisotropy * F.Sz(1) * F.Sz(1)
        source_model = F.ManyBodyModel(spin_representation([S]; name=:hp_source), source_symbolic)
        source_result = validate_numeric_model("holstein-primakoff/source", source_model, source_reference)

        boson = reference_boson_operators([Int(round(2 * S))])
        B = boson.annihilation[1]
        N = boson.number[1]
        identity = identity_matrix(size(B, 1))
        root = Diagonal(ComplexF64.(sqrt.(max.(0.0, 2 * S .- collect(0:Int(round(2 * S)))))))
        Z = S * identity - N
        Plus = Matrix(root) * B
        Minus = adjoint(B) * Matrix(root)
        X = (Plus + Minus) / 2
        target_reference = hz * Z + hx * X + anisotropy * Z * Z

        root_symbolic = TF.SqrtNumberFactor(:boson, 1, 2 * S, -1.0)
        Zsymbolic = S * F.IdentityOperator() - F.nb(1)
        Xsymbolic = (root_symbolic * F.b(1) + F.b(1)' * root_symbolic) / 2
        target_symbolic = hz * Zsymbolic + hx * Xsymbolic + anisotropy * Zsymbolic * Zsymbolic
        target_model = F.ManyBodyModel(holstein_primakoff_representation(S), target_symbolic)
        target_result = validate_numeric_model("holstein-primakoff/target-direct", target_model, target_reference)

        transformed = F.transform(F.HolsteinPrimakoff(), source_model).model
        @test transformed.representation.name == :holstein_primakoff
        transformed_result = validate_numeric_model("holstein-primakoff/transformed", transformed, target_reference)
        permutation = reverse_basis_permutation(size(source_reference, 1))
        @test isapprox(target_reference, permutation * source_reference * adjoint(permutation); atol=ATOL, rtol=RTOL)
        validate_case!(:holstein_primakoff, source_result, target_result, transformed_result)
    end

    @testset "Dyson-Maleev" begin
        S = 1.0
        hz = 0.37
        hx = 0.31
        anisotropy = 0.09
        spin = reference_spin_operators([S])
        source_reference = hz * spin.z[1] + hx * spin.x[1] + anisotropy * spin.z[1] * spin.z[1]
        source_symbolic = hz * F.Sz(1) + hx * F.Sx(1) + anisotropy * F.Sz(1) * F.Sz(1)
        source_model = F.ManyBodyModel(spin_representation([S]; name=:dm_source), source_symbolic)
        source_result = validate_numeric_model("dyson-maleev/source", source_model, source_reference)

        boson = reference_boson_operators([Int(round(2 * S))])
        B = boson.annihilation[1]
        N = boson.number[1]
        identity = identity_matrix(size(B, 1))
        Z = S * identity - N
        Plus = sqrt(2 * S) * B
        Minus = sqrt(2 * S) * adjoint(B) * (identity - N / (2 * S))
        X = (Plus + Minus) / 2
        target_reference = hz * Z + hx * X + anisotropy * Z * Z

        Zsymbolic = S * F.IdentityOperator() - F.nb(1)
        Plussymbolic = sqrt(2 * S) * F.b(1)
        Minussymbolic = sqrt(2 * S) * F.b(1)' * (F.IdentityOperator() - F.nb(1) / (2 * S))
        Xsymbolic = (Plussymbolic + Minussymbolic) / 2
        target_symbolic = hz * Zsymbolic + hx * Xsymbolic + anisotropy * Zsymbolic * Zsymbolic
        target_model = F.ManyBodyModel(dyson_maleev_representation(S), target_symbolic)
        target_result = validate_numeric_model("dyson-maleev/target-direct", target_model, target_reference; expect_biorthogonal=true)

        transformed = F.transform(F.DysonMaleev(), source_model).model
        @test transformed.representation.name == :dyson_maleev
        capabilities = F.numerical_capabilities(transformed)
        @test capabilities.biorthogonal
        @test !(:lanczos in capabilities.compatible_solvers)
        @test_throws ArgumentError F.solve(F.LanczosSolver(nev=1), transformed)
        transformed_result = validate_numeric_model("dyson-maleev/transformed", transformed, target_reference; expect_biorthogonal=true)
        @test isapprox(sorted_eigenvalues(source_reference), sorted_eigenvalues(target_reference); atol=ATOL, rtol=RTOL)
        validate_case!(:dyson_maleev, source_result, target_result, transformed_result; note="biorthogonal ED")
    end

    @testset "Schwinger boson" begin
        S = 1.0
        hz = 0.39
        hx = 0.24
        spin = reference_spin_operators([S])
        source_reference = hz * spin.z[1] + hx * spin.x[1]
        source_symbolic = hz * F.Sz(1) + hx * F.Sx(1)
        source_model = F.ManyBodyModel(spin_representation([S]; name=:schwinger_source), source_symbolic)
        source_result = validate_numeric_model("schwinger/source", source_model, source_reference)

        total = Int(round(2 * S))
        boson_full = reference_boson_operators([total, total])
        A, B = boson_full.annihilation
        Na, Nb = boson_full.number
        Zfull = (Na - Nb) / 2
        Xfull = (adjoint(A) * B + adjoint(B) * A) / 2
        dimension_local = total + 1
        indices = [na * dimension_local + (total - na) + 1 for na in 0:total]
        target_reference = project_matrix(hz * Zfull + hx * Xfull, indices)

        target_symbolic = hz * (F.nb(1) - F.nb(2)) / 2 + hx * (F.b(1)' * F.b(2) + F.b(2)' * F.b(1)) / 2
        target_model = F.ManyBodyModel(schwinger_boson_representation(S), target_symbolic)
        target_result = validate_numeric_model("schwinger/target-direct", target_model, target_reference)

        transformed = F.transform(F.SchwingerBoson(), source_model).model
        @test transformed.representation.name == :schwinger_boson
        transformed_result = validate_numeric_model("schwinger/transformed", transformed, target_reference)
        @test isapprox(source_reference, target_reference; atol=ATOL, rtol=RTOL)
        validate_case!(:schwinger_boson, source_result, target_result, transformed_result)
    end

    @testset "Abrikosov fermion" begin
        hz = 0.47
        hx = 0.21
        spin = reference_spin_operators([0.5])
        source_reference = hz * spin.z[1] + hx * spin.x[1]
        source_symbolic = hz * F.Sz(1) + hx * F.Sx(1)
        source_model = F.ManyBodyModel(spin_representation([1//2]; name=:abrikosov_source), source_symbolic)
        source_result = validate_numeric_model("abrikosov/source", source_model, source_reference)

        fermion_full = reference_fermion_operators(2)
        Cup, Cdn = fermion_full.annihilation
        Nup, Ndn = fermion_full.number
        Zfull = (Nup - Ndn) / 2
        Xfull = (adjoint(Cup) * Cdn + adjoint(Cdn) * Cup) / 2
        target_reference = project_matrix(hz * Zfull + hx * Xfull, [2, 3])
        target_symbolic = hz * (F.nf(1) - F.nf(2)) / 2 + hx * (F.c(1)' * F.c(2) + F.c(2)' * F.c(1)) / 2
        target_model = F.ManyBodyModel(abriko_representation(), target_symbolic)
        target_result = validate_numeric_model("abrikosov/target-direct", target_model, target_reference)

        transformed = F.transform(F.AbrikosovFermion(), source_model).model
        @test transformed.representation.name == :abrikosov_fermion
        transformed_result = validate_numeric_model("abrikosov/transformed", transformed, target_reference)
        @test isapprox(source_reference, target_reference; atol=ATOL, rtol=RTOL)
        validate_case!(:abrikosov_fermion, source_result, target_result, transformed_result)
    end

    @testset "Generic slave-particle" begin
        hz = 0.36
        hx = 0.19
        spin = reference_spin_operators([0.5])
        source_reference = hz * spin.z[1] + hx * spin.x[1]
        source_symbolic = hz * F.Sz(1) + hx * F.Sx(1)
        source_model = F.ManyBodyModel(spin_representation([1//2]; name=:slave_source), source_symbolic)
        source_result = validate_numeric_model("slave/source", source_model, source_reference)

        target_representation = abriko_representation(name=:slave_validation_target)
        target_symbolic = hz * (F.nf(1) - F.nf(2)) / 2 + hx * (F.c(1)' * F.c(2) + F.c(2)' * F.c(1)) / 2
        target_model = F.ManyBodyModel(target_representation, target_symbolic)
        fermion_full = reference_fermion_operators(2)
        Cup, Cdn = fermion_full.annihilation
        Nup, Ndn = fermion_full.number
        target_reference = project_matrix(hz * (Nup - Ndn) / 2 + hx * (adjoint(Cup) * Cdn + adjoint(Cdn) * Cup) / 2, [2, 3])
        target_result = validate_numeric_model("slave/target-direct", target_model, target_reference)

        function generator_map(operator)
            plus = F.c(1)' * F.c(2)
            minus = F.c(2)' * F.c(1)
            if operator isa F.SpinPlus
                return plus
            elseif operator isa F.SpinMinus
                return minus
            elseif operator isa F.SpinZ
                return (F.nf(1) - F.nf(2)) / 2
            elseif operator isa F.SpinX
                return (plus + minus) / 2
            elseif operator isa F.SpinY
                return (plus - minus) / (2im)
            end
            throw(ArgumentError("unsupported validation slave-particle generator $(typeof(operator))"))
        end

        transformation = TF.SlaveParticleTransformation(
            target_representation,
            generator_map;
            domain_predicate=model -> model.representation.algebra isa F.SpinAlgebra,
            label=:slave_particle,
        )
        transformed = F.transform(transformation, source_model).model
        @test transformed.representation.name == :slave_validation_target
        transformed_result = validate_numeric_model("slave/transformed", transformed, target_reference)
        @test isapprox(source_reference, target_reference; atol=ATOL, rtol=RTOL)
        validate_case!(:slave_particle, source_result, target_result, transformed_result)
    end

    @testset "Fermionic Fourier" begin
        coefficient_matrix = ComplexF64[0.70 0.20 + 0.10im; 0.20 - 0.10im 1.10]
        Fourier = ComplexF64[1 1; 1 -1] / sqrt(2)
        fermion = reference_fermion_operators(2)
        source_reference = reference_quadratic(coefficient_matrix, fermion.annihilation)
        source_symbolic = symbolic_quadratic(coefficient_matrix, :fermion)
        source_model = F.ManyBodyModel(fermion_representation(2; name=:fourier_fermion_source), source_symbolic)
        source_result = validate_numeric_model("fourier-fermion/source", source_model, source_reference)

        target_coefficients = Fourier * coefficient_matrix * adjoint(Fourier)
        target_reference = reference_quadratic(target_coefficients, fermion.annihilation)
        target_symbolic = symbolic_quadratic(target_coefficients, :fermion)
        target_model = F.ManyBodyModel(fermion_representation(2; name=:fourier_fermion_direct), target_symbolic)
        target_result = validate_numeric_model("fourier-fermion/target-direct", target_model, target_reference)

        transformation = F.FourierTransformation(Fourier; statistics=:fermion)
        transformed = F.transform(transformation, source_model).model
        @test transformed.representation.name == :fermion_fourier
        transformed_result = validate_numeric_model("fourier-fermion/transformed", transformed, target_reference)
        @test isapprox(sorted_eigenvalues(source_reference), sorted_eigenvalues(target_reference); atol=ATOL, rtol=RTOL)
        validate_case!(:fourier_fermion, source_result, target_result, transformed_result)
    end

    @testset "Bosonic Fourier" begin
        cutoff = 4
        coefficient_matrix = ComplexF64[0.85 0.16; 0.16 1.25]
        Fourier = ComplexF64[1 1; 1 -1] / sqrt(2)
        boson = reference_boson_operators([cutoff, cutoff])
        source_reference = reference_quadratic(coefficient_matrix, boson.annihilation)
        source_symbolic = symbolic_quadratic(coefficient_matrix, :boson)
        source_untruncated = F.ManyBodyModel(boson_representation(2; name=:fourier_boson_source), source_symbolic)
        source_model = F.with_truncation(source_untruncated, F.NumericalTruncation([cutoff, cutoff]))
        source_result = validate_numeric_model("fourier-boson/source", source_model, source_reference)

        target_coefficients = Fourier * coefficient_matrix * adjoint(Fourier)
        target_reference = reference_quadratic(target_coefficients, boson.annihilation)
        target_symbolic = symbolic_quadratic(target_coefficients, :boson)
        target_untruncated = F.ManyBodyModel(boson_representation(2; name=:fourier_boson_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation([cutoff, cutoff]))
        target_result = validate_numeric_model("fourier-boson/target-direct", target_model, target_reference)

        transformation = F.FourierTransformation(Fourier; statistics=:boson)
        transformed = transform_with_reselected_truncation(
            transformation,
            source_model,
            F.NumericalTruncation([cutoff, cutoff]);
            expected_name=:boson_fourier,
        )
        transformed_result = validate_numeric_model("fourier-boson/transformed", transformed, target_reference)
        validate_case!(:fourier_boson, source_result, target_result, transformed_result; note="fixed target-basis cutoff")
    end

    @testset "Fermionic Bogoliubov" begin
        xi = 0.74
        Delta = 0.23
        theta = 0.31
        u = cos(theta)
        v = sin(theta)
        U = ComplexF64[u 0; 0 u]
        V = ComplexF64[0 v; -v 0]
        fermion = reference_fermion_operators(2)
        C1, C2 = fermion.annihilation
        identity = identity_matrix(4)
        source_reference = xi * (fermion.number[1] + fermion.number[2] - identity) + Delta * (adjoint(C1) * adjoint(C2) + C2 * C1)
        source_symbolic = xi * (F.nf(1) + F.nf(2) - F.IdentityOperator()) + Delta * (F.c(1)' * F.c(2)' + F.c(2) * F.c(1))
        source_model = F.ManyBodyModel(fermion_representation(2; name=:bog_fermion_source), source_symbolic)
        source_result = validate_numeric_model("bogoliubov-fermion/source", source_model, source_reference)

        a1_symbolic = symbolic_bogoliubov_annihilator(U, V, 1, :fermion)
        a2_symbolic = symbolic_bogoliubov_annihilator(U, V, 2, :fermion)
        target_symbolic = xi * (a1_symbolic' * a1_symbolic + a2_symbolic' * a2_symbolic - F.IdentityOperator()) +
                          Delta * (a1_symbolic' * a2_symbolic' + a2_symbolic * a1_symbolic)
        a1 = numeric_bogoliubov_annihilator(U, V, fermion.annihilation, 1, :fermion)
        a2 = numeric_bogoliubov_annihilator(U, V, fermion.annihilation, 2, :fermion)
        target_reference = xi * (adjoint(a1) * a1 + adjoint(a2) * a2 - identity) + Delta * (adjoint(a1) * adjoint(a2) + a2 * a1)
        target_model = F.ManyBodyModel(fermion_representation(2; name=:bog_fermion_direct), target_symbolic)
        target_result = validate_numeric_model("bogoliubov-fermion/target-direct", target_model, target_reference)

        transformation = F.BogoliubovTransformation(U, V; statistics=:fermion)
        transformed = F.transform(transformation, source_model).model
        @test transformed.representation.name == :fermionic_quasiparticle
        transformed_result = validate_numeric_model("bogoliubov-fermion/transformed", transformed, target_reference)
        @test isapprox(sorted_eigenvalues(source_reference), sorted_eigenvalues(target_reference); atol=ATOL, rtol=RTOL)
        validate_case!(:bogoliubov_fermion, source_result, target_result, transformed_result)
    end

    @testset "Bosonic Bogoliubov" begin
        cutoff = 10
        omega = 0.91
        drive = 0.07
        r = 0.22
        U = reshape(ComplexF64[cosh(r)], 1, 1)
        V = reshape(ComplexF64[sinh(r)], 1, 1)
        boson = reference_boson_operators([cutoff])
        B = boson.annihilation[1]
        source_reference = omega * boson.number[1] + drive * (B + adjoint(B))
        source_symbolic = omega * F.nb(1) + drive * (F.b(1) + F.b(1)')
        source_untruncated = F.ManyBodyModel(boson_representation(1; name=:bog_boson_source), source_symbolic)
        source_model = F.with_truncation(source_untruncated, F.NumericalTruncation(cutoff))
        source_result = validate_numeric_model("bogoliubov-boson/source", source_model, source_reference)

        a_symbolic = symbolic_bogoliubov_annihilator(U, V, 1, :boson)
        target_symbolic = omega * a_symbolic' * a_symbolic + drive * (a_symbolic + a_symbolic')
        u = real(U[1, 1])
        v = real(V[1, 1])
        projected_bbdag = Diagonal(ComplexF64.(collect(1:(cutoff + 1))))
        target_reference = omega * (u^2 * boson.number[1] - u * v * (adjoint(B) * adjoint(B) + B * B) + v^2 * projected_bbdag) +
                           drive * ((u - v) * B + (u - v) * adjoint(B))
        target_untruncated = F.ManyBodyModel(boson_representation(1; name=:bog_boson_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation(cutoff))
        target_result = validate_numeric_model("bogoliubov-boson/target-direct", target_model, target_reference)

        transformation = F.BogoliubovTransformation(U, V; statistics=:boson)
        transformed = transform_with_reselected_truncation(
            transformation,
            source_model,
            F.NumericalTruncation(cutoff);
            expected_name=:bosonic_quasiparticle,
        )
        transformed_result = validate_numeric_model("bogoliubov-boson/transformed", transformed, target_reference)
        validate_case!(:bogoliubov_boson, source_result, target_result, transformed_result; note="fixed target-basis cutoff")
    end

    @testset "Coordinate-ladder" begin
        mass = 1.30
        omega = 0.82
        cutoff = 8
        source_symbolic = F.pop(1) * F.pop(1) / (2 * mass) + (mass * omega^2 / 2) * F.xop(1) * F.xop(1)
        source_model = F.ManyBodyModel(coordinate_representation(1), source_symbolic)
        source_capabilities = F.numerical_capabilities(source_model)
        @test source_capabilities.coordinate_discretization_required
        @test !source_capabilities.numerically_realizable
        @test_throws ArgumentError F.hamiltonian_matrix(source_model; sparse=false)

        boson = reference_boson_operators([cutoff])
        identity = identity_matrix(cutoff + 1)
        target_reference = omega * (boson.number[1] + 0.5 * identity)
        target_symbolic = omega * (F.nb(1) + 0.5 * F.IdentityOperator())
        target_untruncated = F.ManyBodyModel(boson_representation(1; name=:coordinate_target_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation(cutoff))
        target_result = validate_numeric_model("coordinate-ladder/target-direct", target_model, target_reference)

        transformation = F.CoordinateLadder([mass], [omega])
        capabilities = F.transformation_capabilities(transformation, source_model)
        @test capabilities.applicable
        @test capabilities.numerical.requires_truncation
        transformed_raw = F.transform(transformation, source_model).model
        @test transformed_raw.representation.name == :bosonic_ladder
        @test !F.numerical_capabilities(transformed_raw).transported_truncation
        transformed = F.with_truncation(transformed_raw, F.NumericalTruncation(cutoff))
        transformed_result = validate_numeric_model("coordinate-ladder/transformed", transformed, target_reference)
        validate_case!(
            :coordinate_ladder,
            nothing,
            target_result,
            transformed_result;
            note="raw x,p finite realization intentionally unavailable",
        )
    end

    @testset "Bosonic displacement" begin
        cutoff = 10
        omega = 0.88
        drive = 0.09
        alpha = 0.23 + 0.07im
        boson = reference_boson_operators([cutoff])
        B = boson.annihilation[1]
        identity = identity_matrix(cutoff + 1)
        source_reference = omega * boson.number[1] + drive * (B + adjoint(B))
        source_symbolic = omega * F.nb(1) + drive * (F.b(1) + F.b(1)')
        source_untruncated = F.ManyBodyModel(boson_representation(1; name=:displacement_source), source_symbolic)
        source_model = F.with_truncation(source_untruncated, F.NumericalTruncation(cutoff))
        source_result = validate_numeric_model("displacement/source", source_model, source_reference)

        shifted = B - alpha * identity
        target_reference = omega * adjoint(shifted) * shifted + drive * (shifted + adjoint(shifted))
        shifted_symbolic = F.b(1) - alpha * F.IdentityOperator()
        target_symbolic = omega * shifted_symbolic' * shifted_symbolic + drive * (shifted_symbolic + shifted_symbolic')
        target_untruncated = F.ManyBodyModel(boson_representation(1; name=:displacement_target_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation(cutoff))
        target_result = validate_numeric_model("displacement/target-direct", target_model, target_reference)

        transformation = F.BosonicDisplacement([alpha])
        transformed = transform_with_reselected_truncation(
            transformation,
            source_model,
            F.NumericalTruncation(cutoff);
            expected_name=:displaced_boson,
        )
        transformed_result = validate_numeric_model("displacement/transformed", transformed, target_reference)
        validate_case!(:bosonic_displacement, source_result, target_result, transformed_result; note="fixed target-basis cutoff")
    end

    @testset "Lang-Firsov" begin
        cutoff = 9
        omega = 1.07
        coupling = 0.24
        epsilon = 0.39
        tunneling = 0.08
        lambda = coupling / omega
        fermion = reference_fermion_operators(1)
        boson = reference_boson_operators([cutoff])
        C = kron(fermion.annihilation[1], identity_matrix(cutoff + 1))
        Nf = adjoint(C) * C
        B = kron(identity_matrix(2), boson.annihilation[1])
        Nb = adjoint(B) * B
        source_reference = omega * Nb + coupling * Nf * (B + adjoint(B)) + epsilon * Nf + tunneling * (C + adjoint(C))
        source_symbolic = omega * F.nb(1) + coupling * F.nf(1) * (F.b(1) + F.b(1)') + epsilon * F.nf(1) + tunneling * (F.c(1) + F.c(1)')
        source_untruncated = F.ManyBodyModel(fermion_boson_representation(1, 1; name=:lang_firsov_source), source_symbolic)
        source_model = F.with_truncation(source_untruncated, F.NumericalTruncation(cutoff))
        source_result = validate_numeric_model("lang-firsov/source", source_model, source_reference)

        Dminus_local = reference_displacement(boson.annihilation[1], -lambda)
        Dplus_local = reference_displacement(boson.annihilation[1], lambda)
        Dminus = kron(identity_matrix(2), Dminus_local)
        Dplus = kron(identity_matrix(2), Dplus_local)
        target_reference = omega * Nb + (epsilon - coupling^2 / omega) * Nf + tunneling * (C * Dminus + adjoint(C) * Dplus)
        target_symbolic = omega * F.nb(1) + (epsilon - coupling^2 / omega) * F.nf(1) +
                          tunneling * (
                              F.c(1) * TF.BosonicDisplacementFactor(1, -lambda) +
                              F.c(1)' * TF.BosonicDisplacementFactor(1, lambda)
                          )
        target_untruncated = F.ManyBodyModel(fermion_boson_representation(1, 1; name=:lang_firsov_target_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation(cutoff))
        target_result = validate_numeric_model("lang-firsov/target-direct", target_model, target_reference)

        transformation = F.LangFirsov([lambda])
        transformed = transform_with_reselected_truncation(
            transformation,
            source_model,
            F.NumericalTruncation(cutoff);
            expected_name=:lang_firsov_polaron,
        )
        transformed_result = validate_numeric_model("lang-firsov/transformed", transformed, target_reference)
        validate_case!(:lang_firsov, source_result, target_result, transformed_result; note="fixed target-basis cutoff")
    end

    @testset "Spin-polaron" begin
        cutoff = 9
        omega = 0.96
        coupling = 0.20
        hz = 0.31
        hx = 0.11
        eta = coupling / omega
        spin = reference_spin_operators([0.5])
        boson = reference_boson_operators([cutoff])
        identity_spin = identity_matrix(2)
        identity_boson = identity_matrix(cutoff + 1)
        Sz = kron(spin.z[1], identity_boson)
        Sp = kron(spin.plus[1], identity_boson)
        Sm = kron(spin.minus[1], identity_boson)
        Sx = kron(spin.x[1], identity_boson)
        B = kron(identity_spin, boson.annihilation[1])
        Nb = adjoint(B) * B
        source_reference = omega * Nb + coupling * Sz * (B + adjoint(B)) + hz * Sz + hx * Sx
        source_symbolic = omega * F.nb(1) + coupling * F.Sz(1) * (F.b(1) + F.b(1)') + hz * F.Sz(1) + hx * F.Sx(1)
        source_untruncated = F.ManyBodyModel(spin_boson_representation([1//2], 1; name=:spin_polaron_source), source_symbolic)
        source_model = F.with_truncation(source_untruncated, F.NumericalTruncation(cutoff))
        source_result = validate_numeric_model("spin-polaron/source", source_model, source_reference)

        Dplus = kron(identity_spin, reference_displacement(boson.annihilation[1], eta))
        Dminus = kron(identity_spin, reference_displacement(boson.annihilation[1], -eta))
        target_reference = omega * Nb - (coupling^2 / omega) * Sz * Sz + hz * Sz + (hx / 2) * (Sp * Dplus + Sm * Dminus)
        target_symbolic = omega * F.nb(1) - (coupling^2 / omega) * F.Sz(1) * F.Sz(1) + hz * F.Sz(1) +
                          (hx / 2) * (F.Sp(1) * TF.BosonicDisplacementFactor(1, eta) + F.Sm(1) * TF.BosonicDisplacementFactor(1, -eta))
        target_untruncated = F.ManyBodyModel(spin_boson_representation([1//2], 1; name=:spin_polaron_target_direct), target_symbolic)
        target_model = F.with_truncation(target_untruncated, F.NumericalTruncation(cutoff))
        target_result = validate_numeric_model("spin-polaron/target-direct", target_model, target_reference)

        transformation = F.SpinPolaron([eta])
        transformed = transform_with_reselected_truncation(
            transformation,
            source_model,
            F.NumericalTruncation(cutoff);
            expected_name=:spin_polaron,
        )
        transformed_result = validate_numeric_model("spin-polaron/transformed", transformed, target_reference)
        validate_case!(:spin_polaron, source_result, target_result, transformed_result; note="fixed target-basis cutoff")
    end
end

print_summary()

end
