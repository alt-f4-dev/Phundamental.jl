#!/usr/bin/env julia

module BosonicTruncationConvergenceValidation

using Test
using LinearAlgebra
using Printf
using Phundamental
#include(joinpath(@__DIR__, "..", "src/Phundamental.jl"))
#using .Phundamental

const F = Phundamental
const TF = Phundamental.Transformations

# =============================================================================
# Configuration
# =============================================================================

function parse_cutoffs(value::AbstractString)
    values = sort(unique(parse.(Int, split(value, ','))))
    isempty(values) && throw(ArgumentError("at least one cutoff is required"))
    all(>=(1), values) || throw(ArgumentError("all cutoffs must be positive"))
    return values
end

const CUTOFFS = parse_cutoffs(get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_CUTOFFS", "4,6,8,10,12,16,20,24,28"))
const REFERENCE_CUTOFF = parse(Int, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_REFERENCE_CUTOFF", "48"))
const NLOW = parse(Int, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_NLOW", "6"))
const ENERGY_TOL = parse(Float64, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_ENERGY_TOL", "1e-8"))
const OBSERVABLE_TOL = parse(Float64, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_OBSERVABLE_TOL", "1e-7"))
const PIPELINE_TOL = parse(Float64, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_PIPELINE_TOL", "2e-10"))
const REQUIRED_STABLE_STEPS = parse(Int, get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_STABLE_STEPS", "2"))
const STRICT = lowercase(get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_STRICT", "false")) in ("1", "true", "yes", "on")
const SAVE_CSV = lowercase(get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_CSV", "false")) in ("1", "true", "yes", "on")
const OUTPUT_PATH = get(ENV, "PHUNDAMENTAL_BOSON_CONVERGENCE_OUTPUT", joinpath(pwd(), "bosonic-truncation-convergence-results.csv"))

REFERENCE_CUTOFF > maximum(CUTOFFS) || throw(ArgumentError("reference cutoff must exceed every benchmark cutoff"))
NLOW > 0 || throw(ArgumentError("NLOW must be positive"))
REQUIRED_STABLE_STEPS > 0 || throw(ArgumentError("required stable steps must be positive"))
REQUIRED_STABLE_STEPS <= length(CUTOFFS) || throw(ArgumentError("required stable steps exceed the number of cutoffs"))

const RESULTS = NamedTuple[]

# =============================================================================
# Independent finite-dimensional reference algebra
# =============================================================================

identity_matrix(dimension::Integer) = Matrix{ComplexF64}(I, Int(dimension), Int(dimension))

function local_boson_annihilation(cutoff::Integer)
    cutoff = Int(cutoff)
    operator = zeros(ComplexF64, cutoff + 1, cutoff + 1)
    for n in 1:cutoff
        operator[n, n + 1] = sqrt(n)
    end
    return operator
end

function reference_boson_operators(cutoffs::AbstractVector{<:Integer})
    dimensions = Int.(cutoffs) .+ 1
    annihilation = Matrix{ComplexF64}[]
    for mode in eachindex(cutoffs)
        operators = Matrix{ComplexF64}[]
        for position in eachindex(cutoffs)
            local_op = position == mode ? local_boson_annihilation(cutoffs[position]) : identity_matrix(dimensions[position])
            push!(operators, local_op)
        end
        result = ones(ComplexF64, 1, 1)
        for operator in operators
            result = kron(result, operator)
        end
        push!(annihilation, Matrix{ComplexF64}(result))
    end
    creation = [adjoint(operator) for operator in annihilation]
    number = [creation[i] * annihilation[i] for i in eachindex(annihilation)]
    return (annihilation=annihilation, creation=creation, number=number, dimension=prod(dimensions))
end

function reference_fermion_operators(nmodes::Integer)
    nmodes = Int(nmodes)
    annihilation_local = ComplexF64[0 1; 0 0]
    parity_local = ComplexF64[1 0; 0 -1]
    identity_local = identity_matrix(2)
    annihilation = Matrix{ComplexF64}[]
    for mode in 1:nmodes
        result = ones(ComplexF64, 1, 1)
        for position in 1:nmodes
            local_op = position < mode ? parity_local : position == mode ? annihilation_local : identity_local
            result = kron(result, local_op)
        end
        push!(annihilation, Matrix{ComplexF64}(result))
    end
    creation = [adjoint(operator) for operator in annihilation]
    number = [creation[i] * annihilation[i] for i in eachindex(annihilation)]
    return (annihilation=annihilation, creation=creation, number=number, dimension=2^nmodes)
end

function reference_spin_half_operators()
    plus = ComplexF64[0 0; 1 0]
    minus = adjoint(plus)
    z = ComplexF64[-0.5 0; 0 0.5]
    x = (plus + minus) / 2
    return (x=x, z=z, plus=plus, minus=minus)
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

function hermitian_eigensystem(matrix::AbstractMatrix)
    H = Matrix{ComplexF64}(matrix)
    H = 0.5 .* (H .+ adjoint(H))
    eig = eigen(Hermitian(H))
    return Float64.(real.(eig.values)), Matrix{ComplexF64}(eig.vectors)
end

function low_energies(matrix::AbstractMatrix, nlow::Integer=NLOW)
    values, _ = hermitian_eigensystem(matrix)
    return values[1:min(Int(nlow), length(values))]
end

function ground_expectation(matrix::AbstractMatrix, observable::AbstractMatrix)
    _, vectors = hermitian_eigensystem(matrix)
    psi = view(vectors, :, 1)
    return real(dot(psi, observable * psi))
end

function analytic_two_mode_energies(frequencies::AbstractVector{<:Real}, nlow::Integer)
    max_occupation = max(2 * Int(nlow), 8)
    energies = Float64[]
    for n1 in 0:max_occupation, n2 in 0:max_occupation
        push!(energies, frequencies[1] * n1 + frequencies[2] * n2)
    end
    sort!(energies)
    return energies[1:Int(nlow)]
end

# =============================================================================
# Representation constructors and symbolic helpers
# =============================================================================

function boson_representation(nmodes::Integer; name::Symbol=:convergence_boson)
    nmodes = Int(nmodes)
    return F.Representation(
        name,
        F.BosonFockSpace(nmodes),
        F.BosonAlgebra(nmodes),
        F.BosonOccupationBasis(collect(1:nmodes));
        ordering=collect(1:nmodes),
    )
end

function coordinate_representation(ndof::Integer; name::Symbol=:convergence_coordinate)
    ndof = Int(ndof)
    return F.Representation(
        name,
        F.CoordinateHilbertSpace(ndof),
        F.CoordinateMomentumAlgebra(ndof),
        F.CoordinateBasis(collect(1:ndof));
        ordering=collect(1:ndof),
    )
end

function fermion_boson_representation(nfermions::Integer, nbosons::Integer; name::Symbol=:convergence_fermion_boson)
    nfermions = Int(nfermions)
    nbosons = Int(nbosons)
    return F.Representation(
        name,
        F.CompositeStateSpace(F.FermionFockSpace(nfermions), F.BosonFockSpace(nbosons)),
        F.CompositeAlgebra(F.FermionAlgebra(nfermions), F.BosonAlgebra(nbosons)),
        F.CompositeBasis(F.FermionOccupationBasis(collect(1:nfermions)), F.BosonOccupationBasis(collect(1:nbosons))),
    )
end

function spin_boson_representation(spins, nbosons::Integer; name::Symbol=:convergence_spin_boson)
    nbosons = Int(nbosons)
    nsites = length(spins)
    return F.Representation(
        name,
        F.CompositeStateSpace(F.SpinHilbertSpace(collect(spins)), F.BosonFockSpace(nbosons)),
        F.CompositeAlgebra(F.SpinAlgebra(nsites), F.BosonAlgebra(nbosons)),
        F.CompositeBasis(F.SpinProductBasis(collect(1:nsites)), F.BosonOccupationBasis(collect(1:nbosons))),
    )
end

function symbolic_sum(terms)
    expressions = F.AbstractOperatorExpr[term for term in terms]
    isempty(expressions) && return 0.0 * F.IdentityOperator()
    length(expressions) == 1 && return only(expressions)
    return F.OperatorSum(expressions)
end

function symbolic_quadratic(coefficient_matrix::AbstractMatrix)
    terms = F.AbstractOperatorExpr[]
    for i in axes(coefficient_matrix, 1), j in axes(coefficient_matrix, 2)
        coefficient = coefficient_matrix[i, j]
        abs(coefficient) <= 10eps(Float64) && continue
        push!(terms, coefficient * F.b(i)' * F.b(j))
    end
    return symbolic_sum(terms)
end

function symbolic_bogoliubov_annihilator(U::AbstractMatrix, V::AbstractMatrix, mode::Integer)
    terms = F.AbstractOperatorExpr[]
    for k in axes(U, 1)
        push!(terms, conj(U[k, mode]) * F.b(k))
        push!(terms, -V[k, mode] * F.b(k)')
    end
    return symbolic_sum(terms)
end

function with_boson_cutoff(model, cutoff::Integer)
    return F.with_truncation(model, F.NumericalTruncation(Int(cutoff)); record=false)
end

function transform_with_reselected_cutoff(transformation, source_model, cutoff::Integer)
    raw = F.transform(transformation, source_model).model
    capabilities = F.numerical_capabilities(raw)
    if capabilities.transported_truncation
        @test capabilities.requires_truncation
        @test !capabilities.numerically_realizable
    end
    return with_boson_cutoff(raw, cutoff)
end

function dense_hamiltonian(model)
    realized = F.hamiltonian_matrix(model; sparse=false, closure_tol=1.0e-11)
    return Matrix{ComplexF64}(realized.matrix), realized.diagnostics
end

function solve_dense(model)
    solver = F.ExactDiagonalization(closure_tol=1.0e-11)
    return F.solve(solver, model)
end

function solution_low_energies(solution, nlow::Integer=NLOW)
    count = min(Int(nlow), length(solution.energies))
    return Float64.(real.(solution.energies[1:count]))
end

function symbolic_ground_expectation(solution, observable)
    realized = F.realize(observable, solution.basis; sparse=false)
    psi = view(solution.right_states, :, 1)
    return real(dot(psi, realized.matrix * psi))
end

function max_abs_difference(a::AbstractVector, b::AbstractVector)
    count = min(length(a), length(b))
    count > 0 || return Inf
    return maximum(abs.(a[1:count] .- b[1:count]))
end

function matrix_error(actual::AbstractMatrix, reference::AbstractMatrix)
    size(actual) == size(reference) || return Inf
    return maximum(abs, actual .- reference)
end

# =============================================================================
# Reference problems
# =============================================================================

const FOURIER_K = ComplexF64[0.85 0.16; 0.16 1.25]
const FOURIER_U = ComplexF64[1 1; 1 -1] / sqrt(2)
const FOURIER_TARGET_K = FOURIER_U * FOURIER_K * adjoint(FOURIER_U)
const FOURIER_REFERENCE = analytic_two_mode_energies(sort(real.(eigvals(Hermitian(FOURIER_K)))), NLOW)

const BOG_OMEGA = 0.91
const BOG_PAIRING = 0.23
const BOG_R = 0.17
const BOG_U = reshape(ComplexF64[cosh(BOG_R)], 1, 1)
const BOG_V = reshape(ComplexF64[sinh(BOG_R)], 1, 1)
const BOG_EXACT_FREQUENCY = sqrt(BOG_OMEGA^2 - BOG_PAIRING^2)
const BOG_EXACT_E0 = (BOG_EXACT_FREQUENCY - BOG_OMEGA) / 2
const BOG_REFERENCE = [BOG_EXACT_E0 + n * BOG_EXACT_FREQUENCY for n in 0:(NLOW - 1)]
const BOG_DIAGONAL_R = 0.5 * atanh(BOG_PAIRING / BOG_OMEGA)
const BOG_TARGET_N_REFERENCE = sinh(BOG_DIAGONAL_R - BOG_R)^2

const DISP_OMEGA = 0.88
const DISP_DRIVE = 0.21
const DISP_ALPHA = 0.13
const DISP_EXACT_E0 = -(DISP_DRIVE^2) / DISP_OMEGA
const DISP_REFERENCE = [DISP_EXACT_E0 + n * DISP_OMEGA for n in 0:(NLOW - 1)]
const DISP_TARGET_N_REFERENCE = abs2(DISP_ALPHA - DISP_DRIVE / DISP_OMEGA)

const LF_OMEGA = 1.07
const LF_COUPLING = 0.24
const LF_EPSILON = 0.39
const LF_TUNNELING = 0.08
const LF_LAMBDA = LF_COUPLING / LF_OMEGA

const SP_OMEGA = 0.96
const SP_COUPLING = 0.20
const SP_HZ = 0.31
const SP_HX = 0.11
const SP_ETA = SP_COUPLING / SP_OMEGA

const COORD_MASS = 1.30
const COORD_OMEGA = 0.82
const COORD_QUARTIC = 0.035

function lang_firsov_independent_source(cutoff::Integer)
    fermion = reference_fermion_operators(1)
    boson = reference_boson_operators([cutoff])
    identity_f = identity_matrix(2)
    identity_b = identity_matrix(cutoff + 1)
    C = kron(fermion.annihilation[1], identity_b)
    Nf = adjoint(C) * C
    B = kron(identity_f, boson.annihilation[1])
    Nb = adjoint(B) * B
    H = LF_OMEGA * Nb + LF_COUPLING * Nf * (B + adjoint(B)) + LF_EPSILON * Nf + LF_TUNNELING * (C + adjoint(C))
    return H
end

function lang_firsov_independent_target(cutoff::Integer)
    fermion = reference_fermion_operators(1)
    boson = reference_boson_operators([cutoff])
    identity_f = identity_matrix(2)
    identity_b = identity_matrix(cutoff + 1)
    C = kron(fermion.annihilation[1], identity_b)
    Nf = adjoint(C) * C
    B = kron(identity_f, boson.annihilation[1])
    Nb = adjoint(B) * B
    Dminus = kron(identity_f, reference_displacement(boson.annihilation[1], -LF_LAMBDA))
    Dplus = kron(identity_f, reference_displacement(boson.annihilation[1], LF_LAMBDA))
    H = LF_OMEGA * Nb + (LF_EPSILON - LF_COUPLING^2 / LF_OMEGA) * Nf
    H .+= LF_TUNNELING .* (C * Dminus + adjoint(C) * Dplus)
    return H, Nb
end

function spin_polaron_independent_source(cutoff::Integer)
    spin = reference_spin_half_operators()
    boson = reference_boson_operators([cutoff])
    identity_s = identity_matrix(2)
    identity_b = identity_matrix(cutoff + 1)
    Sz = kron(spin.z, identity_b)
    Sx = kron(spin.x, identity_b)
    B = kron(identity_s, boson.annihilation[1])
    Nb = adjoint(B) * B
    H = SP_OMEGA * Nb + SP_COUPLING * Sz * (B + adjoint(B)) + SP_HZ * Sz + SP_HX * Sx
    return H
end

function spin_polaron_independent_target(cutoff::Integer)
    spin = reference_spin_half_operators()
    boson = reference_boson_operators([cutoff])
    identity_s = identity_matrix(2)
    identity_b = identity_matrix(cutoff + 1)
    Sz = kron(spin.z, identity_b)
    Sp = kron(spin.plus, identity_b)
    Sm = kron(spin.minus, identity_b)
    B = kron(identity_s, boson.annihilation[1])
    Nb = adjoint(B) * B
    Dplus = kron(identity_s, reference_displacement(boson.annihilation[1], SP_ETA))
    Dminus = kron(identity_s, reference_displacement(boson.annihilation[1], -SP_ETA))
    H = SP_OMEGA * Nb - (SP_COUPLING^2 / SP_OMEGA) * Sz * Sz + SP_HZ * Sz
    H .+= (SP_HX / 2) .* (Sp * Dplus + Sm * Dminus)
    return H, Nb
end

function coordinate_independent_target(cutoff::Integer)
    boson = reference_boson_operators([cutoff])
    B = boson.annihilation[1]
    xscale = sqrt(1 / (2 * COORD_MASS * COORD_OMEGA))
    pscale = sqrt(COORD_MASS * COORD_OMEGA / 2)
    X = xscale * (B + adjoint(B))
    P = -im * pscale * (B - adjoint(B))
    H = P * P / (2 * COORD_MASS) + (COORD_MASS * COORD_OMEGA^2 / 2) * X * X + COORD_QUARTIC * X^4
    return H, X * X
end

const LF_REFERENCE_MATRIX = lang_firsov_independent_source(REFERENCE_CUTOFF)
const LF_REFERENCE = low_energies(LF_REFERENCE_MATRIX)
const LF_TARGET_REFERENCE_MATRIX, LF_TARGET_N_MATRIX = lang_firsov_independent_target(REFERENCE_CUTOFF)
const LF_TARGET_N_REFERENCE = ground_expectation(LF_TARGET_REFERENCE_MATRIX, LF_TARGET_N_MATRIX)

const SP_REFERENCE_MATRIX = spin_polaron_independent_source(REFERENCE_CUTOFF)
const SP_REFERENCE = low_energies(SP_REFERENCE_MATRIX)
const SP_TARGET_REFERENCE_MATRIX, SP_TARGET_N_MATRIX = spin_polaron_independent_target(REFERENCE_CUTOFF)
const SP_TARGET_N_REFERENCE = ground_expectation(SP_TARGET_REFERENCE_MATRIX, SP_TARGET_N_MATRIX)

const COORD_REFERENCE_MATRIX, COORD_X2_MATRIX = coordinate_independent_target(REFERENCE_CUTOFF)
const COORD_REFERENCE = low_energies(COORD_REFERENCE_MATRIX)
const COORD_X2_REFERENCE = ground_expectation(COORD_REFERENCE_MATRIX, COORD_X2_MATRIX)

# =============================================================================
# Case builders
# =============================================================================

function fourier_case(cutoff::Integer)
    source_symbolic = symbolic_quadratic(FOURIER_K)
    source_untruncated = F.ManyBodyModel(boson_representation(2; name=:fourier_source), source_symbolic)
    source_model = with_boson_cutoff(source_untruncated, cutoff)

    target_symbolic = symbolic_quadratic(FOURIER_TARGET_K)
    target_untruncated = F.ManyBodyModel(boson_representation(2; name=:fourier_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.FourierTransformation(FOURIER_U; statistics=:boson)
    transformed = transform_with_reselected_cutoff(transformation, source_model, cutoff)
    observable = F.nb(1) + F.nb(2)
    return source_model, target_model, transformed, observable, FOURIER_REFERENCE, 0.0, :analytic
end

function bogoliubov_case(cutoff::Integer)
    source_symbolic = BOG_OMEGA * F.nb(1) + (BOG_PAIRING / 2) * (F.b(1)' * F.b(1)' + F.b(1) * F.b(1))
    source_untruncated = F.ManyBodyModel(boson_representation(1; name=:bogoliubov_source), source_symbolic)
    source_model = with_boson_cutoff(source_untruncated, cutoff)

    a = symbolic_bogoliubov_annihilator(BOG_U, BOG_V, 1)
    target_symbolic = BOG_OMEGA * a' * a + (BOG_PAIRING / 2) * (a' * a' + a * a)
    target_untruncated = F.ManyBodyModel(boson_representation(1; name=:bogoliubov_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.BogoliubovTransformation(BOG_U, BOG_V; statistics=:boson)
    transformed = transform_with_reselected_cutoff(transformation, source_model, cutoff)
    observable = F.nb(1)
    return source_model, target_model, transformed, observable, BOG_REFERENCE, BOG_TARGET_N_REFERENCE, :analytic
end

function displacement_case(cutoff::Integer)
    source_symbolic = DISP_OMEGA * F.nb(1) + DISP_DRIVE * (F.b(1) + F.b(1)')
    source_untruncated = F.ManyBodyModel(boson_representation(1; name=:displacement_source), source_symbolic)
    source_model = with_boson_cutoff(source_untruncated, cutoff)

    shifted = F.b(1) - DISP_ALPHA * F.IdentityOperator()
    target_symbolic = DISP_OMEGA * shifted' * shifted + DISP_DRIVE * (shifted + shifted')
    target_untruncated = F.ManyBodyModel(boson_representation(1; name=:displacement_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.BosonicDisplacement([DISP_ALPHA])
    transformed = transform_with_reselected_cutoff(transformation, source_model, cutoff)
    observable = F.nb(1)
    return source_model, target_model, transformed, observable, DISP_REFERENCE, DISP_TARGET_N_REFERENCE, :analytic
end

function lang_firsov_case(cutoff::Integer)
    source_symbolic = LF_OMEGA * F.nb(1) + LF_COUPLING * F.nf(1) * (F.b(1) + F.b(1)') + LF_EPSILON * F.nf(1)
    source_symbolic += LF_TUNNELING * (F.c(1) + F.c(1)')
    source_untruncated = F.ManyBodyModel(fermion_boson_representation(1, 1; name=:lang_firsov_source), source_symbolic)
    source_model = with_boson_cutoff(source_untruncated, cutoff)

    target_symbolic = LF_OMEGA * F.nb(1) + (LF_EPSILON - LF_COUPLING^2 / LF_OMEGA) * F.nf(1)
    target_symbolic += LF_TUNNELING * (
        F.c(1) * TF.BosonicDisplacementFactor(1, -LF_LAMBDA) + F.c(1)' * TF.BosonicDisplacementFactor(1, LF_LAMBDA)
    )
    target_untruncated = F.ManyBodyModel(fermion_boson_representation(1, 1; name=:lang_firsov_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.LangFirsov([LF_LAMBDA])
    transformed = transform_with_reselected_cutoff(transformation, source_model, cutoff)
    observable = F.nb(1)
    return source_model, target_model, transformed, observable, LF_REFERENCE, LF_TARGET_N_REFERENCE, :high_cutoff_independent
end

function spin_polaron_case(cutoff::Integer)
    source_symbolic = SP_OMEGA * F.nb(1) + SP_COUPLING * F.Sz(1) * (F.b(1) + F.b(1)') + SP_HZ * F.Sz(1) + SP_HX * F.Sx(1)
    source_untruncated = F.ManyBodyModel(spin_boson_representation([1//2], 1; name=:spin_polaron_source), source_symbolic)
    source_model = with_boson_cutoff(source_untruncated, cutoff)

    target_symbolic = SP_OMEGA * F.nb(1) - (SP_COUPLING^2 / SP_OMEGA) * F.Sz(1) * F.Sz(1) + SP_HZ * F.Sz(1)
    target_symbolic += (SP_HX / 2) * (
        F.Sp(1) * TF.BosonicDisplacementFactor(1, SP_ETA) + F.Sm(1) * TF.BosonicDisplacementFactor(1, -SP_ETA)
    )
    target_untruncated = F.ManyBodyModel(spin_boson_representation([1//2], 1; name=:spin_polaron_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.SpinPolaron([SP_ETA])
    transformed = transform_with_reselected_cutoff(transformation, source_model, cutoff)
    observable = F.nb(1)
    return source_model, target_model, transformed, observable, SP_REFERENCE, SP_TARGET_N_REFERENCE, :high_cutoff_independent
end

function coordinate_ladder_case(cutoff::Integer)
    x = F.xop(1)
    p = F.pop(1)
    source_symbolic = p * p / (2 * COORD_MASS) + (COORD_MASS * COORD_OMEGA^2 / 2) * x * x + COORD_QUARTIC * x * x * x * x
    source_model = F.ManyBodyModel(coordinate_representation(1; name=:coordinate_source), source_symbolic)

    xscale = sqrt(1 / (2 * COORD_MASS * COORD_OMEGA))
    pscale = sqrt(COORD_MASS * COORD_OMEGA / 2)
    xladder = xscale * (F.b(1) + F.b(1)')
    pladder = -im * pscale * (F.b(1) - F.b(1)')
    target_symbolic = pladder * pladder / (2 * COORD_MASS)
    target_symbolic += (COORD_MASS * COORD_OMEGA^2 / 2) * xladder * xladder
    target_symbolic += COORD_QUARTIC * xladder * xladder * xladder * xladder
    target_untruncated = F.ManyBodyModel(boson_representation(1; name=:coordinate_target_direct), target_symbolic)
    target_model = with_boson_cutoff(target_untruncated, cutoff)

    transformation = F.CoordinateLadder([COORD_MASS], [COORD_OMEGA])
    transformed_raw = F.transform(transformation, source_model).model
    transformed = with_boson_cutoff(transformed_raw, cutoff)
    observable = xladder * xladder
    return nothing, target_model, transformed, observable, COORD_REFERENCE, COORD_X2_REFERENCE, :high_cutoff_independent
end

const CASE_BUILDERS = (
    (:fourier_boson, fourier_case),
    (:bogoliubov_boson, bogoliubov_case),
    (:bosonic_displacement, displacement_case),
    (:lang_firsov, lang_firsov_case),
    (:spin_polaron, spin_polaron_case),
    (:coordinate_ladder_control, coordinate_ladder_case),
)

# =============================================================================
# Benchmark execution
# =============================================================================

function benchmark_case!(case_name::Symbol, builder, cutoff::Integer)
    source_model, target_model, transformed, observable, reference, observable_reference, reference_kind = builder(cutoff)

    source_solution = nothing
    target_solution = nothing
    transformed_solution = nothing
    elapsed = @elapsed begin
        source_solution = isnothing(source_model) ? nothing : solve_dense(source_model)
        target_solution = solve_dense(target_model)
        transformed_solution = solve_dense(transformed)
    end

    target_matrix, _ = dense_hamiltonian(target_model)
    transformed_matrix, transformed_diagnostics = dense_hamiltonian(transformed)
    @test size(target_matrix) == size(transformed_matrix)
    @test isapprox(transformed_matrix, target_matrix; atol=PIPELINE_TOL, rtol=PIPELINE_TOL)

    source_low = isnothing(source_solution) ? Float64[] : solution_low_energies(source_solution)
    target_low = solution_low_energies(target_solution)
    transformed_low = solution_low_energies(transformed_solution)
    reference_low = Float64.(reference[1:min(length(reference), NLOW)])

    @test isapprox(transformed_low, target_low; atol=PIPELINE_TOL, rtol=PIPELINE_TOL)

    transformed_observable = symbolic_ground_expectation(transformed_solution, observable)
    target_observable = symbolic_ground_expectation(target_solution, observable)
    @test isapprox(transformed_observable, target_observable; atol=PIPELINE_TOL, rtol=PIPELINE_TOL)

    source_low_error = isnothing(source_solution) ? missing : max_abs_difference(source_low, reference_low)
    target_low_error = max_abs_difference(transformed_low, reference_low)
    source_e0_error = isnothing(source_solution) ? missing : abs(first(source_low) - first(reference_low))
    target_e0_error = abs(first(transformed_low) - first(reference_low))
    observable_error = abs(transformed_observable - observable_reference)
    pipeline_matrix_error = matrix_error(transformed_matrix, target_matrix)
    pipeline_spectrum_error = max_abs_difference(transformed_low, target_low)
    leakage = Float64(get(transformed_diagnostics, :truncation_leakage, 0.0))

    push!(RESULTS, (
        case=case_name,
        cutoff=Int(cutoff),
        dimension=length(transformed_solution.basis),
        reference_kind=reference_kind,
        source_e0_error=source_e0_error,
        target_e0_error=target_e0_error,
        source_low_error=source_low_error,
        target_low_error=target_low_error,
        observable_value=transformed_observable,
        observable_reference=observable_reference,
        observable_error=observable_error,
        pipeline_matrix_error=pipeline_matrix_error,
        pipeline_spectrum_error=pipeline_spectrum_error,
        truncation_leakage=leakage,
        elapsed_seconds=elapsed,
    ))
    return nothing
end

function case_rows(case_name::Symbol)
    return [row for row in RESULTS if row.case === case_name]
end

function case_converged(case_name::Symbol)
    rows = case_rows(case_name)
    length(rows) >= REQUIRED_STABLE_STEPS || return false
    tail = rows[(end - REQUIRED_STABLE_STEPS + 1):end]
    energy_ok = all(row -> row.target_e0_error <= ENERGY_TOL && row.target_low_error <= ENERGY_TOL, tail)
    observable_ok = all(row -> row.observable_error <= OBSERVABLE_TOL, tail)
    pipeline_ok = all(row -> row.pipeline_matrix_error <= PIPELINE_TOL && row.pipeline_spectrum_error <= PIPELINE_TOL, rows)
    return energy_ok && observable_ok && pipeline_ok
end

function format_value(value)
    ismissing(value) && return "n/a"
    return @sprintf("%.3e", value)
end

function print_results()
    println("\nBOSONIC TRUNCATION-CONVERGENCE BENCHMARK")
    println("cutoffs = ", CUTOFFS, ", reference cutoff = ", REFERENCE_CUTOFF, ", nlow = ", NLOW)
    println("energy tolerance = ", ENERGY_TOL, ", observable tolerance = ", OBSERVABLE_TOL, ", pipeline tolerance = ", PIPELINE_TOL)
    println(repeat("-", 154))
    @printf(
        "%-27s %5s %6s %10s %10s %10s %10s %10s %10s %10s %9s\n",
        "case", "Nmax", "dim", "src dE0", "xfm dE0", "xfm dElow", "dObs", "pipe dH", "pipe dE", "leakage", "time(s)",
    )
    println(repeat("-", 154))
    for row in RESULTS
        @printf(
            "%-27s %5d %6d %10s %10s %10s %10s %10s %10s %10s %9.3f\n",
            String(row.case),
            row.cutoff,
            row.dimension,
            format_value(row.source_e0_error),
            format_value(row.target_e0_error),
            format_value(row.target_low_error),
            format_value(row.observable_error),
            format_value(row.pipeline_matrix_error),
            format_value(row.pipeline_spectrum_error),
            format_value(row.truncation_leakage),
            row.elapsed_seconds,
        )
    end
    println(repeat("-", 154))
    println("\nCONVERGENCE SUMMARY")
    for (case_name, _) in CASE_BUILDERS
        rows = case_rows(case_name)
        final = last(rows)
        status = case_converged(case_name) ? "CONVERGED" : "NOT CONVERGED"
        @printf(
            "%-27s %-13s final Nmax=%d, dE0=%s, dElow=%s, dObs=%s, leakage=%s, reference=%s\n",
            String(case_name),
            status,
            final.cutoff,
            format_value(final.target_e0_error),
            format_value(final.target_low_error),
            format_value(final.observable_error),
            format_value(final.truncation_leakage),
            String(final.reference_kind),
        )
    end
end

function csv_value(value)
    ismissing(value) && return ""
    value isa Symbol && return String(value)
    return string(value)
end

function write_csv(path::AbstractString)
    headers = (
        :case,
        :cutoff,
        :dimension,
        :reference_kind,
        :source_e0_error,
        :target_e0_error,
        :source_low_error,
        :target_low_error,
        :observable_value,
        :observable_reference,
        :observable_error,
        :pipeline_matrix_error,
        :pipeline_spectrum_error,
        :truncation_leakage,
        :elapsed_seconds,
    )
    open(path, "w") do io
        println(io, join(string.(headers), ','))
        for row in RESULTS
            println(io, join((csv_value(getproperty(row, header)) for header in headers), ','))
        end
    end
    return path
end

@testset "Bosonic truncation convergence" begin
    for (case_name, builder) in CASE_BUILDERS
        @testset "$(case_name)" begin
            for cutoff in CUTOFFS
                benchmark_case!(case_name, builder, cutoff)
            end
            if STRICT
                @test case_converged(case_name)
            end
        end
    end
end

print_results()
if SAVE_CSV
    write_csv(OUTPUT_PATH)
    println("\nCSV results written to: ", OUTPUT_PATH)
end
if !STRICT
    println("Strict convergence gating is disabled. Set PHUNDAMENTAL_BOSON_CONVERGENCE_STRICT=1 to make nonconvergence fail the testset.")
end

end
