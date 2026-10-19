# The acoustic sum rule constrains coefficients of a complete harmonic Hamiltonian, not each invariant basis direction separately.

function _quadratic_displacement_hessian(expr::AbstractOperatorExpr, algebra::CoordinateMomentumAlgebra; tolerance::Real=1e-10)
    canonical = canonicalize(algebra, expr)
    terms = canonical isa OperatorSum ? canonical.terms : AbstractOperatorExpr[canonical]
    K = zeros(Float64, algebra.ndof, algebra.ndof)
    for term in terms
        coefficient, operator = term isa ScaledOperator ? (term.coefficient, term.operator) : (1.0, term)
        abs(imag(coefficient)) <= tolerance || throw(ArgumentError("harmonic displacement coefficients must be real"))
        value = Float64(real(coefficient))
        abs(value) <= tolerance && continue
        operator isa IdentityOperator && continue
        factors = operator isa OperatorProduct ? operator.factors : AbstractOperatorExpr[operator]
        if all(factor -> factor isa MomentumOperator, factors)
            length(factors) == 2 || throw(ArgumentError("only quadratic kinetic terms are supported by acoustic constraints"))
            continue
        end
        length(factors) == 2 && all(factor -> factor isa PositionOperator, factors) ||
            throw(ArgumentError("acoustic constraints require a purely quadratic coordinate potential plus optional quadratic kinetic terms"))
        i, j = factors[1].mode, factors[2].mode
        1 <= i <= algebra.ndof && 1 <= j <= algebra.ndof || throw(ArgumentError("displacement mode index is out of bounds"))
        if i == j
            K[i, i] += 2value
        else
            K[i, j] += value
            K[j, i] += value
        end
    end
    return K
end

"""
    acoustic_sum_rule_constraints(space; nsites, dimension=3, tolerance=1e-10)

Return independent homogeneous `LinearParameterConstraint`s enforcing `sum_j K[iα,jβ] = 0` for every site `i` and Cartesian pair `(α, β)` in the complete harmonic force-constant Hessian. The operator ordering must be site-major with `dimension` Cartesian coordinates per site. Kinetic terms are ignored, but anharmonic or mixed coordinate/momentum terms are rejected.
"""
function acoustic_sum_rule_constraints(space::HamiltonianSpace; nsites::Integer, dimension::Integer=3, tolerance::Real=1e-10)
    algebra = space.specification.algebra
    algebra isa CoordinateMomentumAlgebra || throw(ArgumentError("acoustic sum rules require a CoordinateMomentumAlgebra"))
    dimension == 3 || throw(ArgumentError("the current Cartesian acoustic constraint implementation requires dimension=3"))
    nsites > 0 && algebra.ndof == nsites * dimension || throw(DimensionMismatch("phonon algebra must have exactly three modes per site"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be positive and finite"))
    hessians = [_quadratic_displacement_hessian(operator, algebra; tolerance=tolerance) for operator in space.basis.operators]
    rows = Vector{Float64}[]
    constraints = LinearParameterConstraint[]
    for i in 1:nsites, alpha in 1:dimension, beta in 1:dimension
        row = Float64[sum(K[(i - 1) * dimension + alpha, (j - 1) * dimension + beta] for j in 1:nsites) for K in hessians]
        norm(row) <= tolerance && continue
        if !isempty(rows)
            previous = reduce(vcat, (permutedims(existing) for existing in rows))
            rank(vcat(previous, permutedims(row)); atol=tolerance, rtol=tolerance) == size(previous, 1) && continue
        end
        coefficients = Dict(space.basis.labels[k] => row[k] for k in eachindex(row) if abs(row[k]) > tolerance)
        push!(constraints, LinearParameterConstraint(coefficients, 0.0; name=Symbol("acoustic_", i, "_", alpha, "_", beta),
            atol=tolerance, rtol=tolerance, metadata=Dict(:physical_rule => :acoustic_sum_rule, :site => i, :alpha => alpha, :beta => beta)))
        push!(rows, row)
    end
    return constraints
end

"""
    rotational_sum_rule_constraints(space; positions, nsites, boundary=:open, tolerance=1e-10)

Construct independent homogeneous constraints that make the harmonic Hessian annihilate infinitesimal rigid-body rotations of an explicitly **open finite** set of Cartesian sites. The Cartesian positions must be a 3×nsites or nsites×3 matrix, or a collection of three-vectors; indexing is site-major with three coordinates per site. For each axis `δ`, the required identity is `sum(j,β,γ) K[iα,jβ] * ε[βδγ] * (r[j,γ] - r[i,γ]) = 0`. This finite-reference rule does not certify periodic Born–Huang sum rules, which require image-resolved force constants. `boundary=:periodic` is rejected explicitly.
"""
function rotational_sum_rule_constraints(space::HamiltonianSpace; positions, nsites::Integer, boundary::Symbol=:open, tolerance::Real=1e-10)
    boundary === :open || throw(ArgumentError("rigid-rotation Hessian constraints require an explicitly open finite reference; periodic cells require image-resolved force constants"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be positive and finite"))
    algebra = space.specification.algebra
    algebra isa CoordinateMomentumAlgebra || throw(ArgumentError("rotational constraints require a CoordinateMomentumAlgebra"))
    nsites > 0 && algebra.ndof == 3 * nsites || throw(DimensionMismatch("rotational constraints require exactly three Cartesian coordinates per site"))
    coordinates = if positions isa AbstractMatrix{<:Real}
        size(positions) == (3, nsites) ? Matrix{Float64}(positions) :
            size(positions) == (nsites, 3) ? Matrix{Float64}(permutedims(positions)) :
            throw(DimensionMismatch("Cartesian site positions must form a 3×nsites or nsites×3 matrix"))
    else
        samples = collect(positions)
        length(samples) == nsites && all(sample -> length(sample) == 3, samples) ||
            throw(DimensionMismatch("provide one Cartesian three-vector per site"))
        hcat((Float64.(collect(sample)) for sample in samples)...)
    end
    all(isfinite, coordinates) || throw(ArgumentError("Cartesian site positions must be finite"))
    hessians = [_quadratic_displacement_hessian(operator, algebra; tolerance=tolerance) for operator in space.basis.operators]
    rows = Vector{Float64}[]
    constraints = LinearParameterConstraint[]
    for i in 1:nsites, alpha in 1:3, axis in 1:3
        rotation = zeros(Float64, 3)
        rotation[axis] = 1.0
        displacements = [cross(rotation, coordinates[:, j] - coordinates[:, i]) for j in 1:nsites]
        row = Float64[sum(K[3 * (i - 1) + alpha, 3 * (j - 1) + beta] * displacements[j][beta]
            for j in 1:nsites, beta in 1:3) for K in hessians]
        norm(row) <= tolerance && continue
        if !isempty(rows)
            previous = reduce(vcat, (permutedims(existing) for existing in rows))
            rank(vcat(previous, permutedims(row)); atol=tolerance, rtol=tolerance) == size(previous, 1) && continue
        end
        coefficients = Dict(space.basis.labels[k] => row[k] for k in eachindex(row) if abs(row[k]) > tolerance)
        push!(constraints, LinearParameterConstraint(coefficients, 0.0; name=Symbol("rotational_", i, "_", alpha, "_", axis),
            atol=tolerance, rtol=tolerance, metadata=Dict(:physical_rule => :finite_rigid_rotation, :site => i, :alpha => alpha,
                :axis => axis, :boundary => :open)))
        push!(rows, row)
    end
    return constraints
end
