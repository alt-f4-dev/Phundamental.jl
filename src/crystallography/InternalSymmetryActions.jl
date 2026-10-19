# Explicit internal representations extend, rather than replace, the crystallographic site action.

"""Validated axial-spin action, including antiunitary time reversal, already implemented by the standard crystallographic mapper."""
struct MagneticSymmetryAction end

function (::MagneticSymmetryAction)(expression::AbstractOperatorExpr, operation::SpaceGroupOperation, crystal::CrystalStructure, algebra::SpinAlgebra)
    _require_cartesian_isometry(crystal, operation)
    return standard_symmetry_action(expression, operation, crystal, algebra)
end

"""Spinful fermions indexed by `(site - 1) * (2 * norbitals) + 2 * (orbital - 1) + spin`. Orbital matrices act on orbital columns."""
struct SpinfulFermionAction{O,T}
    norbitals::Int
    orbital_representation::O
    orbital_time_reversal::T
end

function SpinfulFermionAction(; norbitals::Integer=1, orbital_representation=nothing, orbital_time_reversal=nothing)
    norbitals > 0 || throw(ArgumentError("norbitals must be positive"))
    return SpinfulFermionAction(Int(norbitals), orbital_representation, orbital_time_reversal)
end

"""Three Cartesian displacement and conjugate-momentum components per site, with time reversal reversing momenta."""
struct VectorDisplacementAction end

function _require_cartesian_isometry(crystal::CrystalStructure, operation::SpaceGroupOperation; atol::Real=1e-8)
    R = cartesian_rotation(crystal, operation)
    isapprox(R' * R, Matrix{Float64}(I, 3, 3); atol=atol, rtol=atol) ||
        throw(ArgumentError("operation is not orthogonal in the physical Cartesian lattice metric"))
    abs(abs(det(R)) - 1) <= atol || throw(ArgumentError("crystallographic operation does not have determinant ±1"))
    return R
end

function _unitary_representation(matrix, n::Int, description::AbstractString; atol::Real=1e-9)
    matrix isa AbstractMatrix || throw(ArgumentError("$description must be a square matrix"))
    size(matrix) == (n, n) || throw(DimensionMismatch("$description must have size ($n, $n)"))
    U = Matrix{ComplexF64}(matrix)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), U) || throw(ArgumentError("$description contains nonfinite entries"))
    isapprox(U' * U, Matrix{ComplexF64}(I, n, n); atol=atol, rtol=atol) ||
        throw(ArgumentError("$description must be unitary to preserve fermionic anticommutation relations"))
    return U
end

function _spin_half_rotation(R::AbstractMatrix{<:Real})
    # The positive-trace quaternion branch avoids an unstable axis extraction near theta = pi.
    q = zeros(Float64, 4)
    tr = sum(R[i, i] for i in 1:3)
    if tr > 0
        s = 2 * sqrt(1 + tr)
        q .= (s / 4, (R[3, 2] - R[2, 3]) / s, (R[1, 3] - R[3, 1]) / s, (R[2, 1] - R[1, 2]) / s)
    else
        i = argmax((R[1, 1], R[2, 2], R[3, 3]))
        j, k = mod1(i + 1, 3), mod1(i + 2, 3)
        s = 2 * sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k]))
        s > 1e-12 || throw(ArgumentError("unable to resolve the spinor lift of the spatial rotation"))
        q[i + 1] = s / 4
        q[1] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    end
    q ./= norm(q)
    w, x, y, z = q
    U = Matrix{ComplexF64}(undef, 2, 2)
    U[1, 1], U[1, 2] = w - im * z, -im * x - y
    U[2, 1], U[2, 2] = -im * x + y, w + im * z
    return _unitary_representation(U, 2, "spin-1/2 rotation")
end

function _orbital_matrix(action::SpinfulFermionAction, operation::SpaceGroupOperation, crystal::CrystalStructure)
    supplied = action.orbital_representation
    matrix = isnothing(supplied) ? Matrix{ComplexF64}(I, action.norbitals, action.norbitals) :
        supplied isa AbstractMatrix ? supplied : supplied(operation, crystal)
    U = _unitary_representation(matrix, action.norbitals, "orbital symmetry representation")
    if operation.time_reversal
        if isnothing(action.orbital_time_reversal)
            action.norbitals == 1 || throw(ArgumentError(
                "multiorbital time reversal needs an explicit orbital_time_reversal matrix in the declared orbital basis"))
        else
            U *= _unitary_representation(action.orbital_time_reversal, action.norbitals, "orbital time-reversal representation")
        end
    end
    return U
end

function _internal_mode_map(action::SpinfulFermionAction, operation::SpaceGroupOperation, crystal::CrystalStructure)
    R = _require_cartesian_isometry(crystal, operation)
    A = det(R) * R
    # Annihilation fields transform with the adjoint spinor, so c†σc follows the declared axial rotation.
    spin = _spin_half_rotation(A)'
    if operation.time_reversal
        spin *= ComplexF64[0 1; -1 0]
    end
    block = kron(_orbital_matrix(action, operation, crystal), spin)
    permutation = site_permutation(crystal, operation)
    return permutation, block
end

function _linear_mode_image(generator, source::Int, permutation::AbstractVector, block::AbstractMatrix)
    n = size(block, 1)
    site = div(source - 1, n) + 1
    alpha = mod1(source, n)
    1 <= site <= length(permutation) || throw(ArgumentError("mode index $source lies outside the declared site layout"))
    terms = AbstractOperatorExpr[]
    for beta in 1:n
        coefficient = block[beta, alpha]
        abs(coefficient) <= 1e-14 && continue
        push!(terms, coefficient * generator((permutation[site] - 1) * n + beta))
    end
    return isempty(terms) ? 0 * IdentityOperator() : length(terms) == 1 ? terms[1] : OperatorSum(terms)
end

_internal_primitive(op::IdentityOperator, action::SpinfulFermionAction, operation, permutation, block) = op
_internal_primitive(op::FermionAnnihilate, action::SpinfulFermionAction, operation, permutation, block) =
    _linear_mode_image(c, op.mode, permutation, block)
_internal_primitive(op::FermionCreate, action::SpinfulFermionAction, operation, permutation, block) =
    _linear_mode_image(i -> c(i)', op.mode, permutation, conj.(block))
_internal_primitive(op::NumberOperator, action::SpinfulFermionAction, operation, permutation, block) =
    op.kind === :fermion ? _internal_primitive(FermionCreate(op.mode), action, operation, permutation, block) *
        _internal_primitive(FermionAnnihilate(op.mode), action, operation, permutation, block) :
        throw(ArgumentError("spinful fermion action requires fermionic number operators"))

_internal_primitive(op::IdentityOperator, action::VectorDisplacementAction, operation, permutation, block) = op
_internal_primitive(op::PositionOperator, action::VectorDisplacementAction, operation, permutation, block) =
    _linear_mode_image(xop, op.mode, permutation, block)
_internal_primitive(op::MomentumOperator, action::VectorDisplacementAction, operation, permutation, block) =
    _linear_mode_image(pop, op.mode, permutation, operation.time_reversal ? -block : block)

function _internal_primitive(op::AbstractPrimitiveOperator, action, operation, permutation, block)
    throw(ArgumentError("$(typeof(op)) is not supported by the $(typeof(action)) symmetry action"))
end

_internal_transform(expr::AbstractPrimitiveOperator, action, operation, permutation, block) =
    _internal_primitive(expr, action, operation, permutation, block)

function _internal_transform(expr::ScaledOperator, action, operation, permutation, block)
    coefficient = operation.time_reversal ? conj(expr.coefficient) : expr.coefficient
    return coefficient * _internal_transform(expr.operator, action, operation, permutation, block)
end

function _internal_transform(expr::OperatorSum, action, operation, permutation, block)
    return OperatorSum(AbstractOperatorExpr[_internal_transform(term, action, operation, permutation, block) for term in expr.terms])
end

function _internal_transform(expr::OperatorProduct, action, operation, permutation, block)
    return OperatorProduct(AbstractOperatorExpr[_internal_transform(term, action, operation, permutation, block) for term in expr.factors])
end

function _fermion_parity(expr::AbstractPrimitiveOperator)
    expr isa Union{FermionAnnihilate,FermionCreate} && return 1
    expr isa Union{IdentityOperator,NumberOperator} && return 0
    throw(ArgumentError("spinful fermion symmetry supports only fermionic field and number operators"))
end
_fermion_parity(expr::ScaledOperator) = _fermion_parity(expr.operator)
_fermion_parity(expr::OperatorProduct) = mod(sum(_fermion_parity(factor) for factor in expr.factors), 2)
function _fermion_parity(expr::OperatorSum)
    all(iszero, (_fermion_parity(term) for term in expr.terms)) ||
        throw(ArgumentError("spinful fermion invariant generation requires even-fermion-parity candidate terms"))
    return 0
end

function (action::SpinfulFermionAction)(expression::AbstractOperatorExpr, operation::SpaceGroupOperation,
                                        crystal::CrystalStructure, algebra::FermionAlgebra)
    algebra.nmodes == 2 * action.norbitals * length(crystal) ||
        throw(DimensionMismatch("spinful fermion algebra must provide 2 * norbitals modes per crystallographic site"))
    permutation, U = _internal_mode_map(action, operation, crystal)
    return canonicalize(algebra, _internal_transform(expression, action, operation, permutation, U))
end

function (action::VectorDisplacementAction)(expression::AbstractOperatorExpr, operation::SpaceGroupOperation,
                                              crystal::CrystalStructure, algebra::CoordinateMomentumAlgebra)
    algebra.ndof == 3 * length(crystal) || throw(DimensionMismatch("vector displacement algebra requires three coordinates per site"))
    permutation = site_permutation(crystal, operation)
    R = _require_cartesian_isometry(crystal, operation)
    return canonicalize(algebra, _internal_transform(expression, action, operation, permutation, R))
end

"""Check geometric, index-layout, and internal-unitarity preconditions before constructing an invariant family."""
function validate_internal_symmetry(action::Union{MagneticSymmetryAction,SpinfulFermionAction,VectorDisplacementAction},
                                    crystal::CrystalStructure, dataset::CrystallographicDataset, algebra::AbstractOperatorAlgebra)
    if action isa MagneticSymmetryAction
        algebra isa SpinAlgebra && algebra.nsites == length(crystal) || throw(DimensionMismatch("magnetic action requires one spin index per site"))
    elseif action isa SpinfulFermionAction
        algebra isa FermionAlgebra && algebra.nmodes == 2 * action.norbitals * length(crystal) ||
            throw(DimensionMismatch("spinful fermion index layout is incompatible with the crystal"))
    else
        algebra isa CoordinateMomentumAlgebra && algebra.ndof == 3 * length(crystal) ||
            throw(DimensionMismatch("vector displacement index layout is incompatible with the crystal"))
    end
    for operation in dataset.operations
        site_permutation(crystal, operation)
        _require_cartesian_isometry(crystal, operation)
        action isa SpinfulFermionAction && _internal_mode_map(action, operation, crystal)
    end
    return true
end
