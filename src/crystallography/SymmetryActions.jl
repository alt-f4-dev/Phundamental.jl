"""Map one primitive operator under a crystallographic operation using Phundamental's standard scalar/vector conventions."""
function transform_primitive(operator::IdentityOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation::AbstractVector{<:Integer})
    return operator
end

function _spin_components(site::Integer)
    return AbstractOperatorExpr[Sx(site), Sy(site), Sz(site)]
end

function _spin_transform(component::Integer, site::Integer, operation::SpaceGroupOperation, structure::CrystalStructure, permutation)
    R = axial_rotation(structure, operation)
    operation.time_reversal && (R = -R)
    target = permutation[Int(site)]
    components = _spin_components(target)
    terms = AbstractOperatorExpr[]
    for β in 1:3
        abs(R[β, component]) <= 64eps(Float64) && continue
        push!(terms, R[β, component] * components[β])
    end
    isempty(terms) && return 0 * IdentityOperator()
    return length(terms) == 1 ? terms[1] : OperatorSum(terms)
end

transform_primitive(operator::SpinX, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = _spin_transform(1, operator.site, operation, structure, permutation)
transform_primitive(operator::SpinY, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = _spin_transform(2, operator.site, operation, structure, permutation)
transform_primitive(operator::SpinZ, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = _spin_transform(3, operator.site, operation, structure, permutation)
transform_primitive(operator::SpinPlus, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = transform_operator(Sx(operator.site) + im * Sy(operator.site), operation, structure; permutation=permutation)
transform_primitive(operator::SpinMinus, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = transform_operator(Sx(operator.site) - im * Sy(operator.site), operation, structure; permutation=permutation)
transform_primitive(operator::BosonAnnihilate, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = BosonAnnihilate(permutation[operator.mode])
transform_primitive(operator::BosonCreate, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = BosonCreate(permutation[operator.mode])
transform_primitive(operator::FermionAnnihilate, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = FermionAnnihilate(permutation[operator.mode])
transform_primitive(operator::FermionCreate, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = FermionCreate(permutation[operator.mode])
transform_primitive(operator::NumberOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = NumberOperator(operator.kind, permutation[operator.mode])
transform_primitive(operator::MajoranaOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation) = MajoranaOperator(permutation[operator.index])

function transform_primitive(operator::PositionOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation)
    throw(ArgumentError("PositionOperator mode indices do not encode crystallographic vector components; provide a custom symmetry_action for coordinate operators"))
end

function transform_primitive(operator::MomentumOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation)
    throw(ArgumentError("MomentumOperator mode indices do not encode crystallographic vector components; provide a custom symmetry_action for coordinate operators"))
end

function transform_primitive(operator::FactorOperator, operation::SpaceGroupOperation, structure::CrystalStructure, permutation)
    transformed = transform_primitive(operator.operator, operation, structure, permutation)
    transformed isa AbstractPrimitiveOperator && return FactorOperator(operator.factor, transformed)
    return _map_factor_expression(operator.factor, transformed)
end

_map_factor_expression(factor::Integer, operator::AbstractPrimitiveOperator) = FactorOperator(factor, operator)
_map_factor_expression(factor::Integer, operator::ScaledOperator) = operator.coefficient * _map_factor_expression(factor, operator.operator)
_map_factor_expression(factor::Integer, operator::OperatorSum) = OperatorSum(AbstractOperatorExpr[_map_factor_expression(factor, term) for term in operator.terms])
_map_factor_expression(factor::Integer, operator::OperatorProduct) = OperatorProduct(AbstractOperatorExpr[_map_factor_expression(factor, term) for term in operator.factors])

function _resolved_permutation(structure::CrystalStructure, operation::SpaceGroupOperation, permutation, site_tolerance::Real)
    return isnothing(permutation) ? site_permutation(structure, operation; atol=site_tolerance) : permutation
end

function transform_operator(expression::AbstractPrimitiveOperator, operation::SpaceGroupOperation, structure::CrystalStructure; permutation=nothing, site_tolerance::Real=1e-7)
    perm = _resolved_permutation(structure, operation, permutation, site_tolerance)
    return transform_primitive(expression, operation, structure, perm)
end

function transform_operator(expression::ScaledOperator, operation::SpaceGroupOperation, structure::CrystalStructure; permutation=nothing, site_tolerance::Real=1e-7)
    perm = _resolved_permutation(structure, operation, permutation, site_tolerance)
    coefficient = operation.time_reversal ? conj(expression.coefficient) : expression.coefficient
    return coefficient * transform_operator(expression.operator, operation, structure; permutation=perm, site_tolerance=site_tolerance)
end

function transform_operator(expression::OperatorSum, operation::SpaceGroupOperation, structure::CrystalStructure; permutation=nothing, site_tolerance::Real=1e-7)
    perm = _resolved_permutation(structure, operation, permutation, site_tolerance)
    return OperatorSum(AbstractOperatorExpr[transform_operator(term, operation, structure; permutation=perm, site_tolerance=site_tolerance) for term in expression.terms])
end

function transform_operator(expression::OperatorProduct, operation::SpaceGroupOperation, structure::CrystalStructure; permutation=nothing, site_tolerance::Real=1e-7)
    perm = _resolved_permutation(structure, operation, permutation, site_tolerance)
    return OperatorProduct(AbstractOperatorExpr[transform_operator(factor, operation, structure; permutation=perm, site_tolerance=site_tolerance) for factor in expression.factors])
end

function standard_symmetry_action(expression::AbstractOperatorExpr, operation::SpaceGroupOperation, structure::CrystalStructure, algebra::AbstractOperatorAlgebra; site_tolerance::Real=1e-7)
    permutation = site_permutation(structure, operation; atol=site_tolerance)
    transformed = transform_operator(expression, operation, structure; permutation=permutation, site_tolerance=site_tolerance)
    return canonicalize(algebra, transformed)
end
