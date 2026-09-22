# Expand the package's noncommutative AST into scalar × ordered primitive terms.
# This is performed once before hot-loop execution.

struct _CompiledPrimitive
    code::UInt8
    index::Int
end

struct _CompiledTerm
    coefficient::ComplexF64
    primitives::Vector{_CompiledPrimitive}
end

const _OP_IDENTITY = UInt8(0)
const _OP_SPIN_X   = UInt8(1)
const _OP_SPIN_Y   = UInt8(2)
const _OP_SPIN_Z   = UInt8(3)
const _OP_SPIN_P   = UInt8(4)
const _OP_SPIN_M   = UInt8(5)
const _OP_FERMION_A = UInt8(10)
const _OP_FERMION_C = UInt8(11)
const _OP_FERMION_N = UInt8(12)
const _OP_BOSON_A   = UInt8(20)
const _OP_BOSON_C   = UInt8(21)
const _OP_BOSON_N   = UInt8(22)

function _primitive_code(op::IdentityOperator)
    return _CompiledPrimitive(_OP_IDENTITY, 0)
end
_primitive_code(op::SpinX) = _CompiledPrimitive(_OP_SPIN_X, op.site)
_primitive_code(op::SpinY) = _CompiledPrimitive(_OP_SPIN_Y, op.site)
_primitive_code(op::SpinZ) = _CompiledPrimitive(_OP_SPIN_Z, op.site)
_primitive_code(op::SpinPlus) = _CompiledPrimitive(_OP_SPIN_P, op.site)
_primitive_code(op::SpinMinus) = _CompiledPrimitive(_OP_SPIN_M, op.site)
_primitive_code(op::FermionAnnihilate) = _CompiledPrimitive(_OP_FERMION_A, op.mode)
_primitive_code(op::FermionCreate) = _CompiledPrimitive(_OP_FERMION_C, op.mode)
function _primitive_code(op::NumberOperator)
    op.kind === :fermion && return _CompiledPrimitive(_OP_FERMION_N, op.mode)
    op.kind === :boson && return _CompiledPrimitive(_OP_BOSON_N, op.mode)
    throw(ArgumentError("optimized kernel does not support number-operator kind $(op.kind)"))
end
_primitive_code(op::BosonAnnihilate) = _CompiledPrimitive(_OP_BOSON_A, op.mode)
_primitive_code(op::BosonCreate) = _CompiledPrimitive(_OP_BOSON_C, op.mode)

function _primitive_code(op::AbstractPrimitiveOperator)
    throw(ArgumentError("optimized kernel does not support primitive $(typeof(op))"))
end

function _expanded_operator_terms(
    expr::AbstractOperatorExpr;
    max_terms::Integer=1_000_000,
)
    function expand(op)
        if op isa AbstractPrimitiveOperator
            return Tuple{ComplexF64,Vector{_CompiledPrimitive}}[
                (1.0 + 0.0im, _CompiledPrimitive[_primitive_code(op)])
            ]
        elseif op isa ScaledOperator
            inner = expand(op.operator)
            return [(ComplexF64(op.coefficient) * c, p) for (c, p) in inner]
        elseif op isa OperatorSum
            out = Tuple{ComplexF64,Vector{_CompiledPrimitive}}[]
            for term in op.terms
                append!(out, expand(term))
                length(out) <= max_terms || throw(ArgumentError(
                    "operator expansion exceeded max_terms=$max_terms"
                ))
            end
            return out
        elseif op isa OperatorProduct
            acc = Tuple{ComplexF64,Vector{_CompiledPrimitive}}[
                (1.0 + 0.0im, _CompiledPrimitive[])
            ]
            for factor in op.factors
                rhs = expand(factor)
                next = Tuple{ComplexF64,Vector{_CompiledPrimitive}}[]
                length(acc) * length(rhs) <= max_terms || throw(ArgumentError(
                    "operator product expansion exceeded max_terms=$max_terms"
                ))
                for (ca, pa) in acc, (cb, pb) in rhs
                    push!(next, (ca * cb, vcat(pa, pb)))
                end
                acc = next
            end
            return acc
        end
        throw(ArgumentError("unsupported operator expression $(typeof(op))"))
    end

    raw = expand(expr)
    return _CompiledTerm[
        _CompiledTerm(c, p)
        for (c, p) in raw if !iszero(c)
    ]
end

@inline function _bit_at(mask::UInt64, position::Int)
    return (mask >> (position - 1)) & UInt64(1)
end
