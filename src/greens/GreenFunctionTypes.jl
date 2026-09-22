# Matrix-valued propagators defined on explicit frequency axes.

abstract type AbstractFrequencyMatrix end

_metadata_dict(metadata::AbstractDict) = Dict{Symbol,Any}(metadata)
_metadata_dict(metadata::NamedTuple) = Dict{Symbol,Any}(pairs(metadata))

function _frequency_value_cube(axis::AbstractFrequencyAxis, values::AbstractVector{<:Number})
    length(values) == length(axis) || throw(DimensionMismatch("frequency-value count does not match the axis"))
    cube = Array{ComplexF64}(undef, 1, 1, length(axis))
    @inbounds for i in eachindex(values)
        cube[1, 1, i] = ComplexF64(values[i])
    end
    return cube
end

function _frequency_value_cube(axis::AbstractFrequencyAxis, values::AbstractVector{<:AbstractMatrix})
    length(values) == length(axis) || throw(DimensionMismatch("frequency-matrix count does not match the axis"))
    isempty(values) && throw(ArgumentError("frequency matrices cannot be empty"))
    nrow, ncol = size(first(values))
    nrow == ncol || throw(DimensionMismatch("frequency matrices must be square"))
    cube = Array{ComplexF64}(undef, nrow, ncol, length(axis))
    @inbounds for i in eachindex(values)
        size(values[i]) == (nrow, ncol) || throw(DimensionMismatch("all frequency matrices must have the same shape"))
        cube[:, :, i] .= values[i]
    end
    return cube
end

function _frequency_value_cube(axis::AbstractFrequencyAxis, values::AbstractArray{<:Number,3})
    size(values, 1) == size(values, 2) || throw(DimensionMismatch("frequency matrices must be square"))
    size(values, 3) == length(axis) || throw(DimensionMismatch("frequency-value count does not match the axis"))
    return ComplexF64.(values)
end

function _frequency_labels(dimension::Int, labels)
    if isnothing(labels)
        return [Symbol("component_", i) for i in 1:dimension]
    end
    out = Symbol.(collect(labels))
    length(out) == dimension || throw(DimensionMismatch("component-label count must equal the matrix dimension"))
    return out
end

"""Interacting matrix-valued Green function on a discrete frequency axis."""
struct GreenFunction{A<:AbstractFrequencyAxis} <: AbstractFrequencyMatrix
    axis::A
    values::Array{ComplexF64,3}
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

function GreenFunction(axis::A, values; labels=nothing, metadata=Dict{Symbol,Any}()) where {A<:AbstractFrequencyAxis}
    cube = _frequency_value_cube(axis, values)
    return GreenFunction{A}(axis, cube, _frequency_labels(size(cube, 1), labels), _metadata_dict(metadata))
end

"""Noninteracting or Weiss Green function structurally compatible with `GreenFunction`."""
struct NonInteractingGreenFunction{A<:AbstractFrequencyAxis} <: AbstractFrequencyMatrix
    axis::A
    values::Array{ComplexF64,3}
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

function NonInteractingGreenFunction(axis::A, values; labels=nothing, metadata=Dict{Symbol,Any}()) where {A<:AbstractFrequencyAxis}
    cube = _frequency_value_cube(axis, values)
    return NonInteractingGreenFunction{A}(axis, cube, _frequency_labels(size(cube, 1), labels), _metadata_dict(metadata))
end

"""Frequency-dependent self-energy `Σ(z)`."""
struct SelfEnergy{A<:AbstractFrequencyAxis} <: AbstractFrequencyMatrix
    axis::A
    values::Array{ComplexF64,3}
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

function SelfEnergy(axis::A, values; labels=nothing, metadata=Dict{Symbol,Any}()) where {A<:AbstractFrequencyAxis}
    cube = _frequency_value_cube(axis, values)
    return SelfEnergy{A}(axis, cube, _frequency_labels(size(cube, 1), labels), _metadata_dict(metadata))
end

"""Hybridization function `Δ(z)` used by finite-bath impurity embeddings."""
struct HybridizationFunction{A<:AbstractFrequencyAxis} <: AbstractFrequencyMatrix
    axis::A
    values::Array{ComplexF64,3}
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

function HybridizationFunction(axis::A, values; labels=nothing, metadata=Dict{Symbol,Any}()) where {A<:AbstractFrequencyAxis}
    cube = _frequency_value_cube(axis, values)
    return HybridizationFunction{A}(axis, cube, _frequency_labels(size(cube, 1), labels), _metadata_dict(metadata))
end

const FrequencyMatrix = Union{GreenFunction,NonInteractingGreenFunction,SelfEnergy,HybridizationFunction}

Base.length(object::FrequencyMatrix) = length(object.axis)
Base.size(object::FrequencyMatrix) = size(object.values)
Base.getindex(object::FrequencyMatrix, i::Integer) = view(object.values, :, :, Int(i))
Base.getindex(object::FrequencyMatrix, a::Integer, b::Integer, i::Integer) = object.values[Int(a), Int(b), Int(i)]

frequency_axis(object::FrequencyMatrix) = object.axis
frequency_values(object::FrequencyMatrix) = object.values
component_labels(object::FrequencyMatrix) = object.labels
matrix_dimension(object::FrequencyMatrix) = size(object.values, 1)

function scalar_frequency_values(object::FrequencyMatrix)
    matrix_dimension(object) == 1 || throw(DimensionMismatch("scalar_frequency_values requires a 1×1 frequency object"))
    return vec(copy(object.values[1, 1, :]))
end

function _compatible_frequency_objects(a::FrequencyMatrix, b::FrequencyMatrix)
    typeof(a.axis) === typeof(b.axis) || throw(ArgumentError("frequency axes have incompatible types"))
    a.axis.indices == b.axis.indices || throw(ArgumentError("frequency axes carry different Matsubara indices"))
    isapprox(a.axis.beta, b.axis.beta; rtol=0.0, atol=16eps(Float64) * max(abs(a.axis.beta), abs(b.axis.beta), 1.0)) ||
        throw(ArgumentError("frequency axes carry different inverse temperatures"))
    matrix_dimension(a) == matrix_dimension(b) || throw(DimensionMismatch("frequency matrices have different dimensions"))
    return true
end
