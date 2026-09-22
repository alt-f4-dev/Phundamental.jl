# ClassicalTypes.jl

abstract type AbstractClassicalProblem end
abstract type AbstractClassicalSampler end
abstract type AbstractConfigurationEnsemble end

"""
    IsingProblem(J; field=zeros(N), constant=0, positions=nothing,
                   moment_vectors=nothing, metadata=Dict())

Classical Ising energy

    E(s) = constant + 1/2 s' J s + field' s,
    s_i in {-1,+1}.

`J` must be a real symmetric dense or sparse matrix. Its diagonal is allowed
but contributes only a configuration-independent constant because `s_i^2=1`.

`positions` and `moment_vectors` are optional observable metadata. When
provided, `moment_vectors` must be a 3×N matrix whose column `i` is the
physical moment vector for `s_i=+1`. This lets a geometry/model layer supply
local-axis moments without making the sampler depend on `Models`.
"""
struct IsingProblem{M<:AbstractMatrix,V<:AbstractVector,P,MV,T<:Real} <: AbstractClassicalProblem
    J::M
    field::V
    constant::T
    positions::P
    moment_vectors::MV
    metadata::Dict{Symbol,Any}
end

function IsingProblem(
    J::AbstractMatrix{<:Real};
    field=nothing,
    constant::Real=0.0,
    positions=nothing,
    moment_vectors=nothing,
    symmetry_tolerance::Real=1e-12,
    metadata=Dict{Symbol,Any}(),
)
    size(J, 1) == size(J, 2) ||
        throw(DimensionMismatch("Ising coupling matrix J must be square"))
    N = size(J, 1)
    N > 0 || throw(ArgumentError("IsingProblem cannot be empty"))
    all(isfinite, J) || throw(ArgumentError("J must contain only finite values"))

    asym = norm(J - transpose(J)) / max(norm(J), 1.0)
    asym <= symmetry_tolerance ||
        throw(ArgumentError(
            "Ising coupling matrix must be symmetric; relative asymmetry=$asym"
        ))

    Jstored = if J isa SparseMatrixCSC
        sparse(Float64.(J))
    else
        Matrix{Float64}(J)
    end

    h = isnothing(field) ? zeros(Float64, N) : Float64.(collect(field))
    length(h) == N || throw(DimensionMismatch("field must contain one value per Ising site"))
    all(isfinite, h) || throw(ArgumentError("field values must be finite"))

    pos = if isnothing(positions)
        nothing
    else
        length(positions) == N ||
            throw(DimensionMismatch("positions must contain one entry per Ising site"))
        collect(positions)
    end

    moments = if isnothing(moment_vectors)
        nothing
    else
        M = Matrix{Float64}(moment_vectors)
        size(M) == (3, N) ||
            throw(DimensionMismatch("moment_vectors must be a 3×N matrix"))
        M
    end

    return IsingProblem(
        Jstored,
        h,
        Float64(constant),
        pos,
        moments,
        Dict{Symbol,Any}(metadata),
    )
end

Base.length(problem::IsingProblem) = size(problem.J, 1)

"""
    MetropolisSampler(; samples=1000, burnin_sweeps=1000,
                        sweeps_per_sample=10, seed=0,
                        initial=:random, energy_check_interval=100,
                        cache_tolerance=1e-10)

Single-temperature random-site Metropolis sampler. One sweep is `N` uniformly
random single-site proposals.

The proposal `i -> flip(i)` is exactly symmetric because every site is selected
with probability `1/N`, so the acceptance probability is

    min(1, exp(-beta * DeltaE)).

Future work:
    Parallel tempering / replica exchange should be implemented as a separate
    sampler for frustrated and glassy systems. It is intentionally not hidden
    inside this baseline Metropolis algorithm.
"""
struct MetropolisSampler <: AbstractClassicalSampler
    samples::Int
    burnin_sweeps::Int
    sweeps_per_sample::Int
    seed::Int
    initial::Any
    energy_check_interval::Int
    cache_tolerance::Float64
end

function MetropolisSampler(
    ; samples::Integer=1000,
      burnin_sweeps::Integer=1000,
      sweeps_per_sample::Integer=10,
      seed::Integer=0,
      initial=:random,
      energy_check_interval::Integer=100,
      cache_tolerance::Real=1e-10,
)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    burnin_sweeps >= 0 || throw(ArgumentError("burnin_sweeps must be nonnegative"))
    sweeps_per_sample > 0 || throw(ArgumentError("sweeps_per_sample must be positive"))
    energy_check_interval >= 0 ||
        throw(ArgumentError("energy_check_interval must be nonnegative"))
    cache_tolerance > 0 || throw(ArgumentError("cache_tolerance must be positive"))
    return MetropolisSampler(
        Int(samples),
        Int(burnin_sweeps),
        Int(sweeps_per_sample),
        Int(seed),
        initial,
        Int(energy_check_interval),
        Float64(cache_tolerance),
    )
end

"""
Samples are stored column-wise as an `N×Nsamples` matrix. For an Ising problem
the entries are `Int8(±1)`, which keeps large ensembles compact.
"""
struct ConfigurationEnsemble{P,C,E,T<:Real} <: AbstractConfigurationEnsemble
    problem::P
    configurations::C
    energies::E
    temperature::T
    kB::T
    metadata::Dict{Symbol,Any}
end

Base.length(ensemble::ConfigurationEnsemble) = size(ensemble.configurations, 2)
nsamples(ensemble::ConfigurationEnsemble) = length(ensemble)
configuration(ensemble::ConfigurationEnsemble, i::Integer) =
    view(ensemble.configurations, :, Int(i))
