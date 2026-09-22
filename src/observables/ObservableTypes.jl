# Common observable descriptors, conventions, and result containers.

abstract type AbstractObservable end
abstract type AbstractCorrelationObservable <: AbstractObservable end
abstract type AbstractProbe end
abstract type AbstractBroadening end
abstract type AbstractResolution end
abstract type AbstractBosonZeroModePolicy end

"""
    RejectBosonZeroModes(; relative_tol=1e-10)

Default finite-temperature policy for mode-native bosonic response. Any quasiparticle energy at or below `relative_tol * max(maximum(abs, energies), 1)` raises an error because the Bose occupation diverges as the energy approaches zero.
"""
struct RejectBosonZeroModes <: AbstractBosonZeroModePolicy
    relative_tol::Float64
    function RejectBosonZeroModes(; relative_tol::Real=1e-10)
        relative_tol >= 0 || throw(ArgumentError("zero-mode tolerance must be nonnegative"))
        new(Float64(relative_tol))
    end
end

"""
    NumberConservingZeroModeProjection(; relative_tol=1e-10)

Explicit number-conserving policy that removes numerically zero-energy quasiparticle modes from mode-native contractions and spectral lines at all temperatures. This represents the projected noncondensate sector used by number-conserving Bogoliubov formulations; use it only when the omitted mode is known to be the condensate/Goldstone direction. Solver-side `BosonSubspaceProjection` is preferred when the condensate orbital is known because it removes that direction before diagonalization. This policy does not implement a self-consistent finite-temperature condensate/noncondensate evolution scheme.
"""
struct NumberConservingZeroModeProjection <: AbstractBosonZeroModePolicy
    relative_tol::Float64
    function NumberConservingZeroModeProjection(; relative_tol::Real=1e-10)
        relative_tol >= 0 || throw(ArgumentError("zero-mode tolerance must be nonnegative"))
        new(Float64(relative_tol))
    end
end

_zero_mode_relative_tol(policy::RejectBosonZeroModes) = policy.relative_tol
_zero_mode_relative_tol(policy::NumberConservingZeroModeProjection) = policy.relative_tol

"""Reference to an observable registered in `model.observables`."""
struct ObservableRef <: AbstractObservable
    name::Symbol
end
ObservableRef(name::AbstractString) = ObservableRef(Symbol(name))

"""
    SpectrumConvention(; fourier_normalization=:sqrtN,
                         spectral_axis=:energy,
                         hbar=1.0,
                         kB=1.0)

Conventions used by spectral observables.

`fourier_normalization` controls the amplitude in

    O_q = a_N sum_j exp(-i q.r_j) O_j

with `:sqrtN => a_N=1/sqrt(N)`, `:N => a_N=1/N`, and `:none => a_N=1`.
The dynamic structure factor therefore scales as `abs2(a_N)`.

`spectral_axis` is either `:energy` (line positions E_n-E_m) or `:omega`
(line positions (E_n-E_m)/hbar).  Broadenings use the same units as the
selected axis and are normalized to unit integral on that axis.

`kB` must be expressed in energy units per kelvin if `temperature` is supplied
in kelvin. For meV, a useful value is approximately 0.08617333262 meV/K.
"""
struct SpectrumConvention{T<:Real}
    fourier_normalization::Symbol
    spectral_axis::Symbol
    hbar::T
    kB::T

    function SpectrumConvention(
        ; fourier_normalization::Symbol=:sqrtN,
          spectral_axis::Symbol=:energy,
          hbar::Real=1.0,
          kB::Real=1.0,
    )
        fourier_normalization in (:sqrtN, :N, :none) ||
            throw(ArgumentError("fourier_normalization must be :sqrtN, :N, or :none"))
        spectral_axis in (:energy, :omega) ||
            throw(ArgumentError("spectral_axis must be :energy or :omega"))
        hbar > 0 || throw(ArgumentError("hbar must be positive"))
        kB > 0 || throw(ArgumentError("kB must be positive"))
        T = promote_type(typeof(float(hbar)), typeof(float(kB)))
        new{T}(fourier_normalization, spectral_axis, T(hbar), T(kB))
    end
end

"""A field of registered local observables with real-space positions."""
struct MomentumField{K,R} <: AbstractObservable
    keys::K
    positions::R
    normalization::Symbol

    function MomentumField(keys, positions; normalization::Symbol=:sqrtN)
        length(keys) == length(positions) ||
            throw(DimensionMismatch("observable keys and positions must have the same length"))
        isempty(keys) && throw(ArgumentError("a momentum field requires at least one local observable"))
        normalization in (:sqrtN, :N, :none) ||
            throw(ArgumentError("normalization must be :sqrtN, :N, or :none"))
        new{typeof(collect(keys)),typeof(collect(positions))}(
            collect(keys), collect(positions), normalization,
        )
    end
end

"""
Three local spin-component fields used to construct the full tensor
`S^{alpha beta}(q, axis)`.  Each key must resolve through `model.observables`;
therefore an exact representation transformation automatically supplies the
transformed scattering operators when registered observables were transformed
with the model.
"""
struct SpinTensorField{FX,FY,FZ} <: AbstractObservable
    x::FX
    y::FY
    z::FZ
end

function SpinTensorField(
    ; xkeys,
      ykeys,
      zkeys,
      positions,
      normalization::Symbol=:sqrtN,
)
    return SpinTensorField(
        MomentumField(xkeys, positions; normalization=normalization),
        MomentumField(ykeys, positions; normalization=normalization),
        MomentumField(zkeys, positions; normalization=normalization),
    )
end

"""Discrete Lehmann transitions before numerical line broadening."""
struct LehmannLines{X,W,M}
    centers::X
    weights::W
    metadata::M
end

"""Scalar spectrum on a q-by-axis grid."""
struct SpectrumResult{Q,X,V,M}
    q::Q
    axis::X
    intensity::V
    metadata::M
end

"""Static/equal-time structure factor on a momentum grid."""
struct StaticStructureFactorResult{Q,V,M}
    q::Q
    intensity::V
    metadata::M
end

"""Full Cartesian tensor spectrum with values indexed `(iq, ix, alpha, beta)`."""
struct TensorSpectrumResult{Q,X,V,M}
    q::Q
    axis::X
    intensity::V
    metadata::M
end

"""Time-domain correlation result."""
struct CorrelationResult{T,V,M}
    times::T
    values::V
    metadata::M
end

"""Probe-specific scattering intensity on a q-by-axis grid."""
struct ScatteringResult{Q,X,V,M}
    q::Q
    axis::X
    intensity::V
    metadata::M
end

"""Resolve a registered observable in the model's current representation."""
function resolve_observable(model::ManyBodyModel, ref::ObservableRef)
    haskey(model.observables, ref.name) ||
        throw(KeyError("observable $(ref.name) is not registered in this model"))
    op = model.observables[ref.name]
    op isa AbstractOperatorExpr ||
        throw(ArgumentError("registered observable $(ref.name) is not an operator expression"))
    return op
end

resolve_observable(model::ManyBodyModel, name::Symbol) =
    resolve_observable(model, ObservableRef(name))
resolve_observable(::ManyBodyModel, op::AbstractOperatorExpr) = op

function _fourier_amplitude(normalization::Symbol, N::Integer)
    N > 0 || throw(ArgumentError("N must be positive"))
    normalization === :sqrtN && return inv(sqrt(float(N)))
    normalization === :N && return inv(float(N))
    normalization === :none && return 1.0
    throw(ArgumentError("unknown Fourier normalization $normalization"))
end

_phase(q::Number, r::Number) = exp(-1im * q * r)
function _phase(q, r)
    length(q) == length(r) || throw(DimensionMismatch("q and position vectors must have the same dimension"))
    return exp(-1im * sum(qi * ri for (qi, ri) in zip(q, r)))
end

_negate_q(q::Number) = -q
_negate_q(q) = map(x -> -x, q)

"""
    momentum_operator(model, field, q)

Construct the representation-correct `O_q` from observables registered in the
current model.  Using registry keys instead of newly-created source operators
is the safe route after a representation transformation.
"""
function momentum_operator(model::ManyBodyModel, field::MomentumField, q)
    N = length(field.keys)
    aN = _fourier_amplitude(field.normalization, N)
    terms = AbstractOperatorExpr[]
    sizehint!(terms, N)
    for (key, position) in zip(field.keys, field.positions)
        op = key isa ObservableRef ? resolve_observable(model, key) :
             key isa Symbol ? resolve_observable(model, key) :
             key isa AbstractOperatorExpr ? key :
             throw(ArgumentError("MomentumField keys must be Symbol, ObservableRef, or operator expression"))
        push!(terms, (aN * _phase(q, position)) * op)
    end
    return length(terms) == 1 ? terms[1] : OperatorSum(terms)
end


"""Return a copy of `model` with one operator observable registered."""
function register_observable(model::ManyBodyModel, name::Symbol, op::AbstractOperatorExpr)
    obs = Dict{Symbol,Any}()
    for (key, value) in pairs(model.observables)
        obs[key isa Symbol ? key : Symbol(string(key))] = value
    end
    obs[name] = op
    return ManyBodyModel(
        model.representation,
        model.hamiltonian;
        parameters=model.parameters,
        observables=obs,
        provenance=model.provenance,
        specification=model.specification,
        model_space=model.model_space,
        parameterization=model.parameterization,
    )
end

"""
    register_spin_tensor(model, sites, positions; prefix=:spin, normalization=:sqrtN)

Register `Sx(i)`, `Sy(i)`, and `Sz(i)` as individual model observables and
return `(updated_model, field)`. Call this on the *spin representation before*
applying a representation transformation. Because the transformation layer
maps every registered operator observable, the returned `SpinTensorField`
continues to address the correct transformed operators afterward.
"""
function register_spin_tensor(
    model::ManyBodyModel,
    sites,
    positions;
    prefix::Symbol=:spin,
    normalization::Symbol=:sqrtN,
)
    model.representation.algebra isa SpinAlgebra || throw(ArgumentError(
        "register_spin_tensor must be called on a direct SpinAlgebra model before representation transformation"
    ))
    sitevec = Int.(collect(sites))
    posvec = collect(positions)
    length(sitevec) == length(posvec) || throw(DimensionMismatch("sites and positions must have the same length"))

    obs = Dict{Symbol,Any}()
    for (key, value) in pairs(model.observables)
        obs[key isa Symbol ? key : Symbol(string(key))] = value
    end
    xkeys = Symbol[]; ykeys = Symbol[]; zkeys = Symbol[]
    for site in sitevec
        kx = Symbol(string(prefix), "_x_", string(site))
        ky = Symbol(string(prefix), "_y_", string(site))
        kz = Symbol(string(prefix), "_z_", string(site))
        obs[kx] = Sx(site); obs[ky] = Sy(site); obs[kz] = Sz(site)
        push!(xkeys, kx); push!(ykeys, ky); push!(zkeys, kz)
    end

    updated = ManyBodyModel(
        model.representation,
        model.hamiltonian;
        parameters=model.parameters,
        observables=obs,
        provenance=model.provenance,
        specification=model.specification,
        model_space=model.model_space,
        parameterization=model.parameterization,
    )
    field = SpinTensorField(
        xkeys=xkeys, ykeys=ykeys, zkeys=zkeys,
        positions=posvec, normalization=normalization,
    )
    return updated, field
end

function spin_tensor_operators(model::ManyBodyModel, field::SpinTensorField, q)
    return (
        momentum_operator(model, field.x, q),
        momentum_operator(model, field.y, q),
        momentum_operator(model, field.z, q),
    )
end

"""
    ObservableRequest(name, evaluator; metadata=Dict())

Typed intrinsic-observable request for `ForwardProblem`. `evaluator` is called
as `evaluator(model, solution)` and may capture grids, temperature,
broadening, normalization conventions, or other observable-specific settings.
"""
struct ObservableRequest{F,M} <: AbstractObservable
    name::Symbol
    evaluator::F
    metadata::M
end

ObservableRequest(name::Symbol, evaluator; metadata=Dict{Symbol,Any}()) = ObservableRequest(name, evaluator, metadata)
(request::ObservableRequest)(model::ManyBodyModel, solution) = request.evaluator(model, solution)
