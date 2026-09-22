# Registry of standard transformation constructors.

const TRANSFORMATION_REGISTRY = Dict{Symbol,Any}(
    :fourier => FourierTransformation,
    :bogoliubov => BogoliubovTransformation,
    :particle_hole => ParticleHoleTransformation,
    :majorana => MajoranaTransformation,
    :jordan_wigner => JordanWigner,
    :holstein_primakoff => HolsteinPrimakoff,
    :dyson_maleev => DysonMaleev,
    :schwinger_boson => SchwingerBoson,
    :abrikosov_fermion => AbrikosovFermion,
    :slave_particle => SlaveParticleTransformation,
    :coordinate_ladder => CoordinateLadder,
    :bosonic_displacement => BosonicDisplacement,
    :lang_firsov => LangFirsov,
    :spin_polaron => SpinPolaron,
)

"""Register or replace a transformation constructor under `name`."""
function register_transformation!(name::Symbol, constructor)
    TRANSFORMATION_REGISTRY[name] = constructor
    return constructor
end

"""Return the registered constructor for a standard transformation."""
function transformation_constructor(name::Symbol)
    haskey(TRANSFORMATION_REGISTRY, name) ||
        throw(KeyError("unknown transformation: $name"))
    return TRANSFORMATION_REGISTRY[name]
end

"""Construct a registered transformation by name."""
function make_transformation(name::Symbol, args...; kwargs...)
    constructor = transformation_constructor(name)
    return constructor(args...; kwargs...)
end

"""Sorted names of transformations currently available in the registry."""
available_transformations() = sort!(collect(keys(TRANSFORMATION_REGISTRY)))
