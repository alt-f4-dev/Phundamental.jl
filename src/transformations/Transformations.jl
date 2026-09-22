module Transformations

using LinearAlgebra
using ..Core
using ..Algebra

include("TransformationTypes.jl")
include("TransformationResult.jl")
include("Approximations.jl")

include("Fourier.jl")
include("Bogoliubov.jl")
include("ParticleHole.jl")
include("Majorana.jl")
include("JordanWigner.jl")
include("HolsteinPrimakoff.jl")
include("DysonMaleev.jl")
include("SchwingerBoson.jl")
include("AbrikosovFermion.jl")
include("SlaveParticle.jl")
include("CoordinateLadder.jl")
include("BosonicDisplacement.jl")
include("LangFirsov.jl")
include("SpinPolaron.jl")

include("Registry.jl")
include("TransformationGraph.jl")

export AbstractTransformation,
       RelationType,
       UnitaryEquivalence,
       CanonicalMap,
       AlgebraIsomorphism,
       ConstrainedEmbedding,
       SimilarityEquivalence,
       GaugeRedundantEmbedding,
       CompositeRelation,
       TransformationCertificate,
       TransformationResult,
       SqrtNumberFactor,
       BosonicDisplacementFactor,
       transformation_name,
       certificate,
       applicable,
       target_representation,
       transform_generator,
       transform_operator,
       transform_hamiltonian,
       transform_observable,
       transform,
       compose,
       CompositeTransformation,
       has_inverse,
       inverse,
       FourierTransformation,
       BogoliubovTransformation,
       ParticleHoleTransformation,
       MajoranaTransformation,
       JordanWigner,
       HolsteinPrimakoff,
       DysonMaleev,
       SchwingerBoson,
       AbrikosovFermion,
       SlaveParticleTransformation,
       CoordinateLadder,
       BosonicDisplacement,
       LangFirsov,
       SpinPolaron,
       TRANSFORMATION_REGISTRY,
       register_transformation!,
       transformation_constructor,
       make_transformation,
       available_transformations,
       AbstractApproximation,
       ApproximationCertificate,
       ApproximationResult,
       ModelSpaceProjection,
       approximation_name,
       approximation_certificate,
       approximation_applicable,
       apply_approximation,
       transformation_closure_defect,
       TransformationEdge,
       RepresentationGraph,
       add_representation!,
       add_transformation!,
       outgoing_edges,
       transformation_path,
       representation_class,
       transform_along

end # module Transformations
