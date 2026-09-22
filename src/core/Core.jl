module Core

using LinearAlgebra

include("StateSpaces.jl")
include("Algebras.jl")
include("Operators.jl")
include("Representations.jl")
include("ModelSpaces.jl")
include("Models.jl")
include("ReciprocalSpace.jl")
include("Parameters.jl")

# State spaces
export AbstractStateSpace,
       AbstractHilbertSpace,
       AbstractFockSpace,
       SpinHilbertSpace,
       FermionFockSpace,
       BosonFockSpace,
       CoordinateHilbertSpace,
       CompositeStateSpace,
       BiorthogonalSpace,
       PhysicalSubspace,
       NumericalTruncation,
       spacedimension,
       physicaldimension

# Operator algebras
export AbstractOperatorAlgebra,
       SpinAlgebra,
       BosonAlgebra,
       FermionAlgebra,
       MajoranaAlgebra,
       CoordinateMomentumAlgebra,
       CompositeAlgebra,
       algebra_kind

# Noncommutative operator expressions
export AbstractOperatorExpr,
       AbstractPrimitiveOperator,
       IdentityOperator,
       OperatorSum,
       OperatorProduct,
       ScaledOperator,
       SpinX,
       SpinY,
       SpinZ,
       SpinPlus,
       SpinMinus,
       BosonAnnihilate,
       BosonCreate,
       FermionAnnihilate,
       FermionCreate,
       MajoranaOperator,
       PositionOperator,
       MomentumOperator,
       NumberOperator,
       Sx,
       Sy,
       Sz,
       Sp,
       Sm,
       b,
       c,
       γ,
       xop,
       pop,
       nb,
       nf,
       FactorOperator,
       atfactor,
       isprimitive,
       children,
       operator_degree,
       support_size,
       body_order

# Bases and representations
export AbstractBasis,
       SpinProductBasis,
       FermionOccupationBasis,
       BosonOccupationBasis,
       CoordinateBasis,
       CompositeBasis,
       BiorthogonalBasis,
       AbstractWaveVector,
       CartesianWaveVector,
       ReciprocalWaveVector,
       wavevector_coordinates,
       ReciprocalDecomposition,
       decompose_reciprocal_vector,
       GaugeStructure,
       Representation,
       isconstrained,
       istruncated,
       ambientdimension,
       representationdimension,
       physicalspace,
       FactorConstraint,
       FactorTruncation,
       factorconstraint,
       factortruncation

# Hamiltonian model spaces and typed parameters
export AbstractAdmissibilityCondition,
       PredicateAdmissibility,
       GenerativeSpecification,
       HamiltonianBasis,
       HamiltonianSpace,
       admissible,
       model_space_dimension,
       generated_basis,
       generate_hamiltonian_space,
       hamiltonian_from_coefficients,
       ParameterSpec,
       ParameterSpace,
       ParameterPoint,
       parameter_space,
       parameter_point,
       parameter_names,
       parameter_bounds,
       parameter_units,
       parameter_vector,
       free_parameter_indices,
       rebind_parameters,
       with_parameters

# Models
export ManyBodyModel,
       with_provenance,
       remap_model,
       instantiate_model

end # module Core
