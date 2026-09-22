# Representation-specific optimized solver layer.
#
# Included directly inside `Phundamental.Solvers`; no extra physical
# representation namespace is introduced.

include("BackendTypes.jl")
include("OperatorCompilation.jl")

include("Sectors/SectorTypes.jl")
include("Sectors/SymmetryBasis.jl")

include("Krylov/KrylovTypes.jl")
include("Krylov/LanczosKernel.jl")
include("Krylov/ContinuedFraction.jl")
include("Krylov/ExpAction.jl")

include("Spin/BitBasis.jl")
include("Spin/SpinKernels.jl")
include("Spin/SpinLanczos.jl")

include("Fermion/BitFockBasis.jl")
include("Fermion/FermionKernels.jl")
include("Fermion/FermionLanczos.jl")
include("Fermion/FermionGreenFunctions.jl")

include("Boson/PackedBosonBasis.jl")
include("Boson/BosonKernels.jl")

include("Quadratic/QuadraticTypes.jl")
include("Quadratic/FermionicBdG.jl")
include("Quadratic/BosonicBdG.jl")
include("Quadratic/NormalModes.jl")

include("Phonons/DynamicalMatrix.jl")
include("Phonons/PhononModes.jl")

include("SpinWave/LinearSpinWave.jl")
include("SpinWave/SpinWaveSpectrum.jl")

include("Dispatch.jl")
include("Validation.jl")
