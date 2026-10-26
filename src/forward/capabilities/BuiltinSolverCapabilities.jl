# Built-in solver-specific capability rules. Public extensions belong to the `CapabilityDecision` hooks.

# Model-level numerical rules
function _solver_model_support(solver::ExactDiagonalization, model::ManyBodyModel)
    supported, reasons = _finite_basis_solver_support(model)
    extra = String[]
    capabilities = numerical_capabilities(model)
    if capabilities.biorthogonal && solver.hermitian === :yes
        supported = false
        push!(extra, "ExactDiagonalization(hermitian=:yes) cannot be used with a biorthogonal representation")
    end
    return supported, _compatibility_reasons(reasons, extra), true
end

function _solver_model_support(solver::LanczosSolver, model::ManyBodyModel)
    supported, reasons = _finite_basis_solver_support(model)
    extra = String[]
    capabilities = numerical_capabilities(model)
    if capabilities.biorthogonal
        supported = false
        push!(extra, "LanczosSolver requires a Hermitian, non-biorthogonal numerical representation")
    end
    if solver.matrix_free && !(model.hamiltonian isa AbstractOperatorExpr)
        supported = false
        push!(extra, "matrix-free Lanczos requires a symbolic operator-expression Hamiltonian")
    end
    return supported, _compatibility_reasons(reasons, extra), true
end

function _solver_model_support(::TimeEvolutionSolver, model::ManyBodyModel)
    supported, reasons = _finite_basis_solver_support(model)
    return supported, reasons, true
end

function _optimized_sector_support(model::ManyBodyModel, sector)
    isnothing(sector) && return true, (), true
    resolved = try
        Solvers.specialize_sector(model, sector)
    catch error
        return false, (sprint(showerror, error),), true
    end

    algebra = model.representation.algebra
    if algebra isa SpinAlgebra
        resolved isa SpinSector || return false, ("packed spin execution requires a SpinSector-compatible descriptor",), true
    elseif algebra isa FermionAlgebra
        resolved isa FermionSector || return false, ("packed fermion execution requires a FermionSector-compatible descriptor",), true
    elseif algebra isa CompositeAlgebra && _contains_fermion_algebra(algebra)
        resolved isa FermionSector || return false, ("composite fermion execution requires a FermionSector-compatible descriptor",), true
    else
        return false, ("the selected representation has no optimized backend that accepts an additional symmetry sector",), true
    end

    return true, ("sector conservation and operator closure are checked by the specialized backend at execution",), false
end

function _solver_model_support(solver::OptimizedLanczos, model::ManyBodyModel)
    supported, reasons = _finite_basis_solver_support(model)
    extra = String[]
    capabilities = numerical_capabilities(model)
    if capabilities.biorthogonal
        supported = false
        push!(extra, "OptimizedLanczos requires a Hermitian, non-biorthogonal numerical representation")
    end

    sector_supported, sector_reasons, sector_certified = _optimized_sector_support(model, solver.sector)
    if !sector_supported
        supported = false
    end
    append!(extra, sector_reasons)

    algebra = model.representation.algebra
    if solver.backend === :spin && !(algebra isa SpinAlgebra)
        supported = false
        push!(extra, "backend=:spin requires a SpinAlgebra representation")
    elseif solver.backend === :boson && !(algebra isa BosonAlgebra)
        supported = false
        push!(extra, "backend=:boson requires a BosonAlgebra representation")
    elseif solver.backend === :fermion
        fermion_compatible = algebra isa FermionAlgebra
        if algebra isa CompositeAlgebra && _contains_fermion_algebra(algebra)
            sector = try
                Solvers.specialize_sector(model, solver.sector)
            catch
                nothing
            end
            fermion_compatible = sector isa FermionSector
        end
        if !fermion_compatible
            supported = false
            push!(extra, "backend=:fermion requires a fermionic representation or a composite model with a resolved FermionSector")
        end
    elseif solver.backend === :auto && !solver.fallback
        direct_backend = algebra isa SpinAlgebra || algebra isa FermionAlgebra || algebra isa BosonAlgebra
        if algebra isa CompositeAlgebra && _contains_fermion_algebra(algebra)
            sector = try
                Solvers.specialize_sector(model, solver.sector)
            catch
                nothing
            end
            direct_backend = sector isa FermionSector
        end
        if !direct_backend
            supported = false
            push!(extra, "no exact packed OptimizedLanczos backend is available and fallback=false")
        end
    end

    return supported, _compatibility_reasons(reasons, extra), sector_certified
end

function _solver_model_support(solver::QuadraticFermionSolver, model::ManyBodyModel)
    model.representation.algebra isa FermionAlgebra || return false, ("QuadraticFermionSolver requires a FermionAlgebra representation",), true
    Solvers._is_quadratic_fermion(model, solver.tol) || return false, ("the Hamiltonian is not recognized as exactly quadratic fermionic within the solver tolerance",), true
    return true, (), true
end

function _solver_model_support(solver::QuadraticBosonSolver, model::ManyBodyModel)
    model.representation.algebra isa BosonAlgebra || return false, ("QuadraticBosonSolver requires a BosonAlgebra representation",), true
    Solvers._is_quadratic_boson(model, solver.tol) || return false, ("the Hamiltonian is not recognized as exactly quadratic bosonic within the solver tolerance",), true
    return true, (), true
end

function _harmonic_phonon_model_support(model::ManyBodyModel)
    model.representation.algebra isa CoordinateMomentumAlgebra || return false, ("HarmonicPhononSolver requires a CoordinateMomentumAlgebra representation",)
    haskey(model.parameters, :masses) || return false, ("harmonic phonon calculation requires model.parameters[:masses]",)
    has_force_constants = haskey(model.parameters, :force_constants) || haskey(model.parameters, :periodic_force_constants)
    has_force_constants || return false, ("harmonic phonon calculation requires force-constant metadata",)
    return true, ()
end

function _solver_model_support(::HarmonicPhononSolver, model::ManyBodyModel)
    supported, reasons = _harmonic_phonon_model_support(model)
    return supported, reasons, true
end

function _solver_model_support(solver::LinearSpinWaveSolver, model::ManyBodyModel)
    model.representation.algebra isa SpinAlgebra || return false, ("LinearSpinWaveSolver requires a SpinAlgebra representation",), true
    if !isnothing(solver.reference_signs) && length(solver.reference_signs) != model.representation.algebra.nsites
        return false, ("LinearSpinWaveSolver reference_signs length does not match the number of spin sites",), true
    end
    try
        Solvers._spinwave_model_parameters(model)
    catch error
        return false, (sprint(showerror, error),), true
    end
    return true, (), true
end

function _hubbard_forward_model_support(model::ManyBodyModel)
    try
        _single_band_hubbard_forward_parameters(model)
        return true, (), true
    catch error
        return false, (sprint(showerror, error),), true
    end
end

_solver_model_support(::DMFTSolver, model::ManyBodyModel) = _hubbard_forward_model_support(model)
_solver_model_support(::DiagrammaticDMFTSolver, model::ManyBodyModel) = _hubbard_forward_model_support(model)

function _solver_model_support(::NLCESolver, model::ManyBodyModel)
    material = get(model.parameters, :material, nothing)
    reference = get(model.parameters, :cluster, nothing)
    specification = get(model.parameters, :hamiltonian_spec, nothing)
    if material isa Material && reference isa Supercell && specification isa AbstractHamiltonianSpec
        return true, (), true
    end
    if model.representation.algebra isa SpinAlgebra && !isnothing(Models._structural_provenance(model)) &&
       reference isa Supercell && material isa Material
        return true, ("structural spin cluster regeneration requires StructuralNLCEForwardContext",), true
    end
    if model.representation.algebra isa FermionAlgebra && reference isa Supercell && material isa Material
        return true, ("fermionic source reconstruction requires a certified FermionicNLCEForwardContext",), false
    end
    reasons = String[]
    material isa Material || push!(reasons, "NLCE forward integration requires model.parameters[:material]")
    reference isa Supercell || push!(reasons, "NLCE forward integration requires model.parameters[:cluster] to be a Supercell")
    specification isa AbstractHamiltonianSpec || push!(reasons, "NLCE forward integration requires model.parameters[:hamiltonian_spec]")
    return false, Tuple(reasons), true
end

function _solver_model_support(solver::AutoSolver, model::ManyBodyModel)
    algebra = model.representation.algebra
    if algebra isa CoordinateMomentumAlgebra && haskey(model.parameters, :force_constants) && haskey(model.parameters, :masses)
        isnothing(solver.sector) || return false, ("AutoSolver harmonic specialization does not accept a symmetry sector",), true
        return true, (), true
    end

    capabilities = numerical_capabilities(model)
    capabilities.numerically_realizable || return false, capabilities.reasons, true
    if capabilities.biorthogonal
        isnothing(solver.sector) || return false, ("AutoSolver cannot apply an optimized symmetry sector to a biorthogonal representation",), true
        return true, capabilities.reasons, true
    end
    if algebra isa FermionAlgebra && Solvers._is_quadratic_fermion(model, solver.tol)
        isnothing(solver.sector) || return false, ("AutoSolver quadratic-fermion specialization does not apply a many-body symmetry sector",), true
        return true, capabilities.reasons, true
    elseif algebra isa BosonAlgebra && Solvers._is_quadratic_boson(model, solver.tol)
        isnothing(solver.sector) || return false, ("AutoSolver quadratic-boson specialization does not apply a many-body symmetry sector",), true
        return true, capabilities.reasons, true
    end

    optimized = OptimizedLanczos(nev=solver.nev, krylov_dim=solver.krylov_dim, tol=solver.tol, sector=solver.sector, seed=solver.seed,
                                 hbar=solver.hbar, fallback=solver.fallback)
    return _solver_model_support(optimized, model)
end

# Built-in solver result families
_solver_output_family(::ExactDiagonalization, ::ManyBodyModel) = :eigensystem
_solver_output_family(::LanczosSolver, ::ManyBodyModel) = :eigensystem
_solver_output_family(::OptimizedLanczos, ::ManyBodyModel) = :eigensystem
_solver_output_family(::TimeEvolutionSolver, ::ManyBodyModel) = :time_evolution
_solver_output_family(::QuadraticFermionSolver, ::ManyBodyModel) = :quadratic_fermion_mode
_solver_output_family(::QuadraticBosonSolver, ::ManyBodyModel) = :quadratic_boson_mode
_solver_output_family(::HarmonicPhononSolver, ::ManyBodyModel) = :phonon_mode
_solver_output_family(::LinearSpinWaveSolver, ::ManyBodyModel) = :spin_wave
_solver_output_family(::DMFTSolver, ::ManyBodyModel) = :dmft
_solver_output_family(::DiagrammaticDMFTSolver, ::ManyBodyModel) = :ladder_dga
_solver_output_family(::NLCESolver, ::ManyBodyModel) = :nlce

# Automatic solver result family
function _solver_output_family(solver::AutoSolver, model::ManyBodyModel)
    algebra = model.representation.algebra
    if algebra isa CoordinateMomentumAlgebra && haskey(model.parameters, :force_constants) && haskey(model.parameters, :masses)
        return :phonon_mode
    end
    capabilities = numerical_capabilities(model)
    capabilities.biorthogonal && return :eigensystem
    algebra isa FermionAlgebra && Solvers._is_quadratic_fermion(model, solver.tol) && return :quadratic_fermion_mode
    algebra isa BosonAlgebra && Solvers._is_quadratic_boson(model, solver.tol) && return :quadratic_boson_mode
    return :eigensystem
end

# Solver-specific request and sector rules
function _solver_request_support(solver::OptimizedLanczos, model::ManyBodyModel, ::Observables.LocalElectronSpectralFunctionRequest)
    _resolved_fermion_sector(model, solver.sector) || return false, ("local electronic spectral requests require OptimizedLanczos with a resolvable FermionSector",), true
    return true, (), true
end

function _solver_request_support(solver::AutoSolver, model::ManyBodyModel, ::Observables.LocalElectronSpectralFunctionRequest)
    _solver_output_family(solver, model) === :eigensystem || return false, ("AutoSolver selects a mode result for this model, but LocalElectronSpectralFunctionRequest requires a FermionSector SolverResult",), true
    _resolved_fermion_sector(model, solver.sector) || return false, ("local electronic spectral requests require AutoSolver with a resolvable FermionSector",), true
    return true, (), true
end

function _solver_request_support(solver::OptimizedLanczos, model::ManyBodyModel, request::AbstractObservable)
    family = _solver_output_family(solver, model)
    supported, reasons, certified = _request_family_support(request, family)
    if supported && !isnothing(solver.sector)
        reasons = _compatibility_reasons(reasons, ("observable closure inside the requested symmetry sector is verified at execution",))
        certified = false
    end
    return supported, reasons, certified
end

function _solver_request_support(solver::AutoSolver, model::ManyBodyModel, request::AbstractObservable)
    family = _solver_output_family(solver, model)
    supported, reasons, certified = _request_family_support(request, family)
    if supported && !isnothing(solver.sector)
        reasons = _compatibility_reasons(reasons, ("observable closure inside the selected symmetry sector is verified at execution",))
        certified = false
    end
    return supported, reasons, certified
end

function _solver_request_support(::HarmonicPhononSolver, ::ManyBodyModel, request::Observables.PhononStructureFactorRequest)
    return true, (), true
end

function _solver_request_support(::DiagrammaticDMFTSolver, ::ManyBodyModel, ::ObservableRequest)
    return false, ("the Batch 5 high-level ladder-DΓA route requires MatsubaraSusceptibilityRequest so the response channel and Matsubara-domain semantics remain explicit",), true
end

function _solver_request_support(::NLCESolver, ::ManyBodyModel, ::ObservableRequest)
    return true, ("NLCE recursively evaluates the opaque observable on each generated cluster; additive-value closure is checked at execution",), false
end

function _solver_request_support(::NLCESolver, ::ManyBodyModel,
                                 ::Union{ThermodynamicCurveRequest,MagneticSusceptibilityRequest,MagnetizationCurveRequest})
    return true, ("typed extensive curves are evaluated per cluster and combined by inclusion-exclusion",), true
end

function _solver_request_support(::NLCESolver, ::ManyBodyModel, ::AbstractObservable)
    return false, ("the Batch 5 NLCE forward route currently supports ObservableRequest evaluators whose cluster values are additive scalars or arrays",), true
end

# Built-in method-context rules
function _forward_context_support(::DMFTSolver, ::ManyBodyModel, request::AbstractObservable, context::DMFTForwardContext)
    supported, reasons = _request_context_temperature_support(request, context.temperature)
    return supported, reasons, true
end

_forward_context_support(::DMFTSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("high-level DMFT forward execution requires DMFTForwardContext with an explicit lattice embedding and Matsubara domain",), true)

function _forward_context_support(solver::DiagrammaticDMFTSolver, ::ManyBodyModel, request::AbstractObservable, context::LadderDGAForwardContext)
    solver.method isa LadderDGA || return false, ("LadderDGAForwardContext requires DiagrammaticDMFTSolver(method=LadderDGA(...))",), true
    supported, reasons = _request_context_temperature_support(request, context.temperature)
    return supported, reasons, true
end

_forward_context_support(::DiagrammaticDMFTSolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("high-level ladder-DΓA forward execution requires LadderDGAForwardContext with explicit DMFT and momentum-lattice context",), true)

function _forward_context_support(::NLCESolver, model::ManyBodyModel, request::AbstractObservable, context::NLCEForwardContext)
    if model.representation.algebra isa FermionAlgebra &&
       !(get(model.parameters, :hamiltonian_spec, nothing) isa AbstractHamiltonianSpec)
        return false, ("fermionic source without a standard model specification requires FermionicNLCEForwardContext",), true
    end
    inner = solver_compatibility(context.inner_solver, model, request, context.inner_method_context)
    reasons = inner.compatible ? ("inner solver compatibility is established against the source model; cluster-by-cluster closure is checked at execution",) :
                                 _compatibility_reasons(("the configured NLCE inner forward route is incompatible with the source model/request",), inner.reasons)
    return inner.compatible, reasons, false
end

_forward_context_support(::NLCESolver, ::ManyBodyModel, ::AbstractObservable, ::Nothing) =
    (false, ("high-level NLCE forward execution requires NLCEForwardContext with an explicit expansion order and inner solver",), true)
