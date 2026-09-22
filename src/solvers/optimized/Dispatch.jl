"""
    optimized_backend(model; sector=nothing)

Return the exact packed backend naturally associated with the current representation. This does not apply any physical representation transformation.
"""
function optimized_backend(model::ManyBodyModel; sector=nothing)
    alg = model.representation.algebra
    if alg isa SpinAlgebra
        return PackedSpinBackend()
    elseif alg isa FermionAlgebra
        return PackedFermionBackend()
    elseif alg isa BosonAlgebra
        return PackedBosonBackend()
    elseif alg isa CoordinateMomentumAlgebra && haskey(model.parameters, :force_constants) && haskey(model.parameters, :masses)
        return HarmonicPhononBackend()
    end
    return nothing
end

supports_optimized_backend(model::ManyBodyModel; sector=nothing) = !isnothing(optimized_backend(model; sector=sector))

"""
    optimized_matrix_free_operator(model; sector=nothing, backend=:auto, hbar=1,
                                   max_basis_dimension=50_000_000,
                                   materialize_basis=true, threaded=false,
                                   rank_lookup=nothing, populate_rank_lookup=false)

Construct an exact representation-specific matrix-free Hamiltonian operator. The returned named tuple contains `operator`, `basis`, `dimension`, `backend`, and symmetry/backend metadata.

For packed spin-1/2 models, `materialize_basis=false` keeps the full Hilbert space or fixed-`nup` sector implicit and avoids allocating product-state objects, mask arrays, or hash tables. A fixed-`nup` sector may share a dense `Vector{UInt32}` `rank_lookup`, where index `mask+1` stores the one-based rank of that mask in its sector. `populate_rank_lookup=true` fills the entries for the requested sector. This lookup is an optional performance-memory tradeoff and does not alter basis ordering.

When `threaded=true`, a conserved fixed-`S^z` packed-spin sector uses a race-free gather kernel across Julia threads. Other backends retain their established execution path.
"""
function optimized_matrix_free_operator(model::ManyBodyModel; sector=nothing, backend::Symbol=:auto, hbar::Real=1.0, max_basis_dimension::Integer=50_000_000, materialize_basis::Bool=true, threaded::Bool=false, rank_lookup=nothing, populate_rank_lookup::Bool=false)
    backend in (:auto, :spin, :fermion, :boson) || throw(ArgumentError("backend must be :auto, :spin, :fermion, or :boson"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    max_basis_dimension > 0 || throw(ArgumentError("max_basis_dimension must be positive"))
    !isnothing(rank_lookup) && !(rank_lookup isa Vector{UInt32}) && throw(ArgumentError("rank_lookup must be nothing or Vector{UInt32}"))
    sector = specialize_sector(model, sector)

    alg = model.representation.algebra
    selected = backend === :auto ? (alg isa SpinAlgebra ? :spin : alg isa FermionAlgebra ? :fermion : alg isa BosonAlgebra ? :boson : :none) : backend

    if selected === :spin
        alg isa SpinAlgebra || throw(ArgumentError("requested packed-spin backend for $(typeof(alg))"))
        !isnothing(sector) && !(sector isa SpinSector) && throw(ArgumentError("spin backend requires SpinSector or no sector"))
        sector isa SpinSector && !_spin_sector_known_conserved(model, sector) && throw(ArgumentError("cannot certify total-Sz conservation for the requested spin sector"))
        A = _packed_spin_operator(model, sector; hbar=hbar, max_dimension=Int(max_basis_dimension), materialize_basis=materialize_basis, rank_lookup=rank_lookup, populate_rank_lookup=populate_rank_lookup, threaded=threaded)
        return (
            operator=A,
            basis=A.data.basis,
            dimension=A.data.dimension,
            backend=:packed_spin,
            hermitian_certified=_packed_spin_hermitian_certified(A),
            total_sz_conserved=_packed_spin_total_sz_conserved(A),
            global_spin_flip_symmetric=_packed_spin_global_flip_symmetric(A),
            sector=sector,
            rank_lookup=!isnothing(A.data.rank_lookup),
            threaded=A.threaded,
        )
    elseif selected === :fermion
        isnothing(rank_lookup) || throw(ArgumentError("rank_lookup is specific to the packed-spin backend"))
        populate_rank_lookup && throw(ArgumentError("populate_rank_lookup is specific to the packed-spin backend"))
        threaded && throw(ArgumentError("threaded matrix-free execution is currently implemented only for conserved packed-spin sectors"))
        alg isa FermionAlgebra || throw(ArgumentError("requested packed-fermion backend for $(typeof(alg))"))
        !isnothing(sector) && !(sector isa FermionSector) && throw(ArgumentError("fermion backend requires FermionSector or no sector"))
        sector isa FermionSector && !_fermion_sector_known_conserved(model, sector) && throw(ArgumentError("requested fermion-number sector is not conserved by this Hamiltonian"))
        A = _packed_fermion_operator(model, sector; max_dimension=Int(max_basis_dimension))
        return (operator=A, basis=A.data.basis, dimension=length(A.data.basis), backend=:packed_fermion, hermitian_certified=false)
    elseif selected === :boson
        isnothing(rank_lookup) || throw(ArgumentError("rank_lookup is specific to the packed-spin backend"))
        populate_rank_lookup && throw(ArgumentError("populate_rank_lookup is specific to the packed-spin backend"))
        threaded && throw(ArgumentError("threaded matrix-free execution is currently implemented only for conserved packed-spin sectors"))
        alg isa BosonAlgebra || throw(ArgumentError("requested packed-boson backend for $(typeof(alg))"))
        isnothing(sector) || throw(ArgumentError("packed-boson backend does not currently accept an additional symmetry sector"))
        A = _packed_boson_operator(model; max_dimension=Int(max_basis_dimension))
        return (operator=A, basis=A.data.basis, dimension=length(A.data.basis), backend=:packed_boson, hermitian_certified=false)
    end

    throw(ArgumentError("no exact packed matrix-free backend is available for $(typeof(alg))"))
end

function solve(solver::OptimizedLanczos, model::ManyBodyModel)
    capabilities = numerical_capabilities(model)
    capabilities.numerically_realizable || throw(ArgumentError(
        "OptimizedLanczos cannot construct the numerical target representation: $(join(capabilities.reasons, "; "))"
    ))
    capabilities.biorthogonal && throw(ArgumentError(
        "OptimizedLanczos is Hermitian and is not compatible with a biorthogonal/non-Hermitian representation; use ExactDiagonalization(hermitian=:no)"
    ))

    alg = model.representation.algebra
    selected = solver.backend

    if selected === :spin || (selected === :auto && alg isa SpinAlgebra)
        return _solve_packed_spin(solver, model)
    elseif selected === :fermion || (selected === :auto && alg isa FermionAlgebra)
        return _solve_packed_fermion(solver, model)
    elseif selected === :boson || (selected === :auto && alg isa BosonAlgebra)
        return _solve_packed_boson(solver, model)
    end

    solver.fallback || throw(ArgumentError("no optimized Lanczos backend is available for $(typeof(alg))"))
    isnothing(solver.sector) || throw(ArgumentError("generic fallback cannot preserve the requested optimized symmetry sector"))

    generic = LanczosSolver(nev=solver.nev, krylov_dim=solver.krylov_dim, tol=solver.tol, matrix_free=true, reorthogonalize=solver.reorthogonalize, seed=solver.seed, hbar=solver.hbar)
    result = solve(generic, model)
    metadata = copy(result.metadata)
    metadata[:solver] = :optimized_lanczos
    metadata[:backend] = :generic_symbolic_fallback
    metadata[:requested_solver] = :optimized_lanczos
    metadata[:approximation] = :none
    return SolverResult(solver, model, result.energies, result.right_states, result.left_states, result.basis, metadata)
end

function _is_quadratic_fermion(model::ManyBodyModel, tol)
    try
        _fermion_quadratic_matrices(model; tol=tol)
        return true
    catch
        return false
    end
end

function _is_quadratic_boson(model::ManyBodyModel, tol)
    try
        _boson_quadratic_matrices(model; tol=tol)
        return true
    catch
        return false
    end
end

"""
    solve(AutoSolver(), model)

Exact-specialization dispatcher. It never selects an approximation automatically. In particular, linear spin-wave theory must be requested explicitly with `LinearSpinWaveSolver`.
"""
function solve(solver::AutoSolver, model::ManyBodyModel)
    alg = model.representation.algebra

    if alg isa CoordinateMomentumAlgebra && haskey(model.parameters, :force_constants) && haskey(model.parameters, :masses)
        return solve(HarmonicPhononSolver(tol=solver.tol), model)
    end

    if _contains_biorthogonal_space(model.representation.state_space)
        capabilities = numerical_capabilities(model)
        capabilities.numerically_realizable || throw(ArgumentError(
            "AutoSolver cannot construct the biorthogonal numerical target representation: $(join(capabilities.reasons, "; "))"
        ))
        isnothing(solver.sector) || throw(ArgumentError("AutoSolver cannot apply an optimized symmetry sector to a biorthogonal representation"))
        return solve(ExactDiagonalization(hermitian=:no, hbar=solver.hbar), model)
    elseif alg isa FermionAlgebra && _is_quadratic_fermion(model, solver.tol)
        return solve(QuadraticFermionSolver(tol=solver.tol), model)
    elseif alg isa BosonAlgebra && _is_quadratic_boson(model, solver.tol)
        return solve(QuadraticBosonSolver(tol=solver.tol), model)
    end

    capabilities = numerical_capabilities(model)
    capabilities.numerically_realizable || throw(ArgumentError(
        "AutoSolver cannot construct the numerical target representation: $(join(capabilities.reasons, "; "))"
    ))
    return solve(OptimizedLanczos(nev=solver.nev, krylov_dim=solver.krylov_dim, tol=solver.tol, sector=solver.sector, seed=solver.seed, hbar=solver.hbar, fallback=solver.fallback), model)
end
