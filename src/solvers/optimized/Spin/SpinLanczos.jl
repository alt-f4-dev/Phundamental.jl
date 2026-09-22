# Packed-spin Lanczos solver and exact symmetry certification for spin-1/2 sectors.

@inline function _compiled_spin_term_nup_delta(term::_CompiledTerm)
    delta = 0
    @inbounds for primitive in term.primitives
        code = primitive.code
        if code == _OP_IDENTITY || code == _OP_SPIN_Z
            continue
        elseif code == _OP_SPIN_P
            delta += 1
        elseif code == _OP_SPIN_M
            delta -= 1
        else
            return nothing
        end
    end
    return delta
end

function _packed_spin_total_sz_conserved(A::PackedSpinOperator)
    all(kernel -> iszero(kernel.flip_up) && iszero(kernel.flip_down), A.singles) || return false
    all(kernel -> iszero(kernel.flip_same), A.pairs) || return false
    for term in A.residual_terms
        delta = _compiled_spin_term_nup_delta(term)
        isnothing(delta) && return false
        delta == 0 || return false
    end
    return true
end

function _packed_spin_global_flip_symmetric(A::PackedSpinOperator)
    _packed_spin_total_sz_conserved(A) || return false
    isempty(A.residual_terms) || return false
    all(kernel -> iszero(kernel.diagonal_up) && iszero(kernel.flip_up) && iszero(kernel.flip_down), A.singles) || return false
    return true
end

"""
    spin_sector_profile(model; hbar=get(model.parameters, :hbar, 1.0))

Inspect the compiled packed spin-1/2 Hamiltonian and certify the symmetries required for exact fixed-`Sᶻ` decomposition. `conserves_total_sz` is true only when every compiled transition preserves the number of up spins. `global_spin_flip_symmetric` is true only when the compiled Hamiltonian is also certified invariant under global spin inversion, which permits exact pairing of `nup=k` with `nup=N-k` sectors.

The certification is structural: a small but nonzero symmetry-breaking term is not discarded by a numerical tolerance.
"""
function spin_sector_profile(model::ManyBodyModel; hbar::Real=get(model.parameters, :hbar, 1.0))
    model.representation.algebra isa SpinAlgebra || return (supported=false, reason=:not_spin_algebra, nsites=0, conserves_total_sz=false, global_spin_flip_symmetric=false, hermitian_certified=false)
    try
        A = _packed_spin_operator(model, nothing; hbar=hbar, max_dimension=typemax(Int), materialize_basis=false, threaded=false)
        return (
            supported=true,
            reason=:ok,
            nsites=A.data.nsites,
            conserves_total_sz=_packed_spin_total_sz_conserved(A),
            global_spin_flip_symmetric=_packed_spin_global_flip_symmetric(A),
            hermitian_certified=_packed_spin_hermitian_certified(A),
            fused_single_site_kernels=length(A.singles),
            fused_pair_kernels=length(A.pairs),
            residual_compiled_terms=length(A.residual_terms),
        )
    catch err
        if err isa ArgumentError || err isa DimensionMismatch || err isa OverflowError
            return (supported=false, reason=nameof(typeof(err)), nsites=0, conserves_total_sz=false, global_spin_flip_symmetric=false, hermitian_certified=false)
        end
        rethrow()
    end
end

"""
    spin_sector_plan(model; exploit_equivalent_sectors=true, hbar=get(model.parameters, :hbar, 1.0))

Construct an exact fixed-total-`Sᶻ` decomposition plan for a spin-1/2 Hamiltonian. Every retained sector is represented by `SpinSector(nup=k)`. The returned `multiplicities` are one unless global spin inversion has been structurally certified and `exploit_equivalent_sectors=true`, in which case the equivalent `k` and `N-k` sectors are represented once with multiplicity two. For even `N`, the central `k=N/2` sector always has multiplicity one.

The weighted sector dimensions are checked to reconstruct the full Hilbert-space dimension exactly.
"""
function spin_sector_plan(model::ManyBodyModel; exploit_equivalent_sectors::Bool=true, hbar::Real=get(model.parameters, :hbar, 1.0))
    profile = spin_sector_profile(model; hbar=hbar)
    profile.supported || throw(ArgumentError("exact spin-sector decomposition is unavailable for this model ($(profile.reason))"))
    profile.conserves_total_sz || throw(ArgumentError("Hamiltonian does not structurally conserve total S^z"))
    profile.hermitian_certified || throw(ArgumentError("sectorized thermal Lanczos requires a structurally certified Hermitian packed-spin Hamiltonian"))

    N = profile.nsites
    N < Sys.WORD_SIZE - 1 || throw(ArgumentError("full Hilbert dimension 2^N cannot be represented by Int for N=$N"))
    equivalence_used = exploit_equivalent_sectors && profile.global_spin_flip_symmetric
    nup_values = equivalence_used ? collect(0:fld(N, 2)) : collect(0:N)
    sectors = SpinSector[SpinSector(nup=k) for k in nup_values]
    dimensions = [_binomial_checked(N, k) for k in nup_values]
    multiplicities = [equivalence_used && 2 * k != N ? 2 : 1 for k in nup_values]
    total_dimension = Int(1) << N
    reconstructed = sum(multiplicities[i] * dimensions[i] for i in eachindex(dimensions))
    reconstructed == total_dimension || throw(ArgumentError("sector dimensions reconstruct $reconstructed states, expected $total_dimension"))

    return (
        profile=profile,
        nsites=N,
        sectors=sectors,
        nup=nup_values,
        dimensions=dimensions,
        multiplicities=multiplicities,
        total_dimension=total_dimension,
        equivalence_requested=exploit_equivalent_sectors,
        equivalence_used=equivalence_used,
    )
end

function _spin_sector_known_conserved(model::ManyBodyModel, sector::SpinSector)
    profile = spin_sector_profile(model; hbar=get(model.parameters, :hbar, 1.0))
    return profile.supported && profile.conserves_total_sz
end

function _solve_packed_spin(solver::OptimizedLanczos, model::ManyBodyModel)
    sector = specialize_sector(model, solver.sector)
    !isnothing(sector) && !(sector isa SpinSector) && throw(ArgumentError("spin backend requires SpinSector or no sector"))
    if sector isa SpinSector && !_spin_sector_known_conserved(model, sector)
        throw(ArgumentError("cannot certify total-Sz conservation for this model; remove SpinSector or use a Hamiltonian whose compiled spin transitions conserve nup exactly"))
    end

    A = _packed_spin_operator(model, sector; hbar=solver.hbar, max_dimension=solver.max_basis_dimension, materialize_basis=true, threaded=false)
    basis = A.data.basis
    isnothing(basis) && throw(ArgumentError("optimized eigenvector solver requires a materialized computational basis"))
    dim = size(A, 1)
    solver.nev <= dim || throw(ArgumentError("nev=$(solver.nev) exceeds sector dimension $dim"))

    if _packed_spin_hermitian_certified(A)
        herr = 0.0
    else
        ok, herr = _optimized_hermitian_check(A, dim; seed=solver.seed + 17, tol=max(10 * solver.tol, 1e-9))
        ok || throw(ArgumentError("optimized packed-spin Lanczos requires Hermitian projected Hamiltonian; error=$herr"))
    end

    dec = _optimized_lanczos_decomposition(A, dim; krylov_dim=min(dim, max(solver.krylov_dim, solver.nev + 2)), tol=solver.tol, seed=solver.seed, reorthogonalize=solver.reorthogonalize)
    energies, right, residuals = _optimized_ritz(dec, solver.nev; tol=solver.tol)

    metadata = Dict{Symbol,Any}(
        :solver => :optimized_lanczos,
        :backend => :packed_spin,
        :dimension => dim,
        :sector => sector,
        :sector_observables => isnothing(sector) ? :all : :same_sector_only,
        :hermitian => true,
        :hermiticity_error => herr,
        :hermitian_certified => _packed_spin_hermitian_certified(A),
        :biorthogonal => false,
        :iterations => dec.iterations,
        :ritz_residuals => residuals,
        :converged => residuals .<= solver.tol,
        :matrix_free => true,
        :approximation => :none,
        :physical_representation_preserved => true,
        :fused_single_site_kernels => length(A.singles),
        :fused_pair_kernels => length(A.pairs),
        :residual_compiled_terms => length(A.residual_terms),
    )
    return SolverResult(solver, model, energies, right, right, basis, metadata)
end
