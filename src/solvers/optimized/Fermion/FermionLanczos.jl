function _fermion_sector_known_conserved(model::ManyBodyModel, sector::FermionSector)
    # Standard Hubbard and particle-hole transformed number-conserving models
    # can be certified structurally from the expanded terms below. Every term
    # must contain equal creation and annihilation counts globally; nup/ndown
    # additionally require each spin group to be preserved.
    terms = _expanded_operator_terms(model.hamiltonian)
    fmap = get(model.parameters, :fermion_mode_map, nothing)

    mode_spin = Dict{Int,Symbol}()
    if !isnothing(fmap)
        for ((_, sigma), mode) in fmap
            mode_spin[Int(mode)] = sigma
        end
    end

    for term in terms
        delta_total = 0
        delta_up = 0
        delta_down = 0
        for p in term.primitives
            if p.code == _OP_FERMION_C
                delta_total += 1
                get(mode_spin, p.index, :none) === :up && (delta_up += 1)
                get(mode_spin, p.index, :none) === :down && (delta_down += 1)
            elseif p.code == _OP_FERMION_A
                delta_total -= 1
                get(mode_spin, p.index, :none) === :up && (delta_up -= 1)
                get(mode_spin, p.index, :none) === :down && (delta_down -= 1)
            end
        end
        if !isnothing(sector.nparticles)
            delta_total == 0 || return false
        else
            isempty(mode_spin) && return false
            delta_up == 0 && delta_down == 0 || return false
        end
    end
    return true
end

function _solve_packed_fermion(solver::OptimizedLanczos, model::ManyBodyModel)
    sector = specialize_sector(model, solver.sector)
    !isnothing(sector) && !(sector isa FermionSector) &&
        throw(ArgumentError("fermion backend requires FermionSector or no sector"))
    sector isa FermionSector && !_fermion_sector_known_conserved(model, sector) &&
        throw(ArgumentError("requested fermion-number sector is not conserved by this Hamiltonian"))

    A = _packed_fermion_operator(
        model,
        sector;
        max_dimension=solver.max_basis_dimension,
    )
    dim = size(A, 1)
    solver.nev <= dim || throw(ArgumentError("nev=$(solver.nev) exceeds sector dimension $dim"))

    ok, herr = _optimized_hermitian_check(
        A, dim; seed=solver.seed + 19, tol=max(10 * solver.tol, 1e-9),
    )
    ok || throw(ArgumentError(
        "optimized packed-fermion Lanczos requires Hermitian projected Hamiltonian; error=$herr"
    ))

    dec = _optimized_lanczos_decomposition(
        A, dim;
        krylov_dim=min(dim, max(solver.krylov_dim, solver.nev + 2)),
        tol=solver.tol,
        seed=solver.seed,
        reorthogonalize=solver.reorthogonalize,
    )
    energies, right, residuals = _optimized_ritz(dec, solver.nev; tol=solver.tol)

    metadata = Dict{Symbol,Any}(
        :solver => :optimized_lanczos,
        :backend => :packed_fermion,
        :dimension => dim,
        :sector => sector,
        :sector_observables => isnothing(sector) ? :all : :same_sector_only,
        :hermitian => true,
        :hermiticity_error => herr,
        :biorthogonal => false,
        :iterations => dec.iterations,
        :ritz_residuals => residuals,
        :converged => residuals .<= solver.tol,
        :matrix_free => true,
        :approximation => :none,
        :physical_representation_preserved => true,
    )
    return SolverResult(
        solver, model, energies, right, right, A.data.basis, metadata,
    )
end
