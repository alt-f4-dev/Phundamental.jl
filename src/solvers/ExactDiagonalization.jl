# Dense exact diagonalization over the finite computational basis.

struct ExactDiagonalization <: AbstractSolver
    hermitian::Symbol
    check_physical_closure::Bool
    closure_tol::Float64
    hbar::Float64
end

function ExactDiagonalization(; hermitian::Symbol=:auto, check_physical_closure::Bool=true, closure_tol::Real=1e-10, hbar::Real=1.0)
    hermitian in (:auto, :yes, :no) || throw(ArgumentError("hermitian must be :auto, :yes, or :no"))
    return ExactDiagonalization(hermitian, check_physical_closure, Float64(closure_tol), Float64(hbar))
end

solver_name(::ExactDiagonalization) = :exact_diagonalization

function _relative_hermiticity_error(H::AbstractMatrix)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("Hermiticity requires a square matrix"))
    numerator2 = 0.0
    denominator2 = 0.0
    @inbounds for j in axes(H, 2), i in axes(H, 1)
        hij = H[i, j]
        denominator2 += abs2(hij)
        numerator2 += abs2(hij - conj(H[j, i]))
    end
    return sqrt(numerator2) / max(sqrt(denominator2), eps(Float64))
end

function _sort_eigensystem(values, right, left)
    order = sortperm(eachindex(values), by=i -> (real(values[i]), imag(values[i])))
    return values[order], right[:, order], left[:, order]
end

function _exact_dense_problem(solver::ExactDiagonalization, model::ManyBodyModel)
    capabilities = numerical_capabilities(model)
    capabilities.numerically_realizable || throw(ArgumentError(
        "ExactDiagonalization cannot construct the numerical target representation: $(join(capabilities.reasons, "; "))"
    ))

    basis = computational_basis(model)
    realized = realize(model.hamiltonian, basis; hbar=solver.hbar, sparse=false, check_physical_closure=solver.check_physical_closure, closure_tol=solver.closure_tol)
    H = realized.matrix
    herr = _relative_hermiticity_error(H)
    threshold = max(solver.closure_tol, 1e-12)
    if capabilities.biorthogonal && solver.hermitian === :yes
        throw(ArgumentError("a biorthogonal representation cannot be forced through the Hermitian eigensolver; use hermitian=:auto or :no"))
    end
    use_hermitian = !capabilities.biorthogonal && (solver.hermitian === :yes || (solver.hermitian === :auto && herr <= threshold))
    if solver.hermitian === :yes && herr > threshold
        throw(ArgumentError("Hamiltonian was declared Hermitian but relative Hermiticity error is $herr"))
    end
    return H, basis, realized.diagnostics, herr, use_hermitian
end

"""
    energy_spectrum(solver::ExactDiagonalization, model)

Compute the complete exact energy spectrum without retaining eigenvectors or the computational basis in the returned result. For exact canonical thermodynamic curves this avoids the dominant eigenvector storage and copying required by `solve` while preserving the same complete spectrum.
"""
function energy_spectrum(solver::ExactDiagonalization, model::ManyBodyModel)
    H, basis, diagnostics, herr, use_hermitian = _exact_dense_problem(solver, model)
    dim = length(basis)

    energies = if use_hermitian
        values = collect(eigvals!(Hermitian(H)))
        issorted(values) || sort!(values)
        values
    else
        values = collect(eigvals!(H))
        sort!(values; by=x -> (real(x), imag(x)))
        values
    end

    metadata = Dict{Symbol,Any}(
        :solver => :exact_diagonalization,
        :dimension => dim,
        :hermitian => use_hermitian,
        :hermiticity_error => herr,
        :eigenvectors_retained => false,
        :matrix_diagnostics => diagnostics,
    )
    return EnergySpectrumResult(solver, model, energies, dim, metadata)
end

function solve(solver::ExactDiagonalization, model::ManyBodyModel)
    H, basis, diagnostics, herr, use_hermitian = _exact_dense_problem(solver, model)

    if use_hermitian
        F = eigen!(Hermitian(H))
        energies = collect(F.values)
        right = F.vectors
        if !issorted(energies)
            order = sortperm(energies)
            energies = energies[order]
            right = right[:, order]
        end
        left = right
        biorthogonal = false
        conditioning = 1.0
    else
        F = eigen!(H)
        energies = collect(F.values)
        right = F.vectors
        conditioning = cond(right)
        isfinite(conditioning) || throw(ArgumentError("non-Hermitian eigenvector matrix is singular/defective"))
        left = adjoint(inv(right))
        energies, right, left = _sort_eigensystem(energies, right, left)
        biorthogonal = true
    end

    metadata = Dict{Symbol,Any}(
        :solver => :exact_diagonalization,
        :dimension => length(basis),
        :hermitian => use_hermitian,
        :hermiticity_error => herr,
        :biorthogonal => biorthogonal,
        :eigenvector_condition => conditioning,
        :eigenvectors_retained => true,
        :matrix_diagnostics => diagnostics,
    )
    return SolverResult(solver, model, energies, right, left, basis, metadata)
end
