# Extraction and stable paraunitary diagonalization of quadratic bosons.

function _boson_quadratic_matrices(model::ManyBodyModel; tol::Real=1e-10)
    rep = model.representation
    rep.algebra isa BosonAlgebra ||
        throw(ArgumentError("QuadraticBosonSolver requires BosonAlgebra"))
    M = rep.algebra.nmodes
    A = zeros(ComplexF64, M, M)
    B = zeros(ComplexF64, M, M)
    ann = zeros(ComplexF64, M, M)
    constant = 0.0 + 0.0im

    terms = _expanded_operator_terms(model.hamiltonian)
    for term in terms
        ps = [p for p in term.primitives if p.code != _OP_IDENTITY]
        c0 = term.coefficient
        if isempty(ps)
            constant += c0
        elseif length(ps) == 1 && ps[1].code == _OP_BOSON_N
            A[ps[1].index, ps[1].index] += c0
        elseif length(ps) == 2
            p, q = ps
            if p.code == _OP_BOSON_C && q.code == _OP_BOSON_A
                A[p.index, q.index] += c0
            elseif p.code == _OP_BOSON_A && q.code == _OP_BOSON_C
                p.index == q.index && (constant += c0)
                A[q.index, p.index] += c0
            elseif p.code == _OP_BOSON_C && q.code == _OP_BOSON_C
                i, j = p.index, q.index
                if i == j
                    B[i, i] += 2c0
                else
                    B[i, j] += c0
                    B[j, i] += c0
                end
            elseif p.code == _OP_BOSON_A && q.code == _OP_BOSON_A
                i, j = p.index, q.index
                if i == j
                    ann[i, i] += 2c0
                else
                    ann[i, j] += c0
                    ann[j, i] += c0
                end
            else
                throw(ArgumentError("Hamiltonian is not quadratic in bosonic operators"))
            end
        else
            throw(ArgumentError("Hamiltonian contains interactions beyond quadratic boson form"))
        end
    end

    norm(A - adjoint(A)) <= tol * max(norm(A), 1.0) ||
        throw(ArgumentError("quadratic boson normal matrix A is not Hermitian"))
    norm(B - transpose(B)) <= tol * max(norm(B), 1.0) ||
        throw(ArgumentError("bosonic pairing matrix B is not symmetric"))

    pair_error = norm(ann - conj.(B)) / max(norm(B), norm(ann), 1.0)
    pair_error <= max(tol, 1e-12) || throw(ArgumentError(
        "bosonic pairing creation/annihilation terms are not Hermitian conjugates; error=$pair_error"
    ))
    return A, B, constant, pair_error
end

function _symplectic_orthonormalize_positive_modes!(
    modes::Matrix{ComplexF64},
    frequencies::Vector{Float64};
    tol::Real,
)
    M = length(frequencies)
    size(modes) == (2M, M) || throw(DimensionMismatch("positive bosonic BdG modes must have size (2M, M)"))

    scale = max(maximum(abs, frequencies), 1.0)
    degeneracy_atol = max(10 * float(tol) * scale, 100 * eps(Float64) * scale)

    norm_errors = zeros(Float64, M)
    degenerate_blocks = UnitRange{Int}[]
    max_block_gram_error = 0.0

    first_mode = 1

    while first_mode <= M
        last_mode = first_mode

        while last_mode < M && abs(frequencies[last_mode + 1] - frequencies[first_mode]) <= degeneracy_atol
            last_mode += 1
        end

        block = first_mode:last_mode
        block_size = length(block)

        if block_size == 1
            mode = @view modes[:, first_mode]

            metric_norm = 0.0
            @inbounds for i in 1:M
                metric_norm += abs2(mode[i]) - abs2(mode[M + i])
            end

            metric_norm > tol || throw(ArgumentError("positive bosonic mode has nonpositive symplectic norm"))

            mode ./= sqrt(metric_norm)

            normalized_norm = 0.0
            @inbounds for i in 1:M
                normalized_norm += abs2(mode[i]) - abs2(mode[M + i])
            end

            norm_errors[first_mode] = abs(normalized_norm - 1)
        else
            push!(degenerate_blocks, block)

            W = Matrix(@view modes[:, block])
            weighted = copy(W)

            @views weighted[M+1:2M, :] .*= -1

            gram = adjoint(W) * weighted
            gram = 0.5 .* (gram .+ adjoint(gram))

            metric = eigen(Hermitian(gram))
            metric_scale = max(maximum(abs, metric.values), 1.0)
            metric_floor = max(float(tol), 100 * eps(Float64)) * metric_scale

            minimum(metric.values) > metric_floor || throw(ArgumentError(
                "degenerate positive-frequency bosonic subspace has a nonpositive symplectic metric"
            ))

            inverse_sqrt_metric =
                metric.vectors *
                Diagonal(inv.(sqrt.(metric.values))) *
                adjoint(metric.vectors)

            W = W * inverse_sqrt_metric
            @views modes[:, block] .= W

            weighted .= W
            @views weighted[M+1:2M, :] .*= -1

            gram_check = adjoint(W) * weighted

            @inbounds for a in 1:block_size, b in 1:block_size
                target = a == b ? 1.0 : 0.0
                error = abs(gram_check[a, b] - target)
                max_block_gram_error = max(max_block_gram_error, error)
            end

            @inbounds for a in 1:block_size
                norm_errors[first_mode + a - 1] = abs(gram_check[a, a] - 1)
            end
        end

        first_mode = last_mode + 1
    end

    return norm_errors, max_block_gram_error, degenerate_blocks
end

function _bosonic_bdg_diagonalize(A::AbstractMatrix, B::AbstractMatrix; tol::Real=1e-10)
    M = size(A, 1)
    size(A) == (M, M) == size(B) || throw(DimensionMismatch("bosonic BdG blocks must be square and equal-sized"))

    if norm(B) <= tol * max(norm(A), 1.0)
        F = eigen(Hermitian(Matrix(A)))
        scale = max(maximum(abs.(F.values)), 1.0)

        minimum(F.values) >= -tol * scale || throw(ArgumentError(
            "number-conserving boson Hamiltonian is unstable: negative mode energy"
        ))

        return max.(Float64.(F.values), 0.0), Matrix{ComplexF64}(F.vectors), Dict{Symbol,Any}(
            :pairing => false,
            :symplectic => false,
            :symplectic_orthonormalized => false,
        )
    end

    H = [A B; adjoint(B) transpose(A)]
    sigma = Diagonal(vcat(ones(Float64, M), -ones(Float64, M)))
    D = sigma * H
    F = eigen(Matrix{ComplexF64}(D))

    scale = max(maximum(abs.(F.values)), 1.0)
    imagerr = maximum(abs.(imag.(F.values))) / scale

    imagerr <= max(tol, 1e-10) || throw(ArgumentError(
        "bosonic BdG dynamical matrix has complex frequencies; reference state is dynamically unstable"
    ))

    values = real.(F.values)
    positive = findall(value -> value > tol * scale, values)

    length(positive) == M || throw(ArgumentError(
        "stable bosonic BdG problem should contain exactly $M positive frequencies; found $(length(positive))"
    ))

    order = sortperm(values[positive])
    indices = positive[order]

    frequencies = Float64.(values[indices])
    modes = Matrix{ComplexF64}(F.vectors[:, indices])

    norm_errors, block_gram_error, degenerate_blocks =
        _symplectic_orthonormalize_positive_modes!(
            modes,
            frequencies;
            tol=tol,
        )

    return frequencies, modes, Dict{Symbol,Any}(
        :pairing => true,
        :symplectic => true,
        :symplectic_orthonormalized => true,
        :bdg_matrix => H,
        :dynamical_matrix => D,
        :frequency_imaginary_error => imagerr,
        :symplectic_norm_errors => norm_errors,
        :symplectic_block_gram_error => block_gram_error,
        :degenerate_mode_blocks => degenerate_blocks,
    )
end

function _project_bosonic_quadratic_blocks(A::AbstractMatrix, B::AbstractMatrix, projection::BosonSubspaceProjection)
    Q = projection.basis
    size(Q, 1) == size(A, 1) || throw(DimensionMismatch("boson projection basis row count must match the source-mode dimension"))
    Aprojected = adjoint(Q) * A * Q
    Bprojected = adjoint(Q) * B * conj.(Q)
    return Aprojected, Bprojected
end

function _embed_projected_boson_modes(modes::AbstractMatrix, projection::BosonSubspaceProjection)
    Q = projection.basis
    K = size(Q, 2)
    size(modes, 2) == K || throw(DimensionMismatch("projected mode count does not match the retained boson subspace"))

    if size(modes, 1) == K
        return Q * modes
    elseif size(modes, 1) == 2K
        upper = Q * view(modes, 1:K, :)
        lower = conj.(Q) * view(modes, K+1:2K, :)
        return vcat(upper, lower)
    end

    throw(DimensionMismatch("projected bosonic mode matrix must contain K or 2K rows"))
end

function solve(solver::QuadraticBosonSolver, model::ManyBodyModel)
    A, B, constant, pair_error = _boson_quadratic_matrices(model; tol=solver.tol)
    source_dimension = size(A, 1)

    if isnothing(solver.projection)
        omega, modes, meta = _bosonic_bdg_diagonalize(A, B; tol=solver.tol)
        metadata = Dict{Symbol,Any}(
            :solver => :quadratic_boson,
            :backend => :bosonic_bdg,
            :constant => constant,
            :dimension => source_dimension,
            :source_dimension => source_dimension,
            :mode_dimension => length(omega),
            :pair_conjugacy_error => pair_error,
            :approximation => :none_if_input_is_quadratic,
            :mode_space => get(meta, :symplectic, false) ? :nambu : :single_particle,
            :subspace_projected => false,
        )
        merge!(metadata, meta)
        return QuadraticModeResult(solver, model, omega, modes, metadata)
    end

    projection = solver.projection
    Aprojected, Bprojected = _project_bosonic_quadratic_blocks(A, B, projection)
    omega, projected_modes, meta = _bosonic_bdg_diagonalize(Aprojected, Bprojected; tol=solver.tol)
    modes = _embed_projected_boson_modes(projected_modes, projection)

    metadata = Dict{Symbol,Any}(
        :solver => :quadratic_boson,
        :backend => :bosonic_bdg,
        :constant => constant,
        :dimension => source_dimension,
        :source_dimension => source_dimension,
        :mode_dimension => length(omega),
        :pair_conjugacy_error => pair_error,
        :approximation => :quadratic_subspace_projection,
        :mode_space => get(meta, :symplectic, false) ? :nambu : :single_particle,
        :subspace_projected => true,
        :projection_label => projection.label,
        :projection_basis => projection.basis,
        :projected_dimension => size(projection.basis, 2),
        :excluded_dimension => source_dimension - size(projection.basis, 2),
    )
    merge!(metadata, meta)
    return QuadraticModeResult(solver, model, omega, modes, metadata)
end
