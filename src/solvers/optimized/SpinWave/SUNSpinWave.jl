# Exact local spin matrices, product-state SU(2S+1) tangent expansion, and shared bosonic BdG.

function _sun_resolved_reference(ref::MagneticReferenceState, spins::AbstractVector)
    length(spins) == size(ref.directions, 2) || throw(DimensionMismatch("magnetic direction count differs from Hamiltonian"))
    states = Vector{ComplexF64}[]
    for (i, S) in enumerate(spins)
        ops = _sun_spin_matrices(S)
        n = view(ref.directions, :, i)
        H = Hermitian(sum(n[α] .* ops[α] for α in 1:3))
        F = eigen(H)
        z = Vector{ComplexF64}(F.vectors[:, end])
        k = argmax(abs.(z))
        z ./= z[k] / abs(z[k])
        push!(states, z)
    end
    return SUNReferenceState(states)
end


function _sun_quadratic_data(data::_SpinWaveInteractions, ref::SUNReferenceState; tol::Real, stationarity_tol::Real)
    nsites = length(data.spins)
    length(ref.amplitudes) == nsites || throw(DimensionMismatch("SU(N) reference and Hamiltonian site counts disagree"))
    operators = [_sun_spin_matrices(S) for S in data.spins]
    flavors = [length(z) - 1 for z in ref.amplitudes]
    offsets = cumsum(vcat(0, flavors))
    M = offsets[end]
    effective = Matrix{ComplexF64}[]
    moments = zeros(Float64, 3, nsites)
    for i in 1:nsites
        U = ref.bases[i]
        z = ref.amplitudes[i]
        ops = operators[i]
        H = zeros(ComplexF64, length(z), length(z))
        for α in 1:3
            moments[α, i] = real(dot(z, ops[α] * z))
            H .-= data.fields[α, i] .* ops[α]
            for β in 1:3
                H .+= data.anisotropy[i][α, β] .* (ops[α] * ops[β])
            end
        end
        push!(effective, H)
    end
    energy = Float64(data.constant)
    for ((i, j), J) in data.exchange
        energy += dot(view(moments, :, i), J * view(moments, :, j))
        for α in 1:3
            effective[i] .+= (J * view(moments, :, j))[α] .* operators[i][α]
            effective[j] .+= (transpose(J) * view(moments, :, i))[α] .* operators[j][α]
        end
    end
    for i in 1:nsites
        z = ref.amplitudes[i]
        ops = operators[i]
        energy += real(-sum(data.fields[α, i] * dot(z, ops[α] * z) for α in 1:3))
        energy += real(sum(data.anisotropy[i][α, β] * dot(z, ops[α] * ops[β] * z) for α in 1:3, β in 1:3))
    end
    A0 = zeros(ComplexF64, M, M)
    residuals = zeros(Float64, nsites)
    local_amplitudes = Vector{Matrix{ComplexF64}}(undef, nsites)
    for i in 1:nsites
        U = ref.bases[i]
        H = effective[i]
        d = flavors[i]
        e = real(dot(view(U, :, 1), H * view(U, :, 1)))
        tangent = @view U[:, 2:d+1]
        residuals[i] = norm(adjoint(tangent) * H * view(U, :, 1))
        indices = offsets[i]+1:offsets[i+1]
        A0[indices, indices] .= adjoint(tangent) * H * tangent - e * I
        f = zeros(ComplexF64, 3, d)
        for α in 1:3, μ in 1:d
            f[α, μ] = dot(view(U, :, 1), operators[i][α] * view(tangent, :, μ))
        end
        local_amplitudes[i] = f
    end
    scale = max(norm(data.fields), maximum((norm(J) * maximum(data.spins) for J in values(data.exchange)); init=0.0),
                maximum((norm(K) for K in data.anisotropy); init=0.0), 1.0)
    maximum(residuals) <= stationarity_tol * scale || throw(ArgumentError(
        "SU(N) reference is not stationary: maximal local tangent residual $(maximum(residuals)); tolerance $(stationarity_tol * scale)"
    ))
    return (; A0, energy, residuals, offsets, flavors, operators, local_amplitudes, moments, reference=ref)
end

function _sun_interaction_blocks(data::_SpinWaveInteractions, setup; bonds=nothing, q=nothing)
    M = setup.offsets[end]
    A = copy(setup.A0)
    B = zeros(ComplexF64, M, M)
    interactions = isnothing(bonds) ? ((i=i, j=j, exchange=J, displacement=nothing) for ((i, j), J) in data.exchange) : bonds
    for bond in interactions
        i, j, J = bond.i, bond.j, bond.exchange
        i != j || throw(ArgumentError("SU(N) bond must connect distinct magnetic-cell indices"))
        phase = isnothing(q) ? (1.0 + 0im) : cis(dot(q, bond.displacement))
        fi = setup.local_amplitudes[i]
        fj = setup.local_amplitudes[j]
        for μ in 1:setup.flavors[i], ν in 1:setup.flavors[j]
            a, b = setup.offsets[i] + μ, setup.offsets[j] + ν
            hopping = sum(J[α, β] * conj(fi[α, μ]) * fj[β, ν] for α in 1:3, β in 1:3)
            pairing = sum(J[α, β] * conj(fi[α, μ]) * conj(fj[β, ν]) for α in 1:3, β in 1:3)
            A[a, b] += hopping * phase
            A[b, a] += conj(hopping * phase)
            B[a, b] += pairing * phase
            B[b, a] += pairing * conj(phase)
        end
    end
    return A, B
end

function _sun_solve(solver::SpinWaveSolver, model::ManyBodyModel)
    data = _spinwave_extract_interactions(model; tol=solver.tol)
    ref = _sun_resolved_reference(solver.reference, data.spins)
    setup = _sun_quadratic_data(data, ref; tol=solver.tol, stationarity_tol=solver.stationarity_tol)
    A, B = _sun_interaction_blocks(data, setup)
    norm(A - adjoint(A)) <= solver.tol * max(norm(A), 1.0) || throw(ArgumentError("SU(N) quadratic A is not Hermitian"))
    norm(B - transpose(B)) <= solver.tol * max(norm(B), 1.0) || throw(ArgumentError("SU(N) quadratic B is not symmetric"))
    frequencies, modes, diagnostics = _spinwave_bdg(A, B, solver)
    metadata = Dict{Symbol,Any}(
        :solver => :spin_wave,
        :backend => :site_sun_multiboson,
        :formalism => :SUN,
        :approximation => :linear_spin_wave,
        :approximation_certificate => ApproximationCertificate(:linear_spin_wave;
            assumptions=(:stationary_product_reference, :small_generalized_fluctuations),
            validity_conditions=(:quadratic_expansion_adequate,), order=2,
            notes=(:site_suN_multiboson, :exact_local_operator_matrices, :no_entangled_units)),
        :reference_states => [copy(z) for z in ref.amplitudes],
        :local_bases => [copy(U) for U in ref.bases],
        :local_spin_matrices => setup.operators,
        :local_flavors => setup.flavors,
        :flavor_offsets => setup.offsets,
        :reference_spin_expectations => setup.moments,
        :stationarity_residuals => setup.residuals,
        :reference_energy_convention => :exact_local_product_state_expectation,
        :onsite_anisotropy_present => any(K -> any(x -> !iszero(x), K), data.anisotropy),
        :interaction_scope => :bilinear_exchange_and_single_site_quadratic,
        :full_manybody_hilbert_space_constructed => false,
        :entangled_units => :not_supported,
        :multipolar_operator_matrix_elements => :supported,
    )
    merge!(metadata, diagnostics)
    return SpinWaveResult(solver, model, frequencies, modes, A, B, setup.energy, metadata)
end

"""Return reference, creation, annihilation and tangent-sector matrix elements of a local operator.

The operator must be expressed in the local spin basis m=S,S-1,...,-S; this interface also accepts spin quadrupoles and higher multipoles.
"""
function sun_operator_matrix_elements(result::SpinWaveResult, site::Integer, operator::AbstractMatrix)
    get(result.metadata, :formalism, nothing) === :SUN || throw(ArgumentError("local SU(N) matrix elements require formalism=:SUN"))
    bases = result.metadata[:local_bases]
    1 <= site <= length(bases) || throw(BoundsError(bases, site))
    U = bases[site]
    size(operator) == size(U) || throw(DimensionMismatch("local operator dimensions do not match the site Hilbert space"))
    O = adjoint(U) * operator * U
    return (; expectation=O[1, 1], annihilation=copy(O[1, 2:end]), creation=copy(O[2:end, 1]), tangent=copy(O[2:end, 2:end]))
end

"""One-mode amplitudes for one local operator per site in an SU(N) harmonic solution.

The returned vector contains coefficients of normal-mode annihilation operators. For physical spectra, operators should be Hermitian; Goldstone-regularized mode intensities are not certified.
"""
function sun_mode_matrix_elements(result::SpinWaveResult, operators::AbstractVector)
    get(result.metadata, :formalism, nothing) === :SUN || throw(ArgumentError("mode matrix elements require formalism=:SUN"))
    haskey(result.metadata, :zero_mode_regularization) && throw(ArgumentError("regulated SU(N) modes cannot certify spectral weights"))
    offsets = result.metadata[:flavor_offsets]
    length(operators) == length(offsets) - 1 || throw(DimensionMismatch("provide one local operator per site"))
    M = offsets[end]
    U = @view result.modes[1:M, :]
    V = size(result.modes, 1) == 2M ? @view(result.modes[M+1:2M, :]) : nothing
    amplitudes = zeros(ComplexF64, length(result.frequencies))
    for i in eachindex(operators)
        entries = sun_operator_matrix_elements(result, i, operators[i])
        for μ in eachindex(entries.annihilation), ν in eachindex(amplitudes)
            p = offsets[i] + μ
            amplitudes[ν] += entries.annihilation[μ] * U[p, ν]
            isnothing(V) || (amplitudes[ν] += entries.creation[μ] * V[p, ν])
        end
    end
    return amplitudes
end
