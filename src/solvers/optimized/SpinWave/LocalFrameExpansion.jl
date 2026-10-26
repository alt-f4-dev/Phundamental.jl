# Quadratic SU(2) Holstein-Primakoff expansion, including quantum-ordered onsite anisotropy.

"""Match the exact on-site spin matrix elements in the zero-, one-, and two-boson sectors.

For a real symmetric local tensor K, the coefficients reproduce the exact coherent-state expectation, one-magnon diagonal shift, and vacuum-to-two-magnon matrix element. This is a finite-spin kinematic correction to onsite LSWT, not an exact interacting spin-wave solution.
"""
function _spinwave_onsite_components(K::Matrix{Float64}, S::Float64, frame::Matrix{Float64})
    localK = transpose(frame) * K * frame
    traceK = tr(localK)
    longitudinal = localK[3, 3]
    energy = S * (S - 0.5) * longitudinal + (S / 2) * traceK
    normal = (2 * S - 1) * ((localK[1, 1] + localK[2, 2]) / 2 - longitudinal)
    pairfactor = sqrt(max(S * (2 * S - 1), 0.0) / 2)
    pairing = pairfactor * complex(localK[1, 1] - localK[2, 2], 2 * localK[1, 2])
    return energy, normal, pairing
end

function _spinwave_local_blocks(data::_SpinWaveInteractions, ref::MagneticReferenceState; tol::Real, stationarity_tol::Real)
    spins, exchange, fields = data.spins, data.exchange, data.fields
    N = length(spins)
    size(ref.directions) == (3, N) || throw(DimensionMismatch("reference must contain one direction for each spin site"))
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    gradient = -copy(fields)
    Eclass = data.constant
    frames = ref.frames

    for ((i, j), J) in exchange
        ni, nj = view(ref.directions, :, i), view(ref.directions, :, j)
        Si, Sj = spins[i], spins[j]
        longitudinal = dot(ni, J * nj)
        Eclass += Si * Sj * longitudinal
        A[i, i] -= Sj * longitudinal
        A[j, j] -= Si * longitudinal
        gradient[:, i] .+= Sj .* (J * nj)
        gradient[:, j] .+= Si .* (transpose(J) * ni)
        ui = complex.(frames[i][:, 1], -frames[i][:, 2])
        uj = complex.(frames[j][:, 1], -frames[j][:, 2])
        amplitude = sqrt(Si * Sj) / 2
        hopping = amplitude * dot(ui, J * uj)
        pairing = amplitude * dot(ui, J * conj.(uj))
        A[i, j] += hopping
        A[j, i] += conj(hopping)
        B[i, j] += pairing
        B[j, i] += pairing
    end
    for i in 1:N
        ni = view(ref.directions, :, i)
        Si = spins[i]
        Eclass -= Si * dot(view(fields, :, i), ni)
        A[i, i] += dot(view(fields, :, i), ni)
        K = data.anisotropy[i]
        onsite_energy, onsite_normal, onsite_pairing = _spinwave_onsite_components(K, Si, frames[i])
        Eclass += onsite_energy
        A[i, i] += onsite_normal
        B[i, i] += onsite_pairing
        gradient[:, i] .+= (2 * Si - 1) .* (K * ni)
    end
    residuals = [norm(cross(view(ref.directions, :, i), view(gradient, :, i))) for i in 1:N]
    scale = max(norm(fields), maximum((norm(J) * maximum(spins) for J in values(exchange)); init=0.0),
                maximum(((2 * spins[i] - 1) * norm(K) for (i, K) in enumerate(data.anisotropy)); init=0.0), 1.0)
    maximum(residuals) <= stationarity_tol * scale || throw(ArgumentError(
        "reference is not stationary: maximum transverse coherent-state gradient $(maximum(residuals)); tolerance $(stationarity_tol * scale)"
    ))
    norm(A - adjoint(A)) <= tol * max(norm(A), 1.0) || throw(ArgumentError("spin-wave A must be Hermitian"))
    norm(B - transpose(B)) <= tol * max(norm(B), 1.0) || throw(ArgumentError("spin-wave B must be symmetric"))
    return A, B, Eclass, residuals
end

function _spinwave_bdg(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, solver::SpinWaveSolver)
    N = size(A, 1)
    H = [A B; adjoint(B) transpose(A)]
    eigenvalues = eigvals(Hermitian(H))
    scale = max(opnorm(H), 1.0)
    minimum(eigenvalues) >= -solver.tol * scale || throw(ArgumentError("reference has a negative quadratic energy direction"))
    if minimum(eigenvalues) > solver.tol * scale || norm(B) <= solver.tol * max(norm(A), 1.0)
        return _bosonic_bdg_diagonalize(A, B; tol=solver.tol)
    end
    solver.zero_modes === :regularize || throw(ArgumentError(
        "quadratic Hamiltonian has Goldstone/zero modes; select zero_modes=:regularize for an explicitly regulated approximation"
    ))
    delta = solver.regularization * scale
    shifted = A + delta * I
    omega, modes, meta = _bosonic_bdg_diagonalize(Matrix{ComplexF64}(shifted), B; tol=solver.tol)
    meta[:zero_mode_regularization] = delta
    meta[:physical_bdg_minimum_eigenvalue] = minimum(eigenvalues)
    meta[:zero_mode_approximation] = :positive_definite_energy_shift
    meta[:physical_modes_at_zero_regulator] = :not_certified
    return omega, modes, meta
end

function solve(solver::SpinWaveSolver, model::ManyBodyModel)
    solver.formalism === :SUN && return _sun_solve(solver, model)
    data = _spinwave_extract_interactions(model; tol=solver.tol)
    A, B, Eclass, residuals = _spinwave_local_blocks(data, solver.reference; tol=solver.tol, stationarity_tol=solver.stationarity_tol)
    frequencies, modes, diagmeta = _spinwave_bdg(A, B, solver)
    has_onsite = any(K -> any(x -> !iszero(x), K), data.anisotropy)
    metadata = Dict{Symbol,Any}(
        :solver => :spin_wave,
        :backend => :noncollinear_su2_lswt,
        :approximation => :linear_spin_wave,
        :approximation_certificate => ApproximationCertificate(:linear_spin_wave;
            assumptions=(:stationary_ordered_reference, :small_transverse_fluctuations),
            validity_conditions=(:quadratic_expansion_adequate,), order=2,
            notes=(:general_tensor_bilinear_exchange, :local_frame_holstein_primakoff, :quantum_ordered_onsite_anisotropy)),
        :reference_directions => copy(solver.reference.directions),
        :reference_frames => [copy(frame) for frame in solver.reference.frames],
        :stationarity_residuals => residuals,
        :interaction_scope => has_onsite ? :bilinear_exchange_and_single_ion_quadratic : :bilinear_distinct_spin_sites,
        :onsite_anisotropy_present => has_onsite,
        :onsite_anisotropy_expansion => :exact_zero_one_two_local_boson_matrix_elements,
        :reference_energy_convention => :classical_intersite_exact_coherent_onsite,
        :expansion => :quadratic_holstein_primakoff,
        :physical_representation_preserved_in_source_model => true,
        :full_manybody_hilbert_space_constructed => false,
    )
    merge!(metadata, diagmeta)
    return SpinWaveResult(solver, model, frequencies, modes, A, B, Float64(Eclass), metadata)
end
