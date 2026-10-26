# Momentum-resolved BdG matrices require certified directed bond displacements.

"""Modes over momenta measured in inverse Cartesian length, in the magnetic supercell's Brillouin zone."""
struct SpinWaveDispersionResult{S,M,Q,E,V,D} <: AbstractModeResult
    solver::S
    model::M
    qpoints::Q
    frequencies::E
    modes::V
    metadata::D
end

function _spinwave_periodic_bonds(model::ManyBodyModel, data::_SpinWaveInteractions; tol::Real)
    bonds = get(model.parameters, :spinwave_bonds, nothing)
    isnothing(bonds) && throw(ArgumentError("momentum spin waves require certified :spinwave_bonds metadata with translated-image displacements"))
    accumulated = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for bond in bonds
        i, j = bond.i, bond.j
        i != j || throw(ArgumentError("same-index translated-image bonds require a distinct periodic expansion"))
        key = minmax(i, j)
        J = get!(accumulated, key) do
            zeros(Float64, 3, 3)
        end
        J .+= i == key[1] ? bond.exchange : transpose(bond.exchange)
    end
    allkeys = union(keys(accumulated), keys(data.exchange))
    for key in allkeys
        left = get(accumulated, key, zeros(3, 3))
        right = get(data.exchange, key, zeros(3, 3))
        norm(left - right) <= tol * max(norm(right), 1.0) || throw(ArgumentError("periodic bond tensors do not reconstruct the symbolic Hamiltonian"))
    end
    return bonds
end

function _spinwave_bloch_blocks(data::_SpinWaveInteractions, ref::MagneticReferenceState, bonds, q)
    N = length(data.spins)
    Aq = zeros(ComplexF64, N, N)
    Bq = zeros(ComplexF64, N, N)
    for bond in bonds
        i, j, J, δ = bond.i, bond.j, bond.exchange, bond.displacement
        length(q) == length(δ) || throw(DimensionMismatch("momentum and bond displacement dimensions disagree"))
        Si, Sj = data.spins[i], data.spins[j]
        ni, nj = view(ref.directions, :, i), view(ref.directions, :, j)
        longitudinal = dot(ni, J * nj)
        Aq[i, i] -= Sj * longitudinal
        Aq[j, j] -= Si * longitudinal
        ui = complex.(ref.frames[i][:, 1], -ref.frames[i][:, 2])
        uj = complex.(ref.frames[j][:, 1], -ref.frames[j][:, 2])
        factor = sqrt(Si * Sj) / 2
        hopping = factor * dot(ui, J * uj)
        pairing = factor * dot(ui, J * conj.(uj))
        phase = cis(dot(q, δ))
        Aq[i, j] += hopping * phase
        Aq[j, i] += conj(hopping * phase)
        Bq[i, j] += pairing * phase
        Bq[j, i] += pairing * conj(phase)
    end
    for i in 1:N
        Aq[i, i] += dot(view(data.fields, :, i), view(ref.directions, :, i))
        _, onsite_normal, onsite_pairing = _spinwave_onsite_components(data.anisotropy[i], data.spins[i], ref.frames[i])
        Aq[i, i] += onsite_normal
        Bq[i, i] += onsite_pairing
    end
    return Aq, Bq
end

function _spinwave_bloch_diagonalize(Aq, Aminus, Bq, solver::SpinWaveSolver)
    N = size(Aq, 1)
    H = [Aq Bq; adjoint(Bq) transpose(Aminus)]
    norm(H - adjoint(H)) <= solver.tol * max(norm(H), 1.0) || throw(ArgumentError("non-Hermitian Bloch BdG energy matrix"))
    eigenvalues = eigvals(Hermitian(H))
    scale = max(opnorm(H), 1.0)
    minimum(eigenvalues) >= -solver.tol * scale || throw(ArgumentError("unstable Bloch spin-wave reference"))
    meta = Dict{Symbol,Any}()
    if minimum(eigenvalues) <= solver.tol * scale
        solver.zero_modes === :regularize || throw(ArgumentError("Bloch BdG has a zero mode; use zero_modes=:regularize explicitly"))
        delta = solver.regularization * scale
        H += delta * I
        meta[:zero_mode_regularization] = delta
        meta[:zero_mode_approximation] = :positive_definite_energy_shift
    end
    sigma = Diagonal(vcat(ones(Float64, N), -ones(Float64, N)))
    F = eigen(Matrix{ComplexF64}(sigma * H))
    imagerr = maximum(abs, imag.(F.values)) / scale
    imagerr <= max(solver.tol, 1e-10) || throw(ArgumentError("complex Bloch magnon frequencies"))
    indices = findall(value -> real(value) > solver.tol * scale, F.values)
    length(indices) == N || throw(ArgumentError("Bloch BdG should have $N positive-norm positive-energy modes"))
    indices = indices[sortperm(real.(F.values[indices]))]
    frequencies = Float64.(real.(F.values[indices]))
    modes = Matrix{ComplexF64}(F.vectors[:, indices])
    norms, gram_error, blocks = _symplectic_orthonormalize_positive_modes!(modes, frequencies; tol=solver.tol)
    meta[:symplectic_norm_errors] = norms
    meta[:symplectic_block_gram_error] = gram_error
    meta[:degenerate_mode_blocks] = blocks
    meta[:frequency_imaginary_error] = imagerr
    return frequencies, modes, meta
end

"""Compute noncollinear spin-wave Bloch bands for tensor models retaining certified bond-image displacements."""
function spinwave_dispersion(solver::SpinWaveSolver, model::ManyBodyModel, qpoints)
    solver.formalism === :SUN && return _sun_dispersion(solver, model, qpoints)
    data = _spinwave_extract_interactions(model; tol=solver.tol)
    _, _, _, residuals = _spinwave_local_blocks(data, solver.reference; tol=solver.tol, stationarity_tol=solver.stationarity_tol)
    bonds = _spinwave_periodic_bonds(model, data; tol=solver.tol)
    qs = [Float64.(collect(q)) for q in qpoints]
    isempty(qs) && throw(ArgumentError("spin-wave qpoints cannot be empty"))
    N = length(data.spins)
    energies = Matrix{Float64}(undef, length(qs), N)
    modes = Vector{Matrix{ComplexF64}}(undef, length(qs))
    perq = Vector{Dict{Symbol,Any}}(undef, length(qs))
    for (iq, q) in enumerate(qs)
        Aq, Bq = _spinwave_bloch_blocks(data, solver.reference, bonds, q)
        Aminus, Bminus = _spinwave_bloch_blocks(data, solver.reference, bonds, -q)
        norm(Bq - transpose(Bminus)) <= solver.tol * max(norm(Bq), 1.0) || throw(ArgumentError("pairing does not obey B(q)=transpose(B(-q))"))
        ω, vectors, metadata = _spinwave_bloch_diagonalize(Aq, Aminus, Bq, solver)
        energies[iq, :] = ω
        modes[iq] = vectors
        perq[iq] = metadata
    end
    metadata = Dict{Symbol,Any}(
        :solver => :spin_wave,
        :approximation => :linear_spin_wave,
        :momentum_scope => :magnetic_supercell_bloch,
        :momentum_convention => :inverse_cartesian_length,
        :reference_directions => copy(solver.reference.directions),
        :stationarity_residuals => residuals,
        :per_momentum_diagnostics => perq,
        :onsite_anisotropy_present => any(K -> any(x -> !iszero(x), K), data.anisotropy),
        :onsite_anisotropy_expansion => :exact_zero_one_two_local_boson_matrix_elements,
        :scattering_tensors_from_bloch_modes => :elastic_basis_and_one_magnon,
    )
    return SpinWaveDispersionResult(solver, model, qs, energies, modes, metadata)
end

mode_energies(result::SpinWaveDispersionResult) = result.frequencies


function _sun_dispersion(solver::SpinWaveSolver, model::ManyBodyModel, qpoints)
    data = _spinwave_extract_interactions(model; tol=solver.tol)
    ref = _sun_resolved_reference(solver.reference, data.spins)
    setup = _sun_quadratic_data(data, ref; tol=solver.tol, stationarity_tol=solver.stationarity_tol)
    bonds = _spinwave_periodic_bonds(model, data; tol=solver.tol)
    qs = [Float64.(collect(q)) for q in qpoints]
    isempty(qs) && throw(ArgumentError("SU(N) spin-wave qpoints cannot be empty"))
    M = setup.offsets[end]
    energies = Matrix{Float64}(undef, length(qs), M)
    modes = Vector{Matrix{ComplexF64}}(undef, length(qs))
    perq = Vector{Dict{Symbol,Any}}(undef, length(qs))
    for (iq, q) in enumerate(qs)
        Aq, Bq = _sun_interaction_blocks(data, setup; bonds, q)
        Aminus, Bminus = _sun_interaction_blocks(data, setup; bonds, q=-q)
        norm(Bq - transpose(Bminus)) <= solver.tol * max(norm(Bq), 1.0) || throw(ArgumentError("SU(N) Bloch pairing violates B(q)=transpose(B(-q))"))
        omega, vectors, diagnostics = _spinwave_bloch_diagonalize(Aq, Aminus, Bq, solver)
        energies[iq, :] = omega
        modes[iq] = vectors
        perq[iq] = diagnostics
    end
    metadata = Dict{Symbol,Any}(
        :solver => :spin_wave,
        :formalism => :SUN,
        :backend => :site_sun_multiboson,
        :approximation => :linear_spin_wave,
        :momentum_scope => :magnetic_supercell_bloch,
        :momentum_convention => :inverse_cartesian_length,
        :local_flavors => setup.flavors,
        :stationarity_residuals => setup.residuals,
        :per_momentum_diagnostics => perq,
        :scattering_tensors_from_bloch_modes => :elastic_basis_and_one_magnon,
        :entangled_units => :not_supported,
    )
    return SpinWaveDispersionResult(solver, model, qs, energies, modes, metadata)
end
