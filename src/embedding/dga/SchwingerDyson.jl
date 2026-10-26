# Normal-state orbital-resolved Schwinger-Dyson contractions and a separately identified
# particle-hole screened-ladder approximation. A single ph ladder is not parquet DΓA.

"""Momentum- and orbital-resolved normal self-energy with dimensions `(orbital, orbital, k, fermionic_frequency)`."""
struct OrbitalMomentumSelfEnergy{A<:FermionicMatsubaraAxis,K}
    axis::A
    kpoints::K
    labels::Vector{Symbol}
    values::Array{ComplexF64,4}
    metadata::Dict{Symbol,Any}
end

function OrbitalMomentumSelfEnergy(axis::FermionicMatsubaraAxis, kpoints, labels, values; metadata=Dict{Symbol,Any}())
    ks = collect(kpoints)
    names = Symbol.(collect(labels))
    n = length(names)
    n > 0 && length(unique(names)) == n || throw(ArgumentError("orbital labels must be unique and nonempty"))
    size(values) == (n, n, length(ks), length(axis)) || throw(DimensionMismatch("self-energy dimensions must be (orbital, orbital, k, frequency)"))
    all(_isfinite_complex, values) || throw(ArgumentError("momentum self-energy must be finite"))
    return OrbitalMomentumSelfEnergy(axis, ks, names, ComplexF64.(values), Dict{Symbol,Any}(metadata))
end

"""Static Hartree-Fock self-energy `Σ[a,b] = ∑[c,d] U[a,c,b,d] ρ[d,c]` for an antisymmetrized interaction.

The density convention is `ρ[a,b] = ⟨d†[b] d[a]⟩`. The interaction uses
`H_int = (1/4)∑ U[a,b,c,d] d†[a]d†[b]d[d]d[c]`.
"""
function orbital_hartree_fock(interaction::SpinOrbitalInteraction, density::AbstractMatrix)
    n = length(interaction.labels)
    size(density) == (n, n) || throw(DimensionMismatch("density matrix and interaction dimensions differ"))
    all(_isfinite_complex, density) || throw(ArgumentError("density matrix must be finite"))
    isapprox(density, density'; atol=1e-9, rtol=1e-9) || throw(ArgumentError("density matrix must be Hermitian"))
    output = zeros(ComplexF64, n, n)
    U = interaction.tensor
    @inbounds for b in 1:n, a in 1:n, d in 1:n, c in 1:n
        output[a, b] += U[a, c, b, d] * density[d, c]
    end
    return output
end

"""Recover `F` from `χ = χ₀ + χ₀ F χ₀` with factorized linear solves, one bosonic block at a time.

This *particle-hole* vertex does not by itself certify the crossing-complete
vertex needed for a general diagrammatic self-energy.
"""
function orbital_full_vertex(bubble::OrbitalPairSusceptibility, response::OrbitalPairSusceptibility;
                             max_workspace_bytes::Integer=536870912, min_relative_pivot::Real=1e-14)
    _orbital_pair_compatible(bubble, response)
    bubble.bosonic_axis.indices == response.bosonic_axis.indices && bubble.bosonic_axis.beta == response.bosonic_axis.beta ||
        throw(ArgumentError("bosonic frequency axes differ"))
    np, _, nf, _, nb = size(bubble.values)
    d = big(np) * nf
    max_workspace_bytes > 0 && big(16) * (5 * d^2 + big(np)^2 * nf^2 * nb) <= max_workspace_bytes ||
        throw(ArgumentError("full-vertex inversion exceeds max_workspace_bytes"))
    isfinite(min_relative_pivot) && 0 <= min_relative_pivot < 1 || throw(ArgumentError("invalid relative pivot threshold"))
    D = Int(d)
    B = Matrix{ComplexF64}(undef, D, D)
    X = similar(B)
    F = similar(B)
    values = similar(bubble.values)
    diagnostics = NamedTuple[]
    for iΩ in 1:nb
        _orbital_bse_matrix!(B, bubble.values, iΩ)
        _orbital_bse_matrix!(X, response.values, iΩ)
        X .-= B
        factor = lu(B; check=true)
        pivots = abs.(diag(factor.U))
        ratio = minimum(pivots) / max(maximum(pivots), eps(Float64))
        ratio >= min_relative_pivot || throw(LinearAlgebra.SingularException(argmin(pivots)))
        ldiv!(factor, X)
        # Right-division is a factorized solve, not explicit inversion.
        F .= X / B
        _orbital_bse_unflatten!(values, F, iΩ)
        push!(diagnostics, (pivot_ratio=ratio, matrix_dimension=D))
    end
    metadata = Dict{Symbol,Any}(:construction => :particle_hole_full_vertex, :crossing_complete => false,
        :identity => :chi_equals_chi0_plus_chi0_F_chi0, :inverse_formed => false,
        :inversion_diagnostics => diagnostics)
    return OrbitalPairSusceptibility(bubble.fermionic_axis, bubble.bosonic_axis, bubble.pairs, bubble.labels, values;
        metadata=metadata, copy_values=false)
end

"""Callback supplying a four-index Schwinger-Dyson kernel, `fill!(dest, iq, iν, iΩ)`.

A callback must fill `dest[a,b,c,d]`, including its interaction, bubble,
and full-vertex contractions, so the correction is
`-β⁻¹ ∑[q,Ω,c,d] wq K[a,b,c,d;q,ν,Ω] G[c,d](k+q,ν+Ω)`.
The `:externally_certified_crossing` declaration is a provenance assertion by
its *caller*; it is not inferred or certified by this package.
"""
struct OrbitalSchwingerDysonKernel{F}
    fill!::F
    labels::Vector{Symbol}
    provenance::Symbol
end

function OrbitalSchwingerDysonKernel(fill!::F, labels; provenance::Symbol) where {F}
    names = Symbol.(collect(labels))
    !isempty(names) && length(unique(names)) == length(names) || throw(ArgumentError("orbital labels must be unique"))
    provenance in (:externally_certified_crossing, :screened_particle_hole) ||
        throw(ArgumentError("kernel provenance must describe its crossing-complete or screened-ladder approximation"))
    return OrbitalSchwingerDysonKernel{F}(fill!, names, provenance)
end

function _orbital_sde_inputs(dmft::MultiOrbitalDMFTResult, lattice::MatrixMomentumLattice,
                              green::MatrixMomentumResolvedGreenFunction, qmaps, baxis::BosonicMatsubaraAxis,
                              kernel::OrbitalSchwingerDysonKernel, qweights)
    dmft.converged || throw(ArgumentError("Schwinger-Dyson reconstruction requires a converged DMFT reference"))
    dmft.problem.lattice === lattice || throw(ArgumentError("Schwinger-Dyson lattice must be the lattice of the converged DMFT reference"))
    dmft.problem.labels == lattice.labels == green.labels == kernel.labels ||
        throw(ArgumentError("DMFT, lattice, Green and kernel orbital labels differ"))
    dmft.problem.lattice.kpoints == lattice.kpoints == green.kpoints ||
        throw(ArgumentError("lattice momentum ordering differs from the DMFT reference"))
    length(lattice.weights) == size(green.values, 3) || throw(DimensionMismatch("lattice and Green k dimensions differ"))
    isapprox(lattice.weights, green.weights; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("lattice and Green momentum quadrature weights differ"))
    # Expanded grids are permitted; missing external frequencies are checked below.
    all(in(green.axis.indices), dmft.problem.axis.indices) || throw(ArgumentError("single-particle grid lacks DMFT frequencies"))
    maps, shifts = _orbital_ladder_maps(green, qmaps, dmft.problem.axis, baxis)
    n = length(lattice.labels)
    size(green.values, 1) == n || throw(DimensionMismatch("orbital Green dimension differs from lattice"))
    w = isnothing(qweights) ? fill(inv(length(maps)), length(maps)) : Float64.(collect(qweights))
    length(w) == length(maps) || throw(DimensionMismatch("q weights and momentum maps differ"))
    all(x -> isfinite(x) && x >= 0, w) && isapprox(sum(w), 1.0; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("q weights must be finite, nonnegative, and sum to one"))
    return maps, shifts, w
end

"""Contract an explicitly supplied four-index kernel into a matrix self-energy.

The DMFT self-energy is included exactly once. The weighted local momentum
average of the computed correction is subtracted, and the un-subtracted mean
is recorded. This defines a one-shot *local-projection* convention, not an
identity with a general parquet double-counting subtraction.
"""
function orbital_schwinger_dyson(dmft::MultiOrbitalDMFTResult, lattice::MatrixMomentumLattice,
                                 green::MatrixMomentumResolvedGreenFunction, qmaps,
                                 baxis::BosonicMatsubaraAxis, kernel::OrbitalSchwingerDysonKernel;
                                 qweights=nothing, subtract_local::Bool=true,
                                 max_workspace_bytes::Integer=536870912, max_output_bytes::Integer=536870912)
    maps, shifts, qw = _orbital_sde_inputs(dmft, lattice, green, qmaps, baxis, kernel, qweights)
    n, nk, nf, nb = length(lattice.labels), length(lattice.kpoints), length(dmft.problem.axis), length(baxis)
    output_bytes = big(16) * big(n)^2 * nk * nf
    workspace_bytes = big(16) * (big(n)^4 + 2 * big(n)^2 + big(n)^2 * nf + big(n)^2 * nk * nf)
    max_output_bytes > 0 && output_bytes <= max_output_bytes || throw(ArgumentError("self-energy output exceeds max_output_bytes"))
    max_workspace_bytes > 0 && workspace_bytes <= max_workspace_bytes || throw(ArgumentError("Schwinger-Dyson workspace exceeds max_workspace_bytes"))
    correction = zeros(ComplexF64, n, n, nk, nf)
    buffer = Array{ComplexF64}(undef, n, n, n, n)
    kernel_matrix = reshape(buffer, n^2, n^2)
    green_vector = Vector{ComplexF64}(undef, n^2)
    contracted_vector = similar(green_vector)
    beta = dmft.problem.axis.beta
    @inbounds for iq in eachindex(maps), iΩ in 1:nb, iν in 1:nf
        fill!(buffer, 0)
        kernel.fill!(buffer, iq, iν, iΩ)
        all(_isfinite_complex, buffer) || throw(ArgumentError("Schwinger-Dyson kernel returned nonfinite entries"))
        shifted = shifts[iΩ][iν][2]
        scale = -qw[iq] / beta
        for ik in 1:nk
            for d in 1:n, c in 1:n
                green_vector[(d-1)*n+c] = green.values[c, d, maps[iq][ik], shifted]
            end
            mul!(contracted_vector, kernel_matrix, green_vector)
            for b in 1:n, a in 1:n
                correction[a, b, ik, iν] += scale * contracted_vector[(b-1)*n+a]
            end
        end
    end
    local_part = zeros(ComplexF64, n, n, nf)
    @inbounds for iν in 1:nf, ik in 1:nk
        local_part[:, :, iν] .+= lattice.weights[ik] .* view(correction, :, :, ik, iν)
    end
    values = similar(correction)
    @inbounds for iν in 1:nf, ik in 1:nk, b in 1:n, a in 1:n
        values[a, b, ik, iν] = dmft.self_energy.values[a, b, iν] + correction[a, b, ik, iν] -
            (subtract_local ? local_part[a, b, iν] : 0)
    end
    metadata = Dict{Symbol,Any}(:construction => :one_shot_orbital_schwinger_dyson,
        :kernel_provenance => kernel.provenance, :crossing_complete_certified_by_package => false,
        :local_projection_subtracted => subtract_local, :raw_local_correction_norm => norm(local_part),
        :local_double_counting_scheme => subtract_local ? :weighted_local_projection : :none,
        :estimated_output_bytes => Int(output_bytes), :estimated_workspace_bytes => Int(workspace_bytes),
        :frequency_box_complete => true, :vertex_channels_inferred => false, :self_energy_feedback => false)
    return OrbitalMomentumSelfEnergy(dmft.problem.axis, lattice.kpoints, lattice.labels, values; metadata=metadata)
end

"""Evaluate the matrix Dyson equation with a momentum-dependent self-energy."""
function dyson_momentum_green(lattice::MatrixMomentumLattice, sigma::OrbitalMomentumSelfEnergy; chemical_potential::Real=0.0)
    lattice.labels == sigma.labels || throw(ArgumentError("orbital labels differ"))
    lattice.kpoints == sigma.kpoints || throw(ArgumentError("momentum labels differ"))
    n, nk, nf = length(sigma.labels), length(sigma.kpoints), length(sigma.axis)
    values = Array{ComplexF64}(undef, n, n, nk, nf)
    kernel = Matrix{ComplexF64}(undef, n, n)
    for iν in 1:nf, ik in 1:nk
        kernel .= -view(lattice.hamiltonians, :, :, ik) .- view(sigma.values, :, :, ik, iν)
        for a in 1:n
            kernel[a, a] += sigma.axis[iν] + chemical_potential
        end
        values[:, :, ik, iν] .= _solve_inverse_matrix(kernel)
    end
    return MatrixMomentumResolvedGreenFunction(sigma.axis, lattice.kpoints, values; labels=lattice.labels,
        weights=lattice.weights, metadata=Dict{Symbol,Any}(:construction => :orbital_momentum_dyson,
            :chemical_potential => chemical_potential, :bloch_gauge => lattice.gauge))
end

"""Check matrix Dyson closure and the optional zero-local-correction constraint."""
function validate_orbital_schwinger_dyson(dmft::MultiOrbitalDMFTResult, result::OrbitalMomentumSelfEnergy;
                                           atol::Real=1e-10, rtol::Real=1e-8)
    isfinite(atol) && atol >= 0 && isfinite(rtol) && rtol >= 0 || throw(ArgumentError("validation tolerances must be finite and nonnegative"))
    dmft.problem.labels == result.labels || throw(ArgumentError("result and DMFT orbital labels differ"))
    result.axis.indices == dmft.problem.axis.indices || throw(ArgumentError("result and DMFT frequency grids differ"))
    n, _, nk, nf = size(result.values)
    local_delta = zeros(ComplexF64, n, n, nf)
    nk == length(dmft.problem.lattice.weights) || throw(DimensionMismatch("momentum dimensions differ"))
    for ik in 1:nk
        local_delta .+= dmft.problem.lattice.weights[ik] .* (view(result.values, :, :, ik, :) .- dmft.self_energy.values)
    end
    residual = norm(local_delta) / max(norm(dmft.self_energy.values), norm(result.values), 1.0)
    expected = get(result.metadata, :local_projection_subtracted, false)
    finite = all(_isfinite_complex, result.values)
    return (valid=finite && (!expected || residual <= atol + rtol), local_residual=residual,
        crossing_complete_certified_by_package=false, finite=finite)
end

"""Convert an orbital-pair ladder to a *screened particle-hole* kernel.

The kernel uses `W(q,Ω) = U_ph [χ(q,Ω) - ⟨χ(q,Ω)⟩q] U_ph / β²`,
where both fermionic frequency indices of `χ` are summed. It represents a
separately declared fluctuation-exchange-like approximation, **not** the
crossing-completed Schwinger-Dyson kernel of full multiorbital DΓA.
"""
function screened_orbital_ladder_kernel(ladder::OrbitalLadderResult, interaction::SpinOrbitalInteraction;
                                        qweights=nothing, max_kernel_bytes::Integer=536870912)
    responses = ladder.susceptibilities
    isnothing(responses) && throw(ArgumentError("the screened kernel requires return_full=true ladder susceptibilities"))
    isempty(responses) && throw(ArgumentError("the ladder has no momentum points"))
    first_response = first(responses)
    first_response.labels == interaction.labels || throw(ArgumentError("ladder and interaction labels differ"))
    n = length(interaction.labels)
    pairs = [(a, b) for a in 1:n for b in 1:n]
    first_response.pairs == pairs || throw(ArgumentError("screened contraction requires the complete canonical orbital-pair space"))
    nq, nb, np = length(responses), length(first_response.bosonic_axis), n^2
    bytes = big(16) * np^2 * (nq * nb + nb + 2)
    max_kernel_bytes > 0 && bytes <= max_kernel_bytes || throw(ArgumentError("screened ladder kernel exceeds max_kernel_bytes"))
    qw = isnothing(qweights) ? fill(inv(nq), nq) : Float64.(collect(qweights))
    length(qw) == nq && all(w -> isfinite(w) && w >= 0, qw) && isapprox(sum(qw), 1.0; atol=1e-12) ||
        throw(ArgumentError("q weights must be normalized and nonnegative"))
    beta = first_response.fermionic_axis.beta
    contractions = zeros(ComplexF64, np, np, nq, nb)
    for iq in 1:nq
        current = responses[iq]
        _orbital_pair_compatible(first_response, current)
        for iΩ in 1:nb, j in 1:np, i in 1:np
            contractions[i, j, iq, iΩ] = sum(@view current.values[i, j, :, :, iΩ]) / beta^2
        end
    end
    return _screened_kernel_from_contracts(contractions, interaction, qw; max_kernel_bytes=max_kernel_bytes)
end

function _screened_kernel_from_contracts(contractions::Array{ComplexF64,4}, interaction::SpinOrbitalInteraction,
                                         qw::Vector{Float64}; max_kernel_bytes::Integer=536870912)
    n = length(interaction.labels)
    np = n^2
    size(contractions, 1) == np && size(contractions, 2) == np ||
        throw(DimensionMismatch("screened contractions require a complete orbital-pair matrix"))
    _, _, nq, nb = size(contractions)
    length(qw) == nq || throw(DimensionMismatch("screened kernel q weights differ from contractions"))
    storage_bytes = big(16) * np^2 * (nq * nb + nb + 2)
    max_kernel_bytes > 0 && storage_bytes <= max_kernel_bytes || throw(ArgumentError("screened contraction exceeds max_kernel_bytes"))
    local_response = zeros(ComplexF64, np, np, nb)
    for iΩ in 1:nb, iq in 1:nq
        local_response[:, :, iΩ] .+= qw[iq] .* view(contractions, :, :, iq, iΩ)
    end
    pairs = [(a, b) for a in 1:n for b in 1:n]
    interaction_matrix = Matrix{ComplexF64}(undef, np, np)
    for (j, (c, d)) in enumerate(pairs), (i, (a, b)) in enumerate(pairs)
        interaction_matrix[i, j] = interaction.tensor[a, c, b, d]
    end
    storage = Array{ComplexF64}(undef, np, np, nq, nb)
    for iΩ in 1:nb, iq in 1:nq
        storage[:, :, iq, iΩ] .= interaction_matrix *
            (view(contractions, :, :, iq, iΩ) .- view(local_response, :, :, iΩ)) * interaction_matrix
    end
    function fill_screened!(dest::Array{ComplexF64,4}, iq::Int, iν::Int, iΩ::Int)
        @inbounds for b in 1:n, a in 1:n, d in 1:n, c in 1:n
            dest[a, b, c, d] = storage[(a-1)*n+c, (b-1)*n+d, iq, iΩ]
        end
        return nothing
    end
    return OrbitalSchwingerDysonKernel(fill_screened!, interaction.labels; provenance=:screened_particle_hole)
end

"""One-shot screened-ladder electronic problem with an explicit expanded-grid Green function.

The caller must supply a `MatrixMomentumResolvedGreenFunction` evaluated from the
DMFT self-energy on every required shifted Matsubara frequency. No extrapolation
of the self-energy is performed here.
"""
struct OrbitalScreenedLadderProblem{D,G,V,B,M} <: AbstractEmbeddingProblem
    dmft::D
    green::G
    vertex::V
    bosonic_axis::B
    qmaps::M
    qweights::Vector{Float64}
end

function OrbitalScreenedLadderProblem(dmft::MultiOrbitalDMFTResult, green::MatrixMomentumResolvedGreenFunction,
                                       vertex::OrbitalPairIrreducibleVertex, baxis::BosonicMatsubaraAxis;
                                       qmaps=dmft.problem.lattice.qmaps, qweights=nothing)
    isnothing(qmaps) && throw(ArgumentError("explicit momentum-shift maps are required"))
    dmft.problem.labels == green.labels == vertex.labels || throw(ArgumentError("orbital labels do not match"))
    vertex.fermionic_axis.indices == dmft.problem.axis.indices || throw(ArgumentError("vertex must use the DMFT frequency box"))
    vertex.bosonic_axis.indices == baxis.indices && vertex.bosonic_axis.beta == baxis.beta ||
        throw(ArgumentError("bosonic axes differ"))
    dmft.problem.lattice.kpoints == green.kpoints || throw(ArgumentError("DMFT and Green momentum ordering differs"))
    isapprox(dmft.problem.lattice.weights, green.weights; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("DMFT and Green momentum quadrature weights differ"))
    canonical = [(a, b) for a in 1:length(vertex.labels) for b in 1:length(vertex.labels)]
    vertex.pairs == canonical || throw(ArgumentError("screened ladder requires the full canonical orbital-pair ordering"))
    maps, _ = _orbital_ladder_maps(green, qmaps, dmft.problem.axis, baxis)
    w = isnothing(qweights) ? fill(inv(length(maps)), length(maps)) : Float64.(collect(qweights))
    length(w) == length(maps) && all(x -> isfinite(x) && x >= 0, w) && isapprox(sum(w), 1.0; atol=1e-12) ||
        throw(ArgumentError("q weights must be nonnegative and sum to one"))
    return OrbitalScreenedLadderProblem(dmft, green, vertex, baxis, maps, w)
end

"""Opt-in one-shot screened particle-hole ladder solver, distinct from a crossing-complete DΓA solver."""
struct OrbitalScreenedLadderSolver <: AbstractSolver
    max_workspace_bytes::Int
    max_output_bytes::Int
    min_relative_pivot::Float64
end

function OrbitalScreenedLadderSolver(; max_workspace_bytes::Integer=536870912,
                                      max_output_bytes::Integer=536870912, min_relative_pivot::Real=1e-14)
    all(x -> 0 < x <= typemax(Int), (max_workspace_bytes, max_output_bytes)) || throw(ArgumentError("memory limits must be positive"))
    isfinite(min_relative_pivot) && 0 <= min_relative_pivot < 1 || throw(ArgumentError("invalid pivot threshold"))
    return OrbitalScreenedLadderSolver(Int(max_workspace_bytes), Int(max_output_bytes), Float64(min_relative_pivot))
end

solver_name(::OrbitalScreenedLadderSolver) = :orbital_screened_ladder

struct OrbitalScreenedLadderResult{P,L,S,G}
    problem::P
    ladder::L
    nonlocal_self_energy::S
    corrected_green::G
    self_energy::SelfEnergy
    metadata::Dict{Symbol,Any}
end

function solve(solver::OrbitalScreenedLadderSolver, problem::OrbitalScreenedLadderProblem)
    dmft = problem.dmft
    dmft.converged || throw(ArgumentError("a converged DMFT reference is required"))
    lattice = dmft.problem.lattice
    # Stream individual q blocks; only their frequency-contracted pair matrices survive.
    vertex = problem.vertex
    selected = _orbital_bse_selection(vertex.pairs, length(vertex.labels))
    maps, shifts = _orbital_ladder_maps(problem.green, problem.qmaps, vertex.fermionic_axis, vertex.bosonic_axis)
    np, nf, nb, nq = length(selected), length(vertex.fermionic_axis), length(vertex.bosonic_axis), length(maps)
    contraction_bytes = big(16) * np^2 * nb * nq
    contraction_bytes <= solver.max_output_bytes || throw(ArgumentError("frequency-contracted ladder outputs exceed max_output_bytes"))
    contractions = zeros(ComplexF64, np, np, nq, nb)
    diagnostics = Vector{Vector{NamedTuple}}(undef, nq)
    beta = vertex.fermionic_axis.beta
    for iq in 1:nq
        bubble = _orbital_ladder_bubble_at(problem.green, maps[iq], vertex.fermionic_axis, vertex.bosonic_axis, selected, shifts)
        response, information = _orbital_bse_solve(bubble, vertex;
            max_workspace_bytes=solver.max_workspace_bytes, max_output_bytes=solver.max_output_bytes,
            min_relative_pivot=solver.min_relative_pivot)
        diagnostics[iq] = information
        for iΩ in 1:nb, j in 1:np, i in 1:np
            contractions[i, j, iq, iΩ] = sum(@view response.values[i, j, :, :, iΩ]) / beta^2
        end
    end
    metadata_ladder = Dict{Symbol,Any}(:construction => :streamed_frequency_contracted_ladder,
        :full_susceptibilities_retained => false, :frequency_box_complete => true,
        :contracted_pair_matrix_bytes => Int(contraction_bytes))
    ladder = OrbitalLadderResult(Any[collect(1:nq)...], nothing, nothing, diagnostics, metadata_ladder)
    kernel = _screened_kernel_from_contracts(contractions, dmft.problem.interaction, problem.qweights;
        max_kernel_bytes=solver.max_workspace_bytes)
    sigma = orbital_schwinger_dyson(dmft, lattice, problem.green, problem.qmaps, problem.bosonic_axis, kernel;
        qweights=problem.qweights, max_workspace_bytes=solver.max_workspace_bytes, max_output_bytes=solver.max_output_bytes)
    corrected = dyson_momentum_green(lattice, sigma; chemical_potential=dmft.problem.chemical_potential)
    metadata = Dict{Symbol,Any}(:method => :one_shot_screened_particle_hole_ladder,
        :full_multiorbital_dga => false, :crossing_channels_complete => false,
        :local_reference_retained => true, :frequency_box_complete => true, :self_energy_feedback => false,
        :momentum_resolved => true, :dmft_converged => dmft.converged)
    return OrbitalScreenedLadderResult(problem, ladder, sigma, corrected, dmft.self_energy, metadata)
end

"""Evaluate the converged finite-bath DMFT self-energy on an expanded exact Matsubara grid.

This recomputes the impurity's Lehmann propagator and finite-bath resolvent;
no constant, zero, or interpolated self-energy is substituted at missing indices.
"""
function expanded_dmft_momentum_green(dmft::MultiOrbitalDMFTResult, baxis::BosonicMatsubaraAxis)
    dmft.converged || throw(ArgumentError("expanded DMFT momentum Green requires a converged reference"))
    base = dmft.problem.axis
    isapprox(base.beta, baxis.beta; rtol=1e-12, atol=1e-12 * max(base.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic inverse temperatures differ"))
    indices = sort!(unique(vcat(base.indices, [n+m for n in base.indices for m in baxis.indices])))
    axis = FermionicMatsubaraAxis(base.beta, indices)
    impurity = dmft.impurity_result
    problem = MultiOrbitalImpurityProblem(dmft.problem.one_body, dmft.problem.interaction, axis;
        chemical_potential=dmft.problem.chemical_potential, bath=impurity.bath)
    g0 = noninteracting_impurity_green(problem)
    g = multiorbital_impurity_green(impurity.thermal_state, axis)
    sigma = self_energy(g0, g)
    resolved = momentum_green(dmft.problem.lattice, axis, sigma; chemical_potential=dmft.problem.chemical_potential)
    resolved.metadata[:expanded_finite_bath_lehmann] = true
    resolved.metadata[:dmft_reference] = :converged_finite_bath
    return resolved
end

function OrbitalScreenedLadderProblem(dmft::MultiOrbitalDMFTResult, vertex::OrbitalPairIrreducibleVertex,
                                       baxis::BosonicMatsubaraAxis; kwargs...)
    return OrbitalScreenedLadderProblem(dmft, expanded_dmft_momentum_green(dmft, baxis), vertex, baxis; kwargs...)
end
