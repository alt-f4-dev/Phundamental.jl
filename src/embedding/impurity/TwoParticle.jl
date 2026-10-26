# Orbital-pair impurity four-point functions for finite-temperature multiorbital ED.
# This file does not modify the established scalar impurity or ladder-DGA paths.

import ..Greens: impurity_two_particle, _connected_tau_cube_factored, _connected_tau_cube_direct, _fourier_project_particle_hole

"""Frequency-resolved particle-hole susceptibility with selected pairs `(a,b)` and `(c,d)`.

`values[i,j,n,np,m]` represents the connected Fourier transform of
`<T d†_a(τ₁)d_b(τ₂)d†_c(τ₃)d_d(0)>` after subtraction of its direct disconnected
contraction. The Fourier phase is `exp(-iντ₁+i(ν+Ω)τ₂-i(ν′+Ω)τ₃)`.
No spin-conservation or orbital-diagonality assumption is made.
"""
struct OrbitalPairSusceptibility{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    fermionic_axis::F
    bosonic_axis::B
    pairs::Vector{Tuple{Int,Int}}
    labels::Vector{Symbol}
    values::Array{ComplexF64,5}
    metadata::Dict{Symbol,Any}
end

function OrbitalPairSusceptibility(faxis::FermionicMatsubaraAxis, baxis::BosonicMatsubaraAxis,
                                   pairs, labels, values; metadata=Dict{Symbol,Any}(), copy_values::Bool=true)
    selected = Tuple{Int,Int}.(collect(pairs))
    names = Symbol.(collect(labels))
    dimension = length(names)
    !isempty(selected) && length(unique(selected)) == length(selected) &&
        all(pair -> 1 <= pair[1] <= dimension && 1 <= pair[2] <= dimension, selected) ||
        throw(ArgumentError("orbital pairs must be distinct, nonempty, and within the impurity dimension"))
    length(unique(names)) == dimension || throw(ArgumentError("spin-orbital labels must be unique"))
    isapprox(faxis.beta, baxis.beta; atol=16eps(Float64) * max(faxis.beta, 1.0), rtol=0.0) ||
        throw(ArgumentError("fermionic and bosonic axes must have the same inverse temperature"))
    expected = (length(selected), length(selected), length(faxis), length(faxis), length(baxis))
    size(values) == expected || throw(DimensionMismatch("orbital-pair values must have size $expected"))
    all(_isfinite_complex, values) || throw(ArgumentError("orbital-pair values must be finite"))
    cube = !copy_values && values isa Array{ComplexF64,5} ? values : ComplexF64.(values)
    return OrbitalPairSusceptibility(faxis, baxis, selected, names, cube, Dict{Symbol,Any}(metadata))
end

"""Connected impurity four-point result and the independently evaluated exact finite-bath Wick bubble."""
struct MultiOrbitalImpurityTwoParticleResult{S,B}
    susceptibility::S
    bubble::B
    diagnostics::Dict{Symbol,Any}
end

"""Local orbital-pair irreducible vertex, using the same inverse-Bethe–Salpeter sign convention as scalar DΓA."""
struct OrbitalPairIrreducibleVertex{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    fermionic_axis::F
    bosonic_axis::B
    pairs::Vector{Tuple{Int,Int}}
    labels::Vector{Symbol}
    values::Array{ComplexF64,5}
    metadata::Dict{Symbol,Any}
end

function _orbital_pair_selection(n::Int, pairs)
    chosen = isnothing(pairs) ? [(a, b) for a in 1:n for b in 1:n] : Tuple{Int,Int}.(collect(pairs))
    !isempty(chosen) && length(unique(chosen)) == length(chosen) &&
        all(p -> 1 <= p[1] <= n && 1 <= p[2] <= n, chosen) ||
        throw(ArgumentError("pairs must be unique (creation, annihilation) spin-orbital index tuples"))
    return chosen
end

function _orbital_pair_operators(state::MultiOrbitalEDThermalState, modes::Vector{Int})
    # Store eigenbasis annihilation operators once, rather than rebuilding them for each four-point contraction.
    sectors = state.sectors
    ranges = Vector{UnitRange{Int}}(undef, length(sectors))
    next = 1
    for index in eachindex(sectors)
        last = next + length(sectors[index].energies) - 1
        ranges[index] = next:last
        next = last + 1
    end
    dimension = next - 1
    operators = Dict{Int,Matrix{ComplexF64}}()
    for mode in modes
        C = zeros(ComplexF64, dimension, dimension)
        for source_index in eachindex(sectors)
            target_index = _multi_ed_target(state, source_index)
            target_index == 0 && continue
            C[ranges[target_index], ranges[source_index]] .=
                _multi_ed_transition(sectors[source_index], sectors[target_index], mode)
        end
        operators[mode] = C
    end
    return operators, reduce(vcat, (sector.energies for sector in sectors))
end

function _orbital_pair_bubble(state::MultiOrbitalEDThermalState, faxis::FermionicMatsubaraAxis,
                               baxis::BosonicMatsubaraAxis, pairs, labels)
    # Include shifted frequencies in a separate axis; do not silently zero an incomplete Matsubara box.
    all_indices = sort!(unique(vcat(faxis.indices, [n + m for n in faxis.indices for m in baxis.indices])))
    expanded = FermionicMatsubaraAxis(faxis.beta, all_indices)
    green = multiorbital_impurity_green(state, expanded).values
    lookup = Dict(index => i for (i, index) in enumerate(all_indices))
    np, nf, nb = length(pairs), length(faxis), length(baxis)
    output = zeros(ComplexF64, np, np, nf, nf, nb)
    for (iΩ, m) in enumerate(baxis.indices), (iν, n) in enumerate(faxis.indices)
        present = lookup[n]
        shifted = lookup[n + m]
        for (j, (c, d)) in enumerate(pairs), (i, (a, b)) in enumerate(pairs)
            output[i, j, iν, iν, iΩ] = -faxis.beta * green[d, a, present] * green[b, c, shifted]
        end
    end
    return OrbitalPairSusceptibility(faxis, baxis, pairs, labels, output;
        metadata=Dict{Symbol,Any}(:construction => :matrix_wick_bubble, :diagonal_in_fermionic_indices => true,
                                   :frequency_box_complete => true, :fourier_convention => :particle_hole_physical), copy_values=false)
end

"""
    impurity_two_particle(state::MultiOrbitalEDThermalState, fermionic_axis, bosonic_axis;
                          backend=TauGridFourPoint(), pairs=nothing, max_output_bytes=..., max_operator_bytes=...)

Compute selected orbital-pair connected four-point functions on midpoint imaginary-time grids.
The `:factored` backend reuses the validated scalar four-point trace kernel and caches all
fermionic eigenbasis transitions; `:direct` supplies an independent small-system reference.
Only one imaginary-time cube is held at a time. `pairs` defaults to all ordered pairs of
impurity spin-orbitals. Explicit byte limits are checked before the large allocations.
"""
function impurity_two_particle(state::MultiOrbitalEDThermalState, faxis::FermionicMatsubaraAxis,
                               baxis::BosonicMatsubaraAxis; backend::TauGridFourPoint=TauGridFourPoint(), pairs=nothing,
                               max_output_bytes::Integer=536870912, max_operator_bytes::Integer=268435456,
                               max_workspace_bytes::Integer=268435456)
    scale = max(state.beta, 1.0)
    isapprox(state.beta, faxis.beta; rtol=1e-12, atol=1e-12 * scale) || throw(ArgumentError("fermionic axis beta differs from the thermal state"))
    isapprox(state.beta, baxis.beta; rtol=1e-12, atol=1e-12 * scale) || throw(ArgumentError("bosonic axis beta differs from the thermal state"))
    backend.subtract_disconnected || throw(ArgumentError("a connected susceptibility requires backend.subtract_disconnected=true"))
    all(x -> x > 0, (max_output_bytes, max_operator_bytes, max_workspace_bytes)) || throw(ArgumentError("memory limits must be positive"))
    nimp = state.hamiltonian.impurity_dimension
    chosen = _orbital_pair_selection(nimp, pairs)
    active_modes = sort!(unique(vcat([a for (a, b) in chosen], [b for (a, b) in chosen])))
    np, nf, nb = length(chosen), length(faxis), length(baxis)
    dimension = state.metadata[:full_hilbert_dimension]
    dimension <= backend.max_dimension || throw(ArgumentError("Fock dimension $dimension exceeds backend.max_dimension=$(backend.max_dimension)"))
    # The susceptibility, exact bubble, and one frequency-block workspace coexist.
    output_bytes = 2 * big(16) * big(np)^2 * nf^2 * nb
    operators_bytes = 2 * big(16) * length(active_modes) * dimension^2
    workspace_bytes = big(16) * (backend.ntime^3 + 4 * dimension^2 + 2 * nf * backend.ntime^2 + nf^2*nb) +
                      big(8) * (2 * backend.ntime + 1) * dimension
    output_bytes <= max_output_bytes || throw(ArgumentError("estimated susceptibility/bubble storage $output_bytes bytes exceeds max_output_bytes=$max_output_bytes"))
    operators_bytes <= max_operator_bytes || throw(ArgumentError("estimated eigenoperator storage $operators_bytes bytes exceeds max_operator_bytes=$max_operator_bytes"))
    workspace_bytes <= max_workspace_bytes || throw(ArgumentError("estimated four-point workspace $workspace_bytes bytes exceeds max_workspace_bytes=$max_workspace_bytes"))
    operators, energies = _orbital_pair_operators(state, active_modes)
    creation = Dict(mode => Matrix(adjoint(operators[mode])) for mode in active_modes)
    values = Array{ComplexF64}(undef, np, np, nf, nf, nb)
    taus = [(i - 0.5) * state.beta / backend.ntime for i in 1:backend.ntime]
    for (j, (c, d)) in enumerate(chosen), (i, (a, b)) in enumerate(chosen)
        cube = if backend.algorithm === :factored
            _connected_tau_cube_factored(energies, creation[a], operators[b], creation[c], operators[d], state.beta, backend)
        else
            _connected_tau_cube_direct(energies, creation[a], operators[b], creation[c], operators[d], state.beta, backend)
        end
        @views values[i, j, :, :, :] .= _fourier_project_particle_hole(cube, taus, faxis, baxis)
    end
    names = copy(state.hamiltonian.labels[1:nimp])
    metadata = Dict{Symbol,Any}(:method => :midpoint_four_point, :fourier_convention => :particle_hole_physical,
        :disconnected_subtracted => true, :beta => state.beta, :ntime => backend.ntime, :equal_time => backend.equal_time,
        :frequency_box_complete => true, :algorithm => backend.algorithm, :parallel => backend.parallel)
    susceptibility = OrbitalPairSusceptibility(faxis, baxis, chosen, names, values; metadata=metadata, copy_values=false)
    bubble = _orbital_pair_bubble(state, faxis, baxis, chosen, names)
    diagnostics = Dict{Symbol,Any}(:backend => :multiorbital_ed_tau_grid, :ntime => backend.ntime,
        :delta_tau => state.beta / backend.ntime, :hilbert_dimension => dimension, :sector_count => length(state.sectors),
        :operator_bytes => Int(operators_bytes), :output_bytes => Int(output_bytes),
        :estimated_workspace_bytes => Int(workspace_bytes), :pair_count => np, :frequency_box_complete => true,
        :noninteracting_limit => :minus_beta_matrix_GG_diagonal, :convergence_parameter => :ntime)
    return MultiOrbitalImpurityTwoParticleResult(susceptibility, bubble, diagnostics)
end

"""Alias exposing the matrix impurity two-particle route without changing the scalar overload."""
multiorbital_impurity_two_particle(state::MultiOrbitalEDThermalState, faxis::FermionicMatsubaraAxis,
                                   baxis::BosonicMatsubaraAxis; kwargs...) = impurity_two_particle(state, faxis, baxis; kwargs...)

"""Return a scalar-channel view of one specified orbital-pair block for validation and inspection."""
function orbital_pair_block(object::OrbitalPairSusceptibility, first::Tuple{Int,Int}, second::Tuple{Int,Int})
    i = findfirst(==(first), object.pairs)
    j = findfirst(==(second), object.pairs)
    isnothing(i) && throw(ArgumentError("first orbital pair was not evaluated"))
    isnothing(j) && throw(ArgumentError("second orbital pair was not evaluated"))
    values = Array(@view object.values[i, j, :, :, :])
    return GeneralizedSusceptibility(object.fermionic_axis, object.bosonic_axis, values; channel=:orbital_pair,
        metadata=Dict{Symbol,Any}(:first => first, :second => second, :fourier_convention => :particle_hole_physical), copy_values=false)
end

"""Contract an explicitly selected orbital-pair density channel using user-supplied complex weights.

The result is `sum(conj(w[i]) * χ[i,j] * w[j])`. Choose charge/magnetic
weights only when the model and the pair ordering justify the corresponding
spin-density interpretation; no unverified spin symmetry is assumed.
"""
function project_orbital_channel(object::OrbitalPairSusceptibility, weights::AbstractVector{<:Number}; channel::Symbol=:custom)
    length(weights) == length(object.pairs) || throw(DimensionMismatch("projection weights must match the selected pair count"))
    all(_isfinite_complex, weights) || throw(ArgumentError("projection weights must be finite"))
    w = ComplexF64.(weights)
    nf, nb = length(object.fermionic_axis), length(object.bosonic_axis)
    projected = zeros(ComplexF64, nf, nf, nb)
    for j in eachindex(w), i in eachindex(w)
        if iszero(w[i]) || iszero(w[j])
            continue
        end
        @views projected .+= conj(w[i]) * w[j] .* object.values[i, j, :, :, :]
    end
    return GeneralizedSusceptibility(object.fermionic_axis, object.bosonic_axis, projected; channel=channel,
        metadata=Dict{Symbol,Any}(:construction => :explicit_orbital_pair_projection,
            :pairs => copy(object.pairs), :weights => w, :fourier_convention => :particle_hole_physical), copy_values=false)
end

function _orbital_pair_compatible(a::OrbitalPairSusceptibility, b::OrbitalPairSusceptibility)
    a.pairs == b.pairs || throw(ArgumentError("orbital-pair selections differ"))
    a.labels == b.labels || throw(ArgumentError("orbital labels differ"))
    a.fermionic_axis.indices == b.fermionic_axis.indices || throw(ArgumentError("fermionic frequency grids differ"))
    a.bosonic_axis.indices == b.bosonic_axis.indices || throw(ArgumentError("bosonic frequency grids differ"))
    a.fermionic_axis.beta == b.fermionic_axis.beta || throw(ArgumentError("inverse temperatures differ"))
    return nothing
end

"""Relative Frobenius deviation for two grid-convergence or dense/sectorized comparisons."""
function multiorbital_two_particle_residual(a::MultiOrbitalImpurityTwoParticleResult, b::MultiOrbitalImpurityTwoParticleResult)
    _orbital_pair_compatible(a.susceptibility, b.susceptibility)
    x, y = a.susceptibility.values, b.susceptibility.values
    return norm(x - y) / max(norm(x), norm(y), eps(Float64))
end

"""Extract the finite-frequency irreducible orbital-pair vertex, `Γ=χ₀⁻¹-χ⁻¹`.

The inversion is a separate, explicitly regularized operation. Missing orbital pairs,
ill-conditioned frequency boxes, and pseudoinverse truncation do not represent an exact
many-body vertex; inspect the per-bosonic-frequency inversion diagnostics.
"""
function multiorbital_irreducible_vertex(bubble::OrbitalPairSusceptibility, susceptibility::OrbitalPairSusceptibility;
                                         regularization::Real=0.0, inverse_rtol::Real=sqrt(eps(Float64)),
                                         inverse_atol::Real=0.0, pseudoinverse::Bool=false,
                                         max_inversion_bytes::Integer=536870912)
    _orbital_pair_compatible(bubble, susceptibility)
    np, _, nf, _, nb = size(susceptibility.values)
    dimension = np * nf
    big(16) * (4 * big(dimension)^2 + big(np)^2 * nf^2 * nb) <= max_inversion_bytes ||
        throw(ArgumentError("estimated vertex inversion storage exceeds max_inversion_bytes=$max_inversion_bytes"))
    output = similar(susceptibility.values)
    info = Vector{NamedTuple}(undef, nb)
    # Flatten in the explicitly documented pair-major, then fermionic-frequency ordering.
    for iΩ in 1:nb
        χ = Matrix{ComplexF64}(undef, dimension, dimension)
        χ0 = Matrix{ComplexF64}(undef, dimension, dimension)
        for j in 1:np, i in 1:np, nprime in 1:nf, n in 1:nf
            row, column = (i-1)*nf+n, (j-1)*nf+nprime
            χ[row, column] = susceptibility.values[i, j, n, nprime, iΩ]
            χ0[row, column] = bubble.values[i, j, n, nprime, iΩ]
        end
        inverse0, d0 = _stable_inverse(χ0; diagonal_shift=regularization, rtol=inverse_rtol,
            atol=inverse_atol, pseudoinverse=pseudoinverse, return_diagnostics=true)
        inverseχ, dχ = _stable_inverse(χ; diagonal_shift=regularization, rtol=inverse_rtol,
            atol=inverse_atol, pseudoinverse=pseudoinverse, return_diagnostics=true)
        for j in 1:np, i in 1:np, nprime in 1:nf, n in 1:nf
            output[i, j, n, nprime, iΩ] = inverse0[(i-1)*nf+n, (j-1)*nf+nprime] -
                inverseχ[(i-1)*nf+n, (j-1)*nf+nprime]
        end
        info[iΩ] = (bubble=d0, susceptibility=dχ, dimension=dimension)
    end
    metadata = Dict{Symbol,Any}(:construction => :inverse_bethe_salpeter,
        :regularization => Float64(regularization), :inverse_rtol => Float64(inverse_rtol),
        :inverse_atol => Float64(inverse_atol), :pseudoinverse => pseudoinverse,
        :inverse_diagnostics => info, :orbital_pair_order => :pair_major_frequency_minor)
    return OrbitalPairIrreducibleVertex(susceptibility.fermionic_axis, susceptibility.bosonic_axis,
        copy(susceptibility.pairs), copy(susceptibility.labels), output, metadata)
end
