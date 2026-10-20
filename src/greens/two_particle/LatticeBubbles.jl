# Momentum-resolved single-particle propagators and particle-hole bubbles for diagrammatic DMFT.

"""Scalar single-band Green function `G(k,iν)` on an explicit momentum list and fermionic Matsubara axis."""
struct MomentumResolvedGreenFunction{A<:FermionicMatsubaraAxis,K,M}
    axis::A
    kpoints::K
    values::Matrix{ComplexF64}
    metadata::M
end

function MomentumResolvedGreenFunction(axis::A, kpoints, values; metadata=Dict{Symbol,Any}()) where {A<:FermionicMatsubaraAxis}
    ks = collect(kpoints)
    matrix = values isa Matrix{ComplexF64} ? copy(values) : ComplexF64.(values)
    size(matrix) == (length(ks), length(axis)) || throw(DimensionMismatch("momentum Green-function values must have shape (nk, nfermionic)"))
    return MomentumResolvedGreenFunction{A,typeof(ks),typeof(metadata)}(axis, ks, matrix, metadata)
end

"""
    momentum_green(dispersion, axis, self_energy; chemical_potential=0, kpoints=eachindex(dispersion))

Construct the scalar single-band lattice propagator `G(k,iν) = [iν + μ - εₖ - Σ(iν)]⁻¹` from a local DMFT self-energy.
"""
function momentum_green(dispersion::AbstractVector{<:Real}, axis::FermionicMatsubaraAxis, sigma::SelfEnergy;
                        chemical_potential::Real=0.0, kpoints=collect(eachindex(dispersion)))
    matrix_dimension(sigma) == 1 || throw(DimensionMismatch("momentum_green currently supports a scalar local self-energy"))
    _compatible_frequency_objects(sigma, zero_self_energy(axis, 1; labels=[:local]))
    energies = Float64.(collect(dispersion))
    ks = collect(kpoints)
    length(ks) == length(energies) || throw(DimensionMismatch("kpoint count must equal dispersion count"))
    values = Matrix{ComplexF64}(undef, length(energies), length(axis))
    @inbounds for ik in eachindex(energies), iν in eachindex(axis.values)
        values[ik, iν] = inv(axis[iν] + chemical_potential - energies[ik] - sigma[1, 1, iν])
    end
    metadata = Dict{Symbol,Any}(:construction => :local_self_energy_lattice_green, :chemical_potential => Float64(chemical_potential))
    return MomentumResolvedGreenFunction(axis, ks, values; metadata=metadata)
end

"""
    periodic_momentum_shift_maps(shape)

Generate exact `k -> k+q` index maps for every momentum transfer on a periodic Cartesian mesh with first index varying fastest. The returned q labels are integer mesh coordinates.
"""
function periodic_momentum_shift_maps(shape::NTuple{D,<:Integer}) where {D}
    dims = Int.(collect(shape))
    all(>(0), dims) || throw(ArgumentError("all momentum-mesh dimensions must be positive"))
    nk = prod(dims)
    linear_index(coords) = 1 + sum(coords[d] * prod(dims[1:d-1]; init=1) for d in 1:D)
    coordinates = Vector{NTuple{D,Int}}(undef, nk)
    for linear in 0:nk-1
        remainder = linear
        coordinates[linear + 1] = ntuple(d -> begin
            coordinate = mod(remainder, dims[d])
            remainder = div(remainder, dims[d])
            coordinate
        end, D)
    end
    maps = Vector{Vector{Int}}(undef, nk)
    for (iq, q) in enumerate(coordinates)
        mapping = Vector{Int}(undef, nk)
        for (ik, k) in enumerate(coordinates)
            shifted = ntuple(d -> mod(k[d] + q[d], dims[d]), D)
            mapping[ik] = linear_index(shifted)
        end
        maps[iq] = mapping
    end
    return (qpoints=coordinates, maps=maps)
end

function _shifted_fermionic_index(axis::FermionicMatsubaraAxis, bosonic_index::Int)
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    return Int[get(lookup, index + bosonic_index, 0) for index in axis.indices]
end

function _bubble_frequency_maps(axis::FermionicMatsubaraAxis, bosonic_axis::BosonicMatsubaraAxis)
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    shifted = Vector{Vector{Int}}(undef, length(bosonic_axis))
    active = Vector{Vector{Int}}(undef, length(bosonic_axis))
    for (iΩ, m) in enumerate(bosonic_axis.indices)
        frequency_map = Vector{Int}(undef, length(axis))
        @inbounds for iν in eachindex(axis.indices)
            frequency_map[iν] = get(lookup, axis.indices[iν] + m, 0)
        end
        shifted[iΩ] = frequency_map
        active[iΩ] = findall(x -> !iszero(x), frequency_map)
    end
    return shifted, active
end

function _lattice_bubble_at(green::MomentumResolvedGreenFunction, mapping::Vector{Int}, bosonic_axis::BosonicMatsubaraAxis,
                            shifted_frequencies::Vector{Vector{Int}}, active_indices_by_bosonic::Vector{Vector{Int}}, iq::Int,
                            channel::Symbol, missing_frequency::Symbol)
    nk = size(green.values, 1)
    beta = green.axis.beta
    values = zeros(ComplexF64, length(green.axis), length(green.axis), length(bosonic_axis))
    @inbounds for iΩ in eachindex(bosonic_axis.indices)
        shifted_frequency = shifted_frequencies[iΩ]
        for iν in eachindex(green.axis.indices)
            jν = shifted_frequency[iν]
            if iszero(jν)
                missing_frequency === :error && throw(ArgumentError("fermionic frequency box does not contain ν+Ω for ν index $(green.axis.indices[iν]) and Ω index $(bosonic_axis.indices[iΩ])"))
                continue
            end
            accumulator = zero(ComplexF64)
            @simd for ik in 1:nk
                accumulator += green.values[ik, iν] * green.values[mapping[ik], jν]
            end
            values[iν, iν, iΩ] = -beta * accumulator / nk
        end
    end
    metadata = Dict{Symbol,Any}(
        :construction => :particle_hole_lattice_bubble,
        :nk => nk,
        :q_index => iq,
        :missing_frequency_policy => missing_frequency,
        :active_indices_by_bosonic => [copy(active) for active in active_indices_by_bosonic],
        :frequency_box_complete => all(length(active) == length(green.axis) for active in active_indices_by_bosonic),
        :diagonal_in_fermionic_indices => true,
    )
    return GeneralizedSusceptibility(green.axis, bosonic_axis, values; channel=channel, metadata=metadata, copy_values=false)
end

"""
    lattice_bubble(green, qmaps, bosonic_axis; channel=:density, missing_frequency=:error, parallel=:serial)

Construct the generalized particle-hole bubble `χ₀(q;ν,ν′,Ω) = -β δνν′ Nₖ⁻¹ Σₖ G(k,ν)G(k+q,ν+Ω)`. The shifted-frequency lookup is constructed once for the complete q grid, and `parallel=:threads` distributes independent q points across Julia threads. This normalization is paired with `physical_susceptibility(...; beta_normalization=β^2)`.
"""
function lattice_bubble(green::MomentumResolvedGreenFunction, qmaps, bosonic_axis::BosonicMatsubaraAxis;
                        channel::Symbol=:density, missing_frequency::Symbol=:error, parallel::Symbol=:serial)
    isapprox(green.axis.beta, bosonic_axis.beta; rtol=0.0, atol=16eps(Float64) * max(green.axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic Matsubara axes must share beta"))
    missing_frequency in (:error, :zero) || throw(ArgumentError("missing_frequency must be :error or :zero"))
    parallel in (:serial, :threads) || throw(ArgumentError("parallel must be :serial or :threads"))
    maps = [Int.(collect(mapping)) for mapping in qmaps]
    nk = size(green.values, 1)
    all(mapping -> length(mapping) == nk && all(index -> 1 <= index <= nk, mapping), maps) ||
        throw(ArgumentError("each q map must contain one valid shifted momentum index per k point"))
    shifted_frequencies, active_indices_by_bosonic = _bubble_frequency_maps(green.axis, bosonic_axis)
    missing_per_q = sum(length(green.axis) - length(active) for active in active_indices_by_bosonic)
    if missing_frequency === :error && missing_per_q > 0
        for (iΩ, shifted) in enumerate(shifted_frequencies), iν in eachindex(shifted)
            iszero(shifted[iν]) || continue
            throw(ArgumentError("fermionic frequency box does not contain ν+Ω for ν index $(green.axis.indices[iν]) and Ω index $(bosonic_axis.indices[iΩ])"))
        end
    end
    bubbles = Vector{GeneralizedSusceptibility}(undef, length(maps))
    if parallel === :threads && Base.Threads.nthreads() > 1
        Base.Threads.@threads for iq in eachindex(maps)
            bubbles[iq] = _lattice_bubble_at(green, maps[iq], bosonic_axis, shifted_frequencies, active_indices_by_bosonic, iq, channel, missing_frequency)
        end
    else
        for iq in eachindex(maps)
            bubbles[iq] = _lattice_bubble_at(green, maps[iq], bosonic_axis, shifted_frequencies, active_indices_by_bosonic, iq, channel, missing_frequency)
        end
    end
    missing_count = missing_per_q * length(maps)
    return bubbles, (missing_frequency_entries=missing_count, frequency_box_complete=iszero(missing_count))
end

"""Construct the local particle-hole bubble `-β G(ν)G(ν+Ω)δνν′` from a scalar local Green function."""
function local_bubble(green::GreenFunction, bosonic_axis::BosonicMatsubaraAxis; channel::Symbol=:density, missing_frequency::Symbol=:error)
    matrix_dimension(green) == 1 || throw(DimensionMismatch("local_bubble currently supports scalar Green functions"))
    momentum = MomentumResolvedGreenFunction(green.axis, [1], reshape(scalar_frequency_values(green), 1, length(green.axis));
                                             metadata=Dict{Symbol,Any}(:construction => :local_green_wrapper))
    bubbles, diagnostics = lattice_bubble(momentum, [[1]], bosonic_axis; channel=channel, missing_frequency=missing_frequency, parallel=:serial)
    bubble = only(bubbles)
    bubble.metadata[:construction] = :local_particle_hole_bubble
    bubble.metadata[:frequency_box_complete] = diagnostics.frequency_box_complete
    bubble.metadata[:diagonal_in_fermionic_indices] = true
    return bubble
end


"""Matrix momentum Green function with storage order `(a, b, k, ν)` and explicit orbital labels and weights."""
struct MatrixMomentumResolvedGreenFunction{A<:FermionicMatsubaraAxis,K,M}
    axis::A
    kpoints::K
    values::Array{ComplexF64,4}
    weights::Vector{Float64}
    labels::Vector{Symbol}
    metadata::M
end

function MatrixMomentumResolvedGreenFunction(axis::A, kpoints, values::AbstractArray{<:Number,4};
                                              weights=nothing, labels=nothing, metadata=Dict{Symbol,Any}()) where {A<:FermionicMatsubaraAxis}
    cube = ComplexF64.(values)
    n, m, nk, nfreq = size(cube)
    n > 0 && n == m && nk > 0 && nfreq == length(axis) || throw(DimensionMismatch("matrix Green values must have shape (n,n,nk,nfermionic)"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), cube) || throw(ArgumentError("matrix Green values must be finite"))
    ks = collect(kpoints)
    length(ks) == nk || throw(DimensionMismatch("momentum-point count does not match matrix Green values"))
    w = isnothing(weights) ? fill(inv(nk), nk) : Float64.(collect(weights))
    length(w) == nk || throw(DimensionMismatch("Green-function momentum weights have incorrect size"))
    all(x -> isfinite(x) && x >= 0, w) && sum(w) > 0 || throw(ArgumentError("momentum weights must be finite and nonnegative"))
    w ./= sum(w)
    names = isnothing(labels) ? [Symbol("orbital_", i) for i in 1:n] : Symbol.(collect(labels))
    length(names) == n && length(unique(names)) == n || throw(DimensionMismatch("orbital labels must be distinct and match the orbital dimension"))
    return MatrixMomentumResolvedGreenFunction{A,typeof(ks),typeof(metadata)}(axis, ks, cube, w, names, metadata)
end

"""Matrix Dyson equation on a momentum Hamiltonian cube, with a frequency-local matrix self-energy."""
function momentum_green(hamiltonians::AbstractArray{<:Number,3}, axis::FermionicMatsubaraAxis, sigma::SelfEnergy;
                        chemical_potential::Real=0.0, kpoints=collect(axes(hamiltonians, 3)), labels=nothing, weights=nothing)
    n, m, nk = size(hamiltonians)
    n > 0 && n == m && nk > 0 || throw(DimensionMismatch("matrix dispersion must have shape (n,n,nk)"))
    matrix_dimension(sigma) == n || throw(DimensionMismatch("orbital dispersion and self-energy dimensions disagree"))
    _compatible_frequency_objects(sigma, zero_self_energy(axis, n))
    isfinite(chemical_potential) || throw(ArgumentError("chemical potential must be finite"))
    h = ComplexF64.(hamiltonians)
    all(z -> isfinite(real(z)) && isfinite(imag(z)), h) || throw(ArgumentError("orbital dispersion must be finite"))
    for ik in 1:nk
        isapprox(view(h, :, :, ik), adjoint(view(h, :, :, ik)); atol=1e-10, rtol=1e-10) ||
            throw(ArgumentError("matrix Dyson propagation requires a Hermitian single-particle Hamiltonian"))
    end
    declared = isnothing(labels) ? [Symbol("orbital_", i) for i in 1:n] : Symbol.(collect(labels))
    generic = [Symbol("component_", i) for i in 1:n]
    sigma.labels == declared || sigma.labels == generic ||
        throw(ArgumentError("explicit self-energy orbital labels disagree with the matrix Hamiltonian"))
    cube = Array{ComplexF64}(undef, n, n, nk, length(axis))
    identity = Matrix{ComplexF64}(I, n, n)
    @inbounds for iν in eachindex(axis.values), ik in 1:nk
        kernel = (axis[iν] + chemical_potential) * identity - view(h, :, :, ik) - sigma[iν]
        cube[:, :, ik, iν] .= _solve_inverse_matrix(kernel)
    end
    return MatrixMomentumResolvedGreenFunction(axis, kpoints, cube; weights=weights, labels=labels,
        metadata=Dict{Symbol,Any}(:construction => :matrix_local_self_energy_dyson, :chemical_potential => Float64(chemical_potential)))
end

"""Frequency-diagonal orbital-pair bubble with `(a,b,c,d,ν,Ω)` ordering at one momentum transfer.

The component is `-β Σₖ wₖ G[a,c](k+q,ν+Ω) G[d,b](k,ν)` and is not a scalar `GeneralizedSusceptibility`.
"""
struct OrbitalLatticeBubble{F<:FermionicMatsubaraAxis,B<:BosonicMatsubaraAxis}
    fermionic_axis::F
    bosonic_axis::B
    values::Array{ComplexF64,6}
    labels::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

"""Construct an orbital-resolved particle-hole bubble without implying a matrix Bethe-Salpeter or DΓA solver."""
function orbital_lattice_bubble(green::MatrixMomentumResolvedGreenFunction, qmaps, bosonic_axis::BosonicMatsubaraAxis;
                                missing_frequency::Symbol=:error)
    get(green.metadata, :bloch_gauge, :periodic) === :periodic ||
        throw(ArgumentError("bare orbital bubbles require the periodic Bloch gauge; embedded orbital centers require momentum-folding phase transport"))
    isapprox(green.axis.beta, bosonic_axis.beta; atol=16eps(Float64) * max(green.axis.beta, 1.0), rtol=0.0) ||
        throw(ArgumentError("fermionic and bosonic axes must have the same inverse temperature"))
    missing_frequency in (:error, :zero) || throw(ArgumentError("missing_frequency must be :error or :zero"))
    n, _, nk, _ = size(green.values)
    isnothing(qmaps) && throw(ArgumentError("orbital_lattice_bubble requires explicit k-to-k+q index maps"))
    maps = [Int.(collect(mapping)) for mapping in qmaps]
    all(mapping -> length(mapping) == nk && sort(mapping) == collect(1:nk), maps) ||
        throw(ArgumentError("each orbital q map must be a permutation of the momentum grid"))
    shifted, active = _bubble_frequency_maps(green.axis, bosonic_axis)
    missing = sum(length(green.axis) - length(indices) for indices in active)
    if missing_frequency === :error && missing > 0
        throw(ArgumentError("fermionic frequency box does not contain all ν+Ω shifts"))
    end
    output = Vector{OrbitalLatticeBubble}(undef, length(maps))
    @inbounds for (iq, mapping) in enumerate(maps)
        values = zeros(ComplexF64, n, n, n, n, length(green.axis), length(bosonic_axis))
        for iΩ in eachindex(bosonic_axis.indices), iν in eachindex(green.axis.indices)
            jν = shifted[iΩ][iν]
            iszero(jν) && continue
            for ik in 1:nk, a in 1:n, b in 1:n, c in 1:n, d in 1:n
                values[a, b, c, d, iν, iΩ] -= green.axis.beta * green.weights[ik] *
                    green.values[a, c, mapping[ik], jν] * green.values[d, b, ik, iν]
            end
        end
        metadata = Dict{Symbol,Any}(:q_index => iq, :construction => :orbital_particle_hole_bubble,
            :frequency_box_complete => iszero(missing), :missing_frequency_policy => missing_frequency,
            :diagonal_in_fermionic_indices => true, :normalization => :minus_beta_weighted_sum)
        output[iq] = OrbitalLatticeBubble(green.axis, bosonic_axis, values, copy(green.labels), metadata)
    end
    return output, (missing_frequency_entries=missing * length(maps), frequency_box_complete=iszero(missing))
end
