# ============================================================================
# OrbitalCrossingChannels.jl: preserved implementation from src/embedding/OrbitalCrossingChannels.jl
# ============================================================================


# Spin-rotation-invariant density/magnetic particle-hole ladder reconstruction.
# Its crossing relations require complete fermionic and bosonic transfer coverage.

"""Density and magnetic particle-hole full vertices, retaining every orbital-pair and fermionic-frequency index.

The spin decomposition is defined by the caller's SU(2) convention. A construction-time
assertion of spin symmetry does not constitute independent verification of the supplied vertices.
"""
struct OrbitalSpinLadderChannels{F<:OrbitalPairSusceptibility}
    local_density::F
    local_magnetic::F
    density::Vector{F}
    magnetic::Vector{F}
    qmaps::Vector{Vector{Int}}
    momentum_lookup::Matrix{Int}
    fermion_lookup::Dict{Int,Int}
    boson_lookup::Dict{Int,Int}
    metadata::Dict{Symbol,Any}
end

function _orbital_spin_same_grid(reference::OrbitalPairSusceptibility, other::OrbitalPairSusceptibility)
    reference.pairs == other.pairs && reference.labels == other.labels || throw(ArgumentError("spin channels require identical orbital-pair ordering and labels"))
    reference.fermionic_axis.indices == other.fermionic_axis.indices && reference.bosonic_axis.indices == other.bosonic_axis.indices ||
        throw(ArgumentError("spin channels require identical fermionic and bosonic indices"))
    reference.fermionic_axis.beta == other.fermionic_axis.beta && reference.bosonic_axis.beta == other.bosonic_axis.beta ||
        throw(ArgumentError("spin-channel inverse temperatures differ"))
    return nothing
end

"""Construct SU(2) density/magnetic ladders with explicit spin-symmetry acknowledgment and bounded retained storage.

The spatial-orbital pairs must be in complete canonical order. The `qmaps` are momentum
permutations: `qmaps[iq][ik]` is the index of `k+q`. The source vertices must be
full vertices in the `χ = χ₀ + χ₀ F χ₀` convention, not irreducible vertices.
"""
function OrbitalSpinLadderChannels(local_density::OrbitalPairSusceptibility, local_magnetic::OrbitalPairSusceptibility,
                                   density::AbstractVector{<:OrbitalPairSusceptibility}, magnetic::AbstractVector{<:OrbitalPairSusceptibility}, qmaps;
                                   spin_symmetry_certified::Bool=false, max_retained_bytes::Integer=536870912)
    spin_symmetry_certified || throw(ArgumentError("the density/magnetic decomposition requires an explicit SU(2) symmetry declaration"))
    _orbital_spin_same_grid(local_density, local_magnetic)
    norb = length(local_density.labels)
    local_density.pairs == [(a, b) for a in 1:norb for b in 1:norb] ||
        throw(ArgumentError("the crossing reconstruction requires all spatial-orbital pairs in canonical ordering"))
    length(density) == length(magnetic) > 0 || throw(DimensionMismatch("density and magnetic q-block counts must match and be positive"))
    mappings = [Int.(collect(mapping)) for mapping in qmaps]
    length(mappings) == length(density) || throw(DimensionMismatch("momentum-shift maps and q-block counts differ"))
    isempty(mappings[1]) && throw(ArgumentError("momentum maps must not be empty"))
    nk = length(mappings[1])
    all(mapping -> length(mapping) == nk && sort(mapping) == collect(1:nk), mappings) ||
        throw(ArgumentError("every momentum-shift map must be a permutation of the same k grid"))
    length(unique(Tuple.(mappings))) == length(mappings) || throw(ArgumentError("duplicate momentum-shift maps are ambiguous"))
    for item in density
        _orbital_spin_same_grid(local_density, item)
    end
    for item in magnetic
        _orbital_spin_same_grid(local_density, item)
    end
    bytes = big(16) * length(local_density.values) * (2 + length(density) + length(magnetic)) + big(sizeof(Int)) * nk^2
    max_retained_bytes > 0 && bytes <= max_retained_bytes || throw(ArgumentError("retained orbital ladders exceed max_retained_bytes"))
    momentum_lookup = zeros(Int, nk, nk)
    for (iq, mapping) in enumerate(mappings), ik in 1:nk
        iszero(momentum_lookup[ik, mapping[ik]]) || throw(ArgumentError("momentum maps do not assign a unique transfer to each k,kprime pair"))
        momentum_lookup[ik, mapping[ik]] = iq
    end
    fermion_lookup = Dict(index => location for (location, index) in enumerate(local_density.fermionic_axis.indices))
    boson_lookup = Dict(index => location for (location, index) in enumerate(local_density.bosonic_axis.indices))
    metadata = Dict{Symbol,Any}(:spin_assumption => :caller_declared_su2, :crossing_relation => :two_particle_hole_ladders,
        :local_subtraction => :full_local_vertex_at_identical_frequencies, :high_frequency_completion => :none,
        :retained_vertex_bytes => Int(bytes), :full_multiorbital_dga => false)
    return OrbitalSpinLadderChannels(local_density, local_magnetic, collect(density), collect(magnetic), mappings,
        momentum_lookup, fermion_lookup, boson_lookup, metadata)
end

"""Solve density and magnetic particle-hole ladders and recover their full vertices for each momentum transfer.

The input irreducible vertices must already be projected into a consistent SU(2)
spatial-orbital basis. In particular, this method does *not* infer that decomposition
from a generic spin-orbital four-point tensor.
"""
function orbital_spin_ladders(local_bubble::OrbitalPairSusceptibility, local_density_vertex::OrbitalPairIrreducibleVertex,
                              local_magnetic_vertex::OrbitalPairIrreducibleVertex, q_bubbles::AbstractVector{<:OrbitalPairSusceptibility}, qmaps;
                              spin_symmetry_certified::Bool=false, max_workspace_bytes::Integer=536870912,
                              max_retained_bytes::Integer=536870912, min_relative_pivot::Real=1e-14)
    _orbital_bse_compatible(local_bubble, local_density_vertex)
    _orbital_bse_compatible(local_bubble, local_magnetic_vertex)
    nq = length(q_bubbles)
    nq > 0 || throw(ArgumentError("at least one q bubble is required"))
    nbytes = big(16) * length(local_bubble.values) * (2 + 2 * nq)
    max_retained_bytes > 0 && nbytes <= max_retained_bytes || throw(ArgumentError("full vertices exceed max_retained_bytes"))
    local_d = orbital_full_vertex(local_bubble, bethe_salpeter(local_bubble, local_density_vertex;
        max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot);
        max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot)
    local_m = orbital_full_vertex(local_bubble, bethe_salpeter(local_bubble, local_magnetic_vertex;
        max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot);
        max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot)
    density = typeof(local_d)[]
    magnetic = typeof(local_m)[]
    for bubble in q_bubbles
        _orbital_spin_same_grid(local_bubble, bubble)
        response_d = bethe_salpeter(bubble, local_density_vertex;
            max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot)
        response_m = bethe_salpeter(bubble, local_magnetic_vertex;
            max_workspace_bytes=max_workspace_bytes, min_relative_pivot=min_relative_pivot)
        push!(density, orbital_full_vertex(bubble, response_d; max_workspace_bytes=max_workspace_bytes,
            min_relative_pivot=min_relative_pivot))
        push!(magnetic, orbital_full_vertex(bubble, response_m; max_workspace_bytes=max_workspace_bytes,
            min_relative_pivot=min_relative_pivot))
    end
    return OrbitalSpinLadderChannels(local_d, local_m, density, magnetic, qmaps;
        spin_symmetry_certified=spin_symmetry_certified, max_retained_bytes=max_retained_bytes)
end

function _orbital_crossing_lookup(channels::OrbitalSpinLadderChannels, iq::Int, ik::Int, ikprime::Int,
                                   iν::Int, iνprime::Int, iΩ::Int)
    1 <= iq <= length(channels.density) || throw(BoundsError(channels.density, iq))
    1 <= ik <= length(channels.qmaps[1]) || throw(BoundsError(channels.qmaps[1], ik))
    1 <= ikprime <= length(channels.qmaps[1]) || throw(BoundsError(channels.qmaps[1], ikprime))
    fermions = channels.local_density.fermionic_axis.indices
    bosons = channels.local_density.bosonic_axis.indices
    1 <= iν <= length(fermions) && 1 <= iνprime <= length(fermions) && 1 <= iΩ <= length(bosons) ||
        throw(ArgumentError("requested crossing frequency index is outside the supplied Matsubara axes"))
    n, np, m = fermions[iν], fermions[iνprime], bosons[iΩ]
    cross_q = channels.momentum_lookup[ik, ikprime]
    iszero(cross_q) && throw(ArgumentError("momentum maps lack the crossed transfer kprime-k"))
    cross_n = get(channels.fermion_lookup, np - m, 0)
    cross_np = iνprime
    cross_m = get(channels.boson_lookup, np - n, 0)
    iszero(cross_n) && throw(ArgumentError("crossed fermionic index $(np-m) is missing; enlarge the frequency box or supply verified asymptotics"))
    iszero(cross_m) && throw(ArgumentError("crossed bosonic index $(np-n) is missing; enlarge the frequency box or supply verified asymptotics"))
    return (cross_q, cross_n, cross_np, cross_m)
end

"""Return density and magnetic vertex elements from longitudinal and transverse particle-hole ladders.

For external indices `(ν,ν′,Ω)` and momenta `(q,k,k′)`, the crossed transfer
is `(k′-k, ν′-Ω, ν′, ν′-ν)`. Orbital indices `(a,b,c,d)` transform as
`(c,b,a,d)`. The SU(2) coefficients follow the ladder parquet reduction:
`F_d = F_d^loc + ΔF_d(q) - ΔF_d(cross)/2 - 3ΔF_m(cross)/2` and
`F_m = F_m^loc + ΔF_m(q) - ΔF_d(cross)/2 + ΔF_m(cross)/2`.
Incomplete finite-frequency boxes are rejected, never silently zero filled.
"""
function orbital_crossed_vertex(channels::OrbitalSpinLadderChannels, iq::Int, ik::Int, ikprime::Int,
                                 iν::Int, iνprime::Int, iΩ::Int, a::Int, b::Int, c::Int, d::Int)
    n = length(channels.local_density.labels)
    all(x -> 1 <= x <= n, (a, b, c, d)) || throw(ArgumentError("orbital indices are outside the declared spatial basis"))
    cross_q, cross_n, cross_np, cross_m = _orbital_crossing_lookup(channels, iq, ik, ikprime, iν, iνprime, iΩ)
    p = (a - 1) * n + b
    r = (c - 1) * n + d
    pcross = (c - 1) * n + b
    rcross = (a - 1) * n + d
    ld = channels.local_density.values
    lm = channels.local_magnetic.values
    direct_d = channels.density[iq].values[p, r, iν, iνprime, iΩ] - ld[p, r, iν, iνprime, iΩ]
    direct_m = channels.magnetic[iq].values[p, r, iν, iνprime, iΩ] - lm[p, r, iν, iνprime, iΩ]
    cross_d = channels.density[cross_q].values[pcross, rcross, cross_n, cross_np, cross_m] - ld[pcross, rcross, cross_n, cross_np, cross_m]
    cross_magnetic = channels.magnetic[cross_q].values[pcross, rcross, cross_n, cross_np, cross_m] - lm[pcross, rcross, cross_n, cross_np, cross_m]
    return (density=ld[p, r, iν, iνprime, iΩ] + direct_d - (cross_d + 3 * cross_magnetic) / 2,
        magnetic=lm[p, r, iν, iνprime, iΩ] + direct_m - (cross_d - cross_magnetic) / 2)
end

"""Report how many crossed frequency/momentum transfers are unavailable on the supplied finite grids.

The count scans transfer *indices*, not orbital tensor entries; a nonzero count means
that unrestricted channel reconstruction is not defined on the requested grid.
"""
function orbital_crossing_coverage(channels::OrbitalSpinLadderChannels)
    nf = length(channels.local_density.fermionic_axis)
    nb = length(channels.local_density.bosonic_axis)
    nk = length(channels.qmaps[1])
    missing = 0
    examples = String[]
    for ik in 1:nk, ikprime in 1:nk, iΩ in 1:nb, iνprime in 1:nf, iν in 1:nf
        try
            _orbital_crossing_lookup(channels, 1, ik, ikprime, iν, iνprime, iΩ)
        catch error
            error isa ArgumentError || rethrow()
            missing += 1
            length(examples) < 5 && push!(examples, sprint(showerror, error))
        end
    end
    return (complete=iszero(missing), missing=missing, checked=big(nk)^2 * nf^2 * nb, examples=examples)
end

"""Check the spin SU(2) generators against a spin-interleaved local Hamiltonian and interaction tensor.

The spin ordering is `(orbital1_up, orbital1_down, ...)`. The one-body matrix must
commute with both `S⁺` and `Sᶻ`, and the antisymmetrized interaction must be
invariant under their action on all four indices. This certifies the local
Hamiltonian symmetry, *not* the numerical symmetry of independently supplied
frequency-dependent channel vertices or the full lattice.
"""
function orbital_su2_symmetry(one_body::AbstractMatrix, interaction::SpinOrbitalInteraction; tolerance::Real=1e-10)
    n = length(interaction.labels)
    iseven(n) || throw(ArgumentError("spin-interleaved SU(2) validation requires an even spin-orbital dimension"))
    size(one_body) == (n, n) || throw(DimensionMismatch("one-body matrix and interaction dimensions differ"))
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative and finite"))
    all(_isfinite_complex, one_body) || throw(ArgumentError("one-body matrix must be finite"))
    raising = zeros(ComplexF64, n, n)
    zspin = zeros(ComplexF64, n, n)
    for orbital in 1:(n ÷ 2)
        raising[2 * orbital - 1, 2 * orbital] = 1
        zspin[2 * orbital - 1, 2 * orbital - 1] = 0.5
        zspin[2 * orbital, 2 * orbital] = -0.5
    end
    function generator_residual(generator)
        hres = opnorm(generator * one_body - one_body * generator, Inf)
        tensor = interaction.tensor
        maximum_error = 0.0
        @inbounds for a in 1:n, b in 1:n, c in 1:n, d in 1:n
            difference = zero(ComplexF64)
            for e in 1:n
                difference += generator[a, e] * tensor[e, b, c, d] + generator[b, e] * tensor[a, e, c, d]
                difference -= tensor[a, b, e, d] * generator[e, c] + tensor[a, b, c, e] * generator[e, d]
            end
            maximum_error = max(maximum_error, abs(difference))
        end
        return (one_body=hres, interaction=maximum_error)
    end
    transverse = generator_residual(raising)
    longitudinal = generator_residual(zspin)
    scale = max(1.0, opnorm(one_body, Inf), maximum(abs, interaction.tensor))
    residual = max(transverse.one_body, transverse.interaction, longitudinal.one_body, longitudinal.interaction) / scale
    return (valid=residual <= tolerance, relative_residual=residual, raising=transverse,
        zspin=longitudinal, ordering=:interleaved_spin, tolerance=Float64(tolerance))
end

# ============================================================================
# OrbitalCrossingCore.jl: preserved implementation from src/embedding/OrbitalCrossingCore.jl
# ============================================================================


# Separate the physical vertex-frequency core from the larger finite crossing-support box.

"""A core frequency box whose crossing transforms are evaluated on a larger, explicitly supplied support box.

This object only defines crossing lookups. It does not extrapolate missing vertex
values, establish a high-frequency asymptotic expansion, or certify a DGA self-energy.
"""
struct OrbitalCrossingCore{C<:OrbitalSpinLadderChannels}
    channels::C
    core_fermionic::Vector{Int}
    core_bosonic::Vector{Int}
    support_fermionic_locations::Vector{Int}
    support_bosonic_locations::Vector{Int}
    diagnostics::Dict{Symbol,Any}
end

function OrbitalCrossingCore(channels::OrbitalSpinLadderChannels, core_fermionic, core_bosonic;
                             require_complete::Bool=true, max_lookup_bytes::Integer=268435456)
    indices_f = Int.(collect(core_fermionic))
    indices_b = Int.(collect(core_bosonic))
    isempty(indices_f) && throw(ArgumentError("the external fermionic core must not be empty"))
    isempty(indices_b) && throw(ArgumentError("the bosonic core must not be empty"))
    length(unique(indices_f)) == length(indices_f) && length(unique(indices_b)) == length(indices_b) ||
        throw(ArgumentError("core frequency indices must be unique"))
    max_lookup_bytes > 0 || throw(ArgumentError("max_lookup_bytes must be positive"))
    f = [get(channels.fermion_lookup, n, 0) for n in indices_f]
    b = [get(channels.boson_lookup, m, 0) for m in indices_b]
    all(x -> !iszero(x), f) && all(x -> !iszero(x), b) || throw(ArgumentError("requested core frequencies must be present in the support axes"))
    lookup_bytes = big(sizeof(Int)) * (length(f) + length(b))
    lookup_bytes <= max_lookup_bytes || throw(ArgumentError("core crossing lookup exceeds max_lookup_bytes"))
    missing = 0
    examples = Tuple{Int,Int,Int}[]
    for m in indices_b, np in indices_f, n in indices_f
        if !haskey(channels.fermion_lookup, np - m) || !haskey(channels.boson_lookup, np - n)
            missing += 1
            length(examples) < 8 && push!(examples, (n, np, m))
        end
    end
    require_complete && !iszero(missing) && throw(ArgumentError("crossing support is incomplete for $missing core frequency triples; examples=$examples"))
    diagnostics = Dict{Symbol,Any}(:crossing_support_complete => iszero(missing),
        :missing_frequency_triples => missing, :checked_frequency_triples => big(length(f))^2 * length(b),
        :missing_examples => examples, :core_fermion_count => length(f), :core_boson_count => length(b),
        :support_fermion_count => length(channels.local_density.fermionic_axis),
        :support_boson_count => length(channels.local_density.bosonic_axis),
        :high_frequency_tail => :not_supplied, :full_multiorbital_dga => false,
        :estimated_lookup_bytes => Int(lookup_bytes))
    return OrbitalCrossingCore(channels, indices_f, indices_b, f, b, diagnostics)
end

"""Evaluate a core-grid crossed density/magnetic vertex using exact indices in the larger support box."""
function orbital_core_crossed_vertex(core::OrbitalCrossingCore, iq::Int, ik::Int, ikprime::Int,
                                     iν::Int, iνprime::Int, iΩ::Int, a::Int, b::Int, c::Int, d::Int)
    1 <= iν <= length(core.core_fermionic) || throw(BoundsError(core.core_fermionic, iν))
    1 <= iνprime <= length(core.core_fermionic) || throw(BoundsError(core.core_fermionic, iνprime))
    1 <= iΩ <= length(core.core_bosonic) || throw(BoundsError(core.core_bosonic, iΩ))
    return orbital_crossed_vertex(core.channels, iq, ik, ikprime, core.support_fermionic_locations[iν],
        core.support_fermionic_locations[iνprime], core.support_bosonic_locations[iΩ], a, b, c, d)
end

# ============================================================================
# OrbitalSpinProjection.jl: preserved implementation from src/embedding/OrbitalSpinProjection.jl
# ============================================================================


# SU(2) density/magnetic projections from a complete spin-orbital impurity four-point tensor.
# No assumption about the symmetry of independently supplied vertices is made here.

"""Return density and longitudinal magnetic spatial-orbital channels from spin-interleaved spin-orbital data.

For spatial orbitals `a,b,c,d`, the normalization is `χd = χ↑↑ + χ↑↓` and
`χm = χ↑↑ - χ↑↓`, where the arrows identify the spin of the two particle-hole
bilinears. The function independently checks `χ↓↓ = χ↑↑` and `χ↓↑ = χ↑↓`
within `atol + rtol * scale`. The transverse SU(2) Ward identity is not inferred
from these longitudinal tests; validate the Hamiltonian separately with
`orbital_su2_symmetry`.
"""
function orbital_impurity_spin_channels(object::OrbitalPairSusceptibility;
                                        spatial_labels=nothing, atol::Real=1e-9, rtol::Real=1e-7,
                                        max_output_bytes::Integer=536870912)
    nspin = length(object.labels)
    iseven(nspin) || throw(ArgumentError("spin-interleaved projection requires an even number of spin-orbitals"))
    norb = nspin ÷ 2
    selected = [(a, b) for a in 1:nspin for b in 1:nspin]
    object.pairs == selected || throw(ArgumentError("SU(2) projection requires the complete canonical spin-orbital pair space"))
    all(x -> isfinite(x) && x >= 0, (atol, rtol)) || throw(ArgumentError("tolerances must be finite and nonnegative"))
    labels = isnothing(spatial_labels) ? [Symbol("orb", i) for i in 1:norb] : Symbol.(collect(spatial_labels))
    length(labels) == norb && length(unique(labels)) == norb || throw(ArgumentError("spatial labels must be unique and match the orbital count"))
    nf, nb = length(object.fermionic_axis), length(object.bosonic_axis)
    np = norb^2
    retained = big(16) * 2 * big(np)^2 * nf^2 * nb
    max_output_bytes > 0 && retained <= max_output_bytes || throw(ArgumentError("spin channel outputs exceed max_output_bytes"))
    density = Array{ComplexF64}(undef, np, np, nf, nf, nb)
    magnetic = similar(density)
    maximum_residual = 0.0
    maximum_scale = 0.0
    v = object.values
    @inbounds for d in 1:norb, c in 1:norb, b in 1:norb, a in 1:norb
        i = (a - 1) * norb + b
        j = (c - 1) * norb + d
        up_left = (2a - 2) * nspin + 2b - 1
        down_left = (2a - 1) * nspin + 2b
        up_right = (2c - 2) * nspin + 2d - 1
        down_right = (2c - 1) * nspin + 2d
        for m in 1:nb, npf in 1:nf, n in 1:nf
            uu = v[up_left, up_right, n, npf, m]
            ud = v[up_left, down_right, n, npf, m]
            dd = v[down_left, down_right, n, npf, m]
            du = v[down_left, up_right, n, npf, m]
            maximum_residual = max(maximum_residual, abs(uu - dd), abs(ud - du))
            maximum_scale = max(maximum_scale, abs(uu), abs(ud), abs(dd), abs(du))
            density[i, j, n, npf, m] = uu + ud
            magnetic[i, j, n, npf, m] = uu - ud
        end
    end
    bound = atol + rtol * maximum_scale
    maximum_residual <= bound || throw(ArgumentError("spin-orbital four-point functions violate the longitudinal SU(2) equalities: residual=$maximum_residual, bound=$bound"))
    spatial_pairs = [(a, b) for a in 1:norb for b in 1:norb]
    metadata = Dict{Symbol,Any}(:construction => :spin_interleaved_impurity_projection,
        :spin_channel_normalization => :upup_plusminus_updown, :longitudinal_residual => maximum_residual,
        :longitudinal_tolerance => bound, :full_su2_ward_identity_verified => false,
        :frequency_box_complete => get(object.metadata, :frequency_box_complete, false),
        :estimated_output_bytes => Int(retained))
    charge = OrbitalPairSusceptibility(object.fermionic_axis, object.bosonic_axis, spatial_pairs, labels, density;
        metadata=copy(metadata), copy_values=false)
    spin = OrbitalPairSusceptibility(object.fermionic_axis, object.bosonic_axis, spatial_pairs, labels, magnetic;
        metadata=copy(metadata), copy_values=false)
    return (density=charge, magnetic=spin, diagnostics=metadata)
end

"""Project both the measured connected susceptibility and Wick bubble; optionally extract local SU(2) channel vertices.

`compute_vertices=false` avoids finite-box inversion. For vertex extraction, pass the
numerically appropriate regularization/inversion options through `vertex_kwargs`.
"""
function orbital_impurity_spin_channels(result::MultiOrbitalImpurityTwoParticleResult;
                                        compute_vertices::Bool=false, vertex_kwargs::NamedTuple=(;), kwargs...)
    correlated = orbital_impurity_spin_channels(result.susceptibility; kwargs...)
    bubble = orbital_impurity_spin_channels(result.bubble; kwargs...)
    if !compute_vertices
        return (susceptibility=correlated, bubble=bubble)
    end
    vertex_d = multiorbital_irreducible_vertex(bubble.density, correlated.density; vertex_kwargs...)
    vertex_m = multiorbital_irreducible_vertex(bubble.magnetic, correlated.magnetic; vertex_kwargs...)
    return (susceptibility=correlated, bubble=bubble, density_vertex=vertex_d, magnetic_vertex=vertex_m)
end
