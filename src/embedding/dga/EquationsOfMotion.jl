# ============================================================================
# OrbitalImpurityEquationOfMotion.jl: preserved implementation from src/embedding/OrbitalImpurityEquationOfMotion.jl
# ============================================================================


# Independent finite-temperature impurity equation-of-motion reference for multiorbital ED.
# This is an exact *local* benchmark, not an approximation to a lattice ladder self-energy.

"""Return the exact finite-bath interaction-current correlator and impurity self-energy from the equation of motion.

For `H_int = (1/4) Σ U[a,b,c,d] d†[a]d†[b]d[d]d[c]`, the fermionic
interaction current is `J[a]=[d[a],H_int]=(1/2)Σ U[a,b,c,d]d†[b]d[d]d[c]`.
The Lehmann correlator `R[a,b](z)=⟨⟨J[a];d†[b]⟩⟩_z` satisfies `R=ΣG`, so
`Σ=R/G` (right matrix division). All baths and interactions are treated exactly
inside the finite ED Hilbert space. Use this as an independent local numerical
reference for Schwinger–Dyson implementations, not as a nonlocal DGA solver.
"""
function orbital_impurity_equation_of_motion(state::MultiOrbitalEDThermalState,
                                              axis::FermionicMatsubaraAxis; max_workspace_bytes::Integer=536870912)
    isapprox(state.beta, axis.beta; atol=1e-12 * max(state.beta, 1.0), rtol=1e-12) ||
        throw(ArgumentError("equation-of-motion and thermal-state inverse temperatures differ"))
    n = state.hamiltonian.impurity_dimension
    sectors = state.sectors
    nf = length(axis)
    dimension = state.metadata[:full_hilbert_dimension]
    budget = big(16) * (big(3n) * dimension^2 + 4 * big(n)^2 * nf)
    max_workspace_bytes > 0 && budget <= max_workspace_bytes || throw(ArgumentError("equation-of-motion workspace exceeds max_workspace_bytes"))
    residues = zeros(ComplexF64, n, n, nf)
    tensor = state.hamiltonian.interaction.tensor
    for source_index in eachindex(sectors)
        target_index = _multi_ed_target(state, source_index)
        target_index == 0 && continue
        source = sectors[source_index]
        target = sectors[target_index]
        ns, nt = length(source.masks), length(target.masks)
        transitions = [_multi_ed_transition(source, target, a) for a in 1:n]
        currents = Vector{Matrix{ComplexF64}}(undef, n)
        for a in 1:n
            current = zeros(ComplexF64, nt, ns)
            for (column, mask) in enumerate(source.masks), b in 1:n, c in 1:n-1, d in c+1:n
                coefficient = tensor[a, b, c, d]
                iszero(coefficient) && continue
                s1, sign1, ok = _impurity_annihilate(mask, c)
                ok || continue
                s2, sign2, ok = _impurity_annihilate(s1, d)
                ok || continue
                s3, sign3, ok = _impurity_create(s2, b)
                ok || continue
                row = get(target.index, s3, 0)
                iszero(row) && throw(ErrorException("interaction-current transition left the target sector"))
                current[row, column] += coefficient * sign1 * sign2 * sign3
            end
            currents[a] = target.vectors' * current * source.vectors
        end
        for s in eachindex(source.energies), t in eachindex(target.energies)
            weight = source.probabilities[s] + target.probabilities[t]
            iszero(weight) && continue
            shift = target.energies[t] - source.energies[s]
            for b in 1:n, a in 1:n
                amplitude = currents[a][t, s] * conj(transitions[b][t, s])
                iszero(amplitude) && continue
                for iν in 1:nf
                    residues[a, b, iν] += weight * amplitude / (axis[iν] + shift)
                end
            end
        end
    end
    g = multiorbital_impurity_green(state, axis)
    sigma = similar(residues)
    maximum_residual = 0.0
    for iν in 1:nf
        current = @view residues[:, :, iν]
        propagator = @view g.values[:, :, iν]
        quotient = current / propagator
        sigma[:, :, iν] .= quotient
        residual = norm(quotient * propagator - current) / max(norm(current), norm(propagator), eps(Float64))
        maximum_residual = max(maximum_residual, residual)
    end
    labels = copy(state.hamiltonian.labels[1:n])
    metadata = Dict{Symbol,Any}(:construction => :exact_finite_bath_equation_of_motion,
        :interaction_current => :commutator_with_quartic_hamiltonian,
        :identity => :interaction_current_equals_self_energy_times_green,
        :maximum_eom_residual => maximum_residual, :estimated_workspace_bytes => Int(budget),
        :frequency_box_complete => true, :nonlocal_dga_reference => false)
    return (self_energy=SelfEnergy(axis, sigma; labels=labels), composite_green=residues,
        green=g, diagnostics=metadata)
end

"""Compare the ED equation-of-motion self-energy with the independent impurity Dyson result."""
function orbital_impurity_equation_of_motion(result::MultiOrbitalEDResult; max_workspace_bytes::Integer=536870912)
    output = orbital_impurity_equation_of_motion(result.thermal_state, result.green_function.axis;
        max_workspace_bytes=max_workspace_bytes)
    deviation = norm(output.self_energy.values - result.self_energy.values) /
        max(norm(result.self_energy.values), norm(output.self_energy.values), eps(Float64))
    output.diagnostics[:dyson_reference_residual] = deviation
    return output
end

# ============================================================================
# OrbitalDiagrammaticEOM.jl: preserved implementation from src/embedding/OrbitalDiagrammaticEOM.jl
# ============================================================================


# Finite-frequency, spin-orbital equation-of-motion contraction with an explicit
# local diagram counterterm. Vertex crossing and asymptotic closure are not inferred.

"""Input for a finite-box multiorbital equation-of-motion (EOM) reconstruction.

`q_vertices[iq]` and `local_vertex` are full spin-orbital particle-hole vertices in
`OrbitalPairSusceptibility` storage. The first pair is `(a,b)`, and the second
is `(c,d)`; indices on the full vertex are `(ν′,ν,Ω)`. Each vertex must have the
complete canonical spin-orbital pair space. This type does not certify that a
caller-supplied vertex obeys crossing symmetry or has a completed frequency tail.
"""
struct OrbitalDiagrammaticEOMProblem{D,G,V,B,M}
    dmft::D
    green::G
    q_vertices::Vector{V}
    local_vertex::V
    bosonic_axis::B
    qmaps::M
    qweights::Vector{Float64}
end

function OrbitalDiagrammaticEOMProblem(dmft::MultiOrbitalDMFTResult, green::MatrixMomentumResolvedGreenFunction,
                                        q_vertices::AbstractVector{V}, local_vertex::V, baxis::BosonicMatsubaraAxis;
                                        qmaps=dmft.problem.lattice.qmaps, qweights=nothing) where {V<:OrbitalPairSusceptibility}
    dmft.converged || throw(ArgumentError("diagrammatic EOM requires a converged DMFT reference"))
    isnothing(qmaps) && throw(ArgumentError("complete momentum-transfer maps are required"))
    lattice = dmft.problem.lattice
    green.labels == lattice.labels == dmft.problem.interaction.labels == local_vertex.labels ||
        throw(ArgumentError("the lattice, interaction, propagator and local vertex labels must agree"))
    green.kpoints == lattice.kpoints || throw(ArgumentError("momentum order is inconsistent with DMFT"))
    isapprox(green.weights, lattice.weights; atol=1e-12, rtol=1e-12) || throw(ArgumentError("momentum weights differ"))
    pairs = [(a, b) for a in eachindex(lattice.labels) for b in eachindex(lattice.labels)]
    local_vertex.pairs == pairs || throw(ArgumentError("a complete canonical spin-orbital pair space is required"))
    local_vertex.fermionic_axis.indices == dmft.problem.axis.indices &&
        local_vertex.fermionic_axis.beta == dmft.problem.axis.beta || throw(ArgumentError("vertex and DMFT fermionic axes differ"))
    local_vertex.bosonic_axis.indices == baxis.indices && local_vertex.bosonic_axis.beta == baxis.beta ||
        throw(ArgumentError("vertex and bosonic axes differ"))
    maps, _ = _orbital_ladder_maps(green, qmaps, dmft.problem.axis, baxis)
    length(q_vertices) == length(maps) || throw(DimensionMismatch("q-resolved vertex count does not match transfer maps"))
    for vertex in q_vertices
        _orbital_spin_same_grid(local_vertex, vertex)
    end
    # On a general weighted k mesh, replacing a diagram by the local propagator
    # does not necessarily commute with the q convolution. Require a group grid.
    nk = length(lattice.kpoints)
    length(maps) == nk && all(w -> isapprox(w, inv(nk); atol=1e-12, rtol=1e-12), lattice.weights) ||
        throw(ArgumentError("diagrammatic local subtraction requires a complete uniform momentum group"))
    all(==(1), [count(mapping -> mapping[ik] == jk, maps) for jk in 1:nk, ik in 1:nk]) ||
        throw(ArgumentError("momentum transfers do not cover every k-to-kprime pair"))
    weights = isnothing(qweights) ? fill(inv(nk), nk) : Float64.(collect(qweights))
    length(weights) == nk && all(w -> isfinite(w) && isapprox(w, inv(nk); atol=1e-12, rtol=1e-12), weights) ||
        throw(ArgumentError("diagrammatic convolution requires the uniform complete q quadrature"))
    return OrbitalDiagrammaticEOMProblem(dmft, green, collect(q_vertices), local_vertex, baxis, maps, weights)
end

"""Finite frequency-box EOM contraction using a full spin-orbital particle-hole vertex.

In the package's convention `χ⁰[(n,l),(l′,n′)]`, the dynamic diagram is
`-(2β)⁻¹ ∑(q,Ω,ν′,l,l′,n,n′,h,h′) wq U[m,l,h,n] χ⁰q[(n,l),(l′,n′);ν′,Ω] Fq[(n′,l′),(h′,m′);ν′,ν,Ω] G[k-q][h,h′;ν+Ω]`.
The counterterm is the *same diagram* evaluated with the supplied local full
vertex, local Wick bubble and weighted local Green function; no momentum-average
projection of the nonlocal answer is used. The returned answer is the converged
DMFT self-energy plus the difference of these two finite-box diagrams.

An independently certified crossing-consistent full spin-orbital vertex and
converged frequency cutoffs are required before interpreting the answer as DΓA.
"""
function orbital_diagrammatic_self_energy(problem::OrbitalDiagrammaticEOMProblem;
                                          max_workspace_bytes::Integer=536870912, max_output_bytes::Integer=536870912)
    dmft, green = problem.dmft, problem.green
    lattice = dmft.problem.lattice
    n, nk, nf, nb = length(lattice.labels), length(lattice.kpoints), length(dmft.problem.axis), length(problem.bosonic_axis)
    np, nq = n^2, length(problem.qmaps)
    output_bytes = big(16) * big(n)^2 * nk * nf
    work_bytes = big(16) * (7 * big(np)^2 + big(n)^2 * (length(green.axis) + nf + nk * nf))
    max_output_bytes > 0 && output_bytes <= max_output_bytes || throw(ArgumentError("EOM output exceeds max_output_bytes"))
    max_workspace_bytes > 0 && work_bytes <= max_workspace_bytes || throw(ArgumentError("EOM workspace exceeds max_workspace_bytes"))
    maps, shifts = _orbital_ladder_maps(green, problem.qmaps, dmft.problem.axis, problem.bosonic_axis)
    inverse_maps = [invperm(mapping) for mapping in maps]
    U = dmft.problem.interaction.tensor
    A = zeros(ComplexF64, np, np)
    for m in 1:n, h in 1:n, l in 1:n, u in 1:n
        A[(m-1)*n+h, (u-1)*n+l] = U[m, l, h, u]
    end
    local_green = zeros(ComplexF64, n, n, length(green.axis))
    for ik in 1:nk
        local_green .+= lattice.weights[ik] .* view(green.values, :, :, ik, :)
    end
    B = Matrix{ComplexF64}(undef, np, np)
    AB = similar(B)
    F = similar(B)
    K = similar(B)
    local_contribution = zeros(ComplexF64, n, n, nf)
    nonlocal_contribution = zeros(ComplexF64, n, n, nk, nf)
    local_vertex = problem.local_vertex.values
    beta = dmft.problem.axis.beta
    # The local counterterm is evaluated from Gloc and Floc, not the mean of Σ(k).
    for iΩ in 1:nb, iνprime in 1:nf
        present, shifted_prime = shifts[iΩ][iνprime]
        for lprime in 1:n, u_prime in 1:n, l in 1:n, u in 1:n
            row, col = (u-1)*n+l, (lprime-1)*n+u_prime
            B[row, col] = -beta * local_green[u_prime, u, present] * local_green[l, lprime, shifted_prime]
        end
        mul!(AB, A, B)
        for iν in 1:nf
            shifted = shifts[iΩ][iν][2]
            for hprime in 1:n, mprime in 1:n, lprime in 1:n, u_prime in 1:n
                F[(lprime-1)*n+u_prime, (hprime-1)*n+mprime] =
                    local_vertex[(u_prime-1)*n+lprime, (hprime-1)*n+mprime, iνprime, iν, iΩ]
            end
            mul!(K, AB, F)
            for mprime in 1:n, m in 1:n, hprime in 1:n, h in 1:n
                local_contribution[m, mprime, iν] -= K[(m-1)*n+h, (hprime-1)*n+mprime] * local_green[h, hprime, shifted] / (2beta)
            end
        end
    end
    # Stream q and Ω, holding only small orbital matrices and the requested Σ(k,ν).
    for iq in 1:nq, iΩ in 1:nb, iνprime in 1:nf
        present, shifted_prime = shifts[iΩ][iνprime]
        qmap = maps[iq]
        for lprime in 1:n, u_prime in 1:n, l in 1:n, u in 1:n
            row, col = (u-1)*n+l, (lprime-1)*n+u_prime
            total = zero(ComplexF64)
            for ik in 1:nk
                total += green.weights[ik] * green.values[u_prime, u, ik, present] *
                    green.values[l, lprime, qmap[ik], shifted_prime]
            end
            B[row, col] = -beta * total
        end
        mul!(AB, A, B)
        v = problem.q_vertices[iq].values
        for iν in 1:nf
            shifted = shifts[iΩ][iν][2]
            for hprime in 1:n, mprime in 1:n, lprime in 1:n, u_prime in 1:n
                F[(lprime-1)*n+u_prime, (hprime-1)*n+mprime] =
                    v[(u_prime-1)*n+lprime, (hprime-1)*n+mprime, iνprime, iν, iΩ]
            end
            mul!(K, AB, F)
            scale = -problem.qweights[iq] / (2beta)
            for ik in 1:nk, mprime in 1:n, m in 1:n, hprime in 1:n, h in 1:n
                nonlocal_contribution[m, mprime, ik, iν] += scale * K[(m-1)*n+h, (hprime-1)*n+mprime] *
                    green.values[h, hprime, inverse_maps[iq][ik], shifted]
            end
        end
    end
    values = similar(nonlocal_contribution)
    for iν in 1:nf, ik in 1:nk, b in 1:n, a in 1:n
        values[a, b, ik, iν] = dmft.self_energy.values[a, b, iν] +
            nonlocal_contribution[a, b, ik, iν] - local_contribution[a, b, iν]
    end
    local_difference = zeros(ComplexF64, n, n, nf)
    for ik in 1:nk
        local_difference .+= lattice.weights[ik] .* view(nonlocal_contribution, :, :, ik, :)
    end
    local_difference .-= local_contribution
    metadata = Dict{Symbol,Any}(:construction => :spin_orbital_diagrammatic_eom,
        :vertex_convention => :orbital_pair_particle_hole, :local_double_counting_scheme => :matched_local_eom_diagram,
        :frequency_box_complete => true, :high_frequency_completion => :none,
        :crossing_complete_certified_by_package => false, :full_multiorbital_dga => false,
        :local_diagram_difference_norm => norm(local_difference), :local_diagram_norm => norm(local_contribution),
        :estimated_workspace_bytes => Int(work_bytes), :estimated_output_bytes => Int(output_bytes),
        :self_energy_feedback => false)
    return OrbitalMomentumSelfEnergy(dmft.problem.axis, lattice.kpoints, lattice.labels, values; metadata=metadata)
end

"""Compare two frequency-box self-energies on the smaller box without extrapolating missing tails.

Agreement of successive boxes is a *numerical convergence diagnostic*, not a
proof of the analytically correct asymptotic vertex completion.
"""
function orbital_eom_frequency_box_comparison(smaller::OrbitalMomentumSelfEnergy, larger::OrbitalMomentumSelfEnergy;
                                               atol::Real=1e-10, rtol::Real=1e-6)
    smaller.labels == larger.labels && smaller.kpoints == larger.kpoints ||
        throw(ArgumentError("frequency-box comparison requires identical orbital and momentum orderings"))
    smaller.axis.beta == larger.axis.beta || throw(ArgumentError("inverse temperatures differ"))
    all(x -> isfinite(x) && x >= 0, (atol, rtol)) || throw(ArgumentError("comparison tolerances must be finite and nonnegative"))
    lookup = Dict(index => i for (i, index) in enumerate(larger.axis.indices))
    locations = [get(lookup, index, 0) for index in smaller.axis.indices]
    all(>(0), locations) || throw(ArgumentError("the larger frequency box does not contain the smaller frequency box"))
    length(larger.axis) > length(smaller.axis) || throw(ArgumentError("the larger box must add at least one frequency"))
    reference = @view larger.values[:, :, :, locations]
    difference = smaller.values .- reference
    relative = norm(difference) / max(norm(smaller.values), norm(reference), eps(Float64))
    maximum_difference = maximum(abs, difference)
    bound = atol + rtol * max(maximum(abs, smaller.values), maximum(abs, reference))
    return (converged=maximum_difference <= bound, relative_residual=relative, maximum_absolute_difference=maximum_difference,
        compared_frequency_count=length(locations), larger_frequency_count=length(larger.axis),
        high_frequency_asymptotics_certified=false, tolerance=bound)
end

# ============================================================================
# OrbitalChannelDiagrammaticEOM.jl: preserved implementation from src/embedding/OrbitalChannelDiagrammaticEOM.jl
# ============================================================================


# SU(2) channel-combined finite-frequency equation of motion. This consumes
# all three momenta (k, kprime, q), rather than reducing a crossed vertex to F(q).

"""Lift an SU(2) spatial-orbital density/magnetic vertex to one spin-orbital tensor element.

The spin tensor is `(Fd δαβδγδ + Fm σαβ⋅σγδ)/2`, where
`σαβ⋅σγδ = 2 δαδδγβ - δαβδγδ`. The spin-orbital ordering is interleaved.
"""
function _orbital_su2_spin_element(fd::ComplexF64, fm::ComplexF64, a::Int, b::Int, c::Int, d::Int)
    sa, sb, sc, sd = isodd(a), isodd(b), isodd(c), isodd(d)
    value = zero(ComplexF64)
    sa == sb && sc == sd && (value += (fd - fm) / 2)
    sa == sd && sc == sb && (value += fm)
    return value
end

"""Return one spin-orbital full vertex element with all crossed momentum indices retained."""
function orbital_spin_orbital_crossed_vertex(core::OrbitalCrossingCore, iq::Int, ik::Int, ikprime::Int,
                                              iν::Int, iνprime::Int, iΩ::Int, a::Int, b::Int, c::Int, d::Int)
    norb = length(core.channels.local_density.labels)
    all(mode -> 1 <= mode <= 2norb, (a, b, c, d)) || throw(ArgumentError("spin-orbital index outside the SU(2) spatial basis"))
    # The stored particle-hole full vertex uses first and second fermionic
    # frequencies in the same order as orbital_core_crossed_vertex.
    channels = orbital_core_crossed_vertex(core, iq, ik, ikprime, iν, iνprime, iΩ,
        cld(a, 2), cld(b, 2), cld(c, 2), cld(d, 2))
    return _orbital_su2_spin_element(channels.density, channels.magnetic, a, b, c, d)
end

"""Channel-combined, finite-box SU(2) ladder self-energy on a complete translation grid.

The crossed full vertex is evaluated as `F(q,k,k′;ν′,ν,Ω)` rather than `F(q)`.
The local diagram uses the independently supplied local SU(2) full vertex and
local propagators. The returned result does not imply high-frequency completion.
Large calculations require explicit `max_contractions` authorization because
this reference contraction has high polynomial runtime complexity.
"""
function orbital_channel_ladder_self_energy(dmft::MultiOrbitalDMFTResult, green::MatrixMomentumResolvedGreenFunction,
                                             core::OrbitalCrossingCore; qweights=nothing,
                                             max_workspace_bytes::Integer=536870912,
                                             max_output_bytes::Integer=536870912,
                                             max_contractions::Integer=100_000_000,
                                             symmetry_tolerance::Real=1e-10)
    core.diagnostics[:crossing_support_complete] || throw(ArgumentError("incomplete crossed-frequency support cannot enter the diagrammatic self-energy"))
    dmft.converged || throw(ArgumentError("a converged DMFT reference is required"))
    lattice = dmft.problem.lattice
    n, nk, nf, nb = length(lattice.labels), length(lattice.kpoints), length(dmft.problem.axis), length(core.core_bosonic)
    iseven(n) && n == 2length(core.channels.local_density.labels) ||
        throw(ArgumentError("SU(2) channels must contain half as many spatial orbitals as spin-orbitals in the lattice"))
    core.core_fermionic == dmft.problem.axis.indices || throw(ArgumentError("the physical frequency core must equal the DMFT external frequency box"))
    core.channels.qmaps == lattice.qmaps || throw(ArgumentError("crossing momentum maps differ from the DMFT lattice"))
    green.labels == lattice.labels && green.kpoints == lattice.kpoints || throw(ArgumentError("momentum Green and lattice labels differ"))
    isapprox(green.weights, lattice.weights; atol=1e-12, rtol=1e-12) || throw(ArgumentError("momentum Green weights differ"))
    baxis = BosonicMatsubaraAxis(dmft.problem.axis.beta, core.core_bosonic)
    maps, shifts = _orbital_ladder_maps(green, core.channels.qmaps, dmft.problem.axis, baxis)
    nq = length(maps)
    nk == nq && all(w -> isapprox(w, inv(nk); atol=1e-12), lattice.weights) ||
        throw(ArgumentError("channel EOM requires complete uniform momentum translation quadrature"))
    all(==(1), [count(mapping -> mapping[ik] == jk, maps) for jk in 1:nk, ik in 1:nk]) ||
        throw(ArgumentError("momentum transfers are not a complete translation group"))
    weights = isnothing(qweights) ? fill(inv(nq), nq) : Float64.(collect(qweights))
    length(weights) == nq && all(w -> isfinite(w) && isapprox(w, inv(nq); atol=1e-12), weights) ||
        throw(ArgumentError("q quadrature must cover a complete uniform translation group"))
    all(x -> isfinite(x) && x >= 0, (symmetry_tolerance,)) || throw(ArgumentError("invalid SU(2) tolerance"))
    local_symmetry = orbital_su2_symmetry(dmft.problem.one_body, dmft.problem.interaction; tolerance=symmetry_tolerance)
    local_symmetry.valid || throw(ArgumentError("impurity Hamiltonian is not SU(2)-invariant"))
    for ik in 1:nk
        orbital_su2_symmetry(view(lattice.hamiltonians, :, :, ik), dmft.problem.interaction;
            tolerance=symmetry_tolerance).valid || throw(ArgumentError("lattice Bloch Hamiltonian violates SU(2) spin symmetry"))
    end
    np = n^2
    work_bytes = big(16) * (7 * big(np)^2 + 2 * big(div(n, 2))^4 + big(n)^2 * (length(green.axis) + nf + nk * nf))
    output_bytes = big(16) * big(n)^2 * nk * nf
    max_workspace_bytes > 0 && work_bytes <= max_workspace_bytes || throw(ArgumentError("channel EOM workspace exceeds max_workspace_bytes"))
    max_output_bytes > 0 && output_bytes <= max_output_bytes || throw(ArgumentError("channel EOM output exceeds max_output_bytes"))
    estimated_contractions = big(n)^8 * nk^2 * nq * nf^2 * nb
    max_contractions > 0 && estimated_contractions <= max_contractions ||
        throw(ArgumentError("channel EOM contraction count $estimated_contractions exceeds max_contractions=$max_contractions"))
    U = dmft.problem.interaction.tensor
    A = zeros(ComplexF64, np, np)
    for m in 1:n, h in 1:n, l in 1:n, u in 1:n
        A[(m-1)*n+h, (u-1)*n+l] = U[m, l, h, u]
    end
    local_green = zeros(ComplexF64, n, n, length(green.axis))
    for ik in 1:nk
        local_green .+= lattice.weights[ik] .* view(green.values, :, :, ik, :)
    end
    B = Matrix{ComplexF64}(undef, np, np)
    AB = similar(B)
    F = similar(B)
    K = similar(B)
    spatial_density = Array{ComplexF64}(undef, div(n, 2), div(n, 2), div(n, 2), div(n, 2))
    spatial_magnetic = similar(spatial_density)
    counterterm = zeros(ComplexF64, n, n, nf)
    correction = zeros(ComplexF64, n, n, nk, nf)
    inverse_maps = [invperm(mapping) for mapping in maps]
    beta = dmft.problem.axis.beta
    # Matched local diagram: identical tensor contraction, local bubble and Gloc.
    for iΩ in 1:nb, iνprime in 1:nf
        present, shifted_prime = shifts[iΩ][iνprime]
        for lp in 1:n, up in 1:n, l in 1:n, u in 1:n
            B[(u-1)*n+l, (lp-1)*n+up] = -beta * local_green[up, u, present] * local_green[l, lp, shifted_prime]
        end
        mul!(AB, A, B)
        for iν in 1:nf
            shifted = shifts[iΩ][iν][2]
            for hp in 1:n, mp in 1:n, lp in 1:n, up in 1:n
                p = (cld(up, 2)-1)*div(n, 2) + cld(lp, 2)
                r = (cld(hp, 2)-1)*div(n, 2) + cld(mp, 2)
                fd = core.channels.local_density.values[p, r, core.support_fermionic_locations[iνprime],
                    core.support_fermionic_locations[iν], core.support_bosonic_locations[iΩ]]
                fm = core.channels.local_magnetic.values[p, r, core.support_fermionic_locations[iνprime],
                    core.support_fermionic_locations[iν], core.support_bosonic_locations[iΩ]]
                F[(lp-1)*n+up, (hp-1)*n+mp] = _orbital_su2_spin_element(fd, fm, up, lp, hp, mp)
            end
            mul!(K, AB, F)
            for mp in 1:n, m in 1:n, hp in 1:n, h in 1:n
                counterterm[m, mp, iν] -= K[(m-1)*n+h, (hp-1)*n+mp] * local_green[h, hp, shifted] / (2beta)
            end
        end
    end
    # Momentum-resolved crossed vertex: k and kprime remain independent.
    for iq in 1:nq, iΩ in 1:nb, iνprime in 1:nf, ikprime in 1:nk
        present, shifted_prime = shifts[iΩ][iνprime]
        map = maps[iq]
        for lp in 1:n, up in 1:n, l in 1:n, u in 1:n
            B[(u-1)*n+l, (lp-1)*n+up] = -beta * lattice.weights[ikprime] *
                green.values[up, u, ikprime, present] * green.values[l, lp, map[ikprime], shifted_prime]
        end
        mul!(AB, A, B)
        for iν in 1:nf, ik in 1:nk
            shifted = shifts[iΩ][iν][2]
            for d in 1:div(n, 2), c in 1:div(n, 2), b in 1:div(n, 2), a in 1:div(n, 2)
                crossed = orbital_core_crossed_vertex(core, iq, ik, ikprime, iνprime, iν, iΩ, a, b, c, d)
                spatial_density[a, b, c, d] = crossed.density
                spatial_magnetic[a, b, c, d] = crossed.magnetic
            end
            for hp in 1:n, mp in 1:n, lp in 1:n, up in 1:n
                fd = spatial_density[cld(up, 2), cld(lp, 2), cld(hp, 2), cld(mp, 2)]
                fm = spatial_magnetic[cld(up, 2), cld(lp, 2), cld(hp, 2), cld(mp, 2)]
                F[(lp-1)*n+up, (hp-1)*n+mp] = _orbital_su2_spin_element(fd, fm, up, lp, hp, mp)
            end
            mul!(K, AB, F)
            scale = -weights[iq] / (2beta)
            for mp in 1:n, m in 1:n, hp in 1:n, h in 1:n
                correction[m, mp, ik, iν] += scale * K[(m-1)*n+h, (hp-1)*n+mp] *
                    green.values[h, hp, inverse_maps[iq][ik], shifted]
            end
        end
    end
    values = similar(correction)
    for iν in 1:nf, ik in 1:nk, b in 1:n, a in 1:n
        values[a, b, ik, iν] = dmft.self_energy.values[a, b, iν] + correction[a, b, ik, iν] - counterterm[a, b, iν]
    end
    averaged = zeros(ComplexF64, n, n, nf)
    for ik in 1:nk
        averaged .+= lattice.weights[ik] .* view(correction, :, :, ik, :)
    end
    averaged .-= counterterm
    metadata = Dict{Symbol,Any}(:construction => :su2_channel_combined_finite_box_eom,
        :vertex_convention => :su2_pauli_density_magnetic, :frequency_box_complete => true,
        :crossing_support_complete => true, :local_double_counting_scheme => :matched_local_eom_diagram,
        :high_frequency_completion => :none, :full_multiorbital_dga => false,
        :su2_hamiltonian_verified => true, :local_diagram_difference_norm => norm(averaged),
        :estimated_contractions => estimated_contractions, :estimated_workspace_bytes => Int(work_bytes),
        :estimated_output_bytes => Int(output_bytes), :self_energy_feedback => false)
    return OrbitalMomentumSelfEnergy(dmft.problem.axis, lattice.kpoints, lattice.labels, values; metadata=metadata)
end
