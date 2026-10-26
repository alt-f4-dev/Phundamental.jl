# Phase II.A.2: finite-temperature, number-conserving multiorbital Anderson impurity ED.
# The established scalar ED-DMFT code path remains unchanged.

"""
    MultiOrbitalEDSolver(; bath_sites=4, bath_fit=BathFitOptions(), kB=1.0, backend=:sectorized,
                         max_full_dimension=65536, max_sector_dimension=2048, max_eigensystem_bytes=536870912)

Exact finite-bath, finite-temperature multiorbital impurity solver. `:sectorized`
uses conserved total-particle-number sectors; `:dense` uses the complete Fock
space as an independent small-system reference. No spin conservation is assumed.
Both backends require complete eigensystems for thermal Lehmann functions.
The dimension and eigensystem limits are checked *before* allocating matrices.
"""
struct MultiOrbitalEDSolver{O<:BathFitOptions} <: AbstractImpuritySolver
    bath_sites::Int
    bath_fit::O
    kB::Float64
    backend::Symbol
    max_full_dimension::Int
    max_sector_dimension::Int
    max_eigensystem_bytes::Int
end

function MultiOrbitalEDSolver(; bath_sites::Integer=4, bath_fit::BathFitOptions=BathFitOptions(), kB::Real=1.0,
                              backend::Symbol=:sectorized, max_full_dimension::Integer=65536,
                              max_sector_dimension::Integer=2048, max_eigensystem_bytes::Integer=536870912)
    bath_sites >= 0 || throw(ArgumentError("bath_sites must be nonnegative"))
    isfinite(kB) && kB > 0 || throw(ArgumentError("kB must be finite and positive"))
    backend in (:sectorized, :dense) || throw(ArgumentError("backend must be :sectorized or :dense"))
    max_full_dimension > 0 || throw(ArgumentError("max_full_dimension must be positive"))
    max_sector_dimension > 0 || throw(ArgumentError("max_sector_dimension must be positive"))
    max_eigensystem_bytes > 0 || throw(ArgumentError("max_eigensystem_bytes must be positive"))
    return MultiOrbitalEDSolver(Int(bath_sites), bath_fit, Float64(kB), backend, Int(max_full_dimension),
                                Int(max_sector_dimension), Int(max_eigensystem_bytes))
end

solver_name(::MultiOrbitalEDSolver) = :multiorbital_ed_impurity

struct _MultiEDInteractionTerm
    a::Int
    b::Int
    c::Int
    d::Int
    coefficient::ComplexF64
end

struct _MultiEDOneBodyTerm
    a::Int
    b::Int
    coefficient::ComplexF64
end

struct _MultiEDSector
    particle_number::Int
    masks::Vector{UInt64}
    index::Dict{UInt64,Int}
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    probabilities::Vector{Float64}
end

"""Exact grand-canonical thermal state, stored as complete eigensystems in conserved sectors."""
struct MultiOrbitalEDThermalState{H} <: AbstractThermalState
    hamiltonian::H
    temperature::Float64
    kB::Float64
    beta::Float64
    sectors::Vector{_MultiEDSector}
    logZ::Float64
    metadata::Dict{Symbol,Any}
end

"""Single finite-bath multiorbital impurity solve; no DMFT self-consistency is performed."""
struct MultiOrbitalEDResult{H,G,G0,S,B,T}
    hamiltonian::H
    green_function::G
    noninteracting_green_function::G0
    self_energy::S
    density_matrix::Matrix{ComplexF64}
    occupations::Vector{Float64}
    bath::B
    thermal_state::T
    metadata::Dict{Symbol,Any}
end

function _multi_ed_dimensions(nmodes::Int, solver::MultiOrbitalEDSolver)
    nmodes > 0 && nmodes <= 62 || throw(ArgumentError("fermionic Fock-bit representation requires 1–62 modes"))
    total = big(1) << nmodes
    total <= solver.max_full_dimension || throw(ArgumentError("Fock-space dimension $total exceeds max_full_dimension=$(solver.max_full_dimension)"))
    dimensions = solver.backend === :dense ? [Int(total)] : [Int(binomial(big(nmodes), n)) for n in 0:nmodes]
    largest = maximum(dimensions)
    largest <= solver.max_sector_dimension || throw(ArgumentError("largest ED sector $largest exceeds max_sector_dimension=$(solver.max_sector_dimension)"))
    # Count dense Hamiltonian plus eigenvector storage, not just the output eigensystem.
    memory = 32 * sum(big(d)^2 for d in dimensions)
    memory <= solver.max_eigensystem_bytes || throw(ArgumentError("estimated Hamiltonian/eigenvector storage $memory bytes exceeds max_eigensystem_bytes=$(solver.max_eigensystem_bytes)"))
    return Int(total), largest, length(dimensions), Int(memory)
end

function _multi_ed_bases(nmodes::Int, total::Int, backend::Symbol)
    if backend === :dense
        masks = collect(UInt64(0):UInt64(total - 1))
        return [(-1, masks)]
    end
    groups = [UInt64[] for _ in 0:nmodes]
    for integer_mask in 0:total-1
        mask = UInt64(integer_mask)
        push!(groups[count_ones(mask) + 1], mask)
    end
    return [(n, groups[n + 1]) for n in 0:nmodes]
end

function _multi_ed_terms(specification::MultiOrbitalAndersonHamiltonian)
    nimp = specification.impurity_dimension
    nmodes = size(specification.one_body, 1)
    one_body = _MultiEDOneBodyTerm[]
    interactions = _MultiEDInteractionTerm[]
    for b in 1:nmodes, a in 1:nmodes
        h = specification.one_body[a, b]
        iszero(h) || push!(one_body, _MultiEDOneBodyTerm(a, b, h))
    end
    for c in 1:nimp-1, d in c+1:nimp, a in 1:nimp-1, b in a+1:nimp
        coefficient = specification.interaction.tensor[a, b, c, d]
        iszero(coefficient) || push!(interactions, _MultiEDInteractionTerm(a, b, c, d, coefficient))
    end
    return one_body, interactions
end

@inline function _multi_ed_quartic(mask::UInt64, term::_MultiEDInteractionTerm)
    s1, v1, ok = _impurity_annihilate(mask, term.c)
    ok || return mask, 0.0, false
    s2, v2, ok = _impurity_annihilate(s1, term.d)
    ok || return mask, 0.0, false
    s3, v3, ok = _impurity_create(s2, term.b)
    ok || return mask, 0.0, false
    s4, v4, ok = _impurity_create(s3, term.a)
    ok || return mask, 0.0, false
    return s4, v1 * v2 * v3 * v4, true
end

function _multi_ed_hamiltonian(masks::Vector{UInt64}, index::Dict{UInt64,Int},
                               one_body::Vector{_MultiEDOneBodyTerm}, interactions::Vector{_MultiEDInteractionTerm})
    H = zeros(ComplexF64, length(masks), length(masks))
    for (column, mask) in enumerate(masks)
        for term in one_body
            output, sign, ok = _impurity_hop(mask, term.b, term.a)
            ok || continue
            row = get(index, output, 0)
            row > 0 || throw(ErrorException("one-body term left the selected conserved-number sector"))
            H[row, column] += sign * term.coefficient
        end
        for term in interactions
            output, sign, ok = _multi_ed_quartic(mask, term)
            ok || continue
            row = get(index, output, 0)
            row > 0 || throw(ErrorException("interaction term left the selected conserved-number sector"))
            H[row, column] += sign * term.coefficient
        end
    end
    isapprox(H, H'; atol=1e-11, rtol=1e-11) || throw(ErrorException("multiorbital Anderson Hamiltonian is not Hermitian"))
    return H
end

"""
    multiorbital_ed_state(hamiltonian, solver, temperature)

Diagonalize a structured finite-bath multiorbital Anderson Hamiltonian and form
its finite-temperature, grand-canonical Gibbs ensemble without truncation.
"""
function multiorbital_ed_state(specification::MultiOrbitalAndersonHamiltonian, solver::MultiOrbitalEDSolver, temperature::Real)
    isfinite(temperature) && temperature > 0 || throw(ArgumentError("thermal impurity ED requires positive, finite temperature"))
    nmodes = size(specification.one_body, 1)
    total, largest, nsectors, bytes = _multi_ed_dimensions(nmodes, solver)
    one_body, interactions = _multi_ed_terms(specification)
    bases = _multi_ed_bases(nmodes, total, solver.backend)
    energies = Vector{Vector{Float64}}(undef, nsectors)
    vectors = Vector{Matrix{ComplexF64}}(undef, nsectors)
    indices = Vector{Dict{UInt64,Int}}(undef, nsectors)
    for i in eachindex(bases)
        masks = bases[i][2]
        index = Dict(mask => j for (j, mask) in enumerate(masks))
        indices[i] = index
        eigenresult = eigen(Hermitian(_multi_ed_hamiltonian(masks, index, one_body, interactions)))
        energies[i] = Float64.(eigenresult.values)
        vectors[i] = Matrix{ComplexF64}(eigenresult.vectors)
    end
    beta = inv(solver.kB * Float64(temperature))
    reference = minimum(minimum, energies)
    shifted = [exp.(-beta .* (e .- reference)) for e in energies]
    partition = sum(sum, shifted)
    isfinite(partition) && partition > 0 || throw(ErrorException("grand-canonical partition normalization is invalid"))
    sectors = Vector{_MultiEDSector}(undef, nsectors)
    for i in eachindex(bases)
        number, masks = bases[i]
        sectors[i] = _MultiEDSector(number, masks, indices[i], energies[i], vectors[i], shifted[i] ./ partition)
    end
    metadata = Dict{Symbol,Any}(:backend => solver.backend, :full_hilbert_dimension => total,
                                :max_sector_dimension => largest, :sector_count => nsectors,
                                :estimated_eigensystem_bytes => bytes, :impurity_dimension => specification.impurity_dimension,
                                :mode_count => nmodes, :exact_finite_bath => true)
    return MultiOrbitalEDThermalState(specification, Float64(temperature), solver.kB, beta, sectors,
                                     -beta * reference + log(partition), metadata)
end

function _multi_ed_transition(source::_MultiEDSector, target::_MultiEDSector, mode::Int)
    ns = length(source.masks)
    nt = length(target.masks)
    temporary = zeros(ComplexF64, nt, ns)
    for (column, mask) in enumerate(source.masks)
        output, sign, ok = _impurity_annihilate(mask, mode)
        ok || continue
        row = get(target.index, output, 0)
        row > 0 || throw(ErrorException("fermionic annihilation transition missing from target sector"))
        temporary[row, column] = sign
    end
    return target.vectors' * temporary * source.vectors
end

function _multi_ed_target(state::MultiOrbitalEDThermalState, source_index::Int)
    state.metadata[:backend] === :dense && return 1
    return state.sectors[source_index].particle_number == 0 ? 0 : source_index - 1
end

function _multi_ed_spectral_data(state::MultiOrbitalEDThermalState, axis::FermionicMatsubaraAxis)
    scale = max(abs(state.beta), abs(axis.beta), 1.0)
    isapprox(state.beta, axis.beta; rtol=1e-12, atol=1e-12 * scale) || throw(ArgumentError("thermal-state beta does not match Matsubara-axis beta"))
    nimp = state.hamiltonian.impurity_dimension
    nfreq = length(axis)
    values = zeros(ComplexF64, nimp, nimp, nfreq)
    density = zeros(ComplexF64, nimp, nimp)
    # For each N -> N-1 transition, D_a[t,s] = <t,N-1|d_a|s,N>.
    # The formula retains all spin-orbital off-diagonal spectral residues.
    for source_index in eachindex(state.sectors)
        target_index = _multi_ed_target(state, source_index)
        target_index == 0 && continue
        source = state.sectors[source_index]
        target = state.sectors[target_index]
        D = [_multi_ed_transition(source, target, a) for a in 1:nimp]
        for s in eachindex(source.energies), t in eachindex(target.energies)
            ps = source.probabilities[s]
            pt = target.probabilities[t]
            ps + pt == 0 && continue
            energy_shift = target.energies[t] - source.energies[s]
            for b in 1:nimp, a in 1:nimp
                amplitude = D[a][t, s] * conj(D[b][t, s])
                iszero(amplitude) && continue
                density[a, b] += ps * amplitude
                numerator = (ps + pt) * amplitude
                for i in 1:nfreq
                    values[a, b, i] += numerator / (axis[i] + energy_shift)
                end
            end
        end
    end
    return values, density
end

"""
    multiorbital_impurity_green(state, axis)

Exact finite-temperature *matrix* Lehmann Green function of the interacting
impurity spin-orbitals, including complex off-diagonal spectral elements.
"""
function multiorbital_impurity_green(state::MultiOrbitalEDThermalState, axis::FermionicMatsubaraAxis)
    values, _ = _multi_ed_spectral_data(state, axis)
    metadata = Dict{Symbol,Any}(:method => :multiorbital_sector_lehmann, :beta => state.beta,
                                :full_hilbert_dimension => state.metadata[:full_hilbert_dimension],
                                :sector_count => state.metadata[:sector_count], :backend => state.metadata[:backend])
    return GreenFunction(axis, values; labels=state.hamiltonian.labels[1:state.hamiltonian.impurity_dimension], metadata=metadata)
end

"""
    matsubara_green(state::MultiOrbitalEDThermalState, axis; modes=nothing, labels=nothing)

Compatible with the standard Greens API; requested `modes` must be impurity
spin-orbital indices, not bath indices.
"""
function matsubara_green(state::MultiOrbitalEDThermalState, axis::FermionicMatsubaraAxis; modes=nothing, labels=nothing)
    n = state.hamiltonian.impurity_dimension
    selected = isnothing(modes) ? collect(1:n) : Int.(collect(modes))
    !isempty(selected) && length(unique(selected)) == length(selected) && all(a -> 1 <= a <= n, selected) ||
        throw(ArgumentError("modes must be unique impurity spin-orbital indices"))
    names = isnothing(labels) ? state.hamiltonian.labels[selected] : _impurity_labels(labels, length(selected))
    full = multiorbital_impurity_green(state, axis)
    return GreenFunction(axis, full.values[selected, selected, :]; labels=names, metadata=copy(full.metadata))
end

"""
    solve(solver::MultiOrbitalEDSolver, problem::MultiOrbitalImpurityProblem; initial_bath=nothing)

Solve the interacting multiorbital impurity on its fermionic Matsubara axis.
A target hybridization is first discretized using the validated matrix fitter.
This routine performs one exact finite-bath impurity solve, **not** a DMFT loop.
"""
function solve(solver::MultiOrbitalEDSolver, problem::MultiOrbitalImpurityProblem; initial_bath=nothing)
    !isnothing(initial_bath) && !(initial_bath isa MatrixDiscreteBath) && throw(ArgumentError("initial_bath must be a MatrixDiscreteBath or nothing"))
    if !isnothing(problem.bath) && !isnothing(initial_bath)
        throw(ArgumentError("initial_bath is only meaningful when fitting a target hybridization"))
    end
    requested_bath_sites = isnothing(problem.bath) ? solver.bath_sites : length(problem.bath)
    _multi_ed_dimensions(size(problem.one_body, 1) + requested_bath_sites, solver)
    fit = isnothing(problem.bath) ? fit_matrix_bath(problem.target_hybridization, solver.bath_sites;
                                                    chemical_potential=problem.chemical_potential,
                                                    options=solver.bath_fit, initial_bath=initial_bath) : nothing
    bath = isnothing(fit) ? problem.bath : fit.bath
    specification = anderson_hamiltonian(problem; bath=bath)
    temperature = axis_temperature(problem.axis; kB=solver.kB)
    state = multiorbital_ed_state(specification, solver, temperature)
    values, density = _multi_ed_spectral_data(state, problem.axis)
    green = GreenFunction(problem.axis, values; labels=problem.labels,
                          metadata=Dict{Symbol,Any}(:method => :multiorbital_sector_lehmann, :backend => solver.backend,
                                                    :exact_finite_bath => true))
    green0 = noninteracting_impurity_green(problem; bath=bath)
    sigma = self_energy(green0, green)
    occupations = Float64.(real.(diag(density)))
    maximum(abs, density - density') <= 1e-10 || throw(ErrorException("impurity density matrix is not Hermitian"))
    metadata = copy(state.metadata)
    metadata[:temperature] = temperature
    metadata[:beta] = state.beta
    metadata[:bath_sites] = length(bath)
    metadata[:bath_fit_converged] = isnothing(fit) ? nothing : fit.converged
    metadata[:bath_fit_objective] = isnothing(fit) ? nothing : fit.objective
    metadata[:bath_fit_metadata] = isnothing(fit) ? nothing : copy(fit.metadata)
    metadata[:initial_bath_used] = !isnothing(initial_bath)
    return MultiOrbitalEDResult(specification, green, green0, sigma, density, occupations, bath, state, metadata)
end
