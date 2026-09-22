# Spin-resolved exact diagonalization for the normal-state Anderson impurity model.
#
# The full Fock space is decomposed into exact (N_up, N_down) sectors. Basis layouts,
# hopping maps, and impurity annihilation maps depend only on bath size and are cached.

struct _ImpurityHopMap
    rows::Vector{Int}
    columns::Vector{Int}
    amplitudes::Vector{Float64}
end

struct _ImpurityAnnihilationMap
    target_block::Int
    rows::Vector{Int}
    columns::Vector{Int}
    amplitudes::Vector{Float64}
end

struct _ImpuritySectorBlockLayout
    nup::Int
    ndown::Int
    basis::ComputationalBasis
    masks::Vector{UInt64}
    index::Dict{UInt64,Int}
    impurity_up::Vector{Float64}
    impurity_down::Vector{Float64}
    double_occupancy::Vector{Float64}
    bath_occupancy::Matrix{Float64}
    hopping_maps::Vector{_ImpurityHopMap}
end

struct _ImpuritySectorWorkspace
    bath_sites::Int
    spatial_orbitals::Int
    blocks::Vector{_ImpuritySectorBlockLayout}
    lookup::Dict{Tuple{Int,Int},Int}
    up_annihilation::Vector{Union{Nothing,_ImpurityAnnihilationMap}}
    down_annihilation::Vector{Union{Nothing,_ImpurityAnnihilationMap}}
    full_dimension::Int
    max_sector_dimension::Int
end

struct _ImpuritySectorSolution
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    probabilities::Vector{Float64}
end

"""Exact finite-temperature impurity state assembled from spin-resolved particle-number sectors."""
struct SectorizedImpurityThermalState{M,W} <: AbstractThermalState
    model::M
    temperature::Float64
    kB::Float64
    beta::Float64
    workspace::W
    sectors::Vector{_ImpuritySectorSolution}
    logZ::Float64
    metadata::Dict{Symbol,Any}
end

const _IMPURITY_SECTOR_WORKSPACE_CACHE = Dict{Int,_ImpuritySectorWorkspace}()
const _IMPURITY_SECTOR_WORKSPACE_LOCK = ReentrantLock()
const _IMPURITY_THREADING_MODES = (:serial, :threads, :auto)

function _validate_impurity_threading(threading::Symbol)
    threading in _IMPURITY_THREADING_MODES || throw(ArgumentError("threading must be :serial, :threads, or :auto"))
    return threading
end

@inline function _impurity_threading_enabled(threading::Symbol)
    _validate_impurity_threading(threading)
    return threading !== :serial && Threads.nthreads() > 1
end

@inline _effective_impurity_threading(threading::Symbol) = _impurity_threading_enabled(threading) ? :threads : :serial

@inline _impurity_bit_occupied(mask::UInt64, position::Int) = ((mask >> (position - 1)) & UInt64(1)) == UInt64(1)

@inline function _impurity_fermion_parity(mask::UInt64, position::Int)
    position <= 1 && return 1.0
    preceding = (UInt64(1) << (position - 1)) - UInt64(1)
    return isodd(count_ones(mask & preceding)) ? -1.0 : 1.0
end

@inline function _impurity_annihilate(mask::UInt64, position::Int)
    _impurity_bit_occupied(mask, position) || return mask, 0.0, false
    flag = UInt64(1) << (position - 1)
    return mask & ~flag, _impurity_fermion_parity(mask, position), true
end

@inline function _impurity_create(mask::UInt64, position::Int)
    _impurity_bit_occupied(mask, position) && return mask, 0.0, false
    flag = UInt64(1) << (position - 1)
    return mask | flag, _impurity_fermion_parity(mask, position), true
end

@inline function _impurity_hop(mask::UInt64, from_position::Int, to_position::Int)
    intermediate, amplitude1, ok1 = _impurity_annihilate(mask, from_position)
    ok1 || return mask, 0.0, false
    output, amplitude2, ok2 = _impurity_create(intermediate, to_position)
    ok2 || return mask, 0.0, false
    return output, amplitude1 * amplitude2, true
end

function _impurity_hopping_map(masks::Vector{UInt64}, index::Dict{UInt64,Int}, from_position::Int, to_position::Int)
    rows = Int[]
    columns = Int[]
    amplitudes = Float64[]
    for (column, mask) in enumerate(masks)
        output, amplitude, ok = _impurity_hop(mask, from_position, to_position)
        ok || continue
        row = get(index, output, 0)
        row == 0 && throw(ErrorException("Anderson hopping left its conserved (N_up, N_down) sector"))
        push!(rows, row)
        push!(columns, column)
        push!(amplitudes, amplitude)
    end
    return _ImpurityHopMap(rows, columns, amplitudes)
end

function _impurity_sector_block_layout(model::ManyBodyModel, bath_sites::Int, nup::Int, ndown::Int)
    data = _packed_fermion_basis(model, FermionSector(nup=nup, ndown=ndown); max_dimension=50_000_000)
    expected_positions = collect(1:(2 * (bath_sites + 1)))
    data.mode_positions == expected_positions || throw(ArgumentError("sectorized Anderson ED requires canonical fermion mode ordering"))

    dimension = length(data.masks)
    impurity_up = Vector{Float64}(undef, dimension)
    impurity_down = Vector{Float64}(undef, dimension)
    double_occupancy = Vector{Float64}(undef, dimension)
    bath_occupancy = zeros(Float64, bath_sites, dimension)

    @inbounds for column in 1:dimension
        mask = data.masks[column]
        nimp_up = _impurity_bit_occupied(mask, 1) ? 1.0 : 0.0
        nimp_down = _impurity_bit_occupied(mask, 2) ? 1.0 : 0.0
        impurity_up[column] = nimp_up
        impurity_down[column] = nimp_down
        double_occupancy[column] = nimp_up * nimp_down
        for p in 1:bath_sites
            up_position = 2p + 1
            down_position = 2p + 2
            bath_occupancy[p, column] = (_impurity_bit_occupied(mask, up_position) ? 1.0 : 0.0) +
                                        (_impurity_bit_occupied(mask, down_position) ? 1.0 : 0.0)
        end
    end

    hopping_maps = Vector{_ImpurityHopMap}(undef, 2bath_sites)
    for p in 1:bath_sites
        hopping_maps[2p - 1] = _impurity_hopping_map(data.masks, data.index, 2p + 1, 1)
        hopping_maps[2p] = _impurity_hopping_map(data.masks, data.index, 2p + 2, 2)
    end

    return _ImpuritySectorBlockLayout(nup, ndown, data.basis, data.masks, data.index, impurity_up, impurity_down,
                                      double_occupancy, bath_occupancy, hopping_maps)
end

function _impurity_annihilation_map(blocks::Vector{_ImpuritySectorBlockLayout}, lookup::Dict{Tuple{Int,Int},Int}, source_block::Int, spin::Symbol)
    source = blocks[source_block]
    target_key = spin === :up ? (source.nup - 1, source.ndown) : spin === :down ? (source.nup, source.ndown - 1) : throw(ArgumentError("spin must be :up or :down"))
    minimum(target_key) < 0 && return nothing
    target_block = get(lookup, target_key, 0)
    target_block == 0 && return nothing
    target = blocks[target_block]
    position = spin === :up ? 1 : 2
    rows = Int[]
    columns = Int[]
    amplitudes = Float64[]

    for (column, mask) in enumerate(source.masks)
        output, amplitude, ok = _impurity_annihilate(mask, position)
        ok || continue
        row = get(target.index, output, 0)
        row == 0 && throw(ErrorException("impurity annihilation transition could not be located in the adjacent sector"))
        push!(rows, row)
        push!(columns, column)
        push!(amplitudes, amplitude)
    end
    return _ImpurityAnnihilationMap(target_block, rows, columns, amplitudes)
end

function _build_impurity_sector_workspace(model::ManyBodyModel, bath_sites::Int)
    spatial_orbitals = bath_sites + 1
    blocks = _ImpuritySectorBlockLayout[]
    lookup = Dict{Tuple{Int,Int},Int}()
    for nup in 0:spatial_orbitals, ndown in 0:spatial_orbitals
        push!(blocks, _impurity_sector_block_layout(model, bath_sites, nup, ndown))
        lookup[(nup, ndown)] = length(blocks)
    end

    up_annihilation = Vector{Union{Nothing,_ImpurityAnnihilationMap}}(undef, length(blocks))
    down_annihilation = similar(up_annihilation)
    for block_index in eachindex(blocks)
        up_annihilation[block_index] = _impurity_annihilation_map(blocks, lookup, block_index, :up)
        down_annihilation[block_index] = _impurity_annihilation_map(blocks, lookup, block_index, :down)
    end

    full_dimension = sum(length(block.masks) for block in blocks)
    max_sector_dimension = maximum(length(block.masks) for block in blocks)
    expected_dimension = big(4)^spatial_orbitals
    expected_dimension <= typemax(Int) || throw(OverflowError("Anderson Fock-space dimension exceeds Int"))
    full_dimension == Int(expected_dimension) || throw(ErrorException("spin-resolved sector decomposition does not reconstruct the full Anderson Fock space"))
    return _ImpuritySectorWorkspace(bath_sites, spatial_orbitals, blocks, lookup, up_annihilation, down_annihilation,
                                    full_dimension, max_sector_dimension)
end

function _impurity_sector_workspace(model::ManyBodyModel, bath_sites::Int)
    lock(_IMPURITY_SECTOR_WORKSPACE_LOCK)
    try
        return get!(_IMPURITY_SECTOR_WORKSPACE_CACHE, bath_sites) do
            _build_impurity_sector_workspace(model, bath_sites)
        end
    finally
        unlock(_IMPURITY_SECTOR_WORKSPACE_LOCK)
    end
end

function _impurity_sector_hamiltonian(block::_ImpuritySectorBlockLayout, specification::AndersonImpurityModel)
    bath_sites = length(specification.bath_energies)
    dimension = length(block.masks)
    H = zeros(ComplexF64, dimension, dimension)
    impurity_level = specification.impurity_energy - specification.chemical_potential

    @inbounds for column in 1:dimension
        energy = impurity_level * (block.impurity_up[column] + block.impurity_down[column]) +
                 specification.U * block.double_occupancy[column]
        for p in 1:bath_sites
            energy += (specification.bath_energies[p] - specification.chemical_potential) * block.bath_occupancy[p, column]
        end
        H[column, column] = energy
    end

    for p in 1:bath_sites
        coupling = specification.hybridizations[p]
        for map_index in (2p - 1, 2p)
            hopping = block.hopping_maps[map_index]
            @inbounds for k in eachindex(hopping.rows)
                row = hopping.rows[k]
                column = hopping.columns[k]
                amplitude = hopping.amplitudes[k]
                H[row, column] += coupling * amplitude
                H[column, row] += conj(coupling) * amplitude
            end
        end
    end
    return H
end

function _diagonalize_impurity_sector(block::_ImpuritySectorBlockLayout, specification::AndersonImpurityModel)
    H = _impurity_sector_hamiltonian(block, specification)
    eigensystem = eigen!(Hermitian(H))
    return Float64.(eigensystem.values), Matrix{ComplexF64}(eigensystem.vectors)
end

function _sectorized_impurity_state(model::ManyBodyModel, specification::AndersonImpurityModel; temperature::Real, kB::Real=1.0, threading::Symbol=:serial)
    temperature > 0 || throw(ArgumentError("sectorized finite-temperature impurity ED requires temperature > 0"))
    kB > 0 || throw(ArgumentError("kB must be positive"))
    _validate_impurity_threading(threading)
    bath_sites = length(specification.bath_energies)
    workspace = _impurity_sector_workspace(model, bath_sites)
    block_energies = Vector{Vector{Float64}}(undef, length(workspace.blocks))
    block_vectors = Vector{Matrix{ComplexF64}}(undef, length(workspace.blocks))
    effective_threading = _effective_impurity_threading(threading)

    if effective_threading === :threads
        tasks = Vector{Task}(undef, length(workspace.blocks))
        for block_index in eachindex(workspace.blocks)
            block = workspace.blocks[block_index]
            tasks[block_index] = Threads.@spawn _diagonalize_impurity_sector(block, specification)
        end
        for block_index in eachindex(tasks)
            block_energies[block_index], block_vectors[block_index] = fetch(tasks[block_index])
        end
    else
        for block_index in eachindex(workspace.blocks)
            block_energies[block_index], block_vectors[block_index] = _diagonalize_impurity_sector(workspace.blocks[block_index], specification)
        end
    end

    beta = inv(Float64(kB) * Float64(temperature))
    minimum_energy = minimum(minimum(energies) for energies in block_energies)
    shifted_partition = 0.0
    block_weights = Vector{Vector{Float64}}(undef, length(block_energies))
    for block_index in eachindex(block_energies)
        weights = exp.(-beta .* (block_energies[block_index] .- minimum_energy))
        block_weights[block_index] = weights
        shifted_partition += sum(weights)
    end
    isfinite(shifted_partition) && shifted_partition > 0 || throw(ErrorException("sectorized impurity Gibbs normalization is not finite and positive"))

    sectors = Vector{_ImpuritySectorSolution}(undef, length(workspace.blocks))
    for block_index in eachindex(workspace.blocks)
        sectors[block_index] = _ImpuritySectorSolution(block_energies[block_index], block_vectors[block_index], block_weights[block_index] ./ shifted_partition)
    end
    logZ = -beta * minimum_energy + log(shifted_partition)
    metadata = Dict{Symbol,Any}(
        :method => :sectorized_exact_gibbs,
        :backend => :spin_resolved_ed,
        :bath_sites => bath_sites,
        :full_hilbert_dimension => workspace.full_dimension,
        :sector_count => length(workspace.blocks),
        :max_sector_dimension => workspace.max_sector_dimension,
        :cached_workspace => true,
        :threading_requested => threading,
        :threading_effective => effective_threading,
        :julia_threads => Threads.nthreads(),
    )
    return SectorizedImpurityThermalState(model, Float64(temperature), Float64(kB), beta, workspace, sectors, logZ, metadata)
end

function _sectorized_diagonal_expectation(state::SectorizedImpurityThermalState, selector::Symbol)
    value = 0.0
    for block_index in eachindex(state.workspace.blocks)
        block = state.workspace.blocks[block_index]
        solution = state.sectors[block_index]
        diagonal = selector === :density ? block.impurity_up .+ block.impurity_down :
                   selector === :double_occupancy ? block.double_occupancy :
                   selector === :spin_z ? 0.5 .* (block.impurity_up .- block.impurity_down) :
                   throw(ArgumentError("unknown sectorized impurity observable $selector"))
        vectors = solution.vectors
        probabilities = solution.probabilities
        @inbounds for eigenstate_index in eachindex(probabilities)
            expectation_value = 0.0
            for basis_index in eachindex(diagonal)
                expectation_value += abs2(vectors[basis_index, eigenstate_index]) * diagonal[basis_index]
            end
            value += probabilities[eigenstate_index] * expectation_value
        end
    end
    return value
end

function expectation(state::SectorizedImpurityThermalState, observable; hbar::Real=get(state.model.parameters, :hbar, 1.0))
    if observable === :impurity_density
        return ThermalEstimate(_sectorized_diagonal_expectation(state, :density), 0.0)
    elseif observable === :impurity_double_occupancy
        return ThermalEstimate(_sectorized_diagonal_expectation(state, :double_occupancy), 0.0)
    elseif observable === :impurity_spin_z
        return ThermalEstimate(_sectorized_diagonal_expectation(state, :spin_z), 0.0)
    end

    operator = if observable isa Symbol
        haskey(state.model.observables, observable) || throw(KeyError("observable $observable is not registered in this model"))
        state.model.observables[observable]
    else
        observable
    end

    value = 0.0 + 0.0im
    for block_index in eachindex(state.workspace.blocks)
        block = state.workspace.blocks[block_index]
        solution = state.sectors[block_index]
        O = matrix(operator, block.basis; sparse=false, hbar=hbar)
        temporary = O * solution.vectors
        @inbounds for eigenstate_index in eachindex(solution.probabilities)
            value += solution.probabilities[eigenstate_index] * dot(view(solution.vectors, :, eigenstate_index), view(temporary, :, eigenstate_index))
        end
    end
    return ThermalEstimate(value, 0.0)
end

function _sectorized_annihilation_eigenmatrix(state::SectorizedImpurityThermalState, source_block::Int, spin::Symbol)
    transition = spin === :up ? state.workspace.up_annihilation[source_block] :
                 spin === :down ? state.workspace.down_annihilation[source_block] : throw(ArgumentError("spin must be :up or :down"))
    isnothing(transition) && return nothing
    source_solution = state.sectors[source_block]
    target_solution = state.sectors[transition.target_block]
    temporary = zeros(ComplexF64, size(target_solution.vectors, 1), size(source_solution.vectors, 2))
    @inbounds for k in eachindex(transition.rows)
        row = transition.rows[k]
        column = transition.columns[k]
        amplitude = transition.amplitudes[k]
        for eigenstate_index in axes(source_solution.vectors, 2)
            temporary[row, eigenstate_index] += amplitude * source_solution.vectors[column, eigenstate_index]
        end
    end
    return transition.target_block, adjoint(target_solution.vectors) * temporary
end

function _sectorized_source_green_contribution(state::SectorizedImpurityThermalState, axis::FermionicMatsubaraAxis, source_block::Int, spin::Symbol)
    values = zeros(ComplexF64, length(axis))
    transformed = _sectorized_annihilation_eigenmatrix(state, source_block, spin)
    isnothing(transformed) && return values
    target_block, amplitudes = transformed
    source_solution = state.sectors[source_block]
    target_solution = state.sectors[target_block]

    @inbounds for source_state in eachindex(source_solution.energies)
        source_energy = source_solution.energies[source_state]
        source_probability = source_solution.probabilities[source_state]
        for target_state in eachindex(target_solution.energies)
            matrix_element_squared = abs2(amplitudes[target_state, source_state])
            iszero(matrix_element_squared) && continue
            spectral_weight = (source_probability + target_solution.probabilities[target_state]) * matrix_element_squared
            energy_difference = target_solution.energies[target_state] - source_energy
            for frequency_index in eachindex(axis.values)
                values[frequency_index] += spectral_weight / (axis[frequency_index] + energy_difference)
            end
        end
    end
    return values
end

function _sectorized_impurity_spin_green(state::SectorizedImpurityThermalState, axis::FermionicMatsubaraAxis, spin::Symbol)
    scale = max(abs(state.beta), abs(axis.beta), 1.0)
    isapprox(state.beta, axis.beta; rtol=1e-12, atol=1e-12 * scale) || throw(ArgumentError("thermal-state beta does not match Matsubara-axis beta"))
    values = zeros(ComplexF64, length(axis))
    threading = get(state.metadata, :threading_requested, :serial)
    effective_threading = _effective_impurity_threading(threading)

    if effective_threading === :threads
        contributions = Vector{Vector{ComplexF64}}(undef, length(state.sectors))
        tasks = Vector{Task}(undef, length(state.sectors))
        for source_block in eachindex(state.sectors)
            tasks[source_block] = Threads.@spawn _sectorized_source_green_contribution(state, axis, source_block, spin)
        end
        for source_block in eachindex(tasks)
            contributions[source_block] = fetch(tasks[source_block])
        end
        for contribution in contributions
            @inbounds for frequency_index in eachindex(values)
                values[frequency_index] += contribution[frequency_index]
            end
        end
    else
        for source_block in eachindex(state.sectors)
            contribution = _sectorized_source_green_contribution(state, axis, source_block, spin)
            @inbounds for frequency_index in eachindex(values)
                values[frequency_index] += contribution[frequency_index]
            end
        end
    end

    metadata = Dict{Symbol,Any}(
        :method => :sectorized_exact_lehmann,
        :spin => spin,
        :full_hilbert_dimension => state.workspace.full_dimension,
        :max_sector_dimension => state.workspace.max_sector_dimension,
        :threading_requested => threading,
        :threading_effective => effective_threading,
        :julia_threads => Threads.nthreads(),
    )
    return GreenFunction(axis, values; labels=[Symbol("impurity_", spin)], metadata=metadata)
end

function _paramagnetic_impurity_green(state::SectorizedImpurityThermalState, axis::FermionicMatsubaraAxis)
    up = _sectorized_impurity_spin_green(state, axis, :up)
    down = _sectorized_impurity_spin_green(state, axis, :down)
    values = 0.5 .* (up.values .+ down.values)
    spin_difference = maximum(abs.(up.values .- down.values); init=0.0)
    green = GreenFunction(axis, values; labels=[:impurity],
                          metadata=Dict{Symbol,Any}(:method => :sectorized_exact_lehmann, :paramagnetic_average => true,
                                                    :spin_difference => spin_difference,
                                                    :full_hilbert_dimension => state.workspace.full_dimension,
                                                    :max_sector_dimension => state.workspace.max_sector_dimension,
                                                    :threading_requested => get(state.metadata, :threading_requested, :serial),
                                                    :threading_effective => get(state.metadata, :threading_effective, :serial),
                                                    :julia_threads => Threads.nthreads()))
    return green, spin_difference
end
