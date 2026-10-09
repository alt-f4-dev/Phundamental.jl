# Reference finite-temperature impurity four-point functions for ladder diagrammatic DMFT.

"""Abstract backend for evaluating a local impurity two-particle correlation function on Matsubara axes."""
abstract type AbstractImpurityTwoParticleBackend end

"""
    TauGridFourPoint(; ntime=24, max_dimension=256, subtract_disconnected=true, algorithm=:factored, parallel=:serial, equal_time=:symmetric)

Deterministic reference evaluator for the connected impurity four-point function on a midpoint imaginary-time grid. `algorithm=:factored` evaluates the same ordered thermal traces as the original direct implementation while factoring repeated propagator and matrix products across the ordered-time simplices. `algorithm=:direct` retains the original implementation as a validation reference. `parallel=:threads` parallelizes the six ordered-time permutation regions with Julia tasks; each region uses a private reusable workspace. `equal_time=:symmetric` averages all fermionic time orderings on equal-time grid cells, removing the first-order finite-grid bias caused by assigning measure-zero discontinuity surfaces to one arbitrary ordering. `equal_time=:ordered` reproduces the legacy finite-grid convention. Calculations exceeding `max_dimension` are rejected explicitly.
"""
struct TauGridFourPoint <: AbstractImpurityTwoParticleBackend
    ntime::Int
    max_dimension::Int
    subtract_disconnected::Bool
    algorithm::Symbol
    parallel::Symbol
    equal_time::Symbol
end

function TauGridFourPoint(; ntime::Integer=24, max_dimension::Integer=256, subtract_disconnected::Bool=true,
                          algorithm::Symbol=:factored, parallel::Symbol=:serial, equal_time::Symbol=:symmetric)
    ntime >= 4 || throw(ArgumentError("ntime must be at least 4"))
    max_dimension > 0 || throw(ArgumentError("max_dimension must be positive"))
    algorithm in (:factored, :direct) || throw(ArgumentError("algorithm must be :factored or :direct"))
    parallel in (:serial, :threads) || throw(ArgumentError("parallel must be :serial or :threads"))
    equal_time in (:symmetric, :ordered) || throw(ArgumentError("equal_time must be :symmetric or :ordered"))
    return TauGridFourPoint(Int(ntime), Int(max_dimension), subtract_disconnected, algorithm, parallel, equal_time)
end

# Backward-compatible positional construction from the original three-field backend.
TauGridFourPoint(ntime::Integer, max_dimension::Integer, subtract_disconnected::Bool) =
    TauGridFourPoint(; ntime=ntime, max_dimension=max_dimension, subtract_disconnected=subtract_disconnected)

# Preserve the five-argument constructor introduced with the factored/threaded backend.
TauGridFourPoint(ntime::Integer, max_dimension::Integer, subtract_disconnected::Bool, algorithm::Symbol, parallel::Symbol) =
    TauGridFourPoint(; ntime=ntime, max_dimension=max_dimension, subtract_disconnected=subtract_disconnected, algorithm=algorithm, parallel=parallel,
                     equal_time=:ordered)

"""Spin-resolved and charge/magnetic impurity generalized susceptibilities together with quadrature diagnostics."""
struct ImpurityTwoParticleResult{S,C,M,B,D}
    spin_resolved::S
    charge::C
    magnetic::M
    local_bubble::B
    diagnostics::D
end

@inline function _permutation_parity(permutation)
    inversions = 0
    @inbounds for i in 1:length(permutation)-1, j in i+1:length(permutation)
        permutation[i] > permutation[j] && (inversions += 1)
    end
    return isodd(inversions) ? -1.0 : 1.0
end

function _ordered_trace_for_permutation(energies::Vector{Float64}, operators::Vector{Matrix{ComplexF64}}, times::Vector{Float64},
                                                beta::Float64, permutation::Vector{Int})
    sign = _permutation_parity(permutation)
    ordered_times = times[permutation]
    ordered_operators = operators[permutation]
    shifted = energies .- minimum(energies)
    partition = sum(exp.(-beta .* shifted))
    partition > 0 || throw(ErrorException("failed to construct a finite impurity partition function"))
    propagator(dt) = Diagonal(exp.(-dt .* shifted))
    product = Matrix(propagator(beta - ordered_times[1]))
    @inbounds for index in eachindex(ordered_operators)
        product = product * ordered_operators[index]
        interval = index < length(ordered_times) ? ordered_times[index] - ordered_times[index + 1] : ordered_times[index]
        product = product * propagator(interval)
    end
    return sign * tr(product) / partition
end

function _equal_time_orderings(times::Vector{Float64})
    base = sortperm(eachindex(times); by=i -> -times[i], alg=Base.Sort.MergeSort)
    orderings = Vector{Vector{Int}}([collect(base)])
    group_start = 1
    while group_start <= length(base)
        group_stop = group_start
        while group_stop < length(base) && times[base[group_stop + 1]] == times[base[group_start]]
            group_stop += 1
        end
        if group_stop > group_start
            group = collect(base[group_start:group_stop])
            permutations = Vector{Vector{Int}}()
            function append_permutations!(prefix::Vector{Int}, remaining::Vector{Int})
                if isempty(remaining)
                    push!(permutations, copy(prefix))
                    return nothing
                end
                for index in eachindex(remaining)
                    value = remaining[index]
                    push!(prefix, value)
                    next_remaining = copy(remaining)
                    deleteat!(next_remaining, index)
                    append_permutations!(prefix, next_remaining)
                    pop!(prefix)
                end
                return nothing
            end
            append_permutations!(Int[], group)
            expanded = Vector{Vector{Int}}()
            for ordering in orderings, group_permutation in permutations
                candidate = copy(ordering)
                candidate[group_start:group_stop] .= group_permutation
                push!(expanded, candidate)
            end
            orderings = expanded
        end
        group_start = group_stop + 1
    end
    return orderings
end

function _ordered_fermion_trace(energies::Vector{Float64}, operators::Vector{Matrix{ComplexF64}}, times::Vector{Float64}, beta::Float64;
                                equal_time::Symbol=:ordered)
    length(operators) == length(times) || throw(DimensionMismatch("operator and imaginary-time counts differ"))
    isempty(operators) && return 1.0 + 0.0im
    equal_time in (:symmetric, :ordered) || throw(ArgumentError("equal_time must be :symmetric or :ordered"))
    if equal_time === :ordered
        permutation = collect(sortperm(eachindex(times); by=i -> -times[i], alg=Base.Sort.MergeSort))
        return _ordered_trace_for_permutation(energies, operators, times, beta, permutation)
    end
    orderings = _equal_time_orderings(times)
    value = zero(ComplexF64)
    for permutation in orderings
        value += _ordered_trace_for_permutation(energies, operators, times, beta, permutation)
    end
    return value / length(orderings)
end

function _connected_four_point(energies, creation1, annihilation1, creation2, annihilation2, tau1, tau2, tau3, beta, subtract_disconnected, equal_time)
    four = _ordered_fermion_trace(energies, Matrix{ComplexF64}[creation1, annihilation1, creation2, annihilation2],
                                  Float64[tau1, tau2, tau3, 0.0], beta; equal_time=equal_time)
    subtract_disconnected || return four
    pair1 = _ordered_fermion_trace(energies, Matrix{ComplexF64}[creation1, annihilation1], Float64[tau1, tau2], beta; equal_time=equal_time)
    pair2 = _ordered_fermion_trace(energies, Matrix{ComplexF64}[creation2, annihilation2], Float64[tau3, 0.0], beta; equal_time=equal_time)
    return four - pair1 * pair2
end

struct _TauTraceData
    propagators::Matrix{Float64}
    wrap_by_time::Matrix{Float64}
    forward_by_difference::Matrix{Float64}
    partition::Float64
    ntime::Int
end

function _tau_trace_data(energies::Vector{Float64}, beta::Float64, ntime::Int)
    shifted = energies .- minimum(energies)
    partition = sum(exp.(-beta .* shifted))
    partition > 0 || throw(ErrorException("failed to construct a finite impurity partition function"))
    half_step = beta / (2ntime)
    propagators = Matrix{Float64}(undef, length(shifted), 2ntime + 1)
    @inbounds for h in 0:2ntime, state in eachindex(shifted)
        propagators[state, h + 1] = exp(-h * half_step * shifted[state])
    end
    wrap_by_time = Matrix{Float64}(undef, length(shifted), ntime)
    forward_by_difference = Matrix{Float64}(undef, length(shifted), ntime)
    @inbounds for time_index in 1:ntime
        wrap_by_time[:, time_index] .= @view propagators[:, 2(ntime - time_index) + 2]
    end
    @inbounds for difference in 0:ntime-1
        forward_by_difference[:, difference + 1] .= @view propagators[:, 2difference + 1]
    end
    return _TauTraceData(propagators, wrap_by_time, forward_by_difference, partition, ntime)
end

@inline _propagator(data::_TauTraceData, half_steps::Int) = @view data.propagators[:, half_steps + 1]

function _scale_columns!(destination::Matrix{ComplexF64}, matrix::Matrix{ComplexF64}, factors)
    size(destination) == size(matrix) || throw(DimensionMismatch("column-scaling workspace has the wrong shape"))
    @inbounds for column in axes(matrix, 2), row in axes(matrix, 1)
        destination[row, column] = matrix[row, column] * factors[column]
    end
    return destination
end

function _ordered_pair_midpoint_trace(data::_TauTraceData, first::Matrix{ComplexF64}, second::Matrix{ComplexF64}, high::Int, low::Int)
    d0 = _propagator(data, 2(data.ntime - high) + 1)
    d1 = _propagator(data, 2(high - low))
    d2 = _propagator(data, 2low - 1)
    value = zero(ComplexF64)
    @inbounds for a in axes(first, 1), b in axes(first, 2)
        value += d0[a] * d2[a] * first[a, b] * d1[b] * second[b, a]
    end
    return value / data.partition
end

function _ordered_pair_zero_trace(data::_TauTraceData, first::Matrix{ComplexF64}, second::Matrix{ComplexF64}, time_index::Int)
    d0 = _propagator(data, 2(data.ntime - time_index) + 1)
    d1 = _propagator(data, 2time_index - 1)
    value = zero(ComplexF64)
    @inbounds for a in axes(first, 1), b in axes(first, 2)
        value += d0[a] * first[a, b] * d1[b] * second[b, a]
    end
    return value / data.partition
end

function _disconnected_pairs(data::_TauTraceData, creation1::Matrix{ComplexF64}, annihilation1::Matrix{ComplexF64},
                             creation2::Matrix{ComplexF64}, annihilation2::Matrix{ComplexF64}, equal_time::Symbol)
    pair1 = Matrix{ComplexF64}(undef, data.ntime, data.ntime)
    pair2 = Vector{ComplexF64}(undef, data.ntime)
    @inbounds for i in 1:data.ntime, j in 1:data.ntime
        if i > j
            pair1[i, j] = _ordered_pair_midpoint_trace(data, creation1, annihilation1, i, j)
        elseif i < j
            pair1[i, j] = -_ordered_pair_midpoint_trace(data, annihilation1, creation1, j, i)
        elseif equal_time === :symmetric
            forward = _ordered_pair_midpoint_trace(data, creation1, annihilation1, i, j)
            backward = -_ordered_pair_midpoint_trace(data, annihilation1, creation1, j, i)
            pair1[i, j] = 0.5 * (forward + backward)
        else
            pair1[i, j] = _ordered_pair_midpoint_trace(data, creation1, annihilation1, i, j)
        end
    end
    @inbounds for k in 1:data.ntime
        pair2[k] = _ordered_pair_zero_trace(data, creation2, annihilation2, k)
    end
    return pair1, pair2
end

mutable struct _FactoredTraceWorkspace
    scaled::Matrix{ComplexF64}
    tail::Matrix{ComplexF64}
    middle::Matrix{ComplexF64}
    coupling::Matrix{ComplexF64}
    projected::Matrix{ComplexF64}
end

function _FactoredTraceWorkspace(dimension::Int, ntime::Int)
    return _FactoredTraceWorkspace(Matrix{ComplexF64}(undef, dimension, dimension), Matrix{ComplexF64}(undef, dimension, dimension),
                                   Matrix{ComplexF64}(undef, dimension, dimension), Matrix{ComplexF64}(undef, dimension, dimension),
                                   Matrix{ComplexF64}(undef, dimension, ntime))
end

@inline function _original_time_indices(permutation::NTuple{3,Int}, high::Int, middle::Int, low::Int)
    i = permutation[1] == 1 ? high : permutation[2] == 1 ? middle : low
    j = permutation[1] == 2 ? high : permutation[2] == 2 ? middle : low
    k = permutation[1] == 3 ? high : permutation[2] == 3 ? middle : low
    return i, j, k
end

function _fill_ordered_region!(cube::Array{ComplexF64,3}, counts, accumulation_lock, data::_TauTraceData,
                               operators::NTuple{3,Matrix{ComplexF64}}, final_operator::Matrix{ComplexF64},
                               permutation::NTuple{3,Int}, pair1, pair2, subtract_disconnected::Bool, equal_time::Symbol)
    first = operators[permutation[1]]
    second = operators[permutation[2]]
    third = operators[permutation[3]]
    sign = _permutation_parity(permutation)
    symmetric = equal_time === :symmetric

    function store_value!(i::Int, j::Int, k::Int, value::ComplexF64, tied::Bool)
        if symmetric && tied
            Base.lock(accumulation_lock)
            try
                cube[i, j, k] += value
                counts[i, j, k] += UInt8(1)
            finally
                Base.unlock(accumulation_lock)
            end
        else
            cube[i, j, k] = value
            !isnothing(counts) && (counts[i, j, k] = UInt8(1))
        end
        return nothing
    end

    workspace = _FactoredTraceWorkspace(size(first, 1), data.ntime)
    for low in 1:data.ntime
        d3 = _propagator(data, 2low - 1)
        _scale_columns!(workspace.scaled, third, d3)
        mul!(workspace.tail, workspace.scaled, final_operator)
        middle_start = symmetric ? low : permutation[2] > permutation[3] ? low + 1 : low
        @inbounds for middle in middle_start:data.ntime
            d2 = _propagator(data, 2(middle - low))
            _scale_columns!(workspace.scaled, second, d2)
            mul!(workspace.middle, workspace.scaled, workspace.tail)
            for a in axes(first, 1), b in axes(first, 2)
                workspace.coupling[a, b] = first[a, b] * workspace.middle[b, a]
            end
            high_start = symmetric ? middle : permutation[1] > permutation[2] ? middle + 1 : middle
            high_start > data.ntime && continue
            nhigh = data.ntime - high_start + 1
            left = @view data.wrap_by_time[:, high_start:data.ntime]
            right = @view data.forward_by_difference[:, high_start-middle+1:data.ntime-middle+1]
            projected = @view workspace.projected[:, 1:nhigh]
            mul!(projected, workspace.coupling, right)
            for offset in 1:nhigh
                high = high_start + offset - 1
                trace_value = zero(ComplexF64)
                @simd for a in axes(first, 1)
                    trace_value += left[a, offset] * projected[a, offset]
                end
                i, j, k = _original_time_indices(permutation, high, middle, low)
                value = sign * trace_value / data.partition
                subtract_disconnected && (value -= pair1[i, j] * pair2[k])
                store_value!(i, j, k, value, high == middle || middle == low)
            end
        end
    end
    return cube
end

function _connected_tau_cube_factored(energies::Vector{Float64}, creation1::Matrix{ComplexF64}, annihilation1::Matrix{ComplexF64},
                                      creation2::Matrix{ComplexF64}, annihilation2::Matrix{ComplexF64}, beta::Float64,
                                      backend::TauGridFourPoint)
    data = _tau_trace_data(energies, beta, backend.ntime)
    pair1, pair2 = backend.subtract_disconnected ?
                   _disconnected_pairs(data, creation1, annihilation1, creation2, annihilation2, backend.equal_time) : (nothing, nothing)
    symmetric = backend.equal_time === :symmetric
    cube = symmetric ? zeros(ComplexF64, backend.ntime, backend.ntime, backend.ntime) :
                       fill(ComplexF64(NaN, NaN), backend.ntime, backend.ntime, backend.ntime)
    counts = symmetric ? zeros(UInt8, backend.ntime, backend.ntime, backend.ntime) : nothing
    accumulation_lock = symmetric ? ReentrantLock() : nothing
    operators = (creation1, annihilation1, creation2)
    permutations = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    if backend.parallel === :threads && Base.Threads.nthreads() > 1
        @sync for permutation in permutations
            let current_permutation = permutation
                Base.Threads.@spawn _fill_ordered_region!(cube, counts, accumulation_lock, data, operators, annihilation2, current_permutation,
                                                          pair1, pair2, backend.subtract_disconnected, backend.equal_time)
            end
        end
    else
        for permutation in permutations
            _fill_ordered_region!(cube, counts, accumulation_lock, data, operators, annihilation2, permutation, pair1, pair2,
                                  backend.subtract_disconnected, backend.equal_time)
        end
    end
    if symmetric
        all(>(UInt8(0)), counts) || throw(ErrorException("symmetric imaginary-time cube was not filled completely"))
        @inbounds for index in eachindex(cube)
            cube[index] /= counts[index]
        end
    else
        all(isfinite, cube) || throw(ErrorException("factored imaginary-time cube was not filled completely with finite values"))
    end
    all(isfinite, cube) || throw(ErrorException("factored imaginary-time cube contains non-finite values"))
    return cube
end

function _connected_tau_cube_direct(energies::Vector{Float64}, creation1::Matrix{ComplexF64}, annihilation1::Matrix{ComplexF64},
                                    creation2::Matrix{ComplexF64}, annihilation2::Matrix{ComplexF64}, beta::Float64,
                                    backend::TauGridFourPoint)
    delta_tau = beta / backend.ntime
    taus = Float64[(index - 0.5) * delta_tau for index in 1:backend.ntime]
    cube = Array{ComplexF64}(undef, backend.ntime, backend.ntime, backend.ntime)
    @inbounds for i in eachindex(taus), j in eachindex(taus), k in eachindex(taus)
        cube[i, j, k] = _connected_four_point(energies, creation1, annihilation1, creation2, annihilation2,
                                              taus[i], taus[j], taus[k], beta, backend.subtract_disconnected, backend.equal_time)
    end
    return cube
end

function _connected_tau_cube(state::ExactGibbsState, first_mode::Int, second_mode::Int, backend::TauGridFourPoint)
    dimension = length(state.solution.energies)
    dimension <= backend.max_dimension || throw(ArgumentError("TauGridFourPoint dimension $dimension exceeds max_dimension=$(backend.max_dimension); use a smaller ED bath, raise the explicit limit, or supply precomputed two-particle data"))
    energies = Float64.(real.(state.solution.energies))
    first_annihilation = Matrix{ComplexF64}(_eigenbasis_operator(state, c(first_mode)))
    first_creation = Matrix{ComplexF64}(_eigenbasis_operator(state, adjoint(c(first_mode))))
    second_annihilation = Matrix{ComplexF64}(_eigenbasis_operator(state, c(second_mode)))
    second_creation = Matrix{ComplexF64}(_eigenbasis_operator(state, adjoint(c(second_mode))))
    beta = state.beta
    delta_tau = beta / backend.ntime
    taus = Float64[(index - 0.5) * delta_tau for index in 1:backend.ntime]
    cube = backend.algorithm === :factored ?
           _connected_tau_cube_factored(energies, first_creation, first_annihilation, second_creation, second_annihilation, beta, backend) :
           _connected_tau_cube_direct(energies, first_creation, first_annihilation, second_creation, second_annihilation, beta, backend)
    return taus, cube
end

function _fourier_project_particle_hole(cube::Array{ComplexF64,3}, taus::Vector{Float64}, fermionic_axis::FermionicMatsubaraAxis,
                                        bosonic_axis::BosonicMatsubaraAxis)
    ntime = length(taus)
    size(cube) == (ntime, ntime, ntime) || throw(DimensionMismatch("imaginary-time cube has inconsistent dimensions"))
    isapprox(fermionic_axis.beta, bosonic_axis.beta; rtol=0.0, atol=16eps(Float64) * max(fermionic_axis.beta, 1.0)) ||
        throw(ArgumentError("fermionic and bosonic axes must have the same beta"))
    beta = fermionic_axis.beta
    delta_tau = beta / ntime
    fermionic_frequencies = imag.(fermionic_axis.values)
    bosonic_frequencies = imag.(bosonic_axis.values)
    nfermionic = length(fermionic_axis)
    phase1 = Matrix{ComplexF64}(undef, nfermionic, ntime)
    @inbounds for iν in eachindex(fermionic_frequencies), it in eachindex(taus)
        phase1[iν, it] = exp(-im * fermionic_frequencies[iν] * taus[it])
    end
    values = Array{ComplexF64}(undef, nfermionic, nfermionic, length(bosonic_axis))
    cube_matrix = reshape(cube, ntime * ntime, ntime)
    phase2 = Matrix{ComplexF64}(undef, nfermionic, ntime)
    phase3 = Matrix{ComplexF64}(undef, nfermionic, ntime)
    pair_phases = Matrix{ComplexF64}(undef, nfermionic, ntime * ntime)
    contracted3 = Matrix{ComplexF64}(undef, ntime * ntime, nfermionic)
    @inbounds for (iΩ, omega) in enumerate(bosonic_frequencies)
        for iν in eachindex(fermionic_frequencies), it in eachindex(taus)
            phase2[iν, it] = exp(im * (fermionic_frequencies[iν] + omega) * taus[it])
            phase3[iν, it] = exp(-im * (fermionic_frequencies[iν] + omega) * taus[it])
        end
        for iν in 1:nfermionic, j in 1:ntime, i in 1:ntime
            pair_phases[iν, (j - 1) * ntime + i] = phase1[iν, i] * phase2[iν, j]
        end
        mul!(contracted3, cube_matrix, transpose(phase3))
        mul!(@view(values[:, :, iΩ]), pair_phases, contracted3)
    end
    values .*= delta_tau^3
    return values
end

function _connected_particle_hole_values(state::ExactGibbsState, first_mode::Int, second_mode::Int, backend::TauGridFourPoint,
                                         fermionic_axis::FermionicMatsubaraAxis, bosonic_axis::BosonicMatsubaraAxis)
    taus, cube = _connected_tau_cube(state, first_mode, second_mode, backend)
    return _fourier_project_particle_hole(cube, taus, fermionic_axis, bosonic_axis)
end

"""
    impurity_two_particle(state, fermionic_axis, bosonic_axis; backend=TauGridFourPoint(), up_mode=1, down_mode=2)

Evaluate the connected local particle-hole generalized susceptibility of a spinful impurity. The convention is `χ_{σσ′}^{νν′Ω} = ∫dτ₁dτ₂dτ₃ e^{-iντ₁}e^{i(ν+Ω)τ₂}e^{-i(ν′+Ω)τ₃}[⟨T c†σ(τ₁)cσ(τ₂)c†σ′(τ₃)cσ′(0)⟩-disc.]`. For a noninteracting impurity this convention approaches `-β G(ν)G(ν+Ω)δνν′`. The returned channel combinations are `χ_charge = χ_↑↑ + χ_↑↓` and `χ_magnetic = χ_↑↑ - χ_↑↓`; no additional factor of two is hidden in the channel definition.
"""
function impurity_two_particle(state::ExactGibbsState, fermionic_axis::FermionicMatsubaraAxis, bosonic_axis::BosonicMatsubaraAxis;
                               backend::AbstractImpurityTwoParticleBackend=TauGridFourPoint(), up_mode::Integer=1, down_mode::Integer=2)
    backend isa TauGridFourPoint || throw(ArgumentError("unsupported impurity two-particle backend $(typeof(backend))"))
    _validate_axis_temperature(state, fermionic_axis)
    isapprox(state.beta, bosonic_axis.beta; rtol=1e-12, atol=1e-12 * max(state.beta, 1.0)) ||
        throw(ArgumentError("thermal-state beta does not match the bosonic Matsubara axis"))
    same_spin_values = _connected_particle_hole_values(state, Int(up_mode), Int(up_mode), backend, fermionic_axis, bosonic_axis)
    opposite_spin_values = _connected_particle_hole_values(state, Int(up_mode), Int(down_mode), backend, fermionic_axis, bosonic_axis)
    convention = Dict{Symbol,Any}(:fourier_convention => :particle_hole_physical, :channel_normalization => :upup_plusminus_updown,
                                  :noninteracting_limit => :minus_beta_GG_delta, :disconnected_subtracted => backend.subtract_disconnected)
    same_spin = GeneralizedSusceptibility(fermionic_axis, bosonic_axis, same_spin_values; channel=:upup,
                                          metadata=merge(copy(convention), Dict{Symbol,Any}(:spin_pair => (:up, :up))), copy_values=false)
    opposite_spin = GeneralizedSusceptibility(fermionic_axis, bosonic_axis, opposite_spin_values; channel=:updown,
                                              metadata=merge(copy(convention), Dict{Symbol,Any}(:spin_pair => (:up, :down))), copy_values=false)
    charge_values = same_spin_values .+ opposite_spin_values
    magnetic_values = same_spin_values .- opposite_spin_values
    charge = GeneralizedSusceptibility(fermionic_axis, bosonic_axis, charge_values; channel=:charge,
                                       metadata=merge(copy(convention), Dict{Symbol,Any}(:combination => :upup_plus_updown)), copy_values=false)
    magnetic = GeneralizedSusceptibility(fermionic_axis, bosonic_axis, magnetic_values; channel=:magnetic,
                                         metadata=merge(copy(convention), Dict{Symbol,Any}(:combination => :upup_minus_updown)), copy_values=false)
    impurity_green = matsubara_green(state, fermionic_axis; modes=[Int(up_mode)], labels=[:impurity_up])
    bubble = local_bubble(impurity_green, bosonic_axis; channel=:upup, missing_frequency=:zero)
    diagnostics = Dict{Symbol,Any}(:backend => :tau_grid_four_point, :ntime => backend.ntime, :delta_tau => state.beta / backend.ntime,
                                   :hilbert_dimension => length(state.solution.energies), :frequency_box_complete => get(bubble.metadata, :frequency_box_complete, false),
                                   :convergence_parameter => :ntime, :algorithm => backend.algorithm, :parallel => backend.parallel, :equal_time => backend.equal_time,
                                   :julia_threads => Base.Threads.nthreads())
    return ImpurityTwoParticleResult(Dict(:upup => same_spin, :updown => opposite_spin), charge, magnetic, bubble, diagnostics)
end

"""Maximum relative difference between two impurity two-particle calculations, used for explicit imaginary-time-grid convergence checks."""
function impurity_two_particle_residual(a::ImpurityTwoParticleResult, b::ImpurityTwoParticleResult)
    residuals = Float64[]
    for field in (:charge, :magnetic)
        x = getfield(a, field).values
        y = getfield(b, field).values
        size(x) == size(y) || throw(DimensionMismatch("two-particle frequency boxes differ"))
        push!(residuals, norm(x .- y) / max(norm(x), norm(y), eps(Float64)))
    end
    return maximum(residuals)
end
