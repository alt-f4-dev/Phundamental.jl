# Peer-reviewed benchmark descriptors and residual reports for diagrammatic-DMFT observables.

"""Peer-reviewed susceptibility benchmark with explicit model parameters, axes, values, and uncertainty provenance."""
struct SusceptibilityBenchmark{Q,V,U,P,M}
    name::Symbol
    citation::String
    doi::String
    channel::Symbol
    qpoints::Q
    bosonic_indices::Vector{Int}
    values::V
    uncertainty::U
    parameters::P
    metadata::M
end

function SusceptibilityBenchmark(name::Symbol, citation::AbstractString, doi::AbstractString, channel::Symbol, qpoints,
                                 bosonic_indices, values; uncertainty=nothing, parameters=Dict{Symbol,Any}(), metadata=Dict{Symbol,Any}())
    qs = collect(qpoints)
    indices = Int.(collect(bosonic_indices))
    matrix = ComplexF64.(values)
    size(matrix) == (length(qs), length(indices)) || throw(DimensionMismatch("benchmark values must have shape (nq, nbosonic)"))
    errors = isnothing(uncertainty) ? fill(NaN, size(matrix)) : Float64.(uncertainty)
    size(errors) == size(matrix) || throw(DimensionMismatch("benchmark uncertainty must match the value matrix"))
    return SusceptibilityBenchmark(name, String(citation), String(doi), channel, qs, indices, matrix, errors,
                                   Dict{Symbol,Any}(parameters), Dict{Symbol,Any}(metadata))
end

"""Residual report for a susceptibility calculation compared with a peer-reviewed numerical benchmark."""
struct SusceptibilityBenchmarkResult{B,R,N,M}
    benchmark::B
    residuals::R
    normalized_residuals::N
    passed::Bool
    metadata::M
end

function _benchmark_axis_positions(axis::BosonicMatsubaraAxis, indices::AbstractVector{<:Integer})
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    positions = Int[]
    for index in indices
        haskey(lookup, index) || throw(ArgumentError("result bosonic axis does not contain benchmark Matsubara index $index"))
        push!(positions, lookup[index])
    end
    return positions
end

"""
    compare_susceptibility_benchmark(result, benchmark; atol=0, rtol=0, uncertainty_sigma=3)

Compare a `SusceptibilityResult` or `DiagrammaticDMFTResult` against explicitly stored peer-reviewed values. The comparison never digitizes or interpolates published figures internally; reference values and their uncertainty provenance must be supplied when constructing `SusceptibilityBenchmark`.
"""
function compare_susceptibility_benchmark(result::SusceptibilityResult, benchmark::SusceptibilityBenchmark;
                                            atol::Real=0.0, rtol::Real=0.0, uncertainty_sigma::Real=3.0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    uncertainty_sigma > 0 || throw(ArgumentError("uncertainty_sigma must be positive"))
    length(result.qpoints) == length(benchmark.qpoints) || throw(DimensionMismatch("result and benchmark q-point counts differ"))
    result.qpoints == benchmark.qpoints || throw(ArgumentError("result and benchmark q-point labels differ; explicit remapping is required"))
    positions = _benchmark_axis_positions(result.axis, benchmark.bosonic_indices)
    predicted = result.values[:, positions]
    residuals = abs.(predicted .- benchmark.values)
    normalized = fill(NaN, size(residuals))
    passed = true
    for index in eachindex(residuals)
        uncertainty = benchmark.uncertainty[index]
        scale = Float64(atol) + Float64(rtol) * max(abs(predicted[index]), abs(benchmark.values[index]), 1.0)
        if isfinite(uncertainty) && uncertainty > 0
            normalized[index] = residuals[index] / uncertainty
            scale = max(scale, uncertainty_sigma * uncertainty)
        end
        residuals[index] <= scale || (passed = false)
    end
    finite_normalized = filter(isfinite, vec(normalized))
    metadata = Dict{Symbol,Any}(
        :max_absolute_residual => isempty(residuals) ? 0.0 : maximum(residuals),
        :max_normalized_residual => isempty(finite_normalized) ? NaN : maximum(finite_normalized),
        :atol => Float64(atol),
        :rtol => Float64(rtol),
        :uncertainty_sigma => Float64(uncertainty_sigma),
        :benchmark_doi => benchmark.doi,
    )
    return SusceptibilityBenchmarkResult(benchmark, residuals, normalized, passed, metadata)
end

compare_susceptibility_benchmark(result::DiagrammaticDMFTResult, benchmark::SusceptibilityBenchmark; kwargs...) =
    compare_susceptibility_benchmark(susceptibility(result), benchmark; kwargs...)

"""Return the peer-reviewed publications selected as acceptance targets for nonlocal Hubbard-model validation."""
function diagrammatic_dmft_benchmark_references()
    return [
        (name=:gukelberger_2017_dual_fermion, doi="10.1103/PhysRevB.96.035152",
         citation="J. Gukelberger, E. Kozik, and H. Hafermann, Phys. Rev. B 96, 035152 (2017)",
         target=:nonlocal_self_energy_and_convergence_regimes,
         status=:reference_values_not_embedded),
        (name=:leblanc_2019_magnetic_susceptibility, doi="10.1103/PhysRevB.100.075123",
         citation="J. P. F. LeBlanc et al., Phys. Rev. B 100, 075123 (2019)",
         target=:momentum_resolved_magnetic_susceptibility,
         status=:reference_values_not_embedded),
        (name=:leblanc_2015_hubbard_benchmark, doi="10.1103/PhysRevX.5.041041",
         citation="J. P. F. LeBlanc et al., Phys. Rev. X 5, 041041 (2015)",
         target=:independent_hubbard_reference_data,
         status=:reference_values_not_embedded),
    ]
end

"""Peer-reviewed Matsubara self-energy benchmark with explicit momentum/frequency coordinates and uncertainty provenance."""
struct SelfEnergyBenchmark{K,V,U,P,M}
    name::Symbol
    citation::String
    doi::String
    kpoints::K
    fermionic_indices::Vector{Int}
    values::V
    uncertainty::U
    parameters::P
    metadata::M
end

function SelfEnergyBenchmark(name::Symbol, citation::AbstractString, doi::AbstractString, kpoints, fermionic_indices, values;
                             uncertainty=nothing, parameters=Dict{Symbol,Any}(), metadata=Dict{Symbol,Any}())
    ks = collect(kpoints)
    indices = Int.(collect(fermionic_indices))
    matrix = ComplexF64.(values)
    size(matrix) == (length(ks), length(indices)) || throw(DimensionMismatch("self-energy benchmark values must have shape (nk, nfermionic)"))
    errors = isnothing(uncertainty) ? fill(NaN, size(matrix)) : Float64.(uncertainty)
    size(errors) == size(matrix) || throw(DimensionMismatch("self-energy benchmark uncertainty must match the value matrix"))
    return SelfEnergyBenchmark(name, String(citation), String(doi), ks, indices, matrix, errors,
                               Dict{Symbol,Any}(parameters), Dict{Symbol,Any}(metadata))
end

"""Residual report for a momentum-resolved self-energy compared with a peer-reviewed numerical benchmark."""
struct SelfEnergyBenchmarkResult{B,R,N,M}
    benchmark::B
    residuals::R
    normalized_residuals::N
    passed::Bool
    metadata::M
end

function _fermionic_benchmark_positions(axis::FermionicMatsubaraAxis, indices::AbstractVector{<:Integer})
    lookup = Dict(index => position for (position, index) in enumerate(axis.indices))
    return Int[get(lookup, index, 0) == 0 ? throw(ArgumentError("result fermionic axis does not contain benchmark Matsubara index $index")) : lookup[index] for index in indices]
end

function compare_self_energy_benchmark(result::MomentumResolvedSelfEnergy, benchmark::SelfEnergyBenchmark;
                                       atol::Real=0.0, rtol::Real=0.0, uncertainty_sigma::Real=3.0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    uncertainty_sigma > 0 || throw(ArgumentError("uncertainty_sigma must be positive"))
    result.kpoints == benchmark.kpoints || throw(ArgumentError("result and benchmark momentum labels differ; explicit remapping is required"))
    positions = _fermionic_benchmark_positions(result.axis, benchmark.fermionic_indices)
    predicted = result.values[:, positions]
    residuals = abs.(predicted .- benchmark.values)
    normalized = fill(NaN, size(residuals))
    passed = true
    for index in eachindex(residuals)
        uncertainty = benchmark.uncertainty[index]
        scale = Float64(atol) + Float64(rtol) * max(abs(predicted[index]), abs(benchmark.values[index]), 1.0)
        if isfinite(uncertainty) && uncertainty > 0
            normalized[index] = residuals[index] / uncertainty
            scale = max(scale, uncertainty_sigma * uncertainty)
        end
        residuals[index] <= scale || (passed = false)
    end
    finite_normalized = filter(isfinite, vec(normalized))
    metadata = Dict{Symbol,Any}(:max_absolute_residual => isempty(residuals) ? 0.0 : maximum(residuals),
                                :max_normalized_residual => isempty(finite_normalized) ? NaN : maximum(finite_normalized),
                                :atol => Float64(atol), :rtol => Float64(rtol), :uncertainty_sigma => Float64(uncertainty_sigma),
                                :benchmark_doi => benchmark.doi)
    return SelfEnergyBenchmarkResult(benchmark, residuals, normalized, passed, metadata)
end

function compare_self_energy_benchmark(result::LadderDGAResult, benchmark::SelfEnergyBenchmark; kwargs...)
    isnothing(result.self_energy) && throw(ArgumentError("LadderDGAResult does not contain a reconstructed nonlocal self-energy"))
    return compare_self_energy_benchmark(result.self_energy, benchmark; kwargs...)
end

compare_susceptibility_benchmark(result::LadderDGAResult, benchmark::SusceptibilityBenchmark; kwargs...) =
    compare_susceptibility_benchmark(susceptibility(result, benchmark.channel), benchmark; kwargs...)
