# Spin-resolved one-particle spectra reconstructed from finite-cluster Green matrices.

function _electron_spectral_intensity(value::Complex, normalization::Symbol)
    if normalization === :spectral_density || normalization === :minus_im_over_pi
        return -imag(value) / pi
    elseif normalization === :senechal
        return -2 * imag(value)
    end
    throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))
end

"""
    cluster_perturbation_spectral_function(workspace, momenta, axis; eta, t_intercluster, ...)

Evaluate the spin-resolved one-particle spectral function of a one-dimensional cluster perturbation theory reconstruction. The default normalization is `-Im G/pi`, which integrates to unity per spin and momentum when the complete frequency axis is retained. Set `normalization=:senechal` to reproduce the `-2 Im G` convention used by Senechal et al.; set `krylov_dim` to evaluate a shorter prefix of the projected-Lanczos chains already stored in the workspace.
"""
function cluster_perturbation_spectral_function(
    workspace::FermionGreenWorkspace,
    momenta,
    axis;
    eta::Real,
    t_intercluster::Number,
    cluster_translation::Real=length(workspace.sites),
    positions=nothing,
    normalization::Symbol=:spectral_density,
    parallel::Bool=true,
    krylov_dim=nothing,
)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    k_values = Float64.(collect(momenta))
    frequency_axis = Float64.(collect(axis))
    isempty(k_values) && throw(ArgumentError("momenta must not be empty"))
    isempty(frequency_axis) && throw(ArgumentError("axis must not be empty"))
    normalization in (:spectral_density, :minus_im_over_pi, :senechal) ||
        throw(ArgumentError("normalization must be :spectral_density, :minus_im_over_pi, or :senechal"))

    cluster_green = cluster_green_matrix(workspace, frequency_axis; eta=eta, parallel=parallel, krylov_dim=krylov_dim)
    intensity = zeros(Float64, length(k_values), length(frequency_axis))

    worker = function (frequency_index)
        Gc = view(cluster_green, :, :, frequency_index)
        for momentum_index in eachindex(k_values)
            Gk = cluster_perturbation_green(
                Gc,
                k_values[momentum_index];
                t_intercluster=t_intercluster,
                cluster_translation=cluster_translation,
                positions=positions,
            )
            intensity[momentum_index, frequency_index] = _electron_spectral_intensity(Gk, normalization)
        end
    end

    if parallel && Threads.nthreads() > 1 && length(frequency_axis) > 1
        Threads.@threads :static for frequency_index in eachindex(frequency_axis)
            worker(frequency_index)
        end
    else
        for frequency_index in eachindex(frequency_axis)
            worker(frequency_index)
        end
    end

    metadata = Dict{Symbol,Any}(
        :observable => :cluster_perturbation_spectral_function,
        :eta => Float64(eta),
        :normalization => normalization,
        :spin => workspace.spin,
        :ground_sector => workspace.ground_sector,
        :particle_sector => workspace.particle_sector,
        :hole_sector => workspace.hole_sector,
        :t_intercluster => t_intercluster,
        :cluster_translation => Float64(cluster_translation),
        :cpt_equation => :senechal_eqs_6_and_9,
        :krylov_dim => krylov_dim,
        :resolution_applied => false,
    )
    return SpectrumResult(k_values, frequency_axis, intensity, metadata)
end
