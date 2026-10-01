"""Return the mode energies/frequencies carried by a specialized mode result."""
mode_energies(result::QuadraticModeResult) = result.energies
mode_energies(result::PhononModeResult) = result.frequencies
mode_energies(result::PhononDispersionResult) = result.frequencies
mode_energies(result::SpinWaveResult) = result.frequencies

"""Return the numerical mode vectors carried by a specialized mode result. For phonon results these are the orthonormal mass-weighted dynamical-matrix eigenvectors; use `physical_displacement_modes` for physical atomic-displacement polarizations."""
mode_vectors(result::AbstractModeResult) = result.modes

function _phonon_inverse_sqrt_masses(model::ManyBodyModel, ndof::Integer)
    masses = Float64.(collect(model.parameters[:masses]))
    length(masses) == ndof || throw(DimensionMismatch("phonon mass vector must contain one value per displacement degree of freedom"))
    all(>(0.0), masses) || throw(ArgumentError("phonon masses must be positive"))
    return 1 ./ sqrt.(masses)
end

"""
    physical_displacement_modes(result::PhononModeResult)

Return the physical atomic-displacement polarization vectors corresponding to the mass-weighted phonon eigenvectors stored in `result.modes`. If `e` denotes the stored mass-weighted modes and `M` is the diagonal mass matrix, the returned matrix is `u = M^(-1/2)e`; consequently `e' * e ≈ I` implies `u' * M * u ≈ I`. Complex phases at nonzero wavevector and basis freedom inside degenerate subspaces are preserved.
"""
function physical_displacement_modes(result::PhononModeResult)
    ndof = size(result.modes, 1)
    size(result.modes, 2) == ndof || throw(DimensionMismatch("phonon mode matrix must be square"))
    inverse_sqrt_masses = _phonon_inverse_sqrt_masses(result.model, ndof)
    return result.modes .* reshape(inverse_sqrt_masses, ndof, 1)
end

"""
    physical_displacement_modes(result::PhononDispersionResult)

Return physical atomic-displacement polarization vectors for every wavevector in a phonon dispersion. The returned array has the same `(ndof, nbranch, nq)` shape as `result.modes` and is obtained from the stored mass-weighted modes by applying `M^(-1/2)` along the displacement-degree axis.
"""
function physical_displacement_modes(result::PhononDispersionResult)
    ndof, nbranch, nq = size(result.modes)
    nbranch == ndof || throw(DimensionMismatch("phonon dispersion mode matrices must be square"))
    inverse_sqrt_masses = _phonon_inverse_sqrt_masses(result.model, ndof)
    return result.modes .* reshape(inverse_sqrt_masses, ndof, 1, 1)
end

"""
    physical_displacement_modes(result::PhononDispersionResult, iq)

Return physical atomic-displacement polarization vectors for one wavevector in a phonon dispersion without allocating the full transformed `(ndof, nbranch, nq)` array.
"""
function physical_displacement_modes(result::PhononDispersionResult, iq::Integer)
    ndof, nbranch, nq = size(result.modes)
    nbranch == ndof || throw(DimensionMismatch("phonon dispersion mode matrices must be square"))
    1 <= iq <= nq || throw(BoundsError(result.qpoints, iq))
    inverse_sqrt_masses = _phonon_inverse_sqrt_masses(result.model, ndof)
    return view(result.modes, :, :, iq) .* reshape(inverse_sqrt_masses, ndof, 1)
end
