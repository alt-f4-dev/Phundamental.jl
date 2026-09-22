"""Return the zero-point correction `1/2 sum(omega) - 1/2 tr(A)` for an LSWT result."""
function spinwave_zero_point_correction(result::SpinWaveResult)
    return 0.5 * sum(result.frequencies) - 0.5 * real(tr(result.A))
end

"""Classical plus harmonic zero-point ground-state energy estimate."""
spinwave_ground_energy(result::SpinWaveResult) =
    result.classical_energy + spinwave_zero_point_correction(result)
