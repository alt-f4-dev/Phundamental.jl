"""Return the mode energies/frequencies carried by a specialized mode result."""
mode_energies(result::QuadraticModeResult) = result.energies
mode_energies(result::PhononModeResult) = result.frequencies
mode_energies(result::PhononDispersionResult) = result.frequencies
mode_energies(result::SpinWaveResult) = result.frequencies

"""Return the numerical mode vectors carried by a specialized mode result."""
mode_vectors(result::AbstractModeResult) = result.modes
