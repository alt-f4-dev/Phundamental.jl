# Thermodynamic scalar accessors.

free_energy(state::AbstractThermalState) = state.point.free_energy
internal_energy(state::AbstractThermalState) = state.point.internal_energy
entropy(state::AbstractThermalState) = state.point.entropy
heat_capacity(state::AbstractThermalState) = state.point.heat_capacity

free_energy(point::ThermodynamicPoint) = point.free_energy
internal_energy(point::ThermodynamicPoint) = point.internal_energy
entropy(point::ThermodynamicPoint) = point.entropy
heat_capacity(point::ThermodynamicPoint) = point.heat_capacity

free_energy(curve::ThermodynamicCurve) = curve.free_energy
internal_energy(curve::ThermodynamicCurve) = curve.internal_energy
entropy(curve::ThermodynamicCurve) = curve.entropy
heat_capacity(curve::ThermodynamicCurve) = curve.heat_capacity
