# Partition-function accessors.

function log_partition_function(state::AbstractThermalState)
    iszero(state.temperature) && throw(ArgumentError(
        "absolute log(Z) is not reported at exactly T=0; use free_energy(state)"
    ))
    return state.point.logZ
end

function partition_function(state::AbstractThermalState)
    logZ = log_partition_function(state)
    Z = exp(logZ)
    isfinite(Z) || throw(OverflowError(
        "partition function overflows Float64; use log_partition_function(state)"
    ))
    return Z
end

log_partition_function(point::ThermodynamicPoint) = begin
    iszero(point.temperature) && throw(ArgumentError(
        "absolute log(Z) is not reported at exactly T=0"
    ))
    point.logZ
end

partition_function(point::ThermodynamicPoint) = begin
    Z = exp(log_partition_function(point))
    isfinite(Z) || throw(OverflowError(
        "partition function overflows Float64; use log_partition_function(point)"
    ))
    Z
end
