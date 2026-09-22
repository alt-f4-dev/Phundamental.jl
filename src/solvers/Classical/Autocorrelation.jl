# Autocorrelation.jl

"""
    autocorrelation_time(values; maxlag=nothing)

Estimate the integrated autocorrelation time using the initial-positive
sequence of the normalized autocorrelation function:

    tau_int = 1/2 + sum_{t>=1} rho(t),

stopping at the first nonpositive lag. This is intentionally conservative for
a lightweight built-in diagnostic; production uncertainty analysis may use
more sophisticated windowing or batch-means estimators.
"""
function autocorrelation_time(values::AbstractVector; maxlag=nothing)
    x = Float64.(values)
    n = length(x)
    n <= 1 && return 0.5
    mu = mean(x)
    centered = x .- mu
    var0 = dot(centered, centered) / n
    var0 <= eps(Float64) && return 0.5

    lagmax = isnothing(maxlag) ? min(n ÷ 2, 10_000) : min(Int(maxlag), n - 1)
    tau = 0.5
    for lag in 1:lagmax
        rho = dot(
            view(centered, 1:(n - lag)),
            view(centered, (lag + 1):n),
        ) / ((n - lag) * var0)
        rho <= 0 && break
        tau += rho
    end
    return max(tau, 0.5)
end

effective_sample_size(values::AbstractVector) =
    min(length(values), length(values) / (2 * autocorrelation_time(values)))

function correlated_stderr(values::AbstractVector)
    x = Float64.(values)
    length(x) <= 1 && return 0.0
    neff = max(effective_sample_size(x), 1.0)
    return std(x; corrected=true) / sqrt(neff)
end
