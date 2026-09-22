# Ewald.jl
#
# Direct reciprocal-space Ewald reference implementation.
#
# Coulomb:
#   E = 1/2 q' K q
#
# with
#   K_real,ij = sum_R' erfc(alpha*r)/r
#   K_rec,ij  = (4pi/V) sum_G!=0 exp(-G^2/4a^2)/G^2 cos(G.rij)
#   K_self,ii = -2 alpha/sqrt(pi)
#
# and, when a uniform neutralizing background is requested,
#   K_bg,ij = -pi/(alpha^2 V).
#
# Dipoles use T(r)=I/r^3-3rr'/r^5 = -Hessian(1/r).
# The real-space tensor below is the exact -Hessian[erfc(alpha*r)/r],
# the reciprocal tensor is proportional to GG'/G^2, and the self pair
# tensor is -4 alpha^3/(3sqrt(pi)) I. The macroscopic surface term is zero
# for tin-foil/conducting boundary conditions.

# Float64 complementary error function from Julia's platform libm.
# This avoids adding a SpecialFunctions.jl dependency solely for the Ewald
# screening kernel while retaining the correctly rounded system libm routine.
@inline _ewald_erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Cdouble, (Cdouble,), x)
@inline _ewald_erfc(x::Real) = _ewald_erfc(Float64(x))

function _auto_ewald_parameters(
    cluster::Supercell,
    method::EwaldSummation,
)
    V = _supercell_volume(cluster)
    N = nsites(cluster)
    L = cbrt(V)

    alpha = isnothing(method.alpha) ?
        sqrt(pi) * max(N, 1)^(1 / 6) / L :
        method.alpha

    # These cutoffs follow the Gaussian asymptotic scales. They are starting
    # points only; automatic refinement below is the actual convergence gate.
    target = sqrt(max(-log(method.tolerance / 10), 1.0))
    real_cutoff = isnothing(method.real_cutoff) ?
        target / alpha :
        method.real_cutoff
    reciprocal_cutoff = isnothing(method.reciprocal_cutoff) ?
        2 * alpha * target :
        method.reciprocal_cutoff

    return Float64(alpha), Float64(real_cutoff), Float64(reciprocal_cutoff)
end

@inline function _ewald_real_coulomb(r::Real, alpha::Real)
    return _ewald_erfc(alpha * r) / r
end

function _ewald_real_dipolar_tensor(
    rvec::AbstractVector,
    alpha::Real,
    strength::Real,
)
    r2 = dot(rvec, rvec)
    r2 > 0 || throw(ArgumentError("dipolar Ewald kernel is singular at r=0"))
    r = sqrt(r2)
    invr2 = inv(r2)
    invr3 = invr2 / r
    invr4 = invr2^2
    invr5 = invr4 / r

    ar = alpha * r
    e = exp(-(ar^2))
    g = _ewald_erfc(ar)
    c = 2alpha / sqrt(pi)

    # -Hessian[erfc(alpha*r)/r] =
    #   A I - B rr'
    # A = erfc(ar)/r^3 + 2a/sqrt(pi) exp(-a^2 r^2)/r^2
    # B = 3 erfc(ar)/r^5
    #     + 6a/sqrt(pi) exp(-a^2 r^2)/r^4
    #     + 4a^3/sqrt(pi) exp(-a^2 r^2)/r^2
    A = g * invr3 + c * e * invr2
    B = 3g * invr5 + 3c * e * invr4 + 2alpha^2 * c * e * invr2

    return strength .* (
        A .* Matrix{Float64}(I, 3, 3) .-
        B .* (rvec * transpose(rvec))
    )
end

function _ewald_scalar_raw(
    interaction::CoulombInteraction,
    cluster::Supercell,
    alpha::Real,
    real_cutoff::Real,
    reciprocal_cutoff::Real,
    boundary::Symbol,
)
    _require_full_3d_periodicity(cluster)
    boundary === :tin_foil || error("internal boundary validation failure")

    positions = _position_matrix(cluster)
    N = nsites(cluster)
    V = _supercell_volume(cluster)
    real_images = _real_image_vectors(cluster, real_cutoff, positions)
    Gvectors = _reciprocal_vectors(cluster, reciprocal_cutoff)

    realspace = zeros(Float64, N, N)
    reciprocal = zeros(Float64, N, N)
    self = zeros(Float64, N, N)
    background = zeros(Float64, N, N)
    rc2 = real_cutoff^2

    # Real-space pair matrix. Computing only the upper triangle halves the
    # O(N^2) pair work; periodic inversion symmetry guarantees Kji=Kij.
    for i in 1:N
        ri = view(positions, :, i)
        for j in i:N
            dr0 = view(positions, :, j) .- ri
            value = 0.0
            for (n, R) in real_images
                i == j && _zero_shift(n) && continue
                rvec = dr0 .+ R
                r2 = dot(rvec, rvec)
                (r2 <= 0 || r2 > rc2 * (1 + 64eps(Float64))) && continue
                value += _ewald_real_coulomb(sqrt(r2), alpha)
            end
            value *= interaction.strength
            realspace[i, j] = value
            realspace[j, i] = value
        end
    end

    # Reciprocal pair matrix. Each reciprocal vector contributes a rank-one
    # phase correlation, evaluated by BLAS-friendly outer products.
    for (_, G) in Gvectors
        G2 = dot(G, G)
        weight = interaction.strength * (4pi / V) *
                 exp(-G2 / (4alpha^2)) / G2
        theta = vec(transpose(positions) * G)
        z = cis.(theta)
        reciprocal .+= weight .* Base.real.(z * adjoint(z))
    end

    self_coeff = interaction.strength * (-2alpha / sqrt(pi))
    for i in 1:N
        self[i, i] = self_coeff
    end

    if interaction.neutralizing_background
        # 1/2 q' K_bg q = -pi Q^2/(2 alpha^2 V).
        background .= interaction.strength * (-pi / (alpha^2 * V))
    end

    total = realspace + reciprocal + self + background
    metadata = Dict{Symbol,Any}(
        :interaction => :coulomb,
        :summation => :ewald,
        :alpha => Float64(alpha),
        :real_cutoff => Float64(real_cutoff),
        :reciprocal_cutoff => Float64(reciprocal_cutoff),
        :volume => V,
        :boundary => boundary,
        :neutralizing_background => interaction.neutralizing_background,
        :requires_neutrality => !interaction.neutralizing_background,
        :real_vectors => length(real_images),
        :reciprocal_vectors => length(Gvectors),
        :particle_mesh_ewald => :future,
    )
    return ScalarInteractionResult(total, realspace, reciprocal, self, background, metadata)
end

function _ewald_tensor_raw(
    interaction::DipolarInteraction,
    cluster::Supercell,
    alpha::Real,
    real_cutoff::Real,
    reciprocal_cutoff::Real,
    boundary::Symbol,
)
    _require_full_3d_periodicity(cluster)
    boundary === :tin_foil || error("internal boundary validation failure")

    positions = _position_matrix(cluster)
    N = nsites(cluster)
    V = _supercell_volume(cluster)
    real_images = _real_image_vectors(cluster, real_cutoff, positions)
    Gvectors = _reciprocal_vectors(cluster, reciprocal_cutoff)

    realspace = zeros(Float64, N, N, 3, 3)
    reciprocal = zeros(Float64, N, N, 3, 3)
    self = zeros(Float64, N, N, 3, 3)
    surface = zeros(Float64, N, N, 3, 3)  # exactly zero for tin-foil
    rc2 = real_cutoff^2

    for i in 1:N
        ri = view(positions, :, i)
        for j in i:N
            dr0 = view(positions, :, j) .- ri
            Tij = zeros(Float64, 3, 3)
            for (n, R) in real_images
                i == j && _zero_shift(n) && continue
                rvec = dr0 .+ R
                r2 = dot(rvec, rvec)
                (r2 <= 0 || r2 > rc2 * (1 + 64eps(Float64))) && continue
                Tij .+= _ewald_real_dipolar_tensor(
                    rvec, alpha, interaction.strength,
                )
            end
            realspace[i, j, :, :] .= Tij
            realspace[j, i, :, :] .= transpose(Tij)
        end
    end

    for (_, G) in Gvectors
        G2 = dot(G, G)
        scalar = interaction.strength * (4pi / V) *
                 exp(-G2 / (4alpha^2)) / G2
        GG = G * transpose(G)

        theta = vec(transpose(positions) * G)
        z = cis.(theta)
        phase = Base.real.(z * adjoint(z))

        for a in 1:3, b in 1:3
            coeff = scalar * GG[a, b]
            iszero(coeff) && continue
            @views reciprocal[:, :, a, b] .+= coeff .* phase
        end
    end

    self_coeff = interaction.strength * (-4alpha^3 / (3sqrt(pi)))
    for i in 1:N, a in 1:3
        self[i, i, a, a] = self_coeff
    end

    total = realspace + reciprocal + self + surface
    metadata = Dict{Symbol,Any}(
        :interaction => :dipolar,
        :summation => :ewald,
        :alpha => Float64(alpha),
        :real_cutoff => Float64(real_cutoff),
        :reciprocal_cutoff => Float64(reciprocal_cutoff),
        :volume => V,
        :boundary => boundary,
        :surface_term => :zero_for_tin_foil,
        :real_vectors => length(real_images),
        :reciprocal_vectors => length(Gvectors),
        :particle_mesh_ewald => :future,
    )
    return TensorInteractionResult(total, realspace, reciprocal, self, surface, metadata)
end

function _relative_matrix_change(a, b)
    return norm(a - b) / max(norm(b), 1.0)
end

function _refined_ewald(
    rawfun,
    interaction,
    cluster,
    method::EwaldSummation,
)
    alpha, rc, kc = _auto_ewald_parameters(cluster, method)
    result = rawfun(interaction, cluster, alpha, rc, kc, method.boundary)

    if !method.check_convergence
        result.metadata[:converged] = nothing
        result.metadata[:convergence_error_estimate] = NaN
        result.metadata[:refinements] = 0
        return result
    end

    prev = result
    last_error = Inf
    for refinement in 1:method.max_refinements
        rc *= method.refinement_factor
        kc *= method.refinement_factor
        current = rawfun(interaction, cluster, alpha, rc, kc, method.boundary)
        last_error = _relative_matrix_change(prev.total, current.total)

        current.metadata[:convergence_error_estimate] = last_error
        current.metadata[:refinements] = refinement
        current.metadata[:convergence_tolerance] = method.tolerance

        if last_error <= max(method.tolerance, 100eps(Float64))
            current.metadata[:converged] = true
            return current
        end
        prev = current
    end

    prev.metadata[:converged] = false
    prev.metadata[:convergence_error_estimate] = last_error
    prev.metadata[:refinements] = method.max_refinements
    prev.metadata[:convergence_tolerance] = method.tolerance

    method.strict && throw(ArgumentError(
        "Ewald sum failed cutoff-refinement convergence: relative change=$last_error " *
        "after $(method.max_refinements) refinements. Increase max_refinements, " *
        "supply larger cutoffs, or adjust alpha."
    ))
    return prev
end

_interaction_matrix(interaction::CoulombInteraction, cluster::Supercell, method::EwaldSummation) = _refined_ewald(_ewald_scalar_raw, interaction, cluster, method)

_interaction_tensor(interaction::DipolarInteraction, cluster::Supercell, method::EwaldSummation) = _refined_ewald(_ewald_tensor_raw, interaction, cluster, method)

function interaction_matrix(interaction::CoulombInteraction, cluster::Supercell; method::AbstractInteractionSummation=EwaldSummation())
    return _interaction_matrix(interaction, cluster, method)
end

function interaction_tensor(interaction::DipolarInteraction, cluster::Supercell; method::AbstractInteractionSummation=EwaldSummation())
    return _interaction_tensor(interaction, cluster, method)
end
