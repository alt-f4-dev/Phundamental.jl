#!/usr/bin/env julia

# =============================================================================
# morris_2009_dy2ti2o7_paper_faithful.jl
#
# Stand-alone reproduction of the THEORY/CALCULATION panels associated with:
#
#   D. J. P. Morris et al.,
#   "Dirac Strings and Magnetic Monopoles in the Spin Ice Dy2Ti2O7",
#   Science 326, 411-414 (2009),
#   DOI: 10.1126/science.1178868
#
# and its Supporting Online Material.
#
# The script uses Phundamental.jl where its public API is appropriate:
#
#   * crystal geometry;
#   * local <111> site frames;
#   * periodic dipolar/Ewald interactions;
#   * static tensor scattering result containers;
#   * magnetic-neutron polarization/form-factor contraction.
#
# Paper-specific calculations (Debye-Huckel monopole gas, large-N/spherical
# pyrochlore correlations, and Dirac-string random walks) intentionally remain
# in this reproduction script rather than becoming Phundamental core API.
#
# Main-paper reciprocal-space theory panels are rendered in the same geometric
# form used by Morris et al.: the HK0, HK1, and 0KL planes are embedded together
# in a three-dimensional reciprocal-space frame with H,K in [0,5] and L in [0,1].
#
# OUTPUT
# ------
# All generated figures and numerical tables are written to:
#
#     ./Dy2Ti2O7/
#
# EXPECTED PACKAGE LOCATION
# -------------------------
# Run this script from the Phundamental.jl `src/` directory, where
# `Phundamental.jl` is present:
#
#     julia morris_2009_dy2ti2o7_paper_faithful.jl
#
# Plotting dependency:
#
#     import Pkg
#     Pkg.add("GLMakie")
#
# The three extensions (LocalFrames, LongRangeInteractions,
# ClassicalSampling/EnsembleObservables) must already be integrated in the
# package. They were runtime validated together before this reproduction.
#
# REPRODUCIBILITY BOUNDARY
# ------------------------
# The Morris PDFs do not contain the original experimental event/intensity,
# magnetization-sweep, or calorimetry datasets. Consequently this script DOES
# NOT fabricate the experimental panels. It reproduces the theoretical curves,
# reciprocal-space calculations, and analytic phase-boundary information.
#
# The supplement delegates the large-N treatment and Debye-Huckel free energy to cited references. The script implements the large-N construction explicitly and uses the spin-ice-specialized Debye-Huckel equations written out by Castelnovo, Moessner, and Sondhi, Phys. Rev. B 84, 144435 (2011), which identify the parameters used for Morris Fig. 1. The standard
# Dy2Ti2O7 microscopic parameters referenced by Morris through S9/S10 are
#
#     J = -3.72 K,   D = 1.41 K,
#
# which imply
#
#     J_nn = J/3 = -1.24 K,
#     D_nn = 5D/3 = 2.35 K,
#     J_eff = 1.11 K.
#
# They are also consistent with the supplement's stated bare monopole cost
#
#     Delta = 2J/3 + (8/3)(1 + sqrt(2/3))D = 4.35 K.
#
# The standard Dy3+ form-factor coefficients are not tabulated by Morris.
# The conventional dipole-approximation coefficients used below are included
# only to supply the "relevant magnetic form factor" explicitly mentioned in
# the paper. Set USE_DY_FORM_FACTOR=false to reproduce the geometry-only
# normalized cross section.
# =============================================================================

include(joinpath(@__DIR__, "..", "src/Phundamental.jl"))
using .Phundamental

using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles

try
    @eval using GLMakie
catch err
    error(
        "This reproduction script needs GLMakie.jl for figure output.\n" *
        "Install once with: import Pkg; Pkg.add(\"GLMakie\")\n\nOriginal error:\n$err"
    )
end

const M = Phundamental.Models
const O = Phundamental.Observables

# =============================================================================
# 0. User-facing reproduction controls
# =============================================================================

const OUTPUT_DIR = joinpath(@__DIR__, "benchmark-figures/Dy2Ti2O7")

# Reciprocal-space image resolution. 121 is a good default; increase to 181/241
# for publication-quality maps at increased runtime.
const MAP_N = 301

# Brillouin-zone quadrature for the large-N self-consistency condition.
# Runtime scales as LARGE_N_BZ_N^3, but each operation is only 4x4.
const LARGE_N_BZ_N = 12

# Morris states that a string length of order 50 sites is required at 0.7 K,
# h = 5/7 hS.
const STRING_LENGTH = 50

const USE_DY_FORM_FACTOR = true

mkpath(OUTPUT_DIR)

# =============================================================================
# 1. Physical constants and Morris parameterization
# =============================================================================

const kB = 1.380649e-23                 # J/K
const NA = 6.02214076e23                # mol^-1
const Rgas = kB * NA                    # J/(mol K)
const mu0 = 4pi * 1e-7                  # J T^-2 m^-3
const muB = 9.2740100783e-24            # J/T

# Article: magnetic moment per Dy ion ~10 mu_B.
const MU_DY = 10.0 * muB

# Supplement: magnetic monopole charge.
const Q_MONOPOLE = 4.28e-13             # J/(T m) == A m

# The dumbbell mapping gives Q = 2 mu / a_d.
const A_DIAMOND = 2.0 * MU_DY / Q_MONOPOLE
const A_CUBIC = 4.0 * A_DIAMOND / sqrt(3.0)
const R_NN = A_CUBIC * sqrt(2.0) / 4.0

# Phundamental v0.0.1 validates Bravais-lattice nonsingularity using an absolute determinant threshold. Keep the physical constants above in SI units, but represent crystal geometry internally in Angstrom so a valid FCC cell is not rejected solely because its SI volume is numerically small.
const A_CUBIC_GEOM = A_CUBIC * 1e10
const R_NN_GEOM = R_NN * 1e10

# Microscopic dipolar spin-ice parameterization referenced through S9/S10.
const J_EXCHANGE_K = -3.72
const D_DIPOLE_K = 1.41
const J_NN_K = J_EXCHANGE_K / 3.0
const D_NN_K = 5.0 * D_DIPOLE_K / 3.0
const J_EFF_K = J_NN_K + D_NN_K

# Supplement's bare monopole cost.
const DELTA_MONOPOLE_K = 4.35
const DELTA_FROM_JD_K = 2.0 * J_EXCHANGE_K / 3.0 + (8.0 / 3.0) * (1.0 + sqrt(2.0 / 3.0)) * D_DIPOLE_K

# Independent physical calculation of D from the moment and NN spacing.
const D_FROM_GEOMETRY_K = (mu0 / (4pi)) * MU_DY^2 / R_NN^3 / kB

# Biases printed in the main article.
const BIAS_FIG3 = 0.53
const BIAS_FIG4_MID = 0.64
const BIAS_FIG4_HIGH = 0.80

# The single-tetrahedron curve shown in Morris et al. is reproduced by the
# projective-equivalence value J_eff=1.45 K; Castelnovo et al., PRB 84, 144435
# (2011), Appendix A, explicitly identifies this value with Fig. 1 of Morris.
const J_EFF_FIG1_K = 1.45

# Coulomb energy of two monopoles on neighboring diamond-lattice sites.
const E_NN_MONOPOLE_K = mu0 * Q_MONOPOLE^2 / (4pi * A_DIAMOND * kB)

# Paper reciprocal-space extents for the three-plane 3D renderings.
const PAPER_H_RANGE = (-5.0, 5.0)
const PAPER_K_RANGE = (0.0, 5.0)
const PAPER_L_RANGE = (0.0, 1.0)
const PAPER_COLORMAP = :turbo
const PAPER_COLOR_QUANTILE = 0.99

# =============================================================================
# 2. Pyrochlore geometry and local frames using Phundamental
# =============================================================================

"""
Construct the conventional pyrochlore basis as an FCC Bravais lattice with a
four-site basis.

The basis is equivalent to Supporting Table S1 after translating the tetrahedron
center to (1/8,1/8,1/8)a. Local z axes point from the tetrahedron center to each
spin site.
"""
function build_pyrochlore_geometry(; a::Float64=A_CUBIC_GEOM)
    a1 = a .* [0.0, 0.5, 0.5]
    a2 = a .* [0.5, 0.0, 0.5]
    a3 = a .* [0.5, 0.5, 0.0]
    lattice = M.BravaisLattice(hcat(a1, a2, a3))

    # Conventional cubic coordinates in units of a.
    basis_conv = [
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.00],
        [0.25, 0.00, 0.25],
        [0.00, 0.25, 0.25],
    ]

    # Table S1 local <111> axes, in the matching basis order.
    axes = [
        [-1.0, -1.0, -1.0] ./ sqrt(3.0),
        [ 1.0,  1.0, -1.0] ./ sqrt(3.0),
        [ 1.0, -1.0,  1.0] ./ sqrt(3.0),
        [-1.0,  1.0,  1.0] ./ sqrt(3.0),
    ]

    basis = M.BasisSite[]
    for i in 1:4
        cart = a .* basis_conv[i]
        frac = M.cartesian_to_fractional(lattice, cart)
        push!(basis, M.BasisSite(Symbol("Dy$(i)"), :Dy, frac))
    end

    crystal = M.CrystalStructure(lattice, basis)
    cluster = M.Supercell(crystal, [1, 1, 1]; periodic=true)
    frames = M.LocalFrameField(
        crystal,
        [M.LocalFrame(axis) for axis in axes],
    )

    check = M.validate_local_frames(frames, cluster)
    check[:passed] || error("Pyrochlore local-frame validation failed: $check")

    return (
        lattice=lattice,
        crystal=crystal,
        cluster=cluster,
        frames=frames,
        axes=axes,
        basis_conv=basis_conv,
    )
end

const GEO = build_pyrochlore_geometry()

# Positive local-z moment vectors for sigma_i=+1.
const LOCAL_UNIT_MOMENTS =
    M.frame_moment_vectors(GEO.frames, GEO.cluster; magnitudes=1.0)

# =============================================================================
# 3. Dy3+ magnetic form factor
# =============================================================================

"""
Standard Dy3+ dipole-approximation magnetic form factor.

Q may be supplied in the Phundamental scattering API as a three-component
vector in m^-1. The coefficients use s = |Q|/(4pi) in Angstrom^-1.
"""
function dy3_form_factor(Q)
    q_m_inv = Q isa Number ? abs(Float64(Q)) : norm(Float64.(collect(Q)))
    q_A_inv = q_m_inv * 1e-10
    s = q_A_inv / (4pi)
    s2 = s^2

    # <j0>
    A0, a0 = 0.1157, 15.0732
    B0, b0 = 0.3270,  6.7991
    C0, c0 = 0.5821,  3.0202
    D0 = -0.0249

    # <j2>
    A2, a2 = 0.2523, 18.5172
    B2, b2 = 1.0914,  6.7362
    C2, c2 = 0.9345,  2.2082
    D2 = 0.0250

    j0 = A0 * exp(-a0 * s2) +
         B0 * exp(-b0 * s2) +
         C0 * exp(-c0 * s2) + D0

    j2 = s2 * (
        A2 * exp(-a2 * s2) +
        B2 * exp(-b2 * s2) +
        C2 * exp(-c2 * s2) + D2
    )

    gDy = 4.0 / 3.0
    return j0 + (1.0 - 2.0 / gDy) * j2
end

const FORM_FACTOR = USE_DY_FORM_FACTOR ? dy3_form_factor : (Q -> 1.0)
const NEUTRON_PROBE = O.MagneticNeutronProbe(form_factor=FORM_FACTOR)

# =============================================================================
# 4. Phundamental periodic dipolar/Ewald reference model
# =============================================================================

"""
Build the primitive-cell periodic dipolar-spin-ice scalar Ising coupling matrix.

This is not used to replace the article's paper-specific large-N/random-walk
calculations. It validates that the same physical DSI parameterization can be
represented through the Phundamental long-range API.
"""
function build_periodic_dsi_reference()
    # Phundamental's dipolar tensor kernel has units 1/r^3. Multiplying by
    # D * r_nn^3 yields couplings directly in Kelvin for unit local moments.
    strength = D_DIPOLE_K * R_NN_GEOM^3

    dip = M.interaction_tensor(
        M.DipolarInteraction(strength),
        GEO.cluster;
        method=M.EwaldSummation(
            tolerance=1e-8,
            check_convergence=true,
            strict=true,
            boundary=:tin_foil,
        ),
    )

    validation = M.validate_long_range_result(dip; tolerance=1e-7)
    validation[:passed] || error("Dipolar Ewald validation failed: $validation")

    Jdip = M.ising_coupling_matrix(dip, LOCAL_UNIT_MOMENTS)

    # In a four-site primitive pyrochlore cell each site has two NN bonds to
    # each of the other three sublattices. The Ising exchange per physical
    # nearest-neighbor bond is J/3.
    Jex = zeros(Float64, 4, 4)
    for i in 1:4, j in 1:4
        i == j && continue
        Jex[i, j] = 2.0 * J_NN_K
    end

    return (
        dipolar_result=dip,
        Jdip=Jdip,
        Jexchange=Jex,
        Jtotal=Jdip + Jex,
        validation=validation,
    )
end

const DSI_REFERENCE = build_periodic_dsi_reference()

# =============================================================================
# 5. Figure 1B theory: single tetrahedron + Debye-Huckel monopole gas
# =============================================================================

"""
Single-tetrahedron heat capacity in J mol_Dy^-1 K^-1.

For H = J_eff sum_<ij> sigma_i sigma_j:
  6 ice-rule states:        E = -2 J_eff
  8 3-in/1-out states:      E = 0
  2 all-in/all-out states:  E = 6 J_eff

Morris et al. varied J_eff for their single-tetrahedron comparison. The value
J_eff=1.45 K reproduces the projective-equivalence curve identified explicitly
with Morris Fig. 1 in Castelnovo, Moessner, and Sondhi, PRB 84, 144435 (2011),
Appendix A. The factor 1/2 converts one tetrahedral contribution into heat
capacity per Dy spin because N_tetra=N_Dy/2.
"""
function tetrahedron_heat_capacity(T::Float64; Jeff::Float64=J_EFF_FIG1_K)
    beta = 1.0 / T
    energies = [-2Jeff, 0.0, 6Jeff]
    degeneracy = [6.0, 8.0, 2.0]
    emin = minimum(energies)
    w = degeneracy .* exp.(-beta .* (energies .- emin))
    p = w ./ sum(w)
    U = dot(p, energies)
    E2 = dot(p, energies .^ 2)
    return 0.5 * Rgas * (E2 - U^2) / T^2
end

@inline function noninteracting_monopole_density(T::Float64; Delta::Float64=DELTA_MONOPOLE_K)
    fugacity = 2.0 * exp(-Delta / T)
    return fugacity / (1.0 + fugacity)
end

@inline dh_alpha(T::Float64) = sqrt(3.0 * sqrt(3.0) * pi * E_NN_MONOPOLE_K / (2.0 * T))

"""
Equilibrium monopole density per tetrahedron from the spin-ice Debye-Huckel
self-consistency equation.

This is Eq. (2.13) of Castelnovo, Moessner, and Sondhi, PRB 84, 144435 (2011),
which gives the detailed form of the Debye-Huckel calculation first used for
Morris et al. Fig. 1B. The 2009 calculation uses Delta=4.35 K and the monopole
charge Q=4.28e-13 J/(T m).
"""
function equilibrium_monopole_density(T::Float64; Delta::Float64=DELTA_MONOPOLE_K)
    rho = noninteracting_monopole_density(T; Delta=Delta)
    alpha = dh_alpha(T)
    for _ in 1:100
        x = alpha * sqrt(max(rho, 0.0))
        dressed = Delta / T - (E_NN_MONOPOLE_K / (2.0 * T)) * x / (1.0 + x)
        activity = exp(-dressed)
        rho_new = 2.0 * activity / (1.0 + 2.0 * activity)
        abs(rho_new - rho) <= 1e-13 * max(1.0, rho) && return rho_new
        rho = rho_new
    end
    return rho
end

"""
Debye-Huckel free energy per Dy spin in Kelvin, F/(N_Dy k_B).

The expression is Eq. (2.12) of Castelnovo et al. PRB 84, 144435 (2011), using
the same dumbbell-model monopole parameters as the Morris et al. calculation.
"""
function dh_free_energy_per_spin_K(T::Float64; Delta::Float64=DELTA_MONOPOLE_K)
    rho = equilibrium_monopole_density(T; Delta=Delta)
    alpha = dh_alpha(T)
    x = alpha * sqrt(rho)
    f_creation = 0.5 * rho * Delta
    f_mixing = 0.5 * T * rho * log((rho / 2.0) / (1.0 - rho)) + 0.5 * T * log(1.0 - rho)
    f_coulomb = -T / (3.0 * sqrt(3.0) * pi) * (0.5 * x^2 - x + log1p(x))
    return f_creation + f_mixing + f_coulomb
end

"""
Debye-Huckel heat capacity in J mol_Dy^-1 K^-1.

The thermodynamic relation c_V=-R T d^2[F/(N_Dy k_B)]/dT^2 is evaluated with a
five-point central stencil. The Debye-Huckel description is used only as the
low-temperature monopole theory; Morris et al. state that it breaks down above
approximately 1 K.
"""
function dh_heat_capacity(T::Float64)
    h = max(2e-4, 1e-3 * T)
    T > 2h || (h = T / 4.0)
    fm2 = dh_free_energy_per_spin_K(T - 2h)
    fm1 = dh_free_energy_per_spin_K(T - h)
    f0 = dh_free_energy_per_spin_K(T)
    fp1 = dh_free_energy_per_spin_K(T + h)
    fp2 = dh_free_energy_per_spin_K(T + 2h)
    # Use explicit multiplication: in Julia, `30f0` is the Float32 literal 30.0f0, not `30 * f0`.
    d2 = (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h^2)
    return max(-Rgas * T * d2, 0.0)
end

function make_figure1B()
    T = exp.(range(log(0.20), log(4.00); length=260))
    Ctet = [tetrahedron_heat_capacity(t) for t in T]
    Cdh = [dh_heat_capacity(t) for t in T]
    density = [equilibrium_monopole_density(t) for t in T]
    writedlm(joinpath(OUTPUT_DIR, "Fig1B_theory.csv"), hcat(T, Cdh, Ctet, density), ',')

    fig = Figure(size=(760, 560), fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="Temperature [K]", ylabel="Heat capacity [J/mol_Dy K]", xscale=log10, yscale=log10)
    vspan!(ax, 0.20, 1.00; color=(:lightskyblue, 0.22))
    vspan!(ax, 1.00, 4.00; color=(:khaki, 0.22))
    lines!(ax, T, Cdh; color=:blue, linewidth=3.0, label="Debye-Huckel monopoles")
    lines!(ax, T, Ctet; color=:red, linewidth=2.5, label="single tetrahedron, J_eff=1.45 K")
    text!(ax, 0.27, 3.5; text="Spin ice phase", fontsize=14)
    text!(ax, 1.35, 3.5; text="Paramagnetic phase", fontsize=14)
    ax.xticks = ([0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0], ["0.2", "0.4", "0.6", "0.8", "1", "2", "4"])
    ax.yticks = ([0.01, 0.1, 1.0, 5.0], ["0.01", "0.1", "1", "5"])
    xlims!(ax, 0.20, 4.00)
    ylims!(ax, 1e-2, 5.0)
    axislegend(ax; position=:rb, framevisible=false)
    save(joinpath(OUTPUT_DIR, "Fig1B_theory.png"), fig; px_per_unit=2)
    return fig
end

function validate_figure1_theory()
    c07 = dh_heat_capacity(0.7)
    c10 = dh_heat_capacity(1.0)
    all(isfinite, (c07, c10)) || error("Debye-Huckel heat capacity produced a non-finite value")
    0.0 < c07 < 5.0 || error("Debye-Huckel heat capacity at 0.7 K is outside the paper-scale range: $c07 J/mol_Dy/K")
    0.0 < c10 < 5.0 || error("Debye-Huckel heat capacity at 1.0 K is outside the paper-scale range: $c10 J/mol_Dy/K")
    return (C_0p7_K=c07, C_1p0_K=c10)
end

# =============================================================================
# 6. Figure 2B theory: large-N pyrochlore correlations
# =============================================================================

# Work in conventional cubic fractional coordinates for the Fourier model.
const FCC_PRIMITIVES_REDUCED = hcat(
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
)

const PYRO_BASIS_REDUCED = [
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.00],
    [0.25, 0.00, 0.25],
    [0.00, 0.25, 0.25],
]

const PYRO_AXES = [
    [-1.0, -1.0, -1.0] ./ sqrt(3.0),
    [ 1.0,  1.0, -1.0] ./ sqrt(3.0),
    [ 1.0, -1.0,  1.0] ./ sqrt(3.0),
    [-1.0,  1.0,  1.0] ./ sqrt(3.0),
]

"""
List all nearest-neighbor displacements from basis a to b, in units of the
conventional cubic lattice constant.
"""
function pyrochlore_nn_displacements()
    rnn = sqrt(2.0) / 4.0
    tol = 1e-10
    disp = [Vector{Vector{Float64}}() for _ in 1:4, _ in 1:4]

    for a in 1:4, b in 1:4
        for n1 in -1:1, n2 in -1:1, n3 in -1:1
            R = FCC_PRIMITIVES_REDUCED * Float64[n1, n2, n3]
            dr = PYRO_BASIS_REDUCED[b] .+ R .- PYRO_BASIS_REDUCED[a]
            abs(norm(dr) - rnn) <= tol || continue
            push!(disp[a, b], dr)
        end
    end

    # Every site must see exactly six nearest neighbors.
    counts = [
        sum(length(disp[a, b]) for b in 1:4)
        for a in 1:4
    ]
    all(==(6), counts) ||
        error("Pyrochlore NN geometry failed: coordination counts=$counts")
    return disp
end

const NN_DISPLACEMENTS = pyrochlore_nn_displacements()

"""
Four-sublattice nearest-neighbor Ising interaction matrix in momentum space.

q is a Cartesian vector in dimensionless units where the conventional cubic
lattice constant is one, so an (h,k,l) point has q=2pi(h,k,l).
"""
function largeN_interaction_matrix(q::AbstractVector)
    Jq = zeros(ComplexF64, 4, 4)
    for a in 1:4, b in 1:4
        value = 0.0 + 0.0im
        for dr in NN_DISPLACEMENTS[a, b]
            value += cis(dot(q, dr))
        end
        Jq[a, b] = J_EFF_K * value
    end
    return 0.5 .* (Jq + adjoint(Jq))
end

"""
Self-consistent spherical/large-N multiplier enforcing

    (1/4) <Tr C(q)>_BZ = 1,

with

    C(q) = [lambda I + beta J(q)]^-1.
"""
function solve_largeN_lambda(T::Float64; nk::Int=LARGE_N_BZ_N)
    beta = 1.0 / T

    # Reciprocal primitive vectors for the dimensionless FCC lattice.
    B = 2pi .* inv(transpose(FCC_PRIMITIVES_REDUCED))

    qlist = Vector{Vector{Float64}}()
    mineig = Inf
    coords = ((collect(0:nk-1) .+ 0.5) ./ nk) .- 0.5

    for u in coords, v in coords, w in coords
        q = B * Float64[u, v, w]
        push!(qlist, q)
        mineig = min(mineig, minimum(eigvals(Hermitian(largeN_interaction_matrix(q)))))
    end

    lower = -beta * mineig + 1e-10
    upper = max(lower + 1.0, 10.0)

    function constraint(lambda)
        acc = 0.0
        for q in qlist
            A = Hermitian(
                lambda .* Matrix{ComplexF64}(I, 4, 4) .+
                beta .* largeN_interaction_matrix(q)
            )
            C = inv(A)
            acc += real(tr(C)) / 4.0
        end
        return acc / length(qlist) - 1.0
    end

    # At the lower bound the constraint is positive/divergent; increase upper
    # until it is negative.
    while constraint(upper) > 0.0
        upper *= 2.0
        upper > 1e8 && error("failed to bracket large-N Lagrange multiplier")
    end

    for _ in 1:90
        mid = 0.5 * (lower + upper)
        if constraint(mid) > 0.0
            lower = mid
        else
            upper = mid
        end
    end
    return 0.5 * (lower + upper)
end

function largeN_tensor_at_hkl(
    h::Float64,
    k::Float64,
    l::Float64,
    T::Float64,
    lambda::Float64,
)
    qred = 2pi .* [h, k, l]
    beta = 1.0 / T
    C = inv(Hermitian(
        lambda .* Matrix{ComplexF64}(I, 4, 4) .+
        beta .* largeN_interaction_matrix(qred)
    ))

    tensor = zeros(ComplexF64, 3, 3)
    for a in 1:4, b in 1:4
        tensor .+= C[a, b] .* (
            PYRO_AXES[a] * transpose(PYRO_AXES[b])
        )
    end
    return tensor
end

function _physical_q(h, k, l)
    return (2pi / A_CUBIC) .* Float64[h, k, l]
end

function neutron_intensity_from_tensor(q, tensor)
    # Q=0 is excluded by the neutron transverse projector.
    norm(q) <= 1e-14 && return NaN
    result = O.StaticTensorStructureFactorResult(
        [q],
        reshape(ComplexF64.(tensor), 1, 3, 3),
        Dict{Symbol,Any}(:source => :morris_reproduction),
    )
    scattering = O.scattering_intensity(NEUTRON_PROBE, result)
    return scattering.intensity[1]
end

function largeN_intensity_hkl(h, k, l, T, lambda)
    tensor = largeN_tensor_at_hkl(Float64(h), Float64(k), Float64(l), T, lambda)
    return neutron_intensity_from_tensor(_physical_q(h, k, l), tensor)
end

function plane_map(f, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        value = f(Float64(x), Float64(y))
        Z[iy, ix] = isfinite(value) ? value : 0.0
    end
    return Z
end

function normalize_maps(maps...; quantile_level::Float64=PAPER_COLOR_QUANTILE, ceiling::Float64=1.0)
    finite = Float64[]
    for Z in maps
        append!(finite, filter(x -> isfinite(x) && x >= 0.0, vec(Z)))
    end
    isempty(finite) && return Tuple(copy(Z) for Z in maps)
    scale = quantile(finite, quantile_level)
    scale <= 0.0 && (scale = maximum(finite))
    scale <= 0.0 && return Tuple(zeros(size(Z)) for Z in maps)
    return Tuple(clamp.(Z ./ scale, 0.0, ceiling) for Z in maps)
end

function heatmap_panel!(cell, x, y, Z; xlabel, ylabel, title="", colorrange=(0.0, 1.0))
    ax = Axis(cell; xlabel=xlabel, ylabel=ylabel, title=title, aspect=DataAspect())
    hm = heatmap!(ax, x, y, permutedims(Z); colorrange=colorrange, colormap=PAPER_COLORMAP)
    xlims!(ax, first(x), last(x))
    ylims!(ax, first(y), last(y))
    return ax, hm
end

"""
Render the HK1, 0KL, and HK0 reciprocal-space planes as the stepped three-plane
geometry used by Morris et al. The H axis is symmetric about zero: HK1 occupies
-5 <= H <= 0 at L=1, 0KL is the vertical riser at H=0, and HK0 occupies
0 <= H <= 5 at L=0. The intensity is evaluated at the displayed reciprocal
coordinates; only the color normalization is a visualization transform.
"""
function reciprocal_three_plane_figure(intensity; title::String, filename::String)
    Hminus = collect(range(PAPER_H_RANGE[1], 0.0; length=MAP_N))
    Hplus = collect(range(0.0, PAPER_H_RANGE[2]; length=MAP_N))
    K = collect(range(PAPER_K_RANGE[1], PAPER_K_RANGE[2]; length=MAP_N))
    L = collect(range(PAPER_L_RANGE[1], PAPER_L_RANGE[2]; length=max(41, MAP_N ÷ 3)))

    hk1 = plane_map((h, k) -> intensity(h, k, 1.0), Hminus, K)
    hk0 = plane_map((h, k) -> intensity(h, k, 0.0), Hplus, K)
    _0kl = plane_map((k, l) -> intensity(0.0, k, l), K, L)
    hk1n, hk0n, _0kln = normalize_maps(hk1, hk0, _0kl)

    fig = Figure(size=(1180, 720), fontsize=18)
    ax = Axis3(fig[1, 1]; xlabel="H (r.l.u.)", ylabel="K (r.l.u.)", zlabel="L (r.l.u.)", limits=(PAPER_H_RANGE[1], PAPER_H_RANGE[2], PAPER_K_RANGE[1], PAPER_K_RANGE[2], PAPER_L_RANGE[1], PAPER_L_RANGE[2]), aspect=(2.0, 1.0, 0.34), azimuth=0.18pi, elevation=0.17pi, perspectiveness=0.08, viewmode=:fit)

    z1 = ones(length(Hminus), length(K))
    z0 = zeros(length(Hplus), length(K))
    surface!(ax, Hminus, K, z1; color=permutedims(hk1n), colormap=PAPER_COLORMAP, colorrange=(0.0, 1.0), shading=NoShading)
    surface!(ax, Hplus, K, z0; color=permutedims(hk0n), colormap=PAPER_COLORMAP, colorrange=(0.0, 1.0), shading=NoShading)

    H0 = zeros(length(K), length(L))
    Kgrid = repeat(reshape(K, :, 1), 1, length(L))
    Lgrid = repeat(reshape(L, 1, :), length(K), 1)
    surface!(ax, H0, Kgrid, Lgrid; color=permutedims(_0kln), colormap=PAPER_COLORMAP, colorrange=(0.0, 1.0), shading=NoShading)


    Label(fig[0, 1], title; fontsize=19, font=:bold)
    save(joinpath(OUTPUT_DIR, filename), fig; px_per_unit=2)
    return fig, (HK1=hk1, plane0KL=_0kl, HK0=hk0)
end

function make_figure2B()
    T = 0.7
    @printf("Solving large-N self-consistency at T=%.3f K ...\n", T)
    lambda = solve_largeN_lambda(T)
    @printf("  lambda = %.10f\n", lambda)
    intensity = (h, k, l) -> largeN_intensity_hkl(h, k, l, T, lambda)
    fig, _ = reciprocal_three_plane_figure(intensity; title="Fig. 2B  large-N spin-ice correlations, 0.7 K", filename="Fig2B_largeN_3D.png")

    open(joinpath(OUTPUT_DIR, "Fig2B_largeN_metadata.txt"), "w") do io
        println(io, "temperature_K = $T")
        println(io, "lambda = $lambda")
        println(io, "J_eff_K = $J_EFF_K")
        println(io, "BZ_quadrature_n = $LARGE_N_BZ_N")
        println(io, "Dy_form_factor = $USE_DY_FORM_FACTOR")
        println(io, "planes = HK0, HK1, 0KL")
        println(io, "H_range = -5:5")
        println(io, "HK1_H_range = -5:0")
        println(io, "HK0_H_range = 0:5")
        println(io, "K_range = 0:5")
        println(io, "L_range = 0:1")
        println(io, "display_colormap = $(PAPER_COLORMAP)")
        println(io, "display_upper_quantile = $(PAPER_COLOR_QUANTILE)")
    end
    return fig
end

# =============================================================================
# 7. Figures 3C / 4B / 4C / 4D: analytic random-walk string scattering
# =============================================================================

# Directed diamond-lattice steps for strings advancing along +[001].
# Components are in units of the conventional cubic lattice constant.
const WALK_STEPS_ODD = (
    [ 1.0, -1.0, 1.0] ./ 4.0,
    [-1.0,  1.0, 1.0] ./ 4.0,
)
const WALK_STEPS_EVEN = (
    [ 1.0,  1.0, 1.0] ./ 4.0,
    [-1.0, -1.0, 1.0] ./ 4.0,
)

"""
Exact independent-random-walk string tensor for one q point.

No Monte Carlo noise is introduced. The recurrence propagates the first mixed
moment B=<A P*> and second moment C=<A A^†>, where A is the accumulated
magnetic scattering amplitude and P is the phase at the current tetrahedron.

At each link there are exactly two possible forward choices, as described in
the paper. `bias` is the probability of the first branch.
"""
function random_walk_tensor_hkl(
    h::Float64,
    k::Float64,
    l::Float64;
    bias::Float64=0.5,
    length::Int=STRING_LENGTH,
)
    0.0 <= bias <= 1.0 || throw(ArgumentError("bias must lie in [0,1]"))
    length > 0 || throw(ArgumentError("string length must be positive"))

    qred = 2pi .* [h, k, l]
    probs = (bias, 1.0 - bias)

    # B = <A conj(P)> and C = <A A^†>.
    B = zeros(ComplexF64, 3)
    C = zeros(ComplexF64, 3, 3)

    for n in 1:length
        steps = isodd(n) ? WALK_STEPS_ODD : WALK_STEPS_EVEN

        mean_v = zeros(ComplexF64, 3)
        mean_vv = zeros(ComplexF64, 3, 3)
        mean_conj_w = 0.0 + 0.0im
        mean_v_conj_w = zeros(ComplexF64, 3)

        for branch in 1:2
            p = probs[branch]
            d = steps[branch]
            u = d ./ norm(d)

            # Reversing a field-aligned spin changes its moment by -2u.
            # The common factor of 2 affects only absolute normalization, which
            # Morris compares in arbitrary diffuse-scattering intensity units.
            moment = -2.0 .* u

            phase_mid = cis(-0.5 * dot(qred, d))
            w = cis(-dot(qred, d))
            v = ComplexF64.(moment) .* phase_mid

            mean_v .+= p .* v
            mean_vv .+= p .* (v * adjoint(v))
            mean_conj_w += p * conj(w)
            mean_v_conj_w .+= p .* v .* conj(w)
        end

        C .+= B * adjoint(mean_v) +
              mean_v * adjoint(B) +
              mean_vv

        B = B .* mean_conj_w .+ mean_v_conj_w
    end

    return 0.5 .* (C + adjoint(C))
end

function random_walk_intensity_hkl(h, k, l; bias, length=STRING_LENGTH)
    tensor = random_walk_tensor_hkl(
        Float64(h), Float64(k), Float64(l);
        bias=Float64(bias),
        length=Int(length),
    )
    return neutron_intensity_from_tensor(_physical_q(h, k, l), tensor)
end

function random_walk_three_planes(; bias::Float64, title::String, filename::String)
    intensity = (h, k, l) -> random_walk_intensity_hkl(h, k, l; bias=bias)
    fig, _ = reciprocal_three_plane_figure(intensity; title=title, filename=filename)
    return fig
end

function make_figure3C()
    return random_walk_three_planes(bias=BIAS_FIG3, title="Fig. 3C  random-walk strings, 5/7 h_S, bias 0.53:0.47", filename="Fig3C_random_walk_3D.png")
end

function make_figure4B()
    return random_walk_three_planes(bias=BIAS_FIG4_HIGH, title="Fig. 4B  tilted-field strings, 4/7 h_S, bias 0.80:0.20", filename="Fig4B_biased_random_walk_3D.png")
end

function make_figure4C()
    k = collect(range(0.0, 4.0; length=MAP_N))
    l = collect(range(0.0, 1.0; length=max(51, MAP_N ÷ 2)))
    cases = [(BIAS_FIG4_MID, "2/7 h_S   bias 0.64:0.36"), (BIAS_FIG4_HIGH, "4/7 h_S   bias 0.80:0.20")]
    raw = [plane_map((kk, ll) -> random_walk_intensity_hkl(1.0, kk, ll; bias=bias), k, l) for (bias, _) in cases]
    maps = normalize_maps(raw...)

    fig = Figure(size=(1120, 360), fontsize=16)
    Label(fig[0, 1:2], "Fig. 4C  field evolution in the (1,K,L) plane"; fontsize=19, font=:bold)
    for (col, ((_, label), Z)) in enumerate(zip(cases, maps))
        heatmap_panel!(fig[1, col], k, l, Z; xlabel="K (r.l.u.)", ylabel="L (r.l.u.)", title=label)
    end
    save(joinpath(OUTPUT_DIR, "Fig4C_field_dependence.png"), fig; px_per_unit=2)
    return fig
end

function make_figure4D()
    h = collect(range(0.0, 4.0; length=MAP_N))
    k = collect(range(0.0, 1.0; length=max(61, MAP_N ÷ 2)))

    # Morris Fig. 4D defines the wall as (h,k,2eta+k); eta=0 gives (h,k,k).
    Z = plane_map((hh, kk) -> random_walk_intensity_hkl(hh, kk, kk; bias=BIAS_FIG4_HIGH), h, k)
    Zn = normalize_maps(Z)[1]

    fig = Figure(size=(920, 360), fontsize=16)
    ax, hm = heatmap_panel!(fig[1, 1], h, k, Zn; xlabel="H (r.l.u.)", ylabel="K=L (r.l.u.)", title="Fig. 4D  (H,K,K) diffuse wall, eta=0")
    Colorbar(fig[1, 2], hm; label="normalized intensity", width=16)
    save(joinpath(OUTPUT_DIR, "Fig4D_diffuse_wall_eta0.png"), fig; px_per_unit=2)
    return fig
end

# =============================================================================
# 8. Supplemental random-walk slices S3/S5/S7
# =============================================================================

function make_supplement_random_walk_slices()
    h = collect(range(-0.5, 4.0; length=MAP_N))
    k = collect(range(-5.0, 0.0; length=MAP_N))
    Lvalues = [0.0, 0.2, 0.3]
    raw = [plane_map((H, K) -> random_walk_intensity_hkl(H, K, abs(L) < 1e-12 ? 1e-8 : L; bias=BIAS_FIG3), h, k) for L in Lvalues]
    maps = Tuple(normalize_maps(Z)[1] for Z in raw)

    fig = Figure(size=(1320, 430), fontsize=15)
    Label(fig[0, 1:3], "Supporting Figs. S3 / S5 / S7  random-walk model"; fontsize=18, font=:bold)
    for (col, (L, Z)) in enumerate(zip(Lvalues, maps))
        heatmap_panel!(fig[1, col], h, k, Z; xlabel="H (r.l.u.)", ylabel="K (r.l.u.)", title="L=$(L)")
    end
    save(joinpath(OUTPUT_DIR, "Supplement_S3_S5_S7_random_walk.png"), fig; px_per_unit=2)
    return fig
end

# =============================================================================
# 9. Derived ideal Kasteleyn boundary
# =============================================================================

"""
Ideal [001] Kasteleyn boundary derived from the free-energy-per-link argument in Morris et al. This is not Morris Eq. (1); Eq. (1) is the tilted-field step-probability ratio.
"""
kasteleyn_field(T::Float64) =
    (sqrt(3.0) / 2.0) * kB * T * log(2.0) / MU_DY

function make_kasteleyn_figure()
    T = collect(range(0.3, 1.1; length=200))
    B = kasteleyn_field.(T)

    fig = Figure(size=(900, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Field B_K (T)",
        ylabel="Temperature (K)",
        title="Derived ideal 3D Kasteleyn boundary",
    )
    lines!(ax, B, T; linewidth=2.5, label="f_link = 0")
    axislegend(ax; position=:lt)
    save(joinpath(OUTPUT_DIR, "Kasteleyn_line.png"), fig; px_per_unit=2)
    writedlm(joinpath(OUTPUT_DIR, "Kasteleyn_line.csv"), hcat(T, B), ',')
    return fig
end

# =============================================================================
# 10. Parameter/provenance report
# =============================================================================

function write_parameter_report()
    path = joinpath(OUTPUT_DIR, "parameters.txt")
    open(path, "w") do io
        println(io, "Morris et al. Dy2Ti2O7 reproduction")
        println(io, "DOI = 10.1126/science.1178868")
        println(io)
        println(io, "ARTICLE/SUPPLEMENT")
        println(io, "mu_Dy/muB = ", MU_DY / muB)
        println(io, "Q_monopole_J_per_T_m = ", Q_MONOPOLE)
        println(io, "Delta_monopole_K = ", DELTA_MONOPOLE_K)
        println(io, "string_length = ", STRING_LENGTH)
        println(io, "Fig3C_bias = ", BIAS_FIG3, ":", 1-BIAS_FIG3)
        println(io, "Fig4_mid_bias = ", BIAS_FIG4_MID, ":", 1-BIAS_FIG4_MID)
        println(io, "Fig4_high_bias = ", BIAS_FIG4_HIGH, ":", 1-BIAS_FIG4_HIGH)
        println(io)
        println(io, "MICROSCOPIC PARAMETERIZATION REFERENCED THROUGH S9/S10")
        println(io, "J_K = ", J_EXCHANGE_K)
        println(io, "D_K = ", D_DIPOLE_K)
        println(io, "J_nn_K = ", J_NN_K)
        println(io, "D_nn_K = ", D_NN_K)
        println(io, "J_eff_microscopic_K = ", J_EFF_K)
        println(io, "J_eff_Fig1_single_tetrahedron_K = ", J_EFF_FIG1_K)
        println(io, "E_nn_monopole_K = ", E_NN_MONOPOLE_K)
        println(io)
        println(io, "CONSISTENCY CHECKS")
        println(io, "Delta_from_J_D_K = ", DELTA_FROM_JD_K)
        println(io, "D_from_mu_and_geometry_K = ", D_FROM_GEOMETRY_K)
        println(io, "a_diamond_A = ", A_DIAMOND * 1e10)
        println(io, "a_cubic_A = ", A_CUBIC * 1e10)
        println(io, "r_nn_A = ", R_NN * 1e10)
        println(io)
        println(io, "PHUNDAMENTAL EWALD")
        for (k, v) in sort(collect(DSI_REFERENCE.validation); by=x -> string(x[1]))
            println(io, k, " = ", v)
        end
        println(io)
        println(io, "NOT REPRODUCED")
        println(io, "- experimental neutron panels: original 3D data not contained in PDFs")
        println(io, "- experimental magnetization curves: raw sweeps not contained in PDFs")
        println(io, "- experimental heat-capacity points: raw table not contained in PDFs")
        println(io, "- COMSOL demagnetization field map: external FEM calculation")
        println(io, "- Fig. 4D inter-string correlation enhancement: independent-walk model only")
        println(io)
        println(io, "FUTURE CORE NUMERICS")
        println(io, "- particle-mesh Ewald / P3M remains future work")
        println(io, "- parallel tempering / replica exchange remains future work")
    end
    return path
end

# =============================================================================
# 11. Main
# =============================================================================

function main()
    println("Morris et al. Dy2Ti2O7 reproduction")
    println(repeat("=", 72))
    println("Output directory: ", OUTPUT_DIR)
    println()

    @printf("mu_Dy               = %.6f mu_B\n", MU_DY / muB)
    @printf("Q                    = %.6e J/(T m)\n", Q_MONOPOLE)
    @printf("a_diamond            = %.6f A\n", A_DIAMOND * 1e10)
    @printf("a_cubic              = %.6f A\n", A_CUBIC * 1e10)
    @printf("r_nn                 = %.6f A\n", R_NN * 1e10)
    @printf("J, D                 = %.4f K, %.4f K\n", J_EXCHANGE_K, D_DIPOLE_K)
    @printf("J_nn, D_nn, J_eff    = %.4f K, %.4f K, %.4f K\n", J_NN_K, D_NN_K, J_EFF_K)
    @printf("Delta(J,D)           = %.6f K (supplement: %.2f K)\n", DELTA_FROM_JD_K, DELTA_MONOPOLE_K)
    @printf("D(mu,r_nn)           = %.6f K\n", D_FROM_GEOMETRY_K)
    @printf("E_nn(monopoles)      = %.6f K\n", E_NN_MONOPOLE_K)
    println()

    println("[1/7] Figure 1B theoretical heat capacities")
    fig1_check = validate_figure1_theory()
    @printf("  DH C(0.7 K)         = %.6f J/mol_Dy/K\n", fig1_check.C_0p7_K)
    @printf("  DH C(1.0 K)         = %.6f J/mol_Dy/K\n", fig1_check.C_1p0_K)
    make_figure1B()

    println("[2/7] Figure 2B large-N pinch-point calculation")
    make_figure2B()

    println("[3/7] Figure 3C weakly biased [001] random-walk scattering")
    make_figure3C()

    println("[4/7] Figure 4B tilted-field 0.8:0.2 random walk")
    make_figure4B()

    println("[5/7] Figures 4C/4D field-dependent sheets")
    make_figure4C()
    make_figure4D()

    println("[6/7] Supplemental S3/S5/S7 random-walk slices")
    make_supplement_random_walk_slices()

    println("[7/7] Derived Kasteleyn boundary and provenance report")
    make_kasteleyn_figure()
    report = write_parameter_report()

    println()
    println(repeat("-", 72))
    println("Finished.")
    println("Figures and tables saved under:")
    println("  ", OUTPUT_DIR)
    println("Parameter report:")
    println("  ", report)
end

main()
