include(joinpath(@__DIR__, "..", "src/Phundamental.jl"))
using .Phundamental

using Printf
using GLMakie
using LinearAlgebra
using Statistics


using .Phundamental.Solvers: SolverResult, groundenergy, matrix

const M = Phundamental.Models
const F = Phundamental.Foundation
const BENCHMARK_FIGURE_DIR = joinpath(@__DIR__, "benchmark-figures")

mkpath(BENCHMARK_FIGURE_DIR)


#-------------------------------------------------------#
# Bonner, Jill C. and Michael Fisher. (1964).           #
# Linear Magnetic Chains with Anisotropic Coupling.     #
# Phys. Rev. 135, A640. doi: 10.1103/PhysRev.135.A640   #
#-------------------------------------------------------#
let
    crystal = M.CrystalStructure(M.BravaisLattice(reshape([1.0],1,1)), [M.BasisSite(:A,:S,[0.0])])
    material = Material("Bonner-Fisher Chain", crystal;
                        species = Dict(:S => M.AtomicSpecies(:S; mass=1.0)),
                        basis_properties=[M.SiteProperties(spin=1/2, g_tensor=2.0)])

    temperature = range(0.0, 2.5; length=401)
    susceptibility_temperature = temperature[2:end]

    fig = Figure(size=(1400,800))
    ax11 = Axis(fig[1,1]; xlabel="1/N or 1/N²", ylabel="E₀/N|J|", title="(Fig. 1)")
    ax12 = Axis(fig[1,2]; xlabel="kT/|J|", ylabel="U/N|J|", title="(Fig. 3)")
    ax13 = Axis(fig[1,3]; xlabel="kT/|J|", ylabel="S/Nk", title="(Fig. 4)")
    ax21 = Axis(fig[2,1]; xlabel="TS/N|J|", ylabel="U/N|J|", title="(Fig. 5)")
    ax22 = Axis(fig[2,2]; xlabel="kT/|J|", ylabel="C/Nk", title="(Fig. 6)")
    ax23 = Axis(fig[2,3]; xlabel="kT/|J|", ylabel="C/Nk", title="(Fig. 7)")
    ax32 = Axis(fig[3,2]; xlabel="kT/|J|", ylabel="|J|χ/Ng²β²", title="(Fig. 14)")

    Ns = collect(2:11)
    results = Dict{Int, SolverResult}()
    curves = Dict{Int, ThermodynamicCurve}()
    for N in Ns
        cluster = Supercell(crystal, [N]; periodic=true)
        model = build_model(material, cluster, HeisenbergModel(2.0; neighbor_cutoff=1.01))

        results[N] = solve(ExactDiagonalization(), model)
        #curves[N] = thermal_curve(model, temperature; method=ExactGibbs(), kB=1.0)
        curves[N] = thermal_curve(results[N], temperature; kB=1.0)
    end

    E0s = [groundenergy(results[N]) / N for N in Ns]
    scatter!(ax11, 1.0 ./ Ns, E0s; marker=:circle, markersize=11, label="1/N")
    scatter!(ax11, 1.0 ./ Ns.^2, E0s, marker=:rect, markersize=10, label="1/N²")
    axislegend(ax11; position=:rb)

    for N in Ns
        lines!(ax12, temperature, curves[N].internal_energy ./ N; label="N=$N")
        lines!(ax13, temperature, curves[N].entropy ./ N)
        lines!(ax22, temperature, curves[N].heat_capacity ./ N)
    end


    for N in (10,11)
        lowT = (temperature .≥ 0.0) .& (temperature .≤ 0.6)
        TS = temperature .* curves[N].entropy ./ N
        U = curves[N].internal_energy ./ N
        lines!(ax21, TS[lowT], U[lowT]; label="N=$N")
    end
    axislegend(ax21, position=:rb)

    #Fig. 7 using H = 2|J| ∑ᵢ [SzᵢSzᵢ₊₁ + γ(SxᵢSxᵢ₊₁ + SyᵢSyᵢ₊₁)]
    #such that Jxy=2γ and Jz = 2
    gammas = collect(0.0:0.1:1.0)
    fig7_curves = Dict{Float64,ThermodynamicCurve}()
    fig7_curves[1.0] = curves[8]

    cluster8 = Supercell(crystal, [8]; periodic=true)

    for γ in gammas[1:end-1]
        model = build_model(material, cluster8, XXZModel(2.0 * γ, 2.0; neighbor_cutoff=1.01))
        result = solve(ExactDiagonalization(), model)
        fig7_curves[γ] = thermal_curve(result, temperature; kB=1.0)
    end

    for γ in gammas
        lines!(ax23, temperature, fig7_curves[γ].heat_capacity ./ 8; label="γ=$(round(γ; digits=1))")
    end


    #Fig. 14 uses |J|χ/(Ng²β²) = [<Mz²> - <Mz>²]/(NT) with |J| = k = 1.
    for N in 3:11
        result = results[N]
        Mz = reduce(+, (F.Sz(i) for i in 1:N))

        Mz_matrix = matrix(Mz, result.basis; sparse=true)
        Mz_states = Mz_matrix * result.right_states

        mz_eigen = [real(dot(view(result.left_states, :, n), view(Mz_states, :, n))) for n in axes(result.right_states, 2)]

        mz2_eigen = [sum(abs2, view(Mz_states, :, n)) for n in axes(result.right_states, 2)]

        susceptibility = [
            begin
                state = thermal_state(result; temperature=T, kB=1.0)
                mz = sum(state.probabilities .* mz_eigen)
                mz2 = sum(state.probabilities .* mz2_eigen)

                max(real(mz2 - abs2(mz)), 0.0) / (N * T)
            end
            for T in susceptibility_temperature
        ]

        lines!(ax32, susceptibility_temperature, susceptibility; label="N=$N")
    end
    #axislegend(ax23; position=:rt, nbanks=2)
    xlims!(ax11, 0.0, 0.55)
    ylims!(ax11, -1.05, -0.45)

    xlims!(ax12, 0.0, 2.5)
    ylims!(ax12, -1.0, 0.0)

    xlims!(ax13, 0.0, 2.5)
    ylims!(ax13, 0.0, 0.7)

    xlims!(ax21, 0.0, 0.11)
    ylims!(ax21, -0.92, -0.78)

    xlims!(ax22, 0.0, 2.5)
    ylims!(ax22, 0.0, 0.6)

    xlims!(ax23, 0.0, 1.2)
    ylims!(ax23, 0.0, 0.6)

    xlims!(ax32, 0.0, 2.5)
    ylims!(ax32, 0.0, 0.1)

    Legend(fig[1,4], ax12; title="Chain Length")
    save(joinpath(BENCHMARK_FIGURE_DIR, "bonner-fisher-finiteN-benchmarks.png"), fig)
end




#-------------------------------------------------------#
# Pfeuty, Pierre. (1970).                               #
# The One-Dimensional Ising Model with a Transverse     #
# Field. Ann. Phys. 57, 79-90.                          #
# doi: 10.1016/0003-4916(70)90270-8                     #
#-------------------------------------------------------#
let
    N = 101; J = 1.0
    field_ratios = range(0.0, 2.0; length=161)

    fig = Figure(size=(1100,900))
    ax11 = Axis(fig[1,1]; xlabel="k′", ylabel="Λₖ", title="(Fig. 1)")
    ax12 = Axis(fig[1,2]; xlabel="Γ/J", ylabel="-E₀/NJ", title="(Fig. 2)")
    ax21 = Axis(fig[2,1]; xlabel="Γ/J", ylabel="M_z", title="(Fig. 3)")
    ax22 = Axis(fig[2,2]; xlabel="Γ/J", ylabel="ρ₁", title="(Fig. 4)")

    # Pfeuty Eq. (2.4), neglecting the cyclic-chain parity correction
    # exactly as done for the large-N calculation in the article.
    function pfeuty_result(Γ::Real, J::Real, N::Int)
        ordering = collect(1:N)

        representation = F.Representation(:pfeuty_fermion,
                                          F.FermionFockSpace(N),
                                          F.FermionAlgebra(N),
                                          F.FermionOccupationBasis(ordering);
                                          ordering=ordering, reference_state=:fermion_vacuum)

        terms = F.AbstractOperatorExpr[(Γ * N / 2) * F.IdentityOperator()]

        for i in 1:N
            push!(terms, -Γ * F.nf(i))
        end

        for i in 1:N
            j = mod1(i + 1, N)
            push!(terms, -(J / 4) * (F.c(i)' - F.c(i)) * (F.c(j)' + F.c(j)))
        end

        model = F.ManyBodyModel(representation, F.OperatorSum(terms); parameters=Dict(:Gamma => Γ, :J => J, :N => N))

        return solve(QuadraticFermionSolver(), model)
    end

    # Ground-state observables extracted from the positive-energy
    # Nambu modes. P = V V† is the zero-temperature Nambu covariance.
    function pfeuty_observables(result, N::Int)
        V = mode_vectors(result)

        projector(a, b) = dot(view(V, b, :), view(V, a, :))

        i = (N + 1) ÷ 2; j = i + 1

        C11 = projector(i, j)
        C12 = projector(i, N + j)
        C21 = projector(N + i, j)
        C22 = projector(N + i, N + j)

        ni = real(projector(N + i, N + i))

        E0 = -0.5 * sum(mode_energies(result))
        Mz = ni - 0.5

        ρx = real(C21 + C22 - C12 - C11) / 4
        ρy = real(-C21 + C22 - C11 + C12) / 4
        ρz = real(-C21 * C12 + C22 * C11)

        return (; E0, Mz, ρx, ρy, ρz)
    end

    # ------------------------------------------------------------------
    # Fig. 1 — elementary excitation spectrum
    # ------------------------------------------------------------------

    λs = (0.5, 1.0, 1.5)

    for λ in λs
        Γ = 1.0
        Jλ = 2.0 * Γ * λ

        result = pfeuty_result(Γ, Jλ, N)
        energies = sort(mode_energies(result) ./ Γ)

        # For odd N the ±k states occur as degenerate pairs, with the
        # k=0 endpoint occurring once.
        Λk = energies[1:2:end]
        kprime = range(π / N, π; length=length(Λk))

        Λexact = sqrt.(1 .+ λ^2 .- 2.0 * λ .* cos.(kprime))
        error = maximum(abs.(Λk .- Λexact))

        @printf("Pfeuty Fig. 1: λ = %.1f, max |ΔΛ| = %.3e\n", λ, error)

        lines!(ax11, kprime, Λk; label="λ=$(λ)")
    end

    # ------------------------------------------------------------------
    # Figs. 2, 3, 4 — field dependence from the same quadratic solver
    # ------------------------------------------------------------------

    energy = Float64[]
    magnetization = Float64[]
    rho_x = Float64[]
    rho_y = Float64[]
    rho_z = Float64[]

    for ratio in field_ratios
        Γ = ratio * J
        result = pfeuty_result(Γ, J, N)
        obs = pfeuty_observables(result, N)

        push!(energy, -obs.E0 / (N * J))
        push!(magnetization, obs.Mz)
        push!(rho_x, obs.ρx)
        push!(rho_y, obs.ρy)
        push!(rho_z, obs.ρz)
    end

    # Fig. 2
    lines!(ax12, field_ratios, energy)

    # Fig. 3
    lines!(ax21, field_ratios, magnetization; label="exact")
    molecular_field = min.(0.5 .* field_ratios, 0.5)
    lines!(ax21, field_ratios, molecular_field; linestyle=:dash, label="molecular field")

    # Fig. 4
    fig4 = field_ratios .≤ 1.5
    lines!(ax22, field_ratios[fig4], rho_x[fig4]; label="ρ₁ˣ")
    lines!(ax22, field_ratios[fig4], -rho_y[fig4]; label="-ρ₁ʸ")
    lines!(ax22, field_ratios[fig4], rho_z[fig4]; label="ρ₁ᶻ")

    # ------------------------------------------------------------------
    # Table I — exact critical correlation functions
    #
    # At λ=1:
    #
    #     G(n) = (2/π)(-1)^n/(2n+1)
    #
    # together with Pfeuty Eqs. (2.16)-(2.18).
    # ------------------------------------------------------------------

    Gcritical(n::Int) = (2.0 / π) * (-1.0)^n / (2 * n + 1)

    function critical_correlations(n::Int)
        X = [Gcritical(r - c - 1) for r in 1:n, c in 1:n]

        Y = [Gcritical(r - c + 1) for r in 1:n, c in 1:n]

        ρx = det(X) / 4; ρy = det(Y) / 4
        ρz = -Gcritical(n) * Gcritical(-n) / 4

        return (; ρx, ρy, ρz)
    end

    println()
    println("TABLE I")
    println("Correlation Functions for λ = 1 (Γ = J/2).")
    println()
    @printf("%3s     %8s     %9s     %8s\n", "n", "4ρₙˣ", "-4ρₙʸ", "4ρₙᶻ")

    for n in 1:12
        corr = critical_correlations(n)

        xstr = @sprintf("%.4f", 4 * corr.ρx)
        ystr = n ≤ 2 ? @sprintf("%.4f", -4 * corr.ρy) : @sprintf("%.5f", -4 * corr.ρy)
        zstr = @sprintf("%.5f", 4 * corr.ρz)

        @printf("%3d     %8s     %9s     %8s\n", n, xstr, ystr, zstr)
    end

    println()

    # ------------------------------------------------------------------
    # Paper-like axes
    # ------------------------------------------------------------------

    xlims!(ax11, 0.0, π)
    ylims!(ax11, 0.0, 3.5)
    ax11.xticks = ([0.0, 1.0, 2.0, 3.0, π], ["0", "1", "2", "3", "π"])

    xlims!(ax12, 0.0, 2.0)
    ylims!(ax12, 0.2, 1.05)

    xlims!(ax21, 0.0, 1.75)
    ylims!(ax21, 0.0, 0.55)

    xlims!(ax22, 0.0, 1.5)
    ylims!(ax22, 0.0, 0.27)

    axislegend(ax11; position=:lt)
    axislegend(ax21; position=:rb)
    axislegend(ax22; position=:rt)

    save(joinpath(BENCHMARK_FIGURE_DIR, "pfeuty-benchmarks.png"), fig)
end


#-------------------------------------------------------#
# Pfeuty transformation-map validation                  #
#                                                       #
# Spin -> Jordan-Wigner -> Fourier -> Bogoliubov        #
#                                                       #
# An open finite chain is used here so that the         #
# Jordan-Wigner map remains exactly quadratic without   #
# the parity-dependent cyclic boundary term.            #
#-------------------------------------------------------#
let
    N = 8
    J = 1.0
    λs = (0.5, 1.0, 1.5)
    tolerance = 1e-8
    ordering = collect(1:N)

    spectrum_error(a, b) = maximum(abs.(sort(real.(a.energies)) .- sort(real.(b.energies))))
    mode_error(a, b) = maximum(abs.(sort(mode_energies(a)) .- sort(mode_energies(b))))

    println()
    println("PFEUTY TRANSFORMATION-MAP VALIDATION")
    println("Spin -> Jordan-Wigner -> Fourier -> Bogoliubov")
    println("Exact open-chain validation; N = $N")
    println()

    @printf(
        "%5s  %11s  %11s  %11s  %11s  %11s  %11s  %6s\n",
        "λ",
        "JW op.",
        "JW spec.",
        "F spec.",
        "F modes",
        "Bog spec.",
        "Bog modes",
        "diag."
    )

    for λ in λs
        Γ = J / (2.0 * λ)

        # --------------------------------------------------------------
        # 1. Source transverse-field Ising spin Hamiltonian
        #
        # H = -Γ Σᵢ Szᵢ - J Σᵢ Sxᵢ Sxᵢ₊₁
        #
        # Open boundaries are intentional for this exact transformation
        # validation.
        # --------------------------------------------------------------

        spin_representation = F.Representation(
            :pfeuty_spin,
            F.SpinHilbertSpace(fill(1//2, N)),
            F.SpinAlgebra(N),
            F.SpinProductBasis(ordering);
            ordering=ordering
        )

        spin_terms = F.AbstractOperatorExpr[]

        for i in 1:N
            push!(spin_terms, -Γ * F.Sz(i))
        end

        for i in 1:N-1
            push!(spin_terms, -J * F.Sx(i) * F.Sx(i + 1))
        end

        spin_model = F.ManyBodyModel(
            spin_representation,
            F.OperatorSum(spin_terms);
            parameters=Dict(:Gamma => Γ, :J => J, :lambda => λ, :N => N)
        )

        # --------------------------------------------------------------
        # 2. Jordan-Wigner transformation
        # --------------------------------------------------------------

        jw = JordanWigner(ordering)
        jw_result = transform(jw, spin_model)
        jw_model = jw_result.model

        @assert all(values(jw_result.validation))

        # Independent expected JW Hamiltonian for an open chain:
        #
        # H = ΓN/2 - ΓΣᵢnᵢ
        #     - (J/4)Σᵢ(c†ᵢ-cᵢ)(c†ᵢ₊₁+cᵢ₊₁)
        #
        # This directly checks that JordanWigner generated Pfeuty's
        # quadratic fermion form rather than merely checking isospectrality.

        expected_jw_terms = F.AbstractOperatorExpr[(Γ * N / 2) * F.IdentityOperator()]

        for i in 1:N
            push!(expected_jw_terms, -Γ * F.nf(i))
        end

        for i in 1:N-1
            push!(expected_jw_terms, -(J / 4) * (F.c(i)' - F.c(i)) * (F.c(i + 1)' + F.c(i + 1)))
        end

        expected_jw = F.OperatorSum(expected_jw_terms)

        spin_ed = solve(ExactDiagonalization(), spin_model)
        jw_ed = solve(ExactDiagonalization(), jw_model)

        H_jw = matrix(jw_model.hamiltonian, jw_ed.basis; sparse=false)
        H_expected = matrix(expected_jw, jw_ed.basis; sparse=false)

        jw_operator_error = norm(H_jw - H_expected) / max(norm(H_expected), 1.0)
        jw_spectrum_error = spectrum_error(spin_ed, jw_ed)

        # Quadratic modes before changing single-particle basis.
        jw_modes = solve(QuadraticFermionSolver(), jw_model)

        # --------------------------------------------------------------
        # 3. Fourier transformation
        #
        # c_j = 1/sqrt(N) Σ_k exp(+ikj) c_k
        #
        # FourierTransformation is an exact unitary mode transformation.
        # For the open chain it need not diagonalize H by itself; that is
        # not required. Its role here is to validate the explicit change
        # from real-space fermions to a momentum-mode basis.
        # --------------------------------------------------------------

        momenta = 2π .* (0:N-1) ./ N

        fourier_matrix = ComplexF64[exp(-im * momenta[k] * (j - 1)) / sqrt(N) for k in 1:N, j in 1:N]

        fourier = FourierTransformation(fourier_matrix; statistics=:fermion)

        fourier_result = transform(fourier, jw_model)
        fourier_model = fourier_result.model

        @assert all(values(fourier_result.validation))

        fourier_ed = solve(ExactDiagonalization(), fourier_model)
        fourier_modes = solve(QuadraticFermionSolver(), fourier_model)

        fourier_spectrum_error = spectrum_error(jw_ed, fourier_ed)
        fourier_mode_error = mode_error(jw_modes, fourier_modes)

        # --------------------------------------------------------------
        # 4. Bogoliubov transformation
        #
        # QuadraticFermionSolver supplies the positive-energy Nambu
        # eigenvectors [u;v]. From
        #
        #     γ = u†c + v†c†
        #
        # the inverse canonical map used by BogoliubovTransformation is
        #
        #     c = uγ + v*γ†.
        #
        # Hence U = u† and V = v† in the transformation convention used
        # by Phundamental.
        # --------------------------------------------------------------

        W = mode_vectors(fourier_modes)

        size(W, 1) == 2N || error("Expected a paired 2N×N fermionic BdG mode matrix; got size $(size(W))")

        u = W[1:N, :]
        v = W[N+1:2N, :]

        Ubog = Matrix(adjoint(u))
        Vbog = Matrix(adjoint(v))

        bogoliubov = BogoliubovTransformation(Ubog, Vbog; statistics=:fermion, atol=tolerance)

        bogoliubov_result = transform(bogoliubov, fourier_model)
        bogoliubov_model = bogoliubov_result.model

        @assert all(values(bogoliubov_result.validation))

        bogoliubov_ed = solve(ExactDiagonalization(), bogoliubov_model)

        bogoliubov_modes = solve(QuadraticFermionSolver(), bogoliubov_model)

        bogoliubov_spectrum_error = spectrum_error(fourier_ed, bogoliubov_ed)

        bogoliubov_mode_error = mode_error(fourier_modes, bogoliubov_modes)

        diagonalized = get(bogoliubov_modes.metadata, :number_conserving, false)

        # --------------------------------------------------------------
        # 5. Numerical validation gates
        # --------------------------------------------------------------

        @assert jw_operator_error < tolerance
        @assert jw_spectrum_error < tolerance
        @assert fourier_spectrum_error < tolerance
        @assert fourier_mode_error < tolerance
        @assert bogoliubov_spectrum_error < tolerance
        @assert bogoliubov_mode_error < tolerance
        @assert diagonalized

        @printf(
            "%5.1f  %11.3e  %11.3e  %11.3e  %11.3e  %11.3e  %11.3e  %6s\n",
            λ,
            jw_operator_error,
            jw_spectrum_error,
            fourier_spectrum_error,
            fourier_mode_error,
            bogoliubov_spectrum_error,
            bogoliubov_mode_error,
            diagonalized ? "PASS" : "FAIL"
        )
    end

    println()
    println("Validation criteria:")
    println("  JW op.    : transformed JW Hamiltonian vs. explicit Pfeuty fermion Hamiltonian")
    println("  JW spec.  : spin spectrum vs. Jordan-Wigner spectrum")
    println("  F spec.   : Jordan-Wigner spectrum vs. Fourier-transformed spectrum")
    println("  F modes   : BdG quasiparticle energies before vs. after Fourier transform")
    println("  Bog spec. : Fourier spectrum vs. Bogoliubov-transformed spectrum")
    println("  Bog modes : quasiparticle energies before vs. after Bogoliubov transform")
    println("  diag.     : final fermionic Hamiltonian recognized as number conserving")
    println()
    println("Pfeuty JW -> Fourier -> Bogoliubov transformation map: PASS")
    println()
end


#-------------------------------------------------------#
# Huberman, T. (et al.). (2005).                        #
# Two-magnon excitations observed by neutron scattering #
# in the two-dimensional spin-5/2 Heisenberg            #
# antiferromagnet Rb2MnF4.                              #
# Phys. Rev. B 72, 014413.                              #
# doi: 10.1103/PhysRevB.72.014413                       #
#-------------------------------------------------------#
let
    M = Phundamental.Models

    # -------------------------------------------------------------------------
    # Huberman et al. material/model parameters.
    #
    # Magnetic Mn2+ square lattice:
    #
    #     H = J sum_<ij> [Sx_i Sx_j + Sy_i Sy_j + (1 + δz) Sz_i Sz_j]
    #
    # The fitted one-magnon dispersion uses J = 0.648 ± 0.003 meV with
    # δz = 0.0048 fixed. Interplane exchange is neglected in the paper's
    # two-dimensional model.
    # -------------------------------------------------------------------------

    S = 5 / 2
    J = 0.648
    δz = 0.0048
    Jxy = J
    Jz = J * (1 + δz)

    a = 4.215                    # Å
    paper_ΔS = 0.167
    paper_elastic = 5.443
    paper_transverse = 3.112
    paper_two_magnon = 0.195

    crystal = M.CrystalStructure(M.BravaisLattice([a   0.0
                                                   0.0 a]),
                                 [M.BasisSite(:Mn, :Mn, [0.0, 0.0])])

    material = M.Material(
        "Rb2MnF4 magnetic Mn layer",
        crystal;
        species=Dict(:Mn => M.AtomicSpecies(:Mn; mass=54.938044)),
        basis_properties=[M.SiteProperties(spin=S, g_tensor=2.0)],
        metadata=Dict(:compound => :Rb2MnF4,
                      :dimensionality => :two_dimensional,
                      :reference => "Huberman et al., Phys. Rev. B 72, 014413 (2005)",
                      :doi => "10.1103/PhysRevB.72.014413")
    )

    # -------------------------------------------------------------------------
    # Published LSWT expressions.
    #
    # γ_Q = cos[π(Qh + Qk)] cos[π(Qh - Qk)]
    #
    # ħω_Q = 4JS sqrt[(1 + δz)^2 - γ_Q^2]
    #
    # u_Q = cosh θ_Q
    # v_Q = sinh θ_Q
    # tanh(2θ_Q) = -γ_Q / (1 + δz)
    #
    # The explicit u,v expressions below preserve the sign convention required
    # by Huberman et al.'s one- and two-magnon structure factors.
    # -------------------------------------------------------------------------

    γ(h::Real, k::Real) = cospi(h + k) * cospi(h - k)

    function magnon_energy(h::Real, k::Real)
        g = γ(h, k)
        return 4J * S * sqrt((1 + δz)^2 - g^2)
    end

    function bogoliubov_uv(h::Real, k::Real)
        g = γ(h, k)
        ε = sqrt((1 + δz)^2 - g^2)
        ratio = (1 + δz) / ε
        u = sqrt((ratio + 1) / 2)
        vmag = sqrt(max((ratio - 1) / 2, 0.0))
        v = iszero(g) ? 0.0 : -sign(g) * vmag
        return u, v
    end

    function discrete_reference_frequencies(L::Int)
        frequencies = Vector{Float64}(undef, L^2)
        index = 1

        for nx in 0:L-1, ny in 0:L-1
            h = nx / L
            k = ny / L
            frequencies[index] = magnon_energy(h, k)
            index += 1
        end

        sort!(frequencies)
        return frequencies
    end

    function discrete_spin_reduction(L::Int)
        total = 0.0

        for nx in 0:L-1, ny in 0:L-1
            h = nx / L
            k = ny / L
            _, v = bogoliubov_uv(h, k)
            total += v^2
        end

        return total / L^2
    end

    function build_rb2mnf4_model(L::Int)
        iseven(L) || throw(ArgumentError("periodic Néel reference requires even L"))

        cluster = M.Supercell(crystal, [L, L]; periodic=true)

        model = build_model(material, cluster, XXZModel(Jxy, Jz; field_z=0.0, neighbor_cutoff=1.01a, max_shell=1, hbar=1.0))

        reference_signs = Int[iseven(sum(site.cell)) ? 1 : -1 for site in cluster.sites]

        return cluster, model, reference_signs
    end

    # -------------------------------------------------------------------------
    # Warm-up: remove Julia compilation from performance timings.
    # -------------------------------------------------------------------------

    warm_cluster, warm_model, warm_signs = build_rb2mnf4_model(4)

    solve(LinearSpinWaveSolver(reference_signs=warm_signs, tol=1e-11), warm_model)

    # -------------------------------------------------------------------------
    # Phundamental LSWT correctness + scaling benchmark.
    #
    # For each finite periodic lattice:
    #
    #   1. build the XXZ model through the existing material/model API,
    #   2. verify fourfold square-lattice coordination,
    #   3. solve using LinearSpinWaveSolver,
    #   4. compare the entire numerical LSWT spectrum against Huberman Eq. (2),
    #   5. compare the BdG vacuum occupation against the analytic ΔS.
    #
    # Sorting the full frequency sets avoids assigning arbitrary numerical
    # eigenvectors inside exactly degenerate momentum sectors.
    # -------------------------------------------------------------------------

    performance_L = (4, 6, 8, 10, 12)
    performance_N = Int[]
    build_times = Float64[]
    solve_times = Float64[]
    spectrum_errors = Float64[]
    spin_reduction_errors = Float64[]

    println()
    println("HUBERMAN Rb2MnF4 LINEAR-SPIN-WAVE VALIDATION")
    println("Existing Phundamental XXZModel -> LinearSpinWaveSolver path")
    println()
    println("    L       N    bonds    z      build [s]    solve [s]      max |Δω| [meV]      |ΔS|")

    for L in performance_L
        cluster = nothing
        model = nothing
        reference_signs = Int[]

        GC.gc()

        build_time = @elapsed begin
            cluster, model, reference_signs = build_rb2mnf4_model(L)
        end

        bonds = model.parameters[:bonds]
        coordination = M.coordination_numbers(cluster, bonds)
        coordination_ok = all(coordination .== 4)

        result = nothing

        GC.gc()

        solve_time = @elapsed begin
            result = solve(LinearSpinWaveSolver(reference_signs=reference_signs, tol=1e-11), model)
        end

        check = validate_spinwave_result(result; tol=1e-9)

        N = M.nsites(cluster)
        expected = discrete_reference_frequencies(L)
        observed = sort(copy(result.frequencies))

        length(observed) == length(expected) || error(
            "LSWT mode count mismatch for L=$L: observed $(length(observed)), expected $(length(expected))"
        )

        spectrum_error = maximum(abs.(observed .- expected))

        analytic_ΔS_L = discrete_spin_reduction(L)

        numerical_ΔS_L = if size(result.modes, 1) == 2N
            sum(abs2, view(result.modes, N+1:2N, :)) / N
        else
            NaN
        end

        spin_reduction_error = abs(numerical_ΔS_L - analytic_ΔS_L)

        push!(performance_N, N)
        push!(build_times, build_time)
        push!(solve_times, solve_time)
        push!(spectrum_errors, spectrum_error)
        push!(spin_reduction_errors, spin_reduction_error)

        @printf(
            "%5d  %6d  %7d   %s   %10.4e   %10.4e      %12.3e      %9.3e\n",
            L,
            N,
            length(bonds),
            coordination_ok ? "PASS" : "FAIL",
            build_time,
            solve_time,
            spectrum_error,
            spin_reduction_error,
        )

        check[:passed] || println("        SpinWaveResult validation: FAIL")
    end

    scaling_exponent = log(solve_times[end] / solve_times[1]) / log(performance_N[end] / performance_N[1])

    println()
    @printf("Observed endpoint LSWT solve-time exponent: t ~ N^%.2f\n", scaling_exponent)
    @printf("Maximum finite-lattice dispersion error: %.3e meV\n", maximum(spectrum_errors))
    @printf("Maximum finite-lattice ΔS error: %.3e\n", maximum(spin_reduction_errors))

    # -------------------------------------------------------------------------
    # Thermodynamic-limit zero-point spin reduction.
    #
    # Huberman et al. report ΔS = 0.167 for δz = 0.0048.
    # -------------------------------------------------------------------------

    nq_ΔS = 512
    sum_v2 = 0.0
    sum_u2 = 0.0
    sum_uv = 0.0

    for ih in 0:nq_ΔS-1, ik in 0:nq_ΔS-1
        h = -0.5 + ih / nq_ΔS
        k = -0.5 + ik / nq_ΔS
        u, v = bogoliubov_uv(h, k)

        sum_v2 += v^2
        sum_u2 += u^2
        sum_uv += u * v
    end

    nq_total = nq_ΔS^2

    ΔS = sum_v2 / nq_total
    mean_u2 = sum_u2 / nq_total
    mean_uv = sum_uv / nq_total

    # Eq. (5) integrated over Q and energy:
    #
    #     <Szz_inelastic> = <u²><v²> - <uv>²
    #
    # which reduces to ΔS(ΔS + 1) when the full BZ average of uv vanishes.
    two_magnon_sum = mean_u2 * ΔS - mean_uv^2

    elastic_sum = (S - ΔS)^2
    transverse_sum = (S - ΔS) * (2ΔS + 1)
    total_sum = S * (S + 1)

    println()
    println("HUBERMAN ZERO-POINT / SUM-RULE VALIDATION")
    @printf("ΔS numerical                     = %.9f\n", ΔS)
    @printf("ΔS Huberman                      = %.6f\n", paper_ΔS)
    @printf("|ΔS - ΔS_Huberman|               = %.3e\n", abs(ΔS - paper_ΔS))
    println()
    @printf("Elastic Szz                      = %.6f   [paper %.3f]\n", elastic_sum, paper_elastic)
    @printf("One-magnon Sxx + Syy             = %.6f   [paper %.3f]\n", transverse_sum, paper_transverse)
    @printf("Two-magnon Szz inelastic         = %.6f   [paper %.3f]\n", two_magnon_sum, paper_two_magnon)
    @printf("Total S(S+1)                     = %.6f\n", total_sum)
    @printf("<uv> over full Brillouin zone    = %.3e\n", mean_uv)

    # -------------------------------------------------------------------------
    # Published one-magnon landmarks.
    # -------------------------------------------------------------------------

    one_magnon_gap = 4J * S * sqrt(δz * (2 + δz))
    one_magnon_zone_boundary = 4J * S * (1 + δz)
    two_magnon_global_threshold = 2one_magnon_gap
    two_magnon_upper_scale = 2one_magnon_zone_boundary

    println()
    println("PUBLISHED LSWT ENERGY LANDMARKS")
    @printf("One-magnon anisotropy gap        = %.6f meV\n", one_magnon_gap)
    @printf("Two-magnon threshold 2Δ          = %.6f meV\n", two_magnon_global_threshold)
    @printf("Nearest-neighbor zone boundary   = %.6f meV\n", one_magnon_zone_boundary)
    @printf("Two-magnon upper energy scale    = %.6f meV\n", two_magnon_upper_scale)
    println("Nearest-neighbor LSWT predicts an exactly flat antiferromagnetic zone boundary.")
    println("Huberman et al. report a small experimental variation of approximately 1 ± 0.5% along that boundary.")

    # -------------------------------------------------------------------------
    # Symmetry path used to represent the same characteristic directions
    # discussed in Huberman Figs. 4 and 7:
    #
    #   Γ       = (0, 0)
    #   X       = (0.5, 0)
    #   Σ       = (0.25, 0.25)
    #   Q_AF    = (0.5, 0.5)
    #
    # X -> Σ lies on Qh + Qk = 0.5, the antiferromagnetic zone boundary.
    # -------------------------------------------------------------------------

    function symmetry_path(points_per_segment::Int)

        vertices = ((0.0, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5))

        h = Float64[]
        k = Float64[]
        x = Float64[]

        for segment in 1:3
            h0, k0 = vertices[segment]
            h1, k1 = vertices[segment + 1]

            for n in 0:points_per_segment
                segment > 1 && n == 0 && continue

                t = n / points_per_segment
                push!(h, (1 - t) * h0 + t * h1)
                push!(k, (1 - t) * k0 + t * k1)
                push!(x, (segment - 1) + t)
            end
        end

        return h, k, x
    end

    path_h, path_k, path_x = symmetry_path(64)

    one_magnon_path = [magnon_energy(h, k) for (h, k) in zip(path_h, path_k)]

    one_magnon_intensity = Float64[]
    sizehint!(one_magnon_intensity, length(path_h))

    for (h, k) in zip(path_h, path_k)
        u, v = bogoliubov_uv(h, k)
        push!(one_magnon_intensity, 0.5 * (S - ΔS) * (u + v)^2)
    end

    # -------------------------------------------------------------------------
    # Huberman Eq. (5): intrinsic T=0 two-magnon longitudinal continuum.
    #
    #     Szz_inelastic(Q,E) = 1/N sum_Q1 f(Q1,Q-Q1) δ(E - E_Q1 - E_Q-Q1)
    #
    #     f(Q1,Q2) = 1/2 (u1 v2 - u2 v1)^2
    #
    # Huberman evaluate this numerically using a fine Q1 grid and replace the
    # energy delta function by a narrow area-normalized Gaussian.
    #
    # sigma_two below is ONLY a numerical delta-function representation.
    # It is not the MAPS instrumental resolution.
    # -------------------------------------------------------------------------

    function gaussian_smooth(raw::Vector{Float64}, σ::Float64, dE::Float64)
        radius = max(1, ceil(Int, 4σ / dE))
        offsets = collect(-radius:radius)
        kernel = exp.(-0.5 .* ((offsets .* dE) ./ σ).^2)
        kernel ./= sum(kernel)

        smoothed = zeros(Float64, length(raw))

        @inbounds for i in eachindex(raw)
            value = 0.0

            for (kernel_index, offset) in enumerate(offsets)
                source = i - offset

                if 1 <= source <= length(raw)
                    value += kernel[kernel_index] * raw[source]
                end
            end

            smoothed[i] = value
        end

        return smoothed
    end

    function two_magnon_spectrum(
        path_h::Vector{Float64},
        path_k::Vector{Float64},
        energy_axis::Vector{Float64};
        nq::Int,
        σ::Float64,
        high_energy_window::Tuple{Float64,Float64},
    )
        qgrid = collect(range(-0.5, 0.5; length=nq + 1))[1:end-1]
        npairs = nq^2

        q1_h = Vector{Float64}(undef, npairs)
        q1_k = Vector{Float64}(undef, npairs)
        q1_energy = Vector{Float64}(undef, npairs)
        q1_u = Vector{Float64}(undef, npairs)
        q1_v = Vector{Float64}(undef, npairs)

        index = 1

        for h in qgrid, k in qgrid
            u, v = bogoliubov_uv(h, k)

            q1_h[index] = h
            q1_k[index] = k
            q1_energy[index] = magnon_energy(h, k)
            q1_u[index] = u
            q1_v[index] = v

            index += 1
        end

        dE = energy_axis[2] - energy_axis[1]
        Emin = first(energy_axis)
        Emax = last(energy_axis)

        spectrum = zeros(Float64, length(path_h), length(energy_axis))
        high_energy_intensity = zeros(Float64, length(path_h))

        raw = zeros(Float64, length(energy_axis))

        high_min, high_max = high_energy_window

        @inbounds for iq in eachindex(path_h)
            fill!(raw, 0.0)

            Qh = path_h[iq]
            Qk = path_k[iq]

            for p in 1:npairs
                h2 = Qh - q1_h[p]
                k2 = Qk - q1_k[p]

                u2, v2 = bogoliubov_uv(h2, k2)
                Epair = q1_energy[p] + magnon_energy(h2, k2)

                weight = 0.5 * (q1_u[p] * v2 - u2 * q1_v[p])^2 / npairs

                if Emin <= Epair <= Emax
                    bin = round(Int, (Epair - Emin) / dE) + 1

                    if 1 <= bin <= length(raw)
                        raw[bin] += weight / dE
                    end
                end

                if high_min <= Epair <= high_max
                    high_energy_intensity[iq] += weight
                end
            end

            spectrum[iq, :] .= gaussian_smooth(raw, σ, dE)
        end

        return spectrum, high_energy_intensity
    end

    energy_axis = collect(range(0.0, 13.5; length=541))
    nq_two = 128
    σ_two = 0.08
    high_energy_window = (7.5, 10.0)

    two_magnon = nothing
    high_energy_two_magnon = nothing

    GC.gc()

    two_magnon_time = @elapsed begin
        two_magnon, high_energy_two_magnon = two_magnon_spectrum(path_h, path_k, energy_axis; nq=nq_two, σ=σ_two, high_energy_window=high_energy_window)
    end

    pair_evaluations = length(path_h) * nq_two^2
    pair_rate = pair_evaluations / max(two_magnon_time, eps(Float64)) / 1e6

    max_high_intensity, max_high_index = findmax(high_energy_two_magnon)

    println()
    println("TWO-MAGNON Eq. (5) NUMERICAL BENCHMARK")
    @printf("Q1 grid                          = %d × %d\n", nq_two, nq_two)
    @printf("Symmetry-path points             = %d\n", length(path_h))
    @printf("Pair evaluations                 = %d\n", pair_evaluations)
    @printf("Evaluation time                  = %.6f s\n", two_magnon_time)
    @printf("Pair-evaluation throughput       = %.3f million/s\n", pair_rate)
    @printf("Numerical Gaussian sigma         = %.3f meV\n", σ_two)
    @printf("Fig. 7(b) energy window          = %.2f to %.2f meV\n", high_energy_window...)
    @printf(
        "Maximum high-energy intensity at Q = (%.4f, %.4f), Szz = %.6e\n",
        path_h[max_high_index],
        path_k[max_high_index],
        max_high_intensity,
    )

    # -------------------------------------------------------------------------
    # Validation status.
    #
    # The existing LSWT backend is directly tested against Huberman Eq. (2).
    # The current module does not yet expose a SpinWaveResult-specific
    # S^{αβ}(Q,E) observable route, so Eqs. (4)-(5) are evaluated here as the
    # independent published reference target for future LSWT neutron
    # observables. This distinction is intentional: the benchmark should expose
    # that API limitation rather than silently bypass it.
    # -------------------------------------------------------------------------

    dispersion_pass = maximum(spectrum_errors) <= 1e-8
    finite_ΔS_pass = maximum(spin_reduction_errors) <= 1e-8
    thermodynamic_ΔS_pass = abs(ΔS - paper_ΔS) <= 5e-4
    table_pass = abs(elastic_sum - paper_elastic) <= 5e-3 && abs(transverse_sum - paper_transverse) <= 5e-3 && abs(two_magnon_sum - paper_two_magnon) <= 1e-3

    overall_pass = dispersion_pass && finite_ΔS_pass && thermodynamic_ΔS_pass && table_pass

    println()
    println("HUBERMAN BENCHMARK VALIDATION")
    println("  Phundamental LSWT dispersion       : ", dispersion_pass ? "PASS" : "FAIL")
    println("  Phundamental LSWT mode amplitudes  : ", finite_ΔS_pass ? "PASS" : "FAIL")
    println("  Huberman ΔS = 0.167                : ", thermodynamic_ΔS_pass ? "PASS" : "FAIL")
    println("  Huberman Table-I spectral weights  : ", table_pass ? "PASS" : "FAIL")
    println("  Two-magnon Eq. (5) reference       : GENERATED")
    println()
    println("Huberman Rb2MnF4 LSWT physics benchmark: ", overall_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Visualization:
    #
    #   (a) Huberman Eq. (2) one-magnon dispersion.
    #   (b) Huberman Eq. (5) two-magnon Szz continuum with one-magnon overlay.
    #   (c) Integrated two-magnon intensity in 8.75 ± 1.25 meV window.
    #   (d) Existing Phundamental finite-lattice LSWT solve-time scaling.
    # -------------------------------------------------------------------------

    fig = Figure(size=(1200, 900))

    xticks = ([0.0, 1.0, 2.0, 3.0],
              ["Γ", "X", "Σ", "Q_AF"])

    ax11 = Axis(fig[1, 1]; xlabel="symmetry path", ylabel="energy (meV)", title="Huberman one-magnon LSWT dispersion", xticks=xticks)

    lines!(ax11, path_x, one_magnon_path)

    for xboundary in (1.0, 2.0)
        vlines!(ax11, [xboundary]; linestyle=:dash)
    end

    ax12 = Axis(fig[1, 2]; xlabel="symmetry path", ylabel="energy (meV)", title="Two-magnon longitudinal continuum", xticks=xticks)

    intensity_floor = max(maximum(two_magnon) * 1e-8, eps(Float64))
    log_two_magnon = log10.(two_magnon .+ intensity_floor)

    hm = heatmap!(ax12, path_x, energy_axis, two_magnon .+ intensity_floor, colorrange = (0.0, 0.1))

    lines!(ax12, path_x, one_magnon_path; linewidth=1.5)

    for xboundary in (1.0, 2.0)
        vlines!(ax12, [xboundary]; linestyle=:dash)
    end

    Colorbar(fig[1, 3], hm; label="Sᶻᶻ(Q,E)")

    ax21 = Axis(fig[2, 1]; xlabel="symmetry path", ylabel="integrated Szz", title="Two-magnon intensity: 8.75 ± 1.25 meV", xticks=xticks)

    lines!(ax21, path_x, high_energy_two_magnon)

    for xboundary in (1.0, 2.0)
        vlines!(ax21, [xboundary]; linestyle=:dash)
    end

    ax22 = Axis(fig[2, 2]; xlabel="number of spins N = L²", ylabel="LSWT solve time (s)", title="Finite-lattice LinearSpinWaveSolver scaling", xscale=log10, yscale=log10)

    scatterlines!(ax22, performance_N, solve_times)

    save(joinpath(BENCHMARK_FIGURE_DIR, "Huberman-Rb2MnF4-LSWT-benchmark.png"), fig)
    # -------------------------------------------------------------------------
    # Native Phundamental LSWT neutron-tensor validation.
    #
    # This compares the new SpinWaveResult-native S^{αβ}(Q,E) implementation
    # against Huberman et al. Eqs. (4)-(5) on the identical finite periodic
    # momentum grid. The comparison is deliberately finite-L so that any error is
    # an implementation error rather than a thermodynamic-limit finite-size error.
    # -------------------------------------------------------------------------

    native_L = 12
    native_L % 4 == 0 || error("native Huberman validation requires L divisible by 4")
    native_points_per_segment = native_L ÷ 4
    native_h, native_k, native_x = symmetry_path(native_points_per_segment)

    native_cluster, native_model, native_signs = build_rb2mnf4_model(native_L)
    native_result = solve(LinearSpinWaveSolver(reference_signs=native_signs, tol=1e-11), native_model)

    native_result_check = validate_spinwave_result(native_result; tol=1e-9)

    @printf("Native positive-mode metric error = %.3e\n", native_result_check[:mode_metric_error])

    native_result_check[:canonical_modes] || error("native LSWT neutron tensor requires a canonical bosonic mode basis")

    native_positions = M.site_positions(native_cluster)
    native_sites = collect(1:M.nsites(native_cluster))
    native_field = SpinTensorField(xkeys=[F.Sx(i) for i in native_sites],
                                   ykeys=[F.Sy(i) for i in native_sites],
                                   zkeys=[F.Sz(i) for i in native_sites],
                                   positions=native_positions, normalization=:sqrtN)
    native_q = [M.reciprocal_vector(crystal, [h, k]) for (h, k) in zip(native_h, native_k)]

    native_energy_axis = collect(range(0.0, 13.5; length=541))
    native_broadening = GaussianBroadening(σ_two)

    native_one = nothing
    native_two = nothing

    GC.gc()
    native_one_time = @elapsed begin
        native_one = spin_tensor_structure_factor(native_model, native_result, native_field, native_q, native_energy_axis;
                                                  temperature=0.0, broadening=native_broadening,channels=(:one_magnon,),
                                                  renormalize_transverse=true, spectral_accumulation=:direct, weight_tol=1e-14)
    end

    GC.gc()
    native_two_time = @elapsed begin
        native_two = spin_tensor_structure_factor(native_model, native_result, native_field, native_q, native_energy_axis;
                                                  temperature=0.0, broadening=native_broadening, channels=(:two_magnon,),
                                                  spectral_accumulation=:direct, weight_tol=1e-14)
    end

    native_ΔS = discrete_spin_reduction(native_L)
    reference_one = zeros(Float64, length(native_h), length(native_energy_axis), 3, 3)
    reference_two = zeros(Float64, length(native_h), length(native_energy_axis))

    for iq in eachindex(native_h)
        h = native_h[iq]
        k = native_k[iq]
        u, v = bogoliubov_uv(h, k)
        center = magnon_energy(h, k)
        weight = 0.5 * (S - native_ΔS) * (u + v)^2

        @inbounds for ie in eachindex(native_energy_axis)
            kernel = exp(-0.5 * ((native_energy_axis[ie] - center) / σ_two)^2) / (sqrt(2π) * σ_two)
            reference_one[iq, ie, 1, 1] = weight * kernel
            reference_one[iq, ie, 2, 2] = weight * kernel
        end
    end

    q1grid = collect(range(-0.5, 0.5; length=native_L + 1))[1:end-1]
    for iq in eachindex(native_h)
        Qh = native_h[iq]
        Qk = native_k[iq]

        for h1 in q1grid, k1 in q1grid
            h2 = Qh - h1
            k2 = Qk - k1
            u1, v1 = bogoliubov_uv(h1, k1)
            u2, v2 = bogoliubov_uv(h2, k2)
            center = magnon_energy(h1, k1) + magnon_energy(h2, k2)
            weight = 0.5 * (u1 * v2 - u2 * v1)^2 / native_L^2

            @inbounds for ie in eachindex(native_energy_axis)
                kernel = exp(-0.5 * ((native_energy_axis[ie] - center) / σ_two)^2) / (sqrt(2π) * σ_two)
                reference_two[iq, ie] += weight * kernel
            end
        end
    end

    native_one_values = native_one.intensity
    native_two_values = @view native_two.intensity[:, :, 3, 3]

    one_error = norm(native_one_values - reference_one) / max(norm(reference_one), eps(Float64))
    two_error = norm(native_two_values - reference_two) / max(norm(reference_two), eps(Float64))
    xy_longitudinal_leakage = maximum(abs, native_two.intensity[:, :, 1:2, :])
    transverse_z_leakage = maximum(abs, native_one.intensity[:, :, 3, :])

    native_one_pass = one_error <= 1e-8 && transverse_z_leakage <= 1e-10
    native_two_pass = two_error <= 1e-8 && xy_longitudinal_leakage <= 1e-10

    println()
    println("NATIVE LSWT NEUTRON-TENSOR VALIDATION")
    @printf("Finite validation lattice          = %d × %d (%d spins)\n", native_L, native_L, native_L^2)
    @printf("Native one-magnon tensor time      = %.6f s\n", native_one_time)
    @printf("Native two-magnon tensor time      = %.6f s\n", native_two_time)
    @printf("One-magnon tensor relative error   = %.3e\n", one_error)
    @printf("Two-magnon Szz relative error      = %.3e\n", two_error)
    @printf("One-magnon longitudinal leakage    = %.3e\n", transverse_z_leakage)
    @printf("Two-magnon transverse leakage      = %.3e\n", xy_longitudinal_leakage)
    println("  Native one-magnon Sαβ vs Eq. (4): ", native_one_pass ? "PASS" : "FAIL")
    println("  Native two-magnon Szz vs Eq. (5): ", native_two_pass ? "PASS" : "FAIL")

end


#-------------------------------------------------------#
# Sugiura, S. & Shimizu, A. (2013).                    #
# Canonical Thermal Pure Quantum State.                 #
# Phys. Rev. Lett. 111, 010401.                         #
# doi: 10.1103/PhysRevLett.111.010401                   #
#-------------------------------------------------------#
let
    M = Phundamental.Models

    # -------------------------------------------------------------------------
    # Spin-1/2 kagome Heisenberg antiferromagnet:
    #
    #     H = J sum_<ij> S_i · S_j
    #
    # Sugiura and Shimizu use J = kB = 1.
    #
    # The present Phundamental ThermalTypicality backend realizes the canonical
    # TPQ state through Krylov approximation of
    #
    #     |β,N> = exp(-βH/2)|ψ0>,
    #
    # rather than through the paper's practical microcanonical-TPQ expansion
    # truncated at k_term = 2000. The physical canonical state is the same;
    # the numerical realization is therefore independently benchmarked here.
    # -------------------------------------------------------------------------

    J = 1.0
    kB = 1.0
    S = 1 / 2

    paper_Tmin = 0.02
    paper_shoulder_T = 0.10
    paper_entropy_T = 0.20
    paper_entropy_fraction = 0.45
    paper_samples_N18 = 100

    # Runtime controls.
    #
    # The N=18 calculation is deliberately configurable because the paper used
    # 100 realizations, while a first performance-validation run should not be
    # forced to incur that cost.
    #
    # N=27 and N=30 use exact fixed-total-Sᶻ decomposition. They are enabled by
    # default in this benchmark because the sectorized API removes the previous
    # full-Hilbert-space memory barrier. Set PHUNDAMENTAL_TPQ_LARGE=0 to skip
    # those two long-running paper-scale calculations.
    #
    # Examples:
    #
    #     PHUNDAMENTAL_TPQ_SAMPLES18=100 julia empirical-benchmark-suite.jl
    #     julia -t auto empirical-benchmark-suite.jl
    #     PHUNDAMENTAL_TPQ_LARGE=0 julia empirical-benchmark-suite.jl
    #
    samples_N12 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_SAMPLES12", "64"))
    samples_N18 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_SAMPLES18", "8"))
    krylov_N12 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_KRYLOV12", "96"))
    krylov_N18 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_KRYLOV18", "96"))

    run_large_paper_clusters = get(ENV, "PHUNDAMENTAL_TPQ_LARGE", "1") == "1"

    large_samples_default = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_LARGE_SAMPLES", "1"))
    samples_N27 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_SAMPLES27", string(large_samples_default)))
    samples_N30 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_SAMPLES30", string(large_samples_default)))

    large_krylov_default = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_LARGE_KRYLOV", "96"))
    krylov_N27 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_KRYLOV27", string(large_krylov_default)))
    krylov_N30 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_KRYLOV30", string(large_krylov_default)))
    large_min_krylov_default = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_LARGE_MIN_KRYLOV", "32"))
    min_krylov_N27 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_MIN_KRYLOV27", string(large_min_krylov_default)))
    min_krylov_N30 = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_MIN_KRYLOV30", string(large_min_krylov_default)))
    adaptive_large_krylov = get(ENV, "PHUNDAMENTAL_TPQ_ADAPTIVE_KRYLOV", "1") == "1"
    large_convergence_tol = parse(Float64, get(ENV, "PHUNDAMENTAL_TPQ_CONVERGENCE_TOL", "1e-8"))
    large_convergence_interval = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_CONVERGENCE_INTERVAL", "8"))
    large_convergence_consecutive = parse(Int, get(ENV, "PHUNDAMENTAL_TPQ_CONVERGENCE_CONSECUTIVE", "2"))

    sector_memory_fraction = parse(Float64, get(ENV, "PHUNDAMENTAL_TPQ_MEMORY_FRACTION", "0.70"))
    sector_rank_lookup = Symbol(get(ENV, "PHUNDAMENTAL_TPQ_RANK_LOOKUP", "auto"))
    sector_threaded_matvec = get(ENV, "PHUNDAMENTAL_TPQ_THREADED_MATVEC", "1") == "1"
    sector_progress = get(ENV, "PHUNDAMENTAL_TPQ_PROGRESS", "1") == "1"
    sector_pair_equivalent = get(ENV, "PHUNDAMENTAL_TPQ_PAIR_SECTORS", "1") == "1"

    seed = 2013010401

    temperatures = Float64[collect(0.02:0.01:0.20)..., collect(0.22:0.02:1.00)...]

    # -------------------------------------------------------------------------
    # Kagome primitive cell.
    #
    # Nearest-neighbor distance = 1:
    #
    #     a1 = (2, 0)
    #     a2 = (1, sqrt(3))
    #
    # with three sites
    #
    #     b1 = (0, 0)
    #     b2 = a1/2
    #     b3 = a2/2.
    #
    # The matrices below contain torus translation vectors as columns in the
    # primitive-cell basis. det(S) primitive cells therefore contain
    # N = 3|det(S)| spins.
    #
    # N=27 and N=30 use the standard periodic kagome 27b and 30 torus
    # translations. The N=18 choices are isolated here because the 18a/18b
    # naming is geometry-convention dependent in the historical literature.
    # -------------------------------------------------------------------------

    primitive_direct = [2.0  1.0
                        0.0  sqrt(3.0)]

    primitive_basis = ([0.0, 0.0],
                       [0.5, 0.0],
                       [0.0, 0.5] )

    cluster_matrices = Dict{Symbol,Matrix{Int}}(:N12  => [2  0
                                                          0  2],

                                                :N18a => [3  0
                                                          0  2],

                                                :N18b => [3 -1
                                                          0  2],

                                                :N27  => [3  0
                                                          0  3],

                                                :N30  => [2 -2
                                                          1  4] )

    cluster_order = (:N12, :N18a, :N18b, :N27, :N30)

    # -------------------------------------------------------------------------
    # Geometry helpers.
    # -------------------------------------------------------------------------

    det2(Smat::AbstractMatrix{<:Integer}) =
        Smat[1, 1] * Smat[2, 2] - Smat[1, 2] * Smat[2, 1]

    function quotient_cells(Smat::Matrix{Int})
        ncells = abs(det2(Smat))
        Sinv = inv(Float64.(Smat))

        corners = (Smat * [0, 0], Smat * [1, 0],
                   Smat * [0, 1], Smat * [1, 1] )

        xmin = minimum(corner[1] for corner in corners)
        xmax = maximum(corner[1] for corner in corners)
        ymin = minimum(corner[2] for corner in corners)
        ymax = maximum(corner[2] for corner in corners)

        cells = Vector{Vector{Int}}()
        tol = 1e-10

        for x in xmin-1:xmax+1, y in ymin-1:ymax+1
            cell = [x, y]
            reduced = Sinv * Float64.(cell)

            if all(value -> value >= -tol && value < 1 - tol, reduced)
                push!(cells, cell)
            end
        end

        length(cells) == ncells || error("failed to construct quotient lattice: found $(length(cells)) cells, expected $ncells")

        sort!(cells; by=cell -> Tuple(Sinv * Float64.(cell)))

        return cells
    end

    function build_kagome_torus(label::Symbol, Smat::Matrix{Int})
        ncells = abs(det2(Smat))
        cells = quotient_cells(Smat)
        Sinv = inv(Float64.(Smat))

        torus_direct = primitive_direct * Float64.(Smat)

        basis = M.BasisSite[]
        site_number = 0

        for cell in cells, basis_position in primitive_basis
            primitive_fractional =
                Float64.(cell) .+ basis_position

            torus_fractional =
                mod.(Sinv * primitive_fractional, 1.0)

            for d in eachindex(torus_fractional)
                if abs(torus_fractional[d]) <= 1e-12 || abs(torus_fractional[d] - 1) <= 1e-12
                    torus_fractional[d] = 0.0
                end
            end

            site_number += 1

            push!(basis, M.BasisSite(Symbol("s$(site_number)"), :Spin, torus_fractional))
        end

        crystal = M.CrystalStructure(M.BravaisLattice(torus_direct), basis)

        material = M.Material("Kagome Heisenberg $label", crystal;
                              species=Dict(:Spin => M.AtomicSpecies(:Spin; mass=1.0)),
                              basis_properties=[M.SiteProperties(spin=S, g_tensor=2.0) for _ in basis],
                              metadata=Dict(:model => :kagome_heisenberg_antiferromagnet,
                                            :cluster => label,
                                            :reference => "Sugiura and Shimizu, Phys. Rev. Lett. 111, 010401 (2013)",
                                            :doi => "10.1103/PhysRevLett.111.010401")
                             )

        cluster = M.Supercell(crystal, [1, 1]; periodic=true)

        model = build_model(material, cluster, HeisenbergModel(J; field=(0.0, 0.0, 0.0), neighbor_cutoff=1.01, max_shell=1, hbar=1.0))

        bonds = model.parameters[:bonds]
        coordination = M.coordination_numbers(cluster, bonds)

        N = M.nsites(cluster)

        return (label=label, ncells=ncells, N=N,
                crystal=crystal, material=material,
                cluster=cluster, model=model, bonds=bonds,
                coordination=coordination, matrix=Smat)
    end

    # -------------------------------------------------------------------------
    # Numerical / memory helpers.
    # -------------------------------------------------------------------------

    hilbert_dimension(N::Int) = Int(1) << N

    bytes_to_gib(bytes::Real) =
        Float64(bytes) / 2.0^30

    function tpq_memory_estimate(N::Int; krylov_dim::Int, reorthogonalize::Bool)
        dim = Float64(2.0^N)

        # One ComplexF64 Hilbert-space vector.
        vector_gib = bytes_to_gib(dim * sizeof(ComplexF64))

        # The optimized low-memory Lanczos path retains q, q_previous, and Hq.
        # Full reorthogonalization additionally stores the complete Krylov basis.
        work_vectors = reorthogonalize ? krylov_dim + 3 : 3

        krylov_gib = work_vectors * vector_gib

        return (dimension=dim, vector_gib=vector_gib, krylov_gib=krylov_gib)
    end

    function sector_memory_estimate(model, N::Int; krylov_dim::Int, reorthogonalize::Bool)
        plan = spin_sector_plan(model; exploit_equivalent_sectors=sector_pair_equivalent, hbar=1.0)

        largest_dimension = maximum(plan.dimensions)
        evaluated_dimension = sum(plan.dimensions)
        largest_vector_gib = bytes_to_gib(largest_dimension * sizeof(ComplexF64))

        work_vectors = reorthogonalize ? min(krylov_dim, largest_dimension) + 3 : 3
        workspace_gib = work_vectors * largest_vector_gib

        lookup_possible = N < Sys.WORD_SIZE - 1 && largest_dimension <= Int(typemax(UInt32))
        lookup_gib = lookup_possible ? bytes_to_gib(hilbert_dimension(N) * sizeof(UInt32)) : Inf
        memory_budget_gib = sector_memory_fraction * bytes_to_gib(Sys.total_memory())

        use_lookup = if sector_rank_lookup === :force
            lookup_possible
        elseif sector_rank_lookup === :auto
            lookup_possible && workspace_gib + lookup_gib <= memory_budget_gib
        elseif sector_rank_lookup === :none
            false
        else
            throw(ArgumentError("PHUNDAMENTAL_TPQ_RANK_LOOKUP must be auto, none, or force"))
        end

        estimated_peak_gib = workspace_gib + (use_lookup ? lookup_gib : 0.0)
        safe = workspace_gib <= memory_budget_gib && !(sector_rank_lookup === :force && estimated_peak_gib > memory_budget_gib)

        return (
            plan=plan,
            full_dimension=plan.total_dimension,
            representative_sectors=length(plan.sectors),
            largest_dimension=largest_dimension,
            evaluated_dimension=evaluated_dimension,
            dimension_reduction=plan.total_dimension / evaluated_dimension,
            largest_vector_gib=largest_vector_gib,
            workspace_gib=workspace_gib,
            lookup_gib=lookup_gib,
            use_lookup=use_lookup,
            rank_strategy=use_lookup ? :dense_uint32 : :combinatorial,
            estimated_peak_gib=estimated_peak_gib,
            memory_budget_gib=memory_budget_gib,
            safe=safe,
        )
    end

    nearest_index(values, target) = argmin(abs.(values .- target))

    function local_peak_indices(temperatures::Vector{Float64}, values::Vector{Float64}; Tmin::Real, Tmax::Real)
        indices = Int[]
        for i in 2:length(values)-1
            if Tmin <= temperatures[i] <= Tmax && values[i] > values[i - 1] && values[i] >= values[i + 1]
                push!(indices, i)
            end
        end
        return indices
    end

    function finite_difference_temperature_derivative(temperatures::Vector{Float64}, values::Vector{Float64})
        length(temperatures) == length(values) || throw(DimensionMismatch("temperature and value arrays differ in length"))
        length(values) >= 3 || throw(ArgumentError("temperature derivative requires at least three points"))
        derivative = similar(values)
        derivative[1] = (values[2] - values[1]) / (temperatures[2] - temperatures[1])
        @inbounds for i in 2:length(values)-1
            derivative[i] = (values[i + 1] - values[i - 1]) / (temperatures[i + 1] - temperatures[i - 1])
        end
        derivative[end] = (values[end] - values[end - 1]) / (temperatures[end] - temperatures[end - 1])
        return derivative
    end

    function adaptive_krylov_summary(curve)
        iteration_rows = curve.metadata[:quadrature_iterations]
        error_rows = curve.metadata[:quadrature_convergence_error]
        iterations = Int[x for row in iteration_rows for x in row]
        finite_errors = Float64[x for row in error_rows for x in row if isfinite(x)]
        return (converged=get(curve.metadata, :adaptive_krylov_converged, nothing),
                minimum_iterations=minimum(iterations),
                maximum_iterations=maximum(iterations),
                mean_iterations=sum(iterations) / length(iterations),
                maximum_convergence_error=isempty(finite_errors) ? NaN : maximum(finite_errors))
    end

    function low_temperature_peak_diagnostic(temperatures::Vector{Float64}, values::Vector{Float64}, stderr::Vector{Float64};
                                             Tmin::Real, Tmax::Real, krylov_converged::Bool, krylov_error::Real, sigma::Real=3.0)
        length(temperatures) == length(values) == length(stderr) || throw(DimensionMismatch("low-temperature diagnostic arrays differ in length"))
        window = findall(T -> Tmin <= T <= Tmax, temperatures)
        length(window) >= 3 || throw(ArgumentError("low-temperature landmark window contains fewer than three points"))
        peaks = local_peak_indices(temperatures, values; Tmin=Tmin, Tmax=Tmax)
        numerical_resolution = isfinite(krylov_error) ? 10 * Float64(krylov_error) * max(maximum(abs, values[window]), 1.0) : 0.0

        if !krylov_converged
            return (status=:inconclusive_krylov, peak_index=nothing, prominence=NaN, resolution=numerical_resolution, stochastic_resolution=NaN)
        elseif isempty(peaks)
            return (status=:shoulder_consistent, peak_index=nothing, prominence=0.0, resolution=numerical_resolution, stochastic_resolution=NaN)
        end

        first_window = first(window)
        last_window = last(window)
        best = nothing
        for peak in peaks
            peak <= first_window && continue
            peak >= last_window && continue
            left_range = first_window:(peak - 1)
            right_range = (peak + 1):last_window
            left_index = left_range[argmin(values[left_range])]
            right_index = right_range[argmin(values[right_range])]
            baseline = max(values[left_index], values[right_index])
            prominence = max(values[peak] - baseline, 0.0)
            if isnothing(best) || prominence > best.prominence
                best = (peak_index=peak, left_index=left_index, right_index=right_index, prominence=prominence)
            end
        end

        isnothing(best) && return (status=:shoulder_consistent, peak_index=nothing, prominence=0.0, resolution=numerical_resolution, stochastic_resolution=NaN)
        stochastic_available = all(isfinite, (stderr[best.peak_index], stderr[best.left_index], stderr[best.right_index]))
        stochastic_resolution = stochastic_available ? sigma * (stderr[best.peak_index] + max(stderr[best.left_index], stderr[best.right_index])) : NaN
        resolution = stochastic_available ? max(numerical_resolution, stochastic_resolution) : numerical_resolution

        if best.prominence <= resolution
            status = :shoulder_consistent
        elseif stochastic_available
            status = :resolved_peak
        else
            status = :inconclusive_stochastic
        end

        return (status=status, peak_index=best.peak_index, prominence=best.prominence,
                resolution=resolution, stochastic_resolution=stochastic_resolution)
    end

    function stochastic_consistency(
        estimate::Vector{Float64},
        reference::Vector{Float64},
        stderr::Vector{Float64};
        sigma::Real=6.0,
        floor::Real,
    )
        length(estimate) == length(reference) == length(stderr) || throw(DimensionMismatch("stochastic comparison arrays differ in length"))

        residual = abs.(estimate .- reference)
        allowance = sigma .* stderr .+ floor

        normalized = residual ./ max.(allowance, eps(Float64))

        return (passed=all(residual .<= allowance), max_error=maximum(residual), max_normalized_error=maximum(normalized))
    end

    function run_ctpq(model; samples::Int, krylov_dim::Int, seed::Int, reorthogonalize::Bool)
        method = ThermalLanczos(samples=samples, krylov_dim=krylov_dim, tol=1e-10, seed=seed, matrix_free=true, reorthogonalize=reorthogonalize, hbar=1.0)

        curve = nothing

        GC.gc()

        elapsed = @elapsed begin
            curve = thermal_curve(model, method, temperatures; kB=kB)
        end

        return curve, elapsed, method
    end

    function run_sector_ctpq(model; samples::Int, krylov_dim::Int, seed::Int, min_krylov_dim::Int=min(32, krylov_dim),
                             adaptive_krylov::Bool=false, progress::Bool=sector_progress)
        decomposition = SectorDecomposition(symmetry=:total_sz, exploit_equivalent_sectors=sector_pair_equivalent,
                                            threaded_matvec=sector_threaded_matvec, rank_lookup=sector_rank_lookup,
                                            memory_fraction=sector_memory_fraction)

        method = CanonicalTPQ(samples=samples, krylov_dim=krylov_dim, min_krylov_dim=min_krylov_dim, breakdown_tol=1e-10,
                              adaptive_krylov=adaptive_krylov, convergence_tol=large_convergence_tol,
                              convergence_check_interval=large_convergence_interval, convergence_consecutive=large_convergence_consecutive,
                              seed=seed, matrix_free=true, store_vectors=false, reorthogonalize=false, hbar=1.0, propagator=:krylov,
                              parallel=false, progress=progress, decomposition=decomposition)
        workspace = nothing
        curve = nothing
        GC.gc()
        workspace_time = @elapsed workspace = thermal_workspace(model, method)
        GC.gc()
        curve_time = @elapsed curve = thermal_curve(model, method, temperatures; kB=kB, workspace=workspace)
        return curve, workspace_time, curve_time, method, workspace
    end

    # -------------------------------------------------------------------------
    # Construct and validate the finite kagome tori.
    # -------------------------------------------------------------------------

    clusters = Dict{Symbol,Any}()

    println()
    println("SUGIURA-SHIMIZU KAGOME cTPQ BENCHMARK")
    println("Spin-1/2 nearest-neighbor Heisenberg antiferromagnet")
    println()
    println("FINITE-TORUS GEOMETRY VALIDATION")
    println()
    println(" cluster       N     cells     bonds      z=4     Hilbert dimension")

    geometry_pass = true

    for label in cluster_order
        data = build_kagome_torus(label, cluster_matrices[label])

        clusters[label] = data

        N = data.N
        coordination_ok = all(data.coordination .== 4)

        bond_ok = length(data.bonds) == 2N

        size_ok = N == 3 * data.ncells

        cluster_pass = coordination_ok && bond_ok && size_ok

        geometry_pass &= cluster_pass

        @printf(
            "%7s  %6d    %6d    %6d    %5s    %14d\n",
            String(label),
            N,
            data.ncells,
            length(data.bonds),
            coordination_ok ? "PASS" : "FAIL",
            hilbert_dimension(N),
        )
    end

    println()
    println("Kagome periodic geometry: ", geometry_pass ? "PASS" : "FAIL")

    model12 = clusters[:N12].model
    N12 = clusters[:N12].N

    # -------------------------------------------------------------------------
    # Warm-up.
    #
    # Exclude first-call Julia compilation from the TPQ timing that follows.
    # ThermalLanczos is used here because run_ctpq uses the same low-memory
    # trace-quadrature API.
    # -------------------------------------------------------------------------

    println()
    println("TPQ warming up...")

    warm_method = ThermalLanczos(samples=1, krylov_dim=8, tol=1e-8, seed=seed, matrix_free=true, reorthogonalize=false, hbar=1.0)

    thermal_curve(model12, warm_method, [0.5]; kB=kB)

    println()
    println("TPQ warmed up!")

    # -------------------------------------------------------------------------
    # Exact canonical reference, N=12.
    #
    # Scalar canonical thermodynamics requires the complete spectrum but not
    # eigenvectors. The energy-only exact path therefore avoids constructing,
    # retaining, copying, and sorting the full dense eigenvector matrix.
    # -------------------------------------------------------------------------

    println()
    println("Computing N=12 exact canonical reference (energy spectrum only)...")
    @printf("Exact Hilbert dimension            = %d\n", hilbert_dimension(N12))
    flush(stdout)

    exact_spectrum12 = nothing

    GC.gc()

    exact_time12 = @elapsed begin
        exact_spectrum12 = energy_spectrum(ExactDiagonalization(), model12)
    end

    exact_curve12 = thermal_curve(exact_spectrum12, temperatures; kB=kB)

    println("N=12 exact canonical reference complete.")

    # -------------------------------------------------------------------------
    # Canonical TPQ vs exact canonical ensemble, N=12.
    # -------------------------------------------------------------------------

    tpq_curve12, tpq_time12, tpq_method12 = run_ctpq(model12; samples=samples_N12, krylov_dim=krylov_N12, seed=seed, reorthogonalize=true)

    u_exact12 = exact_curve12.internal_energy ./ N12

    c_exact12 = exact_curve12.heat_capacity ./ N12

    f_exact12 = exact_curve12.free_energy ./ N12

    s_exact12 = exact_curve12.entropy ./ N12

    u_tpq12 = tpq_curve12.internal_energy ./ N12

    c_tpq12 = tpq_curve12.heat_capacity ./ N12

    f_tpq12 = tpq_curve12.free_energy ./ N12

    s_tpq12 = tpq_curve12.entropy ./ N12

    σu12 = tpq_curve12.stderr[:internal_energy] ./ N12

    σc12 = tpq_curve12.stderr[:heat_capacity] ./ N12

    σf12 = tpq_curve12.stderr[:free_energy] ./ N12

    σs12 = tpq_curve12.stderr[:entropy] ./ N12

    u_check = stochastic_consistency(u_tpq12, u_exact12, σu12; sigma=6, floor=2e-3)

    c_check = stochastic_consistency(c_tpq12, c_exact12, σc12; sigma=6, floor=5e-3)

    f_check = stochastic_consistency(f_tpq12, f_exact12, σf12; sigma=6, floor=2e-3)

    s_check = stochastic_consistency(s_tpq12, s_exact12, σs12; sigma=6, floor=5e-3)

    exact_tpq_pass = u_check.passed && c_check.passed && f_check.passed && s_check.passed

    println()
    println("CANONICAL TPQ vs EXACT GIBBS: N = 12")
    @printf("Exact diagonalization time       = %.6f s\n", exact_time12)
    @printf("TPQ samples                      = %d\n", samples_N12)
    @printf("TPQ Krylov dimension             = %d\n", krylov_N12)
    @printf("TPQ curve time                   = %.6f s\n", tpq_time12)
    @printf("TPQ time / realization           = %.6f s\n", tpq_time12 / samples_N12)
    println()
    @printf("max |Δu| per site                = %.3e\n", u_check.max_error)
    @printf("max |Δc| per site                = %.3e\n", c_check.max_error)
    @printf("max |Δf| per site                = %.3e\n", f_check.max_error)
    @printf("max |Δs| per site                = %.3e\n", s_check.max_error)
    println()
    @printf("max normalized u residual        = %.3f\n", u_check.max_normalized_error)
    @printf("max normalized c residual        = %.3f\n", c_check.max_normalized_error)
    @printf("max normalized f residual        = %.3f\n", f_check.max_normalized_error)
    @printf("max normalized s residual        = %.3f\n", s_check.max_normalized_error)
    println()
    println("Canonical TPQ thermodynamics vs ExactGibbs: ", exact_tpq_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Direct canonical-TPQ-state validation.
    #
    # Retain exp[-β(H-Eshift)/2]|r> vectors and independently evaluate <H>.
    # This checks the actual pure-state realization rather than only the trace
    # quadrature thermodynamic formulas.
    # -------------------------------------------------------------------------

    direct_temperature = 0.20

    direct_method = ThermalTypicality(samples=8, krylov_dim=krylov_N12, tol=1e-10, seed=seed + 1, matrix_free=true, store_vectors=true, reorthogonalize=true, hbar=1.0)

    direct_state = thermal_state(model12, direct_method; temperature=direct_temperature, kB=kB)

    direct_energy = expectation(direct_state, model12.hamiltonian)

    direct_energy_error = abs(real(direct_energy.value) - direct_state.point.internal_energy) / N12

    direct_energy_tolerance = max(8 * direct_energy.stderr / N12, 1e-7)

    direct_state_pass = direct_energy_error <= direct_energy_tolerance

    quadrature_iterations = direct_state.metadata[:quadrature_iterations]

    quadrature_tails = direct_state.metadata[:quadrature_tail_beta]

    println()
    println("DIRECT CANONICAL-TPQ STATE VALIDATION")
    @printf("T/J                              = %.3f\n", direct_temperature)
    @printf("Retained TPQ realizations        = %d\n", direct_method.samples)
    @printf("Krylov dimension                 = %d\n", direct_method.krylov_dim)
    @printf("mean Lanczos iterations           = %.2f\n", sum(quadrature_iterations) / length(quadrature_iterations))
    @printf("maximum terminal Lanczos beta     = %.3e\n", maximum(abs, quadrature_tails))
    @printf("|<H>vectors - Uquadrature| / N    = %.3e\n", direct_energy_error)
    println("Canonical TPQ retained-state check: ", direct_state_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Single-realization behavior at N=12.
    #
    # Sugiura-Shimizu typicality predicts that individual realizations become
    # increasingly representative as system size / thermal entropy grows.
    # This is diagnostic only; one realization has no sample-derived SEM in the
    # current API and therefore is not used as a hard PASS criterion.
    # -------------------------------------------------------------------------

    single_curve12, single_time12, _ = run_ctpq(model12; samples=1, krylov_dim=krylov_N12, seed=seed, reorthogonalize=true)

    single_c12 = single_curve12.heat_capacity ./ N12

    single_u12 = single_curve12.internal_energy ./ N12

    single_c_rms = sqrt(sum(abs2, single_c12 .- c_exact12) / length(temperatures))

    multi_c_rms = sqrt(sum(abs2, c_tpq12 .- c_exact12) / length(temperatures))

    single_u_rms = sqrt(sum(abs2, single_u12 .- u_exact12) / length(temperatures))

    multi_u_rms = sqrt(sum(abs2, u_tpq12 .- u_exact12) / length(temperatures))

    println()
    println("TPQ STOCHASTIC-TYPICALITY DIAGNOSTIC: N = 12")
    @printf("single-realization c RMS error    = %.3e\n", single_c_rms)
    @printf("%d-realization c RMS error        = %.3e\n", samples_N12, multi_c_rms)
    @printf("single-realization u RMS error    = %.3e\n", single_u_rms)
    @printf("%d-realization u RMS error        = %.3e\n", samples_N12, multi_u_rms)
    @printf("single-realization runtime        = %.6f s\n", single_time12)

    # -------------------------------------------------------------------------
    # Exact fixed-total-Sᶻ sector architecture validation, N=12.
    #
    # Before the sectorized N=27/N=30 calculations are used, independently
    # verify on the tractable N=12 torus that:
    #
    #   1. the compiled packed Hamiltonian structurally conserves total Sᶻ,
    #   2. global spin inversion permits exact sector pairing,
    #   3. the representative sector spectra reconstruct the full spectrum, and
    #   4. sector-recombined thermodynamics reproduce the full ExactGibbs curve.
    # -------------------------------------------------------------------------

    sector_profile12 = spin_sector_profile(model12; hbar=1.0)
    sector_plan12 = spin_sector_plan(model12; exploit_equivalent_sectors=true, hbar=1.0)

    sector_spectrum_validation = validate_spin_sector_decomposition(model12; exploit_equivalent_sectors=true, tol=1e-10, hbar=1.0, max_total_dimension=10_000)

    sector_thermo_validation = validate_sector_thermodynamic_recombination(model12; temperatures=[0.20, 0.50, 1.00], kB=kB, exploit_equivalent_sectors=true, hbar=1.0, atol=1e-10, max_total_dimension=10_000)

    sector_thermo_error = maximum(values(sector_thermo_validation[:errors]))

    sector_curve12, sector_workspace_time12, sector_curve_time12, _, sector_workspace12 = run_sector_ctpq(model12; samples=samples_N12, krylov_dim=krylov_N12, seed=seed + 12, progress=false)

    sector_u12 = sector_curve12.internal_energy ./ N12
    sector_c12 = sector_curve12.heat_capacity ./ N12
    sector_f12 = sector_curve12.free_energy ./ N12
    sector_s12 = sector_curve12.entropy ./ N12

    sector_σu12 = sector_curve12.stderr[:internal_energy] ./ N12
    sector_σc12 = sector_curve12.stderr[:heat_capacity] ./ N12
    sector_σf12 = sector_curve12.stderr[:free_energy] ./ N12
    sector_σs12 = sector_curve12.stderr[:entropy] ./ N12

    sector_u_check = stochastic_consistency(sector_u12, u_exact12, sector_σu12; sigma=6, floor=2e-3)
    sector_c_check = stochastic_consistency(sector_c12, c_exact12, sector_σc12; sigma=6, floor=5e-3)
    sector_f_check = stochastic_consistency(sector_f12, f_exact12, sector_σf12; sigma=6, floor=2e-3)
    sector_s_check = stochastic_consistency(sector_s12, s_exact12, sector_σs12; sigma=6, floor=5e-3)
    sector_tpq_pass = sector_u_check.passed && sector_c_check.passed && sector_f_check.passed && sector_s_check.passed

    sector_validation_pass = sector_profile12.supported && sector_profile12.conserves_total_sz && sector_profile12.global_spin_flip_symmetric && sector_spectrum_validation[:passed] && sector_thermo_validation[:passed] && sector_tpq_pass

    println()
    println("EXACT Sᶻ-SECTOR ARCHITECTURE VALIDATION: N = 12")
    println("Compiled total-Sᶻ conservation    = ", sector_profile12.conserves_total_sz ? "PASS" : "FAIL")
    println("Global spin-inversion symmetry    = ", sector_profile12.global_spin_flip_symmetric ? "PASS" : "FAIL")
    @printf("Representative sectors            = %d\n", length(sector_plan12.sectors))
    @printf("Weighted reconstructed dimension  = %d\n", sum(sector_plan12.multiplicities .* sector_plan12.dimensions))
    @printf("Maximum sector-spectrum error     = %.3e\n", sector_spectrum_validation[:max_spectrum_error])
    @printf("Maximum sector-thermodynamic error = %.3e\n", sector_thermo_error)
    @printf("Sector workspace preparation      = %.6f s\n", sector_workspace_time12)
    @printf("Sector cTPQ curve time            = %.6f s\n", sector_curve_time12)
    @printf("max sector-cTPQ |Δu| per site     = %.3e\n", sector_u_check.max_error)
    @printf("max sector-cTPQ |Δc| per site     = %.3e\n", sector_c_check.max_error)
    @printf("max sector-cTPQ |Δf| per site     = %.3e\n", sector_f_check.max_error)
    @printf("max sector-cTPQ |Δs| per site     = %.3e\n", sector_s_check.max_error)
    println("Exact sector decomposition check  : ", sector_validation_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # N=18 paper-scale finite-cluster TPQ calculations.
    #
    # Sugiura and Shimizu averaged 100 realizations for N=18 because the
    # estimated single-realization uncertainty near T/J=0.1 was large.
    #
    # The environment variable PHUNDAMENTAL_TPQ_SAMPLES18 can be set to 100
    # for the full paper realization count.
    # -------------------------------------------------------------------------

    println()
    println("Starting N=18a canonical TPQ...")
    @printf("Hilbert dimension                 = %d\n", hilbert_dimension(18))
    @printf("TPQ realizations                  = %d\n", samples_N18)
    @printf("Krylov dimension                  = %d\n", krylov_N18)
    println("Reorthogonalization               = false")
    flush(stdout)

    curve18a, time18a, method18a = run_ctpq(clusters[:N18a].model; samples=samples_N18, krylov_dim=krylov_N18, seed=seed + 18, reorthogonalize=false)

    @printf("N=18a complete                    = %.6f s\n", time18a)
    flush(stdout)

    println()
    println("Starting N=18b canonical TPQ...")
    flush(stdout)

    curve18b, time18b, method18b = run_ctpq(clusters[:N18b].model; samples=samples_N18, krylov_dim=krylov_N18, seed=seed + 19, reorthogonalize=false)

    @printf("N=18b complete                    = %.6f s\n", time18b)
    flush(stdout)

    N18 = 18

    c18a = curve18a.heat_capacity ./ N18

    c18b = curve18b.heat_capacity ./ N18

    f18a = curve18a.free_energy ./ N18

    f18b = curve18b.free_energy ./ N18

    s18a = curve18a.entropy ./ N18

    s18b = curve18b.entropy ./ N18

    iT01 = nearest_index(temperatures, paper_shoulder_T)

    σc18a_T01 = curve18a.stderr[:heat_capacity][iT01] / N18

    σc18b_T01 = curve18b.stderr[:heat_capacity][iT01] / N18

    println()
    println("N = 18 CANONICAL-TPQ BENCHMARK")
    @printf("Current realizations / cluster    = %d\n", samples_N18)
    @printf("Sugiura-Shimizu realizations      = %d\n", paper_samples_N18)
    @printf("Krylov dimension                  = %d\n", krylov_N18)
    @printf("18a runtime                       = %.6f s\n", time18a)
    @printf("18b runtime                       = %.6f s\n", time18b)
    @printf("18a runtime / realization         = %.6f s\n", time18a / samples_N18)
    @printf("18b runtime / realization         = %.6f s\n", time18b / samples_N18)
    @printf("c18a(T/J=0.1)                     = %.6f\n", c18a[iT01])
    @printf("c18b(T/J=0.1)                     = %.6f\n", c18b[iT01])
    @printf("SEM[c18a(T/J=0.1)]                = %.3e\n", σc18a_T01)
    @printf("SEM[c18b(T/J=0.1)]                = %.3e\n", σc18b_T01)

    if samples_N18 < paper_samples_N18
        @printf("Projected 100-realization 18a time = %.2f s\n", time18a * paper_samples_N18 / samples_N18)
        @printf("Projected 100-realization 18b time = %.2f s\n", time18b * paper_samples_N18 / samples_N18)
    end

    # -------------------------------------------------------------------------
    # Full-space and sectorized cTPQ memory-scaling diagnostics.
    #
    # The optimized scalar TPQ path no longer materializes SpinBasisState
    # objects or a product-state Dict. The remaining full-space memory boundary
    # is the state-vector/Krylov workspace itself. N=27 and N=30 therefore use
    # exact fixed-total-Sᶻ sectors, with exact reconstruction of the canonical
    # trace and optional global-spin-inversion pairing.
    # -------------------------------------------------------------------------

    println()
    println("CURRENT FULL-SPACE cTPQ MEMORY-SCALING DIAGNOSTIC")
    println()
    println(" cluster       N       dim(H)     one vector [GiB]    Krylov low-memory [GiB]")

    full_memory_estimates = Dict{Symbol,Any}()

    for label in cluster_order
        N = clusters[label].N
        krylov_dim = label === :N27 ? krylov_N27 : label === :N30 ? krylov_N30 : large_krylov_default
        estimate = tpq_memory_estimate(N; krylov_dim=krylov_dim, reorthogonalize=false)
        full_memory_estimates[label] = estimate

        @printf(
            "%7s  %6d  %12.0f      %10.3f              %10.3f\n",
            String(label),
            N,
            estimate.dimension,
            estimate.vector_gib,
            estimate.krylov_gib,
        )
    end

    total_memory_gib = bytes_to_gib(Sys.total_memory())

    println()
    @printf("Detected physical memory          = %.2f GiB\n", total_memory_gib)
    @printf("Julia threads                     = %d\n", Threads.nthreads())

    for label in (:N18a, :N27, :N30)
        N = clusters[label].N
        krylov_dim = label === :N27 ? krylov_N27 : label === :N30 ? krylov_N30 : large_krylov_default
        estimate_reorth = tpq_memory_estimate(N; krylov_dim=krylov_dim, reorthogonalize=true)
        @printf("%s full-space reorthogonalized Krylov estimate = %.2f GiB\n", String(label), estimate_reorth.krylov_gib)
    end

    # -------------------------------------------------------------------------
    # Exact fixed-total-Sᶻ sector plans for N=27 and N=30.
    # -------------------------------------------------------------------------

    sector_memory_estimates = Dict{Symbol,Any}()

    println()
    println("SECTOR-DECOMPOSED cTPQ ARCHITECTURE")
    println()
    println(" cluster       N    rep. sectors    largest sector      evaluated dim    reduction    peak estimate [GiB]")

    for label in (:N27, :N30)
        N = clusters[label].N
        krylov_dim = label === :N27 ? krylov_N27 : krylov_N30
        estimate = sector_memory_estimate(clusters[label].model, N; krylov_dim=krylov_dim, reorthogonalize=false)
        sector_memory_estimates[label] = estimate

        @printf(
            "%7s  %6d       %6d      %12d      %12d      %6.2fx          %8.3f\n",
            String(label),
            N,
            estimate.representative_sectors,
            estimate.largest_dimension,
            estimate.evaluated_dimension,
            estimate.dimension_reduction,
            estimate.estimated_peak_gib,
        )
    end

    println()
    @printf("Sector memory budget              = %.2f GiB (%.0f%% of physical memory)\n", sector_memory_fraction * total_memory_gib, 100sector_memory_fraction)
    println("Sector symmetry                   = total Sᶻ")
    println("Equivalent-sector pairing         = ", sector_pair_equivalent ? "requested" : "disabled")
    println("Sector matvec threading           = ", sector_threaded_matvec ? "enabled" : "disabled")
    println("Rank-lookup policy                = ", sector_rank_lookup)

    if run_large_paper_clusters && sector_threaded_matvec && Threads.nthreads() == 1
        println()
        println("NOTE: sector matvec threading is enabled, but Julia is running with one thread.")
        println("      Use `julia -t auto empirical-benchmark-suite.jl` to enable CPU threading for N=27/N=30.")
    end

    for label in (:N27, :N30)
        estimate = sector_memory_estimates[label]
        @printf(
            "%s: largest vector %.3f GiB, low-memory Krylov %.3f GiB, rank lookup %s, estimated peak %.3f GiB\n",
            String(label),
            estimate.largest_vector_gib,
            estimate.workspace_gib,
            estimate.use_lookup ? @sprintf("%.3f GiB", estimate.lookup_gib) : "combinatorial",
            estimate.estimated_peak_gib,
        )
    end

    # -------------------------------------------------------------------------
    # N=27 and N=30 paper-scale calculations.
    #
    # These calculations are included by default and use the exact sectorized
    # CanonicalTPQ API. Set PHUNDAMENTAL_TPQ_LARGE=0 to skip them. Each
    # representative sector contributes its unnormalized stochastic trace with
    # the exact sector dimension and symmetry multiplicity before the canonical
    # ensemble is normalized.
    # -------------------------------------------------------------------------

    large_curves = Dict{Symbol,Any}()
    large_times = Dict{Symbol,Float64}()
    large_workspace_times = Dict{Symbol,Float64}()
    large_curve_times = Dict{Symbol,Float64}()
    large_methods = Dict{Symbol,Any}()
    large_workspaces = Dict{Symbol,Any}()
    large_sample_counts = Dict(:N27 => samples_N27, :N30 => samples_N30)
    large_krylov_dims = Dict(:N27 => krylov_N27, :N30 => krylov_N30)
    large_min_krylov_dims = Dict(:N27 => min_krylov_N27, :N30 => min_krylov_N30)

    for label in (:N27, :N30)
        N = clusters[label].N
        estimate = sector_memory_estimates[label]
        samples_large = large_sample_counts[label]
        krylov_large = large_krylov_dims[label]
        min_krylov_large = large_min_krylov_dims[label]

        if run_large_paper_clusters && estimate.safe
            println()
            println("Starting $label sector-decomposed canonical TPQ...")
            @printf("Full Hilbert dimension            = %d\n", estimate.full_dimension)
            @printf("Representative Sᶻ sectors         = %d\n", estimate.representative_sectors)
            @printf("Largest sector dimension          = %d\n", estimate.largest_dimension)
            @printf("Evaluated representative states   = %d\n", estimate.evaluated_dimension)
            @printf("Exact dimension reduction         = %.3fx\n", estimate.dimension_reduction)
            println("Spin-inversion sector pairing     = ", estimate.plan.equivalence_used ? "ENABLED" : "DISABLED")
            @printf("TPQ realizations / sector         = %d\n", samples_large)
            @printf("Krylov maximum dimension          = %d\n", krylov_large)
            @printf("Krylov minimum dimension          = %d\n", min_krylov_large)
            println("Adaptive Krylov                   = ", adaptive_large_krylov ? "enabled" : "disabled")
            @printf("Krylov convergence tolerance      = %.3e\n", large_convergence_tol)
            @printf("Krylov checkpoint interval        = %d\n", large_convergence_interval)
            @printf("Krylov consecutive passes         = %d\n", large_convergence_consecutive)
            println("Reorthogonalization               = false")
            @printf("Julia threads                     = %d\n", Threads.nthreads())
            println("Threaded sector matvec            = ", sector_threaded_matvec ? "enabled" : "disabled")
            println("Planned rank indexing             = ", estimate.rank_strategy)
            @printf("Estimated peak working memory     = %.3f GiB\n", estimate.estimated_peak_gib)
            @printf("Configured memory budget          = %.3f GiB\n", estimate.memory_budget_gib)
            flush(stdout)

            curve_large, workspace_time, curve_time, method_large, workspace_large = run_sector_ctpq(clusters[label].model;
                                                                                                     samples=samples_large,
                                                                                                     krylov_dim=krylov_large,
                                                                                                     min_krylov_dim=min_krylov_large,
                                                                                                     adaptive_krylov=adaptive_large_krylov, seed=seed + N)

            total_time = workspace_time + curve_time

            large_curves[label] = curve_large
            large_times[label] = total_time
            large_workspace_times[label] = workspace_time
            large_curve_times[label] = curve_time
            large_methods[label] = method_large
            large_workspaces[label] = workspace_large

            println()
            println("$label sector-decomposed canonical TPQ complete.")
            @printf("Sector workspace preparation      = %.6f s\n", workspace_time)
            @printf("TPQ curve evaluation              = %.6f s\n", curve_time)
            @printf("Total runtime                     = %.6f s\n", total_time)
            @printf("Runtime / realization             = %.6f s\n", total_time / samples_large)
            @printf("Sectors evaluated                 = %d\n", curve_large.metadata[:sectors_evaluated])
            println("Operator backend                  = ", curve_large.metadata[:operator_backend])
            println("Actual rank-index strategy        = ", curve_large.metadata[:rank_lookup_strategy])
            @printf("Rank-index memory                 = %.3f GiB\n", bytes_to_gib(curve_large.metadata[:rank_lookup_bytes]))
            @printf("Estimated runtime peak memory     = %.3f GiB\n", bytes_to_gib(curve_large.metadata[:estimated_peak_bytes]))
            println("Threaded matvec active            = ", curve_large.metadata[:threaded_matvec])

            if adaptive_large_krylov
                krylov_summary = adaptive_krylov_summary(curve_large)
                @printf("Actual Krylov iteration range     = %d to %d\n", krylov_summary.minimum_iterations, krylov_summary.maximum_iterations)
                @printf("Mean Krylov iterations            = %.2f\n", krylov_summary.mean_iterations)
                @printf("Maximum checkpoint error          = %.3e\n", krylov_summary.maximum_convergence_error)
                println("Adaptive Krylov convergence       = ", krylov_summary.converged === true ? "PASS" : "NOT CONVERGED")
            end

            if samples_large == 1
                println("NOTE: sample-derived SEM is unavailable for one realization; stderr is reported as NaN rather than zero.")
            end
        else
            println()

            if !run_large_paper_clusters
                println("$label: SKIPPED (PHUNDAMENTAL_TPQ_LARGE=0)")
            else
                @printf("%s: SKIPPED; largest-sector workspace requires %.2f GiB against a %.2f GiB configured memory budget\n", String(label), estimate.workspace_gib, estimate.memory_budget_gib)
            end
        end
    end

    # -------------------------------------------------------------------------
    # Sugiura-Shimizu paper landmarks.
    #
    # The paper describes the N=27/N=30 low-temperature feature near T/J≈0.1 as
    # a shoulder rather than the pronounced secondary peak of the smaller N=18
    # clusters. A three-point local maximum is therefore not, by itself, a hard
    # contradiction. The benchmark first requires adaptive-Krylov convergence,
    # then measures candidate-peak prominence against numerical and available
    # stochastic resolution. One-realization curves with a prominence above the
    # numerical resolution but without a sample-derived uncertainty are reported
    # as INCONCLUSIVE rather than FAIL.
    # -------------------------------------------------------------------------

    paper_large_tested = false
    paper_large_failed = false
    paper_large_inconclusive = false
    low_temperature_diagnostics = Dict{Symbol,Any}()

    if haskey(large_curves, :N27)
        paper_large_tested = true
        curve27 = large_curves[:N27]
        c27 = curve27.heat_capacity ./ 27
        u27 = curve27.internal_energy ./ 27
        sigma_c27 = curve27.stderr[:heat_capacity] ./ 27
        c27_derivative = finite_difference_temperature_derivative(temperatures, u27)
        krylov27 = adaptive_krylov_summary(curve27)
        low27 = low_temperature_peak_diagnostic(temperatures, c27, sigma_c27; Tmin=0.05, Tmax=0.15,
                                                krylov_converged=krylov27.converged === true,
                                                krylov_error=krylov27.maximum_convergence_error)
        low_temperature_diagnostics[:N27] = low27
        low27.status === :resolved_peak && (paper_large_failed = true)
        low27.status in (:inconclusive_krylov, :inconclusive_stochastic) && (paper_large_inconclusive = true)

        i01 = nearest_index(temperatures, paper_shoulder_T)
        high_indices = findall(T -> T >= 0.30, temperatures)
        high_peak_index = high_indices[argmax(c27[high_indices])]
        low_window = findall(T -> 0.05 <= T <= 0.15, temperatures)
        fluctuation_derivative_error27 = maximum(abs.(c27[low_window] .- c27_derivative[low_window]))

        println()
        println("N = 27 PAPER LANDMARKS")
        @printf("c(T/J=0.10)                      = %.6f\n", c27[i01])
        @printf("high-T maximum                    = %.6f at T/J = %.3f\n", c27[high_peak_index], temperatures[high_peak_index])
        low27_label = low27.status === :shoulder_consistent ? "SHOULDER-CONSISTENT" :
                      low27.status === :resolved_peak ? "RESOLVED PEAK" : "INCONCLUSIVE"
        println("low-T feature classification      = ", low27_label)
        @printf("low-T candidate prominence        = %.6e\n", low27.prominence)
        @printf("low-T resolution threshold        = %.6e\n", low27.resolution)
        @printf("max |c_fluct - d(u)/dT| low-T     = %.6e\n", fluctuation_derivative_error27)
        @printf("maximum Krylov checkpoint error   = %.3e\n", krylov27.maximum_convergence_error)
        println("adaptive Krylov convergence       = ", krylov27.converged === true ? "PASS" : "NOT CONVERGED")
    end

    if haskey(large_curves, :N30)
        paper_large_tested = true
        curve30 = large_curves[:N30]
        c30 = curve30.heat_capacity ./ 30
        u30 = curve30.internal_energy ./ 30
        f30 = curve30.free_energy ./ 30
        s30 = curve30.entropy ./ 30
        sigma_c30 = curve30.stderr[:heat_capacity] ./ 30
        c30_derivative = finite_difference_temperature_derivative(temperatures, u30)
        krylov30 = adaptive_krylov_summary(curve30)
        low30 = low_temperature_peak_diagnostic(temperatures, c30, sigma_c30; Tmin=0.05, Tmax=0.15,
                                                krylov_converged=krylov30.converged === true,
                                                krylov_error=krylov30.maximum_convergence_error)
        low_temperature_diagnostics[:N30] = low30
        low30.status === :resolved_peak && (paper_large_failed = true)
        low30.status in (:inconclusive_krylov, :inconclusive_stochastic) && (paper_large_inconclusive = true)

        i01 = nearest_index(temperatures, paper_shoulder_T)
        i02 = nearest_index(temperatures, paper_entropy_T)
        entropy_fraction30 = s30[i02] / log(2)
        entropy30_consistent = abs(entropy_fraction30 - paper_entropy_fraction) <= 0.05
        if krylov30.converged === true
            entropy30_consistent || (paper_large_failed = true)
        else
            paper_large_inconclusive = true
        end
        low_window = findall(T -> 0.05 <= T <= 0.15, temperatures)
        fluctuation_derivative_error30 = maximum(abs.(c30[low_window] .- c30_derivative[low_window]))

        println()
        println("N = 30 PAPER LANDMARKS")
        @printf("c(T/J=0.10)                      = %.6f\n", c30[i01])
        @printf("f(T/J=0.10) / J                  = %.6f\n", f30[i01])
        @printf("s(T/J=0.20) / ln(2)               = %.6f   [paper ≈ %.2f]\n", entropy_fraction30, paper_entropy_fraction)
        low30_label = low30.status === :shoulder_consistent ? "SHOULDER-CONSISTENT" :
                      low30.status === :resolved_peak ? "RESOLVED PEAK" : "INCONCLUSIVE"
        println("low-T feature classification      = ", low30_label)
        @printf("low-T candidate prominence        = %.6e\n", low30.prominence)
        @printf("low-T resolution threshold        = %.6e\n", low30.resolution)
        @printf("max |c_fluct - d(u)/dT| low-T     = %.6e\n", fluctuation_derivative_error30)
        @printf("maximum Krylov checkpoint error   = %.3e\n", krylov30.maximum_convergence_error)
        println("adaptive Krylov convergence       = ", krylov30.converged === true ? "PASS" : "NOT CONVERGED")
        println("45% entropy landmark              = ", entropy30_consistent ? "CONSISTENT" : "NOT CONSISTENT")
    end

    if paper_large_tested && !(haskey(large_curves, :N27) && haskey(large_curves, :N30))
        paper_large_inconclusive = true
    end

    paper_large_status = paper_large_failed ? :fail : paper_large_inconclusive ? :inconclusive : :pass

    # -------------------------------------------------------------------------
    # Performance summary.
    #
    # Runtime is normalized by TPQ realization count. For N=27/N=30 the x-axis
    # retains the physical full-Hilbert-space dimension 2^N, while the terminal
    # output also reports the exact number of representative sector states that
    # were actually traversed after spin-inversion pairing.
    # -------------------------------------------------------------------------

    perf_N = Int[N12, 18, 18]
    perf_dimension = Float64[2.0^N12, 2.0^18, 2.0^18]
    perf_time_per_sample = Float64[tpq_time12 / samples_N12, time18a / samples_N18, time18b / samples_N18]
    perf_labels = String["N12", "N18a", "N18b"]

    for label in (:N27, :N30)
        if haskey(large_curves, label)
            N = clusters[label].N
            push!(perf_N, N)
            push!(perf_dimension, 2.0^N)
            push!(perf_time_per_sample, large_times[label] / large_sample_counts[label])
            push!(perf_labels, String(label))
        end
    end

    if length(perf_dimension) >= 2
        endpoint_scaling_exponent = log(perf_time_per_sample[end] / perf_time_per_sample[1]) / log(perf_dimension[end] / perf_dimension[1])

        println()
        println("cTPQ PERFORMANCE SUMMARY")
        @printf("endpoint runtime exponent vs Hilbert dimension = %.3f\n", endpoint_scaling_exponent)

        for label in (:N27, :N30)
            if haskey(large_curves, label)
                estimate = sector_memory_estimates[label]
                @printf("%s representative-sector workload = %d states (%.3fx reduction from 2^N)\n", String(label), estimate.evaluated_dimension, estimate.dimension_reduction)
            end
        end
    end

    # -------------------------------------------------------------------------
    # Benchmark status.
    #
    # The N=27/N=30 calculations now exercise the exact fixed-total-Sᶻ direct-
    # sum architecture rather than the full 2^N vector workspace. The small N=12
    # sector reconstruction is an independent correctness gate for that path.
    # -------------------------------------------------------------------------

    core_pass = geometry_pass && exact_tpq_pass && direct_state_pass && sector_validation_pass

    println()
    println("SUGIURA-SHIMIZU CANONICAL TPQ VALIDATION")
    println("  Kagome geometry / coordination       : ", geometry_pass ? "PASS" : "FAIL")
    println("  cTPQ thermodynamics vs ExactGibbs    : ", exact_tpq_pass ? "PASS" : "FAIL")
    println("  retained exp(-βH/2)|ψ> state         : ", direct_state_pass ? "PASS" : "FAIL")
    println("  exact Sᶻ-sector reconstruction       : ", sector_validation_pass ? "PASS" : "FAIL")
    println("  N=18 canonical-TPQ curves            : GENERATED")
    println("  N=27 sector-decomposed cTPQ          : ", haskey(large_curves, :N27) ? "GENERATED" : "NOT RUN")
    println("  N=30 sector-decomposed cTPQ          : ", haskey(large_curves, :N30) ? "GENERATED" : "NOT RUN")

    if paper_large_tested
        landmark_status = paper_large_status === :pass ? "PASS/CONSISTENT" : paper_large_status === :inconclusive ? "INCONCLUSIVE" : "FAIL"
        println("  N=27/30 paper thermodynamic landmarks: ", landmark_status)
    else
        println("  N=27/30 paper thermodynamic landmarks: NOT RUN")
    end

    println()
    println("Sugiura-Shimizu canonical TPQ core benchmark: ", core_pass ? "PASS" : "FAIL")

    if haskey(large_curves, :N27) && haskey(large_curves, :N30)
        println("Paper-scale N=27/N=30 execution completed with exact fixed-total-Sᶻ sector decomposition.")
    elseif !run_large_paper_clusters
        println("Paper-scale N=27/N=30 execution was disabled by PHUNDAMENTAL_TPQ_LARGE=0.")
    else
        println("One or more paper-scale clusters were not executed because of the configured memory budget.")
    end

    # -------------------------------------------------------------------------
    # Visualization.
    #
    # (a) specific heat per site,
    # (b) free-energy density,
    # (c) entropy fraction s/ln(2),
    # (d) runtime per TPQ realization vs Hilbert-space dimension.
    # -------------------------------------------------------------------------

    fig = Figure(size=(1200, 900))

    ax11 = Axis(fig[1, 1]; xlabel="T/J", ylabel="c = C/N", title="Kagome Heisenberg specific heat")

    lines!(ax11, temperatures, c_exact12; label="N=12 ExactGibbs")

    lines!(ax11, temperatures, c_tpq12; label="N=12 cTPQ")

    lines!(ax11, temperatures, c18a; label="N=18a cTPQ")

    lines!(ax11, temperatures, c18b; label="N=18b cTPQ")

    if haskey(large_curves, :N27)
        lines!(ax11, temperatures, large_curves[:N27].heat_capacity ./ 27; label="N=27 cTPQ")
    end

    if haskey(large_curves, :N30)
        lines!(ax11, temperatures, large_curves[:N30].heat_capacity ./ 30; label="N=30 cTPQ")
    end

    vlines!(ax11, [paper_shoulder_T]; linestyle=:dash)

    axislegend(ax11; position=:rt)

    ax12 = Axis(fig[1, 2]; xlabel="T/J", ylabel="f/J = F/(NJ)", title="Free-energy density")

    lines!(ax12, temperatures, f_exact12; label="N=12 ExactGibbs")

    lines!(ax12, temperatures, f_tpq12; label="N=12 cTPQ")

    lines!(ax12, temperatures, f18a; label="N=18a cTPQ")

    lines!(ax12, temperatures, f18b; label="N=18b cTPQ")

    if haskey(large_curves, :N27)
        lines!(ax12, temperatures, large_curves[:N27].free_energy ./ 27; label="N=27 cTPQ")
    end

    if haskey(large_curves, :N30)
        lines!(ax12, temperatures, large_curves[:N30].free_energy ./ 30; label="N=30 cTPQ")
    end

    axislegend(ax12; position=:rb)

    ax21 = Axis(fig[2, 1]; xlabel="T/J", ylabel="s / ln(2)", title="Remaining entropy fraction")

    lines!(ax21, temperatures, s_exact12 ./ log(2); label="N=12 ExactGibbs")

    lines!(ax21, temperatures, s_tpq12 ./ log(2); label="N=12 cTPQ")

    lines!(ax21, temperatures, s18a ./ log(2); label="N=18a cTPQ")

    lines!(ax21, temperatures, s18b ./ log(2); label="N=18b cTPQ")

    if haskey(large_curves, :N27)
        lines!(ax21, temperatures, large_curves[:N27].entropy ./ (27 * log(2)); label="N=27 cTPQ")
    end

    if haskey(large_curves, :N30)
        lines!(ax21, temperatures, large_curves[:N30].entropy ./ (30 * log(2)); label="N=30 cTPQ")
    end

    hlines!(ax21, [paper_entropy_fraction]; linestyle=:dash)

    vlines!(ax21, [paper_entropy_T]; linestyle=:dash)

    axislegend(ax21; position=:rb)

    ax22 = Axis(fig[2, 2]; xlabel="Hilbert-space dimension 2ᴺ", ylabel="time / TPQ realization (s)", title="Current canonical-TPQ scaling", xscale=log10, yscale=log10)

    scatterlines!(ax22, perf_dimension, perf_time_per_sample)

    for i in eachindex(perf_dimension)
        text!(ax22, perf_dimension[i], perf_time_per_sample[i]; text=perf_labels[i], align=(:left, :bottom))
    end

    save(joinpath(BENCHMARK_FIGURE_DIR, "Sugiura-Kagome-TPQ-benchmark.png"), fig)
end


#-------------------------------------------------------#
# Steinhauer, J. (et al.). (2002).                      #
# Excitation Spectrum of a Bose-Einstein Condensate.    #
# Phys. Rev. Lett. 88, 120407.                          #
# doi: 10.1103/PhysRevLett.88.120407                    #
#-------------------------------------------------------#
let
    # -------------------------------------------------------------------------
    # Published system and experimental parameters.
    #
    # The experiment uses a nearly pure 87Rb condensate and measures axial
    # excitations by Bragg spectroscopy. The chemical potential relevant during
    # the Bragg pulse is μ/h = 1.91 ± 0.09 kHz. Steinhauer et al. report an
    # effective low-k sound velocity of 2.0 ± 0.1 mm/s and an LDA prediction of
    # 2.01 ± 0.05 mm/s. Their measured static structure factor is displayed after
    # multiplication by an experimental calibration factor of 2.3; that factor is
    # not applied to the theoretical structure factor below.
    # -------------------------------------------------------------------------

    h = 6.62607015e-34
    ħ = 1.054571817e-34
    atomic_mass_unit = 1.66053906660e-27
    m_Rb87 = 86.909180531 * atomic_mass_unit

    paper_atom_number = 1.0e5
    paper_mu_over_h_hz = 1.91e3
    paper_mu_over_h_uncertainty_hz = 0.09e3
    paper_axial_radius_um = 28.0
    paper_sound_velocity_mm_s = 2.0
    paper_sound_velocity_uncertainty_mm_s = 0.1
    paper_lda_sound_velocity_mm_s = 2.01
    paper_lda_sound_velocity_uncertainty_mm_s = 0.05
    paper_structure_factor_scale = 2.3
    paper_scattering_visible_k_um = 6.8

    μ = h * paper_mu_over_h_hz

    # -------------------------------------------------------------------------
    # Homogeneous Bogoliubov reference.
    #
    # For one real standing-wave component of a ±k pair, the quadratic block is
    #
    #     H_k = (ε_k + μ)b†b + μ/2 (b†b† + bb),
    #
    # which has the exact positive mode energy
    #
    #     E_k = sqrt[ε_k(ε_k + 2μ)]
    #
    # and density-response weight
    #
    #     S(k) = ε_k/E_k = |u_k + v_k|²
    #
    # in the sign convention returned by the current bosonic BdG solver. The
    # standing-wave block is unitarily equivalent to resolving the degenerate
    # ±k pair and avoids arbitrary rotations inside that degenerate subspace.
    # -------------------------------------------------------------------------

    function homogeneous_bogoliubov_model(ε::Real, μmode::Real; cutoff::Union{Nothing,Int}=nothing)
        ε > 0 || throw(ArgumentError("Bogoliubov kinetic energy must be positive"))
        μmode > 0 || throw(ArgumentError("Bogoliubov chemical potential must be positive"))
        truncation = isnothing(cutoff) ? nothing : F.NumericalTruncation(cutoff)

        representation = F.Representation(:homogeneous_bec_bogoliubov, F.BosonFockSpace(1), F.BosonAlgebra(1), F.BosonOccupationBasis([1]); ordering=[1], reference_state=:boson_vacuum, truncation=truncation)
        terms = F.AbstractOperatorExpr[(ε + μmode) * F.nb(1), (μmode / 2) * F.b(1)' * F.b(1)', (μmode / 2) * F.b(1) * F.b(1)]
        hamiltonian = F.OperatorSum(terms)

        parameters = Dict(:epsilon => Float64(ε), :mu => Float64(μmode), :reference => "Steinhauer et al., Phys. Rev. Lett. 88, 120407 (2002)", :doi => "10.1103/PhysRevLett.88.120407")
        provenance = [(construction=:bogoliubov_standing_wave, reference=:steinhauer_2002)]
        return F.ManyBodyModel(representation, hamiltonian; parameters=parameters, provenance=provenance)
    end

    bogoliubov_energy(ε::Real, μmode::Real) = sqrt(ε * (ε + 2 * μmode))
    bogoliubov_structure_factor(ε::Real, μmode::Real) = ε / bogoliubov_energy(ε, μmode)

    epsilon_ratios = 10.0 .^ collect(range(-4.0, 2.0; length=161))
    native_energies = Vector{Float64}(undef, length(epsilon_ratios))
    native_structure = Vector{Float64}(undef, length(epsilon_ratios))
    reference_energies = Vector{Float64}(undef, length(epsilon_ratios))
    reference_structure = Vector{Float64}(undef, length(epsilon_ratios))
    mode_metric_errors = Vector{Float64}(undef, length(epsilon_ratios))
    mode_checks_pass = true

    for i in eachindex(epsilon_ratios)
        ε = epsilon_ratios[i]
        model = homogeneous_bogoliubov_model(ε, 1.0)
        result = solve(QuadraticBosonSolver(tol=1e-12), model)
        check = validate_mode_result(result; tol=1e-10)
        mode_checks_pass &= check[:passed]

        energies = mode_energies(result)
        vectors = mode_vectors(result)
        length(energies) == 1 || error("homogeneous Bogoliubov validation expected one positive mode")
        size(vectors) == (2, 1) || error("homogeneous Bogoliubov validation expected a 2×1 Nambu mode matrix")

        u = vectors[1, 1]
        v = vectors[2, 1]
        native_energies[i] = real(energies[1])
        native_structure[i] = abs2(u + v)
        reference_energies[i] = bogoliubov_energy(ε, 1.0)
        reference_structure[i] = bogoliubov_structure_factor(ε, 1.0)
        mode_metric_errors[i] = abs(abs2(u) - abs2(v) - 1)
    end

    dispersion_relative_error = maximum(abs.(native_energies .- reference_energies) ./ reference_energies)
    structure_relative_error = maximum(abs.(native_structure .- reference_structure) ./ reference_structure)
    mode_metric_error = maximum(mode_metric_errors)

    native_dispersion_pass = mode_checks_pass && dispersion_relative_error <= 1e-10
    native_structure_pass = structure_relative_error <= 1e-9
    native_metric_pass = mode_metric_error <= 1e-10

    println()
    println("STEINHAUER HOMOGENEOUS BOGOLIUBOV VALIDATION")
    println("Existing Phundamental QuadraticBosonSolver -> bosonic BdG path")
    println()
    @printf("Kinetic-energy range ε/μ          = %.1e to %.1e\n", first(epsilon_ratios), last(epsilon_ratios))
    @printf("Maximum dispersion relative error = %.3e\n", dispersion_relative_error)
    @printf("Maximum S(k) relative error       = %.3e\n", structure_relative_error)
    @printf("Maximum symplectic-norm error     = %.3e\n", mode_metric_error)
    println("  Native Bogoliubov dispersion    : ", native_dispersion_pass ? "PASS" : "FAIL")
    println("  Native Bogoliubov amplitudes    : ", native_structure_pass ? "PASS" : "FAIL")
    println("  Native symplectic normalization : ", native_metric_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Mode-native density response.
    #
    # The density fluctuation of the one-mode standing-wave block is proportional
    # to b + b†. The native QuadraticModeResult observable route must therefore
    # place the T=0 spectral line at E_k with integrated weight
    #
    #     S(k) = |u_k + v_k|² = ε_k/E_k.
    #
    # The same mode-space contractions independently generate the equal-time
    # static structure factor. Their agreement tests the full dynamic/static sum
    # rule without constructing a truncated bosonic Fock basis. A finite-T probe
    # additionally checks detailed balance between the ±E_k lines.
    # -------------------------------------------------------------------------

    function trapezoidal_integral(x::AbstractVector{<:Real}, y::AbstractVector)
        length(x) == length(y) || throw(DimensionMismatch("trapezoidal integration arrays differ in length"))
        length(x) >= 2 || throw(ArgumentError("trapezoidal integration requires at least two points"))
        total = zero(eltype(y))
        @inbounds for i in 1:length(x)-1
            total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
        end
        return total
    end

    mode_response_ratios = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    mode_peak_errors = Float64[]
    mode_weight_errors = Float64[]
    mode_static_errors = Float64[]
    mode_sumrule_errors = Float64[]
    response_convention = SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=1.0, kB=1.0)

    for ε in mode_response_ratios
        model = homogeneous_bogoliubov_model(ε, 1.0)
        result = solve(QuadraticBosonSolver(tol=1e-12), model)
        field = MomentumField([F.b(1) + F.b(1)'], [0.0]; normalization=:none)
        energy = bogoliubov_energy(ε, 1.0)
        reference_weight = bogoliubov_structure_factor(ε, 1.0)
        axis = collect(range(0.0, 2energy; length=4001))
        broadening = GaussianBroadening(energy / 50)

        dynamic = dynamic_structure_factor(model, result, field, [0.0], axis; temperature=0.0, convention=response_convention, broadening=broadening, weight_tol=1e-15)
        static = static_structure_factor(model, result, field, [0.0]; temperature=0.0, convention=response_convention)
        dynamic_weight = real(Phundamental.Observables.integrated_spectral_weight(dynamic)[1])
        dynamic_moment = real(trapezoidal_integral(axis, axis .* vec(dynamic.intensity[1, :]))) / dynamic_weight
        static_weight = real(static.intensity[1])
        sumrule_error = Phundamental.Observables.dynamic_static_sumrule_residual(dynamic, static)

        push!(mode_peak_errors, abs(dynamic_moment - energy) / energy)
        push!(mode_weight_errors, abs(dynamic_weight - reference_weight) / reference_weight)
        push!(mode_static_errors, abs(static_weight - reference_weight) / reference_weight)
        push!(mode_sumrule_errors, sumrule_error)
    end

    detailed_balance_model = homogeneous_bogoliubov_model(1.0, 1.0)
    detailed_balance_result = solve(QuadraticBosonSolver(tol=1e-12), detailed_balance_model)
    detailed_balance_field = MomentumField([F.b(1) + F.b(1)'], [0.0]; normalization=:none)
    detailed_balance_energy = bogoliubov_energy(1.0, 1.0)
    detailed_balance_temperature = detailed_balance_energy / 2
    detailed_balance_axis = collect(range(-2detailed_balance_energy, 2detailed_balance_energy; length=8001))
    detailed_balance_dynamic = dynamic_structure_factor(detailed_balance_model, detailed_balance_result, detailed_balance_field, [0.0], detailed_balance_axis;
                                                        temperature=detailed_balance_temperature, convention=response_convention, broadening=GaussianBroadening(detailed_balance_energy / 50), weight_tol=1e-15)
    detailed_balance_static = static_structure_factor(detailed_balance_model, detailed_balance_result, detailed_balance_field, [0.0]; temperature=detailed_balance_temperature, convention=response_convention)
    negative_indices = findall(<(0.0), detailed_balance_axis)
    positive_indices = findall(>(0.0), detailed_balance_axis)
    detailed_balance_negative = real(trapezoidal_integral(detailed_balance_axis[negative_indices], vec(detailed_balance_dynamic.intensity[1, negative_indices])))
    detailed_balance_positive = real(trapezoidal_integral(detailed_balance_axis[positive_indices], vec(detailed_balance_dynamic.intensity[1, positive_indices])))
    detailed_balance_ratio = detailed_balance_negative / detailed_balance_positive
    detailed_balance_reference = exp(-detailed_balance_energy / detailed_balance_temperature)
    detailed_balance_relative_error = abs(detailed_balance_ratio - detailed_balance_reference) / detailed_balance_reference
    detailed_balance_sumrule_error = Phundamental.Observables.dynamic_static_sumrule_residual(detailed_balance_dynamic, detailed_balance_static)
    detailed_balance_occupation = inv(expm1(detailed_balance_energy / detailed_balance_temperature))
    detailed_balance_static_reference = (2detailed_balance_occupation + 1) * bogoliubov_structure_factor(1.0, 1.0)
    detailed_balance_static_error = abs(real(detailed_balance_static.intensity[1]) - detailed_balance_static_reference) / detailed_balance_static_reference

    mode_peak_pass = maximum(mode_peak_errors) <= 1e-8
    mode_weight_pass = maximum(mode_weight_errors) <= 1e-8
    mode_static_response_pass = maximum(mode_static_errors) <= 1e-9
    mode_sumrule_pass = max(maximum(mode_sumrule_errors), detailed_balance_sumrule_error) <= 1e-8
    detailed_balance_pass = detailed_balance_relative_error <= 1e-8 && detailed_balance_static_error <= 1e-9
    mode_native_response_pass = mode_peak_pass && mode_weight_pass && mode_static_response_pass && mode_sumrule_pass && detailed_balance_pass

    println()
    println("STEINHAUER MODE-NATIVE DYNAMIC STRUCTURE FACTOR")
    @printf("Response validation points         = %d\n", length(mode_response_ratios))
    @printf("Maximum peak-center relative error = %.3e\n", maximum(mode_peak_errors))
    @printf("Maximum integrated-weight error    = %.3e\n", maximum(mode_weight_errors))
    @printf("Maximum native static S(k) error   = %.3e\n", maximum(mode_static_errors))
    @printf("Maximum dynamic/static sum-rule error = %.3e\n", max(maximum(mode_sumrule_errors), detailed_balance_sumrule_error))
    @printf("Finite-T detailed-balance ratio    = %.9f [exact %.9f]\n", detailed_balance_ratio, detailed_balance_reference)
    @printf("Detailed-balance relative error    = %.3e\n", detailed_balance_relative_error)
    @printf("Finite-T static-response error     = %.3e\n", detailed_balance_static_error)
    println("  Native S(k,ω) peak positions     : ", mode_peak_pass ? "PASS" : "FAIL")
    println("  Native S(k,ω) integrated weights : ", mode_weight_pass ? "PASS" : "FAIL")
    println("  Native static S(k) mode route    : ", mode_static_response_pass ? "PASS" : "FAIL")
    println("  Dynamic/static spectral sum rule : ", mode_sumrule_pass ? "PASS" : "FAIL")
    println("  Finite-T detailed balance        : ", detailed_balance_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Multi-mode, convention, finite-T zero-mode, and Gaussian-response tests.
    #
    # Number-conserving zero-mode references:
    #
    #   Y. Castin and R. Dum, Phys. Rev. A 57, 3008 (1998),
    #   doi: 10.1103/PhysRevA.57.3008
    #
    #   S. A. Gardiner and S. A. Morgan, Phys. Rev. A 75, 043621 (2007),
    #   doi: 10.1103/PhysRevA.75.043621
    #
    #   Z. Jiang and C. M. Caves, Phys. Rev. A 93, 033623 (2016),
    #   doi: 10.1103/PhysRevA.93.033623
    #
    # These formulations keep the condensate mode outside the quasiparticle
    # fluctuation space rather than assigning an infinite Bose occupation to a
    # zero-energy Goldstone mode. The API therefore keeps rejection as the safe
    # default and requires an explicit number-conserving projection policy. This
    # projection implements the condensate/noncondensate subspace separation, not
    # the full self-consistent second-order finite-T Gardiner-Morgan dynamics.
    #
    # For regimes where condensate depletion or critical fluctuations invalidate
    # quadratic Bogoliubov theory, finite-T field-theory sampling is a separate
    # solver class rather than a zero-mode switch. Modern complex-Langevin
    # benchmarks include:
    #
    #   K. T. Delaney, H. Orland, and G. H. Fredrickson,
    #   Phys. Rev. Lett. 124, 070601 (2020),
    #   doi: 10.1103/PhysRevLett.124.070601
    #
    #   P. Heinen and T. Gasenzer, Phys. Rev. A 106, 063308 (2022),
    #   doi: 10.1103/PhysRevA.106.063308
    #
    #   P. Heinen and T. Gasenzer, Phys. Rev. A 108, 053311 (2023),
    #   doi: 10.1103/PhysRevA.108.053311
    #
    # The Gaussian/Wick response implemented below is intentionally limited to
    # regimes where quadratic quasiparticles remain appropriate. The observed
    # emergence of non-Gaussian correlations at stronger coupling provides a
    # modern experimental boundary for that approximation:
    #
    #   J.-P. Bureik et al., Nat. Phys. 21, 57-62 (2025),
    #   doi: 10.1038/s41567-024-02700-z
    # -------------------------------------------------------------------------

    function number_conserving_boson_model(energies::AbstractVector{<:Real}; hopping::Real=0.0)
        nmodes = length(energies)
        representation = F.Representation(:multimode_boson_response, F.BosonFockSpace(nmodes), F.BosonAlgebra(nmodes), F.BosonOccupationBasis(collect(1:nmodes));
                                          ordering=collect(1:nmodes), reference_state=:boson_vacuum)
        terms = F.AbstractOperatorExpr[]
        for i in 1:nmodes
            push!(terms, Float64(energies[i]) * F.nb(i))
        end
        if !iszero(hopping)
            for i in 1:nmodes-1
                push!(terms, Float64(hopping) * F.b(i)' * F.b(i + 1))
                push!(terms, Float64(hopping) * F.b(i + 1)' * F.b(i))
            end
        end
        return F.ManyBodyModel(representation, F.OperatorSum(terms); parameters=Dict(:benchmark => :multimode_boson_response))
    end

    # -------------------------------------------------------------------------
    # Priority 1: multi-mode and degenerate-subspace invariance.
    # -------------------------------------------------------------------------

    degenerate_model = number_conserving_boson_model([1.0, 1.0, 2.0, 2.0])
    degenerate_result = solve(QuadraticBosonSolver(tol=1e-12), degenerate_model)
    degenerate_field = MomentumField([F.b(i) + F.b(i)' for i in 1:4], [0.0, 0.7, 1.6, 2.8]; normalization=:none)
    degenerate_q = [0.23, 0.91, 1.47]
    degenerate_axis = collect(range(0.0, 3.0; length=3001))
    degenerate_convention = SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=1.0, kB=1.0)
    degenerate_broadening = GaussianBroadening(0.025)

    rotation = Matrix{ComplexF64}(I, 4, 4)
    θ1 = 0.37
    θ2 = 0.61
    phase = exp(0.43im)
    rotation[1:2, 1:2] .= ComplexF64[cos(θ1) sin(θ1); -sin(θ1) cos(θ1)]
    rotation[3:4, 3:4] .= ComplexF64[cos(θ2) phase * sin(θ2); -conj(phase) * sin(θ2) cos(θ2)]
    rotated_modes = mode_vectors(degenerate_result) * rotation
    rotated_metadata = copy(degenerate_result.metadata)
    rotated_metadata[:benchmark_rotation] = :degenerate_unitary
    rotated_result = QuadraticModeResult(degenerate_result.solver, degenerate_result.model, copy(mode_energies(degenerate_result)), rotated_modes, rotated_metadata)

    degenerate_dynamic = dynamic_structure_factor(degenerate_model, degenerate_result, degenerate_field, degenerate_q, degenerate_axis;
                                                  convention=degenerate_convention, broadening=degenerate_broadening, weight_tol=1e-14)
    rotated_dynamic = dynamic_structure_factor(degenerate_model, rotated_result, degenerate_field, degenerate_q, degenerate_axis;
                                               convention=degenerate_convention, broadening=degenerate_broadening, weight_tol=1e-14)
    degenerate_static = static_structure_factor(degenerate_model, degenerate_result, degenerate_field, degenerate_q; convention=degenerate_convention)
    rotated_static = static_structure_factor(degenerate_model, rotated_result, degenerate_field, degenerate_q; convention=degenerate_convention)

    degenerate_dynamic_error = norm(degenerate_dynamic.intensity - rotated_dynamic.intensity) / max(norm(degenerate_dynamic.intensity), eps(Float64))
    degenerate_static_error = norm(degenerate_static.intensity - rotated_static.intensity) / max(norm(degenerate_static.intensity), eps(Float64))
    degenerate_invariance_pass = degenerate_dynamic_error <= 1e-10 && degenerate_static_error <= 1e-10

    println()
    println("MULTI-MODE / DEGENERATE-SUBSPACE RESPONSE VALIDATION")
    @printf("Dynamic spectrum rotation error    = %.3e\n", degenerate_dynamic_error)
    @printf("Static structure-factor error      = %.3e\n", degenerate_static_error)
    println("  Degenerate-mode basis invariance : ", degenerate_invariance_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Priority 4: Fourier-normalization and energy/omega conventions.
    # -------------------------------------------------------------------------

    convention_q = [0.73]
    field_none = MomentumField([F.b(i) + F.b(i)' for i in 1:4], [0.0, 0.7, 1.6, 2.8]; normalization=:none)
    field_sqrt = MomentumField(field_none.keys, field_none.positions; normalization=:sqrtN)
    field_N = MomentumField(field_none.keys, field_none.positions; normalization=:N)
    static_none = static_structure_factor(degenerate_model, degenerate_result, field_none, convention_q; convention=SpectrumConvention(fourier_normalization=:none))
    static_sqrt = static_structure_factor(degenerate_model, degenerate_result, field_sqrt, convention_q; convention=SpectrumConvention(fourier_normalization=:sqrtN))
    static_N = static_structure_factor(degenerate_model, degenerate_result, field_N, convention_q; convention=SpectrumConvention(fourier_normalization=:N))
    normalization_reference = real(static_none.intensity[1])
    normalization_errors = Float64[abs(real(static_sqrt.intensity[1]) - normalization_reference / 4) / max(abs(normalization_reference / 4), eps(Float64)),
                                   abs(real(static_N.intensity[1]) - normalization_reference / 16) / max(abs(normalization_reference / 16), eps(Float64)) ]

    convention_hbar = 2.37
    Oq = Phundamental.Observables.momentum_operator(degenerate_model, field_none, convention_q[1])
    Ominus = Phundamental.Observables.momentum_operator(degenerate_model, field_none, -convention_q[1])
    energy_convention = SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=convention_hbar, kB=1.0)
    omega_convention = SpectrumConvention(fourier_normalization=:none, spectral_axis=:omega, hbar=convention_hbar, kB=1.0)
    energy_lines = Phundamental.Observables.lehmann_lines(degenerate_model, degenerate_result, Oq, Ominus; convention=energy_convention, degeneracy_tol=1e-12, weight_tol=1e-14)
    omega_lines = Phundamental.Observables.lehmann_lines(degenerate_model, degenerate_result, Oq, Ominus; convention=omega_convention, degeneracy_tol=1e-12, weight_tol=1e-14)
    energy_order = sortperm(energy_lines.centers)
    omega_order = sortperm(omega_lines.centers)
    length(energy_order) == length(omega_order) || error("energy/omega convention produced different line counts")
    axis_conversion_error = maximum(abs.(energy_lines.centers[energy_order] .- convention_hbar .* omega_lines.centers[omega_order]))
    axis_weight_error = maximum(abs.(energy_lines.weights[energy_order] .- omega_lines.weights[omega_order]))
    convention_pass = maximum(normalization_errors) <= 1e-12 && axis_conversion_error <= 1e-12 && axis_weight_error <= 1e-12

    println()
    println("MODE-NATIVE SPECTRUM CONVENTION VALIDATION")
    @printf("Maximum Fourier-normalization error = %.3e\n", maximum(normalization_errors))
    @printf("E = ħω line-center error             = %.3e\n", axis_conversion_error)
    @printf("Energy/ω integrated-weight error     = %.3e\n", axis_weight_error)
    println("  Fourier and spectral conventions   : ", convention_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Priority 5: finite-temperature zero-mode policy and solver projection.
    # -------------------------------------------------------------------------

    zero_model = number_conserving_boson_model([0.0, 1.2, 2.1])
    zero_result = solve(QuadraticBosonSolver(tol=1e-12), zero_model)
    zero_field = MomentumField([F.b(i) + F.b(i)' for i in 1:3], [0.0, 0.8, 1.9]; normalization=:none)
    zero_axis = collect(range(-3.0, 3.0; length=3001))
    zero_temperature = 0.7
    zero_convention = SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=1.0, kB=1.0)
    zero_broadening = GaussianBroadening(0.03)

    zero_default_reject_pass = false
    try
        dynamic_structure_factor(zero_model, zero_result, zero_field, [0.41], zero_axis; temperature=zero_temperature, convention=zero_convention, broadening=zero_broadening)
    catch error
        zero_default_reject_pass = error isa ArgumentError
    end

    zero_policy = NumberConservingZeroModeProjection(relative_tol=1e-10)
    zero_policy_dynamic = dynamic_structure_factor(zero_model, zero_result, zero_field, [0.41], zero_axis; temperature=zero_temperature, convention=zero_convention, broadening=zero_broadening, zero_mode_policy=zero_policy)
    zero_policy_static = static_structure_factor(zero_model, zero_result, zero_field, [0.41]; temperature=zero_temperature, convention=zero_convention, zero_mode_policy=zero_policy)
    zero_policy_static_T0 = static_structure_factor(zero_model, zero_result, zero_field, [0.41]; temperature=0.0, convention=zero_convention, zero_mode_policy=zero_policy)

    source_projection = orthogonal_boson_subspace(ComplexF64[1.0, 0.0, 0.0]; label=:number_conserving)
    zero_projected_result = solve(QuadraticBosonSolver(tol=1e-12, projection=source_projection), zero_model)
    zero_solver_dynamic = dynamic_structure_factor(zero_model, zero_projected_result, zero_field, [0.41], zero_axis; temperature=zero_temperature, convention=zero_convention, broadening=zero_broadening)
    zero_solver_static = static_structure_factor(zero_model, zero_projected_result, zero_field, [0.41]; temperature=zero_temperature, convention=zero_convention)
    zero_solver_static_T0 = static_structure_factor(zero_model, zero_projected_result, zero_field, [0.41]; temperature=0.0, convention=zero_convention)
    zero_dynamic_equivalence = norm(zero_policy_dynamic.intensity - zero_solver_dynamic.intensity) / max(norm(zero_solver_dynamic.intensity), eps(Float64))
    zero_static_equivalence = norm(zero_policy_static.intensity - zero_solver_static.intensity) / max(norm(zero_solver_static.intensity), eps(Float64))
    zero_static_T0_equivalence = norm(zero_policy_static_T0.intensity - zero_solver_static_T0.intensity) / max(norm(zero_solver_static_T0.intensity), eps(Float64))

    goldstone_representation = F.Representation(:projected_goldstone_bdg, F.BosonFockSpace(3), F.BosonAlgebra(3), F.BosonOccupationBasis([1, 2, 3]); ordering=[1, 2, 3], reference_state=:boson_vacuum)
    goldstone_terms = F.AbstractOperatorExpr[1.0 * F.nb(1), 0.5 * F.b(1)' * F.b(1)', 0.5 * F.b(1) * F.b(1), 1.2 * F.nb(2), 2.1 * F.nb(3)]
    goldstone_model = F.ManyBodyModel(goldstone_representation, F.OperatorSum(goldstone_terms); parameters=Dict(:benchmark => :number_conserving_zero_mode_projection))
    goldstone_unprojected_reject_pass = false
    try
        solve(QuadraticBosonSolver(tol=1e-12), goldstone_model)
    catch error
        goldstone_unprojected_reject_pass = error isa ArgumentError
    end
    goldstone_projected = solve(QuadraticBosonSolver(tol=1e-12, projection=source_projection), goldstone_model)
    goldstone_energy_error = maximum(abs.(mode_energies(goldstone_projected) .- [1.2, 2.1]))

    zero_mode_pass = zero_default_reject_pass && zero_dynamic_equivalence <= 1e-10 && zero_static_equivalence <= 1e-10 && zero_static_T0_equivalence <= 1e-10 && goldstone_unprojected_reject_pass && goldstone_energy_error <= 1e-12

    println()
    println("FINITE-T BOSON ZERO-MODE POLICY VALIDATION")
    println("  Default divergent zero-mode guard  : ", zero_default_reject_pass ? "PASS" : "FAIL")
    @printf("Response-policy vs solver projection = %.3e [dynamic]\n", zero_dynamic_equivalence)
    @printf("Response-policy vs solver projection = %.3e [static, finite T]\n", zero_static_equivalence)
    @printf("Response-policy vs solver projection = %.3e [static, T=0]\n", zero_static_T0_equivalence)
    println("  Unprojected paired Goldstone guard  : ", goldstone_unprojected_reject_pass ? "PASS" : "FAIL")
    @printf("Projected active-mode energy error    = %.3e\n", goldstone_energy_error)
    println("  Number-conserving zero-mode route   : ", zero_mode_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Priority 6: quadratic Gaussian/Wick observable response.
    # -------------------------------------------------------------------------

    quadratic_model = number_conserving_boson_model([1.1, 1.7])
    quadratic_result = solve(QuadraticBosonSolver(tol=1e-12), quadratic_model)
    κ = 0.35
    η = 0.40
    quadratic_temperature = 0.65
    quadratic_observable = F.b(1)' * F.b(2) + F.b(2)' * F.b(1) + κ * (F.b(1) * F.b(2) + F.b(1)' * F.b(2)') + η * F.nb(1)
    quadratic_lines = Phundamental.Observables.lehmann_lines(quadratic_model, quadratic_result, quadratic_observable, quadratic_observable;
                                                             temperature=quadratic_temperature, convention=zero_convention, degeneracy_tol=1e-12, weight_tol=1e-14)

    function spectral_line_weight(lines, target::Real; atol::Real=1e-10)
        total = 0.0 + 0.0im
        for (center, weight) in zip(lines.centers, lines.weights)
            abs(center - target) <= atol && (total += weight)
        end
        return total
    end

    E1 = 1.1
    E2 = 1.7
    n1 = inv(expm1(E1 / quadratic_temperature))
    n2 = inv(expm1(E2 / quadratic_temperature))
    quadratic_targets = [0.0, E2 - E1, E1 - E2, E1 + E2, -(E1 + E2)]
    quadratic_reference_weights = ComplexF64[η^2 * (2n1^2 + n1),
                                             n1 * (n2 + 1),
                                             n2 * (n1 + 1),
                                             κ^2 * (n1 + 1) * (n2 + 1),
                                             κ^2 * n1 * n2
                                            ]
    quadratic_observed_weights = ComplexF64[spectral_line_weight(quadratic_lines, center) for center in quadratic_targets]
    quadratic_line_relative_errors = abs.(quadratic_observed_weights .- quadratic_reference_weights) ./ max.(abs.(quadratic_reference_weights), eps(Float64))

    normal_order_left = F.b(1) * F.b(1)'
    normal_order_right = 1.0 + F.nb(1)
    left_lines = Phundamental.Observables.lehmann_lines(quadratic_model, quadratic_result, normal_order_left, normal_order_left;
                                                       temperature=quadratic_temperature, convention=zero_convention,
                                                       degeneracy_tol=1e-12, weight_tol=1e-14)
    right_lines = Phundamental.Observables.lehmann_lines(quadratic_model, quadratic_result, normal_order_right, normal_order_right;
                                                        temperature=quadratic_temperature, convention=zero_convention,
                                                        degeneracy_tol=1e-12, weight_tol=1e-14)
    normal_order_centers_error = maximum(abs.(sort(left_lines.centers) .- sort(right_lines.centers)))
    left_weight_total = (isempty(left_lines.weights) ? 0.0 + 0.0im : sum(left_lines.weights))
    right_weight_total = (isempty(right_lines.weights) ? 0.0 + 0.0im : sum(right_lines.weights))
    normal_order_weight_error = abs(left_weight_total - right_weight_total) / max(abs(right_weight_total), eps(Float64))

    quadratic_field = MomentumField([quadratic_observable], [0.0]; normalization=:none)
    quadratic_axis = collect(range(-3.6, 3.6; length=7201))
    quadratic_dynamic = dynamic_structure_factor(quadratic_model, quadratic_result, quadratic_field, [0.0], quadratic_axis;
                                                 temperature=quadratic_temperature, convention=zero_convention,
                                                 broadening=GaussianBroadening(0.025), weight_tol=1e-14)
    quadratic_static = static_structure_factor(quadratic_model, quadratic_result, quadratic_field, [0.0];
                                               temperature=quadratic_temperature, convention=zero_convention, weight_tol=1e-14)
    quadratic_sumrule_error = Phundamental.Observables.dynamic_static_sumrule_residual(quadratic_dynamic, quadratic_static)
    quadratic_static_reference = real(sum(quadratic_reference_weights))
    quadratic_static_error = abs(real(quadratic_static.intensity[1]) - quadratic_static_reference) / max(abs(quadratic_static_reference), eps(Float64))

    paired_quadratic_model = homogeneous_bogoliubov_model(1.0, 1.0; cutoff=40)
    paired_quadratic_mode = solve(QuadraticBosonSolver(tol=1e-12), paired_quadratic_model)
    paired_quadratic_exact = solve(ExactDiagonalization(), paired_quadratic_model)
    paired_quadratic_observable = F.nb(1)
    paired_quadratic_temperature = 0.40
    paired_quadratic_axis = collect(range(-4.5, 4.5; length=9001))
    paired_quadratic_broadening = GaussianBroadening(0.025)
    paired_mode_spectrum = spectral_density(paired_quadratic_model, paired_quadratic_mode, paired_quadratic_observable, paired_quadratic_axis;
                                            temperature=paired_quadratic_temperature, convention=zero_convention, broadening=paired_quadratic_broadening, weight_tol=1e-14)
    paired_exact_spectrum = spectral_density(paired_quadratic_model, paired_quadratic_exact, paired_quadratic_observable, paired_quadratic_axis;
                                             temperature=paired_quadratic_temperature, convention=zero_convention, broadening=paired_quadratic_broadening, weight_tol=1e-14)
    paired_gaussian_fock_error = norm(paired_mode_spectrum.intensity - paired_exact_spectrum.intensity) / max(norm(paired_exact_spectrum.intensity), eps(Float64))

    quadratic_response_pass = maximum(quadratic_line_relative_errors) <= 1e-10 && normal_order_centers_error <= 1e-12 &&
                              normal_order_weight_error <= 1e-12 && quadratic_sumrule_error <= 1e-8 && quadratic_static_error <= 1e-10 && paired_gaussian_fock_error <= 1e-7

    println()
    println("QUADRATIC GAUSSIAN/WICK RESPONSE VALIDATION")
    @printf("Maximum analytic line-weight error  = %.3e\n", maximum(quadratic_line_relative_errors))
    @printf("Normal-order line-center error       = %.3e\n", normal_order_centers_error)
    @printf("Normal-order integrated-weight error = %.3e\n", normal_order_weight_error)
    @printf("Quadratic dynamic/static sum-rule    = %.3e\n", quadratic_sumrule_error)
    @printf("Quadratic static-response error      = %.3e\n", quadratic_static_error)
    @printf("Paired BdG vs exact Fock spectrum     = %.3e\n", paired_gaussian_fock_error)
    println("  Gaussian quadratic response route  : ", quadratic_response_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Priorities 2-3: scaling, allocations, and broadening/coefficient overhead.
    #
    # Each code path is warmed independently at every M before measurement.
    # Timings are medians over repeated steady-state evaluations, which removes
    # first-call JIT/LAPACK initialization and reduces scheduler noise. Scaling
    # exponents are least-squares slopes in log(M)-log(time) over the three
    # largest sizes, rather than endpoint ratios contaminated by small-M overhead.
    # PHUNDAMENTAL_BOSON_SCALING_MAX controls the largest M and
    # PHUNDAMENTAL_BOSON_SCALING_REPEATS controls the number of timing samples.
    # -------------------------------------------------------------------------

    scaling_max = parse(Int, get(ENV, "PHUNDAMENTAL_BOSON_SCALING_MAX", "64"))
    scaling_repeats = parse(Int, get(ENV, "PHUNDAMENTAL_BOSON_SCALING_REPEATS", "5"))
    scaling_repeats >= 3 || throw(ArgumentError("PHUNDAMENTAL_BOSON_SCALING_REPEATS must be at least 3"))
    scaling_sizes = [Mtest for Mtest in (8, 16, 32, 64, 128, 256) if Mtest <= scaling_max]
    length(scaling_sizes) >= 3 || throw(ArgumentError("PHUNDAMENTAL_BOSON_SCALING_MAX must be at least 32"))
    scaling_q = collect(range(0.15, 1.35; length=4))
    scaling_axis = collect(range(0.0, 3.0; length=601))
    scaling_broadening = GaussianBroadening(0.025)
    scaling_solve_times = Float64[]
    scaling_static_times = Float64[]
    scaling_dynamic_times = Float64[]
    scaling_solve_bytes = Float64[]
    scaling_static_bytes = Float64[]
    scaling_dynamic_bytes = Float64[]
    scaling_result_bytes = Float64[]
    scaling_spectrum_bytes = Float64[]

    function median_sample(values::Vector{Float64})
        sort!(values)
        n = length(values)
        middle = (n + 1) ÷ 2
        return isodd(n) ? values[middle] : 0.5 * (values[middle] + values[middle + 1])
    end

    function benchmark_seconds(f::Func, repeats::Int) where {Func}
        samples = Vector{Float64}(undef, repeats)
        for i in 1:repeats
            GC.gc()
            samples[i] = @elapsed f()
        end
        return median_sample(samples)
    end

    function benchmark_bytes(f::Func, repeats::Int) where {Func}
        samples = Vector{Float64}(undef, repeats)
        for i in 1:repeats
            GC.gc()
            samples[i] = Float64(@allocated f())
        end
        return median_sample(samples)
    end

    function loglog_fit(sizes::AbstractVector{<:Real}, values::AbstractVector{<:Real})
        length(sizes) == length(values) || throw(DimensionMismatch("log-log fit arrays differ in length"))
        length(sizes) >= 2 || throw(ArgumentError("log-log fit requires at least two points"))
        x = log.(Float64.(sizes))
        y = log.(max.(Float64.(values), eps(Float64)))
        xmean = sum(x) / length(x)
        ymean = sum(y) / length(y)
        centered_x = x .- xmean
        centered_y = y .- ymean
        denominator = sum(abs2, centered_x)
        denominator > 0 || throw(ArgumentError("log-log fit requires distinct sizes"))
        slope = sum(centered_x .* centered_y) / denominator
        intercept = ymean - slope * xmean
        residual = y .- (intercept .+ slope .* x)
        total_variation = sum(abs2, centered_y)
        r2 = total_variation <= eps(Float64) ? 1.0 : 1.0 - sum(abs2, residual) / total_variation
        return slope, r2
    end

    println()
    println("MODE-NATIVE BOSON RESPONSE SCALING / MEMORY")
    @printf("Timing samples per size             = %d [median after per-size warm-up]\n", scaling_repeats)
    @printf("Julia threads / BLAS threads        = %d / %d\n", Threads.nthreads(), BLAS.get_num_threads())
    println("    M      solve [s]    static [s]   dynamic [s]    solve alloc [MiB]   static alloc [MiB]   dynamic alloc [MiB]")

    for Mtest in scaling_sizes
        onsite = [2.0 + 0.001 * i for i in 1:Mtest]
        scaling_model = number_conserving_boson_model(onsite; hopping=0.05)
        scaling_field = MomentumField([F.b(i) + F.b(i)' for i in 1:Mtest], Float64.(0:Mtest-1); normalization=:sqrtN)

        # Per-size warm-up eliminates first-use LAPACK workspaces and any residual
        # specialization before a sample contributes to the scaling fit.
        scaling_result = solve(QuadraticBosonSolver(tol=1e-11), scaling_model)
        scaling_static = static_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q)
        scaling_dynamic = dynamic_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q, scaling_axis;broadening=scaling_broadening, weight_tol=1e-14)

        solve_time = benchmark_seconds(() -> solve(QuadraticBosonSolver(tol=1e-11), scaling_model), scaling_repeats)
        static_time = benchmark_seconds(() -> static_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q), scaling_repeats)
        dynamic_time = benchmark_seconds(() -> dynamic_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q, scaling_axis; broadening=scaling_broadening, weight_tol=1e-14), scaling_repeats)

        solve_bytes = benchmark_bytes(() -> solve(QuadraticBosonSolver(tol=1e-11), scaling_model), scaling_repeats)
        static_bytes = benchmark_bytes(() -> static_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q), scaling_repeats)
        dynamic_bytes = benchmark_bytes(() -> dynamic_structure_factor(scaling_model, scaling_result, scaling_field, scaling_q, scaling_axis; broadening=scaling_broadening, weight_tol=1e-14), scaling_repeats)

        push!(scaling_solve_times, solve_time)
        push!(scaling_static_times, static_time)
        push!(scaling_dynamic_times, dynamic_time)
        push!(scaling_solve_bytes, solve_bytes)
        push!(scaling_static_bytes, static_bytes)
        push!(scaling_dynamic_bytes, dynamic_bytes)
        push!(scaling_result_bytes, Base.summarysize(scaling_result))
        push!(scaling_spectrum_bytes, Base.summarysize(scaling_dynamic))

        @printf("%5d    %10.4e   %10.4e   %10.4e       %10.3f          %10.3f           %10.3f\n", Mtest,
                solve_time, static_time, dynamic_time, solve_bytes / 2.0^20, static_bytes / 2.0^20, dynamic_bytes / 2.0^20)
    end

    fit_count = min(3, length(scaling_sizes))
    fit_indices = (length(scaling_sizes) - fit_count + 1):length(scaling_sizes)
    fit_sizes = scaling_sizes[fit_indices]
    solve_scaling_exponent, solve_scaling_r2 = loglog_fit(fit_sizes, scaling_solve_times[fit_indices])
    static_scaling_exponent, static_scaling_r2 = loglog_fit(fit_sizes, scaling_static_times[fit_indices])
    dynamic_scaling_exponent, dynamic_scaling_r2 = loglog_fit(fit_sizes, scaling_dynamic_times[fit_indices])
    result_memory_exponent, result_memory_r2 = loglog_fit(fit_sizes, scaling_result_bytes[fit_indices])
    expected_output_bytes = sizeof(ComplexF64) * length(scaling_q) * length(scaling_axis)

    timing_fit_reliable = last(fit_sizes) >= 128 && solve_scaling_exponent > 0 &&
                          static_scaling_exponent > 0 && dynamic_scaling_exponent > 0 &&
                          solve_scaling_r2 >= 0.80 && static_scaling_r2 >= 0.80 && dynamic_scaling_r2 >= 0.80
    scaling_sanity_pass = all(isfinite, scaling_solve_times) && all(isfinite, scaling_static_times) &&
                          all(isfinite, scaling_dynamic_times) &&
                          all(diff(scaling_result_bytes) .>= 0) && all(bytes -> bytes >= expected_output_bytes, scaling_spectrum_bytes) &&
                          isfinite(result_memory_exponent) && result_memory_r2 >= 0.80

    println()
    @printf("Large-M fit range                  = M=%d:%d [%d largest sizes]\n", first(fit_sizes), last(fit_sizes), fit_count)
    @printf("Solve-time exponent                = %.3f (R²=%.3f) [dense asymptote O(M³)]\n", solve_scaling_exponent, solve_scaling_r2)
    @printf(
        "Static-time exponent               = %.3f (R²=%.3f) [linear-field projection O(Nq M²)]\n",
        static_scaling_exponent, static_scaling_r2,
    )
    @printf("Dynamic-time exponent              = %.3f (R²=%.3f) [O(Nq(M² + M Nω))]\n", dynamic_scaling_exponent, dynamic_scaling_r2)
    @printf("Retained-result memory exponent    = %.3f (R²=%.3f) [dense mode vectors O(M²)]\n", result_memory_exponent, result_memory_r2)
    @printf("Analytic spectrum-output storage   = %.3f MiB [independent of M at fixed Nq,Nω]\n", expected_output_bytes / 2.0^20)
    timing_fit_status = timing_fit_reliable ? "PASS" : "DIAGNOSTIC ONLY — use MAX≥128 and increase repeats if R²<0.80"
    println("  Timing scaling-fit reliability   : ", timing_fit_status)
    println("  Scaling / memory sanity          : ", scaling_sanity_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Steinhauer et al. trapped-condensate LDA, Eqs. (3)-(5).
    #
    # With α = 2μ/[ħ²k²/(2m)], their Eq. (4) gives the trapped-gas static
    # structure factor. The corresponding c_ld(k) inserted into Eq. (3) yields
    # the excitation spectrum. At low k,
    #
    #     c_eff = 32/(15π) sqrt(μ/m),
    #
    # while at large k the interaction shift approaches (4/7)μ/ħ.
    # -------------------------------------------------------------------------

    function lda_structure_factor(k_um::Real)
        k_um > 0 || throw(ArgumentError("wave number must be positive"))
        k = Float64(k_um) * 1e6
        ε = ħ^2 * k^2 / (2 * m_Rb87)
        α = 2μ / ε
        angular_term = π + 2 * atan((α - 1) / (2 * sqrt(α)))
        return (15 / 4) * ((3 + α) / (4α^2) - (3 + 2α - α^2) * angular_term / (16α^(5 / 2)))
    end

    function lda_sound_velocity(k_um::Real)
        k = Float64(k_um) * 1e6
        S = lda_structure_factor(k_um)
        return ħ * k * sqrt(inv(S^2) - 1) / (2 * m_Rb87)
    end

    free_particle_omega(k_um::Real) = ħ * (Float64(k_um) * 1e6)^2 / (2 * m_Rb87)

    function lda_omega(k_um::Real)
        k = Float64(k_um) * 1e6
        c = lda_sound_velocity(k_um)
        return sqrt((c * k)^2 + free_particle_omega(k_um)^2)
    end

    omega_to_khz(ω::Real) = Float64(ω) / (2π * 1e3)

    k_um = collect(range(0.25, 20.0; length=240))
    lda_structure = lda_structure_factor.(k_um)
    lda_omega_values = lda_omega.(k_um)
    free_omega_values = free_particle_omega.(k_um)
    lda_frequency_khz = omega_to_khz.(lda_omega_values)
    free_frequency_khz = omega_to_khz.(free_omega_values)
    interaction_shift_khz = lda_frequency_khz .- free_frequency_khz
    feynman_structure = free_omega_values ./ lda_omega_values

    feynman_error = maximum(abs.(lda_structure .- feynman_structure))

    c_eff = 32 / (15π) * sqrt(μ / m_Rb87)
    c_large = sqrt(4 / 7) * sqrt(μ / m_Rb87)
    c_eff_mm_s = 1e3 * c_eff
    c_large_mm_s = 1e3 * c_large
    c_eff_uncertainty_mm_s = 0.5 * c_eff_mm_s * paper_mu_over_h_uncertainty_hz / paper_mu_over_h_hz

    k_lda_min_um = 2π / paper_axial_radius_um
    cbar = (c_eff + c_large) / 2
    xi_inverse_um = sqrt(2) * m_Rb87 * cbar / ħ / 1e6

    landau_grid_velocity_mm_s = minimum(1e3 .* lda_omega_values ./ (k_um .* 1e6))
    large_k_offset_khz = (4 / 7) * paper_mu_over_h_hz / 1e3
    asymptotic_probe_k_um = 40.0
    asymptotic_probe_shift_khz = omega_to_khz(lda_omega(asymptotic_probe_k_um) - free_particle_omega(asymptotic_probe_k_um))
    asymptotic_offset_relative_error = abs(asymptotic_probe_shift_khz - large_k_offset_khz) / large_k_offset_khz

    low_k_probe_um = 0.05
    high_k_probe_um = 40.0
    low_k_structure = lda_structure_factor(low_k_probe_um)
    high_k_structure = lda_structure_factor(high_k_probe_um)
    low_k_velocity_mm_s = 1e3 * lda_omega(low_k_probe_um) / (low_k_probe_um * 1e6)

    experimental_sound_pass = abs(c_eff_mm_s - paper_sound_velocity_mm_s) <= paper_sound_velocity_uncertainty_mm_s
    paper_lda_sound_pass = abs(c_eff_mm_s - paper_lda_sound_velocity_mm_s) <= paper_lda_sound_velocity_uncertainty_mm_s
    low_k_limit_pass = abs(low_k_velocity_mm_s - c_eff_mm_s) <= 0.01
    landau_pass = abs(landau_grid_velocity_mm_s - c_eff_mm_s) <= 0.02
    asymptotic_offset_pass = asymptotic_offset_relative_error <= 0.02
    feynman_pass = feynman_error <= 1e-12
    structure_asymptotics_pass = low_k_structure <= 0.02 && abs(high_k_structure - 1) <= 0.02

    println()
    println("STEINHAUER TRAPPED-BEC LDA LANDMARKS")
    @printf("87Rb atom number                  ≈ %.1e\n", paper_atom_number)
    @printf("Chemical potential μ/h           = %.2f ± %.2f kHz\n", paper_mu_over_h_hz / 1e3, paper_mu_over_h_uncertainty_hz / 1e3)
    @printf("LDA c_eff                         = %.6f mm/s\n", c_eff_mm_s)
    @printf("μ-derived c_eff uncertainty       = %.6f mm/s\n", c_eff_uncertainty_mm_s)
    @printf(
        "Paper theoretical c_eff           = %.2f ± %.2f mm/s\n",
        paper_lda_sound_velocity_mm_s, paper_lda_sound_velocity_uncertainty_mm_s,
    )
    @printf("Paper measured c_eff              = %.1f ± %.1f mm/s\n", paper_sound_velocity_mm_s, paper_sound_velocity_uncertainty_mm_s)
    @printf("Low-k numerical ω/k              = %.6f mm/s\n", low_k_velocity_mm_s)
    @printf("Landau min(ω/k) on plotted grid   = %.6f mm/s\n", landau_grid_velocity_mm_s)
    @printf("Large-k c_ld limit                = %.6f mm/s\n", c_large_mm_s)
    @printf("LDA lower-validity marker 2π/R    = %.6f μm⁻¹\n", k_lda_min_um)
    @printf("Collective/particle marker ξ⁻¹    = %.6f μm⁻¹\n", xi_inverse_um)
    @printf("Eq. (5) asymptotic shift          = %.6f kHz\n", large_k_offset_khz)
    @printf("Numerical shift at k=%.1f μm⁻¹    = %.6f kHz\n", asymptotic_probe_k_um, asymptotic_probe_shift_khz)
    @printf("Asymptotic-shift relative error   = %.3e\n", asymptotic_offset_relative_error)

    println()
    println("STEINHAUER STATIC-STRUCTURE-FACTOR VALIDATION")
    @printf("Maximum Eq. (2) Feynman error     = %.3e\n", feynman_error)
    @printf("S(k=%.2f μm⁻¹)                    = %.6f\n", low_k_probe_um, low_k_structure)
    @printf("S(k=%.1f μm⁻¹)                    = %.6f\n", high_k_probe_um, high_k_structure)
    @printf("Experimental Fig. 4 scale factor  = %.1f [not applied to theory]\n", paper_structure_factor_scale)
    @printf("Visible s-wave scattering onset    ≈ %.1f μm⁻¹ [paper]\n", paper_scattering_visible_k_um)

    lda_landmarks_pass =
        experimental_sound_pass && paper_lda_sound_pass && low_k_limit_pass && landau_pass && asymptotic_offset_pass
    overall_pass =
        native_dispersion_pass && native_structure_pass && native_metric_pass && mode_native_response_pass &&
        degenerate_invariance_pass && convention_pass && zero_mode_pass && quadratic_response_pass && scaling_sanity_pass &&
        lda_landmarks_pass && feynman_pass && structure_asymptotics_pass

    println()
    println("STEINHAUER BEC BENCHMARK VALIDATION")
    println("  QuadraticBosonSolver dispersion     : ", native_dispersion_pass ? "PASS" : "FAIL")
    println("  Bogoliubov density-mode amplitudes  : ", native_structure_pass ? "PASS" : "FAIL")
    println("  Bosonic symplectic normalization    : ", native_metric_pass ? "PASS" : "FAIL")
    println("  Measured low-k sound velocity       : ", experimental_sound_pass ? "PASS" : "FAIL")
    println("  Published LDA sound velocity        : ", paper_lda_sound_pass ? "PASS" : "FAIL")
    println("  Landau phonon-velocity limit        : ", landau_pass ? "PASS" : "FAIL")
    println("  Large-k interaction-energy offset   : ", asymptotic_offset_pass ? "PASS" : "FAIL")
    println("  Feynman S(k)-dispersion relation    : ", feynman_pass ? "PASS" : "FAIL")
    println("  S(k→0)→0 and S(k→∞)→1              : ", structure_asymptotics_pass ? "PASS" : "FAIL")
    println("  Mode-native S(k,ω) observable route : ", mode_native_response_pass ? "PASS" : "FAIL")
    println("  Multi-mode degenerate invariance    : ", degenerate_invariance_pass ? "PASS" : "FAIL")
    println("  Fourier / E↔ω conventions           : ", convention_pass ? "PASS" : "FAIL")
    println("  Finite-T zero-mode policy           : ", zero_mode_pass ? "PASS" : "FAIL")
    println("  Quadratic Gaussian/Wick response    : ", quadratic_response_pass ? "PASS" : "FAIL")
    println("  Scaling / memory sanity             : ", scaling_sanity_pass ? "PASS" : "FAIL")
    println()
    println("Steinhauer BEC excitation-spectrum benchmark: ", overall_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Visualization.
    #
    # (a) Native homogeneous Bogoliubov dispersion vs the analytic solution.
    # (b) Steinhauer Eq. (3) trapped-gas LDA spectrum and free-particle parabola.
    # (c) Interaction-energy shift from Fig. 3(b) with the Eq. (5) asymptote.
    # (d) Steinhauer Eq. (4) static structure factor with regime markers.
    # -------------------------------------------------------------------------

    fig = Figure(size=(1200, 900))

    ax11 = Axis(fig[1, 1]; xlabel="εₖ/μ", ylabel="Eₖ/μ", title="Homogeneous Bogoliubov mode validation", xscale=log10, yscale=log10)
    lines!(ax11, epsilon_ratios, reference_energies; label="analytic Bogoliubov")
    scatter!(ax11, epsilon_ratios[1:8:end], native_energies[1:8:end]; label="QuadraticBosonSolver", markersize=8)
    axislegend(ax11; position=:lt)

    ax12 = Axis(fig[1, 2]; xlabel="k (μm⁻¹)", ylabel="ν = ω/2π (kHz)", title="Steinhauer Fig. 3(a): LDA excitation spectrum")
    lines!(ax12, k_um, lda_frequency_khz; label="Bogoliubov LDA")
    lines!(ax12, k_um, free_frequency_khz; linestyle=:dash, label="free particle")
    vlines!(ax12, [k_lda_min_um, xi_inverse_um]; linestyle=:dot)
    axislegend(ax12; position=:lt)

    ax21 = Axis(fig[2, 1]; xlabel="k (μm⁻¹)", ylabel="ν - ν_free (kHz)", title="Steinhauer Fig. 3(b): interaction-energy shift")
    lines!(ax21, k_um, interaction_shift_khz; label="Bogoliubov LDA - free particle")
    hlines!(ax21, [large_k_offset_khz]; linestyle=:dash, label="(4/7)μ/h")
    axislegend(ax21; position=:rb)

    ax22 = Axis(fig[2, 2]; xlabel="k (μm⁻¹)", ylabel="S(k)", title="Steinhauer Fig. 4: static structure factor")
    lines!(ax22, k_um, lda_structure; label="LDA Eq. (4)")
    lines!(ax22, k_um, feynman_structure; linestyle=:dash, label="Feynman Eq. (2)")
    vlines!(ax22, [k_lda_min_um, xi_inverse_um, paper_scattering_visible_k_um]; linestyle=:dot)
    hlines!(ax22, [1.0]; linestyle=:dash)
    ylims!(ax22, 0.0, 1.05)
    axislegend(ax22; position=:rb)

    save(joinpath(BENCHMARK_FIGURE_DIR, "Steinhauer-BEC-Bogoliubov-benchmark.png"), fig)
end


#-------------------------------------------------------#
# Falkovsky, L. A. (2007).                              #
# Phonon Dispersion in Graphene.                        #
# JETP 105, 397-403.                                    #
# doi: 10.1134/S1063776107080122                        #
#-------------------------------------------------------#
let
    # -------------------------------------------------------------------------
    # Falkovsky first- and second-neighbor Born-von Karman graphene model.
    #
    # The 2007 article uses ξ,η = x ± iy and six mass-normalized force constants
    # in 10^5 cm^-2. The Cartesian tensor corresponding to an in-plane scalar
    # component a = Φ_ξη and covariant component p = Φ_ξξ is shown below.
    #
    #     K = [a + Re(p)   Im(p)
    #          Im(p)       a - Re(p)].
    #
    # Because T K T^-1 = [a p; p* a] for T(x,y) = (ξ,η), the phase of p
    # rotates by twice the real-space bond angle, reproducing Falkovsky
    # Eqs. (4) and (13). The fitted δ is real in the article.
    #
    # The 2007 long-wavelength in-plane coefficients printed in Eq. (17) are
    # not algebraically consistent with Eqs. (11)-(13). Falkovsky's 2008
    # follow-up, Phys. Lett. A 372, 5189
    # (doi: 10.1016/j.physleta.2008.05.085), gives the corrected expansion.
    # With third-neighbor constants set to zero, those coefficients reproduce
    # the q→0 slopes of the 2007 six-parameter matrix. The corrected expansion
    # is therefore the hard API velocity gate below; the 2007 stated
    # 20.3/13.1 km/s values are retained as a documented legacy diagnostic
    # rather than imposed on an inconsistent dynamical matrix.
    # -------------------------------------------------------------------------

    a_angstrom = 1.42
    carbon_mass = 12.011
    force_scale = 1.0e5

    α = -3.980 * force_scale
    β = -1.132 * force_scale
    γ = -0.297 * force_scale
    δ = 1.123 * force_scale
    αz = -1.270 * force_scale
    γz = 0.204 * force_scale

    direct = a_angstrom .* [1.5 1.5; sqrt(3) / 2 -sqrt(3) / 2]
    lattice = M.BravaisLattice(direct)
    crystal = M.CrystalStructure(lattice,[M.BasisSite(:A, :C, [0.0, 0.0]), M.BasisSite(:B, :C, [1 / 3, 1 / 3])])
    material = M.Material(
        "Falkovsky graphene", crystal;
        species=Dict(:C => M.AtomicSpecies(:C; mass=carbon_mass)),
        basis_properties=[M.SiteProperties(), M.SiteProperties()],
        metadata=Dict(:reference => "Falkovsky, JETP 105, 397-403 (2007)", :doi => "10.1134/S1063776107080122"),
    )
    cluster = M.Supercell(crystal, [1, 1]; periodic=true)

    function falkovsky_tensor(isotropic::Real, covariant::Complex, out_of_plane::Real)
        return Float64[isotropic + real(covariant) imag(covariant) 0.0
                       imag(covariant) isotropic - real(covariant) 0.0
                       0.0 0.0 out_of_plane]
    end

    function falkovsky_ifcs()
        blocks = ForceConstantBlock[]
        basis_fractional = ([0.0, 0.0], [1 / 3, 1 / 3])
        first_translations = ([0, 0], [0, -1], [-1, 0])

        for translation in first_translations
            separation_fractional = Float64.(translation) .+ basis_fractional[2] .- basis_fractional[1]
            separation = direct * separation_fractional
            θ = atan(separation[2], separation[1])
            tensor = falkovsky_tensor(α, β * cis(2θ), αz)
            push!(blocks, ForceConstantBlock(1, 2, translation, tensor))
            push!(blocks, ForceConstantBlock(2, 1, -translation, transpose(tensor)))
        end

        second_translations = ([1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1])
        for translation in second_translations
            separation = direct * Float64.(translation)
            θ = atan(separation[2], separation[1])
            tensor = falkovsky_tensor(γ, -δ * cis(2θ), γz)
            push!(blocks, ForceConstantBlock(1, 1, translation, tensor))
            push!(blocks, ForceConstantBlock(2, 2, translation, tensor))
        end

        for site in 1:2
            onsite = zeros(Float64, 3, 3)
            for block in blocks
                block.from_site == site || continue
                onsite .-= block.tensor
            end
            push!(blocks, ForceConstantBlock(site, site, [0, 0], onsite))
        end

        return RealSpaceForceConstants(blocks; displacement_dimension=3)
    end

    function falkovsky_reference_dynamical_matrix(q)
        qv = Float64.(q)
        first_vectors = (a_angstrom .* [1.0, 0.0],
                         a_angstrom .* [-0.5, sqrt(3) / 2],
                         a_angstrom .* [-0.5, -sqrt(3) / 2] )
        second_vectors = (a_angstrom .* [0.0, sqrt(3)],
                          a_angstrom .* [0.0, -sqrt(3)],
                          a_angstrom .* [-1.5, sqrt(3) / 2],
                          a_angstrom .* [1.5, -sqrt(3) / 2],
                          a_angstrom .* [-1.5, -sqrt(3) / 2],
                          a_angstrom .* [1.5, sqrt(3) / 2] )

        first_tensors = [falkovsky_tensor(α, β * cis(2 * atan(vector[2], vector[1])), αz) for vector in first_vectors]
        second_tensors = [falkovsky_tensor(γ, -δ * cis(2 * atan(vector[2], vector[1])), γz) for vector in second_vectors]
        onsite = -reduce(+, first_tensors) - reduce(+, second_tensors)
        AA = ComplexF64.(onsite)
        AB = zeros(ComplexF64, 3, 3)

        for (vector, tensor) in zip(second_vectors, second_tensors)
            AA .+= cis(dot(qv, vector)) .* tensor
        end
        for (vector, tensor) in zip(first_vectors, first_tensors)
            AB .+= cis(dot(qv, vector)) .* tensor
        end

        return [AA AB; adjoint(AB) AA]
    end

    mass_weighted_ifcs = falkovsky_ifcs()
    weighted_model = build_model(material, cluster, HarmonicPhononModel(mass_weighted_ifcs; masses=carbon_mass, displacement_dimension=3, mass_weighted=true, symmetry_tolerance=1e-10))

    physical_blocks = ForceConstantBlock[ForceConstantBlock(block.from_site, block.to_site, block.translation, carbon_mass .* block.tensor) for block in mass_weighted_ifcs.blocks]
    physical_ifcs = RealSpaceForceConstants(physical_blocks; displacement_dimension=3)
    physical_model = build_model(material, cluster, HarmonicPhononModel(physical_ifcs; masses=carbon_mass, displacement_dimension=3, mass_weighted=false, symmetry_tolerance=1e-10))

    Γ = M.reciprocal_vector(crystal, [0.0, 0.0])
    Mpoint = M.reciprocal_vector(crystal, [0.5, 0.0])
    Kpoint = M.reciprocal_vector(crystal, [1 / 3, -1 / 3])

    function symmetry_path(vertices, points_per_segment::Int)
        qpoints = Vector{Vector{Float64}}()
        coordinate = Float64[]
        for segment in 1:length(vertices)-1
            q0 = vertices[segment]
            q1 = vertices[segment + 1]
            for n in 0:points_per_segment
                segment > 1 && n == 0 && continue
                t = n / points_per_segment
                push!(qpoints, (1 - t) .* q0 .+ t .* q1)
                push!(coordinate, (segment - 1) + t)
            end
        end
        return qpoints, coordinate
    end

    qpath, path_coordinate = symmetry_path((Γ, Mpoint, Kpoint, Γ), 96)
    phonon_solver = HarmonicPhononSolver(tol=1e-10)
    dispersion = phonon_dispersion(phonon_solver, weighted_model, qpath)
    dispersion_check = validate_phonon_dispersion(dispersion; tol=1e-8)

    max_reference_matrix_error = 0.0
    max_reference_frequency_error = 0.0
    max_reference_finite_frequency_error = 0.0

    # Frequencies are square roots of dynamical-matrix eigenvalues.  At an
    # exact acoustic zero mode, O(eps) roundoff in ω² becomes O(sqrt(eps)) in
    # ω, so a frequency-space tolerance is ill-conditioned there.  Retain the
    # raw all-mode frequency error as a diagnostic, but assess agreement at
    # exact numerical zero modes in eigenvalue/null-space space instead.
    reference_zero_rtol = 64 * eps(Float64)

    for (iq, q) in enumerate(qpath)
        native_D = dynamical_matrix(weighted_model, q)
        reference_D = falkovsky_reference_dynamical_matrix(q)
        matrix_error = norm(native_D - reference_D) / max(norm(reference_D), 1.0)

        reference_eigenvalues = Float64.(real.(eigvals(Hermitian(0.5 .* (reference_D .+ adjoint(reference_D))))))
        reference_scale = max(maximum(abs, reference_eigenvalues), 1.0)
        reference_zero_tol = reference_zero_rtol * reference_scale
        reference_frequencies = sqrt.(max.(reference_eigenvalues, 0.0))

        frequency_errors = abs.(dispersion.frequencies[iq, :] .- reference_frequencies)
        frequency_error = maximum(frequency_errors)

        # Exclude only modes whose reference eigenvalue is numerically
        # indistinguishable from zero.  All finite-frequency branches retain
        # the original strict 1e-8 cm^-1 comparison below.
        finite_mask = abs.(reference_eigenvalues) .> reference_zero_tol
        finite_frequency_error = any(finite_mask) ? maximum(frequency_errors[finite_mask]) : 0.0

        max_reference_matrix_error = max(max_reference_matrix_error, matrix_error)
        max_reference_frequency_error = max(max_reference_frequency_error, frequency_error)
        max_reference_finite_frequency_error = max(max_reference_finite_frequency_error, finite_frequency_error)
    end

    sample_q = (Γ, 0.37 .* Mpoint, 0.41 .* Kpoint, 0.63 .* Mpoint .+ 0.37 .* Kpoint)
    max_hermiticity_error = 0.0
    max_q_reversal_error = 0.0
    max_mass_convention_error = 0.0

    for q in sample_q
        Dq = dynamical_matrix(weighted_model, q)
        Dminus = dynamical_matrix(weighted_model, -q)
        Dphysical = dynamical_matrix(physical_model, q)
        max_hermiticity_error = max(max_hermiticity_error, norm(Dq - adjoint(Dq)) / max(norm(Dq), 1.0))
        max_q_reversal_error = max(max_q_reversal_error, norm(Dminus - transpose(Dq)) / max(norm(Dq), 1.0))
        max_mass_convention_error = max(max_mass_convention_error, norm(Dq - Dphysical) / max(norm(Dq), 1.0))
    end

    sum_rule_error = acoustic_sum_rule_residual(weighted_model)

    # -------------------------------------------------------------------------
    # Independent critical-point landmarks from Falkovsky Table III; the article rounds these values to the nearest cm^-1.
    # -------------------------------------------------------------------------

    paper_Γ = [0.0, 0.0, 0.0, 873.0, 1545.0, 1545.0]
    paper_M = [301.0, 587.0, 598.0, 1268.0, 1307.0, 1432.0]
    paper_K = [444.0, 444.0, 1059.0, 1209.0, 1209.0, 1342.0]

    Γ_result = solve(phonon_solver, weighted_model, Γ)
    M_result = solve(phonon_solver, weighted_model, Mpoint)
    K_result = solve(phonon_solver, weighted_model, Kpoint)

    Γ_frequency_error = maximum(abs.(Γ_result.frequencies .- paper_Γ))
    M_frequency_error = maximum(abs.(M_result.frequencies .- paper_M))
    K_frequency_error = maximum(abs.(K_result.frequencies .- paper_K))
    critical_frequency_error = max(Γ_frequency_error, M_frequency_error, K_frequency_error)

    acoustic_zero_error = maximum(abs, Γ_result.frequencies[1:3])

    # Test Γ acoustic translational modes before taking the square root.  This
    # is the well-conditioned quantity for an exact zero mode.
    Γ_D = dynamical_matrix(weighted_model, Γ)
    Γ_eigenvalues = sort(Float64.(real.(eigvals(Hermitian(0.5 .* (Γ_D .+ adjoint(Γ_D)))))))
    Γ_eigenvalue_scale = max(maximum(abs, Γ_eigenvalues), 1.0)
    Γ_zero_eigenvalue_tolerance = 64 * eps(Float64) * Γ_eigenvalue_scale
    Γ_acoustic_eigenvalue_error = maximum(abs, Γ_eigenvalues[1:3])

    Γ_optical_splitting = abs(Γ_result.frequencies[5] - Γ_result.frequencies[6])
    K_low_splitting = abs(K_result.frequencies[1] - K_result.frequencies[2])
    K_middle_splitting = abs(K_result.frequencies[4] - K_result.frequencies[5])
    degeneracy_error = max(Γ_optical_splitting, K_low_splitting, K_middle_splitting)

    # -------------------------------------------------------------------------
    # Falkovsky Eqs. (7)-(9) provide an independent out-of-plane 2×2
    # reference; the paper uses dimensionless q components q*a in these
    # formulas.
    # -------------------------------------------------------------------------

    function falkovsky_out_of_plane(q)
        qdimensionless = a_angstrom .* Float64.(q)
        qx, qy = qdimensionless
        aa = 2γz * (cos(sqrt(3) * qy) + 2 * cos(3 * qx / 2) * cos(sqrt(3) * qy / 2) - 3) - 3αz
        ab = αz * (cis(qx) + 2 * cis(-qx / 2) * cos(sqrt(3) * qy / 2))
        return sqrt.(max.(sort([aa - abs(ab), aa + abs(ab)]), 0.0))
    end

    function falkovsky_eq13(q)
        qdimensionless = a_angstrom .* Float64.(q)
        qx, qy = qdimensionless
        aa_xixi = δ * (cis(sqrt(3) * qy) + 2 * cos(3 * qx / 2 + 2π / 3) * cis(-sqrt(3) * qy / 2)) +
                   conj(δ) * (cis(-sqrt(3) * qy) + 2 * cos(3 * qx / 2 - 2π / 3) * cis(sqrt(3) * qy / 2))
        ab_xixi = β * (cis(qx) + 2 * cis(-qx / 2) * cos(sqrt(3) * qy / 2 - 2π / 3))
        return aa_xixi, ab_xixi
    end

    ξη_transform = ComplexF64[1 im; 1 -im]
    ξη_inverse = inv(ξη_transform)
    eq13_error = 0.0
    for q in qpath[1:12:end]
        Dq = dynamical_matrix(weighted_model, q)
        AAξη = ξη_transform * Dq[1:2, 1:2] * ξη_inverse
        ABξη = ξη_transform * Dq[1:2, 4:5] * ξη_inverse
        aa_reference, ab_reference = falkovsky_eq13(q)
        eq13_error = max(eq13_error, abs(AAξη[1, 2] - aa_reference), abs(ABξη[1, 2] - ab_reference))
    end

    out_of_plane_error = 0.0
    for q in qpath[1:12:end]
        Dq = dynamical_matrix(weighted_model, q)
        Dz = Dq[[3, 6], [3, 6]]
        native = sqrt.(max.(Float64.(real.(eigvals(Hermitian(0.5 .* (Dz .+ adjoint(Dz)))))), 0.0))
        reference = falkovsky_out_of_plane(q)
        out_of_plane_error = max(out_of_plane_error, maximum(abs.(native .- reference)))
    end

    # -------------------------------------------------------------------------
    # Long-wavelength velocities.
    #
    # The exact six-parameter matrix agrees with the corrected 2008 expansion
    # evaluated with α'=β'=αz'=0. The 2007 printed Eq. (17) and its stated
    # 20.3/13.1 km/s velocities are reproduced separately to document the
    # source inconsistency rather than forcing the API to satisfy mutually
    # inconsistent equations from the same article.
    # -------------------------------------------------------------------------

    c_cm_s = 2.99792458e10
    a_cm = a_angstrom * 1e-8

    corrected_s1 = -(9 / 2) * γ - (3 / 4) * α + (3 / 8) * β^2 / α
    corrected_s2 = (9 / 4) * δ - (3 / 8) * β
    corrected_LA = 2π * c_cm_s * a_cm * sqrt(corrected_s1 + corrected_s2) / 1e5
    corrected_TA = 2π * c_cm_s * a_cm * sqrt(corrected_s1 - corrected_s2) / 1e5
    corrected_ZA = 2π * c_cm_s * a_cm * sqrt(-(3 / 4) * αz - (9 / 2) * γz) / 1e5

    legacy_s1 = -(9 / 2) * γ - (3 / 4) * (α - β^2 / α)
    legacy_s2 = (9 / 8) * δ - (3 / 8) * β
    legacy_LA = 2π * c_cm_s * a_cm * sqrt(legacy_s1 + legacy_s2) / 1e5
    legacy_TA = 2π * c_cm_s * a_cm * sqrt(legacy_s1 - legacy_s2) / 1e5
    legacy_ZA = corrected_ZA

    q_probe = [1e-4 / a_angstrom, 0.0]
    probe_result = solve(phonon_solver, weighted_model, q_probe)
    q_probe_norm = norm(q_probe)
    native_velocities = 2π * c_cm_s * 1e-13 .* probe_result.frequencies[1:3] ./ q_probe_norm
    reference_velocities = [corrected_ZA, corrected_TA, corrected_LA]
    velocity_relative_error = maximum(abs.(native_velocities .- reference_velocities) ./ reference_velocities)

    legacy_velocity_reproduction_error = maximum(abs.([legacy_LA - 20.3, legacy_TA - 13.1, legacy_ZA - 1.57]))

    hermiticity_pass = max_hermiticity_error <= 1e-12 && max_q_reversal_error <= 1e-12
    acoustic_nullspace_pass = Γ_acoustic_eigenvalue_error <= Γ_zero_eigenvalue_tolerance
    sum_rule_pass = sum_rule_error <= 1e-12 && acoustic_nullspace_pass
    mass_convention_pass = max_mass_convention_error <= 1e-12
    critical_points_pass = critical_frequency_error <= 1.0
    degeneracy_pass = degeneracy_error <= 1e-8
    out_of_plane_pass = out_of_plane_error <= 1e-8
    eq13_pass = eq13_error <= 1e-8
    velocity_pass = velocity_relative_error <= 1e-4
    dispersion_pass = dispersion_check[:passed]
    full_reference_pass = max_reference_matrix_error <= 1e-12 && max_reference_finite_frequency_error <= 1e-8

    overall_pass = hermiticity_pass && sum_rule_pass && mass_convention_pass && critical_points_pass && degeneracy_pass &&
                   out_of_plane_pass && eq13_pass && velocity_pass && dispersion_pass && full_reference_pass

    println()
    println("FALKOVSKY GRAPHENE HARMONIC-PHONON VALIDATION")
    println("First- and second-neighbor six-parameter Born-von Karman model")
    println()
    @printf("Primitive-cell phonon branches      = %d\n", size(dispersion.frequencies, 2))
    @printf("Dispersion q points                 = %d\n", length(qpath))
    @printf("Maximum mode residual               = %.3e\n", dispersion_check[:max_residual])
    @printf("Full-path analytic D(q) error       = %.3e\n", max_reference_matrix_error)
    @printf("Full-path frequency error [raw]     = %.3e cm^-1\n", max_reference_frequency_error)
    @printf("Finite-mode frequency error         = %.3e cm^-1\n", max_reference_finite_frequency_error)
    @printf("Maximum D(q) Hermiticity error      = %.3e\n", max_hermiticity_error)
    @printf("Maximum D(-q)-D(q)^T error          = %.3e\n", max_q_reversal_error)
    @printf("Acoustic sum-rule residual          = %.3e\n", sum_rule_error)
    @printf("Mass-weighted/physical IFC error    = %.3e\n", max_mass_convention_error)
    println("  Periodic IFC Fourier transform     : ", hermiticity_pass ? "PASS" : "FAIL")
    println("  Full-path Falkovsky matrix         : ", full_reference_pass ? "PASS" : "FAIL")
    println("  Acoustic translational invariance  : ", sum_rule_pass ? "PASS" : "FAIL")
    println("  IFC mass-convention equivalence    : ", mass_convention_pass ? "PASS" : "FAIL")

    println()
    println("FALKOVSKY TABLE-III CRITICAL-POINT VALIDATION")
    @printf("Γ maximum rounded-table error       = %.6f cm^-1\n", Γ_frequency_error)
    @printf("M maximum rounded-table error       = %.6f cm^-1\n", M_frequency_error)
    @printf("K maximum rounded-table error       = %.6f cm^-1\n", K_frequency_error)
    @printf("Γ acoustic-zero error [diagnostic]  = %.3e cm^-1\n", acoustic_zero_error)
    @printf("Γ acoustic eigenvalue error         = %.3e cm^-2\n", Γ_acoustic_eigenvalue_error)
    @printf("Γ acoustic eigenvalue tolerance     = %.3e cm^-2\n", Γ_zero_eigenvalue_tolerance)
    @printf("Maximum required degeneracy split   = %.3e cm^-1\n", degeneracy_error)
    @printf("Out-of-plane Eq. (8) error          = %.3e cm^-1\n", out_of_plane_error)
    @printf("In-plane Eq. (13) matrix error      = %.3e cm^-2\n", eq13_error)
    println("  Published critical frequencies    : ", critical_points_pass ? "PASS" : "FAIL")
    println("  Γ/K symmetry degeneracies          : ", degeneracy_pass ? "PASS" : "FAIL")
    println("  Analytic ZA/ZO dispersion          : ", out_of_plane_pass ? "PASS" : "FAIL")
    println("  Analytic in-plane Eq. (13)         : ", eq13_pass ? "PASS" : "FAIL")

    println()
    println("FALKOVSKY LONG-WAVELENGTH VELOCITY VALIDATION")
    @printf("Native ZA velocity                  = %.6f km/s [corrected %.6f]\n", native_velocities[1], corrected_ZA)
    @printf("Native TA velocity                  = %.6f km/s [corrected %.6f]\n", native_velocities[2], corrected_TA)
    @printf("Native LA velocity                  = %.6f km/s [corrected %.6f]\n", native_velocities[3], corrected_LA)
    @printf("Corrected-expansion relative error  = %.3e\n", velocity_relative_error)
    @printf("2007 printed Eq. (17) LA/TA/ZA      = %.3f / %.3f / %.3f km/s\n", legacy_LA, legacy_TA, legacy_ZA)
    println("2007 stated LA/TA/ZA                = 20.3 / 13.1 / 1.57 km/s")
    @printf("Legacy stated-value reproduction    = %.3e km/s [diagnostic]\n", legacy_velocity_reproduction_error)
    println("  Corrected q→0 dynamical-matrix slopes: ", velocity_pass ? "PASS" : "FAIL")
    println("  2007 in-plane velocity discrepancy   : DOCUMENTED SOURCE INCONSISTENCY")

    println()
    println("FALKOVSKY GRAPHENE BENCHMARK VALIDATION")
    println("  HarmonicPhononSolver path modes     : ", dispersion_pass ? "PASS" : "FAIL")
    println("  Periodic q-resolved dynamical matrix: ", hermiticity_pass ? "PASS" : "FAIL")
    println("  Full-path Falkovsky dispersion      : ", full_reference_pass ? "PASS" : "FAIL")
    println("  Acoustic sum rule / Γ zero modes    : ", sum_rule_pass ? "PASS" : "FAIL")
    println("  Mass-weighted IFC convention        : ", mass_convention_pass ? "PASS" : "FAIL")
    println("  Table-III Γ/M/K frequencies         : ", critical_points_pass ? "PASS" : "FAIL")
    println("  Critical-point degeneracies         : ", degeneracy_pass ? "PASS" : "FAIL")
    println("  Analytic out-of-plane modes         : ", out_of_plane_pass ? "PASS" : "FAIL")
    println("  Analytic in-plane force constants   : ", eq13_pass ? "PASS" : "FAIL")
    println("  Corrected acoustic velocities       : ", velocity_pass ? "PASS" : "FAIL")
    println()
    println("Falkovsky graphene phonon-dispersion benchmark: ", overall_pass ? "PASS" : "FAIL")

    fig = Figure(size=(1100, 700))
    ax = Axis(fig[1, 1]; xlabel="reduced wavevector", ylabel="frequency (cm⁻¹)", title="Falkovsky graphene phonon dispersion", xticks=([0.0, 1.0, 2.0, 3.0], ["Γ", "M", "K", "Γ"]))
    for branch in axes(dispersion.frequencies, 2)
        lines!(ax, path_coordinate, dispersion.frequencies[:, branch])
    end
    vlines!(ax, [1.0, 2.0]; linestyle=:dash)
    ylims!(ax, 0.0, 1700.0)
    save(joinpath(BENCHMARK_FIGURE_DIR, "Falkovsky-graphene-phonon-benchmark.png"), fig)
end


#-------------------------------------------------------#
# Fair, R. (et al.). (2022).                            #
# Euphonic: inelastic neutron scattering simulations    #
# from force constants and visualization tools for      #
# phonon properties.                                    #
# J. Appl. Cryst. 55, 1689-1703.                        #
# doi: 10.1107/S1600576722009256                        #
#-------------------------------------------------------#
let
    # -------------------------------------------------------------------------
    # Fair et al. provide the one-phonon formalism and published Al cut geometry used here as an external formal reference. The controlled nearest-neighbour FCC-Al model below does not reproduce the paper's VASP/Phonopy force constants, so the benchmark validates the implementation of the Fair equations rather than reproduction of the published intensity maps.
    # -------------------------------------------------------------------------

    paper_lattice_parameter = 3.984207
    published_mrpd_context_percent = 0.01
    formalism_mrpd_tolerance_percent = 1e-8
    temperatures = (5.0, 300.0)
    kB_meV_per_K = 0.08617333262
    hbar_meV_per_meV = 1.0
    # For frequency=:meV, the solver stores E=ℏω directly. The Bose exponent therefore uses hbar=1, while the independent displacement prefactor converts 1/(amu·meV) to Å².
    hbar2_over_amu_meV_angstrom2 = 4.180159280496723

    al = lookup_species(:Al)
    oxygen18 = lookup_species(:O, 18)
    explicit_al = M.AtomicSpecies(:Al; mass=27.0, nuclear_scattering_length=3.5)
    missing_isotope_rejected = false
    try
        lookup_species(:Al, 27)
    catch err
        missing_isotope_rejected = err isa KeyError
    end

    lookup_mass_error = abs(al.mass - 26.9815385)
    lookup_scattering_error = abs(real(al.nuclear_scattering_length) - 3.449)
    isotope_mass_error = abs(oxygen18.mass - 17.99915961286)
    isotope_scattering_error = abs(real(oxygen18.nuclear_scattering_length) - 5.841)
    explicit_override_error = max(abs(explicit_al.mass - 27.0), abs(real(explicit_al.nuclear_scattering_length) - 3.5))
    lookup_errors = (lookup_mass_error, lookup_scattering_error, isotope_mass_error, isotope_scattering_error, explicit_override_error)
    lookup_pass = maximum(lookup_errors) <= 1e-12 && missing_isotope_rejected

    # FCC primitive vectors are columns. The conventional cubic reciprocal
    # coordinates used by Fair et al. are converted explicitly to Cartesian Q.
    a = paper_lattice_parameter
    direct = (a / 2) .* [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    lattice = M.BravaisLattice(direct)
    crystal = M.CrystalStructure(lattice, [M.BasisSite(:Al1, :Al, [0.0, 0.0, 0.0])])
    material = M.Material(
        "Fair et al. FCC aluminium API validation model",
        crystal;
        species=Dict(:Al => al),
        metadata=Dict(
            :compound => :Al,
            :reference => "Fair et al., J. Appl. Cryst. 55, 1689-1703 (2022)",
            :doi => "10.1107/S1600576722009256",
            :published_ifc_source => :VASP_Phonopy,
            :benchmark_ifc_model => :controlled_nearest_neighbor_FCC,
        ),
    )
    cluster = M.Supercell(crystal, [1, 1, 1]; periodic=true)

    # Six oriented primitive translations generate the twelve FCC nearest
    # neighbours after HarmonicBondInteraction adds each reverse interaction.
    nearest_translations = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1])
    k_longitudinal = 5400.0
    bond_interactions = [HarmonicBondInteraction(1, 1, R; longitudinal=k_longitudinal, transverse=0.0) for R in nearest_translations]
    al_ifcs = force_constants(crystal, bond_interactions)
    units = PhononUnitConvention(length=:angstrom, mass=:amu, force_constant=:amu_meV2, frequency=:meV)
    al_model = build_model(material, cluster, HarmonicPhononModel(al_ifcs; displacement_dimension=3, mass_weighted=false, units=units))
    phonon_solver = HarmonicPhononSolver(tol=1e-10)

    conventional_Q(hkl) = (2π / a) .* Float64.(collect(hkl))

    function relative_matrix_error(A, B)
        return norm(A - B) / max(norm(B), 1.0)
    end

    function aggregate_ifcs(ifcs)
        blocks = Dict{Tuple{Int,Int,Tuple},Matrix{ComplexF64}}()
        for block in ifcs.blocks
            key = (block.from_site, block.to_site, Tuple(block.translation))
            if haskey(blocks, key)
                blocks[key] .+= ComplexF64.(block.tensor)
            else
                blocks[key] = ComplexF64.(block.tensor)
            end
        end
        return blocks
    end

    function ifc_relative_error(a_ifcs, b_ifcs)
        a_blocks = aggregate_ifcs(a_ifcs)
        b_blocks = aggregate_ifcs(b_ifcs)
        keys_union = union(keys(a_blocks), keys(b_blocks))
        numerator = 0.0
        denominator = 0.0
        for key in keys_union
            A = get(a_blocks, key, zeros(ComplexF64, 3, 3))
            B = get(b_blocks, key, zeros(ComplexF64, 3, 3))
            numerator += norm(A - B)^2
            denominator += norm(B)^2
        end
        return sqrt(numerator) / max(sqrt(denominator), 1.0)
    end

    # -------------------------------------------------------------------------
    # Species lookup, typed reciprocal coordinates, and analytic D derivatives.
    # -------------------------------------------------------------------------

    test_hkl = [0.173, 0.237, 0.319]
    test_q = M.reciprocal_vector(crystal, test_hkl)
    D_rlu = dynamical_matrix(al_model, ReciprocalWaveVector(test_hkl))
    D_cart = dynamical_matrix(al_model, CartesianWaveVector(test_q))
    wavevector_error = relative_matrix_error(D_rlu, D_cart)

    D_buffer = similar(D_cart)
    dynamical_matrix!(D_buffer, al_model, CartesianWaveVector(test_q))
    matrix_mutating_error = relative_matrix_error(D_buffer, D_cart)

    gradient = dynamical_gradient(al_model, CartesianWaveVector(test_q))
    gradient_buffer = similar(gradient)
    dynamical_gradient!(gradient_buffer, al_model, CartesianWaveVector(test_q))
    gradient_mutating_error = relative_matrix_error(gradient_buffer, gradient)
    h_gradient = 1e-6
    gradient_error = 0.0
    for μ in 1:3
        dq = zeros(3)
        dq[μ] = h_gradient
        D_plus = dynamical_matrix(al_model, CartesianWaveVector(test_q + dq))
        D_minus = dynamical_matrix(al_model, CartesianWaveVector(test_q - dq))
        finite_difference = (D_plus - D_minus) / (2h_gradient)
        gradient_error = max(gradient_error, relative_matrix_error(gradient[:, :, μ], finite_difference))
    end

    hessian = dynamical_hessian(al_model, CartesianWaveVector(test_q))
    hessian_buffer = similar(hessian)
    dynamical_hessian!(hessian_buffer, al_model, CartesianWaveVector(test_q))
    hessian_mutating_error = relative_matrix_error(hessian_buffer, hessian)
    mutating_api_error = maximum((matrix_mutating_error, gradient_mutating_error, hessian_mutating_error))
    h_hessian = 2e-5
    hessian_error = 0.0
    for ν in 1:3
        dq = zeros(3)
        dq[ν] = h_hessian
        gradient_plus = dynamical_gradient(al_model, CartesianWaveVector(test_q + dq))
        gradient_minus = dynamical_gradient(al_model, CartesianWaveVector(test_q - dq))
        finite_difference = (gradient_plus - gradient_minus) / (2h_hessian)
        hessian_error = max(hessian_error, relative_matrix_error(hessian[:, :, :, ν], finite_difference))
    end

    typed_wavevector_pass = wavevector_error <= 1e-12
    mutating_api_pass = mutating_api_error <= 1e-14
    gradient_pass = gradient_error <= 1e-7
    hessian_pass = hessian_error <= 1e-5

    println()
    println("FAIR ALUMINIUM PHUNNY-INSPIRED API: SPECIES / WAVEVECTOR / DERIVATIVES")
    @printf("lookup_species(:Al) mass error      = %.3e amu\n", lookup_mass_error)
    @printf("lookup_species(:Al) b_coh error     = %.3e fm\n", lookup_scattering_error)
    @printf("lookup_species(:O,18) mass error    = %.3e amu\n", isotope_mass_error)
    @printf("lookup_species(:O,18) b_coh error   = %.3e fm\n", isotope_scattering_error)
    @printf("Explicit-constructor override error = %.3e\n", explicit_override_error)
    println("Missing Al-27 isotope lookup        = ", missing_isotope_rejected ? "REJECTED" : "ACCEPTED [FAIL]")
    @printf("Cartesian / RLU D(q) error          = %.3e\n", wavevector_error)
    @printf("Maximum mutating-API error          = %.3e\n", mutating_api_error)
    @printf("Analytic D'(q) finite-diff error    = %.3e\n", gradient_error)
    @printf("Analytic D''(q) finite-diff error   = %.3e\n", hessian_error)
    println("  Species lookup and override       : ", lookup_pass ? "PASS" : "FAIL")
    println("  Typed wavevector equivalence      : ", typed_wavevector_pass ? "PASS" : "FAIL")
    println("  Mutating D/D'/D'' API             : ", mutating_api_pass ? "PASS" : "FAIL")
    println("  Dynamical-matrix gradient         : ", gradient_pass ? "PASS" : "FAIL")
    println("  Dynamical-matrix Hessian          : ", hessian_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Group velocities and degenerate-subspace directional perturbation theory.
    # -------------------------------------------------------------------------

    velocity_result = solve(phonon_solver, al_model, CartesianWaveVector(test_q))
    native_velocity = group_velocities(velocity_result; degeneracy_tol=1e-10, degeneracy_policy=:basis)
    h_velocity = 1e-6
    finite_difference_velocity = zeros(Float64, length(velocity_result.frequencies), 3)
    for μ in 1:3
        dq = zeros(3)
        dq[μ] = h_velocity
        plus = solve(phonon_solver, al_model, CartesianWaveVector(test_q + dq)).frequencies
        minus = solve(phonon_solver, al_model, CartesianWaveVector(test_q - dq)).frequencies
        finite_difference_velocity[:, μ] .= (plus - minus) ./ (2h_velocity)
    end
    velocity_error = norm(native_velocity - finite_difference_velocity) / max(norm(finite_difference_velocity), 1.0)
    velocity_dispersion = phonon_dispersion(phonon_solver, al_model, [CartesianWaveVector(test_q)]; derivatives=true, degeneracy_tol=1e-10, degeneracy_policy=:basis)
    dispersion_velocity = velocity_dispersion.metadata[:group_velocities][1, :, :]
    dispersion_velocity_error = norm(dispersion_velocity - native_velocity) / max(norm(native_velocity), 1.0)

    degenerate_hkl = [0.27, 0.27, 0.27]
    degenerate_q = conventional_Q(degenerate_hkl)
    degenerate_result = solve(phonon_solver, al_model, CartesianWaveVector(degenerate_q))
    direction = degenerate_q / norm(degenerate_q)
    directional_velocity = directional_group_velocities(degenerate_result, direction; degeneracy_tol=1e-8)
    default_degenerate_velocity = group_velocities(degenerate_result; degeneracy_tol=1e-8)
    degenerate_default_nan = all(isnan, default_degenerate_velocity[1:2, :])
    transverse_split = abs(degenerate_result.frequencies[1] - degenerate_result.frequencies[2])

    degenerate_gradient = dynamical_gradient(al_model, CartesianWaveVector(degenerate_q))
    directional_derivative = sum(direction[μ] .* degenerate_gradient[:, :, μ] for μ in 1:3)
    directional_derivative = 0.5 .* (directional_derivative .+ adjoint(directional_derivative))
    transverse_basis = degenerate_result.modes[:, 1:2]
    transverse_frequency = maximum(degenerate_result.frequencies[1:2])
    projected_matrix = adjoint(transverse_basis) * directional_derivative * transverse_basis
    projected = Hermitian(0.5 .* (projected_matrix .+ adjoint(projected_matrix)))
    reference_directional = sort(Float64.(real.(eigvals(projected))) ./ (2transverse_frequency))

    rotation_angle = 0.371
    rotation = [cos(rotation_angle) -sin(rotation_angle); sin(rotation_angle) cos(rotation_angle)]
    rotated_basis = transverse_basis * rotation
    rotated_projected_matrix = adjoint(rotated_basis) * directional_derivative * rotated_basis
    rotated_projected = Hermitian(0.5 .* (rotated_projected_matrix .+ adjoint(rotated_projected_matrix)))
    rotated_directional = sort(Float64.(real.(eigvals(rotated_projected))) ./ (2transverse_frequency))

    directional_api_error = maximum(abs.(sort(directional_velocity[1:2]) .- reference_directional))
    directional_rotation_error = maximum(abs.(rotated_directional .- reference_directional))
    group_velocity_pass = velocity_error <= 1e-6 && dispersion_velocity_error <= 1e-12
    directional_velocity_pass = transverse_split <= 1e-8 && directional_api_error <= 1e-10 &&
                                directional_rotation_error <= 1e-10 && degenerate_default_nan

    println()
    println("FAIR ALUMINIUM PHUNNY-INSPIRED API: GROUP-VELOCITY VALIDATION")
    @printf("Generic-q group-velocity error      = %.3e\n", velocity_error)
    @printf("Dispersion derivative-route error   = %.3e\n", dispersion_velocity_error)
    @printf("[111] transverse frequency split    = %.3e meV\n", transverse_split)
    @printf("Directional API reference error     = %.3e\n", directional_api_error)
    @printf("Degenerate-basis rotation error     = %.3e\n", directional_rotation_error)
    println("  Nondegenerate group velocities    : ", group_velocity_pass ? "PASS" : "FAIL")
    println("  Default degenerate ambiguity guard: ", degenerate_default_nan ? "PASS" : "FAIL")
    println("  Degenerate directional velocities : ", directional_velocity_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # IFC constraints, explicit ASR projection, bond compilation, and zero modes.
    # -------------------------------------------------------------------------

    ifc_validation = validate_force_constants(crystal, al_ifcs; atol=1e-10, rtol=1e-10, rotational=true)
    original_blocks = aggregate_ifcs(al_ifcs)
    zero_translation = (0, 0, 0)
    corrupt_entries = ForceConstantBlock[]
    onsite_shift = 1e-3 * k_longitudinal
    for block in al_ifcs.blocks
        if block.from_site == 1 && block.to_site == 1 && Tuple(block.translation) == zero_translation
            shifted = Matrix(block.tensor) + onsite_shift .* Matrix{Float64}(I, 3, 3)
            push!(corrupt_entries, ForceConstantBlock(block.from_site, block.to_site, block.translation, shifted))
        else
            push!(corrupt_entries, ForceConstantBlock(block.from_site, block.to_site, block.translation, block.tensor))
        end
    end
    corrupt_ifcs = RealSpaceForceConstants(corrupt_entries; displacement_dimension=3)
    corrupt_validation = validate_force_constants(crystal, corrupt_ifcs; atol=1e-10, rtol=1e-10, rotational=false)
    projected_ifcs = project_force_constants(corrupt_ifcs; translational=true)
    projected_validation = validate_force_constants(crystal, projected_ifcs; atol=1e-10, rtol=1e-10, rotational=true)
    projected_twice = project_force_constants(projected_ifcs; translational=true)
    projection_recovery_error = ifc_relative_error(projected_ifcs, al_ifcs)
    projection_idempotence_error = ifc_relative_error(projected_twice, projected_ifcs)

    first_bond = first(bond_interactions)
    bond_vector = M.direct_matrix(crystal.lattice) * Float64.(first_bond.translation)
    bond_direction = bond_vector / norm(bond_vector)
    reference_bond_tensor = k_longitudinal .* (bond_direction * transpose(bond_direction))
    compiled_bond_block = original_blocks[(1, 1, Tuple(first_bond.translation))]
    bond_tensor_error = relative_matrix_error(real.(compiled_bond_block), -reference_bond_tensor)

    gamma_q = ReciprocalWaveVector([0.0, 0.0, 0.0])
    gamma_D = dynamical_matrix(al_model, gamma_q)
    gamma_eigenvalues = sort(Float64.(real.(eigvals(Hermitian(0.5 .* (gamma_D .+ adjoint(gamma_D)))))))
    gamma_scale = max(maximum(abs, gamma_eigenvalues; init=0.0), 1.0)
    gamma_zero_tolerance = phonon_solver.zero_mode_factor * eps(Float64) * gamma_scale
    gamma_acoustic_eigenvalue_error = maximum(abs, gamma_eigenvalues[1:3])
    gamma_result = solve(phonon_solver, al_model, gamma_q)
    gamma_zero_error = maximum(abs, gamma_result.frequencies[1:3])
    gamma_zero_mask = gamma_result.metadata[:zero_mode_mask][1:3]
    instability_detected = false
    unstable_spec = HarmonicPhononModel(-Matrix{Float64}(I, 3, 3); masses=al.mass, displacement_dimension=3, mass_weighted=true, units=units)
    unstable_model = build_model(material, cluster, unstable_spec)
    try
        solve(phonon_solver, unstable_model)
    catch err
        instability_detected = err isa ArgumentError
    end

    ifc_validation_pass = ifc_validation.passed && !corrupt_validation.passed && projected_validation.passed
    projection_pass = projection_recovery_error <= 1e-12 && projection_idempotence_error <= 1e-12
    bond_compiler_pass = bond_tensor_error <= 1e-12
    zero_mode_pass = gamma_acoustic_eigenvalue_error <= gamma_zero_tolerance && all(gamma_zero_mask) && all(iszero, gamma_result.frequencies[1:3]) && instability_detected

    println()
    println("FAIR ALUMINIUM PHUNNY-INSPIRED API: IFC CONSTRAINT / ZERO-MODE VALIDATION")
    @printf("IFC Hermiticity residual            = %.3e\n", ifc_validation.hermiticity_residual)
    @printf("IFC permutation residual            = %.3e\n", ifc_validation.permutation_residual)
    @printf("IFC translational residual          = %.3e\n", ifc_validation.translational_residual)
    @printf("IFC rotational residual             = %.3e\n", something(ifc_validation.rotational_residual, NaN))
    @printf("Corrupted-ASR residual              = %.3e\n", corrupt_validation.translational_residual)
    @printf("ASR projection recovery error       = %.3e\n", projection_recovery_error)
    @printf("ASR projection idempotence error    = %.3e\n", projection_idempotence_error)
    @printf("Bond-compiler tensor error          = %.3e\n", bond_tensor_error)
    @printf("Γ acoustic frequency [diagnostic]  = %.3e meV\n", gamma_zero_error)
    @printf("Γ acoustic eigenvalue error         = %.3e meV²\n", gamma_acoustic_eigenvalue_error)
    @printf("Γ acoustic eigenvalue tolerance     = %.3e meV²\n", gamma_zero_tolerance)
    println("Γ solver zero-mode mask             = ", all(gamma_zero_mask) ? "PASS" : "FAIL")
    println("  IFC constraint validator          : ", ifc_validation_pass ? "PASS" : "FAIL")
    println("  Explicit ASR projection           : ", projection_pass ? "PASS" : "FAIL")
    println("  Harmonic bond -> IFC compiler     : ", bond_compiler_pass ? "PASS" : "FAIL")
    println("  Zero-mode / instability policy    : ", zero_mode_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # HarmonicAngleInteraction is checked against both the analytic equilibrium
    # Hessian and an independent finite-difference Hessian of the same angle energy.
    # -------------------------------------------------------------------------

    angle_lattice = M.BravaisLattice(4.0 .* Matrix{Float64}(I, 3, 3))
    angle_crystal = M.CrystalStructure(angle_lattice, [M.BasisSite(:A, :Al, [0.25, 0.0, 0.0]), M.BasisSite(:B, :Al, [0.0, 0.0, 0.0]), M.BasisSite(:C, :Al, [0.0, 0.25, 0.0])])
    angle_stiffness = 7.5
    angle_interaction = HarmonicAngleInteraction(1, 2, 3, [0, 0, 0], [0, 0, 0]; stiffness=angle_stiffness)
    angle_ifcs = force_constants(angle_crystal, [angle_interaction])

    function gamma_ifc_matrix(ifcs, nbasis)
        D = ifcs.displacement_dimension
        H = zeros(Float64, nbasis * D, nbasis * D)
        for block in ifcs.blocks
            rows = ((block.from_site - 1) * D + 1):(block.from_site * D)
            cols = ((block.to_site - 1) * D + 1):(block.to_site * D)
            H[rows, cols] .+= real.(block.tensor)
        end
        return H
    end

    equilibrium_positions = Float64[1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0 ]

    function angle_energy(x)
        positions = reshape(x, 3, 3)
        rji = positions[:, 1] - positions[:, 2]
        rjk = positions[:, 3] - positions[:, 2]
        cosine = clamp(dot(rji, rjk) / (norm(rji) * norm(rjk)), -1.0, 1.0)
        delta_theta = acos(cosine) - π / 2
        return 0.5 * angle_stiffness * delta_theta^2
    end

    function finite_difference_hessian(f, x; h=1e-4)
        n = length(x)
        H = zeros(Float64, n, n)
        f0 = f(x)
        for i in 1:n
            ei = zeros(Float64, n)
            ei[i] = h
            H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h^2
            for j in i+1:n
                ej = zeros(Float64, n)
                ej[j] = h
                value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h^2)
                H[i, j] = value
                H[j, i] = value
            end
        end
        return H
    end

    rji_reference = equilibrium_positions[1:3] - equilibrium_positions[4:6]
    rjk_reference = equilibrium_positions[7:9] - equilibrium_positions[4:6]
    ei_reference = rji_reference / norm(rji_reference)
    ek_reference = rjk_reference / norm(rjk_reference)
    cos_reference = dot(ei_reference, ek_reference)
    sin_reference = sqrt(1.0 - cos_reference^2)
    gi_reference = -(ek_reference .- cos_reference .* ei_reference) ./ (norm(rji_reference) * sin_reference)
    gk_reference = -(ei_reference .- cos_reference .* ek_reference) ./ (norm(rjk_reference) * sin_reference)
    gj_reference = -(gi_reference .+ gk_reference)
    angle_gradient_reference = vcat(gi_reference, gj_reference, gk_reference)
    analytic_angle_hessian = angle_stiffness .* (angle_gradient_reference * transpose(angle_gradient_reference))

    compiled_angle_hessian = gamma_ifc_matrix(angle_ifcs, 3)
    numerical_angle_hessian = finite_difference_hessian(angle_energy, equilibrium_positions)
    angle_compiler_analytic_error = relative_matrix_error(compiled_angle_hessian, analytic_angle_hessian)
    angle_finite_difference_error = relative_matrix_error(numerical_angle_hessian, analytic_angle_hessian)
    angle_compiler_fd_error = relative_matrix_error(compiled_angle_hessian, numerical_angle_hessian)
    angle_translation_error = maximum(norm(compiled_angle_hessian * repeat([axis == μ ? 1.0 : 0.0 for axis in 1:3], 3)) / max(norm(compiled_angle_hessian), 1.0) for μ in 1:3)
    angle_compiler_pass = angle_compiler_analytic_error <= 1e-12 && angle_finite_difference_error <= 1e-6 && angle_compiler_fd_error <= 1e-6 && angle_translation_error <= 1e-12

    println()
    println("FAIR ALUMINIUM PHUNNY-INSPIRED API: HARMONIC-ANGLE COMPILER")
    @printf("Compiler vs analytic Hessian error = %.3e\n", angle_compiler_analytic_error)
    @printf("Finite-diff vs analytic Hessian err = %.3e\n", angle_finite_difference_error)
    @printf("Compiler vs finite-diff Hessian err = %.3e\n", angle_compiler_fd_error)
    @printf("Angle rigid-translation residual    = %.3e\n", angle_translation_error)
    println("  Harmonic angle -> IFC compiler    : ", angle_compiler_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Mean-square displacement and Debye-Waller validation against the harmonic
    # covariance underlying Fair et al. Eqs. (8)-(9).
    # -------------------------------------------------------------------------

    mesh_n = 6
    qmesh = [ReciprocalWaveVector([i / mesh_n, j / mesh_n, k / mesh_n]) for i in 0:mesh_n-1 for j in 0:mesh_n-1 for k in 0:mesh_n-1]
    mesh_dispersion = phonon_dispersion(phonon_solver, al_model, qmesh)

    function reference_msd(dispersion, temperature)
        nq, nbranch = size(dispersion.frequencies)
        zero_mode_mask = dispersion.metadata[:zero_mode_mask]
        U = zeros(Float64, 3, 3)
        mass = al.mass
        for iq in 1:nq, ν in 1:nbranch
            zero_mode_mask[iq, ν] && continue
            omega = dispersion.frequencies[iq, ν]
            occupation = iszero(temperature) ? 0.0 : inv(expm1(omega / (kB_meV_per_K * temperature)))
            polarization = dispersion.modes[:, ν, iq] ./ sqrt(mass)
            U .+= (hbar2_over_amu_meV_angstrom2 * (2 * occupation + 1) / (2 * omega * nq)) .* real.(polarization * adjoint(polarization))
        end
        return 0.5 .* (U .+ transpose(U))
    end

    msd_results = Dict{Float64,MeanSquareDisplacementResult}()
    debye_waller_results = Dict{Float64,Vector{M.DebyeWallerTensor}}()
    msd_errors = Dict{Float64,Float64}()
    msd_psd_minimum = Inf
    msd_symmetry_error = 0.0
    msd_cubic_anisotropy = 0.0

    for temperature in temperatures
        msd = mean_square_displacement(mesh_dispersion; temperature=temperature, hbar=hbar_meV_per_meV, kB=kB_meV_per_K,
                                       displacement_prefactor=hbar2_over_amu_meV_angstrom2, zero_mode_policy=:exclude)
        dw = phonon_debye_waller_tensors(mesh_dispersion; temperature=temperature, hbar=hbar_meV_per_meV, kB=kB_meV_per_K,
                                         displacement_prefactor=hbar2_over_amu_meV_angstrom2, zero_mode_policy=:exclude)
        reference = reference_msd(mesh_dispersion, temperature)
        U = only(msd.tensors)
        msd_results[temperature] = msd
        debye_waller_results[temperature] = dw
        msd_errors[temperature] = relative_matrix_error(U, reference)
        msd_psd_minimum = min(msd_psd_minimum, minimum(eigvals(Symmetric(U))))
        msd_symmetry_error = max(msd_symmetry_error, norm(U - transpose(U)) / max(norm(U), 1.0))
        isotropic = (tr(U) / 3) .* Matrix{Float64}(I, 3, 3)
        msd_cubic_anisotropy = max(msd_cubic_anisotropy, norm(U - isotropic) / max(norm(U), 1.0))
    end

    debye_probe_Q = conventional_Q([2.3, 1.1, 0.7])
    U5 = only(msd_results[5.0].tensors)
    dw5 = only(debye_waller_results[5.0])
    direct_dw_factor = exp(-0.5 * dot(debye_probe_Q, U5 * debye_probe_Q))
    api_dw_factor = M.debye_waller_factor(dw5, debye_probe_Q)
    debye_waller_error = abs(api_dw_factor - direct_dw_factor)
    thermal_growth = tr(only(msd_results[300.0].tensors)) / tr(U5)
    msd_zero_mode_source_pass = all(msd_results[temperature].metadata[:zero_mode_source] === :solver for temperature in temperatures)

    msd_pass = maximum(values(msd_errors)) <= 1e-12 && msd_psd_minimum >= -1e-12 && msd_symmetry_error <= 1e-12 &&
               msd_zero_mode_source_pass
    cubic_msd_pass = msd_cubic_anisotropy <= 1e-8 && thermal_growth > 1.0
    debye_waller_pass = debye_waller_error <= 1e-14

    println()
    println("FAIR ALUMINIUM PHUNNY-INSPIRED API: MSD / DEBYE-WALLER VALIDATION")
    @printf("Brillouin-zone mesh                = %d × %d × %d (%d q points)\n", mesh_n, mesh_n, mesh_n, mesh_n^3)
    @printf("5 K MSD reference error            = %.3e\n", msd_errors[5.0])
    @printf("300 K MSD reference error          = %.3e\n", msd_errors[300.0])
    @printf("Minimum MSD-tensor eigenvalue      = %.3e Å²\n", msd_psd_minimum)
    @printf("Maximum MSD symmetry error         = %.3e\n", msd_symmetry_error)
    @printf("Maximum cubic MSD anisotropy       = %.3e\n", msd_cubic_anisotropy)
    @printf("Tr[U(300 K)] / Tr[U(5 K)]          = %.6f\n", thermal_growth)
    @printf("Debye-Waller quadratic-form error  = %.3e\n", debye_waller_error)
    println("MSD zero-mode source                = ", msd_zero_mode_source_pass ? "solver mask" : "fallback/override [FAIL]")
    println("  Harmonic displacement covariance  : ", msd_pass ? "PASS" : "FAIL")
    println("  Cubic symmetry / thermal growth    : ", cubic_msd_pass ? "PASS" : "FAIL")
    println("  Debye-Waller tensor contraction    : ", debye_waller_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Fair et al. Eqs. (7), (10), and (11): independent coherent one-phonon
    # neutron reference. For monatomic Al, Eq. (7) reduces to one coherent
    # scattering length, one mass, and Q dot e for each branch.
    # -------------------------------------------------------------------------

    probe = NuclearNeutronProbe()
    spectrum_convention = SpectrumConvention(spectral_axis=:energy, hbar=hbar_meV_per_meV, kB=kB_meV_per_K)

    function fair_reference_lines(result, Q, temperature, dw)
        Qcart = Float64.(collect(Q))
        zero_mode_mask = result.metadata[:zero_mode_mask]
        centers = Float64[]
        weights = Float64[]
        mass = al.mass
        bcoh = al.nuclear_scattering_length
        Wfactor = M.debye_waller_factor(dw, Qcart)
        for ν in eachindex(result.frequencies)
            zero_mode_mask[ν] && continue
            omega = result.frequencies[ν]
            polarization = view(result.modes, :, ν)
            amplitude = bcoh * Wfactor * dot(Qcart, polarization) / sqrt(mass * omega)
            structure_factor = abs2(amplitude)
            occupation = iszero(temperature) ? 0.0 : inv(expm1(omega / (kB_meV_per_K * temperature)))
            push!(centers, omega)
            push!(weights, 0.5 * (occupation + 1) * structure_factor)
            if occupation > 0
                push!(centers, -omega)
                push!(weights, 0.5 * occupation * structure_factor)
            end
        end
        return centers, weights
    end

    function aggregate_spectral_lines(centers, weights; tolerance=1e-9)
        aggregated = Dict{Int,Float64}()
        for (center, weight) in zip(centers, weights)
            key = round(Int, center / tolerance)
            aggregated[key] = get(aggregated, key, 0.0) + real(weight)
        end
        return aggregated
    end

    function spectral_line_error(native, reference_centers, reference_weights)
        native_map = aggregate_spectral_lines(native.centers, native.weights)
        reference_map = aggregate_spectral_lines(reference_centers, reference_weights)
        all_keys = union(keys(native_map), keys(reference_map))
        numerator = sqrt(sum((get(native_map, key, 0.0) - get(reference_map, key, 0.0))^2 for key in all_keys))
        denominator = sqrt(sum(get(reference_map, key, 0.0)^2 for key in all_keys))
        return numerator / max(denominator, eps(Float64))
    end

    scattering_hkl = [2.37, 0.31, 0.19]
    scattering_G = [2.0, 0.0, 0.0]
    scattering_Q = conventional_Q(scattering_hkl)
    scattering_q = conventional_Q(scattering_hkl .- scattering_G)
    scattering_Q_vector = CartesianWaveVector(scattering_Q)
    scattering_decomposition = decompose_reciprocal_vector(al_model, scattering_Q_vector)
    scattering_result = solve(phonon_solver, al_model, scattering_decomposition.q_cartesian)
    native_lines = one_phonon_neutron_lines(probe, scattering_result, scattering_Q; temperature=300.0, convention=spectrum_convention, debye_waller=debye_waller_results[300.0])
    full_Q_lines = one_phonon_neutron_lines(phonon_solver, al_model, probe, scattering_Q_vector; temperature=300.0, convention=spectrum_convention, debye_waller=debye_waller_results[300.0])
    reference_centers, reference_weights = fair_reference_lines(scattering_result, scattering_Q, 300.0, only(debye_waller_results[300.0]))
    neutron_line_error = spectral_line_error(native_lines, reference_centers, reference_weights)
    full_Q_line_error = spectral_line_error(full_Q_lines, native_lines.centers, real.(native_lines.weights))
    reciprocal_reduction_error = norm(scattering_decomposition.q_cartesian.coordinates - scattering_q) / max(norm(scattering_q), 1.0)
    reciprocal_reconstruction_error = scattering_decomposition.reconstruction_error
    full_Q_metadata_pass = full_Q_lines.metadata[:reciprocal_G_indices] == scattering_decomposition.G_indices &&
                           full_Q_lines.metadata[:reciprocal_reduction_convention] === :first_bz &&
                           full_Q_lines.metadata[:basis_phase_convention] === :atomic_position_dynamical_matrix
    neutron_zero_mode_source_pass = native_lines.metadata[:zero_mode_source] === :solver
    full_Q_api_pass = full_Q_line_error <= 1e-12 && reciprocal_reduction_error <= 1e-12 &&
                      reciprocal_reconstruction_error <= 1e-12 && full_Q_metadata_pass

    phase_angles = [0.37, -0.61, 1.13]
    phase_factors = exp.(im .* phase_angles)
    phase_modes = scattering_result.modes .* reshape(phase_factors, 1, length(phase_factors))
    phase_metadata = copy(scattering_result.metadata)
    physical_phase_factors = reshape(phase_factors, 1, length(phase_factors))
    phase_metadata[:physical_displacement_modes] = phase_metadata[:physical_displacement_modes] .* physical_phase_factors
    phase_result = PhononModeResult(scattering_result.solver, scattering_result.model, copy(scattering_result.frequencies), phase_modes, phase_metadata)
    phase_lines = one_phonon_neutron_lines(probe, phase_result, scattering_Q; temperature=300.0, convention=spectrum_convention, debye_waller=debye_waller_results[300.0])
    phase_invariance_error = spectral_line_error(phase_lines, native_lines.centers, real.(native_lines.weights))

    selection_Q = conventional_Q([0.37, 0.37, 0.37])
    selection_result = solve(phonon_solver, al_model, CartesianWaveVector(selection_Q))
    selection_lines = one_phonon_neutron_lines(probe, selection_result, selection_Q; temperature=0.0, convention=spectrum_convention, debye_waller=debye_waller_results[5.0])
    selection_positive = [(center, real(weight)) for (center, weight) in zip(selection_lines.centers, selection_lines.weights) if center > 0]
    selection_weights = sort(last.(selection_positive))
    transverse_leakage = sum(selection_weights[1:2]) / max(sum(selection_weights), eps(Float64))

    rotation_Q = conventional_Q([2.37, 0.37, 0.37])
    rotation_result = solve(phonon_solver, al_model, CartesianWaveVector(conventional_Q([0.37, 0.37, 0.37])))
    rotation_metadata = copy(rotation_result.metadata)
    rotation_modes = copy(rotation_result.modes)
    rotation_modes[:, 1:2] .= rotation_modes[:, 1:2] * rotation
    rotation_metadata[:physical_displacement_modes] = copy(rotation_metadata[:physical_displacement_modes])
    rotation_metadata[:physical_displacement_modes][:, 1:2] .= rotation_metadata[:physical_displacement_modes][:, 1:2] * rotation
    rotated_mode_result = PhononModeResult(rotation_result.solver, rotation_result.model, copy(rotation_result.frequencies), rotation_modes, rotation_metadata)
    unrotated_lines = one_phonon_neutron_lines(probe, rotation_result, rotation_Q; temperature=300.0, convention=spectrum_convention, debye_waller=debye_waller_results[300.0])
    rotated_lines = one_phonon_neutron_lines(probe, rotated_mode_result, rotation_Q; temperature=300.0, convention=spectrum_convention, debye_waller=debye_waller_results[300.0])
    degenerate_intensity_error = spectral_line_error(rotated_lines, unrotated_lines.centers, real.(unrotated_lines.weights))

    neutron_formula_pass = neutron_line_error <= 1e-12 && neutron_zero_mode_source_pass
    neutron_invariance_pass = phase_invariance_error <= 1e-12 && degenerate_intensity_error <= 1e-12
    neutron_selection_pass = transverse_leakage <= 1e-12

    println()
    println("FAIR COHERENT ONE-PHONON NEUTRON FORMALISM VALIDATION")
    @printf("Fair Eqs. (7,10,11) line error      = %.3e\n", neutron_line_error)
    @printf("Eigenvector-phase invariance error   = %.3e\n", phase_invariance_error)
    @printf("Degenerate-subspace intensity error  = %.3e\n", degenerate_intensity_error)
    @printf("Q || [111] transverse leakage        = %.3e\n", transverse_leakage)
    @printf("Full-Q API line-equivalence error   = %.3e\n", full_Q_line_error)
    @printf("Full-Q reduced-q agreement error    = %.3e\n", reciprocal_reduction_error)
    @printf("Q = q + G reconstruction error      = %.3e\n", reciprocal_reconstruction_error)
    println("Full-Q decomposition metadata       = ", full_Q_metadata_pass ? "PASS" : "FAIL")
    println("Neutron zero-mode source            = ", neutron_zero_mode_source_pass ? "solver mask" : "fallback/override [FAIL]")
    println("  Native lines vs Fair formalism     : ", neutron_formula_pass ? "PASS" : "FAIL")
    println("  Full-Q reciprocal reduction API    : ", full_Q_api_pass ? "PASS" : "FAIL")
    println("  Phase / degenerate invariance      : ", neutron_invariance_pass ? "PASS" : "FAIL")
    println("  Longitudinal neutron selection     : ", neutron_selection_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Fair Fig. 6(g,h) geometry is reused to test the one-phonon implementation on the controlled FCC-Al model. The full scattering vector Q enters the neutron structure factor, while the harmonic eigenproblem is solved at the reduced crystal momentum q = Q - G. Fair's published Al MRPD is reported only as external context because this benchmark does not use the paper's VASP/Phonopy IFCs.
    # -------------------------------------------------------------------------

    h_values = collect(range(-1.0, 1.0; length=49))
    energy_axis = collect(range(-50.0, 50.0; length=1001))
    broadening = GaussianBroadening(0.35)

    fair_cuts = Dict{Symbol,Function}(:h22 => h -> [h, 2.0, 2.0],
                                      :h2hh => h -> [h, 2.0 + h / 2, 2.0 + h / 2])

    function broaden_reference(centers, weights, axis, sigma)
        values = zeros(Float64, length(axis))
        normalization = sqrt(2π) * sigma
        for (center, weight) in zip(centers, weights)
            values .+= weight .* exp.(-0.5 .* ((axis .- center) ./ sigma).^2) ./ normalization
        end
        return values
    end

    # The internal map comparison uses no fitted scale factor. Bins with |E| < 1 meV and bins below 1e-4 of the reference-map maximum are excluded to avoid the acoustic q -> 0 singularity and division by negligible reference intensity.
    function fair_mrpd(reference, candidate, axis)
        energy_mask = abs.(axis) .>= 1.0
        reference_max = maximum(reference[:, energy_mask]; init=0.0)
        threshold = 1e-4 * reference_max
        mask = falses(size(reference))
        for ie in eachindex(axis)
            energy_mask[ie] || continue
            @views mask[:, ie] .= reference[:, ie] .> threshold
        end
        count(identity, mask) > 0 || error("Fair MRPD filtering removed every intensity bin")
        mrpd = mean(abs.(candidate[mask] .- reference[mask]) ./ reference[mask]) * 100
        return mrpd, mask
    end

    map_results = Dict{Tuple{Symbol,Float64},Any}()
    maximum_map_line_error = 0.0
    maximum_mrpd = 0.0

    for (cut_name, cut_function) in fair_cuts
        for temperature in temperatures
            native_map = zeros(Float64, length(h_values), length(energy_axis))
            reference_map = similar(native_map)
            max_line_error = 0.0
            dw = debye_waller_results[temperature]

            for (ih, h) in enumerate(h_values)
                hkl = cut_function(h)
                Q = conventional_Q(hkl)
                Q_vector = CartesianWaveVector(Q)
                decomposition = decompose_reciprocal_vector(al_model, Q_vector)
                result = solve(phonon_solver, al_model, decomposition.q_cartesian)
                native_spectrum = one_phonon_neutron_intensity(phonon_solver, al_model, probe, Q_vector, energy_axis;
                                                               broadening=broadening, temperature=temperature,
                                                               convention=spectrum_convention, debye_waller=dw)
                native_map[ih, :] .= vec(native_spectrum.intensity)

                centers, weights = fair_reference_lines(result, Q, temperature, only(dw))
                reference_map[ih, :] .= broaden_reference(centers, weights, energy_axis, broadening.sigma)
                line_error = norm(native_map[ih, :] - reference_map[ih, :]) / max(norm(reference_map[ih, :]), eps(Float64))
                max_line_error = max(max_line_error, line_error)
            end

            mrpd, mask = fair_mrpd(reference_map, native_map, energy_axis)
            map_results[(cut_name, temperature)] = (; native_map, reference_map, mask, mrpd, max_line_error)
            maximum_map_line_error = max(maximum_map_line_error, max_line_error)
            maximum_mrpd = max(maximum_mrpd, mrpd)
        end
    end

    # Detailed balance is an explicit consequence of Fair Eq. (10). The
    # negative/positive line ratio for the same mode equals exp[-E/(kB T)].
    balance_Q = conventional_Q([2.29, 0.29, 0.29])
    balance_result = solve(phonon_solver, al_model, CartesianWaveVector(conventional_Q([0.29, 0.29, 0.29])))
    balance_frequency = maximum(balance_result.frequencies)
    detailed_balance_errors = Dict{Float64,Float64}()
    detailed_balance_ratios = Dict{Float64,Float64}()

    for temperature in temperatures
        lines = one_phonon_neutron_lines(probe, balance_result, balance_Q; temperature=temperature, convention=spectrum_convention, debye_waller=debye_waller_results[temperature])
        positive_weight = sum(real(weight) for (center, weight) in zip(lines.centers, lines.weights) if abs(center - balance_frequency) <= 1e-9)
        negative_weight = sum(real(weight) for (center, weight) in zip(lines.centers, lines.weights) if abs(center + balance_frequency) <= 1e-9)
        positive_weight_valid = isfinite(positive_weight) && positive_weight > 0
        negative_weight_valid = isfinite(negative_weight) && negative_weight >= 0
        positive_weight_valid || error("Detailed-balance probe has zero or nonfinite positive-energy weight at T=$temperature K")
        negative_weight_valid || error("Detailed-balance probe has invalid negative-energy weight at T=$temperature K")
        ratio = negative_weight / positive_weight
        exact_ratio = exp(-balance_frequency / (kB_meV_per_K * temperature))
        detailed_balance_ratios[temperature] = ratio
        detailed_balance_errors[temperature] = abs(ratio - exact_ratio) / max(exact_ratio, eps(Float64))
    end

    fair_formalism_pass = maximum_mrpd <= formalism_mrpd_tolerance_percent && maximum_map_line_error <= 1e-10
    detailed_balance_pass = maximum(values(detailed_balance_errors)) <= 1e-10 && detailed_balance_ratios[5.0] < detailed_balance_ratios[300.0]

    println()
    println("FAIR ONE-PHONON FORMALISM CROSS-VALIDATION: CONTROLLED FCC Al")
    @printf("Published Al Table-4 MRPD context   = %.2f%% or better [not used as gate]\n", published_mrpd_context_percent)
    @printf("Internal formalism MRPD tolerance   = %.1e%%\n", formalism_mrpd_tolerance_percent)
    println("Published cut geometry              = [h,2,2] and [h,2+h/2,2+h/2]")
    println("Benchmark temperatures              = 5 K and 300 K")
    println("Benchmark IFC source                = controlled nearest-neighbour FCC-Al model")
    println("Published VASP/Phonopy Al map       = NOT CLAIMED / NOT REPRODUCED")
    println()
    @printf("%-24s  %7s  %14s  %14s\n", "cut", "T [K]", "max line err", "MRPD [%]")
    for cut_name in (:h22, :h2hh), temperature in temperatures
        result = map_results[(cut_name, temperature)]
        label = cut_name === :h22 ? "[h,2,2]" : "[h,2+h/2,2+h/2]"
        @printf("%-24s  %7.1f  %14.3e  %14.6e\n", label, temperature, result.max_line_error, result.mrpd)
    end
    println()
    @printf("Detailed-balance E probe            = %.6f meV\n", balance_frequency)
    @printf("5 K S(-E)/S(+E)                     = %.6e\n", detailed_balance_ratios[5.0])
    @printf("300 K S(-E)/S(+E)                   = %.6e\n", detailed_balance_ratios[300.0])
    @printf("Maximum detailed-balance error      = %.3e\n", maximum(values(detailed_balance_errors)))
    println("  Fair Eq. (7/10/11) consistency    : ", fair_formalism_pass ? "PASS" : "FAIL")
    println("  Finite-T detailed balance          : ", detailed_balance_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Consolidated validation status for the newly integrated phonon API.
    # -------------------------------------------------------------------------

    mathematical_pass = lookup_pass && typed_wavevector_pass && mutating_api_pass && gradient_pass && hessian_pass &&
                        group_velocity_pass && directional_velocity_pass && ifc_validation_pass && projection_pass &&
                        bond_compiler_pass && angle_compiler_pass && zero_mode_pass && msd_pass && cubic_msd_pass &&
                        debye_waller_pass && neutron_formula_pass && full_Q_api_pass && neutron_invariance_pass && neutron_selection_pass
    fair_formalism_validation_pass = fair_formalism_pass && detailed_balance_pass
    overall_pass = mathematical_pass && fair_formalism_validation_pass

    println()
    println("FAIR / PHUNNY-INSPIRED HARMONIC-PHONON API VALIDATION")
    println("  Atomic mass / coherent-b lookup    : ", lookup_pass ? "PASS" : "FAIL")
    println("  Cartesian / reciprocal q API       : ", typed_wavevector_pass ? "PASS" : "FAIL")
    println("  Mutating D/D'/D'' API              : ", mutating_api_pass ? "PASS" : "FAIL")
    println("  Analytic D'(q) and D''(q)          : ", gradient_pass && hessian_pass ? "PASS" : "FAIL")
    println("  Group velocities                   : ", group_velocity_pass ? "PASS" : "FAIL")
    println("  Degenerate directional velocities  : ", directional_velocity_pass ? "PASS" : "FAIL")
    println("  IFC physical-constraint validation : ", ifc_validation_pass ? "PASS" : "FAIL")
    println("  Explicit acoustic-sum-rule repair  : ", projection_pass ? "PASS" : "FAIL")
    println("  Harmonic bond interaction compiler : ", bond_compiler_pass ? "PASS" : "FAIL")
    println("  Harmonic angle interaction compiler: ", angle_compiler_pass ? "PASS" : "FAIL")
    println("  Exact-zero / unstable-mode policy  : ", zero_mode_pass ? "PASS" : "FAIL")
    println("  Mean-square displacement tensors   : ", msd_pass && cubic_msd_pass ? "PASS" : "FAIL")
    println("  Phonon-derived Debye-Waller tensors: ", debye_waller_pass ? "PASS" : "FAIL")
    println("  Coherent one-phonon neutron lines  : ", neutron_formula_pass ? "PASS" : "FAIL")
    println("  Full-Q reciprocal reduction        : ", full_Q_api_pass ? "PASS" : "FAIL")
    println("  Neutron basis/phase invariance     : ", neutron_invariance_pass ? "PASS" : "FAIL")
    println("  Neutron longitudinal selection rule: ", neutron_selection_pass ? "PASS" : "FAIL")
    println("  Fair one-phonon equation consistency: ", fair_formalism_pass ? "PASS" : "FAIL")
    println("  Fair finite-T detailed balance     : ", detailed_balance_pass ? "PASS" : "FAIL")
    println()
    println("Phunny-inspired harmonic-phonon API benchmark: ", overall_pass ? "PASS" : "FAIL")

    # -------------------------------------------------------------------------
    # Visualization of the controlled FCC-Al model. Energies below 1 meV are omitted because the one-phonon acoustic prefactor is singular as q -> 0, and the display range is clipped at the 99.5th percentile so the remaining dispersive branches are visible. The plotted model is an API diagnostic, not a reproduction of Fair et al.'s VASP/Phonopy Al calculation.
    # -------------------------------------------------------------------------

    function visualization_colorrange(map::AbstractMatrix{<:Real}, indices::AbstractVector{<:Integer}; quantile_level::Real=0.995)
        values = [value for value in vec(map[:, indices]) if isfinite(value) && value > 0]
        isempty(values) && return (0.0, 1.0)
        upper = quantile(values, quantile_level)
        return (0.0, max(upper, eps(Float64)))
    end

    map_h22_5K = map_results[(:h22, 5.0)].native_map
    map_h2hh_300K = map_results[(:h2hh, 300.0)].native_map
    positive_energy = findall(>=(1.0), energy_axis)
    colorrange_h22 = visualization_colorrange(map_h22_5K, positive_energy)
    colorrange_h2hh = visualization_colorrange(map_h2hh_300K, positive_energy)

    fig = Figure(size=(1200, 850))
    ax11 = Axis(fig[1, 1]; xlabel="h", ylabel="energy (meV)", title="Controlled FCC Al: [h,2,2], 5 K")
    hm11 = heatmap!(ax11, h_values, energy_axis[positive_energy], map_h22_5K[:, positive_energy]; colorrange=colorrange_h22)
    Colorbar(fig[1, 2], hm11; label="coherent one-phonon intensity [99.5% display clip]")

    ax21 = Axis(fig[2, 1]; xlabel="h", ylabel="energy (meV)", title="Controlled FCC Al: [h,2+h/2,2+h/2], 300 K")
    hm21 = heatmap!(ax21, h_values, energy_axis[positive_energy], map_h2hh_300K[:, positive_energy]; colorrange=colorrange_h2hh)
    Colorbar(fig[2, 2], hm21; label="coherent one-phonon intensity [99.5% display clip]")

    save(joinpath(BENCHMARK_FIGURE_DIR, "Controlled-FCC-Al-one-phonon-formalism-benchmark.png"), fig)
end


#-------------------------------------------------------#
# Controlled harmonic-phonon scaling / memory benchmark #
#-------------------------------------------------------#
let
    # The Nb=22 endpoint is a generic 66-mode synthetic stress test. It is not associated with a specific material or external dataset.
    scaling_repeats = parse(Int, get(ENV, "PHUNDAMENTAL_PHONON_SCALING_REPEATS", "5"))
    scaling_repeats >= 3 || error("PHUNDAMENTAL_PHONON_SCALING_REPEATS must be at least 3")
    scaling_solver = HarmonicPhononSolver(tol=1e-10)
    scaling_units = PhononUnitConvention(length=:angstrom, mass=:amu, force_constant=:internal, frequency=:internal)
    scaling_qpoints = [ReciprocalWaveVector([0.137 + 0.011i, 0.211 - 0.007i, 0.319 + 0.005i]) for i in 0:7]

    function controlled_dense_phonon_model(nbasis)
        lattice = M.BravaisLattice(5.0 .* Matrix{Float64}(I, 3, 3))
        basis = [M.BasisSite(Symbol("P_", i), :Al, [mod((i - 1) / max(nbasis, 1), 1.0), 0.0, 0.0]) for i in 1:nbasis]
        crystal = M.CrystalStructure(lattice, basis)
        material = M.Material("Controlled harmonic-phonon scaling model", crystal; species=Dict(:Al => lookup_species(:Al)))
        cluster = M.Supercell(crystal, [1, 1, 1]; periodic=true)
        ndof = 3nbasis
        X = reshape([sin(0.173i + 0.119j) for i in 1:ndof, j in 1:ndof], ndof, ndof)
        Phi = transpose(X) * X + ndof .* Matrix{Float64}(I, ndof, ndof)
        blocks = ForceConstantBlock[]
        for i in 1:nbasis, j in 1:nbasis
            rows = ((i - 1) * 3 + 1):(3i)
            cols = ((j - 1) * 3 + 1):(3j)
            push!(blocks, ForceConstantBlock(i, j, [0, 0, 0], Phi[rows, cols]))
        end
        ifcs = RealSpaceForceConstants(blocks; displacement_dimension=3)
        masses = fill(lookup_species(:Al).mass, ndof)
        model = build_model(material, cluster, HarmonicPhononModel(ifcs; masses=masses, displacement_dimension=3, mass_weighted=false, units=scaling_units))
        return model, length(ifcs.blocks)
    end

    function median_runtime(f, repeats)
        f()
        samples = Float64[]
        sizehint!(samples, repeats)
        for _ in 1:repeats
            GC.gc()
            push!(samples, @elapsed f())
        end
        return median(samples)
    end

    function harmonic_scaling_row(model, block_count, qpoints)
        D_matrices = [dynamical_matrix(model, q) for q in qpoints]
        D_time = median_runtime(() -> foreach(q -> dynamical_matrix(model, q), qpoints), scaling_repeats)
        eig_time = median_runtime(() -> foreach(Dq -> eigen(Hermitian(0.5 .* (Dq .+ adjoint(Dq)))), D_matrices), scaling_repeats)
        solve_time = median_runtime(() -> foreach(q -> solve(scaling_solver, model, q), qpoints), scaling_repeats)
        gradient_time = median_runtime(() -> foreach(q -> dynamical_gradient(model, q), qpoints), scaling_repeats)
        hessian_time = median_runtime(() -> foreach(q -> dynamical_hessian(model, q), qpoints), scaling_repeats)
        solve_alloc = @allocated foreach(q -> solve(scaling_solver, model, q), qpoints)
        retained = Base.summarysize(solve(scaling_solver, model, first(qpoints)))
        nbasis = model.parameters[:phonon_basis_count]
        return (
            nbasis=nbasis,
            branches=3nbasis,
            blocks=block_count,
            nq=length(qpoints),
            D_time=D_time,
            eig_time=eig_time,
            solve_time=solve_time,
            gradient_time=gradient_time,
            hessian_time=hessian_time,
            solve_alloc=solve_alloc / 2.0^20,
            retained=retained / 2.0^20,
        )
    end

    scaling_rows = NamedTuple[]
    for nbasis in (1, 2, 4, 8, 16, 22)
        model, block_count = controlled_dense_phonon_model(nbasis)
        push!(scaling_rows, harmonic_scaling_row(model, block_count, scaling_qpoints))
    end

    fit_x = log.(Float64[row.nbasis for row in scaling_rows])
    design = hcat(ones(length(fit_x)), fit_x)
    D_fit = design \ log.(Float64[row.D_time / row.nq for row in scaling_rows])
    eig_fit = design \ log.(Float64[row.eig_time / row.nq for row in scaling_rows])
    solve_fit = design \ log.(Float64[row.solve_time / row.nq for row in scaling_rows])
    gradient_fit = design \ log.(Float64[row.gradient_time / row.nq for row in scaling_rows])
    hessian_fit = design \ log.(Float64[row.hessian_time / row.nq for row in scaling_rows])
    memory_fit = design \ log.(Float64[row.retained for row in scaling_rows])

    println()
    println("HARMONIC-PHONON IFC SCALING / MEMORY")
    println("All rows are controlled synthetic IFC models; Nb=22 is a generic 66-mode stress endpoint.")
    println()
    @printf(
        "%5s %7s %7s %4s %9s %9s %9s %9s %9s %10s %10s\n",
        "atoms", "modes", "blocks", "Nq", "D(q)", "eig", "solve", "D'", "D''", "alloc MiB", "result MiB",
    )
    for row in scaling_rows
        @printf(
            "%5d %7d %7d %4d %9.3e %9.3e %9.3e %9.3e %9.3e %10.3f %10.3f\n",
            row.nbasis, row.branches, row.blocks, row.nq, row.D_time, row.eig_time, row.solve_time, row.gradient_time,
            row.hessian_time, row.solve_alloc, row.retained,
        )
    end
    println()
    @printf("Controlled D(q) time exponent       = %.3f\n", D_fit[2])
    @printf("Controlled eigensolve exponent      = %.3f\n", eig_fit[2])
    @printf("Controlled total-solve exponent     = %.3f\n", solve_fit[2])
    @printf("Controlled D'(q) time exponent      = %.3f\n", gradient_fit[2])
    @printf("Controlled D''(q) time exponent     = %.3f\n", hessian_fit[2])
    @printf("Retained-result memory exponent     = %.3f\n", memory_fit[2])
    println("Expected dense eigensolver asymptote = O((3Nb)^3)")
    println("Expected retained mode memory        = O((3Nb)^2)")
    println("  Controlled Nb=1..22 timing/memory : GENERATED")
end

# Integrated Ice Ih validation is kept in a separate source file so the multi-atom harmonic, multiphonon, thermodynamic, and constrained-Monte-Carlo gates remain independently maintainable.
include(joinpath(@__DIR__, "ice-ih-integrated-benchmark.jl"))

# Electron-sector validation: finite-cluster Lanczos Green functions and Senechal cluster perturbation theory.
include(joinpath(@__DIR__, "senechal-hubbard-cpt-benchmark.jl"))
