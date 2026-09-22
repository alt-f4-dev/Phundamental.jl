# Deterministic validation suite for the numerical realization layer.
# These are ordinary functions rather than Test.jl tests so users can run them
# from the package itself without a test-only dependency.

function _maxabs(A)
    isempty(A) && return 0.0
    return maximum(abs, A)
end

function _simple_model(rep, H)
    return ManyBodyModel(rep, H)
end

function validate_matrix_realization(; atol::Real=1e-10, verbose::Bool=false)
    results = Dict{Symbol,Bool}()

    # 1. Spin-1/2 su(2): [Sx,Sy] = i Sz for hbar=1.
    spinrep = Representation(
        :spin_test,
        SpinHilbertSpace([1//2]),
        SpinAlgebra(1),
        SpinProductBasis([1]),
    )
    sbasis = computational_basis(spinrep)
    MX = matrix(Sx(1), sbasis; sparse=false)
    MY = matrix(Sy(1), sbasis; sparse=false)
    MZ = matrix(Sz(1), sbasis; sparse=false)
    results[:spin_su2] = _maxabs(MX * MY - MY * MX - im * MZ) <= atol

    # 2. Fermion CAR on two modes.
    frep = Representation(
        :fermion_test,
        FermionFockSpace(2),
        FermionAlgebra(2),
        FermionOccupationBasis([1, 2]),
    )
    fbasis = computational_basis(frep)
    C1 = matrix(c(1), fbasis; sparse=false)
    C1d = matrix(c(1)', fbasis; sparse=false)
    C2d = matrix(c(2)', fbasis; sparse=false)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    results[:fermion_car_same] = _maxabs(C1 * C1d + C1d * C1 - I4) <= atol
    results[:fermion_car_cross] = _maxabs(C1 * C2d + C2d * C1) <= atol

    # Fermionic projector/nilpotency identities required for canonical
    # reduction of Jordan-Wigner transformed quadratic Hamiltonians.
    falgebra = FermionAlgebra(2)

    n2 = canonicalize(falgebra, nf(1) * nf(1))
    cdn = canonicalize(falgebra, c(1)' * nf(1))
    nc = canonicalize(falgebra, nf(1) * c(1))
    ncd = canonicalize(falgebra, nf(1) * c(1)')
    cn = canonicalize(falgebra, c(1) * nf(1))

    n2_ok = n2 isa NumberOperator && n2.kind == :fermion && n2.mode == 1
    cdn_ok = cdn isa ScaledOperator && iszero(cdn.coefficient)
    nc_ok = nc isa ScaledOperator && iszero(nc.coefficient)
    ncd_ok = ncd isa FermionCreate && ncd.mode == 1
    cn_ok = cn isa FermionAnnihilate && cn.mode == 1

    results[:fermion_projector_reductions] = n2_ok && cdn_ok && nc_ok && ncd_ok && cn_ok

    # 3. Bosonic ladder and number operator on a finite numerical basis.
    brep = Representation(
        :boson_test,
        BosonFockSpace(1),
        BosonAlgebra(1),
        BosonOccupationBasis([1]);
        truncation=NumericalTruncation(5),
    )
    bbasis = computational_basis(brep)
    B = matrix(b(1), bbasis; sparse=false)
    Bd = matrix(b(1)', bbasis; sparse=false)
    N = matrix(nb(1), bbasis; sparse=false)
    results[:boson_number] = _maxabs(Bd * B - N) <= atol

    # 4. Majorana Clifford algebra {gamma_a,gamma_b}=2 delta_ab.
    mrep = Representation(
        :majorana_test,
        FermionFockSpace(1),
        MajoranaAlgebra(2),
        FermionOccupationBasis([1]),
    )
    mbasis = computational_basis(mrep)
    G1 = matrix(γ(1), mbasis; sparse=false)
    G2 = matrix(γ(2), mbasis; sparse=false)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    results[:majorana_square] = _maxabs(G1 * G1 - I2) <= atol && _maxabs(G2 * G2 - I2) <= atol
    results[:majorana_cross] = _maxabs(G1 * G2 + G2 * G1) <= atol

    # 5. Jordan-Wigner representation invariance for an open two-site XXZ bond.
    spin2 = Representation(
        :spin_jw_test,
        SpinHilbertSpace([1//2, 1//2]),
        SpinAlgebra(2),
        SpinProductBasis([1, 2]),
    )
    Hspin = (Sp(1) * Sm(2) + Sm(1) * Sp(2)) / 2 + 0.37 * Sz(1) * Sz(2)
    smodel = ManyBodyModel(spin2, Hspin)
    jwmodel = transform(JordanWigner([1, 2]), smodel).model
    Hs = matrix(Hspin, computational_basis(smodel); sparse=false)
    Hf = matrix(jwmodel.hamiltonian, computational_basis(jwmodel); sparse=false)
    results[:jordan_wigner_matrix] = _maxabs(Hs - Hf) <= 100atol

    # 6. Exact Holstein-Primakoff spectrum on the physical spin-1 sector.
    spin1 = Representation(
        :spin_hp_test,
        SpinHilbertSpace([1]),
        SpinAlgebra(1),
        SpinProductBasis([1]),
    )
    hmodel = ManyBodyModel(spin1, 0.31 * Sz(1) + 0.2 * Sx(1))
    hpmodel = transform(HolsteinPrimakoff(), hmodel).model
    Es = eigvals(Hermitian(matrix(hmodel.hamiltonian, computational_basis(hmodel); sparse=false)))
    Eh = eigvals(Hermitian(matrix(hpmodel.hamiltonian, computational_basis(hpmodel); sparse=false)))
    results[:holstein_primakoff_spectrum] = maximum(abs.(sort(Es) - sort(Eh))) <= 100atol

    # 7. Truncated displacement atom is unitary in its chosen finite numerical
    # realization because it is exponentiated from an anti-Hermitian truncated
    # generator.
    D = matrix(BosonicDisplacementFactor(1, 0.23 + 0.11im), bbasis; sparse=false)
    results[:displacement_unitarity] = _maxabs(adjoint(D) * D - Matrix{ComplexF64}(I, size(D)...)) <= 100atol

    # 8. Numerical-closure metadata must distinguish an unbounded boson space
    # from an explicitly truncated target basis.
    bare_brep = Representation(:bare_boson_test, BosonFockSpace(1), BosonAlgebra(1), BosonOccupationBasis([1]))
    bare_bmodel = ManyBodyModel(bare_brep, nb(1))
    bare_caps = numerical_capabilities(bare_bmodel)
    truncated_bmodel = with_truncation(bare_bmodel, NumericalTruncation(4))
    truncated_caps = numerical_capabilities(truncated_bmodel)
    results[:numerical_capabilities_truncation] = bare_caps.requires_truncation && !bare_caps.numerically_realizable &&
                                                  truncated_caps.numerically_realizable &&
                                                  basis_dimension(computational_basis(truncated_bmodel)) == 5

    # 9. A source-basis truncation transported through a basis-changing map is
    # provenance only. It must be explicitly reselected in the target basis.
    displaced = transform(BosonicDisplacement([0.2]), truncated_bmodel).model
    displaced_caps = numerical_capabilities(displaced)
    transported_rejected = try
        computational_basis(displaced)
        false
    catch error
        error isa ArgumentError
    end
    reselected = with_truncation(displaced, NumericalTruncation(4))
    results[:transported_truncation_reselection] = displaced_caps.transported_truncation &&
                                                    displaced_caps.requires_truncation && transported_rejected &&
                                                    numerical_capabilities(reselected).numerically_realizable

    # 10. Dyson-Maleev is numerically realizable through the biorthogonal dense
    # eigensolver, but not through Hermitian Lanczos.
    dm_model = transform(DysonMaleev(), hmodel).model
    dm_caps = numerical_capabilities(dm_model)
    dm_exact = solve(ExactDiagonalization(), dm_model)
    lanczos_rejected = try
        solve(LanczosSolver(nev=1), dm_model)
        false
    catch error
        error isa ArgumentError
    end
    results[:dyson_maleev_solver_capability] = dm_caps.biorthogonal &&
                                               (:exact_diagonalization in dm_caps.compatible_solvers) &&
                                               !(:lanczos in dm_caps.compatible_solvers) &&
                                               isbiorthogonal(dm_exact) && lanczos_rejected

    passed = all(values(results))
    if verbose
        for key in sort!(collect(keys(results)); by=string)
            println(rpad(string(key), 34), results[key] ? "PASSED" : "FAILED")
        end
        println("overall", " "^27, passed ? "PASSED" : "FAILED")
    end
    return (passed=passed, checks=results)
end
