# Standard local electron-phonon and spin-boson hybrid models.

struct HolsteinModel{T,U,M,W,G} <: AbstractHamiltonianSpec
    t::T
    U::U
    chemical_potential::M
    omega::W
    g::G
    phonon_cutoff::Union{Int,Vector{Int}}
    neighbor_cutoff::Float64
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function HolsteinModel(
    t,
    U,
    omega,
    g;
    chemical_potential=0.0,
    phonon_cutoff,
    neighbor_cutoff::Real,
    max_shell::Union{Nothing,Integer}=nothing,
    hbar::Real=1.0,
)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cut = phonon_cutoff isa Integer ? Int(phonon_cutoff) : Int.(collect(phonon_cutoff))
    if cut isa Int
        cut >= 0 || throw(ArgumentError("phonon cutoff must be nonnegative"))
    else
        all(>=(0), cut) || throw(ArgumentError("phonon cutoffs must be nonnegative"))
    end
    return HolsteinModel(
        t, U, chemical_potential, omega, g, cut,
        Float64(neighbor_cutoff), isnothing(max_shell) ? nothing : Int(max_shell), Float64(hbar),
    )
end

struct LongitudinalSpinBosonModel{JXY,JZ,W,G,F} <: AbstractHamiltonianSpec
    Jxy::JXY
    Jz::JZ
    omega::W
    g::G
    field_z::F
    phonon_cutoff::Union{Int,Vector{Int}}
    neighbor_cutoff::Float64
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function LongitudinalSpinBosonModel(
    Jxy,
    Jz,
    omega,
    g;
    field_z=0.0,
    phonon_cutoff,
    neighbor_cutoff::Real,
    max_shell::Union{Nothing,Integer}=nothing,
    hbar::Real=1.0,
)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cut = phonon_cutoff isa Integer ? Int(phonon_cutoff) : Int.(collect(phonon_cutoff))
    if cut isa Int
        cut >= 0 || throw(ArgumentError("phonon cutoff must be nonnegative"))
    else
        all(>=(0), cut) || throw(ArgumentError("phonon cutoffs must be nonnegative"))
    end
    return LongitudinalSpinBosonModel(
        Jxy, Jz, omega, g, field_z, cut,
        Float64(neighbor_cutoff), isnothing(max_shell) ? nothing : Int(max_shell), Float64(hbar),
    )
end

function _site_series(value, cluster::Supercell)
    N = nsites(cluster)
    return [_site_value(value, i, cluster.sites[i]) for i in 1:N]
end

function _boson_truncation(cutoff, N::Int, description::String)
    cuts = cutoff isa Int ? fill(cutoff, N) : copy(cutoff)
    length(cuts) == N || throw(DimensionMismatch("boson cutoff must contain one value per site"))
    return NumericalTruncation(cuts; description=description)
end

function build_model(material::Material, cluster::Supercell, spec::HolsteinModel; bonds=nothing)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    N = nsites(cluster)
    nf_modes = 2 * N
    omega = _site_series(spec.omega, cluster)
    g = _site_series(spec.g, cluster)
    all(>(0), omega) || throw(ArgumentError("all local phonon frequencies must be positive"))

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 4 * length(B) + 8 * N)
    for bond in B
        tij = _bond_coupling(spec.t, bond)
        iszero(tij) && continue
        for sigma in 1:2
            mi = mode_index(bond.i, sigma, 2)
            mj = mode_index(bond.j, sigma, 2)
            push!(terms, (-tij) * (c(mi)' * c(mj)))
            push!(terms, (-conj(tij)) * (c(mj)' * c(mi)))
        end
    end

    for i in 1:N
        up = mode_index(i, 1, 2)
        dn = mode_index(i, 2, 2)
        Ui = _site_value(spec.U, i, cluster.sites[i])
        mui = _site_value(spec.chemical_potential, i, cluster.sites[i])
        !iszero(Ui) && push!(terms, Ui * (nf(up) * nf(dn)))
        !iszero(mui) && begin
            push!(terms, (-mui) * nf(up)); push!(terms, (-mui) * nf(dn))
        end
        push!(terms, (spec.hbar * omega[i]) * nb(i))
        density = nf(up) + nf(dn)
        !iszero(g[i]) && push!(terms, g[i] * density * (b(i) + b(i)'))
    end

    rep = Representation(
        :holstein,
        CompositeStateSpace(FermionFockSpace(nf_modes), BosonFockSpace(N)),
        CompositeAlgebra(FermionAlgebra(nf_modes), BosonAlgebra(N)),
        CompositeBasis(
            FermionOccupationBasis(collect(1:nf_modes)),
            BosonOccupationBasis(collect(1:N)),
        );
        ordering=(fermion=collect(1:nf_modes), boson=collect(1:N)),
        reference_state=:fermion_boson_vacuum,
        truncation=_boson_truncation(spec.phonon_cutoff, N, "Holstein phonon occupation cutoffs"),
    )

    obs = _fermion_spin_observables(N, spec.hbar)
    for i in 1:N
        obs[Symbol("phonon_number_", i)] = nb(i)
        obs[Symbol("phonon_displacement_", i)] = b(i) + b(i)'
    end
    params = _base_parameters(material, cluster, spec, B)
    params[:hbar] = spec.hbar
    params[:frequencies] = omega
    params[:electron_phonon_couplings] = g
    params[:mode_to_phonon] = [cld(m, 2) for m in 1:nf_modes]

    return ManyBodyModel(
        rep,
        _operator_sum(terms);
        parameters=params,
        observables=obs,
        provenance=[(construction=:holstein, material=material.name, sites=N)],
    )
end

function build_model(material::Material, cluster::Supercell, spec::LongitudinalSpinBosonModel; bonds=nothing)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    spins = _spin_values(material, cluster)
    N = length(spins)
    omega = _site_series(spec.omega, cluster)
    g = _site_series(spec.g, cluster)
    all(>(0), omega) || throw(ArgumentError("all local boson frequencies must be positive"))

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 3 * length(B) + 5 * N)
    invh2 = inv(spec.hbar^2)
    invh = inv(spec.hbar)
    for bond in B
        Jxy = _bond_coupling(spec.Jxy, bond)
        Jz = _bond_coupling(spec.Jz, bond)
        if !iszero(Jxy)
            push!(terms, (Jxy * invh2) * (Sx(bond.i) * Sx(bond.j)))
            push!(terms, (Jxy * invh2) * (Sy(bond.i) * Sy(bond.j)))
        end
        !iszero(Jz) && push!(terms, (Jz * invh2) * (Sz(bond.i) * Sz(bond.j)))
    end
    for i in 1:N
        h = _site_value(spec.field_z, i, cluster.sites[i])
        !iszero(h) && push!(terms, (-h * invh) * Sz(i))
        push!(terms, (spec.hbar * omega[i]) * nb(i))
        # Q_i = S_i^z / hbar, so g carries energy units.
        !iszero(g[i]) && push!(terms, (g[i] * invh) * Sz(i) * (b(i) + b(i)'))
    end

    rep = Representation(
        :longitudinal_spin_boson,
        CompositeStateSpace(SpinHilbertSpace(spins), BosonFockSpace(N)),
        CompositeAlgebra(SpinAlgebra(N), BosonAlgebra(N)),
        CompositeBasis(
            SpinProductBasis(collect(1:N)),
            BosonOccupationBasis(collect(1:N)),
        );
        ordering=(spin=collect(1:N), boson=collect(1:N)),
        reference_state=:spin_boson_reference,
        truncation=_boson_truncation(spec.phonon_cutoff, N, "spin-boson occupation cutoffs"),
    )

    obs = _spin_observables(N)
    for i in 1:N
        obs[Symbol("boson_number_", i)] = nb(i)
        obs[Symbol("boson_displacement_", i)] = b(i) + b(i)'
    end
    params = _base_parameters(material, cluster, spec, B)
    params[:hbar] = spec.hbar
    params[:frequencies] = omega
    params[:spin_boson_couplings] = g

    return ManyBodyModel(
        rep,
        _operator_sum(terms);
        parameters=params,
        observables=obs,
        provenance=[(construction=:longitudinal_spin_boson, material=material.name, sites=N)],
    )
end
