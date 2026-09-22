# Standard spinful single-band Hubbard Hamiltonian.

struct HubbardModel{T,U,M} <: AbstractHamiltonianSpec
    t::T
    U::U
    chemical_potential::M
    neighbor_cutoff::Float64
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function HubbardModel(
    t,
    U;
    chemical_potential=0.0,
    neighbor_cutoff::Real,
    max_shell::Union{Nothing,Integer}=nothing,
    hbar::Real=1.0,
)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    return HubbardModel(
        t, U, chemical_potential, Float64(neighbor_cutoff),
        isnothing(max_shell) ? nothing : Int(max_shell), Float64(hbar),
    )
end

function build_model(
    material::Material,
    cluster::Supercell,
    spec::HubbardModel;
    bonds=nothing,
)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    N = nsites(cluster)
    nmodes = 2 * N

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 4 * length(B) + 2 * N)

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
        site = cluster.sites[i]
        up = mode_index(i, 1, 2)
        dn = mode_index(i, 2, 2)
        Ui = _site_value(spec.U, i, site)
        mui = _site_value(spec.chemical_potential, i, site)
        !iszero(Ui) && push!(terms, Ui * (nf(up) * nf(dn)))
        if !iszero(mui)
            push!(terms, (-mui) * nf(up))
            push!(terms, (-mui) * nf(dn))
        end
    end

    rep = Representation(
        :spinful_fermion,
        FermionFockSpace(nmodes),
        FermionAlgebra(nmodes),
        FermionOccupationBasis(collect(1:nmodes));
        ordering=collect(1:nmodes),
        reference_state=:fermion_vacuum,
    )

    params = _base_parameters(material, cluster, spec, B)
    params[:hbar] = spec.hbar
    params[:fermion_mode_map] = Dict(
        (i, sigma) => mode_index(i, sigma === :up ? 1 : 2, 2)
        for i in 1:N for sigma in (:up, :down)
    )

    return ManyBodyModel(
        rep,
        _operator_sum(terms);
        parameters=params,
        observables=_fermion_spin_observables(N, spec.hbar),
        provenance=[(construction=:hubbard, material=material.name, sites=N, fermion_modes=nmodes)],
    )
end
