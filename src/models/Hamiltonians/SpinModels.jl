# Standard localized-spin Hamiltonians.

"""
    HeisenbergModel(J; shell=n, field=(0,0,0), hbar=1)
    HeisenbergModel(J; neighbor_cutoff=r, max_shell=nothing, field=(0,0,0), hbar=1)

Specify an isotropic Heisenberg model. `shell=n` selects exactly geometric neighbor shell `n` without requiring a floating-point cutoff. The existing `neighbor_cutoff`/`max_shell` form remains available for explicit range control.
"""
struct HeisenbergModel{J,F} <: AbstractHamiltonianSpec
    J::J
    field::F
    neighbor_cutoff::Union{Nothing,Float64}
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function HeisenbergModel(
    J;
    field=(0.0, 0.0, 0.0),
    neighbor_cutoff::Union{Nothing,Real}=nothing,
    max_shell::Union{Nothing,Integer}=nothing,
    shell::Union{Nothing,Integer}=nothing,
    hbar::Real=1.0,
)
    length(field) == 3 || throw(DimensionMismatch("field must contain (hx,hy,hz)"))
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cutoff, selected_shell = _normalize_bond_selection(neighbor_cutoff=neighbor_cutoff, max_shell=max_shell, shell=shell)
    return HeisenbergModel(J, tuple(field...), cutoff, selected_shell, Float64(hbar))
end

"""
    XXZModel(Jxy, Jz; shell=n, field_z=0, hbar=1)
    XXZModel(Jxy, Jz; neighbor_cutoff=r, max_shell=nothing, field_z=0, hbar=1)

Specify an XXZ model. `shell=n` selects exactly geometric neighbor shell `n`; the `neighbor_cutoff`/`max_shell` form remains available for explicit range control.
"""
struct XXZModel{JXY,JZ,F} <: AbstractHamiltonianSpec
    Jxy::JXY
    Jz::JZ
    field_z::F
    neighbor_cutoff::Union{Nothing,Float64}
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function XXZModel(
    Jxy,
    Jz;
    field_z=0.0,
    neighbor_cutoff::Union{Nothing,Real}=nothing,
    max_shell::Union{Nothing,Integer}=nothing,
    shell::Union{Nothing,Integer}=nothing,
    hbar::Real=1.0,
)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cutoff, selected_shell = _normalize_bond_selection(neighbor_cutoff=neighbor_cutoff, max_shell=max_shell, shell=shell)
    return XXZModel(Jxy, Jz, field_z, cutoff, selected_shell, Float64(hbar))
end

function _spin_representation(spins)
    N = length(spins)
    return Representation(
        :spin,
        SpinHilbertSpace(spins),
        SpinAlgebra(N),
        SpinProductBasis(collect(1:N));
        ordering=collect(1:N),
        reference_state=nothing,
    )
end

function build_model(material::Material, cluster::Supercell, spec::HeisenbergModel; bonds=nothing)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    spins = _spin_values(material, cluster)
    N = length(spins)

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 3 * length(B) + 3 * N)
    invh2 = inv(spec.hbar^2)
    invh = inv(spec.hbar)
    for bond in B
        J = _bond_coupling(spec.J, bond)
        iszero(J) && continue
        coeff = J * invh2
        push!(terms, coeff * (Sx(bond.i) * Sx(bond.j)))
        push!(terms, coeff * (Sy(bond.i) * Sy(bond.j)))
        push!(terms, coeff * (Sz(bond.i) * Sz(bond.j)))
    end
    hx, hy, hz = spec.field
    for i in 1:N
        !iszero(hx) && push!(terms, (-hx * invh) * Sx(i))
        !iszero(hy) && push!(terms, (-hy * invh) * Sy(i))
        !iszero(hz) && push!(terms, (-hz * invh) * Sz(i))
    end

    params = _base_parameters(material, cluster, spec, B)
    params[:hbar] = spec.hbar
    params[:coupling_convention] = :energy_multiplying_dimensionless_spin_products
    return ManyBodyModel(
        _spin_representation(spins),
        _operator_sum(terms);
        parameters=params,
        observables=_spin_observables(N),
        provenance=[(construction=:heisenberg, material=material.name, sites=N)],
    )
end

function build_model(material::Material, cluster::Supercell, spec::XXZModel; bonds=nothing)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    spins = _spin_values(material, cluster)
    N = length(spins)

    terms = AbstractOperatorExpr[]
    sizehint!(terms, 3 * length(B) + N)
    invh2 = inv(spec.hbar^2)
    invh = inv(spec.hbar)
    for bond in B
        Jxy = _bond_coupling(spec.Jxy, bond)
        Jz = _bond_coupling(spec.Jz, bond)
        if !iszero(Jxy)
            coeff = Jxy * invh2
            push!(terms, coeff * (Sx(bond.i) * Sx(bond.j)))
            push!(terms, coeff * (Sy(bond.i) * Sy(bond.j)))
        end
        !iszero(Jz) && push!(terms, (Jz * invh2) * (Sz(bond.i) * Sz(bond.j)))
    end
    for i in 1:N
        h = _site_value(spec.field_z, i, cluster.sites[i])
        !iszero(h) && push!(terms, (-h * invh) * Sz(i))
    end

    params = _base_parameters(material, cluster, spec, B)
    params[:hbar] = spec.hbar
    params[:coupling_convention] = :energy_multiplying_dimensionless_spin_products
    return ManyBodyModel(
        _spin_representation(spins),
        _operator_sum(terms);
        parameters=params,
        observables=_spin_observables(N),
        provenance=[(construction=:xxz, material=material.name, sites=N)],
    )
end
