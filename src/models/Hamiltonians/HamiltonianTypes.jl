# Common Hamiltonian specification interface and construction helpers.

abstract type AbstractHamiltonianSpec end

function build_model(material::Material, cluster::Supercell, spec::AbstractHamiltonianSpec; kwargs...)
    throw(MethodError(build_model, (material, cluster, spec)))
end

model_cluster(model::ManyBodyModel) = get(model.parameters, :cluster, nothing)
model_positions(model::ManyBodyModel) = get(model.parameters, :positions, nothing)

@inline mode_index(site::Integer, internal::Integer, modes_per_site::Integer) =
    (Int(site) - 1) * Int(modes_per_site) + Int(internal)

_zero_operator() = 0 * IdentityOperator()

function _operator_sum(terms::Vector{<:AbstractOperatorExpr})
    isempty(terms) && return _zero_operator()
    length(terms) == 1 && return terms[1]
    return OperatorSum(AbstractOperatorExpr[terms...])
end

function _bond_coupling(value::Number, bond::Bond)
    return value
end
function _bond_coupling(values::AbstractDict, bond::Bond)
    return haskey(values, bond.shell) ? values[bond.shell] : 0
end
_bond_coupling(values::AbstractVector, bond::Bond) =
    bond.shell <= length(values) ? values[bond.shell] : 0
_bond_coupling(f::Function, bond::Bond) = f(bond)

function _site_value(value::Number, i::Int, site::CrystalSite)
    return value
end
_site_value(values::AbstractVector, i::Int, site::CrystalSite) = values[i]
function _site_value(values::AbstractDict, i::Int, site::CrystalSite)
    haskey(values, i) && return values[i]
    haskey(values, site.label) && return values[site.label]
    haskey(values, site.basis_index) && return values[site.basis_index]
    return 0
end
_site_value(f::Function, i::Int, site::CrystalSite) = f(i, site)

function _validate_material_cluster(material::Material, cluster::Supercell)
    mc = material.crystal
    cc = cluster.crystal
    lattice_dimension(mc.lattice) == lattice_dimension(cc.lattice) ||
        throw(ArgumentError("cluster and material crystal dimensions differ"))
    size(mc.lattice.direct) == size(cc.lattice.direct) ||
        throw(ArgumentError("cluster and material lattice shapes differ"))
    isapprox(mc.lattice.direct, cc.lattice.direct; rtol=1e-12, atol=1e-12) ||
        throw(ArgumentError("cluster lattice does not match material crystal"))
    length(mc.basis) == length(cc.basis) ||
        throw(ArgumentError("cluster basis does not match material crystal"))
    for i in eachindex(mc.basis)
        a, b = mc.basis[i], cc.basis[i]
        (a.label == b.label && a.species == b.species &&
         isapprox(a.fractional, b.fractional; rtol=1e-12, atol=1e-12)) ||
            throw(ArgumentError("cluster basis site $i does not match material crystal"))
    end
    return true
end

function _default_bonds(cluster::Supercell, cutoff::Real, max_shell)
    return neighbor_bonds(cluster; cutoff=cutoff, max_shell=max_shell)
end

function _base_parameters(material, cluster, spec, bonds=nothing)
    params = Dict{Symbol,Any}(
        :material => material,
        :cluster => cluster,
        :hamiltonian_spec => spec,
        :positions => site_positions(cluster),
    )
    !isnothing(bonds) && (params[:bonds] = bonds)
    return params
end

function _spin_values(material::Material, cluster::Supercell)
    spins = Float64[]
    sizehint!(spins, nsites(cluster))
    for i in 1:nsites(cluster)
        S = site_spin(material, cluster, i)
        isnothing(S) && throw(ArgumentError(
            "site $i ($(cluster.sites[i].label)) has no spin quantum number in Material.basis_properties"
        ))
        push!(spins, S)
    end
    return spins
end

function _spin_observables(N::Int)
    obs = Dict{Symbol,Any}()
    sizehint!(obs, 3 * N)
    for i in 1:N
        obs[Symbol("spin_x_", i)] = Sx(i)
        obs[Symbol("spin_y_", i)] = Sy(i)
        obs[Symbol("spin_z_", i)] = Sz(i)
    end
    return obs
end

function _fermion_spin_observables(N::Int, hbar::Real)
    obs = Dict{Symbol,Any}()
    sizehint!(obs, 7 * N)
    for i in 1:N
        up = mode_index(i, 1, 2)
        dn = mode_index(i, 2, 2)
        flip_ud = c(up)' * c(dn)
        flip_du = c(dn)' * c(up)
        obs[Symbol("density_", i)] = nf(up) + nf(dn)
        obs[Symbol("double_occupancy_", i)] = nf(up) * nf(dn)
        obs[Symbol("spin_x_", i)] = (hbar / 2) * (flip_ud + flip_du)
        obs[Symbol("spin_y_", i)] = (hbar / (2im)) * (flip_ud - flip_du)
        obs[Symbol("spin_z_", i)] = (hbar / 2) * (nf(up) - nf(dn))
    end
    return obs
end
