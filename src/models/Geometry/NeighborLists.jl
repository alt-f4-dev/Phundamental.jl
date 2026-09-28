# Reusable finite-cluster neighbor/bond construction.
#
# Periodic bonds are represented explicitly by their image displacement. This
# is important for small periodic clusters: multiple physical links can map to
# the same pair of finite-cluster site indices and must not be collapsed.

struct Bond{T<:AbstractFloat}
    i::Int
    j::Int
    displacement::Vector{T}   # r_j(image) - r_i
    distance::T
    shell::Int
end

@inline _norm2(v) = sum(abs2, v)

function _first_nonzero_positive(v::AbstractVector{<:Integer})
    for x in v
        iszero(x) && continue
        return x > 0
    end
    return false
end

function _enumerate_integer_shifts!(callback, ranges, shift, d::Int=1)
    if d > length(ranges)
        callback(shift)
        return
    end
    for value in ranges[d]
        shift[d] = value
        _enumerate_integer_shifts!(callback, ranges, shift, d + 1)
    end
end

function _pair_image_ranges(supercell::Supercell, delta_fractional, cutoff::Real, sigma_min::Real)
    D = lattice_dimension(supercell.crystal.lattice)
    any(supercell.periodic) || return [0:0 for _ in 1:D]

    # ||M n + dr0|| <= rc implies
    # ||n|| <= (rc + ||dr0||) / sigma_min(M).
    # `sigma_min` is precomputed once per neighbor-list construction.
    dr0 = supercell.crystal.lattice.direct * delta_fractional
    nmax = ceil(Int, (Float64(cutoff) + norm(dr0)) / sigma_min) + 1
    return [supercell.periodic[d] ? (-nmax:nmax) : (0:0) for d in 1:D]
end

function _assign_shells(raw::Vector{Tuple{Int,Int,Vector{T},T}}, atol::Real, rtol::Real) where {T<:AbstractFloat}
    isempty(raw) && return Bond{T}[]
    sort!(raw, by=last)

    bonds = Bond{T}[]
    sizehint!(bonds, length(raw))
    shell = 0
    center = zero(T)
    for (i, j, displacement, distance) in raw
        if shell == 0 || !isapprox(distance, center; atol=atol, rtol=rtol)
            shell += 1
            center = distance
        end
        push!(bonds, Bond{T}(i, j, displacement, distance, shell))
    end
    sort!(bonds, by=b -> (b.shell, b.i, b.j, Tuple(b.displacement)))
    return bonds
end

"""
    neighbor_bonds(supercell; cutoff, minimum=0, max_shell=nothing,
                   atol=1e-8, rtol=1e-8)

Construct all distinct unordered lattice bonds represented in a finite
supercell. Open directions use only sites explicitly present in the cluster.
Periodic directions additionally enumerate translated images within `cutoff`.

Unlike a minimum-image-only list, this preserves distinct links that collapse
to the same finite-cluster pair in a small periodic cell. This is required for
correct Hamiltonian bond multiplicities.
"""
function neighbor_bonds(
    supercell::Supercell;
    cutoff::Real,
    minimum::Real=0,
    max_shell::Union{Nothing,Integer}=nothing,
    atol::Real=1e-8,
    rtol::Real=1e-8,
)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive"))
    minimum >= 0 || throw(ArgumentError("minimum distance must be nonnegative"))
    cutoff >= minimum || throw(ArgumentError("cutoff must be >= minimum"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    !isnothing(max_shell) && max_shell < 1 && throw(ArgumentError("max_shell must be positive"))

    lattice = supercell.crystal.lattice
    T = eltype(lattice.direct)
    raw = Tuple{Int,Int,Vector{T},T}[]
    N = nsites(supercell)
    cutoff2 = T(cutoff)^2
    minimum2 = T(minimum)^2
    sizehint!(raw, max(0, 2 * N))

    sigma_min = if any(supercell.periodic)
        M = lattice.direct * Diagonal(supercell.repetitions)
        value = Base.minimum(svdvals(M))
        value > 0 || throw(ArgumentError("supercell translation matrix is singular"))
        value
    else
        1.0
    end

    # i <= j is enough because translated-image reversal would otherwise count
    # every undirected bond twice. i == j is retained only for nonzero periodic
    # translations and one of ±R is selected canonically.
    for i in 1:N
        jstart = any(supercell.periodic) ? i : i + 1
        jstart > N && continue
        for j in jstart:N
            i == j && !any(supercell.periodic) && continue
            delta = supercell.sites[j].fractional .- supercell.sites[i].fractional
            ranges = _pair_image_ranges(supercell, delta, cutoff, sigma_min)
            shift = zeros(Int, length(ranges))

            _enumerate_integer_shifts!(ranges, shift) do image_shift
                if i == j
                    all(iszero, image_shift) && return
                    # +R and -R are the same undirected self-image bond.
                    _first_nonzero_positive(image_shift) || return
                end

                image_fractional = similar(delta)
                @inbounds for d in eachindex(delta)
                    image_fractional[d] = delta[d] + image_shift[d] * supercell.repetitions[d]
                end
                dr = lattice.direct * image_fractional
                d2 = T(_norm2(dr))
                (d2 <= minimum2 || d2 > cutoff2) && return
                push!(raw, (i, j, dr, sqrt(d2)))
            end
        end
    end

    bonds = _assign_shells(raw, atol, rtol)
    if !isnothing(max_shell)
        filter!(b -> b.shell <= max_shell, bonds)
    end
    return bonds
end

bonds_in_shell(bonds::AbstractVector{<:Bond}, shell::Integer) = [bond for bond in bonds if bond.shell == shell]

function _open_cluster_cutoff(supercell::Supercell, atol::Real, rtol::Real)
    N = nsites(supercell)
    N >= 2 || throw(ArgumentError("an open cluster requires at least two sites to define a neighbor shell"))
    maxdistance = 0.0
    for i in 1:(N - 1), j in (i + 1):N
        maxdistance = max(maxdistance, norm(supercell.sites[j].cartesian - supercell.sites[i].cartesian))
    end
    maxdistance > 0 || throw(ArgumentError("open cluster contains no nonzero pair distance"))
    margin = max(Float64(atol), Float64(rtol) * maxdistance, 10eps(Float64) * max(maxdistance, 1.0))
    return maxdistance + margin
end

function _periodic_shell_cutoff(supercell::Supercell, shell::Integer, atol::Real, rtol::Real)
    lattice = supercell.crystal.lattice
    translation = lattice.direct * Diagonal(supercell.repetitions)
    sigma_min = Base.minimum(svdvals(translation))
    sigma_min > 0 || throw(ArgumentError("supercell translation matrix is singular"))
    cutoff = Float64(sigma_min)

    for _ in 1:64
        bonds = neighbor_bonds(supercell; cutoff=cutoff, atol=atol, rtol=rtol)
        if !isempty(bonds) && maximum(bond.shell for bond in bonds) >= shell
            shell_distance = maximum(bond.distance for bond in bonds if bond.shell == shell)
            margin = max(Float64(atol), Float64(rtol) * shell_distance, 10eps(Float64) * max(shell_distance, 1.0))
            return shell_distance + margin
        end
        cutoff *= 2
        isfinite(cutoff) || break
    end

    throw(ArgumentError("failed to locate neighbor shell $shell while expanding the periodic search radius"))
end

"""
    bonds_in_shell(supercell, shell; atol=1e-8, rtol=1e-8)

Discover and return one geometric neighbor shell without requiring the caller
to choose a distance cutoff. Open clusters are searched over all explicit site
pairs. Periodic clusters expand the image-search radius until the requested
shell is complete.
"""
function bonds_in_shell(supercell::Supercell, shell::Integer; atol::Real=1e-8, rtol::Real=1e-8)
    shell >= 1 || throw(ArgumentError("shell must be positive"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))

    cutoff = any(supercell.periodic) ? _periodic_shell_cutoff(supercell, shell, atol, rtol) : _open_cluster_cutoff(supercell, atol, rtol)
    bonds = neighbor_bonds(supercell; cutoff=cutoff, atol=atol, rtol=rtol)
    selected = bonds_in_shell(bonds, shell)
    isempty(selected) && throw(ArgumentError("cluster does not contain neighbor shell $shell"))
    return selected
end

"""
    bond_cluster(supercell, bond)

Materialize the two endpoints of `bond` as an explicit open two-site cluster.
For a periodic image bond, the second endpoint is placed at the translated
image position encoded by `bond.displacement`. Basis-site properties are copied
from the source crystal so the derived cluster retains local physical metadata.
"""
function bond_cluster(supercell::Supercell, bond::Bond)
    N = nsites(supercell)
    1 <= bond.i <= N || throw(BoundsError(supercell.sites, bond.i))
    1 <= bond.j <= N || throw(BoundsError(supercell.sites, bond.j))

    site_i = supercell.sites[bond.i]
    site_j = supercell.sites[bond.j]
    source_basis = supercell.crystal.basis
    properties_i = source_basis[site_i.basis_index].properties
    properties_j = source_basis[site_j.basis_index].properties

    label_i = site_i.label
    label_j = site_j.label == label_i ? Symbol(string(site_j.label), "_image") : site_j.label
    fractional_i = copy(site_i.fractional)
    cartesian_j = site_i.cartesian + bond.displacement
    fractional_j = cartesian_to_fractional(supercell.crystal.lattice, cartesian_j)

    basis = BasisSite[
        BasisSite(label_i, site_i.species, fractional_i; properties=properties_i),
        BasisSite(label_j, site_j.species, fractional_j; properties=properties_j),
    ]
    crystal = CrystalStructure(supercell.crystal.lattice, basis)
    cluster = Supercell(crystal, fill(1, lattice_dimension(crystal.lattice)); periodic=false)

    displacement = site_position(cluster, 2) - site_position(cluster, 1)
    isapprox(displacement, bond.displacement; rtol=1e-10, atol=1e-10) ||
        throw(ErrorException("materialized bond displacement does not match the source bond"))
    return cluster
end

function coordination_numbers(supercell::Supercell, bonds::AbstractVector{<:Bond})
    z = zeros(Int, nsites(supercell))
    @inbounds for bond in bonds
        if bond.i == bond.j
            # A periodic self-image link has two incident ends on the same
            # finite-cluster site.
            z[bond.i] += 2
        else
            z[bond.i] += 1
            z[bond.j] += 1
        end
    end
    return z
end
