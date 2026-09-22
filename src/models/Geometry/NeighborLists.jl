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
