# General real 3x3 exchange tensors and real symmetric single-ion anisotropy.

"""
    TensorSpinModel(exchange; field=(0,0,0), anisotropy=nothing, shell=n, neighbor_cutoff=r, hbar=1)

Construct `sum_b s_i' J_b s_j + sum_i s_i' K_i s_i - sum_i h_i' s_i`, with `s=S/hbar`.
The exchange tensors may include symmetric anisotropy and antisymmetric DM interactions. The single-ion tensor `anisotropy` must be real symmetric and may be a common 3-by-3 matrix, one matrix per site, or a callable `(i, site)`. For off-diagonal components, `s' K s` uses the Hermitian symmetrized same-site operator product. `anisotropy=nothing` omits the on-site term.
"""
struct TensorSpinModel{J,F,K} <: AbstractHamiltonianSpec
    exchange::J
    field::F
    anisotropy::K
    neighbor_cutoff::Union{Nothing,Float64}
    max_shell::Union{Nothing,Int}
    hbar::Float64
end

function TensorSpinModel(exchange; field=(0.0, 0.0, 0.0), anisotropy=nothing, neighbor_cutoff::Union{Nothing,Real}=nothing,
                         max_shell::Union{Nothing,Integer}=nothing, shell::Union{Nothing,Integer}=nothing, hbar::Real=1.0)
    hbar > 0 || throw(ArgumentError("hbar must be positive"))
    cutoff, selected_shell = _normalize_bond_selection(neighbor_cutoff=neighbor_cutoff, max_shell=max_shell, shell=shell)
    return TensorSpinModel(exchange, field, anisotropy, cutoff, selected_shell, Float64(hbar))
end

function _tensor_spin_exchange(exchange, bond)
    raw = exchange isa AbstractMatrix ? exchange : _bond_coupling(exchange, bond)
    raw isa Number && iszero(raw) && return zeros(Float64, 3, 3)
    raw isa AbstractMatrix || throw(ArgumentError("each bond exchange must be a real 3-by-3 matrix"))
    size(raw) == (3, 3) || throw(DimensionMismatch("bond exchange tensor must be 3 by 3"))
    all(x -> x isa Real && isfinite(x), raw) || throw(ArgumentError("tensor exchange must contain finite real coefficients"))
    return Matrix{Float64}(raw)
end

function _tensor_spin_field(field, i, site, N)
    raw = if field isa Function
        field(i, site)
    elseif field isa Tuple && length(field) == 3 && all(x -> x isa Real, field)
        field
    elseif field isa AbstractVector && length(field) == N && all(x -> x isa AbstractVector || x isa Tuple, field)
        field[i]
    else
        field
    end
    length(raw) == 3 || throw(DimensionMismatch("magnetic field must have three Cartesian components"))
    all(x -> x isa Real && isfinite(x), raw) || throw(ArgumentError("field components must be finite and real"))
    return Float64.(collect(raw))
end

function _tensor_spin_anisotropy(anisotropy, i, site, N)
    raw = if isnothing(anisotropy)
        return zeros(Float64, 3, 3)
    elseif anisotropy isa Function
        anisotropy(i, site)
    elseif anisotropy isa AbstractMatrix
        anisotropy
    elseif anisotropy isa AbstractVector
        length(anisotropy) == N || throw(DimensionMismatch("single-ion anisotropy needs one tensor per spin site"))
        anisotropy[i]
    else
        throw(ArgumentError("anisotropy must be a real symmetric 3-by-3 matrix, a vector of site tensors, or a callable"))
    end
    raw isa AbstractMatrix || throw(ArgumentError("single-ion anisotropy tensors must be matrices"))
    size(raw) == (3, 3) || throw(DimensionMismatch("single-ion anisotropy tensor must be 3 by 3"))
    all(x -> x isa Real && isfinite(x), raw) || throw(ArgumentError("single-ion anisotropy must contain finite real coefficients"))
    K = Matrix{Float64}(raw)
    norm(K - transpose(K)) <= 1e-12 * max(norm(K), 1.0) || throw(ArgumentError("single-ion anisotropy must be symmetric"))
    return (K + transpose(K)) / 2
end

function build_model(material::Material, cluster::Supercell, spec::TensorSpinModel; bonds=nothing)
    _validate_material_cluster(material, cluster)
    B = isnothing(bonds) ? _default_bonds(cluster, spec.neighbor_cutoff, spec.max_shell) : collect(bonds)
    spins = _spin_values(material, cluster)
    N = length(spins)
    terms = AbstractOperatorExpr[]
    components = (Sx, Sy, Sz)
    for bond in B
        bond.i != bond.j || throw(ArgumentError("finite-cell TensorSpinModel cannot fold distinct periodic-image spins onto the same spin index"))
        J = _tensor_spin_exchange(spec.exchange, bond)
        for α in 1:3, β in 1:3
            iszero(J[α, β]) && continue
            push!(terms, (J[α, β] / spec.hbar^2) * (components[α](bond.i) * components[β](bond.j)))
        end
    end
    onsite = Matrix{Float64}[]
    for i in 1:N
        h = _tensor_spin_field(spec.field, i, cluster.sites[i], N)
        for α in 1:3
            iszero(h[α]) && continue
            push!(terms, (-h[α] / spec.hbar) * components[α](i))
        end
        K = _tensor_spin_anisotropy(spec.anisotropy, i, cluster.sites[i], N)
        push!(onsite, K)
        for α in 1:3, β in 1:3
            iszero(K[α, β]) && continue
            push!(terms, (K[α, β] / spec.hbar^2) * (components[α](i) * components[β](i)))
        end
    end
    params = _base_parameters(material, cluster, spec, B)
    params[:spinwave_bonds] = [(i=bond.i, j=bond.j, displacement=copy(bond.displacement), exchange=_tensor_spin_exchange(spec.exchange, bond)) for bond in B]
    params[:single_ion_anisotropy] = onsite
    params[:hbar] = spec.hbar
    params[:coupling_convention] = :energy_multiplying_dimensionless_spin_products
    return ManyBodyModel(_spin_representation(spins), _operator_sum(terms);
        parameters=params, observables=_spin_observables(N),
        provenance=[(construction=:tensor_spin, material=material.name, sites=N)])
end
