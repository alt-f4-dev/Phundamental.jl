# Collinear linear spin-wave expansion for standard Heisenberg/XXZ models.

_opt_bond_value(value::Number, bond) = value
function _opt_bond_value(values::AbstractDict, bond)
    return haskey(values, bond.shell) ? values[bond.shell] : 0
end
_opt_bond_value(values::AbstractVector, bond) =
    bond.shell <= length(values) ? values[bond.shell] : 0
_opt_bond_value(f::Function, bond) = f(bond)

_opt_site_value(value::Number, i::Int, site) = value
_opt_site_value(values::AbstractVector, i::Int, site) = values[i]
function _opt_site_value(values::AbstractDict, i::Int, site)
    haskey(values, i) && return values[i]
    haskey(values, site.label) && return values[site.label]
    haskey(values, site.basis_index) && return values[site.basis_index]
    return 0
end
_opt_site_value(f::Function, i::Int, site) = f(i, site)

function _spinwave_model_parameters(model::ManyBodyModel)
    rep = model.representation
    rep.algebra isa SpinAlgebra ||
        throw(ArgumentError("LinearSpinWaveSolver requires a spin representation"))
    rawspace = rep.state_space isa BiorthogonalSpace ? rep.state_space.ambient : rep.state_space
    rawspace isa SpinHilbertSpace ||
        throw(ArgumentError("LinearSpinWaveSolver requires SpinHilbertSpace"))

    spec = get(model.parameters, :hamiltonian_spec, nothing)
    bonds = get(model.parameters, :bonds, nothing)
    cluster = get(model.parameters, :cluster, nothing)
    isnothing(spec) && throw(ArgumentError("model does not retain hamiltonian_spec metadata"))
    isnothing(bonds) && throw(ArgumentError("model does not retain bond metadata"))
    isnothing(cluster) && throw(ArgumentError("model does not retain cluster metadata"))

    if hasproperty(spec, :Jxy) && hasproperty(spec, :Jz)
        Jxy = getproperty(spec, :Jxy)
        Jz = getproperty(spec, :Jz)
        field_z = getproperty(spec, :field_z)
    elseif hasproperty(spec, :J) && hasproperty(spec, :field)
        field = getproperty(spec, :field)
        (iszero(field[1]) && iszero(field[2])) || throw(ArgumentError(
            "collinear z-axis spin-wave solver does not support transverse magnetic fields"
        ))
        Jxy = getproperty(spec, :J)
        Jz = getproperty(spec, :J)
        field_z = field[3]
    else
        throw(ArgumentError(
            "LinearSpinWaveSolver currently supports standard XXZ and Heisenberg specifications"
        ))
    end

    return Float64.(rawspace.spins), bonds, cluster, Jxy, Jz, field_z
end

function _linear_spinwave_blocks(
    model::ManyBodyModel,
    signs::Vector{Int};
    tol::Real,
)
    spins, bonds, cluster, Jxy_spec, Jz_spec, field_spec =
        _spinwave_model_parameters(model)
    N = length(spins)
    length(signs) == N || throw(DimensionMismatch("reference_signs must have one value per spin"))
    all(s -> s in (-1, 1), signs) ||
        throw(ArgumentError("reference_signs must contain only ±1"))

    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    Eclass = 0.0

    for bond in bonds
        i, j = bond.i, bond.j
        Si, Sj = spins[i], spins[j]
        eta = signs[i] * signs[j]
        Jxy = ComplexF64(_opt_bond_value(Jxy_spec, bond))
        Jz = ComplexF64(_opt_bond_value(Jz_spec, bond))

        abs(imag(Jxy)) <= tol || throw(ArgumentError("spin-wave Jxy must be real for this Hermitian collinear solver"))
        abs(imag(Jz)) <= tol || throw(ArgumentError("spin-wave Jz must be real for Hermitian collinear model"))
        Jxy = real(Jxy)
        Jz = real(Jz)
        Eclass += Jz * eta * Si * Sj
        A[i, i] += -Jz * eta * Sj
        A[j, j] += -Jz * eta * Si

        transverse = Jxy * sqrt(Si * Sj)
        if eta == 1
            A[i, j] += transverse
            A[j, i] += conj(transverse)
        else
            B[i, j] += transverse
            B[j, i] += transverse
        end
    end

    for i in 1:N
        h = ComplexF64(_opt_site_value(field_spec, i, cluster.sites[i]))
        abs(imag(h)) <= tol || throw(ArgumentError("spin-wave field must be real"))
        Eclass -= real(h) * signs[i] * spins[i]
        A[i, i] += h * signs[i]
    end

    norm(A - adjoint(A)) <= tol * max(norm(A), 1.0) ||
        throw(ArgumentError("linear-spin-wave normal block is not Hermitian"))
    norm(B - transpose(B)) <= tol * max(norm(B), 1.0) ||
        throw(ArgumentError("linear-spin-wave pairing block is not symmetric"))

    return A, B, Eclass
end

function solve(solver::LinearSpinWaveSolver, model::ManyBodyModel)
    N = model.representation.algebra isa SpinAlgebra ?
        model.representation.algebra.nsites :
        throw(ArgumentError("LinearSpinWaveSolver requires SpinAlgebra"))
    signs = isnothing(solver.reference_signs) ? ones(Int, N) : solver.reference_signs
    A, B, Eclass = _linear_spinwave_blocks(model, signs; tol=solver.tol)
    omega, modes, meta = _bosonic_bdg_diagonalize(A, B; tol=solver.tol)

    metadata = Dict{Symbol,Any}(
        :solver => :linear_spin_wave,
        :backend => :linear_spin_wave,
        :reference_signs => copy(signs),
        :approximation => :linear_spin_wave,
        :approximation_certificate => ApproximationCertificate(:linear_spin_wave;
            assumptions=(:ordered_reference_state, :small_fluctuations), validity_conditions=(:quadratic_expansion_adequate,), order=2,
            notes=(:holstein_primakoff_quadratic_truncation,)),
        :expansion => :quadratic_holstein_primakoff,
        :physical_representation_preserved_in_source_model => true,
        :full_manybody_hilbert_space_constructed => false,
    )
    merge!(metadata, meta)
    return SpinWaveResult(
        solver, model, omega, modes, A, B, Eclass, metadata,
    )
end
