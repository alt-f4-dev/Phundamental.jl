# Certified extraction from the spin-operator Hamiltonian, including ordered same-site products.

struct _SpinWaveInteractions
    spins::Vector{Float64}
    exchange::Dict{Tuple{Int,Int},Matrix{Float64}}
    fields::Matrix{Float64}
    anisotropy::Vector{Matrix{Float64}}
    constant::Float64
end

@inline function _spinwave_cartesian_components(code::UInt8)
    code == _OP_SPIN_X && return ComplexF64[1, 0, 0]
    code == _OP_SPIN_Y && return ComplexF64[0, 1, 0]
    code == _OP_SPIN_Z && return ComplexF64[0, 0, 1]
    code == _OP_SPIN_P && return ComplexF64[1, im, 0]
    code == _OP_SPIN_M && return ComplexF64[1, -im, 0]
    throw(ArgumentError("spin-wave expansion encountered an unsupported operator code $code"))
end

function _spinwave_extract_interactions(model::ManyBodyModel; tol::Real=1e-10)
    rep = model.representation
    rep.algebra isa SpinAlgebra || throw(ArgumentError("SpinWaveSolver requires SpinAlgebra"))
    space = rep.state_space isa BiorthogonalSpace ? rep.state_space.ambient : rep.state_space
    space isa SpinHilbertSpace || throw(ArgumentError("SpinWaveSolver requires SpinHilbertSpace"))
    spins = Float64.(space.spins)
    N = length(spins)
    hbar = get(model.parameters, :hbar, 1.0)
    hbar isa Real && isfinite(hbar) && hbar > 0 || throw(ArgumentError("model.parameters[:hbar] must be positive and real"))
    bonds = get(model.parameters, :spinwave_bonds, nothing)
    !isnothing(bonds) && any(bond -> bond.i == bond.j, bonds) && throw(ArgumentError("folded same-index periodic bonds are not single-ion anisotropy"))
    fields = zeros(ComplexF64, 3, N)
    exchange = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    onsite = [zeros(ComplexF64, 3, 3) for _ in 1:N]
    constant = 0.0 + 0.0im
    for term in _expanded_operator_terms(model.hamiltonian)
        ps = [p for p in term.primitives if p.code != _OP_IDENTITY]
        length(ps) <= 2 || throw(ArgumentError("SpinWaveSolver supports only constants, linear fields, intersite exchange, and same-site quadratic spin operators"))
        for p in ps
            1 <= p.index <= N || throw(BoundsError(spins, p.index))
        end
        if isempty(ps)
            constant += term.coefficient
        elseif length(ps) == 1
            fields[:, ps[1].index] .-= hbar * term.coefficient .* _spinwave_cartesian_components(ps[1].code)
        else
            p, q = ps
            c = hbar^2 * term.coefficient
            block = c .* (_spinwave_cartesian_components(p.code) * transpose(_spinwave_cartesian_components(q.code)))
            if p.index == q.index
                onsite[p.index] .+= block
            else
                i, j = minmax(p.index, q.index)
                J = get!(exchange, (i, j)) do
                    zeros(ComplexF64, 3, 3)
                end
                J .+= p.index == i ? block : transpose(block)
            end
        end
    end
    # The antisymmetric ordered on-site product is i/2 * C_ab * epsilon_abc * s_c,
    # since [s_a,s_b] = i * epsilon_abc * s_c. Fold it into the existing field.
    symmetric_onsite = Matrix{ComplexF64}[]
    for i in 1:N
        C = onsite[i]
        Csym = (C + transpose(C)) / 2
        Cant = (C - transpose(C)) / 2
        fields[:, i] .-= im .* ComplexF64[Cant[2, 3], Cant[3, 1], Cant[1, 2]]
        push!(symmetric_onsite, Csym)
    end
    scale = max(norm(fields), maximum((norm(J) for J in values(exchange)); init=0.0),
                maximum((norm(K) for K in symmetric_onsite); init=0.0), abs(constant), 1.0)
    all(isfinite, fields) && isfinite(constant) || throw(ArgumentError("nonfinite spin-wave coefficients"))
    abs(imag(constant)) <= tol * scale || throw(ArgumentError("non-Hermitian constant in spin Hamiltonian"))
    maximum(abs, imag.(fields)) <= tol * scale || throw(ArgumentError("non-Hermitian one-spin field coefficients"))
    for ((i, j), J) in exchange
        all(isfinite, J) && maximum(abs, imag.(J)) <= tol * scale || throw(ArgumentError("non-Hermitian tensor exchange on ($i,$j)"))
    end
    for (i, K) in enumerate(symmetric_onsite)
        all(isfinite, K) && maximum(abs, imag.(K)) <= tol * scale || throw(ArgumentError("non-Hermitian single-ion anisotropy on site $i"))
    end
    return _SpinWaveInteractions(spins, Dict{Tuple{Int,Int},Matrix{Float64}}(key => real.(J) for (key, J) in exchange),
                                 real.(fields), [real.(K) for K in symmetric_onsite], real(constant))
end
