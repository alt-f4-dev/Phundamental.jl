# Exact constrained Abrikosov-fermion representation of spin-1/2.

struct AbrikosovFermion <: AbstractTransformation
    hbar::Float64
end
AbrikosovFermion(; hbar::Real=1.0) = AbrikosovFermion(Float64(hbar))

transformation_name(::AbrikosovFermion) = :abrikosov_fermion
transformation_hbar(T::AbrikosovFermion) = T.hbar
certificate(::AbrikosovFermion) = TransformationCertificate(
    true, true, GaugeRedundantEmbedding, :local;
    requires_constraints=true,
    gauge_redundant=true,
    notes=("spin-1/2 represented by two singly-occupied auxiliary fermion modes",),
)

function applicable(::AbrikosovFermion, model::ManyBodyModel)
    R = model.representation
    return R.algebra isa SpinAlgebra && R.state_space isa SpinHilbertSpace &&
           all(S -> S == 1//2 || S == 0.5, R.state_space.spins)
end

_af_up(i::Int) = 2i - 1
_af_dn(i::Int) = 2i

transform_generator(T::AbrikosovFermion, op::SpinPlus) =
    T.hbar * c(_af_up(op.site))' * c(_af_dn(op.site))
transform_generator(T::AbrikosovFermion, op::SpinMinus) =
    T.hbar * c(_af_dn(op.site))' * c(_af_up(op.site))
transform_generator(T::AbrikosovFermion, op::SpinZ) =
    (T.hbar/2) * (nf(_af_up(op.site)) - nf(_af_dn(op.site)))
transform_generator(T::AbrikosovFermion, op::SpinX) =
    (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))
transform_generator(T::AbrikosovFermion, op::SpinY) =
    (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))

function target_representation(::AbrikosovFermion, model::ManyBodyModel)
    N = model.representation.algebra.nsites
    ambient = FermionFockSpace(2N)
    constraints = Tuple((kind=:single_occupancy, site=i,
                         modes=(_af_up(i), _af_dn(i)), total=1) for i in 1:N)
    physical = PhysicalSubspace(ambient; constraints=constraints, dimension=2^N)
    gauge_generators = Tuple(
        nf(_af_up(i)) + nf(_af_dn(i)) - IdentityOperator()
        for i in 1:N
    )
    gauge = GaugeStructure(:U1, gauge_generators; description="local Abrikosov-fermion phase redundancy")
    return Representation(
        :abrikosov_fermion,
        ambient,
        FermionAlgebra(2N),
        FermionOccupationBasis(collect(1:2N));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=collect(1:2N),
    )
end
