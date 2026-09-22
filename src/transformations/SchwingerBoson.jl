# Exact two-boson constrained spin representation.

struct SchwingerBoson <: AbstractTransformation
    hbar::Float64
end
SchwingerBoson(; hbar::Real=1.0) = SchwingerBoson(Float64(hbar))

transformation_name(::SchwingerBoson) = :schwinger_boson
transformation_hbar(T::SchwingerBoson) = T.hbar
certificate(::SchwingerBoson) = TransformationCertificate(
    true, true, GaugeRedundantEmbedding, :local;
    requires_constraints=true,
    gauge_redundant=true,
    notes=("two constrained bosonic modes per spin", "local U(1) redundancy"),
)

applicable(::SchwingerBoson, model::ManyBodyModel) =
    model.representation.algebra isa SpinAlgebra &&
    model.representation.state_space isa SpinHilbertSpace

_sb_a(i::Int) = 2i - 1
_sb_b(i::Int) = 2i

transform_generator(T::SchwingerBoson, op::SpinPlus) =
    T.hbar * b(_sb_a(op.site))' * b(_sb_b(op.site))
transform_generator(T::SchwingerBoson, op::SpinMinus) =
    T.hbar * b(_sb_b(op.site))' * b(_sb_a(op.site))
transform_generator(T::SchwingerBoson, op::SpinZ) =
    (T.hbar/2) * (nb(_sb_a(op.site)) - nb(_sb_b(op.site)))
transform_generator(T::SchwingerBoson, op::SpinX) =
    (1/2) * (transform_generator(T, SpinPlus(op.site)) + transform_generator(T, SpinMinus(op.site)))
transform_generator(T::SchwingerBoson, op::SpinY) =
    (1/(2im)) * (transform_generator(T, SpinPlus(op.site)) - transform_generator(T, SpinMinus(op.site)))

function target_representation(::SchwingerBoson, model::ManyBodyModel)
    spins = model.representation.state_space.spins
    N = length(spins)
    ambient = BosonFockSpace(2N)
    constraints = Tuple((kind=:schwinger_occupancy, site=i,
                         modes=(_sb_a(i), _sb_b(i)), total=2 * spins[i]) for i in 1:N)
    physical = PhysicalSubspace(
        ambient;
        constraints=constraints,
        dimension=prod(Int(round(2S+1)) for S in spins),
    )
    gauge_generators = Tuple(
        nb(_sb_a(i)) + nb(_sb_b(i)) - (2 * spins[i]) * IdentityOperator()
        for i in 1:N
    )
    gauge = GaugeStructure(:U1, gauge_generators; description="local Schwinger-boson phase redundancy")
    return Representation(
        :schwinger_boson,
        ambient,
        BosonAlgebra(2N),
        BosonOccupationBasis(collect(1:2N));
        physical_subspace=physical,
        constraints=constraints,
        gauge_structure=gauge,
        ordering=collect(1:2N),
    )
end
