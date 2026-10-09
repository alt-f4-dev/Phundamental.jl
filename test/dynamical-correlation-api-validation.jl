using Test
using Phundamental

const FDC = Phundamental

@testset "Generic dynamical-correlation API" begin
    @testset "Exact-solver autocorrelation and cross-correlation" begin
        representation = FDC.Representation(:dynamical_correlation_spin, FDC.SpinHilbertSpace([1//2, 1//2]), FDC.SpinAlgebra(2), FDC.SpinProductBasis([1, 2]))
        hamiltonian = (0.71 * FDC.Sz(1) - 0.29 * FDC.Sx(2) +
                       0.43 * (FDC.Sx(1) * FDC.Sx(2) + FDC.Sy(1) * FDC.Sy(2) + FDC.Sz(1) * FDC.Sz(2)))
        observables = Dict{Symbol,Any}(:sx_1 => FDC.Sx(1), :sx_2 => FDC.Sx(2), :sz_1 => FDC.Sz(1), :sz_2 => FDC.Sz(2))
        model = FDC.ManyBodyModel(representation, hamiltonian; observables=observables)
        solution = FDC.solve(FDC.ExactDiagonalization(), model)

        positions = [0.0, 1.0]
        field_x = FDC.MomentumField([:sx_1, :sx_2], positions; normalization=:sqrtN)
        field_z = FDC.MomentumField([:sz_1, :sz_2], positions; normalization=:sqrtN)
        qgrid = [0.0, π]
        axis = collect(range(-2.5, 2.5; length=1001))
        convention = FDC.SpectrumConvention(fourier_normalization=:sqrtN, spectral_axis=:energy, hbar=1.0, kB=1.0)
        broadening = FDC.GaussianBroadening(0.025)

        generic_auto = FDC.dynamical_correlation(model, solution, field_x, qgrid, axis; convention=convention, broadening=broadening)
        legacy_auto = FDC.dynamic_structure_factor(model, solution, field_x, qgrid, axis; convention=convention, broadening=broadening)
        @test FDC.Observables.validate_observable_equivalence(generic_auto, legacy_auto; rtol=1e-12, atol=1e-12)
        @test generic_auto.metadata[:observable] == :dynamical_correlation
        @test generic_auto.metadata[:definition] == :S_AB
        @test legacy_auto.metadata[:observable] == :dynamic_structure_factor

        cross = FDC.dynamical_correlation(model, solution, field_x, field_z, qgrid, axis; convention=convention, broadening=broadening)
        manual = zeros(ComplexF64, length(qgrid), length(axis))
        for (iq, q) in enumerate(qgrid)
            A_q = FDC.Observables.momentum_operator(model, field_x, q)
            B_minus_q = FDC.Observables.momentum_operator(model, field_z, -q)
            lines = FDC.Observables.lehmann_lines(model, solution, A_q, B_minus_q; convention=convention)
            manual[iq, :] .= FDC.Observables.broaden_lines(lines, axis, broadening)
        end
        @test isapprox(cross.intensity, manual; rtol=1e-12, atol=1e-12)

        cross_request = FDC.DynamicalCorrelationRequest(field_x, field_z, qgrid, axis; convention=convention, broadening=broadening,
                                                        metadata=Dict(:channel => :xz))
        cross_forward = FDC.forward(FDC.ForwardProblem(model; solver=FDC.ExactDiagonalization(), observable=cross_request))
        @test FDC.Observables.validate_observable_equivalence(cross_forward.observable_result, cross; rtol=1e-12, atol=1e-12)
        @test cross_forward.observable_result.metadata[:observable_request] == :dynamical_correlation
        @test cross_forward.observable_result.metadata[:request_metadata][:channel] == :xz

        transformed_forward = FDC.forward(FDC.ForwardProblem(model; transformation=FDC.JordanWigner(1:2), solver=FDC.ExactDiagonalization(), observable=cross_request))
        @test FDC.Observables.validate_observable_equivalence(transformed_forward.observable_result, cross; rtol=1e-10, atol=1e-12)

        auto_request = FDC.DynamicalCorrelationRequest(field_x, qgrid, axis; convention=convention, broadening=broadening)
        auto_forward = FDC.forward(FDC.ForwardProblem(model; solver=FDC.ExactDiagonalization(), observable=auto_request))
        @test FDC.Observables.validate_observable_equivalence(auto_forward.observable_result, generic_auto; rtol=1e-12, atol=1e-12)
        convenience_forward = FDC.forward(model; solver=FDC.ExactDiagonalization(), observable=auto_request)
        @test FDC.Observables.validate_observable_equivalence(convenience_forward.observable_result, generic_auto; rtol=1e-12, atol=1e-12)

        incompatible = FDC.MomentumField([:sz_1, :sz_2], positions; normalization=:none)
        @test_throws ArgumentError FDC.DynamicalCorrelationRequest(field_x, incompatible, qgrid, axis; convention=convention)
        @test_throws ArgumentError FDC.dynamical_correlation(model, solution, field_x, incompatible, qgrid, axis; convention=convention)
    end

    @testset "Quadratic-mode cross-correlation" begin
        representation = FDC.Representation(:dynamical_correlation_boson, FDC.BosonFockSpace(1), FDC.BosonAlgebra(1), FDC.BosonOccupationBasis([1]);
                                            ordering=[1], reference_state=:boson_vacuum)
        hamiltonian = 1.4 * FDC.nb(1) + 0.2 * FDC.b(1)' * FDC.b(1)' + 0.2 * FDC.b(1) * FDC.b(1)
        model = FDC.ManyBodyModel(representation, hamiltonian)
        solver = FDC.QuadraticBosonSolver(tol=1e-12)
        solution = FDC.solve(solver, model)

        field_x = FDC.MomentumField([FDC.b(1) + FDC.b(1)'], [0.0]; normalization=:none)
        field_p = FDC.MomentumField([FDC.b(1)' - FDC.b(1)], [0.0]; normalization=:none)
        qgrid = [0.0]
        axis = collect(range(-2.0, 2.0; length=1601))
        convention = FDC.SpectrumConvention(fourier_normalization=:none, spectral_axis=:energy, hbar=1.0, kB=1.0)
        broadening = FDC.GaussianBroadening(0.02)

        generic_auto = FDC.dynamical_correlation(model, solution, field_x, qgrid, axis; convention=convention, broadening=broadening)
        legacy_auto = FDC.dynamic_structure_factor(model, solution, field_x, qgrid, axis; convention=convention, broadening=broadening)
        @test FDC.Observables.validate_observable_equivalence(generic_auto, legacy_auto; rtol=1e-12, atol=1e-12)

        cross = FDC.dynamical_correlation(model, solution, field_x, field_p, qgrid, axis; convention=convention, broadening=broadening)
        A = FDC.Observables.momentum_operator(model, field_x, 0.0)
        B = FDC.Observables.momentum_operator(model, field_p, -0.0)
        lines = FDC.Observables.lehmann_lines(model, solution, A, B; convention=convention)
        manual = FDC.Observables.broaden_lines(lines, axis, broadening)
        @test isapprox(vec(cross.intensity[1, :]), manual; rtol=1e-12, atol=1e-12)

        request = FDC.DynamicalCorrelationRequest(field_x, field_p, qgrid, axis; convention=convention, broadening=broadening)
        forward_result = FDC.forward(FDC.ForwardProblem(model; solver=solver, observable=request))
        @test FDC.Observables.validate_observable_equivalence(forward_result.observable_result, cross; rtol=1e-12, atol=1e-12)
        @test forward_result.observable_result.metadata[:mode_native]
    end
end
