using Test
using LinearAlgebra
using Phundamental

const MCO = Phundamental.Observables

@testset "Milestone C portable predictions, datasets, and numerical instrument response" begin
    @testset "Versioned portable numerical data and statistical comparison" begin
        q = [0.0, 0.5]
        energy = [-1.0, -0.3, 0.2, 1.0]
        values = [1.0 2.0 3.0 4.0; 0.5 1.5 2.5 3.5]
        source = MCO.ScatteringResult(q, energy, values, Dict{Symbol,Any}(:spectral_axis => :energy,
                                 :normalization => :sqrtN, :observable => :magnetic_scattering))
        p = portable_prediction(source; coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                                intensity_unit="arb/meV", normalization="per_spin", provenance=Dict("sample" => "synthetic"))
        wire = export_result(p)
        @test wire["schema"] == "phundamental.experimental"
        @test wire["version"] == 1
        @test wire["stage"] == "intrinsic"
        @test wire["intensity"]["order"] == "column_major"
        @test wire["intensity"]["shape"] == [2, 4]
        @test wire["intensity"]["real"] == collect(vec(values))
        @test wire["grid"]["coordinate_frame"] == "reciprocal"
        p2 = import_result(wire)
        @test p2.intensity == values
        @test p2.grid.spectral == energy
        @test p2.grid.positions[:, 1] == q
        @test export_result(p2)["intensity"]["real"] == wire["intensity"]["real"]
        p_with_error = portable_prediction(source; coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                                           intensity_unit="arb/meV", normalization="per_spin", uncertainty=fill(0.03, size(values)))
        @test import_result(export_result(p_with_error)).uncertainty == fill(0.03, size(values))
        @test export_result(source; coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                            intensity_unit="arb/meV", normalization="per_spin")["version"] == 1

        mask = trues(size(values))
        mask[1, 2] = false
        noisy = values .+ 0.2
        noisy[1, 2] = NaN
        errors = fill(0.5, size(values))
        errors[1, 2] = NaN
        measured = experimental_dataset(p.grid, noisy; valid=mask, uncertainty=errors, intensity_unit="arb/meV",
                                        normalization="per_spin", provenance=Dict("measurement" => "simulated"))
        report = compare_measurement(p, measured)
        @test report.nvalid == 7
        @test report.chi_squared ≈ 7 * (0.2 / 0.5)^2
        recovered = import_result(export_result(measured))
        @test recovered.valid == mask
        @test isnan(recovered.intensity[1, 2])
        @test compare_measurement(p2, recovered).chi_squared ≈ report.chi_squared

        cov = Matrix{Float64}(I, length(values), length(values)) .* 0.25
        measured_cov = experimental_dataset(p.grid, noisy; valid=mask, covariance=cov, intensity_unit="arb/meV",
                                            normalization="per_spin")
        @test compare_measurement(p, measured_cov).chi_squared ≈ report.chi_squared
        @test compare_measurement(p, import_result(export_result(measured_cov))).chi_squared ≈ report.chi_squared
        @test_throws ArgumentError experimental_dataset(p.grid, noisy; valid=mask, intensity_unit="arb/meV", normalization="per_spin")
        @test_throws ArgumentError portable_prediction(source; coordinate_frame=:cartesian, coordinate_unit="", spectral_unit="meV",
                                                        intensity_unit="arb/meV", normalization="per_spin")
        @test_throws ArgumentError import_result(merge(copy(wire), Dict("version" => 2)))
        @test_throws ArgumentError import_result(merge(copy(wire), Dict("kind" => "unsupported")))
        @test_throws ArgumentError compare_measurement(portable_prediction(source; coordinate_frame=:cartesian,
                    coordinate_unit="1/Å", spectral_unit="meV", intensity_unit="arb/meV", normalization="per_spin"), measured)

        tensor = reshape(ComplexF64.(1:24) .+ 0.3im, 2, 3, 2, 2)
        tensor_result = MCO.TensorSpectrumResult(q, [-1.0, 0.0, 1.0], tensor, Dict(:spectral_axis => :energy))
        t = portable_prediction(tensor_result; coordinate_frame=:cartesian, coordinate_unit="1/Å", spectral_unit="meV",
                                intensity_unit="arb/meV", normalization="per_spin")
        @test import_result(export_result(t)).intensity == tensor
        @test export_result(t)["intensity"]["shape"] == [2, 3, 2, 2]
        @test_throws ArgumentError compare_measurement(t, experimental_dataset(t.grid, zeros(2, 3, 2, 2);
                    uncertainty=ones(2, 3, 2, 2), intensity_unit="arb/meV", normalization="per_spin"))
    end

    @testset "Energy-dependent and tabulated Gaussian quadrature" begin
        energy = [-2.0, -1.35, -0.75, -0.2, 0.05, 0.45, 0.9, 1.6, 2.4]
        input = reshape(exp.(-energy.^2), 1, :)
        spectrum = MCO.ScatteringResult([0.0], energy, input, Dict{Symbol,Any}())
        widths = 0.22 .+ 0.035 .* energy.^2
        variable = EnergyDependentGaussianResolution(x -> 0.22 + 0.035 * x^2; nsigma=10)
        tabulated = TabulatedGaussianResolution(energy, widths; nsigma=10)
        generated = convolve_resolution(variable, spectrum)
        from_table = convolve_resolution(tabulated, spectrum)
        @test generated.intensity ≈ from_table.intensity atol=1e-12
        @test generated.metadata[:resolution_role] === :instrument_response
        reference = zeros(length(energy))
        w = [diff(energy)[1] / 2; [(energy[i+1] - energy[i-1]) / 2 for i in 2:length(energy)-1]; diff(energy)[end] / 2]
        for i in eachindex(energy), j in eachindex(energy)
            delta = energy[i] - energy[j]
            reference[i] += w[j] * exp(-delta^2 / (2 * widths[j]^2)) / (sqrt(2π) * widths[j]) * input[1, j]
        end
        @test vec(generated.intensity) ≈ reference atol=1e-11
        density = [exp(-((x-y) / widths[j])^2 / 2) / (sqrt(2π) * widths[j]) for x in energy, (j, y) in enumerate(energy)]
        from_kernel = convolve_resolution(TabulatedEnergyResolution(energy, density), spectrum)
        @test from_kernel.intensity ≈ from_table.intensity atol=1e-11
        @test_throws ArgumentError TabulatedGaussianResolution(energy, -abs.(widths))
        @test_throws ArgumentError convolve_resolution(TabulatedGaussianResolution([-0.5, 0.5], [0.3, 0.3]), spectrum)
        @test_throws ArgumentError convolve_resolution(EnergyDependentGaussianResolution(x -> -1.0), spectrum)
        @test_throws ArgumentError convolve_resolution(GaussianResolution(0.3), spectrum)

        uniform = collect(range(-3.0, 3.0; length=301))
        uniform_input = reshape(exp.(-uniform.^2), 1, :)
        uniform_spectrum = MCO.ScatteringResult([0.0], uniform, uniform_input, Dict{Symbol,Any}())
        optimized = convolve_resolution(GaussianResolution(0.3), uniform_spectrum)
        generalized = convolve_resolution(EnergyDependentGaussianResolution(x -> 0.3), uniform_spectrum)
        @test optimized.intensity[1, 60:240] ≈ generalized.intensity[1, 60:240] atol=4e-5
    end

    @testset "Covariance-coupled Q-energy kernels and detector coverage" begin
        q = [-1.0, 0.0, 1.0]
        energy = [-1.0, 0.0, 1.0]
        input = zeros(3, 3)
        input[2, 2] = 1.0
        spectrum = MCO.ScatteringResult(q, energy, input, Dict{Symbol,Any}())
        covariance = [0.5 0.2; 0.2 0.5]
        resolution = GaussianQEResolution(covariance, reshape(q, :, 1), ones(3);
                                           coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=5.0)
        convolved = convolve_resolution(resolution, spectrum)
        norm = 1 / (2π * sqrt(det(covariance)))
        for iq in eachindex(q), ix in eachindex(energy)
            z = [q[iq], energy[ix]]
            reference = norm * exp(-dot(z, covariance \ z) / 2)
            @test convolved.intensity[iq, ix] ≈ reference atol=1e-12
        end
        @test convolved.intensity[3, 3] > convolved.intensity[3, 1]
        @test_throws PosDefException GaussianQEResolution([1.0 2.0; 2.0 1.0], reshape(q, :, 1), ones(3);
                                                          coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV")
        @test_throws ArgumentError convolve_resolution(GaussianQEResolution(covariance, reshape(q .+ 1, :, 1), ones(3);
                             coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV"), spectrum)
        sparse = SparseQEResolution(3, 3, [5, 7], [5, 5], [0.7, 0.3])
        result = convolve_resolution(sparse, spectrum)
        @test result.intensity[5] == 0.7
        @test result.intensity[7] == 0.3
        @test sum(result.intensity) ≈ 1.0
        @test_throws ArgumentError SparseQEResolution(3, 3, [10], [5], [0.5])

        coverage = trues(3, 3)
        coverage[1, 1] = false
        covered = convolve_resolution(CoveredResolution(resolution, InstrumentCoverage(coverage)), spectrum)
        @test covered.metadata[:coverage_mask] == coverage
        @test covered.intensity == convolved.intensity
        exported = export_result(covered; coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                                 intensity_unit="arb/meV", normalization="per_spin", spectral_kind=:energy)
        @test !exported["valid"][1]
        @test isnothing(exported["intensity"]["real"][1])
    end
end

@testset "F5 raw-count likelihood, spatial resolution, and magnetic static response" begin
    @testset "Raw Poisson counts and stable Cash deviance" begin
        q = [0.0]
        energy = [0.0, 1.0, 2.0, 3.0]
        values = reshape([0.5, 1.1, 3.2, 2.0], 1, :)
        spectrum = MCO.ScatteringResult(q, energy, values, Dict{Symbol,Any}(:spectral_axis => :energy))
        prediction = portable_prediction(spectrum; coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                                         intensity_unit="counts", normalization="raw_counts")
        counts = reshape([0.0, 1.0, 4.0, 2.0], 1, :)
        measured = experimental_dataset(prediction.grid, counts; likelihood=:poisson,
                                        intensity_unit="counts", normalization="raw_counts")
        comparison = compare_measurement(prediction, measured)
        independent = 2 * sum(n == 0 ? μ : μ - n + n * log(n / μ) for (μ, n) in zip(values, counts))
        @test comparison.likelihood == :poisson
        @test comparison.nvalid == 4
        @test isnan(comparison.chi_squared)
        @test comparison.deviance ≈ independent atol=1e-12
        @test sum(abs2, comparison.whitened_residual) ≈ comparison.deviance atol=1e-12
        @test import_result(export_result(measured)).likelihood == :poisson
        old_record = export_result(measured)
        delete!(old_record, "likelihood")
        @test_throws ArgumentError import_result(old_record)
        @test_throws ArgumentError experimental_dataset(prediction.grid, counts; likelihood=:poisson,
                            uncertainty=ones(1, 4), intensity_unit="counts", normalization="raw_counts")
        @test_throws ArgumentError experimental_dataset(prediction.grid, counts .+ 0.5; likelihood=:poisson,
                            intensity_unit="counts", normalization="raw_counts")
        @test_throws ArgumentError experimental_dataset(prediction.grid, counts; likelihood=:poisson,
                            intensity_unit="counts/meV", normalization="per_spin")
        zero_counts = zeros(1, 4)
        zero_observed = experimental_dataset(prediction.grid, zero_counts; likelihood=:poisson,
                                             intensity_unit="counts", normalization="raw_counts")
        @test compare_measurement(prediction, zero_observed).deviance ≈ 2 * sum(values)
        zero_prediction = portable_prediction(MCO.ScatteringResult(q, energy, zeros(1, 4), Dict(:spectral_axis => :energy));
                                              coordinate_frame=:reciprocal, coordinate_unit="rlu", spectral_unit="meV",
                                              intensity_unit="counts", normalization="raw_counts")
        @test isinf(compare_measurement(zero_prediction, measured).deviance)
        mask = trues(1, 4)
        mask[2] = false
        masked = experimental_dataset(prediction.grid, counts; valid=mask, likelihood=:poisson,
                                      intensity_unit="counts", normalization="raw_counts")
        @test compare_measurement(prediction, masked).nvalid == 3
        gaussian = experimental_dataset(prediction.grid, counts; uncertainty=fill(0.2, 1, 4),
                                        intensity_unit="counts", normalization="raw_counts")
        @test compare_measurement(prediction, gaussian).chi_squared ≈ sum(((values .- counts) ./ 0.2).^2)
        @test import_result(export_result(gaussian)).likelihood == :gaussian
        older_gaussian = export_result(gaussian)
        delete!(older_gaussian, "likelihood")
        @test import_result(older_gaussian).likelihood == :gaussian
    end

    @testset "Source-resolved Q-energy covariance and conservative neighborhood search" begin
        q = [-1.0, -0.5, 0.2, 1.0, 1.7]
        x = [-0.8, -0.2, 0.3, 1.0]
        coordinates = reshape(q, :, 1)
        values = reshape([0.5 + abs(sin(2a + 3b)) for a in q, b in x], length(q), length(x))
        spectrum = MCO.ScatteringResult(q, x, values, Dict{Symbol,Any}())
        base = [0.31 0.07; 0.07 0.45]
        covariances = Array{Float64}(undef, 2, 2, length(q), length(x))
        for iq in eachindex(q), ix in eachindex(x)
            covariances[:, :, iq, ix] .= [0.18 + 0.03 * iq 0.04 * sin(x[ix]); 0.04 * sin(x[ix]) 0.20 + 0.05 * ix]
        end
        weights = [0.35, 0.4, 0.3, 0.45, 0.5]
        resolution = SpatialGaussianQEResolution(covariances, coordinates, x, weights;
                                                  coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=5)
        response = convolve_resolution(resolution, spectrum)
        reference = zeros(size(values))
        ew = MCO._energy_quadrature_weights(x)
        for jq in eachindex(q), jx in eachindex(x), iq in eachindex(q), ix in eachindex(x)
            C = covariances[:, :, jq, jx]
            z = [q[iq] - q[jq], x[ix] - x[jx]]
            rho2 = dot(z, C \ z)
            if rho2 <= 25
                reference[iq, ix] += values[jq, jx] * weights[jq] * ew[jx] * exp(-rho2 / 2) / (2π * sqrt(det(C)))
            end
        end
        @test response.intensity ≈ reference atol=2e-12 rtol=2e-12
        @test response.metadata[:resolution_role] == :instrument_response
        from_function = SpatialGaussianQEResolution((qrow, e) -> covariances[:, :, findfirst(==(qrow[1]), q), findfirst(==(e), x)],
                                                    coordinates, x, weights; coordinate_frame=:reciprocal,
                                                    momentum_unit="rlu", energy_unit="meV", nsigma=5)
        @test convolve_resolution(from_function, spectrum).intensity ≈ reference atol=2e-12
        fixed = GaussianQEResolution(base, coordinates, weights; coordinate_frame=:reciprocal,
                                     momentum_unit="rlu", energy_unit="meV", nsigma=5)
        constant = repeat(reshape(base, 2, 2, 1, 1), 1, 1, length(q), length(x))
        variable_constant = SpatialGaussianQEResolution(constant, coordinates, x, weights;
                                                        coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=5)
        @test convolve_resolution(fixed, spectrum).intensity ≈ convolve_resolution(variable_constant, spectrum).intensity atol=1e-12
        @test_throws PosDefException SpatialGaussianQEResolution(repeat(reshape([1.0 2.0; 2.0 1.0], 2, 2, 1, 1), 1, 1, length(q), length(x)),
                                                                  coordinates, x, weights; coordinate_frame=:reciprocal,
                                                                  momentum_unit="rlu", energy_unit="meV")
        @test_throws ArgumentError convolve_resolution(resolution,
                MCO.ScatteringResult(q, x .+ 0.1, values, Dict{Symbol,Any}()))
        @test_throws DimensionMismatch SpatialGaussianQEResolution(zeros(2, 2, 2, 2), coordinates, x, weights;
                                                                    coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV")
        q2 = [(0.0, 0.0), (0.7, -0.3), (1.3, 0.4)]
        q2matrix = [0.0 0.0; 0.7 -0.3; 1.3 0.4]
        x2 = [-0.5, 0.0, 0.7]
        data2 = reshape(collect(1.0:9.0) ./ 9, 3, 3)
        C2 = [0.25 0.03 0.06; 0.03 0.30 -0.04; 0.06 -0.04 0.40]
        field2 = repeat(reshape(C2, 3, 3, 1, 1), 1, 1, 3, 3)
        R2 = SpatialGaussianQEResolution(field2, q2matrix, x2, ones(3);
                                         coordinate_frame=:cartesian, momentum_unit="1/Å", energy_unit="meV", nsigma=6)
        output2 = convolve_resolution(R2, MCO.ScatteringResult(q2, x2, data2, Dict{Symbol,Any}()))
        reference2 = zeros(3, 3)
        ew2 = MCO._energy_quadrature_weights(x2)
        for jq in 1:3, jx in 1:3, iq in 1:3, ix in 1:3
            z = [q2matrix[iq, 1] - q2matrix[jq, 1], q2matrix[iq, 2] - q2matrix[jq, 2], x2[ix] - x2[jx]]
            rho2 = dot(z, C2 \ z)
            if rho2 <= 36
                reference2[iq, ix] += data2[jq, jx] * ew2[jx] * exp(-rho2 / 2) / ((2π)^1.5 * sqrt(det(C2)))
            end
        end
        @test output2.intensity ≈ reference2 atol=2e-12
        # A larger, irregularly ordered grid exercises sorted-neighborhood lookup without all-pairs storage.
        longq = [4.0 * (i - 101) / 100 for i in 1:201]
        longx = collect(range(-1.0, 1.0; length=51))
        isolated = zeros(201, 51)
        isolated[101, 26] = 1.0
        narrow = [0.05^2 0.0; 0.0 0.12^2]
        longR = GaussianQEResolution(narrow, reshape(longq, :, 1), fill(0.04, 201);
                                     coordinate_frame=:reciprocal, momentum_unit="rlu", energy_unit="meV", nsigma=4.2)
        long_output = convolve_resolution(longR, MCO.ScatteringResult(longq, longx, isolated, Dict{Symbol,Any}()))
        long_reference = [0.04 * 0.04 * exp(-0.5 * ((qi / 0.05)^2 + (wi / 0.12)^2)) /
                          (2π * 0.05 * 0.12) * (((qi / 0.05)^2 + (wi / 0.12)^2) <= 4.2^2) for qi in longq, wi in longx]
        @test long_output.intensity ≈ long_reference atol=2e-13
    end

    @testset "Exact Kubo susceptibility and field-resolved magnetization" begin
        representation = Representation(:f5_spin, SpinHilbertSpace([1//2]), SpinAlgebra(1), SpinProductBasis([1]))
        delta = 1.7
        model = ManyBodyModel(representation, delta * Sz(1); observables=Dict{Symbol,Any}(:mx => Sx(1), :mz => Sz(1)))
        temperatures = [0.5, 1.0, 2.0]
        susceptibility = MagneticSusceptibilityRequest(:mx, temperatures; direction=:x, kB=1.0,
                                                        field_unit="field", moment_unit="conjugate_moment")
        report = solver_compatibility(ExactDiagonalization(), model, susceptibility)
        @test report.compatible
        @test !solver_compatibility(LanczosSolver(), model, susceptibility).compatible
        @test !solver_compatibility(AutoSolver(), model, susceptibility).compatible
        @test request_requirements(susceptibility).response_order == 2
        @test request_requirements(susceptibility).frequency_domain == :static
        output = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=susceptibility))
        @test output.observable_result isa MagneticResponseCurve
        @test output.result === output.observable_result
        @test vec(output.result.values) ≈ [tanh(delta / (2T)) / (2delta) for T in temperatures] atol=1e-11
        @test response_kind(output.result) == :static_susceptibility
        longitudinal = MagneticSusceptibilityRequest(:mz, temperatures; direction=:z, field_unit="field", moment_unit="energy/field")
        longitudinal_output = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=longitudinal))
        @test vec(longitudinal_output.result.values) ≈ [1 / (4T * cosh(delta / (2T))^2) for T in temperatures] atol=1e-11
        fields = [-1.0, 0.0, 0.4, 2.0]
        magnetization = MagnetizationCurveRequest(:mx, fields, temperatures; direction=:x, field_unit="field", moment_unit="energy/field")
        magnetic_result = forward(ForwardProblem(model; solver=ExactDiagonalization(), observable=magnetization)).result
        for (ib, field) in enumerate(fields), (it, T) in enumerate(temperatures)
            gap = hypot(delta, field)
            @test magnetic_result.values[ib, it] ≈ field * tanh(gap / (2T)) / (2gap) atol=1e-11
        end
        @test request_requirements(magnetization).response_order == 1
        @test response_kind(magnetic_result) == :magnetization
        zero_temperature = MagnetizationCurveRequest(:mx, [1.0], [0.0]; direction=:x, field_unit="field", moment_unit="energy/field")
        @test forward(ForwardProblem(model; observable=zero_temperature)).result.values[1, 1] ≈ 1 / (2hypot(delta, 1.0))
        @test_throws ArgumentError MagneticSusceptibilityRequest(:mz, [0.0]; direction=:z, field_unit="field", moment_unit="energy/field")
        @test_throws ArgumentError MagnetizationCurveRequest(:mz, [NaN], [1.0]; direction=:z, field_unit="field", moment_unit="energy/field")
        @test_throws ArgumentError MagneticSusceptibilityRequest(:mz, [1.0]; direction=:invalid, field_unit="field", moment_unit="energy/field")
        @test !solver_compatibility(ExactDiagonalization(), model, MagneticSusceptibilityRequest(:missing, [1.0]; direction=:z, field_unit="field", moment_unit="energy/field")).compatible
        @test_throws ArgumentError forward(ForwardProblem(model; observable=MagneticSusceptibilityRequest(:missing, [1.0]; direction=:z, field_unit="field", moment_unit="energy/field")))
    end
end
