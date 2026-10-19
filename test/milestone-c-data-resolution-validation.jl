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
