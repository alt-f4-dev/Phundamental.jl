using Test
using Phundamental

const MSE = Phundamental

struct MilestoneESolver <: MSE.AbstractSolver end
struct MilestoneERequest <: MSE.AbstractObservable end
struct MilestoneEContext <: MSE.AbstractForwardMethodContext end

# A third-party solver integrates using public hooks; no package-private dispatcher is modified.
MSE.solver_model_capability(::MilestoneESolver, ::MSE.ManyBodyModel) = MSE.CapabilityDecision(true)
MSE.solver_output_family(::MilestoneESolver, ::MSE.ManyBodyModel) = :eigensystem
MSE.solver_approximation(::MilestoneESolver) = :none
MSE.request_model_capability(::MilestoneERequest, ::MSE.ManyBodyModel) = MSE.CapabilityDecision(true)
MSE.solver_request_capability(::MilestoneESolver, ::MSE.ManyBodyModel, ::MilestoneERequest) = MSE.CapabilityDecision(true)
MSE.solver_context_capability(::MilestoneESolver, ::MSE.ManyBodyModel, ::MilestoneERequest, ::MilestoneEContext) = MSE.CapabilityDecision(true)
MSE.Solvers.solve(::MilestoneESolver, model::MSE.ManyBodyModel) = MSE.Solvers.solve(MSE.ExactDiagonalization(), model)
(::MilestoneERequest)(model::MSE.ManyBodyModel, solution) = MSE.groundenergy(solution)

@testset "Milestone E extensible capabilities and module ownership" begin
    representation = MSE.Representation(:milestone_e, MSE.SpinHilbertSpace([1//2]), MSE.SpinAlgebra(1), MSE.SpinProductBasis([1]))
    model = MSE.ManyBodyModel(representation, 0.7 * MSE.Sz(1))
    request = MilestoneERequest()
    solver = MilestoneESolver()

    @testset "External solver works through public extension hooks" begin
        @test MSE.solver_model_capability(solver, model).supported
        @test MSE.solver_output_family(solver, model) === :eigensystem
        @test MSE.solver_approximation(solver) === :none
        report = MSE.solver_compatibility(solver, model, request)
        @test report.compatible
        @test report.certified
        @test isempty(MSE.compatibility_issues(report))
        @test MSE.solver_compatibility(solver, model, request, MilestoneEContext()).compatible
        route = MSE.ForwardRoute(solver)
        policy = MSE.ForwardExecutionPolicy(include_default_exact_routes=false)
        problem = MSE.plan_forward(model, request; routes=(route,), policy=policy)
        @test MSE.planning_report(problem).selected_solver == MSE.solver_name(solver)
        @test isapprox(MSE.forward(problem).observable_result, -0.35; atol=1e-12)
    end

    @testset "Typed stage issues do not change existing reasons" begin
        unsupported = MSE.solver_compatibility(MSE.HarmonicPhononSolver(), model, request)
        @test !unsupported.compatible
        @test !isempty(unsupported.reasons)
        @test any(issue -> issue.stage === :model && issue.code === :unsupported, MSE.compatibility_issues(unsupported))
        absent = MSE.solver_compatibility(MilestoneESolver(), model, MSE.ObservableRequest(:opaque, (_, _) -> 0.0))
        @test !absent.certified
        @test any(issue -> issue.stage === :request_model && issue.code === :deferred, MSE.compatibility_issues(absent))
        @test_throws ArgumentError MSE.CapabilityDecision(true; issues=("not an issue",))
        @test_throws ArgumentError MSE.CompatibilityIssue(:invalid, :unknown, "invalid stage")
        @test_throws ArgumentError MSE.CompatibilityIssue(:model, :unsupported, "")
        legacy = MSE.SolverCompatibility(:legacy, false, true, false, true, :none, nothing, ("legacy reason",))
        @test legacy.reasons == ("legacy reason",)
        @test isempty(MSE.compatibility_issues(legacy))
    end

    @testset "Validation contracts and canonical module identity" begin
        @test MSE.Foundation === MSE.PhundamentalCore
        @test MSE.Solvers.AbstractSolver === MSE.AbstractSolver
        @test MSE.Models.Material === MSE.Material
        @test MSE.Observables.AbstractObservable === MSE.AbstractObservable
        @test MSE.Transformations.AbstractTransformation === MSE.AbstractTransformation
        @test MSE.solver_compatibility(MSE.ExactDiagonalization(), model).compatible
        @test !MSE.solver_compatibility(MSE.DMFTSolver(), model).compatible
        @test any(issue -> issue.stage === :model, MSE.compatibility_issues(MSE.solver_compatibility(MSE.DMFTSolver(), model)))
    end
end
