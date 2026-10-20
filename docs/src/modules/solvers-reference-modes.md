```@meta
CurrentModule = Phundamental.Solvers
```

# Solver API — Quadratic and Collective Modes

Reference documentation for quadratic-mode results, harmonic phonon solvers, dynamical matrices, mode derivatives, and linear spin-wave theory.

The [Solvers overview](solvers.md) explains applicability, physical approximation boundaries, and the relationship with the rest of the module.

## Mode solvers

```@docs
QuadraticModeResult
PhononModeResult
PhononDispersionResult
SpinWaveResult
BosonSubspaceProjection
orthogonal_boson_subspace
HarmonicPhononSolver
LinearSpinWaveSolver
mode_energies
mode_vectors
dynamical_matrix
dynamical_matrix!
dynamical_gradient!
dynamical_hessian!
acoustic_sum_rule_residual
group_velocities
directional_group_velocities
phonon_dispersion
phonon_zero_point_energy
spinwave_zero_point_correction
spinwave_ground_energy
```
