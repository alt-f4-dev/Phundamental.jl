# Phundamental optimized solver layer



Main exact accelerated API:

```julia
solve(OptimizedLanczos(), model)
solve(OptimizedLanczos(sector=SpinSector(total_sz=0)), spin_model)
solve(OptimizedLanczos(sector=FermionSector(nup=4, ndown=4)), hubbard_model)
```

Automatic exact specialization:

```julia
solve(AutoSolver(), model)
```

`AutoSolver` never chooses an approximation. It may choose a harmonic normal
mode solver or an exact quadratic/BdG mode solver when the input Hamiltonian is
already harmonic/quadratic; otherwise it chooses packed Lanczos or the generic
fallback.

Explicit mode solvers:

```julia
solve(QuadraticFermionSolver(), quadratic_fermion_model)
solve(QuadraticBosonSolver(), quadratic_boson_model)
solve(HarmonicPhononSolver(), harmonic_phonon_model)
solve(LinearSpinWaveSolver(reference_signs=[1,-1,1,-1]), spin_model)
```

`LinearSpinWaveSolver` is an approximation. It performs a quadratic
Holstein-Primakoff expansion about a collinear ±z reference state and marks the
result metadata with `:approximation => :linear_spin_wave`.

Shared Krylov tools:

```julia
cf = continued_fraction(A, seed; krylov_dim=128)
spectrum = continued_fraction_spectrum(cf, omega; eta=0.02)
y = exp_action(A, x, -0.5beta; krylov_dim=64)
```

Validation:

```julia
validate_optimized_against_generic(model; nev=6)
```

For a symmetry-sector `SolverResult`, ordinary observables that leave the
sector are projected out. Cross-sector dynamical observables require solving
the corresponding target sector(s); result metadata records
`:sector_observables => :same_sector_only`.
