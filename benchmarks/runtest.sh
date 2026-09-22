#!/bin/bash
set -euo pipefail

#-----------#
# Threading #
#-----------#
export JULIA_NUM_THREADS=auto

#----------------------------#
# Thermal Pure Quantum State #
#----------------------------#
export PHUNDAMENTAL_TPQ_LARGE=0 #Large-N

export PHUNDAMENTAL_TPQ_SAMPLES12=64
export PHUNDAMENTAL_TPQ_SAMPLES18=8
export PHUNDAMENTAL_TPQ_KRYLOV12=96
export PHUNDAMENTAL_TPQ_KRYLOV18=96

export PHUNDAMENTAL_TPQ_LARGE_KRYLOV=96
export PHUNDAMENTAL_TPQ_KRYLOV27=96
export PHUNDAMENTAL_TPQ_KRYLOV30=96

export PHUNDAMENTAL_TPQ_LARGE_MIN_KRYLOV=32
export PHUNDAMENTAL_TPQ_MIN_KRYLOV27=32
export PHUNDAMENTAL_TPQ_MIN_KRYLOV30=32

export PHUNDAMENTAL_TPQ_ADAPTIVE_KRYLOV=1
export PHUNDAMENTAL_TPQ_CONVERGENCE_TOL=1e-8
export PHUNDAMENTAL_TPQ_CONVERGENCE_INTERVAL=8
export PHUNDAMENTAL_TPQ_CONVERGENCE_CONSECUTIVE=2

export PHUNDAMENTAL_TPQ_MEMORY_FRACTION=0.70
export PHUNDAMENTAL_TPQ_RANK_LOOKUP=auto
export PHUNDAMENTAL_TPQ_THREADED_MATVEC=1
export PHUNDAMENTAL_TPQ_PROGRESS=1
export PHUNDAMENTAL_TPQ_PAIR_SECTORS=1


#-------------------#
# Boson Solver Test #
#-------------------#
export PHUNDAMENTAL_BOSON_SCALING_MAX=256
export PHUNDAMENTAL_BOSON_SCALING_REPEATS=7

#------------------------#
# Phonon Solver Scaling  #
#------------------------#
export PHUNDAMENTAL_PHONON_SCALING_REPEATS=5

#---------------------------------------------------------------#
# Ice Ih: Monte Carlo, Thermal Diffuse Scattering, Multi-Phonon #
#---------------------------------------------------------------#
export PHUNDAMENTAL_ICE_TDS_PARALLEL=auto
export PHUNDAMENTAL_ICE_TDS_CONVERGENCE_MESHES=8,12,16,20
export PHUNDAMENTAL_ICE_FIG2C_DISPLAY_QUANTILE=0.85

#-----------------------------#
# Cluster Perturbation Theory #
#-----------------------------#
export PHUNDAMENTAL_CPT_KRYLOV_DIM=200
export PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS=96,128,160,200
export PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_TOL=0.05
export PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS=6,8,10
export PHUNDAMENTAL_CPT_OMEGA_SUM_RULE_TOL=0.01
export PHUNDAMENTAL_CPT_RIDGE_COVERAGE_TOL=0.50
export PHUNDAMENTAL_CPT_K_POINTS=161
export PHUNDAMENTAL_CPT_OMEGA_POINTS=481
export PHUNDAMENTAL_CPT_ETA=0.03
export PHUNDAMENTAL_CPT_PARALLEL=true

julia -t auto ~/Repos/Phundamental.jl/benchmarks/morris_2009_dy2ti2o7.jl
julia -t auto ~/Repos/Phundamental.jl/benchmarks/empirical-benchmark-suite.jl
