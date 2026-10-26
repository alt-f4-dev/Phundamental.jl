#!/bin/bash
set -euo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JULIA_EXECUTABLE="${JULIA_EXECUTABLE:-julia}"
cd "$BENCHMARK_DIR"

# Resolve benchmark-only packages through benchmarks/Project.toml.
# Set BLAS threads to one by default for comparable Julia-thread timings.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

run_benchmark() {
    "$JULIA_EXECUTABLE" --project="$BENCHMARK_DIR" --threads "$JULIA_NUM_THREADS" "$BENCHMARK_DIR/$1"
}

#-----------#
# Threading #
#-----------#
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"

#----------------------------#
# Thermal Pure Quantum State #
#----------------------------#
export PHUNDAMENTAL_TPQ_LARGE="${PHUNDAMENTAL_TPQ_LARGE:-0}" # Large-N

export PHUNDAMENTAL_TPQ_SAMPLES12="${PHUNDAMENTAL_TPQ_SAMPLES12:-64}"
export PHUNDAMENTAL_TPQ_SAMPLES18="${PHUNDAMENTAL_TPQ_SAMPLES18:-8}"
export PHUNDAMENTAL_TPQ_KRYLOV12="${PHUNDAMENTAL_TPQ_KRYLOV12:-96}"
export PHUNDAMENTAL_TPQ_KRYLOV18="${PHUNDAMENTAL_TPQ_KRYLOV18:-96}"

export PHUNDAMENTAL_TPQ_LARGE_KRYLOV="${PHUNDAMENTAL_TPQ_LARGE_KRYLOV:-96}"
export PHUNDAMENTAL_TPQ_KRYLOV27="${PHUNDAMENTAL_TPQ_KRYLOV27:-96}"
export PHUNDAMENTAL_TPQ_KRYLOV30="${PHUNDAMENTAL_TPQ_KRYLOV30:-96}"

export PHUNDAMENTAL_TPQ_LARGE_MIN_KRYLOV="${PHUNDAMENTAL_TPQ_LARGE_MIN_KRYLOV:-32}"
export PHUNDAMENTAL_TPQ_MIN_KRYLOV27="${PHUNDAMENTAL_TPQ_MIN_KRYLOV27:-32}"
export PHUNDAMENTAL_TPQ_MIN_KRYLOV30="${PHUNDAMENTAL_TPQ_MIN_KRYLOV30:-32}"

export PHUNDAMENTAL_TPQ_ADAPTIVE_KRYLOV="${PHUNDAMENTAL_TPQ_ADAPTIVE_KRYLOV:-1}"
export PHUNDAMENTAL_TPQ_CONVERGENCE_TOL="${PHUNDAMENTAL_TPQ_CONVERGENCE_TOL:-1e-8}"
export PHUNDAMENTAL_TPQ_CONVERGENCE_INTERVAL="${PHUNDAMENTAL_TPQ_CONVERGENCE_INTERVAL:-8}"
export PHUNDAMENTAL_TPQ_CONVERGENCE_CONSECUTIVE="${PHUNDAMENTAL_TPQ_CONVERGENCE_CONSECUTIVE:-2}"

export PHUNDAMENTAL_TPQ_MEMORY_FRACTION="${PHUNDAMENTAL_TPQ_MEMORY_FRACTION:-0.70}"
export PHUNDAMENTAL_TPQ_RANK_LOOKUP="${PHUNDAMENTAL_TPQ_RANK_LOOKUP:-auto}"
export PHUNDAMENTAL_TPQ_THREADED_MATVEC="${PHUNDAMENTAL_TPQ_THREADED_MATVEC:-1}"
export PHUNDAMENTAL_TPQ_PROGRESS="${PHUNDAMENTAL_TPQ_PROGRESS:-1}"
export PHUNDAMENTAL_TPQ_PAIR_SECTORS="${PHUNDAMENTAL_TPQ_PAIR_SECTORS:-1}"


#-------------------#
# Boson Solver Test #
#-------------------#
export PHUNDAMENTAL_BOSON_SCALING_MAX="${PHUNDAMENTAL_BOSON_SCALING_MAX:-256}"
export PHUNDAMENTAL_BOSON_SCALING_REPEATS="${PHUNDAMENTAL_BOSON_SCALING_REPEATS:-7}"

#------------------------#
# Phonon Solver Scaling  #
#------------------------#
export PHUNDAMENTAL_PHONON_SCALING_REPEATS="${PHUNDAMENTAL_PHONON_SCALING_REPEATS:-5}"

#---------------------------------------------------------------#
# Ice Ih: Monte Carlo, Thermal Diffuse Scattering, Multi-Phonon #
#---------------------------------------------------------------#
export PHUNDAMENTAL_ICE_TDS_PARALLEL="${PHUNDAMENTAL_ICE_TDS_PARALLEL:-auto}"
export PHUNDAMENTAL_ICE_TDS_CONVERGENCE_MESHES="${PHUNDAMENTAL_ICE_TDS_CONVERGENCE_MESHES:-8,12,16,20}"
export PHUNDAMENTAL_ICE_FIG2C_DISPLAY_QUANTILE="${PHUNDAMENTAL_ICE_FIG2C_DISPLAY_QUANTILE:-0.85}"

#-----------------------------#
# Cluster Perturbation Theory #
#-----------------------------#
export PHUNDAMENTAL_CPT_KRYLOV_DIM="${PHUNDAMENTAL_CPT_KRYLOV_DIM:-200}"
export PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS="${PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_DIMS:-96,128,160,200}"
export PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_TOL="${PHUNDAMENTAL_CPT_KRYLOV_CONVERGENCE_TOL:-0.05}"
export PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS="${PHUNDAMENTAL_CPT_OMEGA_HALFWIDTHS:-6,8,10}"
export PHUNDAMENTAL_CPT_OMEGA_SUM_RULE_TOL="${PHUNDAMENTAL_CPT_OMEGA_SUM_RULE_TOL:-0.01}"
export PHUNDAMENTAL_CPT_RIDGE_COVERAGE_TOL="${PHUNDAMENTAL_CPT_RIDGE_COVERAGE_TOL:-0.50}"
export PHUNDAMENTAL_CPT_K_POINTS="${PHUNDAMENTAL_CPT_K_POINTS:-161}"
export PHUNDAMENTAL_CPT_OMEGA_POINTS="${PHUNDAMENTAL_CPT_OMEGA_POINTS:-481}"
export PHUNDAMENTAL_CPT_ETA="${PHUNDAMENTAL_CPT_ETA:-0.03}"
export PHUNDAMENTAL_CPT_PARALLEL="${PHUNDAMENTAL_CPT_PARALLEL:-true}"


#-------------------------#
# Experimental Resolution #
#-------------------------#
export PHUNDAMENTAL_QE_PROFILE=tof
export PHUNDAMENTAL_QE_THREADED=true
export PHUNDAMENTAL_QE_OUTPUT=results/experimental-resolution/threaded-scaling.csv


#----------------#
# BLAS Threading # Set to 1 thread -> Julia threading only
#----------------#
export OPENBLAS_NUM_THREADS=1


#-----------------------#
# Begin Benchmark Suite #
#-----------------------#

echo ""
echo "|-------------------------------------|"
echo "|                                     |"
echo "|         Empirical Benchmarks        |"
echo "|                                     |"
echo "|-------------------------------------|"
echo ""
run_benchmark "morris_2009_dy2ti2o7.jl"
run_benchmark "empirical-benchmark-suite.jl"
run_benchmark "rigol-2007-square-site-nlce.jl"

echo ""
echo "|-------------------------------------|"
echo "|                                     |"
echo "|             DMFT Suite              |"
echo "|                                     |"
echo "|-------------------------------------|"
echo ""
run_benchmark "dmft/ed-dmft-scaling-benchmark.jl"
run_benchmark "dmft/ed-dmft-physical-regimes-benchmark.jl"
run_benchmark "dmft/bath-discretization-benchmark.jl"
run_benchmark "dmft/caffarel-krauth-1994-benchmark.jl"
run_benchmark "diagrammatic-dmft/ladder-dga-convergence.jl"

echo ""
echo "|-------------------------------------|"
echo "|                                     |"
echo "|      Structural Search Scaling      |"
echo "|                                     |"
echo "|-------------------------------------|"
echo ""
run_benchmark "structural-search-scaling-benchmark.jl"



echo ""
echo "|-------------------------------------|"
echo "|                                     |"
echo "|   Experimental Resolution Scaling   |"
echo "|                                     |"
echo "|-------------------------------------|"
echo ""
run_benchmark "experimental-resolution-scaling.jl"
