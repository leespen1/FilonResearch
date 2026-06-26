#!/bin/bash
# Report which (frame, method, s, nsteps) configs of the controlled-Hermite
# CNOT3 sweep are NOT yet cached in the commit-namespaced data dir, so a cleanup
# wave can target exactly the gaps (HDF5-race casualties, 4h timeouts, …).
# Prints one "frame method s nsteps" line per missing config.
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=$(julia --project -e 'using DrWatson; print(gitdescribe(projectdir()))')
DD="../../data/cnot3Convergence/$COMMIT"
INIT=basis

hermite_e=$(seq 4 22)
gmres_e=$(seq 4 16)

missing=0
for frame in rwa norwa lab; do
  for method in hermite filon controlled_filon controlled_hermite; do
    if [ "$method" = hermite ]; then exps=$hermite_e; else exps=$gmres_e; fi
    for s in 0 1 2; do
      for e in $exps; do
        n=$((2**e))
        # DrWatson savename: fields are alphabetical; match the discriminating ones.
        f=$(ls "$DD"/cnot3Convergence_method=${method}_frame=${frame}_s=${s}_*_initialCondition=${INIT}_*_nsteps=${n}.jld2 2>/dev/null | head -1 || true)
        if [ -z "$f" ]; then
          echo "$frame $method $s $n"
          missing=$((missing+1))
        fi
      done
    done
  done
done
echo "# total missing: $missing" >&2
