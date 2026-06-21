#!/bin/bash
# Wave-1 SLURM submission for the CNOT3 convergence campaign with the new
# controlled-Hermite method, across all three frames (rwa, norwa, lab).
#
# Data is namespaced by commit (commit_datadir), so this must run from a CLEAN
# checkout of the commit that adds :controlled_hermite; every (method, frame, s,
# nsteps) solve is cached individually by produce_or_load, so jobs compose and
# TIMEOUTed configs simply stay uncached for a later resubmission.
#
# All jobs go through the node-local-depot script (cnot3_convergence_localdepot.sb):
# the dev node has already warmed ~/.julia, so the in-job precompile is mostly a
# cache-validation pass, while writes land in node-local scratch — immune to the
# shared-depot precompile churn that wrecked earlier waves.
#
# Batching by cost class:
#   * Hermite (explicit, cheap): one multi-worker bulk job over 2^4..2^16, plus
#     one job per s for the deep tail 2^17..2^22.
#   * GMRES family (filon / controlled_filon / controlled_hermite, ~15x Hermite
#     per step): one multi-worker job for 2^4..2^13, then one job per mid
#     exponent (2^14, 2^15, 2^16).  The lab frame's blowup-edge runs (s=2 around
#     2^14..2^15) may exceed the 4h wall; they cache what finishes and are
#     resubmitted as singles afterward.
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON=(-p general-short --time=3:59:00)
GMRES=filon,controlled_filon,controlled_hermite

submit () {   # submit <name> <ntasks> <cpus-per-task> -- <script args...>
    local name=$1 ntasks=$2 cpt=$3; shift 3
    sbatch "${COMMON[@]}" -J "$name" --ntasks-per-node="$ntasks" \
        --cpus-per-task="$cpt" --mem-per-cpu=4G "$SB" "$@"
}

for frame in rwa norwa lab; do
    # --- Hermite (QGD), explicit and cheap ---------------------------------
    submit c3h-$frame-h-bulk 12 1 --frame $frame --method hermite --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536
    for s in 0 1 2; do
        submit c3h-$frame-h-deep-s$s 6 1 --frame $frame --method hermite --s $s \
            --nsteps 131072,262144,524288,1048576,2097152,4194304
    done

    # --- GMRES family: filon / controlled_filon / controlled_hermite -------
    submit c3h-$frame-g-light 12 1 --frame $frame --method $GMRES --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192
    submit c3h-$frame-g-e14 9 1 --frame $frame --method $GMRES --s 0,1,2 \
        --nsteps 16384
    submit c3h-$frame-g-e15 9 1 --frame $frame --method $GMRES --s 0,1,2 \
        --nsteps 32768
    submit c3h-$frame-g-e16 9 1 --frame $frame --method $GMRES --s 0,1,2 \
        --nsteps 65536
done
