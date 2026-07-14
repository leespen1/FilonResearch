#!/bin/bash
# Coarse "scout" pass for the post-qubit-order-fix CNOT3 recollection.  Bundled
# jobs (one Julia process pmap-ing over many configs) over coarse-to-moderate
# nsteps, in the lab and rwa frames, for both initial conditions, at GMRES
# tolerance 1e-15.  Four methods: filon, controlled_filon, hermite (this repo's
# efficient controlled-Hermite) and HermiteQGD (QuantumGateDesign's own Hermite
# solver, a baseline for hermite).
#
# Purpose: establish where each (frame, method, s) curve sits under the corrected
# dynamics so the per-curve deep tails can be sized (see submit_recollect_deep.sh
# / cnot3_summarize.jl).  The Filon-family lab ranges start coarse (2^10) to
# capture the pre-asymptotic onset.  Every (method,s,frame,nsteps) solve is cached
# by produce_or_load, so these compose with the deep pass and can be resubmitted.
#
# Dry run (print the sbatch commands without submitting):
#   DRYRUN=1 ./submit_recollect_coarse.sh
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON="--gmres-atol 1e-15 --gmres-rtol 1e-15 --nruns 1"

# Comma-separated 2^e list for e in [$1, $2].
gen() { local e out=""; for ((e=$1; e<=$2; e++)); do out+="$((2**e)),"; done; echo "${out%,}"; }

# sub JOBNAME PARTITION TIME NTASKS <collect_data args...>
sub() {
    local name=$1 part=$2 time=$3 nt=$4; shift 4
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" \
        --ntasks-per-node="$nt" --cpus-per-task=1 --mem-per-cpu=4G \
        "$SB" "$@" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# ---------------------------------------------------------------------------
# RWA frame: carriers are removed, so every method is cheap.  general-short (4h).
# Start at 2^8 to bracket the pre-asymptotic onset, run to 2^18 for the floor.
# ---------------------------------------------------------------------------
sub rc-rwa-filon general-short 3:59:00 16 \
    --frame rwa --method Filon,ControlledFilon --s 0,1,2 --nsteps "$(gen 8 18)" --init basis,uniform
sub rc-rwa-herm  general-short 3:59:00 16 \
    --frame rwa --method Hermite,HermiteQGD    --s 0,1,2 --nsteps "$(gen 8 18)" --init basis,uniform

# ---------------------------------------------------------------------------
# LAB frame, carrier-folding Filon methods: start at 2^10 (pre-asymptotic edge)
# through 2^18.  GMRES at 1e-15 in the lab is dear and controlled_filon is the
# costliest method, so they go to general-long on separate jobs.
# ---------------------------------------------------------------------------
sub rc-lab-filon general-long 11:59:00 16 \
    --frame lab --method Filon           --s 0,1,2 --nsteps "$(gen 10 18)" --init basis,uniform
sub rc-lab-cf    general-long 23:59:00 12 \
    --frame lab --method ControlledFilon --s 0,1,2 --nsteps "$(gen 10 18)" --init basis,uniform

# ---------------------------------------------------------------------------
# LAB frame, carrier-resolving Hermite methods (hermite + HermiteQGD).  s=1,2
# converge at moderate depth; s=0 is order-2 and slow, so it is bracketed deeper.
# basis is ~4x dearer than uniform (4 gate columns vs 1 state) — the deep s=0
# basis runs get their own longer-wall job.
# ---------------------------------------------------------------------------
sub rc-lab-herm-s12 general-long 23:59:00 16 \
    --frame lab --method Hermite,HermiteQGD --s 1,2 --nsteps "$(gen 14 20)" --init basis,uniform
sub rc-lab-herm-s0-uniform general-long 23:59:00 12 \
    --frame lab --method Hermite,HermiteQGD --s 0   --nsteps "$(gen 16 22)" --init uniform
sub rc-lab-herm-s0-basis   general-long 47:59:00 12 \
    --frame lab --method Hermite,HermiteQGD --s 0   --nsteps "$(gen 16 22)" --init basis

echo "# submitted (or dry-ran) all CNOT3 coarse-scout batches" >&2
