#!/bin/bash
# Deep-tail pass for the post-qubit-order-fix CNOT3 recollection.  Sizes come from
# the scout summary (cnot3Convergence_summary.csv): per (frame, method, s) we
# extend nsteps until the per-method target error is reached (or the curve floors
# / becomes impractical).  Methods use the CURRENT CamelCase names
# (Filon, ControlledFilon, Hermite, HermiteQGD).  GMRES tol 1e-15 throughout.
#
# Packaging follows the rule "very small dt -> single (method,s,nsteps,IC) job;
# coarser -> bundle":
#   * rwa deep tails are cheap (carriers removed) -> bundled per group.
#   * lab e<=21 -> bundled; lab e>=22 (the dear fine-dt runs) -> one job each,
#     so a slow run never shares a wall with others and caches independently.
#
# Targets reached (from scout extrapolation, order p=2(s+1)):
#   lab Filon s0 ->1e-6 @e25 | s1 ->1e-8 @e21 | s2 done@e17
#   lab ControlledFilon s0 ->1e-8 @e25 | s1 ->1e-8 @e19 | s2 done@e17
#   lab Hermite/HermiteQGD s1 ->1e-8 @e25 | s2 ->1e-8 @e22 | s0 NOT reachable
#       (still pre-asymptotic at e22~0.76; ~e32 for 1e-4) -> pushed deepest-practical
#   rwa Filon s0 ->1e-8 @e23 | rwa ControlledFilon s0 ->1e-8 @e21
#   rwa Hermite/HermiteQGD s0 ->1e-6 @e23 | all other rwa already floored @e<=18
#
# Every (method,s,frame,nsteps,IC) solve is cached by produce_or_load, so a job
# that times out still keeps its completed shallower nsteps, and gaps can be
# resubmitted.  Dry run:  DRYRUN=1 ./submit_recollect_deep.sh
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON="--gmres-atol 1e-15 --gmres-rtol 1e-15 --nruns 1"

gen() { local e out=""; for ((e=$1; e<=$2; e++)); do out+="$((2**e)),"; done; echo "${out%,}"; }

# Wall time by exponent for the single deep lab jobs (per-step cost falls as dt
# shrinks; basis is ~4x uniform).  Generous so nothing times out mid-run.
wall() { case $1 in 22) echo 11:59:00;; 23) echo 23:59:00;; 24) echo 35:59:00;;
                    25) echo 71:59:00;; 26) echo 119:59:00;; *) echo 23:59:00;; esac; }

# bndl NAME PART TIME NTASKS <collect_data args...>
bndl() {
    local name=$1 part=$2 time=$3 nt=$4; shift 4
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" \
        --ntasks-per-node="$nt" --cpus-per-task=1 --mem-per-cpu=4G "$SB" "$@" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# one NAME PART TIME METHOD S E IC   (single lab config: one method, one nsteps)
one() {
    local name=$1 part=$2 time=$3 method=$4 s=$5 e=$6 ic=$7
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" \
        --ntasks-per-node=1 --cpus-per-task=4 --mem-per-cpu=4G "$SB" \
        --frame lab --method "$method" --s "$s" --nsteps "$((2**e))" --init "$ic" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# ===========================================================================
# RWA deep tails — only the order-2 s=0 curves need extending; bundled (cheap).
# ===========================================================================
bndl rd-rwa-filon-s0 general-long 11:59:00 12 \
    --frame rwa --method Filon            --s 0 --nsteps "$(gen 19 23)" --init basis,uniform
bndl rd-rwa-cf-s0    general-long 11:59:00 12 \
    --frame rwa --method ControlledFilon  --s 0 --nsteps "$(gen 19 21)" --init basis,uniform
bndl rd-rwa-herm-s0  general-long 11:59:00 12 \
    --frame rwa --method Hermite,HermiteQGD --s 0 --nsteps "$(gen 19 23)" --init basis,uniform

# ===========================================================================
# LAB e<=21 — bundled.  (filon/cf s1 reach target here; s0 starts its tail;
# hermite/HermiteQGD s1,s2 pick up e21.)
# ===========================================================================
bndl rd-lab-filon-lo general-long 23:59:00 12 \
    --frame lab --method Filon           --s 0,1 --nsteps "$(gen 19 21)" --init basis,uniform
bndl rd-lab-cf-lo    general-long 23:59:00 12 \
    --frame lab --method ControlledFilon --s 0,1 --nsteps "$(gen 19 21)" --init basis,uniform
bndl rd-lab-herm-lo  general-long 23:59:00 12 \
    --frame lab --method Hermite,HermiteQGD --s 1,2 --nsteps "$(gen 21 21)" --init basis,uniform

# ===========================================================================
# LAB e>=22 — one job per (method, s, nsteps, IC).  These are the very-small-dt
# runs the user wants isolated.
# ===========================================================================
# Filon s0 -> 1e-6 (order 2): e22..25
for e in 22 23 24 25; do for ic in basis uniform; do
    one rd-lab-filon-s0-e$e-$ic general-long "$(wall $e)" Filon 0 $e $ic
done; done

# ControlledFilon s0 -> 1e-8 (order 2, GMRES-bound): e22..25
for e in 22 23 24 25; do for ic in basis uniform; do
    one rd-lab-cf-s0-e$e-$ic general-long "$(wall $e)" ControlledFilon 0 $e $ic
done; done

# Hermite & HermiteQGD s1 -> 1e-8 (order 4): e22..25
for m in Hermite HermiteQGD; do for e in 22 23 24 25; do for ic in basis uniform; do
    one rd-lab-$m-s1-e$e-$ic general-long "$(wall $e)" $m 1 $e $ic
done; done; done

# Hermite & HermiteQGD s2 -> 1e-8 (order 6): e22
for m in Hermite HermiteQGD; do for ic in basis uniform; do
    one rd-lab-$m-s2-e22-$ic general-long "$(wall 22)" $m 2 22 $ic
done; done

# Hermite & HermiteQGD s0 (order 2, carrier-resolving): cannot reach 1e-4 at any
# practical depth — pushed to the deepest practical to show the order-2 onset.
# basis is 4x dearer, so basis stops at e24, uniform goes to e25.
for m in Hermite HermiteQGD; do
    for e in 23 24;    do one rd-lab-$m-s0-e$e-basis   general-long "$(wall $e)" $m 0 $e basis;   done
    for e in 23 24 25; do one rd-lab-$m-s0-e$e-uniform general-long "$(wall $e)" $m 0 $e uniform; done
done

echo "# submitted (or dry-ran) all CNOT3 deep-tail batches" >&2
