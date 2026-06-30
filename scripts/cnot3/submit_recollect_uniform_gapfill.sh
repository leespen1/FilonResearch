#!/bin/bash
# Uniform-IC gap-fill for the post-qubit-order-fix CNOT3 recollection.  The deep
# tails (submit_recollect_deep.sh) were sized from the BASIS analysis; the uniform
# state populates the fast high-energy storage/guard levels, so the Hermite family
# (and a couple order-2 ControlledFilon s0 points) carry a ~1000x larger error
# constant and need a few more nsteps to reach the same target.  Filon folds the
# carriers, so it is unaffected and already at target for uniform.
#
# Sizes from the uniform trajectories (order p=2(s+1), err~C*dt^p):
#   rwa Hermite/HermiteQGD s0 ->1e-6 @e26 | s1 ->1e-8 @e20
#   rwa ControlledFilon      s0 ->1e-8 @e23
#   lab Hermite/HermiteQGD s1 ->1e-8 @e27 | s2 ->1e-8 @e23
#   lab ControlledFilon      s0 ->1e-8 @e26
# (lab Hermite/HermiteQGD s0 stays unreachable — order-2 carrier-resolving.)
# uniform is 1 state column (cheap), so even the deep lab points are affordable.
#
# Dry run:  DRYRUN=1 ./submit_recollect_uniform_gapfill.sh
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON="--gmres-atol 1e-15 --gmres-rtol 1e-15 --nruns 1 --init uniform"
gen() { local e out=""; for ((e=$1; e<=$2; e++)); do out+="$((2**e)),"; done; echo "${out%,}"; }

bndl() { local name=$1 part=$2 time=$3 nt=$4; shift 4
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" --ntasks-per-node="$nt" \
        --cpus-per-task=1 --mem-per-cpu=4G "$SB" "$@" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}
one() { local name=$1 part=$2 time=$3 method=$4 s=$5 e=$6
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" --ntasks-per-node=1 \
        --cpus-per-task=4 --mem-per-cpu=4G "$SB" --frame lab --method "$method" \
        --s "$s" --nsteps "$((2**e))" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# --- RWA uniform tails (cheap → bundled) ---
bndl gf-rwa-herm-s0 general-long 11:59:00 8 --frame rwa --method Hermite,HermiteQGD --s 0 --nsteps "$(gen 24 26)"
bndl gf-rwa-herm-s1 general-short 3:59:00 8 --frame rwa --method Hermite,HermiteQGD --s 1 --nsteps "$(gen 19 20)"
bndl gf-rwa-cf-s0   general-short 3:59:00 8 --frame rwa --method ControlledFilon    --s 0 --nsteps "$(gen 22 23)"

# --- LAB uniform tails ---
bndl gf-lab-herm-s2 general-long 23:59:00 4 --frame lab --method Hermite,HermiteQGD --s 2 --nsteps "$(gen 23 23)"
for m in Hermite HermiteQGD; do
    one gf-lab-$m-s1-e26 general-long 23:59:00 $m 1 26
    one gf-lab-$m-s1-e27 general-long 35:59:00 $m 1 27
done
one gf-lab-ControlledFilon-s0-e26 general-long 47:59:00 ControlledFilon 0 26

echo "# submitted (or dry-ran) all CNOT3 uniform gap-fill batches" >&2
