#!/bin/bash
# Extend the deepest lab-frame order-2/order-4 tails by 2-3 more nsteps each, now
# that the measured solve times show headroom: the deepest runs took only 0.6-6 h,
# and times grow ~1.8x per doubling, so e26-e28 fit comfortably in a 12-24 h wall.
# One (method, s, nsteps, IC) per job, as requested.  GMRES tol 1e-15.
#
# Extends: lab Filon s0, lab Hermite/HermiteQGD s0, lab Hermite/HermiteQGD s1
# (both basis and uniform).  produce_or_load skips anything already present.
#
# Dry run:  DRYRUN=1 ./submit_recollect_extend.sh
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON="--gmres-atol 1e-15 --gmres-rtol 1e-15 --nruns 1"

# ext METHOD S E IC WALL  — one job, one method, one nsteps.
ext() {
    local method=$1 s=$2 e=$3 ic=$4 wall=$5
    local cmd=(sbatch -p general-long --time="$wall" -J "ext-${method}-s${s}-e${e}-${ic}" \
        --ntasks-per-node=1 --cpus-per-task=4 --mem-per-cpu=8G "$SB" \
        --frame lab --method "$method" --s "$s" --nsteps "$((2**e))" --init "$ic" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# ===== basis IC =====
ext Filon      0 26 basis 12:00:00; ext Filon      0 27 basis 12:00:00; ext Filon      0 28 basis 18:00:00
ext Hermite    0 25 basis 12:00:00; ext Hermite    0 26 basis 12:00:00; ext Hermite    0 27 basis 12:00:00
ext HermiteQGD 0 25 basis 12:00:00; ext HermiteQGD 0 26 basis 12:00:00; ext HermiteQGD 0 27 basis 18:00:00
ext Hermite    1 26 basis 18:00:00; ext Hermite    1 27 basis 24:00:00
ext HermiteQGD 1 26 basis 18:00:00; ext HermiteQGD 1 27 basis 24:00:00

# ===== uniform IC (cheaper, 1 column) =====
ext Filon      0 26 uniform 12:00:00; ext Filon      0 27 uniform 12:00:00; ext Filon      0 28 uniform 12:00:00
ext Hermite    0 26 uniform 12:00:00; ext Hermite    0 27 uniform 12:00:00; ext Hermite    0 28 uniform 12:00:00
ext HermiteQGD 0 26 uniform 12:00:00; ext HermiteQGD 0 27 uniform 12:00:00; ext HermiteQGD 0 28 uniform 12:00:00
ext Hermite    1 28 uniform 18:00:00; ext Hermite    1 29 uniform 24:00:00
ext HermiteQGD 1 28 uniform 18:00:00; ext HermiteQGD 1 29 uniform 24:00:00

echo "# submitted (or dry-ran) all CNOT3 deep-extension jobs" >&2
