#!/bin/bash
# Extend the Filon-family CNOT3 sweep to Δt ≈ 1e-4 (nsteps up to 2^22) in ALL
# three frames, for filon / controlled_filon / controlled_hermite, s = 0,1,2.
# Cross-product submissions; produce_or_load skips configs already present.
#
# Cost (measured): rwa/norwa GMRES ≈ 7 iters → cheap; lab ≈ 85 iters → dear, and
# controlled_filon is the costly method.  Cheap groups go to general-short;
# the deep controlled_filon tail (e = 21,22; norwa/lab s=2 2^22 ≈ 5 h) goes to a
# 24 h wall so it never times out (one resubmit would cost more wall than the run).
set -euo pipefail
cd "$(dirname "$0")"
SB=cnot3_convergence_localdepot.sb

# nsteps helpers
e()  { echo $((2**$1)); }
csv() { local out=""; for x in "$@"; do out="$out,$(e $x)"; done; echo "${out:1}"; }

short () {  # short <name> <ntasks> -- <args...>
    local name=$1 nt=$2; shift 2
    sbatch -p general-short --time=3:59:00 -J "$name" --ntasks-per-node="$nt" \
        --cpus-per-task=1 --mem-per-cpu=4G "$SB" "$@"
}
long () {   # long <name> <ntasks> -- <args...>
    local name=$1 nt=$2; shift 2
    sbatch -p general-long --time=23:59:00 -J "$name" --ntasks-per-node="$nt" \
        --cpus-per-task=1 --mem-per-cpu=4G "$SB" "$@"
}

DEEP=$(csv 17 18 19 20 21 22)     # full extension range (cached coarser ones skipped)
MID=$(csv 17 18 19 20)
TAIL=$(csv 21 22)

for fr in rwa norwa; do
    # filon + controlled_hermite: cheap everywhere → one general-short job, full range
    short xt-$fr-fh 12 --frame $fr --method filon,controlled_hermite --s 0,1,2 --nsteps "$DEEP"
    # controlled_filon: mid range general-short, deep tail on the long wall
    short xt-$fr-cf   12 --frame $fr --method controlled_filon --s 0,1,2 --nsteps "$MID"
    long  xt-$fr-cfd   6 --frame $fr --method controlled_filon --s 0,1,2 --nsteps "$TAIL"
done

# lab: filon/ch cheap (general-short, full deep range — coarser already present);
# controlled_filon mid general-short, deep tail on the long wall.
short xt-lab-fh  12 --frame lab --method filon,controlled_hermite --s 0,1,2 --nsteps "$(csv 19 20 21 22)"
short xt-lab-cf   6 --frame lab --method controlled_filon --s 0,1,2 --nsteps "$(csv 19 20)"
long  xt-lab-cfd  6 --frame lab --method controlled_filon --s 0,1,2 --nsteps "$TAIL"
