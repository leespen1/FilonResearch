#!/bin/bash
# Stage-1 SLURM submission for the CNOT3 norwa/lab-frame convergence campaign.
# Run from scripts/cnot3 (logs land in ./log, data is commit-namespaced).
#
# Batching: small/medium configs are grouped into multi-worker jobs (the
# collection script spawns one Distributed worker per SLURM task and pmaps the
# configs); deep hermite runs get one single-task job each, with the deepest
# exponent per (frame, s) chosen so a flat per-step-cost extrapolation from the
# dev-node probes fits the 4h general-short wall.  TIMEOUTed configs simply
# stay uncached and define the 4h depth frontier.
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_collect_data.sb
COMMON=(-p general-short --time=3:59:00)

submit () {
    local name=$1 ntasks=$2 mem=$3; shift 3
    sbatch "${COMMON[@]}" -J "$name" --ntasks-per-node="$ntasks" \
        --mem-per-cpu="$mem" "$SB" "$@"
}

for frame in norwa lab; do
    # Filon-family bulk
    submit c3-$frame-fc-light 12 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192
    submit c3-$frame-fc-mid 6 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 16384,32768
    submit c3-$frame-fc-16 6 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 65536

    # Hermite bulk
    submit c3-$frame-h-bulk 12 4G --frame $frame --method hermite --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072
    submit c3-$frame-h-mid 6 4G --frame $frame --method hermite --s 0,1,2 \
        --nsteps 262144,524288
done

# Deep hermite singles: nsteps = 2^20 and up, one job per config.
deep () { # deep <frame> <s> <exponents...>
    local frame=$1 s=$2; shift 2
    for e in "$@"; do
        submit c3-$frame-h-s$s-e$e 1 8G --frame $frame --method hermite \
            --s $s --nsteps $((2**e))
    done
}
deep norwa 0 20 21 22 23 24
deep norwa 1 20 21 22 23
deep norwa 2 20 21 22 23
deep lab   0 20 21 22 23 24
deep lab   1 20 21 22 23
deep lab   2 20
