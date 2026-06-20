#!/bin/bash
# Stage-3: floor-hunting extensions and lab-frame gap fill, all through the
# node-local-depot batch script (immune to the shared-depot precompile churn).
#
# Dropped vs earlier plans: norwa hermite s=1 2^24 and s=2 2^23 (floors already
# reached at 2^22), lab cfilon s=2 beyond 2^14 (deep blowup region, cannot fit
# the 4h wall even optimistically).
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON=(-p general-short --time=3:59:00)

submit () {
    local name=$1 ntasks=$2 cpt=$3; shift 3
    sbatch "${COMMON[@]}" -J "$name" --ntasks-per-node="$ntasks" \
        --cpus-per-task="$cpt" --mem-per-cpu=4G "$SB" "$@"
}

# norwa Filon-family floor extensions (cheap, grouped; wall = largest nsteps)
submit c3s3-norwa-f-s0  4 1 --frame norwa --method filon --s 0 --nsteps 131072,262144,524288,1048576
submit c3s3-norwa-f-s1  3 1 --frame norwa --method filon --s 1 --nsteps 131072,262144,524288
submit c3s3-norwa-f-s2  2 1 --frame norwa --method filon --s 2 --nsteps 131072,262144
submit c3s3-norwa-cf-s0 3 1 --frame norwa --method controlled_filon --s 0 --nsteps 131072,262144,524288
submit c3s3-norwa-cf-s1 2 1 --frame norwa --method controlled_filon --s 1 --nsteps 131072,262144
submit c3s3-norwa-cf-s2 1 4 --frame norwa --method controlled_filon --s 2 --nsteps 131072

# lab hermite depth (order-6 toward floor; s=1 as deep as 4h allows)
submit c3s3-lab-h-s1-e24 1 4 --frame lab --method hermite --s 1 --nsteps 16777216
submit c3s3-lab-h-s2-e22 1 4 --frame lab --method hermite --s 2 --nsteps 4194304
submit c3s3-lab-h-s2-e23 1 4 --frame lab --method hermite --s 2 --nsteps 8388608

# lab filon s=0 depth (order-2 line)
submit c3s3-lab-f-s0 2 1 --frame lab --method filon --s 0 --nsteps 131072,262144

# lab filon/cfilon onset hunting: one single per config; TIMEOUTs mark the
# 4h frontier (costs fall sharply once GMRES leaves the blowup regime).
single () { # single <method-tag> <method> <s> <exponents...>
    local tag=$1 method=$2 s=$3; shift 3
    for e in "$@"; do
        submit c3s3-lab-$tag-s$s-e$e 1 4 --frame lab --method $method \
            --s $s --nsteps $((2**e))
    done
}
single f  filon            1 14 15 16
single f  filon            2 13 14 15 16
single cf controlled_filon 0 14 15
single cf controlled_filon 1 14 15 16
single cf controlled_filon 2 12 13 14
