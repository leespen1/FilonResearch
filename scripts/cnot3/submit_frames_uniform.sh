#!/bin/bash
# Uniform-IC campaign for the CNOT3 norwa/lab frames, sized from the measured
# basis-IC costs (a uniform run is ~0.37x a basis run: one column at ~1.5x
# per-column cost).  Everything goes through the node-local-depot script.
# Notable depth gains over basis: lab filon/cfilon s=2 2^15 now fit a 4h wall,
# hermite s=1 reaches 2^25 in both frames.
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON=(-p general-short --time=3:59:00)

submit () {
    local name=$1 ntasks=$2 cpt=$3; shift 3
    sbatch "${COMMON[@]}" -J "$name" --ntasks-per-node="$ntasks" \
        --cpus-per-task="$cpt" --mem-per-cpu=4G "$SB" --init uniform "$@"
}

for frame in norwa lab; do
    submit u-$frame-fc-light 12 1 --frame $frame \
        --method filon,controlled_filon --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192
    submit u-$frame-fc-mid 6 1 --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 16384,32768
    submit u-$frame-fc-16 6 1 --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 65536
    submit u-$frame-h-bulk 12 1 --frame $frame --method hermite --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072
    submit u-$frame-h-mid 6 1 --frame $frame --method hermite --s 0,1,2 \
        --nsteps 262144,524288
done

# norwa Filon-family floor extensions (mirrors the basis stage-3 ranges, with
# the extra cross-product points cheap enough to keep)
submit u-norwa-f-ext   12 1 --frame norwa --method filon --s 0,1,2 \
    --nsteps 131072,262144,524288,1048576
submit u-norwa-cf-ext   6 1 --frame norwa --method controlled_filon --s 0,1,2 \
    --nsteps 131072,262144
submit u-norwa-cf-ext2  2 1 --frame norwa --method controlled_filon --s 0,1 \
    --nsteps 524288

# lab Filon-family depth beyond the bulk jobs
submit u-lab-fc-17    6 1 --frame lab --method filon,controlled_filon --s 0,1,2 \
    --nsteps 131072
submit u-lab-f-s0-e18 1 4 --frame lab --method filon --s 0 --nsteps 262144

# Deep hermite, grouped per (frame, s); every member fits the wall at the
# measured uniform-scaled per-step costs
submit u-norwa-h-s0deep 5 1 --frame norwa --method hermite --s 0 \
    --nsteps 1048576,2097152,4194304,8388608,16777216
submit u-norwa-h-s1deep 6 1 --frame norwa --method hermite --s 1 \
    --nsteps 1048576,2097152,4194304,8388608,16777216,33554432
submit u-norwa-h-s2deep 5 1 --frame norwa --method hermite --s 2 \
    --nsteps 1048576,2097152,4194304,8388608,16777216
submit u-lab-h-s0deep   6 1 --frame lab --method hermite --s 0 \
    --nsteps 1048576,2097152,4194304,8388608,16777216,33554432
submit u-lab-h-s1deep   6 1 --frame lab --method hermite --s 1 \
    --nsteps 1048576,2097152,4194304,8388608,16777216,33554432
submit u-lab-h-s2deep   5 1 --frame lab --method hermite --s 2 \
    --nsteps 1048576,2097152,4194304,8388608,16777216
submit u-lab-h-s2-e25   1 4 --frame lab --method hermite --s 2 --nsteps 33554432
