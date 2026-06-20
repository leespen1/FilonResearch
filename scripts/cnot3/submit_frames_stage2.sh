#!/bin/bash
# Stage-2 resubmission for the CNOT3 norwa/lab campaign: stage 1 lost most of
# its wall time to a shared-depot precompile storm (33 cold jobs churning each
# other's caches), so the batch jobs deposited little.  The depot has since
# been fully precompiled in one sequential pass; these resubmissions skip every
# cached config via produce_or_load.
#
# Depth extensions beyond stage 1, sized from measured deep per-step costs:
#   hermite s=1 -> 2^24 in both frames (~0.73 ms/step, ~3.4h)
#   lab hermite s=2 -> 2^20, 2^21 (per-step cost falls ~3x at fine dt)
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
    submit c3-$frame-fc-light 12 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192
    submit c3-$frame-fc-mid 6 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 16384,32768
    submit c3-$frame-fc-16 6 4G --frame $frame \
        --method filon,controlled_filon --s 0,1,2 --nsteps 65536
    submit c3-$frame-h-bulk 12 4G --frame $frame --method hermite --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072
done
# norwa h-mid is the canary (submitted separately); lab h-mid completed in stage 1.

deep () { # deep <frame> <s> <exponents...>
    local frame=$1 s=$2; shift 2
    for e in "$@"; do
        submit c3-$frame-h-s$s-e$e 1 8G --frame $frame --method hermite \
            --s $s --nsteps $((2**e))
    done
}
# Resubmits of stage-1 TIMEOUTs (solve times fit easily with a warm cache)
deep norwa 0 21 22 24
deep norwa 1 20 22
deep norwa 2 21 23
deep lab   0 21
deep lab   1 23
deep lab   2 20
# New depth extensions
deep norwa 1 24
deep lab   1 24
deep lab   2 21
