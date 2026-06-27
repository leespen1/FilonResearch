#!/bin/bash
# Re-collection of the CNOT3 sweep with GMRES iteration tracking (mean/median/
# max/std per run), plus lab-frame depth extensions.
#
# Hermite (QGD) is the one method WITHOUT GMRES, and its output is byte-for-byte
# unchanged by the tracking edit, so its files are copied from the previous data
# dir instead of recomputed (see copy step in the session notes) — this script
# therefore submits NO base-range hermite, only the Filon-family rerun and the
# lab-frame deepenings (incl. new deep hermite).
#
# All jobs use the node-local-depot script (immune to shared-depot churn).
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

# --- Filon-family base-range rerun, all frames (2^4 .. 2^16) -----------------
for frame in rwa norwa lab; do
    submit gt-$frame-g-light 12 1 --frame $frame --method $GMRES --s 0,1,2 \
        --nsteps 16,32,64,128,256,512,1024,2048,4096,8192
    submit gt-$frame-g-e14 9 1 --frame $frame --method $GMRES --s 0,1,2 --nsteps 16384
    submit gt-$frame-g-e15 9 1 --frame $frame --method $GMRES --s 0,1,2 --nsteps 32768
    submit gt-$frame-g-e16 9 1 --frame $frame --method $GMRES --s 0,1,2 --nsteps 65536
done

# --- Lab-frame depth extensions ---------------------------------------------
# Lab dynamics carry the full carriers (no rotating frame), so the iterative
# methods need finer Δt than 2^16 to converge.  Past the coarse-step blowup edge
# (~2^14-15) per-step GMRES cost FALLS, so 2^17/2^18 are reachable within 4h;
# each is its own single-task job and a TIMEOUT just marks the 4h frontier.
labg () {   # labg <method> <s> <exponents...>
    local method=$1 s=$2; shift 2
    for e in "$@"; do
        submit gt-lab-$method-s$s-e$e 1 4 --frame lab --method $method \
            --s $s --nsteps $((2**e))
    done
}
for method in filon controlled_filon controlled_hermite; do
    for s in 0 1 2; do labg $method $s 17 18; done
done

# Lab hermite is order-6-still-descending at 2^22; push toward its floor.
labh () { local s=$1; shift; for e in "$@"; do
    submit gt-lab-h-s$s-e$e 1 4 --frame lab --method hermite --s $s --nsteps $((2**e)); done; }
labh 0 23 24
labh 1 23 24
labh 2 23
