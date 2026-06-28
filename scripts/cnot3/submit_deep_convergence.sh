#!/bin/bash
# Deep CNOT3 convergence campaign in the lab and rotating (rwa, norwa) frames,
# for both the essential-basis and uniform-superposition initial conditions, at
# GMRES tolerance 1e-15 and nRuns=1.
#
# Timestep ranges per (frame, method, s) start just before the asymptotic regime
# (a little pre-asymptotic data is fine) and go deep enough that every method
# reaches at least ~1e-3 relative final-state error.  Low-order Hermite in the
# lab frame must resolve the carriers and converges very slowly: order-2 (s=0)
# only enters its asymptotic regime around 2^26 and reaches 1e-3 near 2^30
# (verified against the 1e-15 Vern9 reference), so it is pushed that deep — a
# multi-day run for the basis initial condition.
#
# Every (frame, method, s, nsteps) solve is cached individually by
# produce_or_load, so completed configs survive a job hitting its wall-clock
# limit, and jobs compose / can be resubmitted to fill gaps.
#
# Dry run (print the sbatch commands without submitting):
#   DRYRUN=1 ./submit_deep_convergence.sh
set -euo pipefail
cd "$(dirname "$0")"

SB=cnot3_convergence_localdepot.sb
COMMON="--gmres-atol 1e-15 --gmres-rtol 1e-15 --nruns 1"

# Comma-separated 2^e list for e in [$1, $2].
gen() { local e out=""; for ((e=$1; e<=$2; e++)); do out+="$((2**e)),"; done; echo "${out%,}"; }

# sub JOBNAME PARTITION TIME NTASKS CPUS <collect_data args...>
sub() {
    local name=$1 part=$2 time=$3 nt=$4 cpus=$5; shift 5
    local cmd=(sbatch -p "$part" --time="$time" -J "$name" \
        --ntasks-per-node="$nt" --cpus-per-task="$cpus" --mem-per-cpu=4G \
        "$SB" "$@" $COMMON)
    if [ -n "${DRYRUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# ---------------------------------------------------------------------------
# Rotating frames (rwa, norwa): carriers are slow/removed, so every method
# converges fast and cheap.  general-short (4 h) is plenty.
# ---------------------------------------------------------------------------
sub rot-filon    general-short 3:59:00 16 1 \
    --frame rwa,norwa --method filon,controlled_filon,controlled_hermite \
    --s 0,1,2 --nsteps "$(gen 10 18)" --init basis,uniform
sub rot-herm-s0  general-short 3:59:00 16 1 \
    --frame rwa,norwa --method hermite --s 0 --nsteps "$(gen 11 22)" --init basis,uniform
sub rot-herm-s12 general-short 3:59:00 16 1 \
    --frame rwa,norwa --method hermite --s 1,2 --nsteps "$(gen 12 20)" --init basis,uniform

# ---------------------------------------------------------------------------
# Lab frame, carrier-folding Filon methods: reach 1e-3 by ~2^18, swept a bit
# deeper to the round-off floor.  GMRES at 1e-15 is dearer, so general-long.
# ---------------------------------------------------------------------------
# s=0 is swept as deep as the carrier-resolving Hermite (below) so the curves
# span the same Δt range: filon all the way to 2^30; controlled_filon is GMRES-
# bound (~1.8 ms/step) so 2^30 would take ~3 weeks — capped at 2^27 (the deepest
# that fits the 7-day partition).
sub lab-filon-s0  general-long 119:59:00 12 1 \
    --frame lab --method filon --s 0 --nsteps "$(gen 14 30)" --init basis,uniform
sub lab-filon-s1  general-long 11:59:00 16 1 \
    --frame lab --method filon --s 1 --nsteps "$(gen 14 21)" --init basis,uniform
sub lab-filon-s2  general-long 11:59:00 16 1 \
    --frame lab --method filon --s 2 --nsteps "$(gen 13 20)" --init basis,uniform
sub lab-cf-s0     general-long 95:59:00 14 1 \
    --frame lab --method controlled_filon --s 0 --nsteps "$(gen 12 27)" --init basis,uniform
sub lab-cf-s1     general-long 11:59:00 16 1 \
    --frame lab --method controlled_filon --s 1 --nsteps "$(gen 14 20)" --init basis,uniform
sub lab-cf-s2     general-long 23:59:00 12 1 \
    --frame lab --method controlled_filon --s 2 --nsteps "$(gen 14 19)" --init basis,uniform

# ---------------------------------------------------------------------------
# Lab frame, carrier-resolving Hermite methods.  s=1,2 reach 1e-3 at moderate
# depth; s=0 (order 2) is the slow case and gets pushed very deep below.
# ---------------------------------------------------------------------------
sub lab-herm-s0-sh general-long 11:59:00 16 1 \
    --frame lab --method hermite,controlled_hermite --s 0 --nsteps "$(gen 14 24)" --init basis,uniform
sub lab-herm-s1    general-long 11:59:00 16 1 \
    --frame lab --method hermite,controlled_hermite --s 1 --nsteps "$(gen 16 24)" --init basis,uniform
sub lab-herm-s2    general-long 11:59:00 16 1 \
    --frame lab --method hermite,controlled_hermite --s 2 --nsteps "$(gen 14 23)" --init basis,uniform

# ---------------------------------------------------------------------------
# Lab frame, deep low-order Hermite (s=0): the headline deep-convergence runs.
# Basis is ~4x dearer than uniform (4 gate columns vs 1 state), so they get
# separate jobs / wall times.  Each nsteps is cached as it finishes, so even if
# 2^28 does not complete, 2^25..2^27 are saved.
# ---------------------------------------------------------------------------
sub lab-herm-s0-deep-basis   general-long 143:59:00 12 1 \
    --frame lab --method hermite,controlled_hermite --s 0 --nsteps "$(gen 25 30)" --init basis
sub lab-herm-s0-deep-uniform general-long  47:59:00 12 1 \
    --frame lab --method hermite,controlled_hermite --s 0 --nsteps "$(gen 25 30)" --init uniform

echo "# submitted (or dry-ran) all CNOT3 deep-convergence batches" >&2
