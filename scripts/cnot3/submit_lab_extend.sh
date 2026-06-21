#!/bin/bash
# Lab-frame depth extension to complete the convergence picture.  Sized from the
# measured deep per-step costs (controlled_hermite ~1-2 ms/step, filon ~3 ms,
# controlled_filon ~7-15 ms; all fall at finer Δt).  s=0 is order-2 and cannot
# floor before ~2^26, so it is left as the clean order-2 line and not extended.
#
# floors targeted (order p = 2(s+1), error ~ Δt^p from the 2^18 values):
#   filon s=1            -> 2^21      controlled_filon s=1 -> 2^20
#   controlled_hermite   -> still resolving carriers at 2^18; push s=2 to a floor
#                           (~2^21-22) and s=1 as deep as cheap (2^24)
#   hermite s=1          -> 2^24 (timed out on general-short; resubmit longer)
set -euo pipefail
cd "$(dirname "$0")"
SB=cnot3_convergence_localdepot.sb

short () { local name=$1 s=$2 method=$3 e=$4
    sbatch -p general-short --time=3:59:00 -J "$name" --ntasks-per-node=1 \
        --cpus-per-task=4 --mem-per-cpu=4G "$SB" --frame lab --method "$method" \
        --s "$s" --nsteps $((2**e)); }
long () { local name=$1 s=$2 method=$3 e=$4
    sbatch -p general-long --time=23:59:00 -J "$name" --ntasks-per-node=1 \
        --cpus-per-task=4 --mem-per-cpu=4G "$SB" --frame lab --method "$method" \
        --s "$s" --nsteps $((2**e)); }

# --- Filon family s=1 toward floor (general-short; all < 4h) ---
for e in 19 20 21; do short xl-f-s1-e$e   1 filon            $e; done
for e in 19 20;     do short xl-cf-s1-e$e  1 controlled_filon $e; done

# --- Controlled-Hermite: now entering convergence, push to floor (cheap) ---
for e in 19 20 21 22; do short xl-ch-s2-e$e 2 controlled_hermite $e; done
for e in 19 20 21 22; do short xl-ch-s1-e$e 1 controlled_hermite $e; done

# --- Deepest runs: general-long (7-day partition, intel18 via the .sb) ---
long xl-ch-s1-e23 1 controlled_hermite 23
long xl-ch-s1-e24 1 controlled_hermite 24
long xl-h-s1-e24  1 hermite            24
