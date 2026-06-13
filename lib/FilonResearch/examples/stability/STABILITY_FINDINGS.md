# Filon / controlled-Filon stability — empirical findings

Numerical stability study of `filon_solve_hardcoded` and `controlled_filon_solve`
(orders s = 0,1,2 → 2,4,6). Scripts: `stab_compute.jl` (compute → `data/stability/stab_data.jls`),
`stab_plot.jl` (→ `plots/stability/*.png`), `stab_probe.jl` (harness validation).
Run on SLURM (`srun -p general-short`); the amplification factor is computed
*exactly* — a 1×1 `SMatrix` operator takes the static `\` path (no GMRES).

The one-step multiplier of the scalar test equation u′ = λu, ansatz frequency ω,
depends only on `z = λΔt` and `θ = ωΔt` (verified to 0 error under Δt rescaling).
The hand-derived `f4` in `stability_plot_order2.jl` does **not** match the
implemented method (max diff ≈ 1.9) — a different convention; do not trust it.

## A. Constant coefficient, scalar / diagonal-in-ansatz generator
- **Exact at the design point** z = iθ: R = e^{iθ} to roundoff, all s.
- **Norm-preserving on the imaginary axis**: sup_ν ‖R(iν,θ)|−1| at roundoff for
  all s and all θ ∈ [0,100]. (s=2, θ≳50 shows ~1e-10 from catastrophic
  cancellation in F₂ = Ȧ + A² − Ω² − 2iΩA, *not* a true instability.)
- **Stability region = closed left half z-plane**, boundary = the imaginary axis,
  **independent of θ and s** → the method is **A-stable**.
- **Not L-stable**: |R(−X)| → 1 as X → ∞ (intermediate decay rates are damped,
  more so at higher s; infinitely stiff decay is not). This is a Gauss/Cayley-type
  stability function.

→ **Theorem candidate 1**: for a constant generator diagonal in the ansatz basis
(A = iΩ + diagonal), the Filon one-step map is a Cayley-like transform with
|R(iμ)| ≡ 1 for all real μ; the method is A-stable and exactly norm-preserving on
iℝ, unconditionally in Δt. Likely route: a conjugate symmetry W_I = conj(W_E)·phase
between the implicit/explicit Filon weights ⇒ |S_E| = |S_I| on iℝ.

## B. Scalar variable coefficient  u′ = iω₀(1 + ε cos Ωt) u  (true |u| ≡ 1)
- ε = 0 → norm preserved to roundoff (consistent with A).
- ε > 0 → growth; **decreases sharply with order** at fixed coarseness
  (s=0 ≫ s=1 ≫ s=2). Mechanism: the one-step map staggers A(t_n) (explicit)
  against A(t_{n+1}) (implicit), so |R_n| ≠ 1 even though each frozen value is
  on iℝ. At very coarse Δt all orders eventually grow.
- Controlled vs plain: comparable; controlled is sometimes **worse** at low order
  / strong modulation (s=0, ε=1: ctrl ~14–25× vs plain ~5×).

## C. Non-normal 2×2 (diagonal drift + carrier coupling) — the CNOT3 analogue
A(t) = [iω₁, g e^{iω_c t}; −g e^{−iω_c t}, iω₂], skew-Hermitian, true ‖ψ‖ ≡ 1.
- The coupling is **off-diagonal — not absorbed by the diagonal ansatz of plain
  Filon** — so the one-step matrix M = S_I⁻¹S_E is non-normal. At coarse steps
  ‖M‖₂ > 1 **and** ρ(M) > 1 for s ≥ 1 (genuine eigenvalue instability, not just
  transient non-normal growth). Even the *frozen* (constant-A) step has ‖M‖₂ > 1:
  the instability is fundamentally the generator-vs-ansatz mismatch, amplified by
  coarse Δt; time-variation / non-commuting [A(t₁),A(t₂)] adds to it.
- **Plain Filon blows up catastrophically at coarse steps** (G = max_n‖ψ_n‖ up to
  1e75–1e110), reproducing the CNOT3 coarse-step blow-up.
- **Controlled Filon** gives the carrier coupling its own ansatz (ω + ω_c):
  - constant envelope → **unconditionally norm-stable** (G ≡ 1.000 to roundoff at
    every coarseness and coupling tested). Conjecture: the step is *exactly*
    unitary for constant-envelope carrier couplings (the 2-level constant-envelope
    rotating drive is exactly solvable).
  - varying envelope (realistic, CNOT3-like) → both eventually blow up, but
    controlled is orders of magnitude more stable and the advantage grows with
    order (s=2: plain ~1e20 vs ctrl ~1e6).

→ **Theorem candidate 2**: plain Filon's stability for a coupled generator is
governed by ‖M‖₂; instability arises from the part of A not diagonalized by the
ansatz, with onset in ‖A_offdiag‖·Δt and ω Δt.
→ **Theorem candidate 3**: the controlled ansatz enlarges what is absorbed; for
constant-envelope carrier terms the step is (conjecturally) exactly unitary, and
for varying envelopes the growth bound improves with order.

## Headline
Constant-coefficient (diagonal) Filon is A-stable and norm-preserving on iℝ
**unconditionally** — so the CNOT3 coarse-step blow-up is **not** a
constant-coefficient instability. It is driven by generator components the ansatz
does not absorb (off-diagonal couplings, then time variation). Controlled Filon
absorbs the carrier couplings and is markedly more stable; this is the strongest
stability argument for the controlled method found so far.
