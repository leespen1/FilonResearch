# =============================================================================
# Convergence study of Filon / controlled-Filon on problems with NO clean
# rotating frame (where the oscillation cannot be analytically transformed away,
# so an integrator must handle it directly).  Three small test problems:
#
#   P1. Parametric flux modulation (2 qubits, 4-dim): a flux-modulated qubit
#       frequency ω_b(t)=ω_b0 + A cos(ω_m t).  The rotating transform of the
#       modulated b†b produces a Bessel comb in the coupling — no finite fixed
#       carrier set.  Drift diagonal is constant ⇒ Filon ansatz = bare levels.
#   P2. Chirped drive (2-level): E₀cos(ω₀t + ½βt²) — the carrier frequency
#       sweeps, so there is no fixed carrier.  Controlled Filon can still put ω₀
#       in the ansatz, leaving a chirped (complex) envelope.
#   P3. Few-cycle Gaussian pulse (2-level): E₀ e^{-(t-t₀)²/2τ²} cos(ω₀t) with
#       τ ≈ 1.5 carrier periods — the envelope varies on the carrier timescale
#       (no scale separation), stressing the "slow envelope" assumption.
#
# Subjects: filon_solve_hardcoded, controlled_filon_solve (s=0,1,2 → order 2,4,6).
# Competitor: classical fixed-step RK4 (order 4).  Reference (ground truth):
# a very fine independent RK4, cross-checked against fine Filon s=2.
# Errors are final-time 2-norm vs the reference (true ‖ψ‖≡1).
#
# Run under the umbrella env via srun, then osc_plot.jl renders figures:
#   srun -p general-short --constraint=intel18 -n1 -c1 --mem=4G -t 0:25:00 \
#       julia --project=. scripts/oscillatory/osc_compute.jl
# =============================================================================

using FilonResearch
using LinearAlgebra
using StaticArrays
using Serialization
using Printf

# ---- operators ----
const Z2 = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)
const X2 = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)
const I2 = SMatrix{2,2,ComplexF64}(1, 0, 0, 1)
kron4(A, B) = SMatrix{4,4,ComplexF64}(kron(Matrix(A), Matrix(B)))

# Custom complex control from its analytic derivatives of orders 0,1,2 (all the
# hard-coded Filon method needs for s ≤ 2).
mkfc(d0, d1, d2) = FunctionControl{ComplexF64}((t, n) ->
    n == 0 ? ComplexF64(d0(t)) :
    n == 1 ? ComplexF64(d1(t)) :
    n == 2 ? ComplexF64(d2(t)) :
    throw(ArgumentError("derivative order $n > 2 not provided")))

# ansatz frequencies ω_k from a diagonal generator-drift A_drift = i·diag(ω).
ansatz_freqs(A_drift) = SVector(real.(-im .* diag(A_drift))...)

# ---------------------------------------------------------------------------
# P1 — parametric flux modulation (2 qubits)
# ---------------------------------------------------------------------------
function prob_parametric()
    ωa, ωb0, Amod, ωm, g, T = 5.0, 4.0, 0.5, 1.0, 0.05, 100.0
    Za, Zb, XaXb = kron4(Z2, I2), kron4(I2, Z2), kron4(X2, X2)
    Hdrift = (ωa / 2) * Za + (ωb0 / 2) * Zb
    Adrift = -im * Hdrift
    freqs = ansatz_freqs(Adrift)
    # plain: modulation as a single cosine coefficient on Z_b; coupling constant
    co_plain = ControlledOperator(
        (ConstantControl(ComplexF64(1)),
         FourierControl(ComplexF64(0), [ComplexF64(-im * Amod / 2)], [ComplexF64(0)], ωm),
         ConstantControl(ComplexF64(-im * g))),
        (Adrift, Zb, XaXb))
    # controlled: modulation split into ±ω_m carriers (envelope −iA/4 each)
    env = ConstantControl(ComplexF64(-im * Amod / 4))
    co_ctrl = ControlledOperator(
        (ConstantControl(ComplexF64(1)), ConstantControl(ComplexF64(-im * g)),
         CarrierControl(env, ωm), CarrierControl(env, -ωm)),
        (Adrift, XaXb, Zb, Zb))
    ψ0 = SVector{4,ComplexF64}(0, 1, 0, 0)             # |01>
    return (; label = "parametric", co_plain, co_ctrl, freqs, ψ0, T,
              erange = 6:18, eref = 21)
end

# ---------------------------------------------------------------------------
# P2 — chirped drive (2-level)
# ---------------------------------------------------------------------------
function prob_chirp()
    ω0, E0, T = 20.0, 2.0, 10.0
    β = 2 * 5.0 / T                                    # sweep ±5 over [0,T]
    Hdrift = (ω0 / 2) * Z2; Adrift = -im * Hdrift
    freqs = ansatz_freqs(Adrift)
    φ(t) = ω0 * t + β * t^2 / 2; φp(t) = ω0 + β * t
    h0(t) = cos(φ(t)); h1(t) = -φp(t) * sin(φ(t))
    h2(t) = -β * sin(φ(t)) - φp(t)^2 * cos(φ(t))
    drive = mkfc(t -> -im * E0 * h0(t), t -> -im * E0 * h1(t), t -> -im * E0 * h2(t))
    co_plain = ControlledOperator((ConstantControl(ComplexF64(1)), drive), (Adrift, X2))
    # controlled: carriers ±ω0, chirped complex envelope −i(E0/2)e^{±iβt²/2}
    ec(sgn) = (e0 = t -> cis(sgn * β * t^2 / 2);
               (mkfc(t -> -im * (E0 / 2) * e0(t),
                     t -> -im * (E0 / 2) * im * sgn * β * t * e0(t),
                     t -> -im * (E0 / 2) * (im * sgn * β - β^2 * t^2) * e0(t))))
    co_ctrl = ControlledOperator(
        (ConstantControl(ComplexF64(1)), CarrierControl(ec(+1), ω0), CarrierControl(ec(-1), -ω0)),
        (Adrift, X2, X2))
    ψ0 = SVector{2,ComplexF64}(1, 0)
    return (; label = "chirp", co_plain, co_ctrl, freqs, ψ0, T, erange = 5:18, eref = 21)
end

# ---------------------------------------------------------------------------
# P3 — few-cycle Gaussian pulse (2-level)
# ---------------------------------------------------------------------------
function prob_fewcycle()
    ω0, E0, T = 20.0, 10.0, 5.0
    τ = 1.5 * (2π / ω0); t0 = T / 2
    Hdrift = (ω0 / 2) * Z2; Adrift = -im * Hdrift
    freqs = ansatz_freqs(Adrift)
    G(t) = exp(-((t - t0) / τ)^2 / 2)
    Gp(t) = -((t - t0) / τ^2) * G(t)
    Gpp(t) = (((t - t0)^2 - τ^2) / τ^4) * G(t)
    c0(t) = cos(ω0 * t); c1(t) = -ω0 * sin(ω0 * t); c2(t) = -ω0^2 * cos(ω0 * t)
    f0(t) = G(t) * c0(t)
    f1(t) = Gp(t) * c0(t) + G(t) * c1(t)
    f2(t) = Gpp(t) * c0(t) + 2 * Gp(t) * c1(t) + G(t) * c2(t)
    drive = mkfc(t -> -im * E0 * f0(t), t -> -im * E0 * f1(t), t -> -im * E0 * f2(t))
    co_plain = ControlledOperator((ConstantControl(ComplexF64(1)), drive), (Adrift, X2))
    env = mkfc(t -> -im * (E0 / 2) * G(t), t -> -im * (E0 / 2) * Gp(t), t -> -im * (E0 / 2) * Gpp(t))
    co_ctrl = ControlledOperator(
        (ConstantControl(ComplexF64(1)), CarrierControl(env, ω0), CarrierControl(env, -ω0)),
        (Adrift, X2, X2))
    ψ0 = SVector{2,ComplexF64}(1, 0)
    return (; label = "fewcycle", co_plain, co_ctrl, freqs, ψ0, T, erange = 5:18, eref = 21)
end

# ---------------------------------------------------------------------------
# RK4 (fixed step) for dψ/dt = A(t)ψ, using the dense A(t) from the operator.
# ---------------------------------------------------------------------------
A_at(co, t) = materialize(evaluate(co, t, Derivative{0}()))
function rk4_final(co, ψ0::SVector{N}, T, nsteps) where {N}
    h = T / nsteps; ψ = ψ0
    @inbounds for n in 0:nsteps-1
        t = n * h
        k1 = A_at(co, t) * ψ
        k2 = A_at(co, t + h/2) * (ψ + (h/2) * k1)
        k3 = A_at(co, t + h/2) * (ψ + (h/2) * k2)
        k4 = A_at(co, t + h) * (ψ + h * k3)
        ψ = ψ + (h/6) * (k1 + 2k2 + 2k3 + k4)
    end
    return ψ
end

# Gauss–Legendre implicit RK tableaux (orders 2,4,6 for m=1,2,3 stages).
# Structure-preserving (unitary for skew-Hermitian A) and A-stable — a fair,
# strong matched-order competitor: GL order 2m vs Filon s=m−1 (order 2m).
const GLTAB = Dict(
    1 => (c = [0.5], a = reshape([0.5], 1, 1), b = [1.0]),
    2 => (c = [0.5 - √3/6, 0.5 + √3/6],
          a = [0.25 0.25-√3/6; 0.25+√3/6 0.25], b = [0.5, 0.5]),
    3 => (c = [0.5 - √15/10, 0.5, 0.5 + √15/10],
          a = [5/36 2/9-√15/15 5/36-√15/30;
               5/36+√15/24 2/9 5/36-√15/24;
               5/36+√15/30 2/9+√15/15 5/36],
          b = [5/18, 4/9, 5/18]))

# One GL step for the linear ODE dψ/dt=A(t)ψ: the stage derivatives K solve the
# (m·N) linear system  K_i − h Σ_j a_ij A_i K_j = A_i ψ.
function gl_final(co, ψ0::SVector{N}, T, nsteps, m) where {N}
    tab = GLTAB[m]; h = T / nsteps; ψ = ψ0
    M = zeros(ComplexF64, m*N, m*N); rhs = zeros(ComplexF64, m*N)
    @inbounds for n in 0:nsteps-1
        t = n * h
        fill!(M, 0)
        for i in 1:m
            Ai = A_at(co, t + tab.c[i] * h); ri = (i-1)*N
            for k in 1:N; M[ri+k, ri+k] += 1; end
            for j in 1:m
                cj = (j-1)*N; aij = h * tab.a[i, j]
                for r in 1:N, c in 1:N; M[ri+r, cj+c] -= aij * Ai[r, c]; end
            end
            Aiψ = Ai * ψ
            for k in 1:N; rhs[ri+k] = Aiψ[k]; end
        end
        K = M \ rhs
        acc = zero(SVector{N,ComplexF64})
        for i in 1:m
            ri = (i-1)*N
            acc += tab.b[i] * SVector{N,ComplexF64}(ntuple(k -> K[ri+k], N))
        end
        ψ = ψ + h * acc
    end
    return ψ
end

# final state for a named method (s = order parameter for filon/cfilon, stage
# count m for gl, ignored for rk4)
function solve_final(method, s, P, nsteps)
    if method === :filon
        return SVector(filon_solve_hardcoded(P.co_plain, P.ψ0, P.freqs, P.T / nsteps, nsteps, s;
                                             save_final_only = true)...)
    elseif method === :cfilon
        return SVector(controlled_filon_solve(P.co_ctrl, P.ψ0, P.freqs, P.T / nsteps, nsteps, s;
                                              save_final_only = true)...)
    elseif method === :gl
        return gl_final(P.co_plain, P.ψ0, P.T, nsteps, s)
    else
        return rk4_final(P.co_plain, P.ψ0, P.T, nsteps)
    end
end

# ---------------------------------------------------------------------------
function fd_check(P)
    h = 1e-5; mx = 0.0
    for co in (P.co_plain, P.co_ctrl)
        for ctrl in co.controls, t in (0.13, 0.37 * P.T, 0.71 * P.T)
            v(τ) = derivative(ctrl, τ, Derivative{0}())
            d1 = derivative(ctrl, t, Derivative{1}()); d2 = derivative(ctrl, t, Derivative{2}())
            fd1 = (v(t + h) - v(t - h)) / (2h)
            fd2 = (v(t + h) - 2v(t) + v(t - h)) / h^2
            mx = max(mx, abs(d1 - fd1) / (abs(fd1) + 1e-8), abs(d2 - fd2) / (abs(fd2) + 1e-6))
        end
    end
    return mx
end

function run_problem(P)
    println("\n===== Problem: $(P.label)  (N=$(length(P.ψ0)), T=$(P.T)) =====")
    @printf("  control-derivative FD check (max rel err): %.2e\n", fd_check(P))

    nref = 2^P.eref
    ψref = rk4_final(P.co_plain, P.ψ0, P.T, nref)
    # cross-check the reference against fine Filon s=2 (independent method)
    ψfil = solve_final(:filon, 2, P, 2^min(P.eref - 3, 18))
    @printf("  reference RK4(2^%d) vs Filon s=2(2^%d) agree to %.2e  (‖ψref‖=%.3f)\n",
            P.eref, min(P.eref - 3, 18), norm(ψref - ψfil), norm(ψref))

    methods = [(:filon, 0), (:filon, 1), (:filon, 2),
               (:cfilon, 0), (:cfilon, 1), (:cfilon, 2),
               (:gl, 1), (:gl, 2), (:gl, 3), (:rk4, 0)]
    err = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    tim = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    es = collect(P.erange)
    for (m, s) in methods
        solve_final(m, s, P, 2^4)                       # warm up / compile
        ev = Float64[]; tv = Float64[]
        for e in es
            ns = 2^e
            t = @elapsed ψf = solve_final(m, s, P, ns)
            push!(ev, norm(ψf - ψref)); push!(tv, t)
        end
        err[(m, s)] = ev; tim[(m, s)] = tv
        tag = m === :rk4 ? "rk4" : "$(m) s=$s"
        @printf("  %-10s  err: %.2e → %.2e  (over 2^%d..2^%d)\n",
                tag, ev[1], minimum(ev), first(es), last(es))
    end
    return (; label = P.label, N = length(P.ψ0), T = P.T, es, methods, err, tim, nref)
end

function main()
    results = map(run_problem, (prob_parametric(), prob_chirp(), prob_fewcycle()))
    outdir = normpath(joinpath(@__DIR__, "..", "..", "data", "oscillatory")); mkpath(outdir)
    path = joinpath(outdir, "osc_data.jls")
    serialize(path, results)
    println("\nSerialized → ", path)
    println("OSC_COMPUTE_DONE")
end
main()
