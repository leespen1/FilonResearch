# =============================================================================
# Numerical stability study of the hard-coded Filon and controlled-Filon methods.
#
# Three experiments, increasing in structure:
#
#   A. CONSTANT-COEFFICIENT amplification factor R(z,θ) for the scalar test
#      equation u' = λu (z = λΔt, θ = ωΔt).  Maps the stability region in the
#      z-plane and tests norm preservation on the imaginary axis (skew-Hermitian
#      / Hamiltonian spectrum).  Exact: a 1×1 SMatrix takes the static `\` path.
#
#   B. SCALAR VARIABLE-COEFFICIENT  u' = i ω0 (1 + ε cos Ωt) u  (true |u| ≡ 1).
#      The one-step map staggers A(t_n) (explicit) against A(t_{n+1}) (implicit),
#      breaking the constant-coefficient unitarity.  Measures growth vs stepsize
#      and modulation depth; plain Filon vs controlled Filon (carrier-resolved).
#
#   C. NON-NORMAL 2×2 SYSTEM  dψ/dt = A(t)ψ, A skew-Hermitian, diagonal drift +
#      carrier-modulated coupling (a minimal CNOT3 analogue; true ‖ψ‖ ≡ 1).  The
#      product of (per step unitary, when frozen) one-step maps can still grow
#      because A(t_n) and A(t_{n+1}) do not commute.  This is the conjectured
#      CNOT3 blow-up mechanism.  Compares plain vs controlled Filon, and the
#      actual one-step matrix norm ‖M_n‖₂ against its frozen (commuting) value.
#
# Run (compute only; no plotting) via srun, e.g.
#   srun -p general-short --constraint=intel18 -n1 -c1 --mem=4G -t 0:20:00 \
#       julia --project=. lib/FilonResearch/examples/stability/stab_compute.jl
# Serializes results to data/stability/stab_data.jls for stab_plot.jl.
# =============================================================================

using FilonResearch
using LinearAlgebra
using StaticArrays
using Serialization
using Printf

geom(a, b, n) = exp.(range(log(a), log(b), length = n))

# -----------------------------------------------------------------------------
# Problem builders
# -----------------------------------------------------------------------------

# Plain-Filon exact one-step multiplier for u' = z u (Δt = 1), ansatz freq θ.
function R_filon_scalar(z::Complex, θ::Real, s::Int)
    co = ControlledOperator((ConstantControl(one(ComplexF64)),),
                            (SMatrix{1,1,ComplexF64}(z),))
    ψ = filon_solve_hardcoded(co, SVector{1,ComplexF64}(1.0), SVector(float(θ)),
                              1.0, 1, s; save_final_only = true)
    return ψ[1]
end

# Scalar variable-coefficient A(t) = i ω0 (1 + ε cos Ω t).  Returns matched
# plain (single Fourier coefficient) and controlled (drift + ±Ω carriers) co's.
function scalar_var_cos(ω0, ε, Ω)
    M1 = (SMatrix{1,1,ComplexF64}(1.0),)
    plain = ControlledOperator(
        (FourierControl(ComplexF64(im * ω0), [ComplexF64(im * ω0 * ε)],
                        [ComplexF64(0)], float(Ω)),), M1)
    g = ComplexF64(im * ω0 * ε / 2)
    ctrls = (ConstantControl(ComplexF64(im * ω0)),
             CarrierControl(ConstantControl(g),  float(Ω)),
             CarrierControl(ConstantControl(g), -float(Ω)))
    M3 = ntuple(_ -> SMatrix{1,1,ComplexF64}(1.0), 3)
    controlled = ControlledOperator(ctrls, M3)
    return plain, controlled
end

# 2×2 skew-Hermitian A(t) = [iω1, g e^{iωc t}; -g e^{-iωc t}, iω2].  One co serves
# both solvers: plain Filon folds the carrier into A(t); controlled Filon reads
# the carrier frequencies off the CarrierControls.
function twolevel_carrier(ω1, ω2, g, ωc)
    D   = SMatrix{2,2,ComplexF64}(im * ω1, 0, 0, im * ω2)   # [iω1 0; 0 iω2]
    E12 = SMatrix{2,2,ComplexF64}(0, 0, 1, 0)               # [0 1; 0 0]
    E21 = SMatrix{2,2,ComplexF64}(0, 1, 0, 0)               # [0 0; 1 0]
    ctrls = (ConstantControl(ComplexF64(1)),
             CarrierControl(ConstantControl(ComplexF64(g)),  float(ωc)),
             CarrierControl(ConstantControl(ComplexF64(-g)), -float(ωc)))
    co = ControlledOperator(ctrls, (D, E12, E21))
    return co, SVector(float(ω1), float(ω2))
end

# Same 2-level model but with a slowly *varying* coupling envelope
# env(t) = g(1 + ½cos Ωe t) (mirrors a CNOT3 B-spline envelope: controlled Filon
# is then finite-order, not near-exact as it is for a constant envelope).
function twolevel_varenv(ω1, ω2, g, ωc, Ωe)
    D   = SMatrix{2,2,ComplexF64}(im * ω1, 0, 0, im * ω2)
    E12 = SMatrix{2,2,ComplexF64}(0, 0, 1, 0)
    E21 = SMatrix{2,2,ComplexF64}(0, 1, 0, 0)
    envp = FourierControl(ComplexF64(g),  [ComplexF64(0.5g)],  [ComplexF64(0)], float(Ωe))
    envm = FourierControl(ComplexF64(-g), [ComplexF64(-0.5g)], [ComplexF64(0)], float(Ωe))
    ctrls = (ConstantControl(ComplexF64(1)),
             CarrierControl(envp,  float(ωc)),
             CarrierControl(envm, -float(ωc)))
    return ControlledOperator(ctrls, (D, E12, E21)), SVector(float(ω1), float(ω2))
end

# Closed-form A(t) for the 2-level model (for frozen-operator diagnostics).
Afull_2lvl(ω1, ω2, g, ωc, t) =
    SMatrix{2,2,ComplexF64}(im*ω1, -g*cis(-ωc*t), g*cis(ωc*t), im*ω2)  # col-major [iω1 ge^{iωct}; -ge^{-iωct} iω2]

# Trajectory of ‖ψ_n‖ (true ≡ ‖ψ0‖) for one solver/order.
function norm_traj(co, freqs, ψ0, Δt, nsteps, s; controlled::Bool)
    sol = controlled ?
        controlled_filon_solve(co, ψ0, freqs, Δt, nsteps, s; save_every = 1) :
        filon_solve_hardcoded(co, ψ0, freqs, Δt, nsteps, s; save_every = 1)
    return vec(sqrt.(sum(abs2, sol; dims = 1)))
end

# Actual one-step matrix M_n (plain Filon) advancing from t_n by Δt; columns are
# the images of the basis vectors.
function step_matrix_plain(co, freqs, Δt, s, t_n, ::Val{N}) where {N}
    wp = filon_weight_phases(co, freqs, Δt, s; variant = :static)
    cols = ntuple(N) do j
        e = SVector{N,ComplexF64}(ntuple(i -> i == j ? 1.0 : 0.0, N))
        Vector(filon_timestep_hardcoded(co, e, t_n, Δt, wp))
    end
    return reduce(hcat, cols)
end

# Frozen one-step matrix: constant generator Afrozen (Ȧ = 0).
function step_matrix_frozen(Afrozen::SMatrix{N,N}, freqs, Δt, s) where {N}
    co = ControlledOperator((ConstantControl(ComplexF64(1)),), (Afrozen,))
    wp = filon_weight_phases(co, freqs, Δt, s; variant = :static)
    cols = ntuple(N) do j
        e = SVector{N,ComplexF64}(ntuple(i -> i == j ? 1.0 : 0.0, N))
        Vector(filon_timestep_hardcoded(co, e, 0.0, Δt, wp))
    end
    return reduce(hcat, cols)
end

# =============================================================================
# Experiment A — constant-coefficient amplification / stability region
# =============================================================================
function experiment_A()
    println("\n===== Experiment A: constant-coefficient amplification R(z,θ) =====")
    svals = (0, 1, 2)

    # (A1) imaginary-axis norm preservation: sup_ν ||R(iν,θ)|-1|
    θ_axis = (0.0, 1.0, 5.0, 20.0, 50.0, 100.0)
    imax = Dict{Tuple{Int,Float64},Float64}()
    for s in svals, θ in θ_axis
        m = 0.0
        for ν in range(-3θ - 50, 3θ + 50, length = 6001)
            m = max(m, abs(abs(R_filon_scalar(im * ν, θ, s)) - 1))
        end
        imax[(s, θ)] = m
        @printf("  s=%d θ=%6.1f  sup_ν ||R(iν)|-1| = %.3e\n", s, θ, m)
    end

    # (A2) A-stability / L-stability: |R| in the left half-plane & at -∞.
    println("  -- left/right half-plane classification (θ=20) & |R(-X)| as X→∞ --")
    lhp = Dict{Int,NTuple{2,Float64}}()   # (max|R| over Re z<0 grid, min|R| over Re z>0 grid)
    linf = Dict{Int,Vector{Float64}}()
    for s in svals
        maxL = 0.0; minR = Inf
        for x in range(-30, 30, length = 121), y in range(-60, 60, length = 241)
            r = abs(R_filon_scalar(complex(x, y), 20.0, s))
            if x < -1e-9
                maxL = max(maxL, r)
            elseif x > 1e-9
                minR = min(minR, r)
            end
        end
        lhp[s] = (maxL, minR)
        linf[s] = [abs(R_filon_scalar(complex(-X, 0.0), 20.0, s)) for X in (10.0, 100.0, 1e3, 1e6)]
        @printf("  s=%d  max|R|(Re z<0)=%.4f  min|R|(Re z>0)=%.4f  |R(-X)|→ %s\n",
                s, maxL, minR, string(round.(linf[s]; digits = 4)))
    end

    # (A3) region maps |R(z,θ)| for plotting.
    xs = range(-25, 10, length = 281)
    ys = range(-25, 75, length = 401)
    region = Dict{Tuple{Int,Float64},Matrix{Float64}}()
    for s in svals, θ in (0.0, 20.0)
        Z = [abs(R_filon_scalar(complex(x, y), θ, s)) for y in ys, x in xs]
        region[(s, θ)] = Z
    end

    return (; svals, θ_axis, imax, lhp, linf, xs = collect(xs), ys = collect(ys), region)
end

# =============================================================================
# Experiment B — scalar variable coefficient u' = iω0(1+ε cos Ωt) u
# =============================================================================
function experiment_B()
    println("\n===== Experiment B: scalar variable-coefficient growth =====")
    ω0 = 1.0; Ω = 0.25; T = 200.0
    coarse = geom(0.03, 12.0, 44)           # ω0·Δt (steps-per-ansatz-period = 2π/coarse)
    εvals = (0.0, 0.25, 0.5, 1.0)
    svals = (0, 1, 2)
    ψ0 = SVector{1,ComplexF64}(1.0)

    # growth factor G = max_n |u_n| over [0,T]; true ≡ 1.
    Gp = Dict{Tuple{Int,Float64},Vector{Float64}}()   # plain:      (s, ε) -> vec over coarse
    Gc = Dict{Tuple{Int,Float64},Vector{Float64}}()   # controlled: (s, ε) -> vec over coarse
    for ε in εvals
        plain, controlled = scalar_var_cos(ω0, ε, Ω)
        freqs = SVector(ω0)
        for s in svals
            gp = Float64[]; gc = Float64[]
            for c in coarse
                Δt = c / ω0; nsteps = max(round(Int, T / Δt), 2)
                push!(gp, maximum(norm_traj(plain,      freqs, ψ0, Δt, nsteps, s; controlled = false)))
                push!(gc, maximum(norm_traj(controlled, freqs, ψ0, Δt, nsteps, s; controlled = true)))
            end
            Gp[(s, ε)] = gp; Gc[(s, ε)] = gc
            @printf("  ε=%.2f s=%d  plain max-G=%.2e  controlled max-G=%.2e\n",
                    ε, s, maximum(gp), maximum(gc))
        end
    end

    # one representative trajectory (coarse, deep modulation) to show the shape.
    ε = 0.6; c = 2.0; Δt = c / ω0; nsteps = max(round(Int, T / Δt), 2)
    plain, controlled = scalar_var_cos(ω0, ε, Ω)
    traj = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    for s in svals
        traj[(:plain, s)]      = norm_traj(plain,      SVector(ω0), ψ0, Δt, nsteps, s; controlled = false)
        traj[(:controlled, s)] = norm_traj(controlled, SVector(ω0), ψ0, Δt, nsteps, s; controlled = true)
    end
    tgrid = collect(0:nsteps) .* Δt

    return (; ω0, Ω, T, coarse = collect(coarse), εvals, svals, Gp, Gc,
              traj, tgrid, traj_params = (; ε, c, Δt, nsteps))
end

# =============================================================================
# Experiment C — non-normal 2×2 (CNOT3 analogue)
# =============================================================================
# Max over one carrier period (where the staggered one-step map varies) of the
# actual per-step amplification ‖M_n‖₂ and spectral radius ρ(M_n) (plain Filon),
# and the frozen (constant-A, midpoint) ‖M‖₂.  Decoupled from nsteps.
function step_norms_over_period(co, freqs, ω1, ω2, g, ωc, Δt, s)
    period = 2π / ωc
    σa = 0.0; ρa = 0.0; σf = 0.0
    for t_n in range(0, period, length = 60)
        M = step_matrix_plain(co, freqs, Δt, s, t_n, Val(2))
        σa = max(σa, opnorm(M, 2)); ρa = max(ρa, maximum(abs, eigvals(M)))
        Af = Afull_2lvl(ω1, ω2, g, ωc, t_n + Δt / 2)
        σf = max(σf, opnorm(step_matrix_frozen(Af, freqs, Δt, s), 2))
    end
    return σa, ρa, σf
end

function experiment_C()
    println("\n===== Experiment C: non-normal 2×2 (CNOT3 analogue) =====")
    ω1 = 5.0; ω2 = 4.0
    ψ0 = SVector{2,ComplexF64}(1 / sqrt(2), 1 / sqrt(2))
    svals = (0, 1, 2)
    Nstep = 200                              # fixed step count so coarse Δt still accrues growth

    # (C1) per-step amplification ‖M_n‖₂ vs coarseness (the mechanism, plain Filon):
    #      actual (staggered, non-commuting) vs frozen (constant-A → unitary).
    g1 = 0.5; ωc1 = 1.0
    co1, fr1 = twolevel_carrier(ω1, ω2, g1, ωc1)
    coarseM = geom(0.1, 100.0, 46)          # ω1·Δt up to ~CNOT3 coarse regime
    σA = Dict{Int,Vector{Float64}}(); ρA = Dict{Int,Vector{Float64}}(); σF = Dict{Int,Vector{Float64}}()
    for s in svals
        sa = Float64[]; ra = Float64[]; sf = Float64[]
        for c in coarseM
            a, r, f = step_norms_over_period(co1, fr1, ω1, ω2, g1, ωc1, c / ω1, s)
            push!(sa, a); push!(ra, r); push!(sf, f)
        end
        σA[s] = sa; ρA[s] = ra; σF[s] = sf
        @printf("  C1 s=%d  max‖M‖₂(actual)=%.3e  max‖M‖₂(frozen)=%.4f  (max ρ=%.4f)\n",
                s, maximum(sa), maximum(sf), maximum(ra))
    end

    # (C2) fixed-step-count growth G = max_n ‖ψ_n‖ vs coarseness, plain vs controlled.
    #      T = Nstep·Δt grows with Δt; true ‖ψ‖ ≡ 1 at every T, so G>1 is numerical.
    coarseG = geom(0.1, 60.0, 40)
    GpC = Dict{Int,Vector{Float64}}(); GcC = Dict{Int,Vector{Float64}}()
    for s in svals
        gp = Float64[]; gc = Float64[]
        for c in coarseG
            Δt = c / ω1
            push!(gp, maximum(norm_traj(co1, fr1, ψ0, Δt, Nstep, s; controlled = false)))
            push!(gc, maximum(norm_traj(co1, fr1, ψ0, Δt, Nstep, s; controlled = true)))
        end
        GpC[s] = gp; GcC[s] = gc
        @printf("  C2 s=%d  plain max-G=%.3e  controlled max-G=%.3e\n", s, maximum(gp), maximum(gc))
    end

    # (C3) coupling sweep at fixed coarse Δt (ω1·Δt = 20): does coupling drive it?
    c_fixed = 20.0; Δt3 = c_fixed / ω1
    gvals = geom(0.02, 8.0, 36)
    GpG = Dict{Int,Vector{Float64}}(); GcG = Dict{Int,Vector{Float64}}()
    for s in svals
        gp = Float64[]; gc = Float64[]
        for g in gvals
            co, fr = twolevel_carrier(ω1, ω2, g, ωc1)
            push!(gp, maximum(norm_traj(co, fr, ψ0, Δt3, Nstep, s; controlled = false)))
            push!(gc, maximum(norm_traj(co, fr, ψ0, Δt3, Nstep, s; controlled = true)))
        end
        GpG[s] = gp; GcG[s] = gc
        @printf("  C3 s=%d  plain max-G=%.3e  controlled max-G=%.3e\n", s, maximum(gp), maximum(gc))
    end

    # (C4) one representative coarse trajectory (ω1·Δt = 20, s=1) for a time series.
    s4 = 1; Δt4 = 20.0 / ω1
    trajC = Dict{Symbol,Vector{Float64}}(
        :plain      => norm_traj(co1, fr1, ψ0, Δt4, Nstep, s4; controlled = false),
        :controlled => norm_traj(co1, fr1, ψ0, Δt4, Nstep, s4; controlled = true))
    tgridC = collect(0:Nstep) .* Δt4
    @printf("  C4 s=%d ω1Δt=20  final ‖ψ‖: plain=%.3e controlled=%.3e\n",
            s4, trajC[:plain][end], trajC[:controlled][end])

    # (C5) realistic comparison: slowly *varying* coupling envelope, so controlled
    #      Filon is finite-order too.  G vs coarseness, plain vs controlled.
    Ωe = 0.3
    cov, frv = twolevel_varenv(ω1, ω2, g1, ωc1, Ωe)
    GpV = Dict{Int,Vector{Float64}}(); GcV = Dict{Int,Vector{Float64}}()
    for s in svals
        gp = Float64[]; gc = Float64[]
        for c in coarseG
            Δt = c / ω1
            push!(gp, maximum(norm_traj(cov, frv, ψ0, Δt, Nstep, s; controlled = false)))
            push!(gc, maximum(norm_traj(cov, frv, ψ0, Δt, Nstep, s; controlled = true)))
        end
        GpV[s] = gp; GcV[s] = gc
        @printf("  C5 s=%d (var env)  plain max-G=%.3e  controlled max-G=%.3e\n",
                s, maximum(gp), maximum(gc))
    end

    return (; ω1, ω2, svals, g1, ωc1, Nstep,
              coarseM = collect(coarseM), σA, ρA, σF,
              coarseG = collect(coarseG), GpC, GcC,
              c_fixed, gvals = collect(gvals), GpG, GcG,
              s4, trajC, tgridC, Ωe, GpV, GcV)
end

# -----------------------------------------------------------------------------
function main()
    A = experiment_A()
    B = experiment_B()
    C = experiment_C()
    outdir = joinpath(@__DIR__, "..", "..", "..", "..", "data", "stability")
    outdir = normpath(outdir)
    mkpath(outdir)
    path = joinpath(outdir, "stab_data.jls")
    serialize(path, (; A, B, C))
    println("\nSerialized → ", path)
    println("STAB_COMPUTE_DONE")
end
main()
