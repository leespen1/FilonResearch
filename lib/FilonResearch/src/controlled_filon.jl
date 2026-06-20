# =============================================================================
# Controlled Filon method (Appendix B) for  dψ/dt = A(t) ψ  where
#
#       A(t) = Σ_{k=0}^{n_c} c̃_k(t) e^{i ω_{c,k} t} A_k ,
#
# i.e. each term is a constant matrix A_k times a control built from a slow
# *envelope* c̃_k and a *carrier wave* e^{i ω_{c,k} t}.  (The drift is the k=0
# term with c̃_0 = 1, ω_{c,0} = 0.)  Compared with the regular Filon method, each
# control term gets its OWN oscillatory ansatz: the quadrature for term k uses
# the *modified* frequencies ω + ω_{c,k}.
#
# Operators are expected to be `ControlledOperator`s whose controls are
# `CarrierControl`s (or ordinary controls, treated as zero-carrier).  The carrier
# frequency and envelope of each term are recovered through `carrier_frequency`
# and `envelope`.  The ansatz frequencies ω (one per state component) are passed
# to the solver, exactly as for the regular method.
#
# Each step solves  S^s_I ψ_{n+1} = S^s_E ψ_n  with (Appendix B)
#
#   S^s_* = I ± Σ_k φ_{*,k} A_k [ Σ_j Σ_m C(j,m) c̃_k^{(j-m)} W^s_{*,j,k} F_m ]
#
# evaluated at t_n (explicit, +) / t_{n+1} (implicit, −), where
#   φ_{*,k} = e^{i ω_{c,k} t̄_n},   t̄_n = t_n + Δt/2   (same midpoint phase both sides),
#   F_0 = I,  F_1 = A - iΩ,  F_2 = -Ω² - 2iΩA + Adot + A²   (A = full A(t), with carriers),
#   Ω = diag(ω) (ansatz frequencies), and the per-control weight-phase diagonals are
#   W^s_{*,j,k} = (Δt/2)^{j+1} diag( e^{∓i ω̂_m} b^{[-1,1],s}_{*,j}((ω_m + ω_{c,k}) Δt/2) ),
#   ω̂_m = ω_m Δt/2   (note: the PHASE uses the unmodified ω_m; the weight uses ω_m+ω_{c,k}).
#
# When every ω_{c,k} = 0 this reduces exactly to the regular Filon method
# (Appendix A) — that is the primary correctness check.
#
# As with the regular method there are two implementations:
#   * STATIC  — SMatrix-tuple operators; forms S_E, S_I and solves with `\`.
#   * DYNAMIC — Vector operators; matrix-free `mul!` + GMRES.
# =============================================================================

# The envelope-only operator: same matrices, controls replaced by their envelopes.
# `evaluate(envco, t, Derivative{j})` then yields the per-control envelope
# derivatives c̃_k^{(j)}(t) as its coefficient vector, while `evaluate(co, …)`
# (carriers intact) yields A⁽ᵐ⁾.
_envelope_operator(co::ControlledOperator) =
    ControlledOperator(map(envelope, co.controls), co.matrices)

# Per-control envelope derivative coefficients (c̃_k^{(0)}, c̃_k^{(1)}, c̃_k^{(2)}).
# Orders beyond S are not needed and are aliased to the 0-th (never read).
@inline function _envelope_coeffs(envco, t, ::Val{S}) where {S}
    c0 = evaluate(envco, t, Derivative{0}()).coeffs
    c1 = S >= 1 ? evaluate(envco, t, Derivative{1}()).coeffs : c0
    c2 = S >= 2 ? evaluate(envco, t, Derivative{2}()).coeffs : c0
    return (c0, c1, c2)
end

# -----------------------------------------------------------------------------
# Weight-phase precompute (one diagonal per (order j, control k))
# -----------------------------------------------------------------------------

# Diagonal entries of W^s_{E,j,k} and W^s_{I,j,k} for a single control's carrier
# frequency ωc.  The Hermite weight b uses the modified frequency ω+ωc; the phase
# e^{∓iω̂} uses the unmodified ω.  (ωc = 0 reproduces the regular weight.)
@inline function _carrier_weight_phase_entries(frequencies, ωc, Δt, ::Val{S}) where {S}
    halfdt = Δt / 2
    scaled = frequencies .* halfdt                       # ω̂_m  (phase)
    scaled_mod = (frequencies .+ ωc) .* halfdt           # (ω_m+ωc) Δt/2  (weight)
    wa, wb = filon_weights(scaled_mod, S, -1, 1)
    phaseE = cis.(scaled)
    phaseI = cis.(-scaled)
    WE = ntuple(jp -> (halfdt^jp) .* phaseE .* wa[jp], Val(S + 1))
    WI = ntuple(jp -> (halfdt^jp) .* phaseI .* wb[jp], Val(S + 1))
    return WE, WI
end

"""
    StaticControlledFilonWeights{S}

Precomputed data for the static controlled-Filon method of order `s = S`: per
control term, the order-`0…S` weight-phase diagonals (`Diagonal{…,SVector}`), the
ansatz `Ω`, and the carrier frequencies.  Built by [`controlled_filon_weights`](@ref).
"""
struct StaticControlledFilonWeights{S,WT,OT,FT}
    WE::WT          # NTuple{ncontrol} of NTuple{S+1} of Diagonal{ComplexF64,SVector{N}}
    WI::WT
    Ω::OT           # Diagonal{ComplexF64,SVector{N}}
    ωc::FT          # NTuple{ncontrol,Float64}
end

"""
    DynamicControlledFilonWeights{S}

Precomputed data for the dynamic (matrix-free) controlled-Filon method: per
control term, the weight-phase diagonals as plain complex vectors; plus the
ansatz frequency vector and the carrier frequencies.
"""
struct DynamicControlledFilonWeights{S,WT,FT}
    WE::WT          # NTuple{ncontrol} of NTuple{S+1} of Vector{ComplexF64}
    WI::WT
    freqs::Vector{Float64}
    ωc::FT          # Vector{Float64}, one per control term
end

function _static_controlled_weights(co, frequencies, Δt, ::Val{S}, ::Val{N}) where {S,N}
    toD(v) = Diagonal(SVector{N,ComplexF64}(v))
    pairs = map(co.controls) do ctrl
        WEk, WIk = _carrier_weight_phase_entries(frequencies, carrier_frequency(ctrl), Δt, Val(S))
        (map(toD, WEk), map(toD, WIk))
    end
    WE = map(first, pairs)
    WI = map(last, pairs)
    Ω = Diagonal(SVector{N,ComplexF64}(ComplexF64.(frequencies)))
    ωc = map(c -> float(carrier_frequency(c)), co.controls)
    return StaticControlledFilonWeights{S,typeof(WE),typeof(Ω),typeof(ωc)}(WE, WI, Ω, ωc)
end

function _dynamic_controlled_weights(co, frequencies, Δt, ::Val{S}) where {S}
    pairs = map(co.controls) do ctrl
        WEk, WIk = _carrier_weight_phase_entries(frequencies, carrier_frequency(ctrl), Δt, Val(S))
        (map(v -> convert(Vector{ComplexF64}, v), WEk),
         map(v -> convert(Vector{ComplexF64}, v), WIk))
    end
    WE = map(first, pairs)
    WI = map(last, pairs)
    freqs = collect(float.(frequencies))
    ωc = Float64[float(carrier_frequency(c)) for c in co.controls]
    return DynamicControlledFilonWeights{S,typeof(WE),typeof(ωc)}(WE, WI, freqs, ωc)
end

"""
    controlled_filon_weights(co, frequencies, Δt, s; variant = :auto)

Precompute the per-control weight-phase data for the controlled-Filon method of
order `s ∈ {0,1,2}`, for the controlled operator `co` (whose controls carry the
carrier frequencies) with ansatz `frequencies` and stepsize `Δt`.  Returns a
[`StaticControlledFilonWeights`](@ref) or [`DynamicControlledFilonWeights`](@ref).
"""
function controlled_filon_weights(co::ControlledOperator, frequencies::AbstractVector,
                                  Δt::Real, s::Integer; variant::Symbol = :auto)
    0 <= s <= 2 || throw(ArgumentError("controlled Filon supports s ∈ {0,1,2}; got s=$s"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    if _resolve_variant(co, variant) === :static
        return _static_controlled_weights(co, frequencies, Δt, Val(Int(s)), Val(_nstate(co)))
    else
        return _dynamic_controlled_weights(co, frequencies, Δt, Val(Int(s)))
    end
end

# -----------------------------------------------------------------------------
# STATIC — accumulate  Σ_k φ_k A_k bracket_k  (so S_* = I ± acc), then solve
# -----------------------------------------------------------------------------

@inline function _controlled_acc_static(co, envco, t, W, Ω, ωc, mid, ::Val{0}, ::Val{N}) where {N}
    c0, _, _ = _envelope_coeffs(envco, t, Val(0))
    mats = co.matrices
    acc = zero(SMatrix{N,N,ComplexF64})
    @inbounds for k in eachindex(mats)
        acc += cis(ωc[k] * mid) * c0[k] * (mats[k] * W[k][1])
    end
    return acc
end

@inline function _controlled_acc_static(co, envco, t, W, Ω, ωc, mid, ::Val{1}, ::Val{N}) where {N}
    A = materialize(evaluate(co, t, Derivative{0}()))
    F1 = A - im * Ω
    c0, c1, _ = _envelope_coeffs(envco, t, Val(1))
    mats = co.matrices
    acc = zero(SMatrix{N,N,ComplexF64})
    @inbounds for k in eachindex(mats)
        Wk0, Wk1 = W[k][1], W[k][2]
        bracket = c0[k] * Wk0 + c1[k] * Wk1 + c0[k] * (Wk1 * F1)
        acc += cis(ωc[k] * mid) * (mats[k] * bracket)
    end
    return acc
end

@inline function _controlled_acc_static(co, envco, t, W, Ω, ωc, mid, ::Val{2}, ::Val{N}) where {N}
    A = materialize(evaluate(co, t, Derivative{0}()))
    Ad = materialize(evaluate(co, t, Derivative{1}()))
    F1 = A - im * Ω
    F2 = Ad + A^2 - Ω^2 - 2im * (Ω * A)
    c0, c1, c2 = _envelope_coeffs(envco, t, Val(2))
    mats = co.matrices
    acc = zero(SMatrix{N,N,ComplexF64})
    @inbounds for k in eachindex(mats)
        Wk0, Wk1, Wk2 = W[k][1], W[k][2], W[k][3]
        bracket = c0[k] * Wk0 + c1[k] * Wk1 + c0[k] * (Wk1 * F1) +
                  c2[k] * Wk2 + 2 * c1[k] * (Wk2 * F1) + c0[k] * (Wk2 * F2)
        acc += cis(ωc[k] * mid) * (mats[k] * bracket)
    end
    return acc
end

@inline function _controlled_step_static(co, envco, ψ, t_n, Δt, wp::StaticControlledFilonWeights{S},
                                          ::Val{N}) where {S,N}
    mid = t_n + Δt / 2
    accE = _controlled_acc_static(co, envco, t_n,      wp.WE, wp.Ω, wp.ωc, mid, Val(S), Val(N))
    accI = _controlled_acc_static(co, envco, t_n + Δt, wp.WI, wp.Ω, wp.ωc, mid, Val(S), Val(N))
    return (I - accI) \ ((I + accE) * ψ)
end

# -----------------------------------------------------------------------------
# DYNAMIC — matrix-free application of  M_*  (so S_* x = x ± M_* x)
# -----------------------------------------------------------------------------

struct _ControlledDynWS{V<:AbstractVector,P<:AbstractVector}
    Ax::V
    F1x::V
    F2x::V
    t1::V
    t2::V
    bk::V
    Akbk::V
    Mψ::V
    rhs::V
    φ::P        # length ncontrol
end
_ControlledDynWS(N::Integer, ncontrol::Integer) = _ControlledDynWS(
    (zeros(ComplexF64, N) for _ in 1:9)..., zeros(ComplexF64, ncontrol))

# out ← M x = Σ_k φ_k A_k bracket_k x.  `Aop`/`Adop` are A(t), Adot(t) as Operators;
# `mats` are the constant A_k; `W` the per-control weight vectors; `cc` the envelope
# coefficients (c0,c1,c2); `φ` the per-control midpoint phases; `freqs` the ansatz ω.
function _controlled_apply_M!(out, x, mats, Aop, Adop, W, cc, φ, freqs, ws, ::Val{S}) where {S}
    fill!(out, zero(eltype(out)))
    if S >= 1
        mul!(ws.Ax, Aop, x)
        @. ws.F1x = ws.Ax - im * freqs * x                       # (A - iΩ) x
    end
    if S >= 2
        mul!(ws.t1, Aop, ws.Ax)                                  # A(Ax)
        mul!(ws.t2, Adop, x)                                     # Adot x
        @. ws.F2x = ws.t2 + ws.t1 - (freqs^2) * x - 2im * freqs * ws.Ax
    end
    c0, c1, c2 = cc
    @inbounds for k in eachindex(mats)
        Wk = W[k]
        if S == 0
            @. ws.bk = c0[k] * Wk[1] * x
        elseif S == 1
            @. ws.bk = c0[k] * Wk[1] * x + c1[k] * Wk[2] * x + c0[k] * Wk[2] * ws.F1x
        else
            @. ws.bk = c0[k] * Wk[1] * x + c1[k] * Wk[2] * x + c0[k] * Wk[2] * ws.F1x +
                       c2[k] * Wk[3] * x + 2 * c1[k] * Wk[3] * ws.F1x + c0[k] * Wk[3] * ws.F2x
        end
        mul!(ws.Akbk, mats[k], ws.bk)
        @. out += φ[k] * ws.Akbk
    end
    return out
end

# Callable applying x ↦ S_I x = x - M_I x, wrapped once in a LinearMap.  The
# implicit-side operators (Aop, Adop) and envelope coefficients (c0,c1,c2) are
# refreshed in place each step (their types are fixed across steps), so the same
# LinearMap and GMRES workspace are reused — the loop allocates nothing.  The
# per-control phases live in `ws.φ` (also refreshed in place each step).
mutable struct _ControlledImplicit{S,MT,OT,WT,CT,FT,WST}
    mats::MT
    Aop::OT
    Adop::OT
    W::WT
    c0::CT
    c1::CT
    c2::CT
    freqs::FT
    ws::WST
end

@inline function (ia::_ControlledImplicit{S})(out, x) where {S}
    _controlled_apply_M!(out, x, ia.mats, ia.Aop, ia.Adop, ia.W,
                         (ia.c0, ia.c1, ia.c2), ia.ws.φ, ia.freqs, ia.ws, Val(S))
    @. out = x - out
    return out
end

function _controlled_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                                   save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    ncontrol = length(co.matrices)
    envco = _envelope_operator(co)
    wp = _dynamic_controlled_weights(co, frequencies, Δt, Val(S))
    ws = _ControlledDynWS(N, ncontrol)
    mats = co.matrices
    freqs = wp.freqs
    ψ = Vector{ComplexF64}(undef, N); ψ .= ψ0

    # Reusable implicit operator + LinearMap + GMRES workspace (built once).
    Aop0 = evaluate(co, Δt, Derivative{0}())
    Adop0 = S >= 2 ? evaluate(co, Δt, Derivative{1}()) : Aop0
    c00, c10, c20 = _envelope_coeffs(envco, Δt, Val(S))
    ia = _ControlledImplicit{S,typeof(mats),typeof(Aop0),typeof(wp.WI),typeof(c00),
                             typeof(freqs),typeof(ws)}(mats, Aop0, Adop0, wp.WI, c00, c10, c20,
                                                       freqs, ws)
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    kws = Krylov.krylov_workspace(Val(:gmres), L, ws.rhs)
    _stats_init!(stats, nsteps, true)
    warned = false

    save_final_only || (save_idx = _save_indices(nsteps, save_every))
    save_final_only || (history = Matrix{ComplexF64}(undef, N, length(save_idx)))
    save_final_only || (history[:, 1] .= ψ)
    col = 2

    for n in 1:nsteps
        t0 = _stats_tick(stats)
        t_n = (n - 1) * Δt; t_np1 = n * Δt; mid = t_n + Δt / 2
        @. ws.φ = cis(wp.ωc * mid)
        # explicit side:  rhs = ψ + M_E ψ   (applied to the state, in place)
        AE = evaluate(co, t_n, Derivative{0}())
        AdE = S >= 2 ? evaluate(co, t_n, Derivative{1}()) : AE
        ccE = _envelope_coeffs(envco, t_n, Val(S))
        _controlled_apply_M!(ws.Mψ, ψ, mats, AE, AdE, wp.WE, ccE, ws.φ, freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  refresh ia, solve (I - M_I) ψ_{n+1} = rhs matrix-free
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        c0I, c1I, c2I = _envelope_coeffs(envco, t_np1, Val(S))
        ia.c0 = c0I; ia.c1 = c1I; ia.c2 = c2I
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        _stats_record_dynamic!(stats, t0, kws)
        if !warned && !Krylov.issolved(kws)
            warned = true
            @warn "GMRES did not converge at a Filon timestep; continuing" step = n niters = Krylov.iteration_count(kws) atol rtol
        end
        if !save_final_only && col <= length(save_idx) && save_idx[col] == n
            history[:, col] .= ψ
            col += 1
        end
    end
    return save_final_only ? ψ : history
end

# -----------------------------------------------------------------------------
# STATIC driver
# -----------------------------------------------------------------------------

function _controlled_solve_static(co, ψ0, frequencies, Δt, nsteps, vs::Val, save_every,
                                  save_final_only, stats)
    return _controlled_solve_static(co, ψ0, frequencies, Δt, nsteps, vs, save_every,
                                    save_final_only, stats, Val(_nstate(co)))
end

function _controlled_solve_static(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                                  save_final_only, stats, ::Val{N}) where {S,N}
    envco = _envelope_operator(co)
    wp = _static_controlled_weights(co, frequencies, Δt, Val(S), Val(N))
    _stats_init!(stats, nsteps, false)
    ψ = SVector{N,ComplexF64}(ψ0)

    if save_final_only
        @inbounds for n in 1:nsteps
            t0 = _stats_tick(stats)
            ψ = _controlled_step_static(co, envco, ψ, (n - 1) * Δt, Δt, wp, Val(N))
            _stats_record_static!(stats, t0)
        end
        return Vector(ψ)
    end

    save_idx = _save_indices(nsteps, save_every)
    history = Matrix{ComplexF64}(undef, N, length(save_idx))
    history[:, 1] .= ψ
    col = 2
    @inbounds for n in 1:nsteps
        t0 = _stats_tick(stats)
        ψ = _controlled_step_static(co, envco, ψ, (n - 1) * Δt, Δt, wp, Val(N))
        _stats_record_static!(stats, t0)
        if col <= length(save_idx) && save_idx[col] == n
            history[:, col] .= ψ
            col += 1
        end
    end
    return history
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    controlled_filon_solve(co, ψ0, frequencies, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` with the **controlled** Filon method of order `s ∈ {0,1,2}`,
where `co` is a [`ControlledOperator`](@ref) whose controls carry per-term carrier
frequencies (use [`CarrierControl`](@ref); ordinary controls are treated as
zero-carrier).  `frequencies` are the ansatz frequencies ω (one per state
component); each control term `k` is integrated with the modified ansatz
`ω + ω_{c,k}`.

Pass `warm_start = true` (dynamic variant only) to seed each GMRES solve with the
previous step's solution; it helps when GMRES takes several iterations.

The keyword arguments (`save_every`, `save_final_only`, `warm_start`, `variant`,
`gmres_atol`, `gmres_rtol`, `stats`) and the return value match
[`filon_solve_hardcoded`](@ref): an `N × nsaves` history matrix, or just the final
state if `save_final_only`.

When every carrier frequency is zero this method coincides with
[`filon_solve_hardcoded`](@ref) (regular Filon).
"""
function controlled_filon_solve(co::ControlledOperator, ψ0::AbstractVector,
                                frequencies::AbstractVector, Δt::Real, nsteps::Integer,
                                s::Integer; save_every::Integer = 1,
                                save_final_only::Bool = false, warm_start::Bool = false,
                                variant::Symbol = :auto,
                                gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                                stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("controlled Filon supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    if _resolve_variant(co, variant) === :static
        return _controlled_solve_static(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                                        save_every, save_final_only, stats)
    else
        return _controlled_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                                         save_every, save_final_only, warm_start, gmres_atol, gmres_rtol,
                                         stats)
    end
end
