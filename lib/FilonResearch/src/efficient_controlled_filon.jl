# =============================================================================
# Efficient (generator-form) Controlled Filon method (Appendix B) for
#
#       dψ/dt = A(t) ψ ,   A(t) = Σ_k A_k Σ_l c̃_{k,l}(t) e^{i ν_{k,l} t} ,
#
# where a single control matrix A_k may be driven by SEVERAL carrier waves
# (frequencies ν_{k,l}, envelopes c̃_{k,l}).  This is the same method as
# `controlled_filon_solve`, but reorganized into the efficient form of Appendix B:
# every carrier sharing a matrix A_k is gathered into a single diagonal generator
#
#   G^s_{*,k,m}(t) = Σ_l φ_{*,k,l}(t) Σ_{j=m}^s C(j,m) c̃_{k,l}^{(j-m)}(t) W^s_{*,j,k,l},
#
# so that the timestep matrices
#
#   S^s_* = I ± Σ_k A_k Σ_{m=0}^s [ G^s_{*,k,m} F_m ]_t           (* = E at t_n, I at t_{n+1})
#
# apply each *distinct* matrix A_k only once in the outer sum, plus s applications
# while building the shared F_m ψ (one onto each of ψ^{(0)},…,ψ^{(s-1)}; the
# per-matrix products A_k x are retained so Ax and Adotx share them) — s+1 in total.
# The F_m ψ are computed once and reused:
#   F_0 = I,  F_1 = A - iΩ,  F_2 = Adot + A² - Ω² - 2iΩA   (A = full A(t), Ω = diag(ω)),
#   φ_{*,k,l}(t) = e^{i ν_{k,l}(t ± Δt/2)} = e^{i ν_{k,l} t̄_n}  (same midpoint phase both sides),
#   W^s_{*,j,k,l} = (Δt/2)^{j+1} diag( e^{∓i ω̂_m} b^{[-1,1],s}_{*,j}((ω_m + ν_{k,l}) Δt/2) ).
#
# The number of dense matrix-vector products per step therefore scales with the
# number of distinct matrices A_k, NOT with the number of carriers — the whole
# point of the efficient form.  Contrast `controlled_filon_solve`, which keeps one
# matrix per carrier term and so applies each physical matrix once per carrier.
#
# The grouping is carried by the operator itself: `co.matrices[k]` is paired with
# `co.controls[k]`, a `SumControl` over that matrix's carriers (a single
# `CarrierControl`/`ConstantControl` is treated as a one-carrier group).  The
# `SumControl`'s `derivative` gives the combined coefficient for the F_m matvecs
# (each matrix applied once); its `components` give the per-carrier envelopes and
# frequencies for the generator G.
#
# Only the dynamic (matrix-free `mul!` + GMRES) implementation is provided, as
# this is the path that matters for large controlled systems (e.g. CNOT3).  When
# every carrier frequency is zero this reduces to the regular Filon method.
# =============================================================================

# Flatten the controls' carriers into a single tuple of *envelopes*, compile-time
# unrolled so the (heterogeneous) result tuple is concretely typed.
@inline _flatten_envelopes(::Tuple{}) = ()
@inline _flatten_envelopes(controls::Tuple) =
    (map(envelope, components(first(controls)))..., _flatten_envelopes(Base.tail(controls))...)

# Carrier-index range of each control entry within the flattened carrier list.
function _carrier_ranges(controls)
    ranges = UnitRange{Int}[]
    g = 0
    for ctrl in controls
        lo = g + 1
        g += length(components(ctrl))
        push!(ranges, lo:g)
    end
    return ranges
end

# Write each envelope's order-0…S derivatives (an SVector{S+1}) into the
# preallocated buffer.  Compile-time-unrolled over the (heterogeneous) tuple, so
# the writes are type-stable and allocation-free.
@inline _write_env_derivs!(buf, controls::Tuple, t, d) = _wed!(buf, controls, t, d, 1)
@inline _wed!(buf, ::Tuple{}, t, d, k) = buf
@inline function _wed!(buf, controls::Tuple, t, d, k)
    @inbounds buf[k] = derivative(first(controls), t, d)
    return _wed!(buf, Base.tail(controls), t, d, k + 1)
end

# -----------------------------------------------------------------------------
# Weight-phase precompute (one diagonal per (order j, carrier))
# -----------------------------------------------------------------------------

"""
    DynamicEfficientControlledFilonWeights{S}

Precomputed data for the efficient (generator-form) controlled-Filon method of
order `s = S`: per *carrier*, the order-`0…S` weight-phase diagonals as plain
complex vectors; the carrier frequencies `ν`; the carrier-index ranges grouping
carriers under their shared matrix; and the ansatz frequency vector.  Built by
[`efficient_controlled_filon_weights`](@ref).
"""
struct DynamicEfficientControlledFilonWeights{S,WT}
    WE::WT                      # Vector{NTuple{S+1,Vector{ComplexF64}}}, one per carrier
    WI::WT
    ranges::Vector{UnitRange{Int}}   # carriers of matrix k are ranges[k]
    ν::Vector{Float64}          # carrier frequency of each carrier
    freqs::Vector{Float64}      # ansatz frequencies ω
end

function _efficient_dynamic_weights(co, frequencies, Δt, ::Val{S}) where {S}
    WE = NTuple{S + 1,Vector{ComplexF64}}[]
    WI = NTuple{S + 1,Vector{ComplexF64}}[]
    ν = Float64[]
    for ctrl in co.controls
        for c in components(ctrl)
            νc = float(carrier_frequency(c))
            WEk, WIk = _carrier_weight_phase_entries(frequencies, νc, Δt, Val(S))
            push!(WE, map(v -> convert(Vector{ComplexF64}, v), WEk))
            push!(WI, map(v -> convert(Vector{ComplexF64}, v), WIk))
            push!(ν, νc)
        end
    end
    ranges = _carrier_ranges(co.controls)
    freqs = collect(float.(frequencies))
    return DynamicEfficientControlledFilonWeights{S,typeof(WE)}(WE, WI, ranges, ν, freqs)
end

"""
    efficient_controlled_filon_weights(co, frequencies, Δt, s)

Precompute the per-carrier weight-phase data for the efficient controlled-Filon
method of order `s ∈ {0,1,2}`, for the grouped controlled operator `co` (each
matrix paired with a [`SumControl`](@ref) over its carriers) with ansatz
`frequencies` and stepsize `Δt`.  Returns a
[`DynamicEfficientControlledFilonWeights`](@ref).
"""
function efficient_controlled_filon_weights(co::ControlledOperator, frequencies::AbstractVector,
                                            Δt::Real, s::Integer)
    0 <= s <= 2 || throw(ArgumentError("efficient controlled Filon supports s ∈ {0,1,2}; got s=$s"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    return _efficient_dynamic_weights(co, frequencies, Δt, Val(Int(s)))
end

# -----------------------------------------------------------------------------
# DYNAMIC — matrix-free application of  M_*  (so S_* x = x ± M_* x)
# -----------------------------------------------------------------------------

struct _EffControlledDynWS{V<:AbstractVector,P,GT}
    Ax::V
    F1x::V
    F2x::V
    t1::V
    bracket::V
    Akbk::V
    Mψ::V
    rhs::V
    Px::P       # nmat per-matrix products A_k x, retained so Ax and Adotx share them
    G::GT       # G[k][m+1]: per-step generator diagonal (matrix k, order m), reused across GMRES iters
end
function _EffControlledDynWS(N::Integer, nmat::Integer, nord::Integer)
    return _EffControlledDynWS(ntuple(_ -> zeros(ComplexF64, N), Val(8))...,
                               [zeros(ComplexF64, N) for _ in 1:nmat],
                               [[zeros(ComplexF64, N) for _ in 1:nord] for _ in 1:nmat])
end

# Per-matrix products P[k] = mats[k]·x, retained so several combined operators
# sharing these matrices (here A and Adot) can be assembled by cheap coefficient
# gathers instead of re-applying every A_k.  This lets the s = 2 step apply each
# A_k only s+1 times (Appendix B), rather than recomputing A_k x when forming Adotx.
@inline function _apply_each!(P, mats, x)
    @inbounds for k in eachindex(mats)
        mul!(P[k], mats[k], x)
    end
    return P
end

# y ← y + Σ_k c[k]·P[k]  (combine retained per-matrix products with coefficients c)
@inline function _combine_add!(y, P, c)
    @inbounds for k in eachindex(c)
        axpy!(c[k], P[k], y)
    end
    return y
end

# y ← Σ_k c[k]·P[k]
@inline function _combine!(y, P, c)
    fill!(y, zero(eltype(y)))
    return _combine_add!(y, P, c)
end

# Build the per-step generator diagonals
#   G[k][m+1] = Σ_{l∈ranges[k]} φ_l Σ_{j=m}^S C(j,m) ed_l[j-m+1] W_l[j+1].
# These fold the whole carrier × derivative-order sum into one diagonal per (matrix,
# order) and depend only on the step time (envelope derivatives `ed`, midpoint phases
# `φ`), not on the Krylov vector.  Building them once per step/side lets each GMRES
# apply skip the carrier expansion, so the repeated diagonal work scales with the
# number of matrices rather than the number of carriers.
function _build_generators!(G, W, ranges, ed, φ, ::Val{S}) where {S}
    @inbounds for k in eachindex(ranges)
        Gk = G[k]
        for m in 0:S
            fill!(Gk[m + 1], zero(ComplexF64))
        end
        for l in ranges[k]
            Wl = W[l]; e = ed[l]; p = φ[l]
            for m in 0:S, j in m:S
                axpy!(p * binomial(j, m) * e[j - m + 1], Wl[j + 1], Gk[m + 1])
            end
        end
    end
    return G
end

# Hand-vectorized diagonal kernels.  At the qudit problem's N (a few hundred) the
# diagonal weight-phase work dominates the apply (the constant Hamiltonians are very
# sparse), and a generic complex broadcast spends much of its time in broadcast
# machinery rather than arithmetic.  A flat `@inbounds @simd` loop over the
# (loop-invariant-hoisted) diagonal vectors is ~2.5× faster on the fused bracket,
# which is the single hottest operation in the apply.  Each kernel computes exactly
# the broadcast it replaces, element by element with no cross-i reassociation.
@inline function _f1x_kernel!(F1x, Ax, freqs, x)            # F_1 x = (A - iΩ) x
    @inbounds @simd for i in eachindex(F1x)
        F1x[i] = Ax[i] - im * freqs[i] * x[i]
    end
    return F1x
end
@inline function _f2x_kernel!(F2x, ψ2, freqs, x, Ax)       # F_2 x = ψ⁽²⁾ - Ω²x - 2iΩ Ax,  ψ⁽²⁾ = Ȧx + A(Ax)
    @inbounds @simd for i in eachindex(F2x)
        F2x[i] = ψ2[i] - (freqs[i]^2) * x[i] - 2im * freqs[i] * Ax[i]
    end
    return F2x
end
@inline function _bracket_kernel!(br, G, x, F1x, F2x, ::Val{S}) where {S}
    if S == 0
        G1 = G[1]
        @inbounds @simd for i in eachindex(br)
            br[i] = G1[i] * x[i]
        end
    elseif S == 1
        G1 = G[1]; G2 = G[2]
        @inbounds @simd for i in eachindex(br)
            br[i] = G1[i] * x[i] + G2[i] * F1x[i]
        end
    else
        G1 = G[1]; G2 = G[2]; G3 = G[3]
        @inbounds @simd for i in eachindex(br)
            br[i] = G1[i] * x[i] + G2[i] * F1x[i] + G3[i] * F2x[i]
        end
    end
    return br
end
@inline function _accum_kernel!(out, y)
    @inbounds @simd for i in eachindex(out)
        out[i] += y[i]
    end
    return out
end

# out ← M x = Σ_k A_k Σ_m [G_{k,m} F_m] x.  The per-step generator diagonals
# G_{k,m} = ws.G[k][m+1] (built by `_build_generators!`) already fold the carrier ×
# order sum, so each apply just gathers them against the F_m x vectors and applies
# each A_k once.  `Aop`/`Adop` are the combined A(t), Adot(t) Operators (for the F_m
# vectors); `freqs` = ω.
function _eff_apply_M!(out, x, mats, Aop, Adop, freqs, ws, ::Val{S}) where {S}
    fill!(out, zero(eltype(out)))
    if S == 1
        mul!(ws.Ax, Aop, x)
        _f1x_kernel!(ws.F1x, ws.Ax, freqs, x)
    elseif S >= 2
        _apply_each!(ws.Px, mats, x)                             # Px[k] = A_k x  (one matvec each)
        _combine!(ws.Ax, ws.Px, Aop.coeffs)                     # Ax = Σ_k c_k A_k x
        _f1x_kernel!(ws.F1x, ws.Ax, freqs, x)
        mul!(ws.t1, Aop, ws.Ax)                                  # A(Ax)
        _combine_add!(ws.t1, ws.Px, Adop.coeffs)                 # + Adotx = Σ_k ċ_k A_k x  ⇒ ψ⁽²⁾  (reuses Px)
        _f2x_kernel!(ws.F2x, ws.t1, freqs, x, ws.Ax)
    end
    @inbounds for k in eachindex(mats)
        _bracket_kernel!(ws.bracket, ws.G[k], x, ws.F1x, ws.F2x, Val(S))
        mul!(ws.Akbk, mats[k], ws.bracket)
        _accum_kernel!(out, ws.Akbk)
    end
    return out
end

# Callable applying x ↦ S_I x = x - M_I x, wrapped once in a LinearMap.  The
# implicit-side operators (Aop, Adop) are refreshed in place each step, and the
# generator diagonals ws.G are rebuilt each step before the solve, so the same
# LinearMap and GMRES workspace are reused — the loop allocates nothing.
mutable struct _EffControlledImplicit{S,MT,OT,FT,WST}
    mats::MT
    Aop::OT
    Adop::OT
    freqs::FT
    ws::WST
end

@inline function (ia::_EffControlledImplicit{S})(out, x) where {S}
    _eff_apply_M!(out, x, ia.mats, ia.Aop, ia.Adop, ia.freqs, ia.ws, Val(S))
    @. out = x - out
    return out
end

function _efficient_controlled_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, ::Val{S},
                                             save_every, save_final_only, warm_start, atol, rtol,
                                             stats) where {S}
    N = _nstate(co)
    mats = co.matrices
    wp = _efficient_dynamic_weights(co, frequencies, Δt, Val(S))
    ws = _EffControlledDynWS(N, length(mats), S + 1)
    freqs = wp.freqs

    # Flattened carrier envelopes and their per-step derivative buffers.  Endpoint
    # quantities at tₙ are identical to the previous step's t_{n+1} (same time, same
    # values), so the envelope-derivative evaluation and operator realization run
    # once per step instead of twice: `edL`/`edR` ping-pong the envelope-derivative
    # buffers between steps and `AL`/`AdL` carry the left-endpoint operators, both
    # primed below at the first step's left endpoint t = 0.
    env_tuple = _flatten_envelopes(co.controls)
    ncarrier = length(env_tuple)
    edL = Vector{SVector{S + 1,ComplexF64}}(undef, ncarrier)
    edR = Vector{SVector{S + 1,ComplexF64}}(undef, ncarrier)
    φbuf = Vector{ComplexF64}(undef, ncarrier)

    ψ = Vector{ComplexF64}(undef, N); ψ .= ψ0

    # Reusable implicit operator + LinearMap + GMRES workspace (built once).
    Aop0 = evaluate(co, Δt, Derivative{0}())
    Adop0 = S >= 2 ? evaluate(co, Δt, Derivative{1}()) : Aop0
    ia = _EffControlledImplicit{S,typeof(mats),typeof(Aop0),typeof(freqs),typeof(ws)}(
        mats, Aop0, Adop0, freqs, ws)
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    kws = Krylov.krylov_workspace(Val(:gmres), L, ws.rhs)
    _stats_init!(stats, nsteps, true)
    warned = false

    save_final_only || (save_idx = _save_indices(nsteps, save_every))
    save_final_only || (history = Matrix{ComplexF64}(undef, N, length(save_idx)))
    save_final_only || (history[:, 1] .= ψ)
    col = 2

    # Prime the first step's left endpoint (t = 0): envelope derivatives and operators.
    _write_env_derivs!(edL, env_tuple, zero(Δt), DerivativeUpTo{S}())
    AL = evaluate(co, zero(Δt), Derivative{0}())
    AdL = S >= 2 ? evaluate(co, zero(Δt), Derivative{1}()) : AL

    for n in 1:nsteps
        t0 = _stats_tick(stats)
        t_np1 = n * Δt; mid = (n - 1) * Δt + Δt / 2
        @. φbuf = cis(wp.ν * mid)                       # shared midpoint phase, both sides
        # explicit side:  rhs = ψ + M_E ψ.  The left-endpoint envelope derivatives
        # (edL) and operators (AL, AdL) were realized as the previous step's right
        # endpoint; only the WE-weighted generators (which also depend on the
        # current midpoint phase) are rebuilt here.
        _build_generators!(ws.G, wp.WE, wp.ranges, edL, φbuf, Val(S))
        _eff_apply_M!(ws.Mψ, ψ, mats, AL, AdL, freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  realize the operators at t_{n+1}, rebuild the implicit
        # generators G_I (reused across GMRES iterations), then solve
        # (I - M_I) ψ_{n+1} = rhs matrix-free.
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        _write_env_derivs!(edR, env_tuple, t_np1, DerivativeUpTo{S}())
        _build_generators!(ws.G, wp.WI, wp.ranges, edR, φbuf, Val(S))
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        _stats_record_dynamic!(stats, t0, kws)
        if !warned && !Krylov.issolved(kws)
            warned = true
            @warn "GMRES did not converge at an efficient controlled-Filon timestep; continuing" step = n niters = Krylov.iteration_count(kws) atol rtol
        end
        if !save_final_only && col <= length(save_idx) && save_idx[col] == n
            history[:, col] .= ψ
            col += 1
        end
        # This step's right endpoint t_{n+1} is the next step's left endpoint tₙ.
        edL, edR = edR, edL
        AL = ia.Aop; AdL = ia.Adop
    end
    return save_final_only ? ψ : history
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    efficient_controlled_filon_solve(co, ψ0, frequencies, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` with the **efficient (generator-form)** controlled Filon
method of order `s ∈ {0,1,2}` (Appendix B), where `co` is a
[`ControlledOperator`](@ref) whose controls are [`SumControl`](@ref)s grouping
each matrix's carriers (a single [`CarrierControl`](@ref)/[`ConstantControl`](@ref)
is treated as a one-carrier group).  `frequencies` are the ansatz frequencies ω
(one per state component); carrier `l` of matrix `k` is integrated with the
modified ansatz `ω + ν_{k,l}`.

This computes the same approximation as [`controlled_filon_solve`](@ref), but
gathers every carrier sharing a matrix into a single diagonal generator, so the
dense matrix-vector product count scales with the number of distinct matrices and
**not** with the number of carriers.  Only the matrix-free (GMRES) implementation
is provided.

Pass `warm_start = true` to seed each GMRES solve with the previous step's
solution (a good guess since the state changes little per step); it pays off
when GMRES takes several iterations, which needs a system larger than a handful
of states (for very small `N`, GMRES already converges in `≤ N` iterations).

The keyword arguments (`save_every`, `save_final_only`, `warm_start`, `gmres_atol`,
`gmres_rtol`, `stats`) and the return value match [`controlled_filon_solve`](@ref):
an `N × nsaves` history matrix, or just the final state if `save_final_only`.

When every carrier frequency is zero this method coincides with
[`filon_solve_hardcoded`](@ref) (regular Filon).
"""
function efficient_controlled_filon_solve(co::ControlledOperator, ψ0::AbstractVector,
                                          frequencies::AbstractVector, Δt::Real, nsteps::Integer,
                                          s::Integer; save_every::Integer = 1,
                                          save_final_only::Bool = false, warm_start::Bool = false,
                                          gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                                          stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("efficient controlled Filon supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    return _efficient_controlled_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                                               save_every, save_final_only, warm_start, gmres_atol,
                                               gmres_rtol, stats)
end
