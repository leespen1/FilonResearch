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

struct _EffControlledDynWS{V<:AbstractVector,P}
    Ax::V
    F1x::V
    F2x::V
    t1::V
    t2::V
    bracket::V
    Akbk::V
    Mψ::V
    rhs::V
    Px::P       # nmat per-matrix products A_k x, retained so Ax and Adotx share them
end
function _EffControlledDynWS(N::Integer, nmat::Integer)
    return _EffControlledDynWS(ntuple(_ -> zeros(ComplexF64, N), Val(9))...,
                               [zeros(ComplexF64, N) for _ in 1:nmat])
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

# out ← M x = Σ_k A_k Σ_m [G^s_{k,m} F_m] x, with the carrier sum folded into the
# per-matrix bracket *before* the single matvec with A_k.  `Aop`/`Adop` are the
# combined A(t), Adot(t) Operators (each matrix applied once for the F_m vectors);
# `mats` the distinct matrices; `W` the per-carrier weight vectors; `ranges` group
# carriers by matrix; `ed[l]` holds the carrier-l envelope derivatives
# (c̃^{(0)},…,c̃^{(S)}); `φ[l] = e^{i ν_l t̄}` the carrier midpoint phase; `freqs` = ω.
function _eff_apply_M!(out, x, mats, Aop, Adop, W, ranges, ed, φ, freqs, ws, ::Val{S}) where {S}
    fill!(out, zero(eltype(out)))
    if S == 1
        mul!(ws.Ax, Aop, x)
        @. ws.F1x = ws.Ax - im * freqs * x                       # F_1 x = (A - iΩ) x
    elseif S >= 2
        _apply_each!(ws.Px, mats, x)                             # Px[:,k] = A_k x  (one matvec each)
        _combine!(ws.Ax, ws.Px, Aop.coeffs)                     # Ax = Σ_k c_k A_k x
        @. ws.F1x = ws.Ax - im * freqs * x                       # F_1 x = (A - iΩ) x
        mul!(ws.t1, Aop, ws.Ax)                                  # A(Ax)
        _combine!(ws.t2, ws.Px, Adop.coeffs)                     # Adotx = Σ_k ċ_k A_k x  (reuses Px)
        @. ws.F2x = ws.t2 + ws.t1 - (freqs^2) * x - 2im * freqs * ws.Ax
    end
    @inbounds for k in eachindex(mats)
        fill!(ws.bracket, zero(eltype(ws.bracket)))
        for l in ranges[k]
            Wl = W[l]
            e = ed[l]
            p = φ[l]
            if S == 0
                @. ws.bracket += p * (e[1] * Wl[1] * x)
            elseif S == 1
                @. ws.bracket += p * (e[1] * Wl[1] * x + e[2] * Wl[2] * x +
                                      e[1] * Wl[2] * ws.F1x)
            else
                @. ws.bracket += p * (e[1] * Wl[1] * x + e[2] * Wl[2] * x +
                                      e[1] * Wl[2] * ws.F1x + e[3] * Wl[3] * x +
                                      2 * e[2] * Wl[3] * ws.F1x + e[1] * Wl[3] * ws.F2x)
            end
        end
        mul!(ws.Akbk, mats[k], ws.bracket)
        @. out += ws.Akbk
    end
    return out
end

# Callable applying x ↦ S_I x = x - M_I x, wrapped once in a LinearMap.  The
# implicit-side operators (Aop, Adop), envelope derivatives (ed) and phases (φ)
# are refreshed in place each step (their types are fixed across steps), so the
# same LinearMap and GMRES workspace are reused — the loop allocates nothing.
mutable struct _EffControlledImplicit{S,MT,OT,WT,RT,ET,PT,FT,WST}
    mats::MT
    Aop::OT
    Adop::OT
    W::WT
    ranges::RT
    ed::ET
    φ::PT
    freqs::FT
    ws::WST
end

@inline function (ia::_EffControlledImplicit{S})(out, x) where {S}
    _eff_apply_M!(out, x, ia.mats, ia.Aop, ia.Adop, ia.W, ia.ranges, ia.ed, ia.φ,
                  ia.freqs, ia.ws, Val(S))
    @. out = x - out
    return out
end

function _efficient_controlled_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, ::Val{S},
                                             save_every, save_final_only, atol, rtol,
                                             stats) where {S}
    N = _nstate(co)
    mats = co.matrices
    wp = _efficient_dynamic_weights(co, frequencies, Δt, Val(S))
    ws = _EffControlledDynWS(N, length(mats))
    freqs = wp.freqs

    # Flattened carrier envelopes and their per-step derivative buffer.
    env_tuple = _flatten_envelopes(co.controls)
    ncarrier = length(env_tuple)
    edbuf = Vector{SVector{S + 1,ComplexF64}}(undef, ncarrier)
    φbuf = Vector{ComplexF64}(undef, ncarrier)

    ψ = Vector{ComplexF64}(undef, N); ψ .= ψ0

    # Reusable implicit operator + LinearMap + GMRES workspace (built once).
    Aop0 = evaluate(co, Δt, Derivative{0}())
    Adop0 = S >= 2 ? evaluate(co, Δt, Derivative{1}()) : Aop0
    ia = _EffControlledImplicit{S,typeof(mats),typeof(Aop0),typeof(wp.WI),typeof(wp.ranges),
                                typeof(edbuf),typeof(φbuf),typeof(freqs),typeof(ws)}(
        mats, Aop0, Adop0, wp.WI, wp.ranges, edbuf, φbuf, freqs, ws)
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
        @. φbuf = cis(wp.ν * mid)                       # shared midpoint phase, both sides
        # explicit side:  rhs = ψ + M_E ψ
        AE = evaluate(co, t_n, Derivative{0}())
        AdE = S >= 2 ? evaluate(co, t_n, Derivative{1}()) : AE
        _write_env_derivs!(edbuf, env_tuple, t_n, DerivativeUpTo{S}())
        _eff_apply_M!(ws.Mψ, ψ, mats, AE, AdE, wp.WE, wp.ranges, edbuf, φbuf, freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  refresh ia (operators + envelope derivatives at t_{n+1}),
        # then solve (I - M_I) ψ_{n+1} = rhs matrix-free.  edbuf === ia.ed.
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        _write_env_derivs!(edbuf, env_tuple, t_np1, DerivativeUpTo{S}())
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

The keyword arguments (`save_every`, `save_final_only`, `gmres_atol`,
`gmres_rtol`, `stats`) and the return value match [`controlled_filon_solve`](@ref):
an `N × nsaves` history matrix, or just the final state if `save_final_only`.

When every carrier frequency is zero this method coincides with
[`filon_solve_hardcoded`](@ref) (regular Filon).
"""
function efficient_controlled_filon_solve(co::ControlledOperator, ψ0::AbstractVector,
                                          frequencies::AbstractVector, Δt::Real, nsteps::Integer,
                                          s::Integer; save_every::Integer = 1,
                                          save_final_only::Bool = false,
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
                                               save_every, save_final_only, gmres_atol,
                                               gmres_rtol, stats)
end
