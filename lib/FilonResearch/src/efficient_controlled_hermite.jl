# =============================================================================
# Efficient controlled Hermite method for  dψ/dt = A(t) ψ,  A(t) = Σ_k c_k(t) A_k.
#
# This is the ω = 0 (Hermite) counterpart of the efficient controlled Filon method
# (`efficient_controlled_filon.jl`): the quadrature is split over each control
# matrix A_k and the time-derivatives are moved onto the scalar controls, so each
# distinct A_k is applied only s+1 times per step instead of the (s+1)(s+2)/2
# applications of the regular Hermite method (`hermite_solve_hardcoded`).
#
# Hermite is Filon with the ansatz frequencies ω = 0, so Ω = diag(ω) = 0, the
# weight-phase matrices collapse to SCALAR weights, and the envelope-derivative
# matrices reduce to F_0 = I, F_1 = A, F_2 = Adot + A²  (no -iΩ terms).  No frequency
# is considered: each control matrix is paired with its full control c_k(t) (any
# carrier is folded in and differentiated as a generic scalar), so the operator has
# one matrix per Hamiltonian and no carrier grouping is needed.
#
# Each step solves  S^s_I ψ_{n+1} = S^s_E ψ_n  with (Hermite specialization of
# Appendix B, gathering the derivatives onto the scalar control)
#
#   S^s_* = I ± Σ_k A_k Σ_{m=0}^s [ G^s_{*,k,m} F_m ]_t ,
#   G^s_{*,k,m}(t) = Σ_{j=m}^s C(j,m) c_k^{(j-m)}(t) w^s_{*,j} ,
#
# where w^s_{*,j} = (Δt/2)^{j+1} b^{[-1,1],s}_{*,j}(0) are the scalar Hermite
# weights and * = E (explicit, +, at t_n) / I (implicit, −, at t_{n+1}).  The F_m ψ
# are computed once and reused; building them applies each A_k once to ψ^{(0)},…,
# ψ^{(s-1)} (the per-matrix products A_k x are retained so Ax and Adotx share them),
# and each A_k is applied once more in the outer sum — s+1 applications in total.
#
# This computes exactly the same approximation as `hermite_solve_hardcoded` (the
# regular Hermite propagator over the full A(t)) — the two are algebraically
# identical because each A_k is constant — but with fewer dense matvecs.  Only the
# dynamic (matrix-free `mul!` + GMRES) implementation is provided.
# =============================================================================

# -----------------------------------------------------------------------------
# Weight precompute (scalar Hermite weights, reused from hardcoded_hermite.jl)
# -----------------------------------------------------------------------------

"""
    DynamicEfficientControlledHermiteWeights{S}

Precomputed scalar weights for the efficient controlled Hermite method of order
`s = S`: the order-`0…S` weights `w^s_{E,j}`, `w^s_{I,j}` as
`NTuple{S+1,ComplexF64}`.  Built by [`efficient_controlled_hermite_weights`](@ref).
"""
struct DynamicEfficientControlledHermiteWeights{S,WT}
    WE::WT          # NTuple{S+1,ComplexF64}
    WI::WT
end

function _efficient_dynamic_hermite_weights(::Val{S}, Δt) where {S}
    WE, WI = _hermite_weight_entries(Δt, Val(S))
    return DynamicEfficientControlledHermiteWeights{S,typeof(WE)}(WE, WI)
end

"""
    efficient_controlled_hermite_weights(co, Δt, s)

Precompute the scalar weights for the efficient controlled Hermite method of order
`s ∈ {0,1,2}` with stepsize `Δt`.  (Like [`hermite_weight_phases`](@ref), there are
no ansatz frequencies — Hermite is the ω = 0 case.)  Returns a
[`DynamicEfficientControlledHermiteWeights`](@ref).
"""
function efficient_controlled_hermite_weights(co::ControlledOperator, Δt::Real, s::Integer)
    0 <= s <= 2 || throw(ArgumentError("efficient controlled Hermite supports s ∈ {0,1,2}; got s=$s"))
    return _efficient_dynamic_hermite_weights(Val(Int(s)), Δt)
end

# -----------------------------------------------------------------------------
# DYNAMIC — matrix-free application of  M_*  (so S_* x = x ± M_* x)
# -----------------------------------------------------------------------------

struct _EffControlledHermiteWS{V<:AbstractVector,P}
    Ax::V
    F2x::V
    bracket::V
    Akbk::V
    Mψ::V
    rhs::V
    Px::P       # nmat per-matrix products A_k x, retained so Ax and Adotx share them
end
function _EffControlledHermiteWS(N::Integer, nmat::Integer)
    return _EffControlledHermiteWS(ntuple(_ -> zeros(ComplexF64, N), Val(6))...,
                                   [zeros(ComplexF64, N) for _ in 1:nmat])
end

# out ← M x = Σ_k A_k Σ_m [G^s_{k,m} F_m] x, with the derivatives on the scalar
# control gathered into the per-matrix bracket before the single matvec with A_k.
# `Aop`/`Adop` are the combined A(t), Adot(t) Operators (each matrix applied once for
# the F_m vectors); `mats` the distinct matrices; `W` the scalar weights; `ed[k]`
# holds the control-k derivatives (c_k^{(0)},…,c_k^{(S)}).
function _eff_hermite_apply_M!(out, x, mats, Aop, Adop, W, ed, ws, ::Val{S}) where {S}
    fill!(out, zero(eltype(out)))
    if S == 1
        mul!(ws.Ax, Aop, x)                                      # F_1 x = A x  (Ω = 0)
    elseif S >= 2
        _apply_each!(ws.Px, mats, x)                             # Px[:,k] = A_k x  (one matvec each)
        _combine!(ws.Ax, ws.Px, Aop.coeffs)                     # Ax = Σ_k c_k A_k x
        mul!(ws.F2x, Aop, ws.Ax)                                 # A(A x)
        _combine_add!(ws.F2x, ws.Px, Adop.coeffs)               # + Adotx = Σ_k ċ_k A_k x  ⇒ F_2 x  (reuses Px)
    end
    @inbounds for k in eachindex(mats)
        c = ed[k]
        if S == 0
            @. ws.bracket = c[1] * W[1] * x
        elseif S == 1
            @. ws.bracket = c[1] * W[1] * x + c[2] * W[2] * x + c[1] * W[2] * ws.Ax
        else
            @. ws.bracket = c[1] * W[1] * x + c[2] * W[2] * x + c[1] * W[2] * ws.Ax +
                            c[3] * W[3] * x + 2 * c[2] * W[3] * ws.Ax + c[1] * W[3] * ws.F2x
        end
        mul!(ws.Akbk, mats[k], ws.bracket)
        @. out += ws.Akbk
    end
    return out
end

# Callable applying x ↦ S_I x = x - M_I x, wrapped once in a LinearMap.  The
# implicit-side operators (Aop, Adop) and control derivatives (ed) are refreshed in
# place each step, so the same LinearMap and GMRES workspace are reused.
mutable struct _EffControlledHermiteImplicit{S,MT,OT,WT,ET,WST}
    mats::MT
    Aop::OT
    Adop::OT
    W::WT
    ed::ET
    ws::WST
end

@inline function (ia::_EffControlledHermiteImplicit{S})(out, x) where {S}
    _eff_hermite_apply_M!(out, x, ia.mats, ia.Aop, ia.Adop, ia.W, ia.ed, ia.ws, Val(S))
    @. out = x - out
    return out
end

function _efficient_controlled_hermite_solve_dynamic(co, ψ0, Δt, nsteps, ::Val{S}, save_every,
                                                     save_final_only, atol, rtol, stats) where {S}
    N = _nstate(co)
    mats = co.matrices
    nmat = length(mats)
    wp = _efficient_dynamic_hermite_weights(Val(S), Δt)
    ws = _EffControlledHermiteWS(N, nmat)

    # Per-step control-derivative buffer: one SVector{S+1} per matrix.
    edbuf = Vector{SVector{S + 1,ComplexF64}}(undef, nmat)

    ψ = Vector{ComplexF64}(undef, N); ψ .= ψ0

    # Reusable implicit operator + LinearMap + GMRES workspace (built once).
    Aop0 = evaluate(co, Δt, Derivative{0}())
    Adop0 = S >= 2 ? evaluate(co, Δt, Derivative{1}()) : Aop0
    ia = _EffControlledHermiteImplicit{S,typeof(mats),typeof(Aop0),typeof(wp.WI),
                                       typeof(edbuf),typeof(ws)}(mats, Aop0, Adop0, wp.WI, edbuf, ws)
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
        t_n = (n - 1) * Δt; t_np1 = n * Δt
        # explicit side:  rhs = ψ + M_E ψ
        AE = evaluate(co, t_n, Derivative{0}())
        AdE = S >= 2 ? evaluate(co, t_n, Derivative{1}()) : AE
        _write_env_derivs!(edbuf, co.controls, t_n, DerivativeUpTo{S}())
        _eff_hermite_apply_M!(ws.Mψ, ψ, mats, AE, AdE, wp.WE, edbuf, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  refresh ia (operators + control derivatives at t_{n+1}),
        # then solve (I - M_I) ψ_{n+1} = rhs matrix-free.  edbuf === ia.ed.
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        _write_env_derivs!(edbuf, co.controls, t_np1, DerivativeUpTo{S}())
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        _stats_record_dynamic!(stats, t0, kws)
        if !warned && !Krylov.issolved(kws)
            warned = true
            @warn "GMRES did not converge at an efficient controlled-Hermite timestep; continuing" step = n niters = Krylov.iteration_count(kws) atol rtol
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
    efficient_controlled_hermite_solve(co, ψ0, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` with the **efficient controlled Hermite** method of order
`s ∈ {0,1,2}` (orders 2/4/6), where `co` is a [`ControlledOperator`](@ref)
`A(t) = Σ_k c_k(t) A_k` with one matrix per control Hamiltonian.  This is the
ω = 0 counterpart of [`efficient_controlled_filon_solve`](@ref): the quadrature is
split over each `A_k` and the derivatives are moved onto the scalar controls, so
each distinct `A_k` is applied only a few times per step.  No frequency is
considered — any carrier in `c_k(t)` is folded into the control and differentiated
as a generic scalar; it takes no ansatz `frequencies`.

The result is identical (to round-off) to [`hermite_solve_hardcoded`](@ref) on the
same operator — only the matrix-vector ordering differs — but uses fewer dense
products.  Only the matrix-free (GMRES) implementation is provided.

The keyword arguments (`save_every`, `save_final_only`, `gmres_atol`,
`gmres_rtol`, `stats`) and the return value match [`hermite_solve_hardcoded`](@ref):
an `N × nsaves` history matrix, or just the final state if `save_final_only`.
"""
function efficient_controlled_hermite_solve(co::ControlledOperator, ψ0::AbstractVector,
                                            Δt::Real, nsteps::Integer, s::Integer;
                                            save_every::Integer = 1,
                                            save_final_only::Bool = false,
                                            gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                                            stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("efficient controlled Hermite supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    return _efficient_controlled_hermite_solve_dynamic(co, ψ0, Δt, nsteps, Val(Int(s)),
                                                       save_every, save_final_only, gmres_atol,
                                                       gmres_rtol, stats)
end
