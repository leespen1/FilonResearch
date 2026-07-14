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

struct _EffControlledHermiteWS{V<:AbstractVector,P,GT}
    Ax::V
    F2x::V
    bracket::V
    Akbk::V
    Mψ::V
    rhs::V
    Px::P       # nmat per-matrix products A_k x, retained so Ax and Adotx share them
    G::GT       # G[k][m+1]: per-step generator scalar (matrix k, order m), reused across GMRES iters
end
function _EffControlledHermiteWS(N::Integer, nmat::Integer, nord::Integer)
    return _EffControlledHermiteWS(ntuple(_ -> zeros(ComplexF64, N), Val(6))...,
                                   [zeros(ComplexF64, N) for _ in 1:nmat],
                                   [zeros(ComplexF64, nord) for _ in 1:nmat])
end

# Build the per-step generator scalars  G_{k,m} = Σ_{j=m}^s C(j,m) c_k^{(j-m)} w^s_j.
# Unlike the Filon generators (per-component diagonals), the Hermite weights w^s_j
# are scalar and there is one control per matrix, so each generator is a scalar.
# They do not depend on x, so building them once per step lets every GMRES iteration
# reuse them (and collapses the bracket from C(s+2,2) scalar·vector terms to s+1).
function _build_hermite_generators!(G, W, ed, ::Val{S}) where {S}
    @inbounds for k in eachindex(ed)
        Gk = G[k]; c = ed[k]
        for m in 0:S
            g = zero(ComplexF64)
            for j in m:S
                g += binomial(j, m) * c[j - m + 1] * W[j + 1]
            end
            Gk[m + 1] = g
        end
    end
    return G
end

# out ← M x = Σ_k A_k Σ_m [G^s_{k,m} F_m] x.  The per-step generator scalars
# G_{k,m} = ws.G[k][m+1] (built by `_build_hermite_generators!`) already fold the
# control-derivative × weight sum, so each apply just gathers them against the F_m x
# vectors and applies each A_k once.  `Aop`/`Adop` are the combined A(t), Adot(t)
# Operators (for the F_m vectors); `mats` the distinct matrices.
function _eff_hermite_apply_M!(out, x, mats, Aop, Adop, ws, ::Val{S}) where {S}
    fill!(out, zero(eltype(out)))
    if S == 1
        mul!(ws.Ax, Aop, x)                                      # F_1 x = A x  (Ω = 0)
    elseif S >= 2
        _apply_each!(ws.Px, mats, x)                             # Px[k] = A_k x  (one matvec each)
        _combine!(ws.Ax, ws.Px, Aop.coeffs)                     # Ax = Σ_k c_k A_k x
        mul!(ws.F2x, Aop, ws.Ax)                                 # A(A x)
        _combine_add!(ws.F2x, ws.Px, Adop.coeffs)               # + Adotx = Σ_k ċ_k A_k x  ⇒ F_2 x  (reuses Px)
    end
    @inbounds for k in eachindex(mats)
        # Ω = 0 here, so F_1 x = ws.Ax.  Hermite generators are scalars (not the
        # diagonal vectors of the Filon method), so the bracket is a scalar-weighted
        # combination — distinct from the Filon `_bracket_kernel!`.
        _hermite_bracket_kernel!(ws.bracket, ws.G[k], x, ws.Ax, ws.F2x, Val(S))
        mul!(ws.Akbk, mats[k], ws.bracket)
        _accum_kernel!(out, ws.Akbk)
    end
    return out
end

# Scalar-generator analogue of `_bracket_kernel!`: G[m] is a scalar weight, so the
# bracket is g·vector rather than (diagonal vector)·vector.
@inline function _hermite_bracket_kernel!(br, G, x, F1x, F2x, ::Val{S}) where {S}
    if S == 0
        g1 = G[1]
        @inbounds @simd for i in eachindex(br)
            br[i] = g1 * x[i]
        end
    elseif S == 1
        g1 = G[1]; g2 = G[2]
        @inbounds @simd for i in eachindex(br)
            br[i] = g1 * x[i] + g2 * F1x[i]
        end
    else
        g1 = G[1]; g2 = G[2]; g3 = G[3]
        @inbounds @simd for i in eachindex(br)
            br[i] = g1 * x[i] + g2 * F1x[i] + g3 * F2x[i]
        end
    end
    return br
end

# Callable applying x ↦ S_I x = x - M_I x, wrapped once in a LinearMap.  The
# implicit-side operators (Aop, Adop) and generators (ws.G) are refreshed in place
# each step, so the same LinearMap and GMRES workspace are reused.
mutable struct _EffControlledHermiteImplicit{S,MT,OT,WST}
    mats::MT
    Aop::OT
    Adop::OT
    ws::WST
end

@inline function (ia::_EffControlledHermiteImplicit{S})(out, x) where {S}
    _eff_hermite_apply_M!(out, x, ia.mats, ia.Aop, ia.Adop, ia.ws, Val(S))
    @. out = x - out
    return out
end

function _efficient_controlled_hermite_solve_dynamic(co, ψ0, Δt, nsteps, ::Val{S}, save_every,
                                                     save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    mats = co.matrices
    nmat = length(mats)
    wp = _efficient_dynamic_hermite_weights(Val(S), Δt)
    ws = _EffControlledHermiteWS(N, nmat, S + 1)

    # Per-step control-derivative buffers: one SVector{S+1} per matrix (feeds the
    # generator build; not read by the apply itself).  The left endpoint tₙ repeats
    # the previous step's right endpoint t_{n+1}, so `edL`/`edR` ping-pong between
    # steps and `AL`/`AdL` carry the left-endpoint operators (primed below at t = 0),
    # making the control evaluation run once per step instead of twice.
    edL = Vector{SVector{S + 1,ComplexF64}}(undef, nmat)
    edR = Vector{SVector{S + 1,ComplexF64}}(undef, nmat)

    ψ = Vector{ComplexF64}(undef, N); ψ .= ψ0

    # Reusable implicit operator + LinearMap + GMRES workspace (built once).
    Aop0 = evaluate(co, Δt, Derivative{0}())
    Adop0 = S >= 2 ? evaluate(co, Δt, Derivative{1}()) : Aop0
    ia = _EffControlledHermiteImplicit{S,typeof(mats),typeof(Aop0),typeof(ws)}(mats, Aop0, Adop0, ws)
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    kws = Krylov.krylov_workspace(Val(:gmres), L, ws.rhs)
    _stats_init!(stats, nsteps, true)
    warned = false

    save_final_only || (save_idx = _save_indices(nsteps, save_every))
    save_final_only || (history = Matrix{ComplexF64}(undef, N, length(save_idx)))
    save_final_only || (history[:, 1] .= ψ)
    col = 2

    # Prime the first step's left endpoint (t = 0): control derivatives and operators.
    _write_env_derivs!(edL, co.controls, zero(Δt), DerivativeUpTo{S}())
    AL = evaluate(co, zero(Δt), Derivative{0}())
    AdL = S >= 2 ? evaluate(co, zero(Δt), Derivative{1}()) : AL

    for n in 1:nsteps
        t0 = _stats_tick(stats)
        t_np1 = n * Δt
        # explicit side:  rhs = ψ + M_E ψ.  The left-endpoint control derivatives
        # (edL) and operators (AL, AdL) were realized as the previous step's right
        # endpoint, so only the WE-weighted generators are rebuilt here.
        _build_hermite_generators!(ws.G, wp.WE, edL, Val(S))
        _eff_hermite_apply_M!(ws.Mψ, ψ, mats, AL, AdL, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  realize the operators at t_{n+1}, rebuild the implicit
        # generators G_I (reused across GMRES iterations), then solve
        # (I - M_I) ψ_{n+1} = rhs matrix-free.
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        _write_env_derivs!(edR, co.controls, t_np1, DerivativeUpTo{S}())
        _build_hermite_generators!(ws.G, wp.WI, edR, Val(S))
        warm_start && Krylov.warm_start!(kws, ψ)
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

Pass `warm_start = true` to seed each GMRES solve with the previous step's
solution; it helps when GMRES takes several iterations (i.e. for systems larger
than a handful of states).

The keyword arguments (`save_every`, `save_final_only`, `warm_start`, `gmres_atol`,
`gmres_rtol`, `stats`) and the return value match [`hermite_solve_hardcoded`](@ref):
an `N × nsaves` history matrix, or just the final state if `save_final_only`.
"""
function efficient_controlled_hermite_solve(co::ControlledOperator, ψ0::AbstractVector,
                                            Δt::Real, nsteps::Integer, s::Integer;
                                            save_every::Integer = 1,
                                            save_final_only::Bool = false, warm_start::Bool = false,
                                            gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                                            stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("efficient controlled Hermite supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    return _efficient_controlled_hermite_solve_dynamic(co, ψ0, Δt, nsteps, Val(Int(s)),
                                                       save_every, save_final_only, warm_start,
                                                       gmres_atol, gmres_rtol, stats)
end
