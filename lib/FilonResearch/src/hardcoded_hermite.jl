# =============================================================================
# Hard-coded Hermite method for linear ODEs  dψ/dt = A(t) ψ,  using the
# ControlledOperator interface.  This is the ω = 0 specialization of the
# hard-coded Filon method (Appendix A): choosing the ansatz frequencies ω = 0
# makes Ω = diag(ω) = 0, so the diagonal weight-phase matrices collapse to
# *scalar* weights and the envelope-derivative matrices F_m reduce to the plain
# state-derivative matrices D_m (D_0 = I, D_1 = A, D_2 = Ȧ + A², …).
#
# Each timestep advances ψ_n → ψ_{n+1} by solving  S^s_I ψ_{n+1} = S^s_E ψ_n
# with the explicit / implicit propagators
#
#   S^s_E = [ I + Σ_{j=0}^s Σ_{k=0}^j C(j,k) A^{(j-k)} w^s_{E,j} D_k ]_{t_n}
#   S^s_I = [ I - Σ_{j=0}^s Σ_{k=0}^j C(j,k) A^{(j-k)} w^s_{I,j} D_k ]_{t_{n+1}}
#
# where the scalar weights absorb the rescaling factor,
#
#   w^s_{E/I,j} = (Δt/2)^{j+1} b^{[-1,1],s}_{E/I,j}(0),
#
# i.e. `filon_weights(0, s, -1, 1)` scaled by (Δt/2)^{j+1}.  Expanded (with
# D_1 = A, D_2 = Ȧ + A²):
#
#   s=0:  S_E = I + w_{E,0} A
#   s=1:  S_E = I + w_{E,0} A + w_{E,1} Ȧ + w_{E,1} A²
#   s=2:  S_E = I + w_{E,0} A + w_{E,1} Ȧ + w_{E,2} Ä
#                 + w_{E,1} A² + 2 w_{E,2} Ȧ A + w_{E,2} A(Ȧ + A²)
#
# (S_I is the same with -w_{I,j}.)  This is exactly `hardcoded_filon.jl` with the
# per-component frequency data and the -iΩ terms removed; the structure (static
# SMatrix path + dynamic matrix-free/GMRES path, s ∈ {0,1,2}, allocation-free
# steps) and the shared helpers (`_matderivs`, `_opderivs`, `_FilonDynWS`,
# `_save_indices`, the stats hooks) are reused unchanged.  Because the Hermite
# method has no oscillatory ansatz, the public API takes no `frequencies`.
# =============================================================================

# -----------------------------------------------------------------------------
# Weight precompute (done once, reused across every timestep)
# -----------------------------------------------------------------------------

# Scalar weights w^s_{E,j}, w^s_{I,j} for j = 0 … S, as two NTuple{S+1} of
# ComplexF64.  Depends only on (s, Δt) — there are no frequencies.
@inline function _hermite_weight_entries(Δt, ::Val{S}) where {S}
    halfdt = Δt / 2
    wa, wb = filon_weights(0.0, S, -1, 1)               # ω = 0 ⇒ scalar weights
    WE = ntuple(jp -> ComplexF64(halfdt^jp * wa[jp]), Val(S + 1))
    WI = ntuple(jp -> ComplexF64(halfdt^jp * wb[jp]), Val(S + 1))
    return WE, WI
end

"""
    StaticHermiteWeights{S}

Precomputed scalar weights for the **static** hard-coded Hermite method of order
`s = S`.  Holds the weights `w^s_{E,j}`, `w^s_{I,j}` (`j = 0 … S`) as
`NTuple{S+1,ComplexF64}`, so a timestep is allocation-free.  Build with
[`hermite_weight_phases`](@ref).
"""
struct StaticHermiteWeights{S,WT}
    WE::WT          # NTuple{S+1,ComplexF64}
    WI::WT
end

"""
    DynamicHermiteWeights{S}

Precomputed scalar weights for the **dynamic** (matrix-free) hard-coded Hermite
method of order `s = S`.  Same scalar weights as [`StaticHermiteWeights`](@ref);
a distinct type so a step dispatches to the matrix-free / GMRES path.  Build with
[`hermite_weight_phases`](@ref).
"""
struct DynamicHermiteWeights{S,WT}
    WE::WT
    WI::WT
end

function _static_hermite_weights(::Val{S}, Δt) where {S}
    WE, WI = _hermite_weight_entries(Δt, Val(S))
    return StaticHermiteWeights{S,typeof(WE)}(WE, WI)
end

function _dynamic_hermite_weights(::Val{S}, Δt) where {S}
    WE, WI = _hermite_weight_entries(Δt, Val(S))
    return DynamicHermiteWeights{S,typeof(WE)}(WE, WI)
end

"""
    hermite_weight_phases(co, Δt, s; variant = :auto)

Precompute the scalar weights `w^s_{E,j}`, `w^s_{I,j}` shared by every timestep of
the hard-coded Hermite method of order `s ∈ {0,1,2}` for the controlled operator
`co` with stepsize `Δt`.  (Unlike [`filon_weight_phases`](@ref), there are no
ansatz frequencies — Hermite is the ω = 0 case.)

Returns a [`StaticHermiteWeights`](@ref) for the static layout (tuple of
`SMatrix`) or a [`DynamicHermiteWeights`](@ref) for the dynamic layout (vector of
matrices); `variant` may force either path.  Pass the result to
[`hermite_timestep_hardcoded`](@ref).
"""
function hermite_weight_phases(co::ControlledOperator, Δt::Real, s::Integer;
                               variant::Symbol = :auto)
    0 <= s <= 2 || throw(ArgumentError("hard-coded Hermite supports s ∈ {0,1,2}; got s=$s"))
    if _resolve_variant(co, variant) === :static
        return _static_hermite_weights(Val(Int(s)), Δt)
    else
        return _dynamic_hermite_weights(Val(Int(s)), Δt)
    end
end

# -----------------------------------------------------------------------------
# STATIC step — forms S_E, S_I as SMatrices and solves (reads like the header)
# -----------------------------------------------------------------------------

@inline function _hermite_step_static(An, Anp1, ψ, wp::StaticHermiteWeights{0})
    A = An[1]
    S_E = I + wp.WE[1] * A
    A = Anp1[1]
    S_I = I - wp.WI[1] * A
    return S_I \ (S_E * ψ)
end

@inline function _hermite_step_static(An, Anp1, ψ, wp::StaticHermiteWeights{1})
    A, dA = An
    S_E = I + wp.WE[1] * A + wp.WE[2] * dA + wp.WE[2] * A * A
    A, dA = Anp1
    S_I = I - wp.WI[1] * A - wp.WI[2] * dA - wp.WI[2] * A * A
    return S_I \ (S_E * ψ)
end

@inline function _hermite_step_static(An, Anp1, ψ, wp::StaticHermiteWeights{2})
    A, dA, ddA = An
    F2 = dA + A * A
    S_E = I + wp.WE[1] * A + wp.WE[2] * dA + wp.WE[3] * ddA +
              wp.WE[2] * A * A + 2 * wp.WE[3] * dA * A + wp.WE[3] * A * F2
    A, dA, ddA = Anp1
    F2 = dA + A * A
    S_I = I - wp.WI[1] * A - wp.WI[2] * dA - wp.WI[3] * ddA -
              wp.WI[2] * A * A - 2 * wp.WI[3] * dA * A - wp.WI[3] * A * F2
    return S_I \ (S_E * ψ)
end

# -----------------------------------------------------------------------------
# DYNAMIC step — matrix-free application of the propagators (mul! + GMRES).
# Mirrors `_apply_M!` (Filon) with Ω = 0 and scalar weights `W[j]`; `ws` is a
# reused `_FilonDynWS`.  `ws.v` holds A x (= D_1 x = F_1 x at Ω = 0); `ws.w`
# holds F_2 x = Ȧ x + A² x.
# -----------------------------------------------------------------------------

@inline function _apply_M_hermite!(out, x, ops, W, ws, ::Val{0})
    @. ws.buf = W[1] * x
    mul!(out, ops[1], ws.buf)                       # A (w_0 x)
    return out
end

@inline function _apply_M_hermite!(out, x, ops, W, ws, ::Val{1})
    A, dA = ops
    @. ws.buf = W[1] * x
    mul!(out, A, ws.buf)                            # A (w_0 x)
    @. ws.buf = W[2] * x
    mul!(out, dA, ws.buf, 1, 1)                     # + Ȧ (w_1 x)
    mul!(ws.v, A, x)                                # v = A x
    @. ws.buf = W[2] * ws.v
    mul!(out, A, ws.buf, 1, 1)                       # + A w_1 (A x)
    return out
end

@inline function _apply_M_hermite!(out, x, ops, W, ws, ::Val{2})
    A, dA, ddA = ops
    @. ws.buf = W[1] * x
    mul!(out, A, ws.buf)                            # A (w_0 x)
    @. ws.buf = W[2] * x
    mul!(out, dA, ws.buf, 1, 1)                     # + Ȧ (w_1 x)
    mul!(ws.v, A, x)                                # v = A x
    @. ws.buf = W[2] * ws.v
    mul!(out, A, ws.buf, 1, 1)                       # + A w_1 (A x)
    @. ws.buf = W[3] * x
    mul!(out, ddA, ws.buf, 1, 1)                    # + Ä (w_2 x)
    @. ws.buf = W[3] * ws.v                          # ws.v still holds A x
    mul!(out, dA, ws.buf, 2, 1)                     # + 2 Ȧ w_2 (A x)
    mul!(ws.w, A, ws.v)                             # w = A(A x) = A² x
    mul!(ws.w, dA, x, 1, 1)                          # + Ȧ x  ⇒ w = F_2 x
    @. ws.buf = W[3] * ws.w
    mul!(out, A, ws.buf, 1, 1)                       # + A w_2 F_2 x
    return out
end

# Callable wrapped once in a LinearMap: x ↦ S_I x = x - M^s_I x.  `ops` is
# refreshed in place each timestep, so the same LinearMap and GMRES workspace are
# reused (mirrors `_ImplicitApply`).
mutable struct _ImplicitApplyHermite{S,OT,WT,WS}
    ops::OT
    W::WT
    ws::WS
end

@inline function (ia::_ImplicitApplyHermite{S})(out, x) where {S}
    _apply_M_hermite!(out, x, ia.ops, ia.W, ia.ws, Val(S))
    @. out = x - out
    return out
end

# -----------------------------------------------------------------------------
# Timestepping drivers
# -----------------------------------------------------------------------------

function _solve_static_hermite(co, ψ0, Δt, nsteps, vs::Val, save_every,
                               save_final_only, stats)
    return _solve_static_hermite(co, ψ0, Δt, nsteps, vs, save_every,
                                 save_final_only, stats, Val(_nstate(co)))
end

function _solve_static_hermite(co, ψ0, Δt, nsteps, ::Val{S}, save_every,
                               save_final_only, stats, ::Val{N}) where {S,N}
    wp = _static_hermite_weights(Val(S), Δt)
    _stats_init!(stats, nsteps, false)
    ψ = SVector{N,ComplexF64}(ψ0)
    A_n = _matderivs(co, zero(Δt), Val(S))

    if save_final_only
        @inbounds for n in 1:nsteps
            t0 = _stats_tick(stats)
            A_np1 = _matderivs(co, n * Δt, Val(S))
            ψ = _hermite_step_static(A_n, A_np1, ψ, wp)
            A_n = A_np1
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
        A_np1 = _matderivs(co, n * Δt, Val(S))
        ψ = _hermite_step_static(A_n, A_np1, ψ, wp)
        A_n = A_np1
        _stats_record_static!(stats, t0)
        if col <= length(save_idx) && save_idx[col] == n
            history[:, col] .= ψ
            col += 1
        end
    end
    return history
end

function _solve_dynamic_hermite(co, ψ0, Δt, nsteps, ::Val{S}, save_every,
                                save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    wp = _dynamic_hermite_weights(Val(S), Δt)
    ws = _FilonDynWS(N)
    ψ = Vector{ComplexF64}(undef, N)
    ψ .= ψ0

    A_n = _opderivs(co, zero(Δt), Val(S))
    A_np1 = _opderivs(co, Δt, Val(S))               # placeholder (fixes the field type)
    ia = _ImplicitApplyHermite{S,typeof(A_np1),typeof(wp.WI),typeof(ws)}(
        A_np1, wp.WI, ws)
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    kws = Krylov.krylov_workspace(Val(:gmres), L, ws.rhs)
    _stats_init!(stats, nsteps, true)
    warned = false

    save_final_only || (save_idx = _save_indices(nsteps, save_every))
    save_final_only || (history = Matrix{ComplexF64}(undef, N, length(save_idx)))
    save_final_only || (history[:, 1] .= ψ)
    col = 2

    @inbounds for n in 1:nsteps
        t0 = _stats_tick(stats)
        A_np1 = _opderivs(co, n * Δt, Val(S))
        _apply_M_hermite!(ws.Mψ, ψ, A_n, wp.WE, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        ia.ops = A_np1
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        A_n = A_np1
        _stats_record_dynamic!(stats, t0, kws)
        if !warned && !Krylov.issolved(kws)
            warned = true
            @warn "GMRES did not converge at a Hermite timestep; continuing" step = n niters = Krylov.iteration_count(kws) atol rtol
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
    hermite_timestep_hardcoded(co, ψ, t_n, Δt, wp; atol, rtol) -> ψ_next

Advance the state `ψ` by one hard-coded Hermite step of size `Δt`, from `t_n` to
`t_n + Δt`, for the linear ODE `dψ/dt = A(t) ψ` with `A = co`.  `wp` is the
precomputed weight data from [`hermite_weight_phases`](@ref); its type selects
the order `s` and the static / dynamic implementation.

The static path (`wp::StaticHermiteWeights`) returns an `SVector` and allocates
nothing; the dynamic path (`wp::DynamicHermiteWeights`) returns a `Vector` and
uses GMRES (the `atol`/`rtol` keywords set its tolerances).

For many steps, prefer [`hermite_solve_hardcoded`](@ref), which reuses the GMRES
workspace and avoids re-evaluating `A` at shared step boundaries.
"""
function hermite_timestep_hardcoded(co::ControlledOperator, ψ, t_n::Real, Δt::Real,
                                    wp::StaticHermiteWeights{S}) where {S}
    A_n = _matderivs(co, t_n, Val(S))
    A_np1 = _matderivs(co, t_n + Δt, Val(S))
    return _hermite_step_static(A_n, A_np1, ψ, wp)
end

function hermite_timestep_hardcoded(co::ControlledOperator, ψ, t_n::Real, Δt::Real,
                                    wp::DynamicHermiteWeights{S};
                                    atol::Real = 1e-13, rtol::Real = 1e-13) where {S}
    N = length(ψ)
    ws = _FilonDynWS(N)
    A_n = _opderivs(co, t_n, Val(S))
    A_np1 = _opderivs(co, t_n + Δt, Val(S))
    _apply_M_hermite!(ws.Mψ, ψ, A_n, wp.WE, ws, Val(S))
    @. ws.rhs = ψ + ws.Mψ
    ia = _ImplicitApplyHermite{S,typeof(A_np1),typeof(wp.WI),typeof(ws)}(
        A_np1, wp.WI, ws)
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    sol, _ = Krylov.gmres(L, ws.rhs; atol = atol, rtol = rtol)
    return sol
end

"""
    hermite_solve_hardcoded(co, ψ0, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` (with `A = co`, a [`ControlledOperator`](@ref)) over
`nsteps` steps of fixed size `Δt`, starting from `ψ0`, using the hard-coded
Hermite method of order `s ∈ {0,1,2}` (orders 2/4/6).  This is the ω = 0 case of
[`filon_solve_hardcoded`](@ref); it takes no ansatz `frequencies`.

Returns, by default, an `N × nsaves` matrix whose columns are the state at the
saved times — the initial state, then every `save_every`-th step, then always
the final step.

# Keyword arguments
- `save_every::Integer = 1` — store the state every `save_every` steps.
- `save_final_only::Bool = false` — return just the final state vector.
- `variant::Symbol = :auto` — `:static` forms the propagator matrices and solves
  with `\\` (allocation-free for an `SMatrix`-backed `co`); `:dynamic` is
  matrix-free and solves with GMRES.  `:auto` chooses by the layout of `co`.
- `gmres_atol`, `gmres_rtol` — GMRES tolerances for the dynamic variant.
- `warm_start::Bool = false` — for the dynamic variant, seed each GMRES solve
  with the previous step's solution (ignored by the static variant).
- `stats::Union{Nothing,FilonSolveStats} = nothing` — pass a
  [`FilonSolveStats`](@ref) to collect per-step wall times and (dynamic variant
  only) GMRES iteration counts and convergence flags.
"""
function hermite_solve_hardcoded(co::ControlledOperator, ψ0::AbstractVector,
                                 Δt::Real, nsteps::Integer, s::Integer;
                                 save_every::Integer = 1,
                                 save_final_only::Bool = false, variant::Symbol = :auto,
                                 gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                                 warm_start::Bool = false,
                                 stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("hard-coded Hermite supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    if _resolve_variant(co, variant) === :static
        return _solve_static_hermite(co, ψ0, Δt, nsteps, Val(Int(s)),
                                     save_every, save_final_only, stats)
    else
        return _solve_dynamic_hermite(co, ψ0, Δt, nsteps, Val(Int(s)),
                                      save_every, save_final_only, warm_start,
                                      gmres_atol, gmres_rtol, stats)
    end
end
