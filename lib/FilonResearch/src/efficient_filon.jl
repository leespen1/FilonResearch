# =============================================================================
# Efficient Filon method for  dψ/dt = A(t) ψ,  A(t) = Σ_k c_k(t) A_k.
#
# This is the A_k-factored (Appendix B) reorganization of the *regular* Filon
# method (`filon_solve_hardcoded`), applied WITHOUT carrier resolution: the
# quadrature is split over each control matrix A_k and the time-derivatives are
# moved onto the scalar controls, so each distinct A_k is applied only s+1 times
# per step instead of the (s+1)(s+2)/2 applications of the regular method.  The
# ansatz frequencies ω enter only through the diagonal weight-phase matrices
# W^s_j(ω) and the envelope-derivative matrices F_m — each control c_k(t) (any
# carrier folded into the envelope) is differentiated as a generic scalar.
#
# Because each A_k is constant, this is the SAME approximation as
# `filon_solve_hardcoded` on the same operator — only the matrix-vector ordering
# differs — but with fewer dense products.  This is the ω ≠ 0 generalization of
# `efficient_controlled_hermite_solve` (which is its scalar-weight ω = 0 case).
#
# It is DISTINCT from two similarly named things:
#   * `filon_solve_hardcoded(; efficient=true)` reorders the *full-operator*
#     apply (Appendix A); it still applies the combined A(t) and its derivatives
#     (s+1)(s+2)/2 times and so costs (s+1)(s+2)/2·nHam dense matvecs — it does
#     NOT factor the individual A_k.
#   * `efficient_controlled_filon_solve` additionally RESOLVES each carrier into
#     frequency-shifted weights W^s_j(ω + ν); that is a different (carrier-aware)
#     approximation, not identical to regular Filon.
#
# Each step solves  S^s_I ψ_{n+1} = S^s_E ψ_n  with (Filon specialization of
# Appendix B, gathering the derivatives onto the scalar control)
#
#   S^s_* = I ± Σ_k A_k Σ_{m=0}^s [ G^s_{*,k,m} F_m ]_t ,
#   G^s_{*,k,m}(t) = Σ_{j=m}^s C(j,m) c_k^{(j-m)}(t) W^s_{*,j} ,
#
# where W^s_{*,j} = (Δt/2)^{j+1} diag( e^{∓i ω̂_m} b^{[-1,1],s}_{*,j}(ω_m Δt/2) )
# are the diagonal Filon weight-phase matrices (one set, shared across matrices —
# contrast the per-carrier weights of `efficient_controlled_filon_solve`) and
# * = E (explicit, +, at t_n) / I (implicit, −, at t_{n+1}).  The F_m ψ are
# computed once and reused:
#   F_0 = I,  F_1 = A - iΩ,  F_2 = Adot + A² - Ω² - 2iΩA   (A = full A(t), Ω = diag(ω)),
# building them applies each A_k once to ψ^{(0)},…,ψ^{(s-1)} (the per-matrix
# products A_k x are retained so Ax and Adotx share them), and each A_k is applied
# once more in the outer sum — s+1 applications in total.
#
# Only the dynamic (matrix-free `mul!` + GMRES) implementation is provided.  The
# runtime apply machinery (`_EffControlledDynWS`, `_eff_apply_M!`,
# `_EffControlledImplicit`, `_write_env_derivs!`) and the diagonal weight builder
# `_weight_phase_entries` are shared with the efficient-controlled-Filon and
# hard-coded-Filon files; this file must be `include`d after both.
# =============================================================================

# -----------------------------------------------------------------------------
# Weight precompute (one shared diagonal Filon weight set, νc = 0)
# -----------------------------------------------------------------------------

"""
    DynamicEfficientFilonWeights{S}

Precomputed diagonal weights for the efficient Filon method of order `s = S`: the
order-`0…S` weight-phase diagonals `W^s_{E,j}`, `W^s_{I,j}` as
`NTuple{S+1,Vector{ComplexF64}}` (one set, shared across all matrices), and the
ansatz frequency vector.  Built by [`efficient_filon_weights`](@ref).
"""
struct DynamicEfficientFilonWeights{S,WT}
    WE::WT          # NTuple{S+1,Vector{ComplexF64}}
    WI::WT
    freqs::Vector{Float64}
end

function _efficient_filon_weights(co::ControlledOperator, frequencies, Δt, ::Val{S}) where {S}
    WE, WI = _weight_phase_entries(frequencies, Δt, Val(S))
    WEc = map(v -> convert(Vector{ComplexF64}, v), WE)
    WIc = map(v -> convert(Vector{ComplexF64}, v), WI)
    freqs = collect(float.(frequencies))
    return DynamicEfficientFilonWeights{S,typeof(WEc)}(WEc, WIc, freqs)
end

"""
    efficient_filon_weights(co, frequencies, Δt, s)

Precompute the diagonal weight-phase data for the efficient Filon method of order
`s ∈ {0,1,2}`, for the controlled operator `co` (one control per matrix, carriers
folded into the envelope) with ansatz `frequencies` ω and stepsize `Δt`.  Returns
a [`DynamicEfficientFilonWeights`](@ref).
"""
function efficient_filon_weights(co::ControlledOperator, frequencies::AbstractVector,
                                 Δt::Real, s::Integer)
    0 <= s <= 2 || throw(ArgumentError("efficient Filon supports s ∈ {0,1,2}; got s=$s"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    return _efficient_filon_weights(co, frequencies, Δt, Val(Int(s)))
end

# -----------------------------------------------------------------------------
# DYNAMIC — matrix-free application of  M_*  (so S_* x = x ± M_* x)
# -----------------------------------------------------------------------------

# Build the per-step generator diagonals  G[k][m+1] = Σ_{j=m}^S C(j,m) c_k^{(j-m)} W^s_j.
# Like `_build_hermite_generators!` but with the (frequency-dependent) diagonal
# Filon weights, so each generator is a length-N vector accumulated by `axpy!`.
# Unlike `_build_generators!` (efficient controlled Filon) there is one control
# per matrix and no carrier ranges or midpoint phases — the carrier, if any, is
# already folded into the control and differentiated as a generic scalar.  The
# weights `W` are the single shared set (not indexed by carrier).  These do not
# depend on x, so building them once per step lets every GMRES iteration reuse
# them (and collapses the bracket from C(s+2,2) diagonal·vector terms to s+1).
function _build_efficient_filon_generators!(G, W, ed, ::Val{S}) where {S}
    @inbounds for k in eachindex(ed)
        Gk = G[k]; c = ed[k]
        for m in 0:S
            fill!(Gk[m + 1], zero(ComplexF64))
        end
        for m in 0:S, j in m:S
            axpy!(binomial(j, m) * c[j - m + 1], W[j + 1], Gk[m + 1])
        end
    end
    return G
end

function _efficient_filon_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                                        save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    mats = co.matrices
    nmat = length(mats)
    wp = _efficient_filon_weights(co, frequencies, Δt, Val(S))
    freqs = wp.freqs
    ws = _EffControlledDynWS(N, nmat, S + 1)

    # Per-step control-derivative buffer: one SVector{S+1} per matrix (feeds the
    # generator build; not read by the apply itself).
    edbuf = Vector{SVector{S + 1,ComplexF64}}(undef, nmat)

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

    for n in 1:nsteps
        t0 = _stats_tick(stats)
        t_n = (n - 1) * Δt; t_np1 = n * Δt
        # explicit side:  rhs = ψ + M_E ψ
        AE = evaluate(co, t_n, Derivative{0}())
        AdE = S >= 2 ? evaluate(co, t_n, Derivative{1}()) : AE
        _write_env_derivs!(edbuf, co.controls, t_n, DerivativeUpTo{S}())
        _build_efficient_filon_generators!(ws.G, wp.WE, edbuf, Val(S))
        _eff_apply_M!(ws.Mψ, ψ, mats, AE, AdE, freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # implicit side:  refresh the operators at t_{n+1}, rebuild the implicit
        # generators G_I (reused across GMRES iterations), then solve
        # (I - M_I) ψ_{n+1} = rhs matrix-free.
        ia.Aop = evaluate(co, t_np1, Derivative{0}())
        ia.Adop = S >= 2 ? evaluate(co, t_np1, Derivative{1}()) : ia.Aop
        _write_env_derivs!(edbuf, co.controls, t_np1, DerivativeUpTo{S}())
        _build_efficient_filon_generators!(ws.G, wp.WI, edbuf, Val(S))
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        _stats_record_dynamic!(stats, t0, kws)
        if !warned && !Krylov.issolved(kws)
            warned = true
            @warn "GMRES did not converge at an efficient-Filon timestep; continuing" step = n niters = Krylov.iteration_count(kws) atol rtol
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
    efficient_filon_solve(co, ψ0, frequencies, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` with the **efficient Filon** method of order `s ∈ {0,1,2}`
(orders 2/4/6), where `co` is a [`ControlledOperator`](@ref) `A(t) = Σ_k c_k(t) A_k`
with one matrix per control Hamiltonian and `frequencies` are the ansatz
frequencies ω (one per state component).

This is the A_k-factored (Appendix B) reorganization of regular Filon: the
quadrature is split over each distinct matrix `A_k` and the derivatives are moved
onto the scalar controls, so each `A_k` is applied only `s+1` times per step
instead of `(s+1)(s+2)/2`.  The result is identical (to round-off) to
[`filon_solve_hardcoded`](@ref) on the same operator — only the matrix-vector
ordering differs — but uses fewer dense products.  Only the matrix-free (GMRES)
implementation is provided.

Any carrier in `c_k(t)` is folded into the control and differentiated as a
generic scalar; ω enters only through the diagonal weights.  This makes the method
**distinct** from:
  * `filon_solve_hardcoded(; efficient=true)`, which reorders the full-operator
    apply but still costs `(s+1)(s+2)/2·nHam` dense matvecs (it does not factor
    the individual `A_k`); and
  * [`efficient_controlled_filon_solve`](@ref), which instead **resolves** each
    carrier into shifted weights `W^s_j(ω + ν)` — a different approximation.
The ω = 0 scalar-weight specialization is [`efficient_controlled_hermite_solve`](@ref).

Pass `warm_start = true` to seed each GMRES solve with the previous step's
solution; it helps when GMRES takes several iterations (i.e. for systems larger
than a handful of states).

The keyword arguments (`save_every`, `save_final_only`, `warm_start`, `gmres_atol`,
`gmres_rtol`, `stats`) and the return value match [`filon_solve_hardcoded`](@ref):
an `N × nsaves` history matrix, or just the final state if `save_final_only`.
"""
function efficient_filon_solve(co::ControlledOperator, ψ0::AbstractVector,
                               frequencies::AbstractVector, Δt::Real, nsteps::Integer,
                               s::Integer; save_every::Integer = 1,
                               save_final_only::Bool = false, warm_start::Bool = false,
                               gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                               stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("efficient Filon supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one ansatz frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    return _efficient_filon_solve_dynamic(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                                          save_every, save_final_only, warm_start, gmres_atol,
                                          gmres_rtol, stats)
end
