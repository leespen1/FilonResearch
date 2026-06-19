# =============================================================================
# Hard-coded Filon method for linear ODEs  dψ/dt = A(t) ψ,  using the
# ControlledOperator interface.  This implements Appendix A of the project
# writeup for s = 0, 1, 2 (orders 2, 4, 6).
#
# Each timestep advances ψ_n → ψ_{n+1} by solving
#
#       S^s_I ψ_{n+1} = S^s_E ψ_n
#
# with the explicit / implicit propagators
#
#   S^s_E = [ I + Σ_{j=0}^s Σ_{k=0}^j C(j,k) A^{(j-k)} W^s_{E,j} F_k ]_{t_n}
#   S^s_I = [ I - Σ_{j=0}^s Σ_{k=0}^j C(j,k) A^{(j-k)} W^s_{I,j} F_k ]_{t_{n+1}}
#
# where, writing Ω = diag(ω) for the ansatz frequencies,
#
#   F_0 = I,   F_1 = A - iΩ,   F_2 = -Ω² - 2iΩA + Ȧ + A² ,
#
# and the diagonal *weight-phase* matrices (which absorb the rescaling factor
# and the ansatz phase R(t) = e^{-iΩt}) are
#
#   W^s_{E,j} = (Δt/2)^{j+1} diag( e^{+i ω̂_k} b^{[-1,1],s}_{E,j}(ω̂_k) ),
#   W^s_{I,j} = (Δt/2)^{j+1} diag( e^{-i ω̂_k} b^{[-1,1],s}_{I,j}(ω̂_k) ),
#   ω̂_k = ω_k Δt/2.
#
# Expanded explicitly (Appendix A), the propagators are
#
#   s=0:  S_E = I + A W_{E,0}
#         S_I = I - A W_{I,0}
#
#   s=1:  S_E = I + A W_{E,0} + Ȧ W_{E,1} + A W_{E,1}(A - iΩ)
#         S_I = I - A W_{I,0} - Ȧ W_{I,1} - A W_{I,1}(A - iΩ)
#
#   s=2:  S_E = I + A W_{E,0} + Ȧ W_{E,1} + Ä W_{E,2}
#                 + A W_{E,1}(A - iΩ) + 2 Ȧ W_{E,2}(A - iΩ) + A W_{E,2} F_2
#         S_I = I - A W_{I,0} - Ȧ W_{I,1} - Ä W_{I,2}
#                 - A W_{I,1}(A - iΩ) - 2 Ȧ W_{I,2}(A - iΩ) - A W_{I,2} F_2
#
# (The Appendix's S²_E prints W^2_{I,0} in its first term; that is a typo — the
# explicit propagator uses the explicit weight W^2_{E,0}, as used here.)
#
# Two implementations share the weight-phase precompute and the timestepping
# loop, differing only in how a single step is taken:
#
#   * STATIC ("readable") — for a ControlledOperator whose matrices are a Tuple
#     of SMatrix.  It materializes A, Ȧ, Ä into SMatrices and forms S_E, S_I
#     with `+`/`*` exactly as written above, then solves with `\`.  Every
#     operation is on StaticArrays, so a step allocates nothing.
#
#   * DYNAMIC (matrix-free) — for a ControlledOperator whose matrices are a
#     Vector.  It never forms S_E or S_I.  The explicit side is applied to the
#     state as a composition of non-allocating `mul!` matrix-vector products and
#     broadcast vector-adds; the implicit side is the same composition wrapped
#     in a LinearMap and solved with GMRES (Krylov.jl) using a reusable
#     workspace, so the inner loop is allocation-free.
#
# The frequencies ω are supplied separately (one per state component): a
# ControlledOperator describes the modulation A(t), not the oscillatory ansatz.
# =============================================================================

# Number of state components N (the matrix dimension).  For the static layout
# this is a compile-time constant (the SMatrix size); for the dynamic layout it
# is read from the first stored matrix.
@inline _nstate(co::ControlledOperator) = size(first(co.matrices), 1)

@inline function _resolve_variant(co::ControlledOperator, variant::Symbol)
    variant === :auto && return co.matrices isa Tuple ? :static : :dynamic
    variant in (:static, :dynamic) ||
        throw(ArgumentError("variant must be :auto, :static or :dynamic; got :$variant"))
    return variant
end

# -----------------------------------------------------------------------------
# Weight-phase precompute (done once, reused across every timestep)
# -----------------------------------------------------------------------------

# Diagonal entries of W^s_{E,j} and W^s_{I,j} for j = 0 … S, returned as two
# NTuple{S+1} of length-N complex vectors.  Depends only on (frequencies, s, Δt).
@inline function _weight_phase_entries(frequencies, Δt, ::Val{S}) where {S}
    halfdt = Δt / 2
    scaled = frequencies .* halfdt                      # ω̂_k = ω_k Δt/2
    wa, wb = filon_weights(scaled, S, -1, 1)            # wa: explicit (left), wb: implicit (right)
    phaseE = cis.(scaled)                               # e^{+i ω̂_k}
    phaseI = cis.(-scaled)                              # e^{-i ω̂_k}
    # jp = j + 1, so the rescaling factor (Δt/2)^{j+1} is halfdt^jp.
    WE = ntuple(jp -> (halfdt^jp) .* phaseE .* wa[jp], Val(S + 1))
    WI = ntuple(jp -> (halfdt^jp) .* phaseI .* wb[jp], Val(S + 1))
    return WE, WI
end

"""
    StaticFilonWeights{S}

Precomputed weight-phase data for the **static** hard-coded Filon method of
order `s = S`.  Holds the diagonal weight-phase matrices `W^s_{E,j}`, `W^s_{I,j}`
(`j = 0 … S`) and `Ω = diag(ω)` as StaticArrays `Diagonal{…,SVector}`, so a
timestep is allocation-free.  Build with [`filon_weight_phases`](@ref).
"""
struct StaticFilonWeights{S,WT,OT}
    WE::WT          # NTuple{S+1} of Diagonal{ComplexF64,SVector{N}}
    WI::WT
    Ω::OT           # Diagonal{ComplexF64,SVector{N}}
end

"""
    DynamicFilonWeights{S}

Precomputed weight-phase data for the **dynamic** (matrix-free) hard-coded Filon
method of order `s = S`.  Holds the weight-phase diagonals as plain length-N
complex vectors and the frequency vector, which the in-place matvecs apply by
broadcasting.  Build with [`filon_weight_phases`](@ref).
"""
struct DynamicFilonWeights{S,WT,FT}
    WE::WT          # NTuple{S+1} of Vector{ComplexF64}
    WI::WT
    freqs::FT       # Vector of the ansatz frequencies ω
end

# N threaded in as a Val so the SVector sizes are compile-time constants and the
# returned struct is concretely typed (this runs once, behind a function barrier).
function _static_weights(co, frequencies, Δt, ::Val{S}, ::Val{N}) where {S,N}
    WEv, WIv = _weight_phase_entries(frequencies, Δt, Val(S))
    toD = v -> Diagonal(SVector{N,ComplexF64}(v))
    WE = map(toD, WEv)
    WI = map(toD, WIv)
    Ω = Diagonal(SVector{N,ComplexF64}(ComplexF64.(frequencies)))
    return StaticFilonWeights{S,typeof(WE),typeof(Ω)}(WE, WI, Ω)
end

function _dynamic_weights(co, frequencies, Δt, ::Val{S}) where {S}
    WEv, WIv = _weight_phase_entries(frequencies, Δt, Val(S))
    WE = map(v -> convert(Vector{ComplexF64}, v), WEv)
    WI = map(v -> convert(Vector{ComplexF64}, v), WIv)
    freqs = collect(float.(frequencies))
    return DynamicFilonWeights{S,typeof(WE),typeof(freqs)}(WE, WI, freqs)
end

"""
    filon_weight_phases(co, frequencies, Δt, s; variant = :auto)

Precompute the weight-phase matrices `W^s_{E,j}`, `W^s_{I,j}` (and `Ω`) shared by
every timestep of the hard-coded Filon method of order `s ∈ {0,1,2}` for the
controlled operator `co` with the given ansatz `frequencies` and stepsize `Δt`.

Returns a [`StaticFilonWeights`](@ref) for the static layout (tuple of `SMatrix`)
or a [`DynamicFilonWeights`](@ref) for the dynamic layout (vector of matrices);
`variant` may force either path.  Pass the result to [`filon_timestep_hardcoded`](@ref).
"""
function filon_weight_phases(co::ControlledOperator, frequencies::AbstractVector,
                             Δt::Real, s::Integer; variant::Symbol = :auto)
    0 <= s <= 2 || throw(ArgumentError("hard-coded Filon supports s ∈ {0,1,2}; got s=$s"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    if _resolve_variant(co, variant) === :static
        return _static_weights(co, frequencies, Δt, Val(Int(s)), Val(_nstate(co)))
    else
        return _dynamic_weights(co, frequencies, Δt, Val(Int(s)))
    end
end

# -----------------------------------------------------------------------------
# Operator-derivative tuples  (A, Ȧ, …, A^{(S)})  at a single time
# -----------------------------------------------------------------------------

# Static: materialize each derivative into an SMatrix.  Compile-time-unrolled.
@inline _matderivs(co, t, ::Val{0}) = (materialize(evaluate(co, t, Derivative{0}())),)
@inline function _matderivs(co, t, ::Val{M}) where {M}
    return (_matderivs(co, t, Val(M - 1))..., materialize(evaluate(co, t, Derivative{M}())))
end

# Dynamic: keep the realized Operators (share co's matrices) for use with mul!.
@inline _opderivs(co, t, ::Val{0}) = (evaluate(co, t, Derivative{0}()),)
@inline function _opderivs(co, t, ::Val{M}) where {M}
    return (_opderivs(co, t, Val(M - 1))..., evaluate(co, t, Derivative{M}()))
end

# -----------------------------------------------------------------------------
# STATIC step — forms S_E, S_I as SMatrices and solves (reads like Appendix A)
# -----------------------------------------------------------------------------

@inline function _filon_step_static(An, Anp1, ψ, wp::StaticFilonWeights{0})
    A = An[1]
    S_E = I + A * wp.WE[1]
    A = Anp1[1]
    S_I = I - A * wp.WI[1]
    return S_I \ (S_E * ψ)
end

@inline function _filon_step_static(An, Anp1, ψ, wp::StaticFilonWeights{1})
    Ω = wp.Ω
    A, dA = An
    S_E = I + A * wp.WE[1] + dA * wp.WE[2] + A * wp.WE[2] * (A - im * Ω)
    A, dA = Anp1
    S_I = I - A * wp.WI[1] - dA * wp.WI[2] - A * wp.WI[2] * (A - im * Ω)
    return S_I \ (S_E * ψ)
end

@inline function _filon_step_static(An, Anp1, ψ, wp::StaticFilonWeights{2})
    Ω = wp.Ω
    A, dA, ddA = An
    F2 = dA + A^2 - Ω^2 - 2im * Ω * A
    S_E = I + A * wp.WE[1] + dA * wp.WE[2] + ddA * wp.WE[3] +
              A * wp.WE[2] * (A - im * Ω) + 2 * dA * wp.WE[3] * (A - im * Ω) +
              A * wp.WE[3] * F2
    A, dA, ddA = Anp1
    F2 = dA + A^2 - Ω^2 - 2im * Ω * A
    S_I = I - A * wp.WI[1] - dA * wp.WI[2] - ddA * wp.WI[3] -
              A * wp.WI[2] * (A - im * Ω) - 2 * dA * wp.WI[3] * (A - im * Ω) -
              A * wp.WI[3] * F2
    return S_I \ (S_E * ψ)
end

# -----------------------------------------------------------------------------
# DYNAMIC step — matrix-free application of the propagators (mul! + GMRES)
# -----------------------------------------------------------------------------

# Scratch buffers reused by every matvec and by the timestepping loop.
struct _FilonDynWS{V<:AbstractVector}
    buf::V      # holds (W_j .* something) before a mul!
    v::V        # holds (A - iΩ)x  /  A x
    w::V        # holds F_2 x
    Mψ::V       # holds (M_E ψ) when forming the RHS
    rhs::V      # the GMRES right-hand side  ψ + M_E ψ
end
_FilonDynWS(N::Integer) =
    _FilonDynWS(ntuple(_ -> zeros(ComplexF64, N), Val(5))...)

# out ← M^s x, the bracketed sum (without the leading I) of S_E / S_I, applied to
# x.  `ops = (A, [Ȧ, [Ä]])` are realized Operators; `W = (W_0, …, W_S)` are the
# weight-phase diagonals (as vectors); `freqs` is ω.  Uses only mul! and
# broadcast vector ops into the preallocated workspace, so it never allocates.
@inline function _apply_M!(out, x, ops, W, freqs, ws, ::Val{0})
    @. ws.buf = W[1] * x
    mul!(out, ops[1], ws.buf)                       # A (W_0 x)
    return out
end

@inline function _apply_M!(out, x, ops, W, freqs, ws, ::Val{1})
    A, dA = ops
    @. ws.buf = W[1] * x
    mul!(out, A, ws.buf)                            # A (W_0 x)
    @. ws.buf = W[2] * x
    mul!(out, dA, ws.buf, 1, 1)                     # + Ȧ (W_1 x)
    mul!(ws.v, A, x)
    @. ws.v = ws.v - im * freqs * x                 # v = (A - iΩ) x
    @. ws.buf = W[2] * ws.v
    mul!(out, A, ws.buf, 1, 1)                       # + A W_1 (A - iΩ) x
    return out
end

@inline function _apply_M!(out, x, ops, W, freqs, ws, ::Val{2})
    A, dA, ddA = ops
    # j = 0, 1 terms (identical to the s = 1 expansion)
    @. ws.buf = W[1] * x
    mul!(out, A, ws.buf)                            # A (W_0 x)
    @. ws.buf = W[2] * x
    mul!(out, dA, ws.buf, 1, 1)                     # + Ȧ (W_1 x)
    mul!(ws.v, A, x)
    @. ws.v = ws.v - im * freqs * x                 # v = (A - iΩ) x
    @. ws.buf = W[2] * ws.v
    mul!(out, A, ws.buf, 1, 1)                       # + A W_1 (A - iΩ) x
    # j = 2 terms
    @. ws.buf = W[3] * x
    mul!(out, ddA, ws.buf, 1, 1)                    # + Ä (W_2 x)
    @. ws.buf = W[3] * ws.v                          # ws.v still holds (A - iΩ) x
    mul!(out, dA, ws.buf, 2, 1)                     # + 2 Ȧ W_2 (A - iΩ) x
    # F_2 x = Ȧ x + A(A x) - Ω² x - 2iΩ (A x)
    mul!(ws.v, A, x)                                # v = A x
    mul!(ws.w, A, ws.v)                             # w = A(A x)
    @. ws.w = ws.w - (freqs^2) * x - 2im * freqs * ws.v
    mul!(ws.w, dA, x, 1, 1)                          # + Ȧ x
    @. ws.buf = W[3] * ws.w
    mul!(out, A, ws.buf, 1, 1)                       # + A W_2 F_2 x
    return out
end

# Callable, wrapped once in a LinearMap: applies x ↦ S_I x = x - M_I x.  The
# operator-derivative tuple `ops` is refreshed in place each timestep (its type
# is fixed across timesteps, so the update is allocation-free), letting the same
# LinearMap and GMRES workspace be reused for every step.
mutable struct _ImplicitApply{S,OT,WT,FT,WS}
    ops::OT
    W::WT
    freqs::FT
    ws::WS
end

@inline function (ia::_ImplicitApply{S})(out, x) where {S}
    _apply_M!(out, x, ia.ops, ia.W, ia.freqs, ia.ws, Val(S))
    @. out = x - out
    return out
end

# -----------------------------------------------------------------------------
# DYNAMIC step, EFFICIENT ordering — matrix-free apply of the propagator's
# bracketed sum using Appendix A's "second ordering"
#
#   M^s = Σ_{i=0}^s A^{(i)} Σ_{j=i}^s C(j,i) W^s_j F_{j-i},
#
# which is algebraically identical to the naive Σ_j Σ_k form used by _apply_M!
# above, but applies each A^{(i)} only once (to the output of the inner sum) and
# computes each envelope-derivative vector F_m x only once.  Writing
# F_0 x = x, F_1 x = (A - iΩ)x, F_2 x = (Ȧ + A² - Ω² - 2iΩA)x, the expansions are
#
#   s=0:  M x = A (W_0 x)
#   s=1:  M x = A (W_0 x + W_1 F_1 x) + Ȧ (W_1 x)
#   s=2:  M x = A (W_0 x + W_1 F_1 x + W_2 F_2 x)
#               + Ȧ (W_1 x + 2 W_2 F_1 x) + Ä (W_2 x).
#
# Matvec counts per apply: s=1 → 2×A, 1×Ȧ (vs 3×A in the naive form); s=2 →
# 3×A, 2×Ȧ, 1×Ä = 6 (vs 9).  Like _apply_M!, every step uses only mul! and
# broadcasts into the preallocated workspace, so it never allocates.
# -----------------------------------------------------------------------------

# Scratch buffers for the efficient apply (separate from _FilonDynWS so the old
# path is untouched).  g1, g2 hold F_1 x, F_2 x; Ax holds A x then A(A x); inner
# is the inner-sum accumulator fed to each A^{(i)} mul!; Mψ and rhs are used by
# the timestepping loop.
struct _FilonEfficientWS{V<:AbstractVector}
    g1::V
    g2::V
    Ax::V
    inner::V
    Mψ::V
    rhs::V
end
_FilonEfficientWS(N::Integer) =
    _FilonEfficientWS(ntuple(_ -> zeros(ComplexF64, N), Val(6))...)

@inline function _apply_M_efficient!(out, x, ops, W, freqs, ws, ::Val{0})
    @. ws.inner = W[1] * x                          # W_0 F_0 x  (F_0 = I)
    mul!(out, ops[1], ws.inner)                     # A (…)
    return out
end

@inline function _apply_M_efficient!(out, x, ops, W, freqs, ws, ::Val{1})
    A, dA = ops
    mul!(ws.Ax, A, x)
    @. ws.g1 = ws.Ax - im * freqs * x               # F_1 x = (A - iΩ) x
    @. ws.inner = W[1] * x + W[2] * ws.g1           # i=0 inner sum
    mul!(out, A, ws.inner)                          # A (W_0 F_0 x + W_1 F_1 x)
    @. ws.inner = W[2] * x                          # i=1 inner sum
    mul!(out, dA, ws.inner, 1, 1)                   # + Ȧ (W_1 F_0 x)
    return out
end

@inline function _apply_M_efficient!(out, x, ops, W, freqs, ws, ::Val{2})
    A, dA, ddA = ops
    mul!(ws.Ax, A, x)                               # A x
    @. ws.g1 = ws.Ax - im * freqs * x               # F_1 x = (A - iΩ) x
    mul!(ws.g2, A, ws.Ax)                           # A² x
    @. ws.g2 = ws.g2 - (freqs^2) * x - 2im * freqs * ws.Ax
    mul!(ws.g2, dA, x, 1, 1)                         # F_2 x = Ȧx + A²x - Ω²x - 2iΩAx
    @. ws.inner = W[1] * x + W[2] * ws.g1 + W[3] * ws.g2
    mul!(out, A, ws.inner)                          # A (W_0 F_0 x + W_1 F_1 x + W_2 F_2 x)
    @. ws.inner = W[2] * x + 2 * W[3] * ws.g1
    mul!(out, dA, ws.inner, 1, 1)                   # + Ȧ (W_1 F_0 x + 2 W_2 F_1 x)
    @. ws.inner = W[3] * x
    mul!(out, ddA, ws.inner, 1, 1)                  # + Ä (W_2 F_0 x)
    return out
end

# Callable wrapped once in a LinearMap: x ↦ S_I x = x - M^s_I x, efficient form.
# Mirrors _ImplicitApply; `ops` is refreshed in place each timestep so the same
# LinearMap and GMRES workspace are reused.
mutable struct _ImplicitApplyEfficient{S,OT,WT,FT,WS}
    ops::OT
    W::WT
    freqs::FT
    ws::WS
end

@inline function (ia::_ImplicitApplyEfficient{S})(out, x) where {S}
    _apply_M_efficient!(out, x, ia.ops, ia.W, ia.freqs, ia.ws, Val(S))
    @. out = x - out
    return out
end

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

# Step indices at which the state is stored: 0 (initial), every `save_every`-th
# step, and always the final step `nsteps`.
function _save_indices(nsteps::Integer, save_every::Integer)
    idx = collect(0:save_every:nsteps)
    (isempty(idx) || idx[end] != nsteps) && push!(idx, nsteps)
    return idx
end

# -----------------------------------------------------------------------------
# Timestepping drivers
# -----------------------------------------------------------------------------

function _solve_static(co, ψ0, frequencies, Δt, nsteps, vs::Val, save_every,
                       save_final_only, stats)
    return _solve_static(co, ψ0, frequencies, Δt, nsteps, vs, save_every,
                         save_final_only, stats, Val(_nstate(co)))
end

function _solve_static(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                       save_final_only, stats, ::Val{N}) where {S,N}
    wp = _static_weights(co, frequencies, Δt, Val(S), Val(N))
    _stats_init!(stats, nsteps, false)
    ψ = SVector{N,ComplexF64}(ψ0)
    A_n = _matderivs(co, zero(Δt), Val(S))

    if save_final_only
        @inbounds for n in 1:nsteps
            t0 = _stats_tick(stats)
            A_np1 = _matderivs(co, n * Δt, Val(S))
            ψ = _filon_step_static(A_n, A_np1, ψ, wp)
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
        ψ = _filon_step_static(A_n, A_np1, ψ, wp)
        A_n = A_np1
        _stats_record_static!(stats, t0)
        if col <= length(save_idx) && save_idx[col] == n
            history[:, col] .= ψ
            col += 1
        end
    end
    return history
end

function _solve_dynamic(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                        save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    wp = _dynamic_weights(co, frequencies, Δt, Val(S))
    ws = _FilonDynWS(N)
    ψ = Vector{ComplexF64}(undef, N)
    ψ .= ψ0

    A_n = _opderivs(co, zero(Δt), Val(S))
    A_np1 = _opderivs(co, Δt, Val(S))               # placeholder (fixes the field type)
    ia = _ImplicitApply{S,typeof(A_np1),typeof(wp.WI),typeof(wp.freqs),typeof(ws)}(
        A_np1, wp.WI, wp.freqs, ws)
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
        # RHS = ψ + M_E ψ   (explicit side applied to the state)
        _apply_M!(ws.Mψ, ψ, A_n, wp.WE, wp.freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # Solve  (I - M_I) ψ_{n+1} = RHS   matrix-free with GMRES.  Optionally
        # warm-start from the previous step's solution ψ_n (consecutive S_I and
        # solutions are nearly identical); costs one extra residual matvec, so it
        # only pays off in high-iteration regimes — see the docstring.
        ia.ops = A_np1
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        A_n = A_np1
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

# Efficient-ordering counterpart of _solve_dynamic.  Identical control flow and
# weight data (DynamicFilonWeights); only the matrix-free apply differs, using
# _apply_M_efficient!/_ImplicitApplyEfficient and the _FilonEfficientWS buffers.
function _solve_dynamic_efficient(co, ψ0, frequencies, Δt, nsteps, ::Val{S}, save_every,
                                  save_final_only, warm_start, atol, rtol, stats) where {S}
    N = _nstate(co)
    wp = _dynamic_weights(co, frequencies, Δt, Val(S))
    ws = _FilonEfficientWS(N)
    ψ = Vector{ComplexF64}(undef, N)
    ψ .= ψ0

    A_n = _opderivs(co, zero(Δt), Val(S))
    A_np1 = _opderivs(co, Δt, Val(S))               # placeholder (fixes the field type)
    ia = _ImplicitApplyEfficient{S,typeof(A_np1),typeof(wp.WI),typeof(wp.freqs),typeof(ws)}(
        A_np1, wp.WI, wp.freqs, ws)
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
        # RHS = ψ + M_E ψ   (explicit side applied to the state)
        _apply_M_efficient!(ws.Mψ, ψ, A_n, wp.WE, wp.freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        # Solve  (I - M_I) ψ_{n+1} = RHS   matrix-free with GMRES.
        ia.ops = A_np1
        warm_start && Krylov.warm_start!(kws, ψ)
        Krylov.gmres!(kws, L, ws.rhs; atol = atol, rtol = rtol)
        ψ .= Krylov.solution(kws)
        A_n = A_np1
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
# Public API
# -----------------------------------------------------------------------------

"""
    filon_timestep_hardcoded(co, ψ, t_n, Δt, wp; atol, rtol) -> ψ_next

Advance the state `ψ` by one hard-coded Filon step of size `Δt`, from `t_n` to
`t_n + Δt`, for the linear ODE `dψ/dt = A(t) ψ` with `A = co`.  `wp` is the
precomputed weight-phase data from [`filon_weight_phases`](@ref); its type
selects the order `s` and the static / dynamic implementation.

The static path (`wp::StaticFilonWeights`) returns an `SVector` and allocates
nothing; the dynamic path (`wp::DynamicFilonWeights`) returns a `Vector` and uses
GMRES (the `atol`/`rtol` keywords set its tolerances).  On the dynamic path,
`efficient = true` applies the propagator in the reorganized "second ordering"
of Appendix A (each `A^{(i)}` applied once, each `F_m ψ` computed once), which
reduces the matrix–vector products per apply while giving the same result.

For many steps, prefer [`filon_solve_hardcoded`](@ref), which reuses the GMRES
workspace and avoids re-evaluating `A` at shared step boundaries.
"""
function filon_timestep_hardcoded(co::ControlledOperator, ψ, t_n::Real, Δt::Real,
                                  wp::StaticFilonWeights{S}) where {S}
    A_n = _matderivs(co, t_n, Val(S))
    A_np1 = _matderivs(co, t_n + Δt, Val(S))
    return _filon_step_static(A_n, A_np1, ψ, wp)
end

function filon_timestep_hardcoded(co::ControlledOperator, ψ, t_n::Real, Δt::Real,
                                  wp::DynamicFilonWeights{S};
                                  atol::Real = 1e-13, rtol::Real = 1e-13,
                                  efficient::Bool = false) where {S}
    N = length(ψ)
    A_n = _opderivs(co, t_n, Val(S))
    A_np1 = _opderivs(co, t_n + Δt, Val(S))
    if efficient
        ws = _FilonEfficientWS(N)
        _apply_M_efficient!(ws.Mψ, ψ, A_n, wp.WE, wp.freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        ia = _ImplicitApplyEfficient{S,typeof(A_np1),typeof(wp.WI),typeof(wp.freqs),typeof(ws)}(
            A_np1, wp.WI, wp.freqs, ws)
    else
        ws = _FilonDynWS(N)
        _apply_M!(ws.Mψ, ψ, A_n, wp.WE, wp.freqs, ws, Val(S))
        @. ws.rhs = ψ + ws.Mψ
        ia = _ImplicitApply{S,typeof(A_np1),typeof(wp.WI),typeof(wp.freqs),typeof(ws)}(
            A_np1, wp.WI, wp.freqs, ws)
    end
    L = LinearMap{ComplexF64}(ia, N; ismutating = true)
    sol, _ = Krylov.gmres(L, ws.rhs; atol = atol, rtol = rtol)
    return sol
end

"""
    filon_solve_hardcoded(co, ψ0, frequencies, Δt, nsteps, s; kwargs...)

Solve `dψ/dt = A(t) ψ` (with `A = co`, a [`ControlledOperator`](@ref)) over
`nsteps` steps of fixed size `Δt`, starting from `ψ0`, using the hard-coded
Filon method of order `s ∈ {0,1,2}` and ansatz `frequencies` (one per state
component).

Returns, by default, an `N × nsaves` matrix whose columns are the state at the
saved times — the initial state, then every `save_every`-th step, then always
the final step.  The saved times are `Δt .* (0:save_every:nsteps)` (plus the
final step if it is not a multiple of `save_every`).

# Keyword arguments
- `save_every::Integer = 1` — store the state every `save_every` steps (use a
  large value to compare against other methods without keeping every step).
- `save_final_only::Bool = false` — return just the final state vector instead
  of the history matrix.
- `variant::Symbol = :auto` — `:static` forms the propagator matrices and solves
  with `\\` (allocation-free for an `SMatrix`-backed `co`); `:dynamic` is
  matrix-free and solves with GMRES (for a `Vector`-backed `co`).  `:auto`
  chooses by the layout of `co`.
- `gmres_atol`, `gmres_rtol` — GMRES tolerances for the dynamic variant.
- `warm_start::Bool = false` — for the dynamic variant, seed each GMRES solve
  with the previous step's solution `ψ_n` as the initial guess (consecutive
  systems `S_I` and their solutions are nearly identical).  It costs one extra
  matvec per step to form the initial residual, so it only helps when GMRES
  otherwise takes *many* iterations; in the well-conditioned, few-iteration
  regime typical of a converged solve it is a wash (or slightly slower), which
  is why it defaults off.  Ignored by the static variant (which solves directly).
- `efficient::Bool = false` — for the dynamic variant, apply the propagator in
  the reorganized "second ordering" of Appendix A
  (`Σ_i A^{(i)} Σ_{j≥i} C(j,i) W_j F_{j-i}`), which applies each matrix
  derivative `A^{(i)}` once and reuses each envelope-derivative vector `F_m ψ`,
  cutting the matrix–vector products per step (e.g. 6 vs 9 at `s=2`) while
  returning the same answer.  The per-step apply remains allocation-free.  Only
  valid for the dynamic variant; combining it with `:static` throws.
- `stats::Union{Nothing,FilonSolveStats} = nothing` — pass a
  [`FilonSolveStats`](@ref) to collect per-step wall times and (dynamic variant
  only) GMRES iteration counts and convergence flags.  The collector is emptied
  at the start of the solve.  The default `nothing` adds zero overhead.
"""
function filon_solve_hardcoded(co::ControlledOperator, ψ0::AbstractVector,
                               frequencies::AbstractVector, Δt::Real, nsteps::Integer,
                               s::Integer; save_every::Integer = 1,
                               save_final_only::Bool = false, variant::Symbol = :auto,
                               gmres_atol::Real = 1e-13, gmres_rtol::Real = 1e-13,
                               warm_start::Bool = false, efficient::Bool = false,
                               stats::Union{Nothing,FilonSolveStats} = nothing)
    0 <= s <= 2 || throw(ArgumentError("hard-coded Filon supports s ∈ {0,1,2}; got s=$s"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1; got $nsteps"))
    save_every >= 1 || throw(ArgumentError("save_every must be ≥ 1; got $save_every"))
    length(frequencies) == _nstate(co) ||
        throw(DimensionMismatch("need one frequency per state component "*
            "($(_nstate(co))); got $(length(frequencies))"))
    length(ψ0) == _nstate(co) ||
        throw(DimensionMismatch("ψ0 has length $(length(ψ0)); expected $(_nstate(co))"))

    if _resolve_variant(co, variant) === :static
        efficient && throw(ArgumentError("efficient=true applies only to the "*
            "matrix-free dynamic variant; the static variant forms the full "*
            "propagator matrix, where the reorganization gives no benefit. "*
            "Pass variant=:dynamic to use it."))
        return _solve_static(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                             save_every, save_final_only, stats)
    elseif efficient
        return _solve_dynamic_efficient(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                             save_every, save_final_only, warm_start, gmres_atol, gmres_rtol,
                             stats)
    else
        return _solve_dynamic(co, ψ0, frequencies, Δt, nsteps, Val(Int(s)),
                             save_every, save_final_only, warm_start, gmres_atol, gmres_rtol,
                             stats)
    end
end
