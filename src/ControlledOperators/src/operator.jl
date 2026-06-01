# ---------------------------------------------------------------------------------------------
# Operator — the realized A(t) at a fixed time
# ---------------------------------------------------------------------------------------------

"""
    Operator{T,CT,MC} <: AbstractMatrix{T}

A *realized* time-dependent operator: the value of a [`Generator`](@ref) frozen at one time,

    A = Σₖ coeffs[k] · matrices[k].

It stores

* `coeffs::CT`   — the realized scalar coefficients (`SVector` for the static layout, `Vector`
  for the dynamic layout), and
* `matrices::MC` — **the very same matrix container the generator holds** (never copied).

`Operator <: AbstractMatrix{T}` with `T = promote_type(eltype(coeffs), eltype(matrix))`, so it
implements `size`, `eltype`, `getindex`, `mul!` and `*` and drops straight into iterative
solvers such as Krylov.jl, or into a direct solve via [`materialize`](@ref) / `\\`.
"""
struct Operator{T,CT,MC} <: AbstractMatrix{T}
    coeffs::CT
    matrices::MC
end

@inline function _realized_eltype(coeffs, matrices)
    return promote_type(eltype(coeffs), eltype(first(matrices)))
end

function Operator(coeffs::CT, matrices::MC) where {CT,MC}
    # The coefficient and matrix containers are indexed in lockstep (`coeffs[k]`,
    # `matrices[k]`), so they must have the same length *and* start at index 1 — the latter
    # because the `mul!`/`getindex`/`materialize` loops assume 1-based access (and use
    # `@inbounds`).  Both checks fold away at compile time for the static (tuple/SVector)
    # layout, so they cost nothing there.
    Base.require_one_based_indexing(coeffs, matrices)
    length(coeffs) == length(matrices) || throw(DimensionMismatch(
        "number of coefficients ($(length(coeffs))) must equal number of matrices ($(length(matrices)))"))
    T = _realized_eltype(coeffs, matrices)
    return Operator{T,CT,MC}(coeffs, matrices)
end

# Realize the coefficient container from a container of controls at time `t`.
@inline function _coeffs(controls::Tuple, t, d)
    return SVector(map(c -> derivative(c, t, d), controls))
end

@inline function _coeffs(controls::AbstractVector, t, d)
    return [derivative(c, t, d) for c in controls]
end

"""
    evaluate(gen, t, order = Derivative{0}()) -> Operator

Realize the generator `gen` at time `t`, returning an [`Operator`](@ref) that shares `gen`'s
matrices.  With `order = Derivative{N}()` the coefficients are each control's `N`-th
derivative, giving the realized `N`-th time-derivative `Aᴺ(t)` — structurally identical, only
the coefficients differ.

For a fully static generator (tuple matrices) this allocates nothing.  For repeated
refreshing at a fixed `t` across solver iterations, build a reusable operator with
[`Operator(gen, t)`](@ref) and call [`evaluate!`](@ref).
"""
@inline function evaluate(gen::Generator, t::Real, d::Derivative = Derivative{0}())
    coeffs = _coeffs(gen.controls, t, d)
    return Operator(coeffs, gen.matrices)
end

"""
    Operator(gen::Generator, t, order = Derivative{0}()) -> Operator

Build a **reusable, mutable** operator from `gen` at time `t`, forcing a `Vector` coefficient
buffer (one allocation up front).  Subsequent [`evaluate!`](@ref) calls refresh that buffer in
place with zero allocation, even when `gen`'s controls are a heterogeneous tuple.
"""
function Operator(gen::Generator, t::Real, d::Derivative = Derivative{0}())
    coeffs = collect(_coeffs(gen.controls, t, d))
    return Operator(coeffs, gen.matrices)
end

"""
    evaluate!(op, gen, t, order = Derivative{0}()) -> op

Refresh `op`'s coefficients in place from `gen` at time `t` (the matrices are untouched).
`op` must have a mutable coefficient buffer — build it with [`Operator(gen, t)`](@ref) or any
`evaluate` that produced `Vector` coefficients.  Allocation-free, including for a heterogeneous
tuple of controls (the write loop is unrolled).
"""
@inline function evaluate!(op::Operator, gen::Generator, t::Real, d::Derivative = Derivative{0}())
    _evaluate_coeffs!(op.coeffs, gen.controls, t, d)
    return op
end

@inline function _evaluate_coeffs!(coeffs::AbstractVector, controls::AbstractVector, t, d)
    @inbounds for k in eachindex(controls, coeffs)
        coeffs[k] = derivative(controls[k], t, d)
    end
    return coeffs
end

# Tuple of (possibly heterogeneous) controls → unrolled, type-stable writes into the buffer.
@inline function _evaluate_coeffs!(coeffs::AbstractVector, controls::Tuple, t, d)
    return _write_coeffs!(coeffs, controls, t, d, 1)
end

"""
    _write_coeffs!(coeffs, controls::Tuple, t, d, k) -> coeffs

Write the realized coefficients of a **tuple** of controls into the mutable buffer `coeffs`,
starting at index `k`, by *compile-time–unrolled recursion*.

A tuple encodes its length in its **type** (`Tuple{C1,C2,…}`), so the recursion below is
resolved entirely by dispatch: the compiler expands it into straight-line
`coeffs[k] = …; coeffs[k+1] = …; …` with no runtime loop and no actual recursive calls.  That
is what keeps the write type-stable and allocation-free even when the controls are
*heterogeneous* — at each unrolled step `first(controls)` has a single concrete type the
compiler can see, so `derivative` is dispatched statically.

The recursion is formed by two methods:

* the **general** method peels off `first(controls)`, writes its `derivative`, and recurses on
  `Base.tail(controls)` (the tuple with its first element dropped — `(a,b,c)→(b,c)→(c,)→()`);
* the **base case** `::Tuple{}` matches *only the empty tuple* `()`, the value `Base.tail`
  ultimately produces, and stops the recursion by returning `coeffs` unchanged.

`Tuple{}` is the concrete type of `()` specifically: a non-empty `(a, b)` has type
`Tuple{typeof(a),typeof(b)}`, which is **not** `Tuple{}`, so it takes the general `::Tuple`
method.  When the tuple *is* empty both signatures apply, but `Tuple{}` is strictly more
specific and therefore wins.  (`Tuple{}` is distinct from `Tuple`, the abstract supertype of
all tuples.)

Used by `_evaluate_coeffs!` for tuple-valued control containers; the `AbstractVector` (dynamic)
container uses an ordinary `for` loop instead.
"""
@inline function _write_coeffs!(coeffs, ::Tuple{}, t, d, k)
    return coeffs
end

@inline function _write_coeffs!(coeffs, controls::Tuple, t, d, k)
    @inbounds coeffs[k] = derivative(first(controls), t, d)
    return _write_coeffs!(coeffs, Base.tail(controls), t, d, k + 1)
end

function _evaluate_coeffs!(::StaticVector, controls, t, d)
    return throw(ArgumentError(
        "evaluate! needs a mutable coefficient buffer; build the operator with `Operator(gen, t)` " *
        "or use `evaluate(gen, t)` (which is itself allocation-free for static generators)."))
end

# ---------------------------------------------------------------------------------------------
# AbstractMatrix interface
# ---------------------------------------------------------------------------------------------

function Base.size(op::Operator)
    return size(first(op.matrices))
end

function Base.size(op::Operator, d::Integer)
    return size(first(op.matrices), d)
end

Base.@propagate_inbounds function Base.getindex(op::Operator{T}, i::Int, j::Int) where {T}
    acc = zero(T)
    @inbounds for k in 1:length(op.coeffs)
        acc += op.coeffs[k] * op.matrices[k][i, j]
    end
    return acc
end

# ---------------------------------------------------------------------------------------------
# mul!  —  y ← β·y + α·A·x
# ---------------------------------------------------------------------------------------------

# The 3-arg `mul!(y, op, x)` is provided by LinearAlgebra's generic fallback, which forwards
# to the 5-arg methods below (with α = true, β = false).

# Dynamic layout (Vector of matrices): plain loop, non-allocating (BLAS gemv! / sparse mul!).
function LinearAlgebra.mul!(y::AbstractVector, op::Operator{T,CT,<:AbstractVector},
                            x::AbstractVector, α::Number, β::Number) where {T,CT}
    mats = op.matrices
    c = op.coeffs
    n = length(mats)
    n == 0 && return _scale!(y, β)
    @inbounds mul!(y, mats[1], x, α * c[1], β)
    @inbounds for k in 2:n
        mul!(y, mats[k], x, α * c[k], true)
    end
    return y
end

# Static layout (Tuple of matrices): unrolled via tuple recursion, type-stable & non-allocating.
function LinearAlgebra.mul!(y::AbstractVector, op::Operator{T,CT,<:Tuple},
                            x::AbstractVector, α::Number, β::Number) where {T,CT}
    _mul_tuple!(y, op.matrices, op.coeffs, x, α, β)
    return y
end

"""
    _mul_tuple!(y, mats::Tuple, c, x, α, β)

Apply the *first* term of a tuple-stored operator: `y ← β·y + α·c[1]·(mats[1]·x)`, then hand
off to `_mul_tuple_rest!` to accumulate the remaining terms.  Split out from the rest because
only this leading term carries the caller's `β` (every later term accumulates with `β = true`);
this is also the only term whose coefficient index is statically known to be `1`, so unlike
`_mul_tuple_rest!` it needs no running `k`.
"""
@inline function _mul_tuple!(y, mats::Tuple, c, x, α, β)
    mul!(y, first(mats), x, α * c[1], β)               # first term applies β
    return _mul_tuple_rest!(y, Base.tail(mats), c, x, α, 2)
end

"""
    _mul_tuple_rest!(y, mats::Tuple, c, x, α, k) -> y

Accumulate the remaining tuple terms in place: `y += α·c[k]·(mats[k]·x)`, recursing on
`Base.tail(mats)`.  Like [`_write_coeffs!`](@ref), this is compile-time–unrolled tuple
recursion whose base case `::Tuple{}` matches *only the empty tuple* and stops it.  The empty
base case is the right one here because the work is an in-place accumulation into `y`: once no
terms remain there is simply nothing to add, so it returns `y` unchanged.
"""
@inline function _mul_tuple_rest!(y, ::Tuple{}, c, x, α, k)
    return y
end

@inline function _mul_tuple_rest!(y, mats::Tuple, c, x, α, k)
    mul!(y, first(mats), x, α * c[k], true)            # subsequent terms accumulate
    return _mul_tuple_rest!(y, Base.tail(mats), c, x, α, k + 1)
end

function _scale!(y, β)
    iszero(β) ? fill!(y, zero(eltype(y))) : rmul!(y, β)
    return y
end

# ---------------------------------------------------------------------------------------------
# *  —  application to a vector
# ---------------------------------------------------------------------------------------------

# Dynamic: allocate the result (allowed), then a non-allocating mul!.
function Base.:*(op::Operator{T,CT,<:AbstractVector}, x::AbstractVector) where {T,CT}
    y = similar(x, promote_type(T, eltype(x)), size(op, 1))
    return mul!(y, op, x, true, false)
end

# Static (tuple matrices, static vector): unrolled, returns an SVector with no heap allocation.
function Base.:*(op::Operator{T,CT,<:Tuple}, x::StaticVector) where {T,CT}
    return _times_tuple(op.matrices, op.coeffs, x, 1)
end

"""
    _times_tuple(mats::Tuple, c, x, k) -> SVector

Compute `Σₖ c[k]·(mats[k]·x)` over a tuple of static matrices by compile-time–unrolled
recursion, returning the freshly *constructed* `SVector` (no in-place buffer, no allocation).

Note the base case differs from [`_write_coeffs!`](@ref): it is `::Tuple{Any}`, matching a
**one-element** tuple, not `::Tuple{}`.  Because this builds and returns a value (a sum of
`SVector`s) rather than mutating a buffer, there is no natural neutral element of the right
static type to start an empty sum from — so the recursion bottoms out at the *last* term and
returns `c[k]·(mats[k]·x)` directly.  (`Tuple{Any}` is the type of any length-1 tuple `(a,)`;
a longer tuple matches the general `::Tuple` method.)  This assumes the operator has ≥1 term,
which `Generator`/`evaluate` guarantee for the static layout.
"""
@inline function _times_tuple(mats::Tuple{Any}, c, x, k)
    return c[k] * (first(mats) * x)
end

@inline function _times_tuple(mats::Tuple, c, x, k)
    return c[k] * (first(mats) * x) + _times_tuple(Base.tail(mats), c, x, k + 1)
end

# ---------------------------------------------------------------------------------------------
# Arithmetic on realized operators (concatenate term lists)
# ---------------------------------------------------------------------------------------------

function _neg(c::AbstractVector)
    return -c
end

"""
    +(a::Operator, b::Operator)
    -(a::Operator, b::Operator)

Combine realized operators by **concatenating their term lists**.  Subtraction negates the
right-hand side's (already realized) coefficients; the matrices are shared.  No
operator×operator product is defined.
"""
function Base.:+(a::Operator, b::Operator)
    return Operator(_cat(a.coeffs, b.coeffs), _cat(a.matrices, b.matrices))
end

function Base.:-(a::Operator, b::Operator)
    return Operator(_cat(a.coeffs, _neg(b.coeffs)), _cat(a.matrices, b.matrices))
end

function Base.:-(op::Operator)
    return Operator(_neg(op.coeffs), op.matrices)
end

function Base.:*(s::Number, op::Operator)
    return Operator(s .* op.coeffs, op.matrices)
end

function Base.:*(op::Operator, s::Number)
    return s * op
end

# ---------------------------------------------------------------------------------------------
# Materialization — form Σ cₖ Aₖ as a concrete matrix
# ---------------------------------------------------------------------------------------------

"""
    materialize(op) -> matrix
    materialize!(out, op) -> out

Form the dense (or static) matrix `Σₖ coeffs[k]·matrices[k]`.  `materialize(op)` returns an
`SMatrix` for the static (tuple) layout and a dense `Matrix` for the dynamic layout;
`materialize!` accumulates into the preallocated `out`.  See also `Matrix(op)`, `sparse(op)`
and `collect(op)`.
"""
function materialize! end

function materialize!(out::AbstractMatrix, op::Operator)
    fill!(out, zero(eltype(out)))
    @inbounds for k in 1:length(op.coeffs)
        out .+= op.coeffs[k] .* op.matrices[k]
    end
    return out
end

function materialize(op::Operator{T,CT,<:AbstractVector}) where {T,CT}
    return Matrix(op)
end

function materialize(op::Operator{T,CT,<:Tuple}) where {T,CT}
    return _sum_terms(op.matrices, op.coeffs, 1)
end

"""
    _sum_terms(mats::Tuple, c, k) -> SMatrix

Form `Σₖ c[k]·mats[k]` over a tuple of static matrices by compile-time–unrolled recursion,
returning the constructed `SMatrix`.  Like [`_times_tuple`](@ref), it builds a value, so its
base case is the **one-element** `::Tuple{Any}` (returning `c[k]·mats[k]`) rather than the
empty `::Tuple{}` — there is no zero `SMatrix` of known type to seed an empty sum.  Assumes ≥1
term, which the static layout guarantees.
"""
@inline function _sum_terms(mats::Tuple{Any}, c, k)
    return c[k] * first(mats)
end

@inline function _sum_terms(mats::Tuple, c, k)
    return c[k] * first(mats) + _sum_terms(Base.tail(mats), c, k + 1)
end

function Base.Matrix(op::Operator)
    return materialize!(Matrix{eltype(op)}(undef, size(op)...), op)
end

function Base.collect(op::Operator)
    return Matrix(op)
end

function SparseArrays.sparse(op::Operator)
    s = op.coeffs[1] * sparse(op.matrices[1])
    @inbounds for k in 2:length(op.coeffs)
        s = s + op.coeffs[k] * sparse(op.matrices[k])
    end
    return s
end

# ---------------------------------------------------------------------------------------------
# Pretty printing (override AbstractMatrix's grid display)
# ---------------------------------------------------------------------------------------------

function Base.show(io::IO, op::Operator)
    K = length(op.coeffs)
    print(io, "Operator(", K, " term", K == 1 ? "" : "s", ", ")
    print(io, op.matrices isa Tuple ? "static" : "dynamic", ")")
end

function Base.show(io::IO, ::MIME"text/plain", op::Operator)
    m, n = size(op)
    K = length(op.coeffs)
    print(io, m, "×", n, " Operator{", eltype(op), "} — ", K, " term", K == 1 ? "" : "s", " (",
          op.matrices isa Tuple ? "static" : "dynamic", ")")
end
