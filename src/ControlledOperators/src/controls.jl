"""
    AbstractControl{T}

A scalar, time-dependent control `c(t)` whose values have element type `T`.

Subtypes implement the single interface method

    derivative(control, t, order)

where `order` is a compile-time singleton — either [`Derivative{N}`](@ref) (the `N`-th time
derivative; `N == 0` is plain evaluation) or [`DerivativeUpTo{N}`](@ref) (an `SVector{N+1,T}`
of derivatives `0..N`).  A control is also callable: `control(t)` is shorthand for the `0`-th
derivative.

All concrete controls in this package are *named structs* (never anonymous closures), so they
never leak a closure type into user-visible signatures.
"""
abstract type AbstractControl{T} end

function Base.eltype(::Type{<:AbstractControl{T}}) where {T}
    return T
end

function Base.eltype(c::AbstractControl)
    eltype(typeof(c))
end

"""
Evaluate the control `c` at time `t`. I.e., the programming expression `c(t)`
matches the mathematical expression c(t).
"""
@inline function (c::AbstractControl)(t::Real)
    return derivative(c, t, Derivative{0}())
end

"""
    derivative(control, t::Real, order) -> value

Evaluate a time-derivative of a scalar `control` at time `t`.

The built-in controls constrain `t` to `Real`.  This both documents intent (time is real) and
rejects nonsensical arguments, while still admitting differentiation *through* `t`: the AD types
this package relies on — `TaylorDiff.TaylorScalar` and `ForwardDiff.Dual` — are themselves
`<: Real`.  When you define your *own* control, type its `t` argument `t::Real` (or leave it
untyped) — but **not** a concrete type such as `t::Float64`, or the Taylor-mode fallback below
cannot reach it (the seed it passes is a `TaylorScalar`, not a `Float64`).

`order` is a compile-time singleton:

* `Derivative{N}()`     → the `N`-th time-derivative (`N == 0` is plain evaluation), a scalar
  of type `eltype(control)`.
* `DerivativeUpTo{N}()` → an `SVector{N+1}` holding derivatives of orders `0..N`, computed in
  a single call.

Loading the `TaylorDiff` extension supplies a generic `Derivative`/`DerivativeUpTo` fallback
(via Taylor-mode AD) for any control that only defines the `Derivative{0}` evaluation.
"""
function derivative end

# ---------------------------------------------------------------------------------------------
# ConstantControl
# ---------------------------------------------------------------------------------------------

"""
    ConstantControl(value)

A control that is constant in time: `c(t) = value`, with all higher derivatives zero.  Use
`ConstantControl(1)` for the drift term `H₀` of a controlled operator.
"""
struct ConstantControl{T} <: AbstractControl{T}
    value::T
end

@inline function derivative(c::ConstantControl{T}, t::Real, ::Derivative{0}) where {T}
    return c.value
end

@inline function derivative(c::ConstantControl{T}, t::Real, ::Derivative{N}) where {T,N}
    return zero(T)
end
@inline function derivative(c::ConstantControl{T}, t::Real, ::DerivativeUpTo{N}) where {T,N}
    return SVector(ntuple(m -> m == 1 ? c.value : zero(T), Val(N + 1)))
end

# ---------------------------------------------------------------------------------------------
# FourierControl
# ---------------------------------------------------------------------------------------------

"""
    FourierControl(a0, a, b, ω)

A truncated Fourier series control

    c(t) = a0 + Σₙ aₙ cos(n ω t) + bₙ sin(n ω t),   n = 1 … length(a)

with analytic derivatives of every order

    cᴺ(t) = Σₙ aₙ (nω)ᴺ cos(n ω t + Nπ/2) + bₙ (nω)ᴺ sin(n ω t + Nπ/2).

`a` and `b` are stored as `SVector`s (and may be empty for a pure constant).  The value type
`T = promote_type(typeof(a0), eltype(a), eltype(b))` may be real or complex.
"""
struct FourierControl{T,NF,V<:Real} <: AbstractControl{T}
    a0::T
    a::SVector{NF,T}
    b::SVector{NF,T}
    ω::V
end

function FourierControl(a0, a::AbstractVector, b::AbstractVector, ω::Real)
    length(a) == length(b) ||
        throw(DimensionMismatch("cosine/sine coefficient vectors must have equal length"))
    NF = length(a)
    T = promote_type(typeof(a0), eltype(a), eltype(b))
    FourierControl{T,NF,typeof(ω)}(convert(T, a0), SVector{NF,T}(a), SVector{NF,T}(b), ω)
end

# Scalar `m`-th derivative.  `m` is a plain `Int`, but the result is always typed `T`, so the
# method is type-stable regardless of the (runtime) value of `m`.
@inline function _fourier_deriv(c::FourierControl{T,NF}, t::Real, m::Int) where {T,NF}
    acc = m == 0 ? c.a0 : zero(T)
    @inbounds for n in 1:NF
        w = n * c.ω
        ang = w * t + m * (oftype(float(w * t), π) / 2)
        s, co = sincos(ang)
        wm = w^m
        acc += wm * (c.a[n] * co + c.b[n] * s)
    end
    return acc
end

@inline function derivative(c::FourierControl, t::Real, ::Derivative{N}) where {N}
    return _fourier_deriv(c, t, N)
end
@inline function derivative(c::FourierControl{T}, t::Real, ::DerivativeUpTo{N}) where {T,N}
    return SVector(ntuple(m -> _fourier_deriv(c, t, m - 1), Val(N + 1)))
end

# ---------------------------------------------------------------------------------------------
# FunctionControl
# ---------------------------------------------------------------------------------------------

"""
    FunctionControl{T}(f)

Wrap a user callback `f(t, order::Int)` that returns the `order`-th time-derivative of the
control as a value of type `T`.  The return value is asserted to be `T`, which keeps the
control type-stable even if `f`'s own inference is imperfect.

By default the closure type `F` is kept (the fastest option, but `F` appears in the
control's type).  Loading the `FunctionWrappers` extension enables the opt-in,
*type-erased* constructor [`erase_type`](@ref).
"""
struct FunctionControl{T,F} <: AbstractControl{T}
    f::F
end

function FunctionControl{T}(f::F) where {T,F}
    return FunctionControl{T,F}(f)
end

@inline function derivative(c::FunctionControl{T}, t::Real, ::Derivative{N}) where {T,N}
    return c.f(t, N)::T
end

@inline function derivative(c::FunctionControl{T}, t::Real, ::DerivativeUpTo{N}) where {T,N}
    return SVector(ntuple(m -> c.f(t, m - 1)::T, Val(N + 1)))
end

"""
    erase_type(f, ::Type{T}; timetype = Float64) -> FunctionControl

Build a *type-erased* [`FunctionControl`](@ref) from the callback `f(t, order::Int)::T`,
wrapping it in a `FunctionWrappers.FunctionWrapper` so the concrete closure type does not
appear in the control's type.  Requires `using FunctionWrappers` (provided by a package
extension); without it, a helpful error is raised.
"""
function erase_type(args...; kwargs...)
    return error(
        "ControlledOperators.erase_type requires FunctionWrappers.jl. " *
        "Run `using FunctionWrappers` to enable the type-erased FunctionControl constructor.",
    )
end

# ---------------------------------------------------------------------------------------------
# ScaledControl
# ---------------------------------------------------------------------------------------------

"""
    ScaledControl(s, control)

A control whose every derivative is `s` times that of `control`: `(s·c)(t) = s · c(t)`.  Used
internally to implement subtraction (`s = -1`) and scalar multiplication of controlled operators and
operators.
"""
struct ScaledControl{T,C<:AbstractControl} <: AbstractControl{T}
    factor::T
    control::C
    # Explicit inner constructor suppresses the auto-generated outer one (which would clash
    # with the promoting constructor below) and converts the factor to the element type `T`.
    function ScaledControl{T,C}(factor, control) where {T,C<:AbstractControl}
        return new{T,C}(convert(T, factor), control)
    end
end

function ScaledControl(s, c::AbstractControl{Tc}) where {Tc}
    T = promote_type(typeof(s), Tc)
    return ScaledControl{T,typeof(c)}(s, c)
end

@inline function derivative(sc::ScaledControl, t::Real, ::Derivative{N}) where {N}
    return sc.factor * derivative(sc.control, t, Derivative{N}())
end

@inline function derivative(sc::ScaledControl, t::Real, d::DerivativeUpTo)
    return sc.factor .* derivative(sc.control, t, d)
end
