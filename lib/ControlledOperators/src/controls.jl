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

# ---------------------------------------------------------------------------------------------
# CarrierControl
# ---------------------------------------------------------------------------------------------

"""
    CarrierControl(envelope, ωc)

A control `c(t) = envelope(t) · e^{i ωc t}`: a slowly-varying `envelope` (any
[`AbstractControl`](@ref)) modulated by a complex-exponential *carrier wave* of frequency `ωc`.
Its value type is complex.  Time-derivatives follow the Leibniz rule

    c⁽ᴺ⁾(t) = e^{i ωc t} Σ_{i=0}^N binom(N,i) · envelope⁽ⁱ⁾(t) · (i ωc)^{N-i}.

Unlike folding the carrier into a generic control, a `CarrierControl` keeps the `envelope` and
`ωc` *recoverable* (via [`envelope`](@ref) and [`carrier_frequency`](@ref)).  Methods that must
treat the envelope and carrier separately — notably the controlled Filon method, which gives
each control term its own oscillatory ansatz — rely on that.
"""
struct CarrierControl{T,Tω<:Real,E<:AbstractControl} <: AbstractControl{T}
    envelope::E
    ωc::Tω
end

function CarrierControl(env::AbstractControl, ωc::Real)
    Te = eltype(env)
    T = Complex{promote_type(real(Te), typeof(float(ωc)))}
    return CarrierControl{T,typeof(ωc),typeof(env)}(env, ωc)
end

@inline function derivative(c::CarrierControl{T}, t::Real, ::Derivative{N}) where {T,N}
    env = derivative(c.envelope, t, DerivativeUpTo{N}())   # SVector{N+1}: envelope⁽⁰⁾ … envelope⁽ᴺ⁾
    iω = im * c.ωc
    acc = zero(T)
    @inbounds for i in 0:N
        acc += binomial(N, i) * env[i+1] * iω^(N - i)
    end
    return acc * cis(c.ωc * t)
end

@inline function derivative(c::CarrierControl{T}, t::Real, ::DerivativeUpTo{N}) where {T,N}
    env = derivative(c.envelope, t, DerivativeUpTo{N}())   # envelope⁽⁰⁾ … envelope⁽ᴺ⁾
    iω = im * c.ωc
    e = cis(c.ωc * t)
    # The p-th carrier derivative is e^{iωt} Σ_{i=0}^p binom(p,i) envelope⁽ⁱ⁾ (iω)^{p-i}.
    # Nothing here depends on the tuple index at the type level, so this stays type-stable.
    return SVector(ntuple(Val(N + 1)) do p1
        acc = zero(T)
        @inbounds for i in 0:(p1 - 1)
            acc += binomial(p1 - 1, i) * env[i+1] * iω^(p1 - 1 - i)
        end
        e * acc
    end)
end

# ---------------------------------------------------------------------------------------------
# Envelope / carrier introspection — a uniform interface over all controls
# ---------------------------------------------------------------------------------------------

"""
    carrier_frequency(control) -> ωc

The carrier-wave frequency of a control: `ωc` for a [`CarrierControl`](@ref), and `0` for any
other control (no carrier).  Lets a method scan a heterogeneous set of controls uniformly.
"""
carrier_frequency(::AbstractControl) = 0
carrier_frequency(c::CarrierControl) = c.ωc

"""
    envelope(control) -> control

The slowly-varying envelope of a control: the wrapped envelope for a [`CarrierControl`](@ref),
and the control itself otherwise (its envelope *is* itself, with zero carrier).
"""
envelope(c::AbstractControl) = c
envelope(c::CarrierControl) = c.envelope

# ---------------------------------------------------------------------------------------------
# SumControl
# ---------------------------------------------------------------------------------------------

"""
    SumControl(controls...)
    SumControl(controls::Tuple)

A control equal to the sum of its component controls: `c(t) = Σₗ cₗ(t)`, with every derivative
likewise the sum of the components' derivatives.  The components are held in a `Tuple` so that
a heterogeneous mix (e.g. several [`CarrierControl`](@ref)s at different frequencies) stays
type-stable, and the summing recursion is unrolled at compile time.

This is the natural representation of a single control pulse built from several carrier waves,

    cₖ(t) = Σₗ c̃ₖ,ₗ(t) · e^{i νₖ,ₗ t},

all multiplying the *same* control matrix.  A `SumControl` keeps the individual carriers
recoverable through [`components`](@ref); the efficient controlled Filon method reaches into the
components for their separate envelopes and carrier frequencies, while [`derivative`](@ref) of
the sum supplies the combined coefficient (and its time-derivatives) for the operator's matvec.
"""
struct SumControl{T,CC<:Tuple} <: AbstractControl{T}
    controls::CC
    function SumControl{T,CC}(controls) where {T,CC<:Tuple}
        return new{T,CC}(controls)
    end
end

function SumControl(controls::Tuple)
    isempty(controls) && throw(ArgumentError("SumControl needs at least one component control"))
    all(c -> c isa AbstractControl, controls) ||
        throw(ArgumentError("SumControl components must all be AbstractControls"))
    T = promote_type(map(eltype, controls)...)
    return SumControl{T,typeof(controls)}(controls)
end

SumControl(controls::AbstractControl...) = SumControl(controls)

# The natural promoted type of the sum is exactly `T`, but we deliberately do *not* assert it:
# under the Taylor-mode `derivative` fallback `t` may be a `TaylorScalar`, so the realized type
# differs from the nominal value type while differentiating through `t`.
@inline function derivative(sc::SumControl, t::Real, d::Derivative)
    return _sum_derivative(sc.controls, t, d)
end

@inline function derivative(sc::SumControl, t::Real, d::DerivativeUpTo)
    return _sum_derivative(sc.controls, t, d)
end

# Compile-time-unrolled sum over the component tuple (same idea as `_write_coeffs!`): the tuple
# type encodes its length, so this expands to straight-line additions with no runtime recursion.
@inline _sum_derivative(controls::Tuple{Any}, t, d) = derivative(controls[1], t, d)
@inline _sum_derivative(controls::Tuple, t, d) =
    derivative(first(controls), t, d) + _sum_derivative(Base.tail(controls), t, d)

"""
    components(control) -> Tuple

The component controls of a [`SumControl`](@ref).  For any other control the "sum" is the
control itself, so a one-element tuple `(control,)` is returned — letting a method iterate the
carriers of a control entry uniformly, whether or not it is a sum.
"""
components(sc::SumControl) = sc.controls
components(c::AbstractControl) = (c,)
