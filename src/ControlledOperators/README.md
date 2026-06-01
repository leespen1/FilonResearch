# ControlledOperators.jl

A small, self-contained implementation of **time-dependent operators**

```
A(t) = Σₖ cₖ(t) · Aₖ
```

for quantum optimal control. It mirrors the vocabulary of
[QuantumPropagators.jl](https://github.com/JuliaQuantumControl/QuantumPropagators.jl) —
a symbolic **`Generator`** vs. a realized **`Operator`**, connected by **`evaluate`** /
**`evaluate!`** — but is an independent implementation with **no runtime dependency** on it.

The design goal is a clean, **type-stable, allocation-free** realization layer: build a
`Generator` once from scalar controls `cₖ(t)` and constant matrices `Aₖ`, then `evaluate` it
at a time `t` into an `Operator` that **shares** (never copies) the matrices and applies via
non-allocating `mul!` inside an iterative solver, or `materialize`s for a direct solve.

## Installation

```julia
pkg> dev /path/to/ControlledOperators
```

Runtime dependencies are only `StaticArrays`, `LinearAlgebra` and `SparseArrays`. Two optional
package extensions add a type-erased control constructor (via `FunctionWrappers`) and a
Taylor-mode derivative fallback (via `TaylorDiff`).

## Quick start — a two-term operator `H(t) = H₀ + ε(t)·H₁`

```julia
using ControlledOperators, LinearAlgebra

# Drift H₀ (constant coefficient 1) and a Fourier control ε(t) = sin(2t) on H₁.
H₀ = [1.0 0.0; 0.0 -1.0]
H₁ = [0.0 1.0; 1.0 0.0]

drift = ConstantControl(1.0)
ε     = FourierControl(0.0, [0.0], [1.0], 2.0)     # a0=0, cos-coeffs=[0], sin-coeffs=[1], ω=2

gen = Generator((drift, ε), [H₀, H₁])              # symbolic A(t) = H₀ + ε(t)·H₁

op  = evaluate(gen, 0.5)                            # realize at t = 0.5  (shares H₀, H₁)

x = [1.0, 2.0]
y = similar(x)
mul!(y, op, x)                                      # non-allocating  y = A(0.5)·x
op * x                                              # convenience      A(0.5)·x
Matrix(op)                                          # materialize Σ cₖ·Aₖ for a direct solve
```

`op` is an `AbstractMatrix`, so it drops straight into Krylov.jl:

```julia
using Krylov
sol, stats = Krylov.cg(op, y)                       # op used directly as the linear operator
```

### In-place refresh across solver iterations

At a fixed time, refresh only the coefficients (zero allocation), reusing the same operator:

```julia
op = Operator(gen, 0.0)            # reusable, mutable coefficient buffer
for t in times
    evaluate!(op, gen, t)          # in-place; matrices untouched
    # … use mul!(y, op, x) …
end
```

### Time-derivatives of the operator

`Aᴺ(t)` is structurally identical — same matrices, the `N`-th derivative of each control:

```julia
dA = evaluate(gen, 0.5, Derivative{1}())            # realized Ȧ(0.5)
dA.matrices === gen.matrices                         # true — matrices are shared, never copied
```

## Static (fully unrolled, isbits) layout

Use a `Tuple` of `SMatrix` for small fixed-size problems; `evaluate`, `mul!` and `*` are then
fully unrolled and allocate nothing:

```julia
using StaticArrays
S₀ = SMatrix{2,2}(H₀); S₁ = SMatrix{2,2}(H₁)
sgen = Generator((drift, ε), (S₀, S₁))
sop  = evaluate(sgen, 0.5)
sop * SVector(1.0, 2.0)                              # returns an SVector, no heap allocation
```

## Controls

| Control | Meaning |
|---|---|
| `ConstantControl(v)` | `c(t) = v`, zero derivatives (use `ConstantControl(1)` for a drift) |
| `FourierControl(a0, a, b, ω)` | `a0 + Σ aₙcos(nωt) + bₙsin(nωt)`, analytic derivatives of all orders |
| `FunctionControl{T}(f)` | wraps a user callback `f(t, order::Int)::T` |
| `ScaledControl(s, c)` | `s · c(t)` (used internally by `-` and scalar `*`) |

Each control implements the single interface method

```julia
derivative(control, t, order)
```

where `order` is a compile-time singleton — `Derivative{N}()` for the `N`-th derivative
(`N = 0` is plain evaluation) or `DerivativeUpTo{N}()` for an `SVector{N+1}` of derivatives
`0..N` in one pass.

### Optional extensions

* **FunctionWrappers** — `erase_type(f, T)` builds a *type-erased* `FunctionControl`, hiding
  the closure type behind a `FunctionWrapper` (opt-in; raw closures are the faster default).
* **TaylorDiff** — supplies a generic `derivative` fallback (via Taylor-mode AD) for any
  control that only defines the `Derivative{0}` evaluation.

```julia
using FunctionWrappers                # enables erase_type
c = erase_type((t, n) -> sin(t + n*π/2), Float64)

using TaylorDiff                      # enables the derivative fallback
struct MyRamp <: AbstractControl{Float64}; α::Float64; end
ControlledOperators.derivative(c::MyRamp, t, ::Derivative{0}) = exp(c.α*t)
derivative(MyRamp(0.7), 0.4, DerivativeUpTo{3}())     # higher orders via Taylor mode
```

## Quality bar

The test suite **asserts** type stability (`@inferred`) and zero allocation (`@allocated == 0`
and `AllocCheck`) on every in-place path, no dynamic dispatch on the hot paths (`JET`),
finite-difference correctness of all analytic derivatives, matrix reuse, and a Krylov solve.
Run it with:

```julia
pkg> test ControlledOperators
```
