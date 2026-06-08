"""
    Derivative{N}

Compile-time singleton selecting the `N`-th time-derivative as the `order` argument of
[`derivative`](@ref) and [`evaluate`](@ref).  `Derivative{0}()` denotes plain evaluation.

Because `N` is encoded in the type, dispatch on `order` happens at compile time and
`Derivative{0}` is *more specific* than the generic `Derivative{N}` — so a control may give
a fast path for evaluation and a generic path for higher derivatives.

```jldoctest
julia> Derivative{2}()
Derivative{2}()
```
"""
struct Derivative{N} end

"""
    DerivativeUpTo{N}

Compile-time singleton requesting *all* derivatives of orders `0..N` returned together as an
`SVector{N+1}`, ideally computed in a single pass.  Pass as the `order` argument of
[`derivative`](@ref).

```jldoctest
julia> DerivativeUpTo{3}()
DerivativeUpTo{3}()
```
"""
struct DerivativeUpTo{N} end

# Convenience constructors from a runtime integer.  These are only type-stable when `N` is a
# constant the compiler can see; in hot code prefer writing `Derivative{2}()` directly.
@inline function Derivative(N::Integer)
    return Derivative{Int(N)}()
end

@inline function DerivativeUpTo(N::Integer)
    return DerivativeUpTo{Int(N)}()
end

function Base.show(io::IO, ::Derivative{N}) where {N}
    return print(io, "Derivative{", N, "}()")
end

function Base.show(io::IO, ::DerivativeUpTo{N}) where {N}
    return print(io, "DerivativeUpTo{", N, "}()")
end
