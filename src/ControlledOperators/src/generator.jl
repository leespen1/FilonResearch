# ---------------------------------------------------------------------------------------------
# Generator — the symbolic A(t) = Σₖ cₖ(t)·Aₖ
# ---------------------------------------------------------------------------------------------

"""
    Generator(controls, matrices)

A *symbolic* time-dependent operator

    A(t) = Σₖ controls[k](t) · matrices[k].

`controls` and `matrices` are two **parallel containers** (a struct-of-arrays layout).  The
two containers are independent and may be:

* a `Tuple` (or `SVector`) — the *static* layout; heterogeneous control types stay fully
  type-stable because the container is unrolled.  Matrices are typically `SMatrix`.
* a `Vector` — the *dynamic* layout, for many same-type entries or large dense/sparse
  matrices.

For type stability you must keep each container **homogeneous in element type** (all matrices
the same type; for a `Vector` of controls, all the same control type).  A `Tuple` of controls
may freely mix control types (e.g. a [`ConstantControl`](@ref) drift plus several
[`FourierControl`](@ref)s) — that is the recommended way to build a drift+controls generator.

Call [`evaluate`](@ref) to realize the generator at a time `t` into an [`Operator`](@ref); the
realized operator references the *same* matrices (they are never copied).
"""
struct Generator{CC,MC}
    controls::CC
    matrices::MC
    function Generator{CC,MC}(controls::CC, matrices::MC) where {CC,MC}
        length(controls) == length(matrices) || throw(DimensionMismatch(
            "number of controls ($(length(controls))) must equal number of matrices ($(length(matrices)))"))
        if length(matrices) > 0
            s = size(first(matrices))
            all(A -> size(A) == s, matrices) ||
                throw(DimensionMismatch("all matrices must share the same size"))
        end
        return new{CC,MC}(controls, matrices)
    end
end

function Generator(controls::CC, matrices::MC) where {CC,MC}
    return Generator{CC,MC}(controls, matrices)
end

"""
    get_controls(gen) -> controls

Return the container of controls held by `gen`.  Mirrors QuantumPropagators's duck-typed
interface so a generator can be introspected the same way.
"""
function get_controls(gen::Generator)
    return gen.controls
end

function Base.length(gen::Generator)
    return length(gen.matrices)
end

# Concatenate parallel containers, preserving the static/dynamic layout where possible.
function _cat(a::Tuple, b::Tuple)
    return (a..., b...)
end

function _cat(a::Tuple, b)
    return (a..., b...)
end

function _cat(a, b::Tuple)
    return (a..., b...)
end

function _cat(a::AbstractVector, b::AbstractVector)
    return vcat(a, b)
end

function _scale_controls(s, controls::Tuple)
    return map(c -> ScaledControl(s, c), controls)
end

function _scale_controls(s, controls::AbstractVector)
    return [ScaledControl(s, c) for c in controls]
end

"""
    +(g1::Generator, g2::Generator)
    -(g1::Generator, g2::Generator)

Combine generators by **concatenating their term lists**.  Subtraction wraps the controls of
the right-hand generator in [`ScaledControl`](@ref)`(-1, …)` (the matrices are shared, never
copied or negated).  No operator×operator product is defined.
"""
function Base.:+(g1::Generator, g2::Generator)
    return Generator(_cat(g1.controls, g2.controls), _cat(g1.matrices, g2.matrices))
end

function Base.:-(g1::Generator, g2::Generator)
    return Generator(
        _cat(g1.controls, _scale_controls(-1, g2.controls)),
        _cat(g1.matrices, g2.matrices),
    )
end

function Base.:*(s::Number, g::Generator)
    return Generator(_scale_controls(s, g.controls), g.matrices)
end

function Base.:*(g::Generator, s::Number)
    return s * g
end

function Base.:-(g::Generator)
    return (-1) * g
end

function Base.show(io::IO, gen::Generator)
    K = length(gen)
    print(io, "Generator(", K, " term", K == 1 ? "" : "s", ")")
    if K > 0
        A = first(gen.matrices)
        print(io, ": ", size(A, 1), "×", size(A, 2), " ")
        print(io, gen.matrices isa Tuple ? "static" : "dynamic")
        print(io, " {", eltype(A), "}")
    end
end
