# ---------------------------------------------------------------------------------------------
# ControlledOperator — the symbolic A(t) = Σₖ cₖ(t)·Aₖ
# ---------------------------------------------------------------------------------------------

"""
    ControlledOperator(controls, matrices)

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
[`FourierControl`](@ref)s) — that is the recommended way to build a drift-plus-controls
operator.

Call [`evaluate`](@ref) to realize the controlled operator at a time `t` into an
[`Operator`](@ref); the realized operator references the *same* matrices (they are never
copied).
"""
struct ControlledOperator{CC,MC}
    controls::CC
    matrices::MC
    function ControlledOperator{CC,MC}(controls::CC, matrices::MC) where {CC,MC}
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

function ControlledOperator(controls::CC, matrices::MC) where {CC,MC}
    return ControlledOperator{CC,MC}(controls, matrices)
end

"""
    get_controls(co) -> controls

Return the container of controls held by `co`.  Mirrors QuantumPropagators's duck-typed
interface so a controlled operator can be introspected the same way.
"""
function get_controls(co::ControlledOperator)
    return co.controls
end

function Base.length(co::ControlledOperator)
    return length(co.matrices)
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
    +(co1::ControlledOperator, co2::ControlledOperator)
    -(co1::ControlledOperator, co2::ControlledOperator)

Combine controlled operators by **concatenating their term lists**.  Subtraction wraps the
controls of the right-hand operand in [`ScaledControl`](@ref)`(-1, …)` (the matrices are
shared, never copied or negated).  No operator×operator product is defined.
"""
function Base.:+(co1::ControlledOperator, co2::ControlledOperator)
    return ControlledOperator(_cat(co1.controls, co2.controls), _cat(co1.matrices, co2.matrices))
end

function Base.:-(co1::ControlledOperator, co2::ControlledOperator)
    return ControlledOperator(
        _cat(co1.controls, _scale_controls(-1, co2.controls)),
        _cat(co1.matrices, co2.matrices),
    )
end

function Base.:*(s::Number, co::ControlledOperator)
    return ControlledOperator(_scale_controls(s, co.controls), co.matrices)
end

function Base.:*(co::ControlledOperator, s::Number)
    return s * co
end

function Base.:-(co::ControlledOperator)
    return (-1) * co
end

function Base.show(io::IO, co::ControlledOperator)
    K = length(co)
    print(io, "ControlledOperator(", K, " term", K == 1 ? "" : "s", ")")
    if K > 0
        A = first(co.matrices)
        print(io, ": ", size(A, 1), "×", size(A, 2), " ")
        print(io, co.matrices isa Tuple ? "static" : "dynamic")
        print(io, " {", eltype(A), "}")
    end
end
