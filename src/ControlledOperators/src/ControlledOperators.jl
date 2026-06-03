"""
    ControlledOperators

A small, self-contained implementation of time-dependent operators

    A(t) = Σₖ cₖ(t) · Aₖ

for quantum optimal control.  It mirrors the vocabulary of QuantumPropagators.jl —
[`ControlledOperator`](@ref) (symbolic) vs [`Operator`](@ref) (realized), connected by
[`evaluate`](@ref) / [`evaluate!`](@ref) — but is an independent implementation with no
runtime dependency on it.

The design goal is a clean, type-stable, allocation-free realization layer:

* scalar [`AbstractControl`](@ref)s `cₖ(t)` with analytic [`derivative`](@ref)s selected by
  the compile-time singletons [`Derivative{N}`](@ref) / [`DerivativeUpTo{N}`](@ref);
* a [`ControlledOperator`](@ref) storing controls and matrices as two parallel containers
  (struct-of-arrays), in either a *static* (tuple/`SMatrix`) or *dynamic* (`Vector`) layout;
* an [`Operator`](@ref) that realizes the coefficients while **sharing** the controlled operator's
  matrices, with non-allocating `mul!`/`*`, `materialize`, and a full `AbstractMatrix`
  interface so it drops straight into Krylov.jl or a direct solve.

Two optional package extensions add a type-erased control constructor ([`erase_type`](@ref),
via FunctionWrappers.jl) and a Taylor-mode derivative fallback (via TaylorDiff.jl).

The main use-case for the type-erased control constructor is to turn a
heterogeneous collection of controls in to a homogeneous vector of controls,
which is easier to parse in error stacktraces. However, the evaluation of the
control functions may be slower compared to providing a tuple of heterogeneous
controls.

The Taylor-mode derivative fallback allows for easy creation of new controls by
just programming a function that gives the value of the control function. It is
not necessary to program the derivatives as well. I believe I have done this in
a way that is non-allocating and efficient.
"""
module ControlledOperators

using LinearAlgebra
using SparseArrays
using StaticArrays

export AbstractControl, ConstantControl, FourierControl, FunctionControl, ScaledControl
export CarrierControl, carrier_frequency, envelope
export Derivative, DerivativeUpTo, derivative, erase_type
export ControlledOperator, Operator, get_controls, evaluate, evaluate!
export materialize, materialize!

include("derivatives.jl")
include("controls.jl")
include("controlled_operator.jl")
include("operator.jl")

end # module
