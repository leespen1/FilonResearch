"""
Type for faciliting operations of the type
    c₁*A₁ + c₂*A₂ + …
"""
struct ControlledOp{T <: Union{Tuple, AbstractVector}}
    operators::T
    coefficients::Vector{ComplexF64}
    function ControlledOp(operators::T, coefficients::AbstractVector{<: Number}) where {T <: Union{Tuple, AbstractVector}}
        @assert length(operators) == length(coefficients) "Must provide one coefficient for each operator."
        @assert all(x -> hasmethod(*, (typeof(x), Vector{ComplexF64})), operators) "Each operator in operators must be able to multiple a Vector{ComplexF64}"
        new{T}(operators, convert(Vector{ComplexF64}, coefficients))
    end
end

function Base.:*(scalar::Real, controlled_op::ControlledOp)
    return ControlledOp(controlled_op.operators, controlled_op.coefficients .* scalar)
end

function Base.:*(a::ControlledOp, b::AbstractVector{<: Number})
    return sum(
        a.coefficients[i] * (a.operators[i] * b)
        for i in eachindex(a.coefficients, a.operators)
    )
end

function full_op(a::ControlledOp)
    return sum(
        a.coefficients[i] * a.operators[i]
        for i in eachindex(a.coefficients, a.operators)
    )
end

"""
Type for faciliting operations of the type
    c₁(t)*A₁ + c₂(t)*A₂ + …
Works by storing the functions, and functioning as a callable struct which
returns isntances of ControlledOp, which represents 
    c₁(t)*A₁ + c₂(t)*A₂ + …
for a particular value of t.
"""
struct ControlledFunctionOp{T1 <: Union{Tuple, AbstractVector}, T2 <: Tuple} <: Function
    operators::T1 
    coefficient_functions::T2
    function ControlledFunctionOp(operators::T1, coefficient_functions::T2) where {T1 <: Union{Tuple, AbstractVector}, T2 <: Tuple}
        @assert length(operators) == length(coefficient_functions) "Must provide one function for each operator."
        @assert all(x -> isa(x, Function), coefficient_functions) "Each element of A_deriv_funcs must be a function."
        @assert all(x -> hasmethod(x, (Float64,)), coefficient_functions) "Each function `f` in A_deriv_funcs must have a method `f(t::Float64)` defined."
        new{T1, T2}(operators, coefficient_functions)
    end
end

function (self::ControlledFunctionOp)(t::Real)
    coefficients = [f(t) for f in self.coefficient_functions]
    return ControlledOp(self.operators, coefficients)
end

