"""
Type for faciliting operations of the type
    c₁*A₁ + c₂*A₂ + …
"""
struct ControlledOp{OpT, CoeffT}
    operators::Vector{OpT} 
    coefficients::Vector{CoeffT}
end
# TODO: make inner constructor that checks for same indices on operators
# and control coefficients.



function Base.:*(a::ControlledOp{OpT, CoeffT}, b)
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
