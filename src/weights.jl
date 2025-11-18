"""

"""
function convert_weights_to_other_side(weights)
    # make tuple (1, -1, 1, -1, ...)
    other_side_weights = map(enumerate(weights)) do (i, w)
        ifelse(isodd(i-1), -1, 1) * conj(w)
    end
    return other_side_weights
end


"""
Tuple version (some care is needed to make sure output is a tuple).
"""
function convert_weights_to_other_side(weights::NTuple{N,T}) where {N,T}
    # make tuple (1, -1, 1, -1, ...)
    sign_tup = ntuple(i -> ifelse(isodd(i-1), -1, 1), N)
    sign_tup = ntuple(i -> (isodd(i-1) ? -1 : 1),  N)
    return sign_tup .* conj.(weights)
end


"""
Generate s=0 (second-order) Filon Weights
"""
function second_order_weights(ω)
    b_20 = zero(ComplexF64) # For type stability

    if ω == 0
        b_20 += 1.0
    else
        #b_10 = im*exp(-im*ω)/ω - im*sin(ω)/ω^2
        b_20 += -im*exp(im*ω)/ω + im*sin(ω)/ω^2
    end
    right_weights = tuple(b_20)
    left_weights = convert_weights_to_other_side(right_weights)
    return left_weights, right_weights
end

"""
Generate s=1 (fourth-order) Filon Weights
"""
function fourth_order_weights(ω)
    b_20 = zero(ComplexF64) # For type stability
    b_21 = zero(ComplexF64)

    if ω == 0
        #b_10 = 1
        b_20 += 1
        #b_11 = 1/3
        b_21 += -1/3 
    else
        #b_10 = exp(-ω*im)im/ω
        #b_10 += 3im*cos(ω)/(ω^3)
        #b_10 += -3im*sin(ω)/(ω^4)
        b_20 += -im*exp(ω*im)/ω
        b_20 += -3im*cos(ω)/(ω^3)
        b_20 += 3im*sin(ω)/(ω^4)

        #b_11 = -exp(-ω*im)/(ω^2)
        #b_11 += im*(2*exp(-im*ω) + exp(im*ω))/(ω^3)
        #b_11 += -3im*sin(ω)/(ω^4)

        b_21 += exp(ω*im)/(ω^2)
        b_21 += im*(exp(-im*ω) + 2*exp(im*ω))/(ω^3)
        b_21 += -3im*sin(ω)/(ω^4)
    end


    right_weights = (b_20, b_21)
    left_weights = convert_weights_to_other_side(right_weights)
    return left_weights, right_weights
end
