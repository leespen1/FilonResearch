"""
Compute the moments

    \\mu_n(x) = \\int_a^b x^n e^{iwx} dx

and store them in a vector.
"""
function filon_moments(degree::Integer, w::Real, a::Real=-1, b::Real=1)
    @assert degree >= 0 "degree must be non-negative."
    T = promote_type(typeof(float(w)), typeof(float(a)), typeof(float(b)))
    moments = Vector{Complex{T}}(undef, 1+degree)

    if (w == 0)
        for n in 0:degree
            moments[1+n] = (b^(n+1) - a^(n+1)) / (n+1)
        end
    elseif USE_TAYLOR_MOMENTS[] && abs(w) < 1.0
        # Taylor series for small w to avoid catastrophic cancellation in the
        # oscillatory recurrence. Controlled by USE_TAYLOR_MOMENTS toggle. Uses:
        #   μ_n(w) = ∫_a^b x^n e^{iwx} dx = Σ_{k=0}^{K} (iw)^k/k! × (b^{n+k+1} - a^{n+k+1})/(n+k+1)
        # For |w| < 2 and K = 30, the truncation error is |w|^{K+1}/(K+1)! < 8e-24.
        K = 30
        for n in 0:degree
            val = zero(Complex{T})
            iw_power = one(Complex{T})  # (iw)^k / k!, starting at k=0
            for k in 0:K
                monomial_moment = (b^(n+k+1) - a^(n+k+1)) / (n+k+1)
                val += iw_power * monomial_moment
                iw_power *= im*w / (k+1)
            end
            moments[1+n] = val
        end
    else
        exp_a = cis(w*a)
        exp_b = cis(w*b)

        moments[1] = (-im/w)*(exp_b - exp_a)
        for n in 1:degree
            moments[1+n] = (-im/w)*(exp_b*b^n - exp_a*a^n) + (im*n/w)*moments[1+n-1]
        end
    end
    return moments
end

# The Hermite cardinal polynomials ℓ_{k,j} on [a,b] depend only on (s, a, b), NOT
# on the frequency ω — but `filon_weights` is called once per frequency (and, for
# the controlled methods, once per carrier on top of that).  Rebuilding the
# cardinals (an LU solve, in `hermite_interpolating_polynomial`) on every call
# dominated the weight-setup cost, so cache their coefficient vectors by (s,a,b).
const _CARDINAL_CACHE = Dict{Tuple{Int,Float64,Float64},
                             Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}}()
const _CARDINAL_LOCK = ReentrantLock()

# Left/right Hermite cardinal coefficient vectors for order `s` on `[a,b]`:
# `ca[1+i]` interpolates f⁽ⁱ⁾(a) = δ (zero at b); `cb[1+i]` the mirror at b.
function _filon_cardinal_coeffs(s::Integer, a::Real, b::Real)
    key = (Int(s), Float64(a), Float64(b))
    lock(_CARDINAL_LOCK) do
        get!(_CARDINAL_CACHE, key) do
            ca = Vector{Vector{Float64}}(undef, 1+s)
            cb = Vector{Vector{Float64}}(undef, 1+s)
            fa = zeros(Int, 1+s); fb = zeros(Int, 1+s)
            for i in 0:s
                fa .= 0; fb .= 0; fa[1+i] = 1
                ca[1+i] = Polynomials.coeffs(hermite_interpolating_polynomial(a, b, fa, fb))
                fa .= 0; fb .= 0; fb[1+i] = 1
                cb[1+i] = Polynomials.coeffs(hermite_interpolating_polynomial(a, b, fa, fb))
            end
            return (ca, cb)
        end
    end
end

"""
Compute the weights

    b_{k,j}(\\omega) = I_\\omega[\\ell_{k,j}] = \\int_a^b \\ell_{k,j}(x)e^{i\\omega x} dx

The Hermite cardinal polynomials ℓ_{k,j} are frequency-independent and cached by
(s, a, b); only the moments μ_n(ω) are recomputed per call.
"""
function filon_weights(w::Real, s::Integer, a::Real=-1, b::Real=1)
    degree = 2*s+1
    moments = filon_moments(degree, w, a, b)
    T = promote_type(typeof(float(w)), typeof(float(a)), typeof(float(b)))
    ca, cb = _filon_cardinal_coeffs(s, a, b)

    weights_a = Vector{Complex{T}}(undef, 1+s)
    weights_b = Vector{Complex{T}}(undef, 1+s)
    # Manual dot over the overlapping length (coeffs may be shorter than moments):
    # the two-iterable `mapreduce(*, +, moments, coeffs)` allocated on every call.
    for i in 0:s
        weights_a[1+i] = _dot_min(moments, ca[1+i])
        weights_b[1+i] = _dot_min(moments, cb[1+i])
    end
    return weights_a, weights_b
end

# Σ u[k]·v[k] over the overlapping length, allocation-free.
@inline function _dot_min(u, v)
    acc = zero(promote_type(eltype(u), eltype(v)))
    @inbounds for k in 1:min(length(u), length(v))
        acc += u[k] * v[k]
    end
    return acc
end

"""
Compute the weights
    
    b_{k,j}(\\omega) = I_\\omega[\\ell_{k,j}] = \\int_a^b \\ell_{k,j}(x)e^{i\\omega x} dx

For multiple frequencies. Store them in a vector of vectors, where the outer
vector changes the "derivative order" j.
"""
function filon_weights(ωs::AbstractVector{<: Real}, s::Integer, a::Real=-1, b::Real=1)
    T = promote_type(float(eltype(ωs)), typeof(float(a)), typeof(float(b)))
    nfreq = length(ωs)
    weights_a = [zeros(Complex{T}, nfreq) for _ in 0:s]
    weights_b = [zeros(Complex{T}, nfreq) for _ in 0:s]
    for j in 1:nfreq 
        scalar_weights_a, scalar_weights_b = filon_weights(ωs[j], s, a, b)
        for i in eachindex(scalar_weights_a, scalar_weights_b, weights_a, weights_b)
            weights_a[i][j] = scalar_weights_a[i]
            weights_b[i][j] = scalar_weights_b[i]
        end
    end
    return weights_a, weights_b
end

"""
Generate Filon weights (on the interval [-1,1])
"""
function hardcoded_filon_weights(w::Real, s::Integer)
    T = typeof(float(w))

    b_20 = zero(Complex{T}) # For type stability
    b_21 = zero(Complex{T})
    right_weights = zeros(Complex{T}, 1+s)

    if s == 0
        if w == 0
            b_20 += 1
        else
            #b_10 = im*exp(-im*w)/w - im*sin(w)/w^2
            b_20 += -im*exp(im*w)/w + im*sin(w)/w^2
        end
        right_weights[1] = b_20
    elseif s == 1
        if w == 0
            #b_10 = 1
            b_20 += 1
            #b_11 = 1/3
            b_21 += -1/3 
        else
            #b_10 = exp(-w*im)im/w
            #b_10 += 3im*cos(w)/(w^3)
            #b_10 += -3im*sin(w)/(w^4)
            b_20 += -im*exp(w*im)/w
            b_20 += -3im*cos(w)/(w^3)
            b_20 += 3im*sin(w)/(w^4)

            #b_11 = -exp(-w*im)/(w^2)
            #b_11 += im*(2*exp(-im*w) + exp(im*w))/(w^3)
            #b_11 += -3im*sin(w)/(w^4)

            b_21 += exp(w*im)/(w^2)
            b_21 += im*(exp(-im*w) + 2*exp(im*w))/(w^3)
            b_21 += -3im*sin(w)/(w^4)
        end
        right_weights[1] = b_20
        right_weights[2] = b_21
    else
        throw(DomainError("s must be 0 or 1 for hard-coded Filon weights"))
    end
    left_weights = [(-1)^j * conj(right_weights[1+j]) for j in 0:s]
    return left_weights, right_weights
end
