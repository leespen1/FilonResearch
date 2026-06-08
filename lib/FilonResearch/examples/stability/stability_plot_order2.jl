using Plots

#f(z) = (1 + 2im*real(z) - abs2(z)) / abs2(1-im*z)
#f(z) = (1 + real(z)^2 +2im*real(z) - imag(z)^2 ) / abs2(1-im*z)
#f(z) = (1+im*z) / (1-im*z)
f(z) = (1+z) / (1-z)

# Grid in the complex plane
x_range = range(-0.5, 0.5, length=200)
y_range = range(-10, 10, length=200)

# Evaluate |f(z)| on the grid
Z = [abs(f(complex(x,y))) for x in x_range, y in y_range]

# Contour plot
pl1 = contour(x_range, y_range, Z,
    xlabel="Re(z)",
    ylabel="Im(z)",
    title="Contour of |f(z)|"
)


function new_f(ωₖΔt)
    ω̂ₖ = ωₖΔt / 2
    iωₖWᴵ = im * ω̂ₖ * cis(ω̂ₖ) * ((-im*cis(ω̂ₖ) / ω̂ₖ) + im*sin(ω̂ₖ)/(ω̂ₖ)^2)
    return  (1 - conj(iωₖWᴵ)) / (1 - iωₖWᴵ)
end

# Evaluate |f(z)| on the grid
new_Z = [abs(new_f(complex(x,y))) for x in x_range, y in y_range]

# Contour plot
pl2 = contour(x_range, y_range, new_Z,
    xlabel="Re(ωₖΔt)",
    ylabel="Im(ωₖΔt)",
    title="Contour of |f(ωₖΔt)|"
)


function f3(a)
    # (a = ωΔt/2)
    β = cis(a)*((-im*cis(a)/a) + im*sin(a)/a^2)
    return (1+im*a*conj(β)) / (1-im*a*β)
end


α_range = LinRange(-20, 20, 100)
#pl3 = plot(α_range, abs.(f3.(α_range)), xlabel="α", ylabel="|f(α)|")
pl3 = plot(α_range, abs.(f3.(α_range)))


function f4(λ, a)
    β = cis(a)*((-im*cis(a)/a) + im*sin(a)/a^2)
    return (1 + (λ/2)*conj(β)) / (1 - (λ/2)*β)
end


fixed_a = 10.0
Z_λ = [abs(f4(complex(x,y), fixed_a)) for x in x_range, y in y_range]
pl4 = contour(x_range, y_range, Z_λ,
    xlabel="Re(λ)",
    ylabel="Im(λ)",
    title="Contour of |f(λ; α=$fixed_a)|"
)



