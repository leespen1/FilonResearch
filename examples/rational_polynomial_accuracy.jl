using FilonResearch
using SpecialFunctions: expint
#using GLMakie
using CairoMakie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

using LaTeXStrings
"""
Plot accuracy of the explicit Filon method when evaluating the integral

    \\int_{-1}^{1} f(x)e^{i \\omega x} dx

with f(x) = x / (1+x^2).
"""
function collect_errors(ω_range=0:10, s_range=0:1, a=-1, b=1)

    f(x) = x / (1+x^2)
    df(x) = (1 - x^2) / (1+x^2)^2
    df²(x) = (8*x^3/(1+x^2)^3) - (6*x/(1+x^2)^2)
    df³(x) = -6/(x^2+1)^2 + 48/(x^2+1)^3 - 48/(x^2+1)^4

    fa_derivs = [f(a), df(a), df²(a), df³(a)]
    fb_derivs = [f(b), df(b), df²(b), df³(b)]

    Ei(x) = -expint(-x)
    indefinite_integral(x, ω) = ifelse(
        ω == 0,
        0.5*log(x^2+1),
        0.5*(exp(ω)*Ei(im*ω*(x+im)) + exp(-ω)*Ei(im*x*ω + ω))
    )
    true_solution(ω) = indefinite_integral(b, ω) - indefinite_integral(a, ω)
    
    errors_mat = zeros(length(ω_range), length(s_range))

    for (i, ω) in enumerate(ω_range)
        true_sol = true_solution(ω)
        solutions = [explicit_filon_integral(ω, s, a, b, fa_derivs[1:1+s], fb_derivs[1:1+s]) for s in s_range]
        errors = abs.(true_sol .- solutions)
        errors_mat[i,:] .= errors
        #@show i ω true_sol solutions errors
    end

    true_sols = true_solution.(ω_range)

    return errors_mat, true_sols
end

function main(ω_range=1:200, s_range=0:3, a=-1, b=1)
    envelope(x) = x / (1+x^2)
    integrand(x, ω) = envelope(x) * exp(im*ω*x) |> real


    xs = LinRange(-1, 1, 10_000)

    inch = 96
    fig = Figure(size=(10inch, 5inch))

    # Envelop plot
    
    ax_envelope = Axis(fig[1,1], xlabel=L"x", ylabel=L"f(x)", title=L"f(x) = \frac{x}{1+x^2}")
    ax_ω1 = Axis(fig[1,2], xlabel=L"x", ylabel=L"f(x)e^{i\omega x}", title=L"f(x)e^{i\omega x}, \quad \omega=10")
    ax_ω2 = Axis(fig[2,1], xlabel=L"x", ylabel=L"f(x)e^{i\omega x}", title=L"f(x)e^{i\omega x}, \quad \omega=100")
    ax_ω3 = Axis(fig[2,2], xlabel=L"x", ylabel=L"f(x)e^{i\omega x}", title=L"f(x)e^{i\omega x}, \quad \omega=200")

    lines!(ax_envelope, xs, envelope.(xs))
    lines!(ax_ω1, xs, integrand.(xs, 10))
    lines!(ax_ω2, xs, integrand.(xs, 100))
    lines!(ax_ω3, xs, integrand.(xs, 200))
    for ax in (ax_ω1, ax_ω2, ax_ω3)
        lines!(ax, xs, envelope.(xs), color=:black, linestyle=:dash, linewidth=1)
        lines!(ax, xs, -1 .* envelope.(xs), color=:black, linestyle=:dash, linewidth=1)
    end

    fig2 = Figure(size=(10inch, 5inch))
    ax_errors = Axis(fig2[1,1], xlabel=L"\omega", ylabel=L"\log_{10}(\textrm{Error})", title="Approximation Error vs Frequency", yticks = 0:-2:-16, xticks=0:20:200)
    errors_mat, true_sols = collect_errors(ω_range, s_range, a, b)
    replace!(x -> abs(x) < 1e-14 ? NaN : x, errors_mat)
    for i in 1:size(errors_mat, 2)
        lines!(ax_errors, ω_range, log10.(errors_mat[:,i]), label="s=$(i-1)")
    end


    mkpath("./Plots/")
    save("./Plots/integrand.svg", fig)
    save("./Plots/integrand.png", fig)
    save("./Plots/error.svg", fig2)
    save("./Plots/error.png", fig2)

    return fig, fig2
end
    



