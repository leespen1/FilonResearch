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

    #=
    # Old, hardcoded derivatives of f(x)
    f(x) = x / (1+x^2)
    df(x) = (1 - x^2) / (1+x^2)^2
    df²(x) = (8*x^3/(1+x^2)^3) - (6*x/(1+x^2)^2)
    df³(x) = -6/(x^2+1)^2 + 48/(x^2+1)^3 - 48/(x^2+1)^4
    fa_derivs = [f(a), df(a), df²(a), df³(a)]
    fb_derivs = [f(b), df(b), df²(b), df³(b)]
    =#

    if maximum(s_range) > 5
        @warn "maximum(s_range) > 5. Results for s > 5 are unreliable, due to polynoimal coefficients sizes pushing the limits of Int64 storage."
    end
    
    f = Polynomial([0, 1]) // Polynomial([1,0,1])
    fa_derivs = [derivative(f, s)(a) for s in 0:maximum(s_range)]
    fb_derivs = [derivative(f, s)(b) for s in 0:maximum(s_range)]
    @show fa_derivs fb_derivs

    Ei(x) = -expint(-x)
    indefinite_integral(x, ω) = ifelse(
        ω == 0,
        0.5*log(x^2+1), # ω = 0 case
        0.5*(exp(ω)*Ei(im*ω*(x+im)) + exp(-ω)*Ei(im*x*ω + ω)) # ω != 0 case
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

function main(ω_range=1:200, s_range=0:3, a=-1, b=1, relative=false)
    envelope(x) = x / (1+x^2)
    integrand(x, ω) = envelope(x) * exp(im*ω*x) |> real


    xs = LinRange(a, b, 10_000)

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

    errors_mat, true_sols = collect_errors(ω_range, s_range, a, b)

    fig2 = Figure(size=(10inch, 5inch))
    ax_errors = Axis(
        fig2[1,1],
        xlabel=L"\omega",
        ylabel=L"\log_{10}(\textrm{Absolute Error})",
        title=L"\textbf{Filon Method: Approximation Error vs Frequency},\quad \int_{%$(a)}^{%$(b)} \frac{x}{(1+x)^2} e^{i \omega x}dx",
        xticks=0:20:200,
        yticks = 0:-2:-16,
        xminorticks=0:10:200,
        yminorticks=0:-1:-16,
        xminorticksvisible=true,
        yminorticksvisible=true,
        yaxisposition=:right,
        limits=((0, maximum(ω_range)), (-16,0)),
        #limits=(nothing, (-16,0)),
    )

    graph_errors_mat = replace(x -> iszero(x) ? NaN : x, errors_mat)
    for i in 1:size(errors_mat, 2)
        lines!(ax_errors, ω_range, log10.(graph_errors_mat[:,i]), label="$(i-1)")
    end
    axislegend(ax_errors, "Number of Derivatives Taken", position=:rt, orientation=:horizontal)



    fig3 = Figure(size=(10inch, 5inch))
    ax_rel_errors = Axis(
        fig3[1,1],
        xlabel=L"\omega",
        ylabel=L"\log_{10}(\textrm{Relative Error})",
        title=L"\textbf{Filon Method: Approximation Error vs Frequency},\quad \int_{%$(a)}^{%$(b)} \frac{x}{(1+x)^2} e^{i \omega x}dx",
        xticks=0:20:200,
        yticks = 0:-2:-16,
        xminorticks=0:10:200,
        yminorticks=0:-1:-16,
        xminorticksvisible=true,
        yminorticksvisible=true,
        yaxisposition=:right,
        limits=((0, maximum(ω_range)), (-16,0)),
        #limits=(nothing, (-16,0)),
    )
    graph_rel_errors_mat = graph_errors_mat ./ abs.(true_sols)
    for i in 1:size(errors_mat, 2)
        lines!(ax_rel_errors, ω_range, log10.(graph_rel_errors_mat[:,i]), label="$(i-1)")
    end
    axislegend(ax_rel_errors, "Number of Derivatives Taken", position=:lb, orientation=:horizontal)


    fig4 = Figure(size=(10inch, 5inch))
    ax_true_sol = Axis(
        fig4[1,1],
        xlabel=L"\omega",
        ylabel=L"\log_{10}(\textrm{Analytical Value})",
        title=L"\int_{%$(a)}^{%$(b)} \frac{x}{(1+x)^2} e^{i \omega x}dx",
        xticks=0:20:200,
        yticks = 0:-2:-16,
        xminorticks=0:10:200,
        yminorticks=0:-1:-16,
        xminorticksvisible=true,
        yminorticksvisible=true,
        yaxisposition=:right,
        limits=((0, maximum(ω_range)), (-16,0)),
        #limits=(nothing, (-16,0)),
    )
    lines!(ax_true_sol, ω_range, log10.(abs.(true_sols)))


    mkpath("./Plots/")
    save("./Plots/integrand_a=$(a)_b=$(b).svg", fig)
    save("./Plots/integrand_a=$(a)_b=$(b).png", fig)
    save("./Plots/error_a=$(a)_b=$(b).svg", fig2)
    save("./Plots/error_a=$(a)_b=$(b).png", fig2)
    save("./Plots/rel_error_a=$(a)_b=$(b).svg", fig3)
    save("./Plots/rel_error_a=$(a)_b=$(b).png", fig3)
    save("./Plots/true_sol_a=$(a)_b=$(b).svg", fig4)
    save("./Plots/true_sol_a=$(a)_b=$(b).png", fig4)

    return fig, fig2, errors_mat
end
    



