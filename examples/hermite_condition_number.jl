using FilonResearch, CairoMakie
using LinearAlgebra: cond

function interpolating_mat(a, b, n_deriv)
    degree = 2*n_deriv + 1
    a_mat = derivative_monomial_matrix(a, n_deriv, degree)
    b_mat = derivative_monomial_matrix(b, n_deriv, degree)
    LHS = vcat(a_mat, b_mat)
    return LHS
end

function main(n_deriv)
    as = LinRange(-1.5, -0.5, 100)
    bs = LinRange(0.5, 1.5, 100)

    f(a,b) = interpolating_mat(a, b, n_deriv) |> cond |> log10

    fig = Figure()
    ax = Axis(
        fig[1,1],
        xlabel="a",
        ylabel="b",
        title="(Log10) Condition Number for interpolating f(a), f'(a), ..., f(b), f'(b), ..., up to $n_deriv",
    )
    #ax = Axis(f[1, 1])
    hm = heatmap!(ax, as, bs, f)
    Colorbar(fig[:,end+1], hm)
    return fig
end

function main2(upto_n_deriv=5)
    as = LinRange(-1.5, -0.5, 100)
    bs = LinRange(0.5, 1.5, 100)

    f(x, n_deriv) = interpolating_mat(-x, x, n_deriv) |> cond
    Δts = 10.0 .^ (-5:5)

    fig = Figure()
    ax = Axis(
        fig[1,1],
        xlabel="Log10(x)",
        ylabel="Log10(Condition Number)",
        title="Condition Number for interpolating f(-x), f'(-x), ..., f(x), f'(x), ...",
        yminorticks = 0:20,
        yminorticksvisible = true,
        yminorgridvisible = true,
    )
    for n_deriv in 0:upto_n_deriv
        lines!(ax, log10.(Δts), log10.(f.(Δts, n_deriv)), label="$n_deriv derivatives")
    end
    ylims!(ax, 0, 20)
    Legend(fig[1,end+1], ax)
    return fig
end


