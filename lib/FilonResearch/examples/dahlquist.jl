using Plots, FilonResearch

function dahlquist_true_solution(λ, u₀, t)
    return exp(λ*t)*u₀
end

function main(λ, ω, n_refinements, s=1, tf=1, u₀=1)
    u_tf_true = dahlquist_true_solution(λ, u₀, tf)
    nsteps_vec = 10 .^ (0:n_refinements)
    solutions = [filon_dahlquist(λ, ω, s, nsteps, tf, u₀) for nsteps in nsteps_vec]
    errors = abs.(solutions .- u_tf_true)

    pl = plot(log10.(nsteps_vec), log10.(errors))
    return pl
end
