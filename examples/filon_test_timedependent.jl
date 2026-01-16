using FilonResearch, Plots

s = 1

y0 = ComplexF64[1.0]
α = 10im
β = 0.1im
ω = 3
y(t) = y0 * exp(α*t + β*sin(ω*t))

T = 2.0

# Part 1
A_derivs = (
    t ->  ComplexF64[α + β*ω*cos(ω*t);;],
    t ->  ComplexF64[-β*ω^2*sin(ω*t);;],
    t ->  ComplexF64[-β*ω^3*cos(ω*t);;],
    t ->  ComplexF64[β*ω^4*sin(ω*t);;],
    t ->  ComplexF64[β*ω^5*cos(ω*t);;],
)


frequencies = [imag(α)]
#frequencies = [0]
errors = Float64[]
nsteps_vec = [2^i for i in 0:10]
for nsteps in nsteps_vec
    ts = LinRange(0, T, 1+nsteps)
    local num_sol = filon_solve(A_derivs, y0, frequencies, T, nsteps, s)
    local true_sol = reduce(hcat, y.(ts))
    #display(hcat(sol, true_sol))
    error = sum(abs, num_sol - true_sol)
    push!(errors, error)
end
error_ratios = [errors[i] / errors[i+1] for i in 1:length(errors)-1]
cvg_orders = log2.(error_ratios)
@show round.(errors, sigdigits=2)
@show round.(cvg_orders, sigdigits=2)

pl_nsteps = 1000
pl_ts = LinRange(0, T, 1+pl_nsteps)
true_sol = [y(t)[1] for t in pl_ts]
pl = plot(pl_ts, real.(true_sol), label="True Sol")

num_sol = filon_solve(A_derivs, y0, frequencies, T, pl_nsteps, s)[1,:]
plot!(pl, pl_ts, real.(num_sol), label="Numerical Sol")
display(pl)


