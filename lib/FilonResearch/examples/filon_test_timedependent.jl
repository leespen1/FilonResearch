using FilonResearch, Plots, LinearAlgebra, OrdinaryDiffEq

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

# Part 2 - controlled operator framework works exactly the same as hard-coded derivatives
operators = [ComplexF64[1;;], ComplexF64[1;;]]
op_A_derivs = (
            ControlledFunctionOp(operators, (t -> α, t -> β*ω*cos(ω*t))),
            ControlledFunctionOp(operators, (t -> 0, t -> -β*ω^2*sin(ω*t))),
            ControlledFunctionOp(operators, (t -> 0, t -> -β*ω^3*cos(ω*t))),
            ControlledFunctionOp(operators, (t -> 0, t -> β*ω^4*sin(ω*t))),
            ControlledFunctionOp(operators, (t -> 0, t -> β*ω^5*cos(ω*t))),
)


cmp_nsteps = 1000
num_sol_1 = filon_solve(A_derivs, y0, frequencies, T, cmp_nsteps, s)
num_sol_2 = filon_solve(op_A_derivs, y0, frequencies, T, cmp_nsteps, s)
@show maximum(abs, num_sol_1 .- num_sol_2)

# Part 3 Check that Filon converges with the correct order of accuracy for a
# more complicated example (non-scalar, multiple frequencies, etc.)

ω1 = 10
ω2 = 15
ω3 = 20
non_freq_strength = 0.00
control_mat = [1; 2; 3;; 4; 5; 6;; 7; 8; 9]
A(t) = Diagonal([ω1*im, ω2*im, ω3*im]) + cos(t)*non_freq_strength*control_mat
function du(u, p, t)
    return A(t)*u
end
u0 = ComplexF64[1; 2; 3]
tspan = (0.0, 10.0)
prob = ODEProblem(du, u0, tspan)
sol = solve(prob, Tsit5(), abstol=1e-10)


operators = [Array(Diagonal([ω1*im, ω2*im, ω3*im])), control_mat]
op_A_derivs = (
    ControlledFunctionOp(operators, (t -> 1, t -> non_freq_strength*cos(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> -non_freq_strength*sin(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> -non_freq_strength*cos(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> non_freq_strength*sin(t))),
    ControlledFunctionOp(operators, (t -> 0, t -> non_freq_strength*cos(t))),
)

frequencies = [ω1, ω2, ω3]
filonprob = FilonProblem(u0, tspan[end], frequencies, op_A_derivs)

filon_sol = filon_solve(filonprob, 10, 1)
