using FilonResearch, Plots

#=
#
# This example shows that the solution obtained using Filon on a system of ODEs
# whose solution is p(t)exp(iwt), where p(t) is a polynomial, is *exact* s is
# high enough.
#
# Also checks that the order of convergence is correct when using Hermite
# (ansatz frequencies zero)
#
# Possible shortcomings: only one frequency here.
=#


w = 10.0
T = 1.0

#=
pl_degree = 2
pl_ts = LinRange(0, t, 1001)
pl_sol = transpose(hcat(poly_osc_solution.(w, pl_degree, pl_ts)...))
pl = plot(pl_ts, real.(pl_sol))
plot!(pl, pl_ts, (x -> 1).(pl_ts), label="1", linestyle=:dash)
plot!(pl, pl_ts, (x -> 1+x).(pl_ts), label="1+t", linestyle=:dash)
plot!(pl, pl_ts, (x -> 1+x+x^2).(pl_ts), label="1+t+t^2", linestyle=:dash)
=#

# Checking for exact solutions
for s in 0:3
    for degree in 0:2*(s+1)
        A = poly_osc_ode_mat(w, degree)
        N = size(A, 1)
        ws = fill(w, N)
        u0 = ones(N)
        true_sol = poly_osc_solution(w, degree, T)
        sol = filon_timestep_static(A, u0, ws, 0.0, T, s)
        @show s degree maximum(abs, sol - true_sol)
        println()
        #println("A")
        #display(A)
        #println("sol vs true sol")
        #display(hcat(sol, true_sol))
    end
end

begin
    # Checking order of convergence for hermite-version (zero frequency)
    w = 2.0
    T = 5.0
    degree = 16
    A = poly_osc_ode_mat(w, degree)
    N = size(A, 1)
    ws = zeros(N)
    #ws = fill(3, N)
    u0 = ones(N)
    nsteps_vec = [2^i for i in 0:4]
    true_sol = poly_osc_solution(w, degree, T)
    errors = Float64[]
    s=3
    for nsteps in nsteps_vec
        sol = filon_solve_static(A, u0, ws, T, nsteps, s)[:,end]
        #display(hcat(sol, true_sol))
        error = sum(abs, sol - true_sol)
        push!(errors, error)
    end
    error_ratios = [errors[i] / errors[i+1] for i in 1:length(errors)-1]
    cvg_orders = log2.(error_ratios)
    @show round.(errors, sigdigits=2)
    @show round.(cvg_orders, sigdigits=2)

end




