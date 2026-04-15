using FilonResearch
using LinearAlgebra
using Printf

ω₁, ω₂, ω₃ = 10.0, 25.0, 40.0
imD = ComplexF64.(im * Diagonal([ω₁, ω₂, ω₃]))
A_deriv_funcs = (t -> imD, t -> zero(imD), t -> zero(imD), t -> zero(imD))
u₀ = ComplexF64[1.0, 1.0, 1.0]
frequencies = [ω₁, ω₂, ω₃]
T = 550.0
u_exact = [exp(im*ω*T) for ω in frequencies]

println("Error vs nsteps for T=$T, ω_max=$ω₃")
println(@sprintf("%10s  %12s  %12s  %12s  %12s", "nsteps", "s=0", "s=1", "s=2", "s=3"))
for k in 0:14
    nsteps = 2^k
    errs = Float64[]
    for s in 0:3
        sol = filon_solve(A_deriv_funcs, u₀, frequencies, T, nsteps, s, rescale=true)
        push!(errs, norm(sol[:, end] - u_exact))
    end
    @printf("%10d  %12.3e  %12.3e  %12.3e  %12.3e\n", nsteps, errs...)
end

# Also check: does the error scale as ω^(s+1)?
println("\n\nError vs ω_max for T=100, nsteps=4")
for ω_max in [5.0, 10.0, 20.0, 40.0, 80.0]
    freqs = [ω_max/4, ω_max/1.6, ω_max]
    iD = ComplexF64.(im * Diagonal(freqs))
    A_funcs = (t -> iD, t -> zero(iD), t -> zero(iD), t -> zero(iD))
    u_ex = [exp(im*ω*100.0) for ω in freqs]
    errs = Float64[]
    for s in 0:3
        sol = filon_solve(A_funcs, u₀, freqs, 100.0, 4, s, rescale=true)
        push!(errs, norm(sol[:, end] - u_ex))
    end
    @printf("ω_max=%5.1f:  s=0: %10.3e  s=1: %10.3e  s=2: %10.3e  s=3: %10.3e\n", ω_max, errs...)
end
