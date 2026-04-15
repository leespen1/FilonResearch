using FilonResearch
using LinearAlgebra
using Printf

# Same setup as oscillatory_3x3_convergence.jl with ε=0
ω₁, ω₂, ω₃ = 10.0, 25.0, 40.0
D = Diagonal([ω₁, ω₂, ω₃])
ε = 0.0

A₀(t) = ComplexF64.(ε * [sin(t) cos(t) sin(2t); cos(t) sin(3t) cos(t); sin(2t) cos(t) sin(t)])
dA₀(t) = ComplexF64.(ε * [cos(t) -sin(t) 2cos(2t); -sin(t) 3cos(3t) -sin(t); 2cos(2t) -sin(t) cos(t)])
d2A₀(t) = ComplexF64.(ε * [-sin(t) -cos(t) -4sin(2t); -cos(t) -9sin(3t) -cos(t); -4sin(2t) -cos(t) -sin(t)])
d3A₀(t) = ComplexF64.(ε * [-cos(t) sin(t) -8cos(2t); sin(t) -27cos(3t) sin(t); -8cos(2t) sin(t) -cos(t)])

imD = ComplexF64.(im * D)
A_deriv_funcs = (t -> imD + A₀(t), t -> dA₀(t), t -> d2A₀(t), t -> d3A₀(t))

u₀ = ComplexF64[1.0, 1.0, 1.0]
frequencies = [ω₁, ω₂, ω₃]

# True analytic solution: u_k(T) = exp(i*ω_k*T)
for T in [1.0, 10.0, 100.0, 550.0]
    u_exact = [exp(im*ω₁*T), exp(im*ω₂*T), exp(im*ω₃*T)]
    println("\n=== T = $T ===")
    println("Analytic solution norm: ", norm(u_exact))

    for nsteps in [1, 2, 4, 8, 16]
        for s in [0, 1, 2, 3]
            sol = filon_solve(A_deriv_funcs, u₀, frequencies, T, nsteps, s, rescale=true)
            err = norm(sol[:, end] - u_exact)
            if err > 1e-12
                @printf("  nsteps=%3d, s=%d: error = %.3e\n", nsteps, s, err)
            else
                @printf("  nsteps=%3d, s=%d: error = %.3e  (exact)\n", nsteps, s, err)
            end
        end
    end
end
