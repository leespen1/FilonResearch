using FilonResearch
using LinearAlgebra
using Printf

# Check the intermediate quantities in Algorithm1 for the ε=0 case
ω₁, ω₂, ω₃ = 10.0, 25.0, 40.0
frequencies = [ω₁, ω₂, ω₃]
imD = ComplexF64.(im * Diagonal([ω₁, ω₂, ω₃]))
A_derivs = [imD, zero(imD), zero(imD), zero(imD)]

u = ComplexF64[1.0, 1.0, 1.0]

for s in 0:3
    for t in [0.0, 275.0, 550.0]
        # Reproduce what Algorithm1 does
        u_at_t = [exp(im*ω*t) for ω in frequencies]
        u_derivs = FilonResearch.linear_ode_derivs(A_derivs, u_at_t, s)
        freq_factor_derivs = FilonResearch.exp_iωt_derivs(-frequencies, t, s)
        f_derivs = FilonResearch.multiple_general_leibniz_rule(freq_factor_derivs, u_derivs)

        # f_derivs[1] should be u₀ = [1,1,1], f_derivs[2:end] should be zero
        println("s=$s, t=$t:")
        for (m, fd) in enumerate(f_derivs)
            println("  f^($(m-1)) = ", fd, "  |f^($(m-1))| = ", norm(fd))
        end

        # Show magnitudes of individual terms in the Leibniz sum for f'
        if s >= 1
            println("  --- Leibniz terms for f'(t) ---")
            for k in 0:1
                term = binomial(1, k) .* freq_factor_derivs[1+1-k] .* u_derivs[1+k]
                println("    k=$k: ", norm(term), " (individual components: ", abs.(term), ")")
            end
        end
        println()
    end
end
