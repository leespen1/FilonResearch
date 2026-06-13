# Probe: validate the amplification-factor harness for the hard-coded Filon
# method and test whether the constant-coefficient scalar step is unitary on the
# imaginary axis (|R(iν,θ)| ≡ 1).  Run under --project=lib/FilonResearch.
using FilonResearch, LinearAlgebra, StaticArrays, Printf

# Exact one-step multiplier R(z,θ;s) of the plain Filon method for the scalar
# test equation u' = λ u with ansatz frequency ω, where z = λΔt, θ = ωΔt.
# 1×1 SMatrix ⇒ static `\` path ⇒ no GMRES, so R is exact to roundoff.
function R_filon(z::Complex, θ::Real, s::Int; Δt::Float64 = 1.0)
    co = ControlledOperator((ConstantControl(one(ComplexF64)),),
                            (SMatrix{1,1,ComplexF64}(z / Δt),))   # λ = z/Δt
    ψ = filon_solve_hardcoded(co, SVector{1,ComplexF64}(1.0),
                              SVector(θ / Δt), Δt, 1, s; save_final_only = true)
    return ψ[1]
end

# Hand-derived s=0 amplification factor from stability_plot_order2.jl (cross-check).
function f4(λ, a)
    β = cis(a) * ((-im * cis(a) / a) + im * sin(a) / a^2)
    return (1 + (λ / 2) * conj(β)) / (1 - (λ / 2) * β)
end

function main()
    println("== scaling invariance R(z,θ) independent of Δt ==")
    for s in 0:2
        r1 = R_filon(0.3 + 2.0im, 5.0, s; Δt = 1.0)
        r2 = R_filon(0.3 + 2.0im, 5.0, s; Δt = 0.25)
        @printf("  s=%d  |R(Δt=1) - R(Δt=0.25)| = %.2e\n", s, abs(r1 - r2))
    end

    println("== s=0 harness vs hand-derived f4 ==")
    mx = 0.0
    for z in (0.1 + 0.3im, -0.5 + 2im, 1.0 + 10im, -2.0 + 0.5im), θ in (1.0, 5.0, 20.0)
        mx = max(mx, abs(R_filon(z, θ, 0) - f4(z, θ / 2)))
    end
    @printf("  max|R_filon - f4| = %.2e\n", mx)

    println("== exactness at the design point z = iθ (should give e^{iθ}) ==")
    for s in 0:2, θ in (1.0, 20.0)
        r = R_filon(im * θ, θ, s)
        @printf("  s=%d θ=%5.1f  |R - e^{iθ}| = %.2e  (|R|=%.6f)\n", s, θ, abs(r - cis(θ)), abs(r))
    end

    println("== unitarity on the imaginary axis: sup_ν | |R(iν,θ)| - 1 | ==")
    for s in 0:2
        for θ in (1.0, 5.0, 20.0, 100.0)
            m = 0.0; worstν = 0.0; gmax = 0.0
            for ν in range(-2θ - 30, 2θ + 30, length = 4001)
                g = abs(R_filon(im * ν, θ, s))
                gmax = max(gmax, g)
                if abs(g - 1) > m; m = abs(g - 1); worstν = ν; end
            end
            @printf("  s=%d θ=%6.1f  sup||R|-1| = %.3e  (sup|R| = %.6f at ν=%.2f)\n",
                    s, θ, m, gmax, worstν)
        end
    end
    println("PROBE_DONE")
end
main()
