# Where is Filon most effective?  The highly-oscillatory regime: a driven
# 2-level system in the lab frame, H(t)=½ω₀Z + Ω_R cos(ω₀t) X, with the carrier
# ω₀ swept up.  The physical dynamics (Rabi flopping at Ω_R) is fixed, but the
# oscillation gets faster.  Filon's ansatz absorbs ω₀, so its step count to a
# target accuracy is set by Ω_R (≈ constant in ω₀); every standard method
# (Gauss–Legendre IRK, RK4) must resolve ω₀, so its step count grows ∝ ω₀.
#
# Run under the umbrella env via srun; osc_omega_plot via osc_plot-style code at
# the end writes plots/oscillatory/osc_omega.png.
using FilonResearch, LinearAlgebra, StaticArrays, Serialization, Printf

const Z2 = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)
const X2 = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)
mkfc(d0, d1, d2) = FunctionControl{ComplexF64}((t, n) ->
    n == 0 ? ComplexF64(d0(t)) : n == 1 ? ComplexF64(d1(t)) : n == 2 ? ComplexF64(d2(t)) :
    throw(ArgumentError("order $n")))
A_at(co, t) = materialize(evaluate(co, t, Derivative{0}()))

function rk4_final(co, ψ0::SVector{N}, T, nsteps) where {N}
    h = T / nsteps; ψ = ψ0
    @inbounds for n in 0:nsteps-1
        t = n*h
        k1 = A_at(co, t)*ψ; k2 = A_at(co, t+h/2)*(ψ+(h/2)*k1)
        k3 = A_at(co, t+h/2)*(ψ+(h/2)*k2); k4 = A_at(co, t+h)*(ψ+h*k3)
        ψ = ψ + (h/6)*(k1+2k2+2k3+k4)
    end
    return ψ
end
const GLTAB3 = (c = [0.5-√15/10, 0.5, 0.5+√15/10],
    a = [5/36 2/9-√15/15 5/36-√15/30; 5/36+√15/24 2/9 5/36-√15/24; 5/36+√15/30 2/9+√15/15 5/36],
    b = [5/18, 4/9, 5/18])
function gl6_final(co, ψ0::SVector{N}, T, nsteps) where {N}
    tab = GLTAB3; h = T/nsteps; ψ = ψ0; m = 3
    M = zeros(ComplexF64, m*N, m*N); rhs = zeros(ComplexF64, m*N)
    @inbounds for n in 0:nsteps-1
        t = n*h; fill!(M, 0)
        for i in 1:m
            Ai = A_at(co, t+tab.c[i]*h); ri=(i-1)*N
            for k in 1:N; M[ri+k, ri+k]+=1; end
            for j in 1:m
                cj=(j-1)*N; aij=h*tab.a[i,j]
                for r in 1:N, c in 1:N; M[ri+r,cj+c]-=aij*Ai[r,c]; end
            end
            Aiψ=Ai*ψ; for k in 1:N; rhs[ri+k]=Aiψ[k]; end
        end
        K = M\rhs; acc = zero(SVector{N,ComplexF64})
        for i in 1:m; ri=(i-1)*N; acc += tab.b[i]*SVector{N,ComplexF64}(ntuple(k->K[ri+k],N)); end
        ψ = ψ + h*acc
    end
    return ψ
end

# Filon's sweet spot: a fast KNOWN DIAGONAL splitting ω0 (absorbed by the ansatz)
# with a SLOW transverse coupling g0 cos(Ω_s t) X (Ω_s ≪ ω0, no fast carrier) —
# a Larmor/NMR-type setup.  The off-diagonal coupling between the fast ±ω0/2
# components makes the coupling integrand oscillate at ω0, which Filon's moment
# weights handle analytically; so Filon's step count is set by Ω_s, g0 — not ω0.
# (cfilon ≡ filon here: the coupling carries no carrier.)
function driven(ω0, g0)
    Ωs = 0.3
    Adrift = -im*(ω0/2)*Z2
    freqs = SVector(-ω0/2, ω0/2)
    g = mkfc(t -> -im*g0*cos(Ωs*t), t -> im*g0*Ωs*sin(Ωs*t), t -> im*g0*Ωs^2*cos(Ωs*t))
    co_plain = ControlledOperator((ConstantControl(ComplexF64(1)), g), (Adrift, X2))
    return (; co_plain, co_ctrl = co_plain, freqs, ψ0 = SVector{2,ComplexF64}(1, 0), T = 10.0)
end

solve(method, P, n) =
    method === :filon  ? SVector(filon_solve_hardcoded(P.co_plain, P.ψ0, P.freqs, P.T/n, n, 2; save_final_only=true)...) :
    method === :cfilon ? SVector(controlled_filon_solve(P.co_ctrl, P.ψ0, P.freqs, P.T/n, n, 2; save_final_only=true)...) :
    method === :gl6    ? gl6_final(P.co_plain, P.ψ0, P.T, n) :
                         rk4_final(P.co_plain, P.ψ0, P.T, n)

function main()
    g0 = 1.0; ω0s = [20.0, 40.0, 80.0, 160.0, 320.0]; target = 1e-6
    methods = (:filon, :cfilon, :gl6, :rk4)
    minN = Dict(m => Int[] for m in methods); minT = Dict(m => Float64[] for m in methods)
    for ω0 in ω0s
        P = driven(ω0, g0)
        ψref = rk4_final(P.co_plain, P.ψ0, P.T, 2^21)
        @printf("ω0=%5.0f  (‖ref‖=%.3f)\n", ω0, norm(ψref))
        for m in methods
            solve(m, P, 2^4)                              # warm up
            nbest = 0; tbest = Inf
            for e in 4:19
                n = 2^e; t = @elapsed ψ = solve(m, P, n)
                if norm(ψ - ψref) <= target; nbest = n; tbest = t; break; end
            end
            push!(minN[m], nbest); push!(minT[m], tbest)
            @printf("    %-7s  min nsteps→1e-6 = %-8s  time = %s\n", m,
                    nbest == 0 ? ">2^19" : "2^$(Int(log2(nbest)))",
                    isfinite(tbest) ? @sprintf("%.2e s", tbest) : "—")
        end
    end
    out = normpath(joinpath(@__DIR__, "..", "..", "data", "oscillatory")); mkpath(out)
    serialize(joinpath(out, "osc_omega.jls"), (; ω0s, methods, minN, minT, target, g0))
    println("OSC_OMEGA_DONE")
end
main()
