# Quantitative summary of the oscillatory convergence study: cheapest run (by
# wall-clock) that reaches an accuracy target, per method/problem, + the winner.
using Serialization, Printf
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
results = deserialize(joinpath(REPO, "data", "oscillatory", "osc_data.jls"))

tag(m, s) = m === :rk4 ? "RK4" : (m === :filon ? "Filon" : "cFilon") * " s=$s"

# cheapest wall-time among points reaching err ≤ target (Inf if never)
function time_to(P, m, s, target)
    e = P.err[(m, s)]; t = P.tim[(m, s)]
    best = Inf
    for i in eachindex(e)
        e[i] <= target && (best = min(best, t[i]))
    end
    return best
end

for P in results
    println("\n===== $(P.label) =====")
    for target in (1e-6, 1e-10)
        @printf("  target err ≤ %.0e :\n", target)
        rows = [(tag(m, s), time_to(P, m, s, target)) for (m, s) in P.methods]
        sort!(rows; by = x -> x[2])
        for (nm, tt) in rows
            @printf("     %-12s  %s\n", nm, isfinite(tt) ? @sprintf("%.3e s", tt) : "—")
        end
    end
    # coarse-step error (stability indicator) at the coarsest nsteps
    @printf("  coarse-step err (nsteps=2^%d):\n", P.es[1])
    for (m, s) in P.methods
        @printf("     %-12s  %.2e\n", tag(m, s), P.err[(m, s)][1])
    end
end
println("OSC_REPORT_DONE")
