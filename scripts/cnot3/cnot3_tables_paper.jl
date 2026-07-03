# =============================================================================
# CNOT3 paper tables: speedup / Δt requirements vs target final-time error
# =============================================================================
#
# Speedup / Δt tables: "to reach error X, which method is fastest (or coarsest)
# and by how much?"  Columns are target final-time errors 1e-1..1e-7; rows are
# (method, s); lab and RWA frames are stacked as blocks of one tabular.  For
# each initial condition four tables are produced:
#
#   1. speedup_time  — time-to-solution ratio to the fastest method ("--")
#   2. speedup_dt    — Δt ratio to the coarsest-stepping method ("--")
#   3. required_dt   — largest Δt reaching each target (sanity check)
#   4. required_time — minimal time reaching each target (sanity check)
#
# Values come from the same filtered curves as the figures, with log-log
# interpolation between consecutive points.  The cheapest run reaching a target
# need not be the coarsest run that crosses it: GMRES iteration counts vary
# with Δt, so a finer run can also be a faster one, and any run that lands
# below X also serves every looser target.
#
# Shares src/cnot3_paper_common.jl with the figures script
# (cnot3_convergence_paper.jl) but loads no plotting backend, so table tweaks
# don't pay the Makie tax.  Tables are written to the DrWatson plots dir and
# (only if the Overleaf dir exists) copied into FilonProjectOverleaf/Tables/.
#
# Run:  julia --project=. scripts/cnot3/cnot3_tables_paper.jl
#       (CNOT3_INIT=basis to restrict to one initial condition)
# =============================================================================

using DrWatson
@quickactivate "FilonExperiments"

using Printf

include(srcdir("cnot3_paper_common.jl"))

const TABLE_TARGETS = [10.0^(-k) for k in 1:7]
const TABLE_ROWS = [(m, s) for m in (:Hermite, :Filon, :ControlledFilon) for s in SVALS]
# Method labels, one line per row of the group (s = 0, 1, 2 top to bottom), so
# long names can wrap instead of widening the method column.
const TABLE_ROW_LABELS = Dict(:Hermite => ["Hermite"], :Filon => ["Filon"],
                              :ControlledFilon => ["C-Filon"])
function row_label(m, s)
    si = findfirst(isequal(s), SVALS)
    return get(TABLE_ROW_LABELS[m], si, "")
end

# Optimal `vals` subject to error <= X on the piecewise-linear (in log-log)
# curve through (vals_i, errs_i), points connected in nsteps order as plotted.
# Along a segment both coordinates are monotone in the interpolation parameter,
# so the optimum over the feasible part of each segment is attained either at a
# data point already below X or at a threshold crossing; `pick` is `minimum`
# (time to compute) or `maximum` (Δt).  Returns `nothing` if X is never reached.
function value_at_target(vals, errs, X, pick)
    cands = [vals[i] for i in eachindex(vals) if errs[i] <= X]
    for i in firstindex(vals):lastindex(vals)-1
        e1, e2 = errs[i], errs[i+1]
        min(e1, e2) < X < max(e1, e2) || continue
        λ = (log(X) - log(e1)) / (log(e2) - log(e1))
        push!(cands, exp((1 - λ) * log(vals[i]) + λ * log(vals[i+1])))
    end
    return isempty(cands) ? nothing : pick(cands)
end

# Per-target (Δt, time) requirements for one (method, s) series, as a vector of
# NamedTuples over TABLE_TARGETS, or `nothing` if the series has no usable runs.
# Targets the data never reaches are extrapolated from the finest kept run at
# the design order O(Δt^{2(s+1)}): Δt* = Δt_end (X/err_end)^{1/p}, and the time
# scales with the step count, t* = t_end · Δt_end/Δt*.  Extrapolated entries
# carry `ex = true`.
function series_requirements(df, m, s)
    sub = seriesof(df, m, s)
    keep = in_window(sub.final_error) .& isfinite.(sub.t_elapsed)
    dt = Vector{Float64}(sub.dt[keep])
    err = Vector{Float64}(sub.final_error[keep])
    tel = Vector{Float64}(sub.t_elapsed[keep])
    isempty(err) && return nothing
    p = 2 * (s + 1)
    return map(TABLE_TARGETS) do X
        dtX = value_at_target(dt, err, X, maximum)
        if dtX === nothing
            dt_star = dt[end] * (X / err[end])^(1 / p)
            (dt = dt_star, time = tel[end] * dt[end] / dt_star, ex = true)
        else
            (dt = dtX, time = value_at_target(tel, err, X, minimum), ex = false)
        end
    end
end

# Mantissa(exponent) notation: "1.1(2)" for 110, "3.7(-5)" for 3.7e-5.
function fmt_mant_exp(v)
    e = floor(Int, log10(v))
    m = round(v / 10.0^e; digits = 1)
    m >= 10 && (m /= 10; e += 1)
    return @sprintf("%.1f(%d)", m, e)
end

# Table entries, two significant digits throughout: single-digit numbers
# plainly ("1.5", "9.9"), everything else in mantissa(exponent) notation
# ("2.3(1)", "7.3(-1)").
fmt_num(v) = 0.95 <= v < 9.95 ? @sprintf("%.1f", v) : fmt_mant_exp(v)

# Entry matrix (TABLE_ROWS × TABLE_TARGETS) for a ratio table: the best row of
# each column gets "--" (so near-ties don't masquerade as winners), the rest
# the ratio (>= 1) of their value to the best.  `pick` is :min (time) or :max
# (Δt); `ratio(v, best)` orients the factor.  An entry is starred when it, or
# the best value it is measured against, is extrapolated.
function ratio_table(reqs; getval, pick, ratio)
    ent = fill("n/a", length(TABLE_ROWS), length(TABLE_TARGETS))
    for j in eachindex(TABLE_TARGETS)
        cells = [reqs[row] === nothing ? nothing : reqs[row][j] for row in TABLE_ROWS]
        idx = findall(!isnothing, cells)
        isempty(idx) && continue
        scores = [getval(cells[i]) for i in idx]
        best = idx[pick == :min ? argmin(scores) : argmax(scores)]
        for i in idx
            star = (cells[best].ex || cells[i].ex) ? "*" : ""
            ent[i, j] = i == best ? (cells[best].ex ? "*" : "") * "--" :
                        star * fmt_num(ratio(getval(cells[i]), getval(cells[best])))
        end
    end
    return ent
end

# Entry matrix of raw values (the sanity-check tables), starred when extrapolated.
function raw_table(reqs; getval)
    ent = fill("n/a", length(TABLE_ROWS), length(TABLE_TARGETS))
    for j in eachindex(TABLE_TARGETS), (i, row) in enumerate(TABLE_ROWS)
        reqs[row] === nothing && continue
        c = reqs[row][j]
        ent[i, j] = (c.ex ? "*" : "") * fmt_num(getval(c))
    end
    return ent
end

# Write one booktabs tabular (no surrounding table environment, so the paper
# controls placement/caption/label) to the plots dir and (if present) Overleaf
# Tables/.  The title is a \multicolumn row under the toprule, set off from the
# body by a heavy rule; `blocks` is a vector of (block_title, entries) pairs
# stacked vertically under a shared column header, each introduced by a
# \multicolumn row in the style of the physical-parameters table in the paper.
# The caption text is kept as a comment for reference.
function save_table(basename, title, caption, blocks)
    ncols = 2 + length(TABLE_TARGETS)
    header = "Method & \$s\$ & " * join(["\$10^{-$k}\$" for k in 1:7], " & ") * " \\\\"
    lines = String[
        "% Auto-generated by scripts/cnot3/cnot3_tables_paper.jl; do not edit.",
        "% $caption",
        "\\begin{tabular}{ll" * "r"^length(TABLE_TARGETS) * "}",
        "    \\toprule",
        "    \\multicolumn{$ncols}{c}{$title} \\\\",
        "    \\midrule[\\heavyrulewidth]",
        "    & & \\multicolumn{$(length(TABLE_TARGETS))}{c}{Target final-time error} \\\\",
        "    \\cmidrule(lr){3-$ncols}",
        "    " * header,
    ]
    for (block_title, entries) in blocks
        append!(lines, ["    \\midrule",
                        "    \\multicolumn{$ncols}{c}{$block_title} \\\\",
                        "    \\midrule"])
        for (i, (m, s)) in enumerate(TABLE_ROWS)
            s == first(SVALS) && i > 1 && push!(lines, "    \\addlinespace")
            push!(lines, "    $(row_label(m, s)) & $s & " * join(entries[i, :], " & ") * " \\\\")
        end
    end
    append!(lines, ["    \\bottomrule", "\\end{tabular}", ""])
    dests = [plotsdir("cnot3", "$basename.tex")]
    overleaf = projectdir("FilonProjectOverleaf")
    isdir(overleaf) && push!(dests, joinpath(overleaf, "Tables", "$basename.tex"))
    for p in dests
        mkpath(dirname(p))
        write(p, join(lines, "\n"))
        println("  saved → ", p)
    end
end

# Aligned stdout rendering of an entry matrix (the on-screen sanity check).
function print_table(title, entries)
    println("\n  ", title)
    w = max(11, maximum(length, entries) + 2)
    wm = maximum(length, Iterators.flatten(values(TABLE_ROW_LABELS))) + 2
    println("    ", rpad("method", wm), rpad("s", 3),
            join([lpad("1e-$k", w) for k in 1:7]))
    for (i, (m, s)) in enumerate(TABLE_ROWS)
        println("    ", rpad(row_label(m, s), wm), rpad(s, 3),
                join([lpad(e, w) for e in entries[i, :]]))
    end
end

const EXTRAP_NOTE = "Asterisked entries rely on extrapolation from the finest " *
                    "run at the design order \$\\mathcal{O}(\\Delta t^{2(s+1)})\$."

const TABLE_FRAMES = ("lab" => "Lab Frame", "rwa" => "RWA Frame")

function tables_for(df_all, init)
    ent = Dict()
    for (frame, _) in TABLE_FRAMES
        df = frame_df(df_all, frame, init)
        reqs = Dict(row => series_requirements(df, row...) for row in TABLE_ROWS)
        ent[frame, :speedup] = ratio_table(reqs; getval = c -> c.time, pick = :min,
                                           ratio = (v, b) -> v / b)
        ent[frame, :dtratio] = ratio_table(reqs; getval = c -> c.dt, pick = :max,
                                           ratio = (v, b) -> b / v)
        ent[frame, :raw_dt] = raw_table(reqs; getval = c -> c.dt)
        ent[frame, :raw_time] = raw_table(reqs; getval = c -> c.time)
    end
    blocks(kind) = [(block_title, ent[frame, kind]) for (frame, block_title) in TABLE_FRAMES]
    pretty = (init == "basis" ? "gate-basis" : init) * " initial condition"
    ctx = "CNOT3 experiment, $pretty; the blocks are the lab and RWA frames, " *
          "compared within each block."

    save_table("cnot3_speedup_time_$init", "Relative Time to Compute Solution",
        "$ctx For each target final-time error the fastest (method, \$s\$) pair " *
        "is marked ``--'' and every other row reports the ratio of its minimal " *
        "time reaching that error to the fastest time; \$a(b)\$ denotes " *
        "\$a \\times 10^{b}\$. $EXTRAP_NOTE", blocks(:speedup))
    save_table("cnot3_speedup_dt_$init", "Relative Step Size",
        "$ctx For each target final-time error the (method, \$s\$) pair reaching " *
        "it at the largest \$\\Delta t\$ is marked ``--'' and every other row " *
        "reports the ratio of that largest \$\\Delta t\$ to its own required " *
        "\$\\Delta t\$; \$a(b)\$ denotes \$a \\times 10^{b}\$. $EXTRAP_NOTE",
        blocks(:dtratio))
    save_table("cnot3_required_dt_$init", "Required Step Size (ns)",
        "$ctx Largest \$\\Delta t\$ reaching each target final-time error; " *
        "\$a(b)\$ denotes \$a \\times 10^{b}\$. $EXTRAP_NOTE", blocks(:raw_dt))
    save_table("cnot3_required_time_$init", "Required Time to Solution (s)",
        "$ctx Minimal time to solution (seconds) reaching each target final-time " *
        "error; \$a(b)\$ denotes \$a \\times 10^{b}\$. $EXTRAP_NOTE", blocks(:raw_time))

    for (frame, block_title) in TABLE_FRAMES
        where_ = "$block_title, $pretty"
        print_table("Relative Time to Compute Solution ($where_)", ent[frame, :speedup])
        print_table("Relative Step Size ($where_)", ent[frame, :dtratio])
        print_table("Required Step Size, ns ($where_)", ent[frame, :raw_dt])
        print_table("Required Time to Solution, s ($where_)", ent[frame, :raw_time])
    end
end

df_all = load_cnot3_runs()
for init in INITS
    tables_for(df_all, init)
end
println("Done.")
