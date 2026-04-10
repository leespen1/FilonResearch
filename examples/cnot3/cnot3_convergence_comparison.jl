# convergence_comparison.jl
#
# Compare accuracy of Filon and Hermite methods for CNOT3 gate design problem.
#
# Part 1: Richardson extrapolation error estimates (doubling nsteps)
# Part 2: Error vs most accurate Hermite solution
# Part 3: Error vs most accurate Filon solution
# Part 4: Cross-method error comparison
#
# Run for zero Filon frequencies, then for rotating frame frequencies.

using QuantumGateDesign
using FilonResearch
using LinearAlgebra
using Printf
using CairoMakie
using LaTeXStrings
using Quadmath

# ============================================================
# Experiment parameters - Frequently Changed
# ============================================================

# Subsystem dimensions
N_osc_levels = 10 # default 10
N_guard_levels = 2 # default 2
Tmax = 100.0 # default 550.0

# Which methods to run (Filon with rotating-frame frequencies is always run)
run_hermite = true
run_filon_zero = false

# Whether to use Taylor series branch in filon_moments for small ω.
# Set to false to reproduce catastrophic cancellation blow-up.
use_taylor_moments = true

# To control which versions of Filon, which stepsizes are used
s_values = [0, 1, 2]
min_power = 2       # start at 2^2 = 4 timesteps
max_power = 12      # cap at 2^12 = 4096 timesteps

# To control initial condition (will be normalized)
# CNOT3 Gate intitial conditions for HOHO paper example are e⃗₁, e⃗₂, e⃗₅, and e⃗₆
ψ0_unnormalized = zeros(ComplexF64, N_osc_levels*(2+N_guard_levels)^2) 
ψ0_unnormalized .= 1
#ψ0_unnormalized[6] = 1

# Change FloatType to change precision in Filon solves
FloatType = Float64

# Physical constants
fa = 4.10595
fb = 4.81526
fs = 7.8447
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2/xa
xab = 1.0e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)

#  Control function parameters
pcof = [0.014206703047363163,-0.0007404000550756441,-0.0009984714672589814,-0.0005694728281868248,-0.002501715745115248,0.0038132941334898416,0.0007116462317267991,-0.006841223438087476,-0.0023070585150656102,0.002726912304185669,0.0021534345504655482,0.0008411826515444047,-0.0018506535209815973,-0.003016058630555202,0.016049633966934613,0.017528890160984745,-0.031046857921623247,-0.010471231793747007,0.00576673415918385,-0.003037235085607017,0.0019034563937884086,-0.00269621548025083,-0.0022405789846467296,0.0027608356869845583,0.0042540873895280185,-0.0019410612339501392,0.002351716734598865,-0.0015891035993499884,0.00027566389089670037,0.00533573820808008,-0.005142978495560767,-0.02031166808124263,0.0038872595006349928,0.001749401701453946,-0.00641073199837208,-0.006889073357649663,-0.008116885234440926,0.0009660405685911396,0.008956073522034327,0.007549775920374156,0.011851558070851547,0.0051721975016686215,0.0025305278942518504,-0.004615597716686936,-0.009188552704360168,-0.0009826751804701444,0.004137740604789321,0.001976294076873065,-0.004581053667415471,0.007842651828502228,0.008731345857587123,-0.0006410027001324836,-0.010967403591470118,-0.014259223484843993,-0.00650306287617129,-0.0019148512764557185,-0.002211672527070954,0.007902705812502554,0.00807921721894839,0.0074589666546925745,0.005585593479441905,-0.006683495538458927,-0.009384665905412622,0.003874246382540184,-0.0031242840168878615,0.001928120687756253,0.0006693266711531669,0.0037148072679799022,-0.003641568161234867,-0.00365062772307532,0.0032894196045000716,-0.0023920420539574015,0.004577443983171143,-0.0055931230951403715,-5.092968148866822e-5,0.0030609312881264747,-0.002360386820624622,0.0044397005102192275,0.0009736977770266375,0.0005161127086999689,-0.0051961848866031515,0.002635932048507368,0.00024038401297035025,-0.0008219857847728555,0.0017384780236516897,0.003964850215556359,-0.0029353474747559377,-0.0013307597929656716,3.295984420467496e-5,0.0033227639167528154,-0.00042180929769978105,-0.00378001225736713,0.0016127840500289967,-0.0001641888872019785,0.00019477933106718437,0.0035008013949833147,-0.0004072308017653091,0.0003674091175372138,0.0003587496430154635,-0.0007882651771858813,0.00014953288551707066,9.65224553036758e-5,0.001632583121587648,-0.0026455651231152896,0.0008689711764723674,8.234711269603958e-5,0.002100667499337898,-0.0031970690652175266,0.00048645460193190486,0.002236433194188343,-0.002215606735306239,0.0013359251223616047,0.0006786858931989296,-0.00171056558916783,0.0026090900465467045,-0.002311818405538429,0.0013342673317499987,-0.0008398793061069168,0.0006900616402598058,0.0008538827347281442,-0.002991595583182856,0.003557834976491093,-0.0025393314050465073,0.0005187002242051925,0.002017397908124808,-0.003518869036150964,0.0032994378826123813,-0.002684348735972484,-0.003891929590288017,-0.0012408024098798096,-0.0020244985089209857,0.001948797501147367,0.0023378069280728313,0.004924588948523233,0.0020794463996769755,-0.0014518589868631176,-0.0028807598621154533,-0.0009757427275965665,-0.007492536497771161,-0.001967638334279097,-0.0069178517108224435,-0.003918863672180314,-0.0026997382411946577,-0.003926668256901013,-0.0029796892270391815,0.005347537191108047,0.0017589907868812273,-0.0014754664858328303,-0.003375183456047392,0.0032180271047125937,0.0032137265037543526,0.0009783243302176985,0.0005800037566817189,-0.0045722695461994116,0.001488089005718878,-0.0023791221121741185,0.005614233508721144,0.004921531955982609,0.006362313990562208,-0.0006638371794847855,-0.004420850857388265,0.0012792962083531915,0.002881926345854144,0.007177648839411087,0.004578372939404998,0.0026567829612900527,0.005007583283451031,-0.0037930939300424432,-0.0005138965300716757,-0.0017017194085404014,-0.010783731144485529,-0.005449130765573482,-0.003615890442526304,0.0013185552744913193,-0.0013183658842076998,-0.002154001346376969,-0.0028805908826307148,-0.0028118638088218337,-0.004574963295667305,-0.005676726099109537,-0.011402465810840888,-0.006172005547979964,-0.00243035402018065,-0.0026786376567304745,-0.00809900093687264,-0.006439216489041175,-0.005330018439044402,-0.007820064636981483,-0.006606544399090594,-0.00278589522725575,-0.0008859868160328908,-0.001625519360551106,-0.0064692279658987726,-0.005775482515237348,0.016096215609218936,0.01649559327099624,0.019180291791383346,0.009706953114914619,-0.005578470311832401,-0.011285077032252538,-0.018273317833363105,-0.022511383681479105,-0.02002655882802859,-0.009508994119235394,0.0038860973226832007,0.017264582004569068,0.011412631863814948,-0.00641615516327001,0.00733280934478686,-0.01903061930555173,-0.0097084864374851,0.0028568002178964245,0.014388776133406379,0.020246837176269936,0.015197591921245741,0.012299701185289143,0.01111772098734325,-0.0007089547830225728,-0.016105120105593288,-0.02041123528277712,-0.018073390784366965,-0.009957528977978495,0.013698829094931735,0.002013370882117593,-0.007993908433986962,0.0012440446038925045,0.009417974818047303,0.0012693278272154647,-0.009029360619055793,-0.0068655980256444905,0.005415329524354024,0.003248079323309337,0.0018391359620955011,0.0009221324721135398,-0.0020272555833737117,-0.0020206341012867558,0.0026046605057047693,0.005822756404979163,-0.0034692657131796642,0.002373016665043436,0.00272067852400584,-0.012695647240846767,-0.0003759775382934763,0.007319318614630666,-0.0013253767701881158,-0.00597078764978222,-0.00502170865852333,0.0019366438500330871,0.002118403413013919,0.0025342464538708693,-0.0006561622871415026,-0.003433363091115776,0.0011550408057884771,-0.0004474437102222443,0.006061587296459784,-0.011212339271104576,-0.002037669914416563,-0.002018667528883307,0.004055171134696443,0.0035202098739393656,-0.007138072391258802,0.0001536015205630304,0.003694298088620965,0.00831692297690751,-0.00021040861013524323,-0.0053531640693303024,-0.005814150889267475,0.0017112603264415136,0.008703334038138477,3.455283113401425e-5,-0.006150902914373094,0.001695514442260664,0.009962047875121036,-0.008693306597882311,0.007012436868809998,0.011943692643412487,-0.001262942508456683,-0.005823008403364433,-0.00502543509839941,-0.0011517450525467994,0.006027010068473857,0.001839485294707081,0.0013492114647962392,-0.0037273154288677084,-0.0028904197805052867,-0.0007386937212103546,0.007904299964579984,-0.005178057940816483]
degree = 14
D1 = 16

Nctrl = 3 # Number of control functions
Nfreq = 3 # Number of carrier wave frequencies for each control function
# Currently not handling time-dependence in H(t), so best keep these zero
Cfreq = zeros(Nctrl, Nfreq) 
## Values used in HOHO experiment
#Cfreq[1:2,2] .= -2.0*pi*xa
#Cfreq[1:2,3] .= -2.0*pi*xb
#Cfreq[3,2] = -2.0*pi*xas
#Cfreq[3,3] = -2.0*pi*xbs

# ============================================================
# Helper functions (from bspline_test.jl)
# ============================================================

function QGD_control_to_func_p(control::QuantumGateDesign.AbstractControl, pcof::AbstractVector{<: Real}, derivative_order::Integer)
    return t -> eval_p_derivative(control, t, pcof, derivative_order)
end

function QGD_control_to_func_q(control::QuantumGateDesign.AbstractControl, pcof::AbstractVector{<: Real}, derivative_order::Integer)
    return t -> eval_q_derivative(control, t, pcof, derivative_order)
end

function QGD_prob_to_controlled_op(prob::QuantumGateDesign.SchrodingerProb, controls, pcof, derivative_order::Integer)
    system_control_func = t -> iszero(derivative_order)

    p_control_funcs = ntuple(
        i -> QGD_control_to_func_p(
            controls[i],
            QuantumGateDesign.get_control_vector_slice(pcof, controls, i),
            derivative_order),
        length(controls),
    )

    q_control_funcs = ntuple(
        i -> QGD_control_to_func_q(
            controls[i],
            QuantumGateDesign.get_control_vector_slice(pcof, controls, i),
            derivative_order),
        length(controls),
    )

    coefficient_functions = tuple(system_control_func, p_control_funcs..., q_control_funcs...)

    system_hamiltonian = prob.system_sym + (im*prob.system_asym)
    hamiltonians = vcat([system_hamiltonian], complex.(prob.sym_operators), im .* prob.asym_operators)

    return ControlledFunctionOp(-im .* hamiltonians, coefficient_functions)
end

function get_controls(method_order::Integer, D1::Integer, Cfreq::AbstractMatrix{<: Real}, tf::Real)
    degree = method_order
    base_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
    base_control = QuantumGateDesign.FortranBSplineControl2(base_bspline, tf)
    controls = [CarrierControl(base_control, freqs) for freqs in eachrow(Cfreq)]
    return controls
end

# L2-in-time error helpers
function l2_time_error_subsample(hist_fine, hist_coarse, T)
    N_fine = size(hist_fine, 2) - 1
    N_coarse = size(hist_coarse, 2) - 1
    stride = N_fine ÷ N_coarse
    dt = T / N_coarse
    err_sq = sum(norm(hist_fine[:, 1 + (k-1)*stride] - hist_coarse[:, k])^2 for k in 1:N_coarse+1)
    return sqrt(dt * err_sq)
end

function l2_time_diff(hist_a, hist_b, T)
    @assert size(hist_a) == size(hist_b)
    N = size(hist_a, 2) - 1
    dt = T / N
    err_sq = sum(norm(hist_a[:, k] - hist_b[:, k])^2 for k in 1:N+1)
    return sqrt(dt * err_sq)
end

# ============================================================
# Problem setup 
# ============================================================

subsystem_sizes = (N_osc_levels, 2+N_guard_levels, 2+N_guard_levels)
essential_subsystem_sizes = (1, 2, 2)

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs

kerr_coeffs = Symmetric(
    [xs   xbs   xas;
     xbs  xb    xab;
     xas  xab   xa],
    :U
)

nsteps = 1_024 # (not used, will be overriden)
rot_prob_original = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    Tmax,
    nsteps,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)

#=
lab_prob_original = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    (0,0,0),
    kerr_coeffs,
    Tmax,
    nsteps,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)
=#

rot_controls = get_controls(degree, D1, Cfreq, Tmax)

ψ0_normalized = ψ0_unnormalized / norm(ψ0_unnormalized)
u0_vec = Complex{FloatType}.(ψ0_normalized)

rot_vec_prob = QuantumGateDesign.VectorSchrodingerProb(rot_prob_original, 1)
rot_vec_prob.u0 .= real(ψ0_normalized)
rot_vec_prob.v0 .= imag(ψ0_normalized)

# Frequency vectors, to determine Filon frequencies to use
frequencies_zero = zeros(div(rot_vec_prob.real_system_size, 2))
frequencies_rot = -1.0 .* Array(diag(rot_vec_prob.system_sym))

# Derivative functions for Filon (need at least s+1 for each s; max s=3 -> need 4)
max_deriv_order = maximum(s_values) + 2
A_deriv_funcs = ntuple(
    i -> QGD_prob_to_controlled_op(rot_vec_prob, rot_controls, pcof, i-1),
    max_deriv_order
)


# Apply Taylor moments toggle
FilonResearch.USE_TAYLOR_MOMENTS[] = use_taylor_moments

# ============================================================
# Run experiments
# ============================================================

@show maximum(abs, frequencies_rot) minimum(abs, frequencies_rot)

# Storage: s => (nsteps_list, [full_history_matrices])
hermite_data = Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}()
filon_zero_data = Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}()
filon_rot_data = Dict{Int, Tuple{Vector{Int}, Vector{Matrix{ComplexF64}}}}()


fmt(x) = @sprintf("%12.4e", x)

for s in s_values
    order = 2*(s+1)

    println("\n" * "="^100)
    println("s=$s (order $order)")
    println("="^100)
    header_parts = [@sprintf("%-8s", "nsteps")]
    if run_hermite
        push!(header_parts, @sprintf("%-14s %-6s", "Rich(H)", "p(H)"))
    end
    push!(header_parts, @sprintf("%-14s %-6s", "Rich(Fr)", "p(Fr)"))
    if run_hermite && run_filon_zero
        push!(header_parts, @sprintf("%-14s", "|H-F0|"))
    end
    if run_hermite;    push!(header_parts, @sprintf("%-12s", "|u|_H"));  end
    if run_filon_zero; push!(header_parts, @sprintf("%-12s", "|u|_F0")); end
    push!(header_parts, @sprintf("%-12s", "|u|_Fr"))
    push!(header_parts, "time")
    println("  " * join(header_parts, "  "))
    println("  " * "-"^130)

    h_hists = Matrix{ComplexF64}[]
    f0_hists = Matrix{ComplexF64}[]
    fr_hists = Matrix{ComplexF64}[]
    nsteps_h = Int[]
    nsteps_f0 = Int[]
    nsteps_fr = Int[]

    prev_h_rich = NaN
    prev_fr_rich = NaN

    for k in min_power:max_power
        ns = 2^k

        t_start = time()

        # Hermite
        if run_hermite
            rot_vec_prob.nsteps = ns
            hist = eval_forward(rot_vec_prob, rot_controls, pcof, order=order)
            push!(h_hists, Matrix{ComplexF64}(hist))
            push!(nsteps_h, ns)
        end

        # Filon (zero frequencies)
        if run_filon_zero
            hist = filon_solve(A_deriv_funcs, u0_vec, frequencies_zero, Tmax, ns, s)
            push!(f0_hists, Matrix{ComplexF64}(hist))
            push!(nsteps_f0, ns)
        end

        # Filon (rotating frame frequencies)
        hist = filon_solve(A_deriv_funcs, u0_vec, frequencies_rot, Tmax, ns, s)
        push!(fr_hists, Matrix{ComplexF64}(hist))
        push!(nsteps_fr, ns)

        # Richardson errors and convergence orders (L2-in-time norm)
        if run_hermite && length(h_hists) >= 2
            h_rich_val = l2_time_error_subsample(h_hists[end], h_hists[end-1], Tmax) * 2^order / (2^order - 1)
            h_rich_str = fmt(h_rich_val)
            h_ord_str = !isnan(prev_h_rich) && h_rich_val > 0 ? @sprintf("%5.2f ", log2(prev_h_rich / h_rich_val)) : "  --- "
            prev_h_rich = h_rich_val
        else
            h_rich_str = "    ---       "
            h_ord_str  = "  --- "
        end

        if length(fr_hists) >= 2
            fr_rich_val = l2_time_error_subsample(fr_hists[end], fr_hists[end-1], Tmax) * 2^order / (2^order - 1)
            fr_rich_str = fmt(fr_rich_val)
            fr_ord_str = !isnan(prev_fr_rich) && fr_rich_val > 0 ? @sprintf("%5.2f ", log2(prev_fr_rich / fr_rich_val)) : "  --- "
            prev_fr_rich = fr_rich_val
        else
            fr_rich_str = "    ---       "
            fr_ord_str  = "  --- "
        end

        # Elapsed time
        elapsed = time() - t_start
        time_str = elapsed < 60 ? @sprintf("%.1fs", elapsed) : @sprintf("%.1fm", elapsed/60)

        # Build output line
        row_parts = [@sprintf("%-8d", ns)]
        if run_hermite
            push!(row_parts, @sprintf("%-14s %-6s", h_rich_str, h_ord_str))
        end
        push!(row_parts, @sprintf("%-14s %-6s", fr_rich_str, fr_ord_str))
        if run_hermite && run_filon_zero
            push!(row_parts, fmt(l2_time_diff(h_hists[end], f0_hists[end], Tmax)))
        end
        if run_hermite;    push!(row_parts, @sprintf("%12.2e", norm(h_hists[end][:,end])));  end
        if run_filon_zero; push!(row_parts, @sprintf("%12.2e", norm(f0_hists[end][:,end]))); end
        push!(row_parts, @sprintf("%12.2e", norm(fr_hists[end][:,end])))
        push!(row_parts, time_str)

        println("  " * join(row_parts, "  "))
    end

    hermite_data[s] = (nsteps_h, h_hists)
    filon_zero_data[s] = (nsteps_f0, f0_hists)
    filon_rot_data[s] = (nsteps_fr, fr_hists)
end

# ============================================================
# Output functions
# ============================================================

function print_section(title)
    println("\n" * "="^80)
    println(title)
    println("="^80)
end

function print_subsection(title)
    println("\n  --- $title ---")
end

function print_richardson(label, nsteps_list, hists, order, T)
    println(@sprintf("\n  %-12s  %-22s  %-12s", "nsteps", "$label Rich. err", "rate"))
    for i in 1:length(nsteps_list)-1
        # Richardson error estimate: L2-in-time ||u_{2N} - u_N|| * 2^p/(2^p-1)
        diff = l2_time_error_subsample(hists[i+1], hists[i], T)
        err = diff * 2^order / (2^order - 1)
        if i >= 2
            prev_diff = l2_time_error_subsample(hists[i], hists[i-1], T)
            rate = prev_diff > 0 ? log2(prev_diff / diff) : NaN
            println(@sprintf("  %-12d  %-22.6e  %-12.2f", nsteps_list[i], err, rate))
        else
            println(@sprintf("  %-12d  %-22.6e", nsteps_list[i], err))
        end
    end
end

function print_error_vs_ref(label, nsteps_list, hists, ref_hist, order, T)
    println(@sprintf("\n  %-12s  %-22s  %-12s", "nsteps", "$label error", "rate"))
    prev_err = NaN
    for i in 1:length(nsteps_list)
        err = l2_time_error_subsample(ref_hist, hists[i], T)
        if err == 0.0
            continue
        end
        if !isnan(prev_err) && prev_err > 0
            rate = log2(prev_err / err)
            println(@sprintf("  %-12d  %-22.6e  %-12.2f", nsteps_list[i], err, rate))
        else
            println(@sprintf("  %-12d  %-22.6e", nsteps_list[i], err))
        end
        prev_err = err
    end
end

# ============================================================
# Print results
# ============================================================

#=
for s in s_values
    order = 2*(s+1)

    h_nsteps, h_hists = hermite_data[s]
    f0_nsteps, f0_hists = filon_zero_data[s]
    fr_nsteps, fr_hists = filon_rot_data[s]

    h_ref = h_hists[end]
    f0_ref = f0_hists[end]
    fr_ref = fr_hists[end]

    # ======================================================
    # ZERO FREQUENCY MODE
    # ======================================================
    print_section("s=$s, order=$order — ZERO FREQUENCIES (w=0)")

    print_subsection("Part 1: Richardson Extrapolation")
    print_richardson("Hermite", h_nsteps, h_hists, order, Tmax)
    print_richardson("Filon(w=0)", f0_nsteps, f0_hists, order, Tmax)

    print_subsection("Part 2: Hermite error vs most accurate Hermite (nsteps=$(h_nsteps[end]))")
    print_error_vs_ref("Hermite", h_nsteps, h_hists, h_ref, order, Tmax)

    print_subsection("Part 3: Filon(w=0) error vs most accurate Filon(w=0) (nsteps=$(f0_nsteps[end]))")
    print_error_vs_ref("Filon(w=0)", f0_nsteps, f0_hists, f0_ref, order, Tmax)

    print_subsection("Part 4a: Filon(w=0) error vs best Hermite (nsteps=$(h_nsteps[end]))")
    print_error_vs_ref("Filon(w=0)", f0_nsteps, f0_hists, h_ref, order, Tmax)

    print_subsection("Part 4b: Hermite error vs best Filon(w=0) (nsteps=$(f0_nsteps[end]))")
    print_error_vs_ref("Hermite", h_nsteps, h_hists, f0_ref, order, Tmax)

    # ======================================================
    # ROTATING FRAME FREQUENCY MODE
    # ======================================================
    print_section("s=$s, order=$order — ROTATING FRAME FREQUENCIES")

    print_subsection("Part 1: Richardson Extrapolation")
    print_richardson("Hermite", h_nsteps, h_hists, order, Tmax)
    print_richardson("Filon(rot)", fr_nsteps, fr_hists, order, Tmax)

    print_subsection("Part 2: Hermite error vs most accurate Hermite (nsteps=$(h_nsteps[end]))")
    print_error_vs_ref("Hermite", h_nsteps, h_hists, h_ref, order, Tmax)

    print_subsection("Part 3: Filon(rot) error vs most accurate Filon(rot) (nsteps=$(fr_nsteps[end]))")
    print_error_vs_ref("Filon(rot)", fr_nsteps, fr_hists, fr_ref, order, Tmax)

    print_subsection("Part 4a: Filon(rot) error vs best Hermite (nsteps=$(h_nsteps[end]))")
    print_error_vs_ref("Filon(rot)", fr_nsteps, fr_hists, h_ref, order, Tmax)

    print_subsection("Part 4b: Hermite error vs best Filon(rot) (nsteps=$(fr_nsteps[end]))")
    print_error_vs_ref("Hermite", h_nsteps, h_hists, fr_ref, order, Tmax)

    # Cross-check: do methods agree?
    # Compare the most accurate solutions from each method
    println()
    println("  Cross-method agreement (best solutions):")
    println(@sprintf("    |Hermite_best - Filon(w=0)_best| = %.6e", l2_time_diff(h_ref, f0_ref, Tmax)))
    println(@sprintf("    |Hermite_best - Filon(rot)_best| = %.6e", l2_time_diff(h_ref, fr_ref, Tmax)))
    println(@sprintf("    |Filon(w=0)_best - Filon(rot)_best| = %.6e", l2_time_diff(f0_ref, fr_ref, Tmax)))
end
=#

# ============================================================
# Build descriptive filename prefix from experiment parameters
# ============================================================

# 1. Control vector: "pcof-orig" (many distinct values), "pcof-zero", "pcof-const<val>", or "pcof-other"
if all(iszero, pcof)
    _pcof_tag = "pcof-zero"
elseif length(unique(pcof)) == 1
    _pcof_tag = "pcof-const$(pcof[1])"
elseif length(unique(pcof)) > 10
    _pcof_tag = "pcof-orig"
else
    _pcof_tag = "pcof-other"
end

# 2-3. System size
_size_tag = "Nosc$(N_osc_levels)_Nguard$(N_guard_levels)"

# 4. Taylor moments
_taylor_tag = use_taylor_moments ? "taylor-on" : "taylor-off"

# 5. Initial condition: single component, uniform, or other
_nonzero_idx = findall(!iszero, ψ0_unnormalized)
if length(_nonzero_idx) == 1
    _ic_tag = "ic-e$(_nonzero_idx[1])"
elseif length(_nonzero_idx) == length(ψ0_unnormalized) && length(unique(ψ0_unnormalized)) == 1
    _ic_tag = "ic-uniform"
else
    _ic_tag = "ic-other"
end

# 6. Carrier frequencies
_cfreq_tag = all(iszero, Cfreq) ? "Cfreq-zero" : "Cfreq-nonzero"

# 7. Tmax
_tmax_tag = "T$(Tmax)"

# 8. Float type used in Filon solves
_float_tag = "$(FloatType)"

_plot_prefix = join([_pcof_tag, _size_tag, _taylor_tag, _ic_tag, _cfreq_tag, _tmax_tag, _float_tag], "_")
println("\nPlot filename prefix: $(_plot_prefix)")

# ============================================================
# Plotting
# ============================================================

CairoMakie.set_theme!(CairoMakie.theme_latexfonts())
inch = 96
colors = Makie.wong_colors()

hermite_color = colors[1]
filon0_color  = colors[2]
filonr_color  = colors[3]

hermite_marker = :circle
filon0_marker  = :rect
filonr_marker  = :diamond

mkpath(joinpath(@__DIR__, "../..", "Plots"))

# Helper: compute Richardson error vector from a list of history matrices (L2-in-time)
function richardson_errors(hists, order, T)
    return [l2_time_error_subsample(hists[i+1], hists[i], T) * 2^order / (2^order - 1)
            for i in 1:length(hists)-1]
end

# ---- Figure 1: Control Functions ----

nplot = 500
tplot = range(0, Tmax, length=nplot)
ctrl_labels = [L"p_1(t)", L"p_2(t)", L"p_3(t)", L"q_1(t)", L"q_2(t)", L"q_3(t)"]

fig1 = Figure(size=(6.5inch, 4inch), fontsize=12)
ax1 = Axis(fig1[1, 1],
    xlabel=L"t",
    ylabel=L"\mathrm{Control\; amplitude}",
    title=L"\textbf{Control Functions}",
)

for (i, ctrl) in enumerate(rot_controls)
    pcof_i = QuantumGateDesign.get_control_vector_slice(pcof, rot_controls, i)
    p_vals = [eval_p_derivative(ctrl, t, pcof_i, 0) for t in tplot]
    q_vals = [eval_q_derivative(ctrl, t, pcof_i, 0) for t in tplot]
    lines!(ax1, collect(tplot), p_vals, color=colors[i], label=ctrl_labels[i])
    lines!(ax1, collect(tplot), q_vals, color=colors[i+3], linestyle=:dash, label=ctrl_labels[i+3])
end
Legend(fig1[1, 2], ax1, framevisible=false)

save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_control_functions.png"), fig1)
#save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_control_functions.pdf"), fig1)
println("\nSaved: Plots/$(_plot_prefix)_control_functions.{png,pdf}")

# ---- Figure 2: State Evolution ----
# Use the finest Hermite solution (highest s, max nsteps) for the state history

s_finest = maximum(s_values)
order_finest = 2*(s_finest+1)
ns_fine = 2^max_power
rot_vec_prob.nsteps = ns_fine
hist_fine = eval_forward(rot_vec_prob, rot_controls, pcof, order=order_finest)

N = size(hist_fine, 1)
t_fine = range(0, Tmax, length=size(hist_fine, 2))

fig2 = Figure(size=(6.5inch, 5.5inch), fontsize=12)
ax2_re = Axis(fig2[1, 1],
    ylabel=L"\mathrm{Re}(u_k)",
    title=L"\textbf{State Evolution}\;\;(\mathrm{nsteps}=%$ns_fine,\; s=%$s_finest)",
)
ax2_im = Axis(fig2[2, 1],
    xlabel=L"t",
    ylabel=L"\mathrm{Im}(u_k)",
)

state_cmap = cgrad(:viridis, N, categorical=true)
for k in 1:N
    lines!(ax2_re, collect(t_fine), real.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
    lines!(ax2_im, collect(t_fine), imag.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
end
Colorbar(fig2[1:2, 2], colormap=:viridis, limits=(1, N), label=L"k")

save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_state_evolution.png"), fig2)
#save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_state_evolution.pdf"), fig2)
println("Saved: Plots/$(_plot_prefix)_state_evolution.{png,pdf}")

# ---- Figure 3: Method Convergence (Richardson Errors) ----

n_s = length(s_values)
fig3 = Figure(size=(min(6.5, 2.5*n_s) * inch, 3.5inch), fontsize=12)
Label(fig3[0, 1:n_s],
    L"\textbf{Richardson Extrapolation Error Estimates}",
    fontsize=14)

ax_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)
    h_nsteps, h_hists = hermite_data[s]
    f0_nsteps, f0_hists = filon_zero_data[s]
    fr_nsteps, fr_hists = filon_rot_data[s]

    ax = Axis(fig3[1, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    # Hermite
    if run_hermite
        h_re = richardson_errors(h_hists, order, Tmax)
        scatterlines!(ax, h_nsteps[1:end-1], h_re,
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    # Filon (w=0)
    if run_filon_zero
        f0_re = richardson_errors(f0_hists, order, Tmax)
        scatterlines!(ax, f0_nsteps[1:end-1], f0_re,
            color=filon0_color, marker=filon0_marker, label=L"\mathrm{Filon}\;(\omega=0)")
    end

    # Filon (rot)
    fr_re = richardson_errors(fr_hists, order, Tmax)
    scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
        color=filonr_color, marker=filonr_marker, label=L"\mathrm{Filon}\;(\omega=\mathrm{rot})")

    # Reference slope
    ref_ns = [2.0^k for k in min_power:max_power]
    ref_C = fr_re[1] * fr_nsteps[1]^order
    ref_line = ref_C ./ (ref_ns .^ order)
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")

    if col == 1
        global ax_first = ax
    end
end
Legend(fig3[2, 1:n_s], ax_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_convergence_richardson.png"), fig3)
#save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_convergence_richardson.pdf"), fig3)
println("Saved: Plots/$(_plot_prefix)_convergence_richardson.{png,pdf}")

# ---- Figure 4: Combined (controls, state evolution, convergence) ----

fig4 = Figure(size=(6.5inch, 10inch), fontsize=12)

# Row 1: Control functions
ax4_ctrl = Axis(fig4[1, 1:n_s],
    xlabel=L"t",
    ylabel=L"\mathrm{Control\; amplitude}",
    title=L"\textbf{Control Functions}",
)
for (i, ctrl) in enumerate(rot_controls)
    pcof_i = QuantumGateDesign.get_control_vector_slice(pcof, rot_controls, i)
    p_vals = [eval_p_derivative(ctrl, t, pcof_i, 0) for t in tplot]
    q_vals = [eval_q_derivative(ctrl, t, pcof_i, 0) for t in tplot]
    lines!(ax4_ctrl, collect(tplot), p_vals, color=colors[i], label=ctrl_labels[i])
    lines!(ax4_ctrl, collect(tplot), q_vals, color=colors[i+3], linestyle=:dash, label=ctrl_labels[i+3])
end
Legend(fig4[1, n_s+1], ax4_ctrl, framevisible=false)

# Rows 2-3: State evolution (Re and Im)
ax4_re = Axis(fig4[2, 1:n_s],
    ylabel=L"\mathrm{Re}(u_k)",
    title=L"\textbf{State Evolution}\;\;(\mathrm{nsteps}=%$ns_fine,\; s=%$s_finest)",
)
ax4_im = Axis(fig4[3, 1:n_s],
    xlabel=L"t",
    ylabel=L"\mathrm{Im}(u_k)",
)
for k in 1:N
    lines!(ax4_re, collect(t_fine), real.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
    lines!(ax4_im, collect(t_fine), imag.(hist_fine[k, :]), color=(state_cmap[k], 0.5))
end
Colorbar(fig4[2:3, n_s+1], colormap=:viridis, limits=(1, N), label=L"k")

# Row 4: Convergence (Richardson errors)
ax4_first = nothing
for (col, s) in enumerate(s_values)
    order = 2*(s+1)
    h_nsteps, h_hists = hermite_data[s]
    f0_nsteps, f0_hists = filon_zero_data[s]
    fr_nsteps, fr_hists = filon_rot_data[s]

    ax = Axis(fig4[4, col],
        title=L"s=%$s \;\; (p=%$order)",
        xlabel=L"\mathrm{nsteps}",
        ylabel= col == 1 ? L"\mathrm{Error}" : "",
        xscale=log10, yscale=log10,
        yticklabelsvisible = true,
    )

    if run_hermite
        h_re = richardson_errors(h_hists, order, Tmax)
        scatterlines!(ax, h_nsteps[1:end-1], h_re,
            color=hermite_color, marker=hermite_marker, label="Hermite")
    end

    if run_filon_zero
        f0_re = richardson_errors(f0_hists, order, Tmax)
        scatterlines!(ax, f0_nsteps[1:end-1], f0_re,
            color=filon0_color, marker=filon0_marker, label="Filon (w=0)")
    end

    fr_re = richardson_errors(fr_hists, order, Tmax)
    scatterlines!(ax, fr_nsteps[1:end-1], fr_re,
        color=filonr_color, marker=filonr_marker, label="Filon (w=rot)")

    ref_ns = [2.0^k for k in min_power:max_power]
    ref_C = fr_re[1] * fr_nsteps[1]^order
    ref_line = ref_C ./ (ref_ns .^ order)
    lines!(ax, ref_ns, ref_line, linestyle=:dot, color=:gray, linewidth=2, label="O(dt^p)")

    if col == 1
        global ax4_first = ax
    end
end
Legend(fig4[5, 1:n_s], ax4_first, orientation=:horizontal, tellheight=true, tellwidth=false)

save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_cnot3_combined.png"), fig4)
#save(joinpath(@__DIR__, "../..", "Plots", "$(_plot_prefix)_cnot3_combined.pdf"), fig4)
println("Saved: Plots/$(_plot_prefix)_cnot3_combined.{png,pdf}")

display(fig1)
display(fig2)
display(fig3)
display(fig4)
