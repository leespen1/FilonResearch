# cnot3_conditioning.jl
#
# Check the conditioning of S_+ (the implicit LHS matrix) in the CNOT3 example
# as a function of nsteps and s, for both zero and rotating-frame frequencies.

using QuantumGateDesign
using FilonResearch
using LinearAlgebra
using Printf
using CairoMakie

# ============================================================
# Experiment parameters - Frequently Changed
# ============================================================

# Subsystem dimensions
N_osc_levels = 10 # default 10
N_guard_levels = 2 # default 2
Tmax = 550.0 # default 550.0

# Which methods to run (Filon with rotating-frame frequencies is always run)
run_hermite = true
run_filon_zero = false

# Whether to use Taylor series branch in filon_moments for small ω.
# Set to false to reproduce catastrophic cancellation blow-up.
use_taylor_moments = true

# To control which versions of Filon, which stepsizes are used
s_values = [0, 1, 2]
min_power = 2       # start at 2^2 = 4 timesteps
max_power = 14      # cap at 2^12 = 4096 timesteps

# Representative nsteps values for time-resolved conditioning plots
time_resolved_nsteps = [8, 16, 64, 128]

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
Cfreq[1:2,2] .= -2.0*pi*xa
Cfreq[1:2,3] .= -2.0*pi*xb
Cfreq[3,2] = -2.0*pi*xas
Cfreq[3,3] = -2.0*pi*xbs

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

const Tmax_original = 550.0 # Want to use 550.0 so that the controls aren't "squeezed" into a smaller timeframe.
rot_controls = get_controls(degree, D1, Cfreq, Tmax_original)

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

A_matrix_funcs = ntuple(
    i -> (t -> FilonResearch.full_op(A_deriv_funcs[i](t))),
    length(A_deriv_funcs)
)


# Apply Taylor moments toggle
FilonResearch.USE_TAYLOR_MOMENTS[] = use_taylor_moments


# ============================================================
# Conditioning study — collect data
# ============================================================

nsteps_list = [2^k for k in min_power:max_power]
dt_list = Tmax ./ nsteps_list
n_ns = length(nsteps_list)
n_s = length(s_values)

# Storage: cond(S+), cond(S-), min singular value of S+
cond_Sp_0  = zeros(n_ns, n_s)
cond_Sm_0  = zeros(n_ns, n_s)
cond_Sp_r  = zeros(n_ns, n_s)
cond_Sm_r  = zeros(n_ns, n_s)
min_sv_Sp_0 = zeros(n_ns, n_s)
min_sv_Sp_r = zeros(n_ns, n_s)

# Spectral radius of G = S+⁻¹ S-
rho_0 = zeros(n_ns, n_s)
rho_r = zeros(n_ns, n_s)

println("=" ^ 120)
println("Conditioning of S_+ (implicit LHS) for CNOT3 problem")
println("System size N = $(length(frequencies_rot))")
println("=" ^ 120)

for (js, s) in enumerate(s_values)
    order = 2*(s+1)
    println("\n" * "-"^120)
    println("s=$s (order $order)")
    println("-"^120)
    println(@sprintf("  %-10s  %-12s  %-14s %-14s %-14s %-14s %-14s %-14s",
        "nsteps", "dt",
        "cond(S+,ω=0)", "cond(S-,ω=0)",
        "cond(S+,ω=rot)", "cond(S-,ω=rot)",
        "min|σ(S+,0)|", "min|σ(S+,rot)|"))

    for (jn, ns) in enumerate(nsteps_list)
        dt = dt_list[jn]
        t_n = 0.0
        t_np1 = dt

        # Zero frequencies
        Sp_0_, Sm_0_ = filon_S_plus_S_minus(A_matrix_funcs, frequencies_zero, t_n, t_np1, s)
        cond_Sp_0[jn, js]  = cond(Sp_0_)
        cond_Sm_0[jn, js]  = cond(Sm_0_)
        min_sv_Sp_0[jn, js] = minimum(svdvals(Sp_0_))
        G_0 = Sp_0_ \ Sm_0_
        rho_0[jn, js] = maximum(abs.(eigvals(G_0)))

        # Rotating frame frequencies
        Sp_r_, Sm_r_ = filon_S_plus_S_minus(A_matrix_funcs, frequencies_rot, t_n, t_np1, s)
        cond_Sp_r[jn, js]  = cond(Sp_r_)
        cond_Sm_r[jn, js]  = cond(Sm_r_)
        min_sv_Sp_r[jn, js] = minimum(svdvals(Sp_r_))
        G_r = Sp_r_ \ Sm_r_
        rho_r[jn, js] = maximum(abs.(eigvals(G_r)))

        @printf("  %-10d  %-12.4e  %-14.4e %-14.4e %-14.4e %-14.4e %-14.4e %-14.4e\n",
            ns, dt,
            cond_Sp_0[jn, js], cond_Sm_0[jn, js],
            cond_Sp_r[jn, js], cond_Sm_r[jn, js],
            min_sv_Sp_0[jn, js], min_sv_Sp_r[jn, js])
    end
end

# Print spectral radius table
println("\n\n" * "=" ^ 100)
println("Spectral radius of G = S_+^{-1} S_- (amplification matrix) vs nsteps")
println("  ρ(G) > 1 means the method is unstable at that resolution")
println("=" ^ 100)

println(@sprintf("\n  %-10s  %-12s  %-18s %-18s %-18s  %-18s %-18s %-18s",
    "nsteps", "dt",
    "ρ(G), s=0, ω=0", "ρ(G), s=1, ω=0", "ρ(G), s=2, ω=0",
    "ρ(G), s=0, ω=rot", "ρ(G), s=1, ω=rot", "ρ(G), s=2, ω=rot"))
println("  " * "-"^140)

for (jn, ns) in enumerate(nsteps_list)
    @printf("  %-10d  %-12.4e  %-18.10f %-18.10f %-18.10f  %-18.10f %-18.10f %-18.10f\n",
        ns, dt_list[jn], rho_0[jn, :]..., rho_r[jn, :]...)
end


# ============================================================
# Plots
# ============================================================

plot_dir = joinpath(@__DIR__, "../..", "Plots")
mkpath(plot_dir)

# Build filename prefix from experiment parameters
if all(iszero, pcof)
    _pcof_tag = "pcof-zero"
elseif length(unique(pcof)) == 1
    _pcof_tag = "pcof-const$(pcof[1])"
elseif length(unique(pcof)) > 10
    _pcof_tag = "pcof-orig"
else
    _pcof_tag = "pcof-other"
end
_size_tag = "Nosc$(N_osc_levels)_Nguard$(N_guard_levels)"
_taylor_tag = use_taylor_moments ? "taylor-on" : "taylor-off"
_nonzero_idx = findall(!iszero, ψ0_unnormalized)
if length(_nonzero_idx) == 1
    _ic_tag = "ic-e$(_nonzero_idx[1])"
elseif length(_nonzero_idx) == length(ψ0_unnormalized) && length(unique(ψ0_unnormalized)) == 1
    _ic_tag = "ic-uniform"
else
    _ic_tag = "ic-other"
end
_cfreq_tag = all(iszero, Cfreq) ? "Cfreq-zero" : "Cfreq-nonzero"
_tmax_tag = "T$(Tmax)"
_float_tag = "$(FloatType)"

_plot_prefix = join([_pcof_tag, _size_tag, _taylor_tag, _ic_tag, _cfreq_tag, _tmax_tag, _float_tag], "_")
println("\nPlot filename prefix: $(_plot_prefix)")

s_labels = ["s=$s (order $(2*(s+1)))" for s in s_values]
s_markers = [:circle, :utriangle, :rect]
s_colors = Makie.wong_colors()

# --- Plot 1: cond(S+) vs dt ---

fig1 = Figure(size=(900, 500))
ax1 = Axis(fig1[1, 1],
    xlabel=L"$\Delta t$", ylabel=L"\kappa(S_+)",
    xscale=log10, yscale=log10,
    title="Condition number of S₊ vs Δt")

for js in 1:n_s
    scatterlines!(ax1, dt_list, cond_Sp_0[:, js],
        label="$(s_labels[js]), ω=0",
        marker=s_markers[js], color=(s_colors[js], 0.5), linestyle=:dash)
    scatterlines!(ax1, dt_list, cond_Sp_r[:, js],
        label="$(s_labels[js]), ω=rot",
        marker=s_markers[js], color=s_colors[js])
end

Legend(fig1[1, 2], ax1, framevisible=false)

save(joinpath(plot_dir, "cnot3_cond_Splus_vs_dt_$(_plot_prefix).png"), fig1)
println("\nSaved: Plots/cnot3_cond_Splus_vs_dt_$(_plot_prefix).png")


# --- Plot 2: min singular value of S+ vs dt ---

fig2 = Figure(size=(900, 500))
ax2 = Axis(fig2[1, 1],
    xlabel=L"$\Delta t$", ylabel=L"\sigma_{\min}(S_+)",
    xscale=log10, yscale=log10,
    title="Minimum singular value of S₊ vs Δt")

for js in 1:n_s
    scatterlines!(ax2, dt_list, min_sv_Sp_0[:, js],
        label="$(s_labels[js]), ω=0",
        marker=s_markers[js], color=(s_colors[js], 0.5), linestyle=:dash)
    scatterlines!(ax2, dt_list, min_sv_Sp_r[:, js],
        label="$(s_labels[js]), ω=rot",
        marker=s_markers[js], color=s_colors[js])
end

Legend(fig2[1, 2], ax2, framevisible=false)

save(joinpath(plot_dir, "cnot3_min_sv_Splus_vs_dt_$(_plot_prefix).png"), fig2)
println("Saved: Plots/cnot3_min_sv_Splus_vs_dt_$(_plot_prefix).png")


# --- Plot 3: spectral radius ρ(G) vs dt ---

fig3 = Figure(size=(900, 500))
ax3 = Axis(fig3[1, 1],
    xlabel=L"$\Delta t$", ylabel=L"\rho(G)",
    xscale=log10,
    title="Spectral radius of G = S₊⁻¹S₋ vs Δt")

hlines!(ax3, [1.0], color=:red, linestyle=:dash, linewidth=1, label="ρ = 1")

for js in 1:n_s
    scatterlines!(ax3, dt_list, rho_0[:, js],
        label="$(s_labels[js]), ω=0",
        marker=s_markers[js], color=(s_colors[js], 0.5), linestyle=:dash)
    scatterlines!(ax3, dt_list, rho_r[:, js],
        label="$(s_labels[js]), ω=rot",
        marker=s_markers[js], color=s_colors[js])
end

Legend(fig3[1, 2], ax3, framevisible=false)

save(joinpath(plot_dir, "cnot3_spectral_radius_vs_dt_$(_plot_prefix).png"), fig3)
println("Saved: Plots/cnot3_spectral_radius_vs_dt_$(_plot_prefix).png")


# ============================================================
# Time-resolved conditioning study
# ============================================================

println("\n\n" * "=" ^ 100)
println("Time-resolved conditioning analysis")
println("=" ^ 100)

nsteps_labels = ["nsteps=$ns" for ns in time_resolved_nsteps]
n_tr = length(time_resolved_nsteps)

# Compute ρ(G), κ(S₊), σ_min(S₊) at every timestep interval for each (nsteps, s)
# _0 = zero frequencies, _r = rotating frame frequencies
tr_rho_0   = Vector{Matrix{Float64}}(undef, n_tr)
tr_cond_0  = Vector{Matrix{Float64}}(undef, n_tr)
tr_minsv_0 = Vector{Matrix{Float64}}(undef, n_tr)
tr_rho_r   = Vector{Matrix{Float64}}(undef, n_tr)
tr_cond_r  = Vector{Matrix{Float64}}(undef, n_tr)
tr_minsv_r = Vector{Matrix{Float64}}(undef, n_tr)
tr_tmid    = Vector{Vector{Float64}}(undef, n_tr)

for (j_ns, ns) in enumerate(time_resolved_nsteps)
    dt = Tmax / ns
    tr_rho_0[j_ns]   = zeros(ns, n_s)
    tr_cond_0[j_ns]  = zeros(ns, n_s)
    tr_minsv_0[j_ns] = zeros(ns, n_s)
    tr_rho_r[j_ns]   = zeros(ns, n_s)
    tr_cond_r[j_ns]  = zeros(ns, n_s)
    tr_minsv_r[j_ns] = zeros(ns, n_s)
    tr_tmid[j_ns]    = [(n + 0.5) * dt for n in 0:ns-1]

    println("\nnsteps=$ns, dt=$(@sprintf("%.4e", dt))")
    for (js, s) in enumerate(s_values)
        for n in 0:ns-1
            t_n = n * dt
            t_np1 = (n + 1) * dt

            Sp_0, Sm_0 = filon_S_plus_S_minus(A_matrix_funcs, frequencies_zero, t_n, t_np1, s)
            G_0 = Sp_0 \ Sm_0
            tr_rho_0[j_ns][n+1, js]   = maximum(abs.(eigvals(G_0)))
            tr_cond_0[j_ns][n+1, js]  = cond(Sp_0)
            tr_minsv_0[j_ns][n+1, js] = minimum(svdvals(Sp_0))

            Sp_r, Sm_r = filon_S_plus_S_minus(A_matrix_funcs, frequencies_rot, t_n, t_np1, s)
            G_r = Sp_r \ Sm_r
            tr_rho_r[j_ns][n+1, js]   = maximum(abs.(eigvals(G_r)))
            tr_cond_r[j_ns][n+1, js]  = cond(Sp_r)
            tr_minsv_r[j_ns][n+1, js] = minimum(svdvals(Sp_r))
        end
        @printf("  s=%d: ρ(G,ω=0)   ∈ [%.6f, %.6f]  κ(S₊,ω=0)   ∈ [%.2e, %.2e]\n",
            s, minimum(tr_rho_0[j_ns][:, js]), maximum(tr_rho_0[j_ns][:, js]),
            minimum(tr_cond_0[j_ns][:, js]), maximum(tr_cond_0[j_ns][:, js]))
        @printf("  s=%d: ρ(G,ω=rot) ∈ [%.6f, %.6f]  κ(S₊,ω=rot) ∈ [%.2e, %.2e]\n",
            s, minimum(tr_rho_r[j_ns][:, js]), maximum(tr_rho_r[j_ns][:, js]),
            minimum(tr_cond_r[j_ns][:, js]), maximum(tr_cond_r[j_ns][:, js]))
    end
end


# --- Plot 4: ρ(G) vs t ---

fig4 = Figure(size=(350 * n_s + 150, 500))
for js in 1:n_s
    ax_0 = Axis(fig4[1, js],
        ylabel=js == 1 ? L"\rho(G)" : "",
        title="$(s_labels[js]), ω=0")
    hlines!(ax_0, [1.0], color=:red, linestyle=:dash, linewidth=1)
    for j_ns in 1:n_tr
        lines!(ax_0, tr_tmid[j_ns], tr_rho_0[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end

    ax_r = Axis(fig4[2, js],
        xlabel=L"t", ylabel=js == 1 ? L"\rho(G)" : "",
        title="$(s_labels[js]), ω=rot")
    hlines!(ax_r, [1.0], color=:red, linestyle=:dash, linewidth=1)
    for j_ns in 1:n_tr
        lines!(ax_r, tr_tmid[j_ns], tr_rho_r[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end
end
Legend(fig4[1:2, n_s + 1],
    [LineElement(color=s_colors[j]) for j in 1:n_tr],
    nsteps_labels, framevisible=false)

save(joinpath(plot_dir, "cnot3_spectral_radius_vs_t_$(_plot_prefix).png"), fig4)
println("\nSaved: Plots/cnot3_spectral_radius_vs_t_$(_plot_prefix).png")


# --- Plot 5: κ(S₊) vs t ---

fig5 = Figure(size=(350 * n_s + 150, 500))
for js in 1:n_s
    ax_0 = Axis(fig5[1, js],
        ylabel=js == 1 ? L"\kappa(S_+)" : "",
        yscale=log10,
        title="$(s_labels[js]), ω=0")
    for j_ns in 1:n_tr
        lines!(ax_0, tr_tmid[j_ns], tr_cond_0[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end

    ax_r = Axis(fig5[2, js],
        xlabel=L"t", ylabel=js == 1 ? L"\kappa(S_+)" : "",
        yscale=log10,
        title="$(s_labels[js]), ω=rot")
    for j_ns in 1:n_tr
        lines!(ax_r, tr_tmid[j_ns], tr_cond_r[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end
end
Legend(fig5[1:2, n_s + 1],
    [LineElement(color=s_colors[j]) for j in 1:n_tr],
    nsteps_labels, framevisible=false)

save(joinpath(plot_dir, "cnot3_cond_Splus_vs_t_$(_plot_prefix).png"), fig5)
println("Saved: Plots/cnot3_cond_Splus_vs_t_$(_plot_prefix).png")


# --- Plot 6: σ_min(S₊) vs t ---

fig6 = Figure(size=(350 * n_s + 150, 500))
for js in 1:n_s
    ax_0 = Axis(fig6[1, js],
        ylabel=js == 1 ? L"\sigma_{\min}(S_+)" : "",
        yscale=log10,
        title="$(s_labels[js]), ω=0")
    for j_ns in 1:n_tr
        lines!(ax_0, tr_tmid[j_ns], tr_minsv_0[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end

    ax_r = Axis(fig6[2, js],
        xlabel=L"t", ylabel=js == 1 ? L"\sigma_{\min}(S_+)" : "",
        yscale=log10,
        title="$(s_labels[js]), ω=rot")
    for j_ns in 1:n_tr
        lines!(ax_r, tr_tmid[j_ns], tr_minsv_r[j_ns][:, js],
            label=nsteps_labels[j_ns], color=s_colors[j_ns])
    end
end
Legend(fig6[1:2, n_s + 1],
    [LineElement(color=s_colors[j]) for j in 1:n_tr],
    nsteps_labels, framevisible=false)

save(joinpath(plot_dir, "cnot3_min_sv_Splus_vs_t_$(_plot_prefix).png"), fig6)
println("Saved: Plots/cnot3_min_sv_Splus_vs_t_$(_plot_prefix).png")
