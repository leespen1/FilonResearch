using QuantumGateDesign
using FilonResearch
using LinearAlgebra


"""
Given a "control" from QuantumGateDesign and a control vector, convert to a
function which takes time `t` as input, and outputs the value of the
real/p part of the control at that time (with that control vector).
"""
function QGD_control_to_func_p(control::QuantumGateDesign.AbstractControl, pcof::AbstractVector{<: Real}, derivative_order::Integer)
    return t -> eval_p_derivative(control, t, pcof, derivative_order)
end

"""
Given a "control" from QuantumGateDesign and a control vector, convert to a
function which takes time `t` as input, and outputs the value of the
imaginary/q part of the control (as a complex scalar) at that time (with that control vector).
"""
function QGD_control_to_func_q(control::QuantumGateDesign.AbstractControl, pcof::AbstractVector{<: Real}, derivative_order::Integer)
    return t -> eval_q_derivative(control, t, pcof, derivative_order)
end


"""
Given a QuantumGateDesign problem, a set of controls, a control vector, and a derivative order,
return the controled function op which implements -iH(t;θ) (with derivatives taken)
"""
function QGD_prob_to_controlled_op(prob::QuantumGateDesign.SchrodingerProb, controls, pcof, derivative_order::Integer)

    # The system hamiltonian is ignored if we take a derivative, since it is time-independent.
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
    hamiltonians = vcat([system_hamiltonian], prob.sym_operators, im .* prob.asym_operators)

    return ControlledFunctionOp(-im .* hamiltonians, coefficient_functions)
end

"""
Copied from JCP_HighOrderHermiteOptimization branch of QuantumGateDesign.jl.

Given JuqboxParams, construct BCarrier control functions with the same number of
control coefficients, but with the appropriate level of smoothness for the
method order.
"""
function get_controls(method_order::Integer, D1::Integer, Cfreq::AbstractMatrix{<: Real}, tf::Real)
    # A degree N BSpline has continuous derivatives up to order N-1 
    degree = method_order
    base_control = FortranBSplineControl(degree, D1, tf)

    base_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
    base_control = QuantumGateDesign.FortranBSplineControl2(base_bspline, tf)
    controls = [CarrierControl(base_control, freqs) for freqs in eachrow(Cfreq)]

    return controls
end


N_osc_levels = 10
Tmax = 550.0
nsteps = 1_024

degree = 14
D1 = 16

subsystem_sizes = (N_osc_levels, 4, 4)
essential_subsystem_sizes = (1, 2, 2)

fa = 4.10595
fb = 4.81526
fs = 7.8447
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2/xa
xab = 1.0e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs # Rotating Frame

kerr_coeffs = Symmetric(
    [xs   xbs   xas;
     xbs  xb    xab;
     xas  xab   xa],
    :U
)


rot_prob = dispersive_qudits_problem(
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

lab_prob = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    (0,0,0), # For lab frame, use zero as each rotation frequency
    kerr_coeffs,
    Tmax,
    nsteps,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)

U0 = lab_prob.u0 + im.*lab_prob.v0

rot_op_T = compsys_rot_frame_op(rotation_freqs, subsystem_sizes, Tmax)

lab_target = U0*CNOT_gate()
rot_target = rot_op_T*lab_target

Nctrl = 3
Nfreq = 3
Cfreq = zeros(Nctrl,Nfreq)
Cfreq[1:2,2] .= -2.0*pi*xa # carrier freq's for ctrl Hamiltonian 1 & 2
Cfreq[1:2,3] .= -2.0*pi*xb # carrier freq's for ctrl Hamiltonian 1 & 2
Cfreq[3,2] = -2.0*pi*xas # carrier freq 2 for ctrl Hamiltonian #3
Cfreq[3,3] = -2.0*pi*xbs # carrier freq 2 for ctrl Hamiltonian #3

Cfreq_lab_shift = repeat(collect(rotation_freqs), 1, Nfreq) .* 2.0*pi
Cfreq_lab = Cfreq .+ Cfreq_lab_shift

rot_controls = get_controls(degree, D1, Cfreq, Tmax)

#= #Need to implement these in Main before I can do this here
lab_controls = [QuantumGateDesign.DoubleRealPartControl(control)
                for control in get_controls(degree, D1, Cfreq_lab, Tmax)]

# Alternative implementation, which should be slower
lab_controls2 = [QuantumGateDesign.LabFrameControl(rot_controls[i], 2.0*pi*rotation_freqs[i])
                 for i in 1:3]
=#

# Hard-coded control vector, resulting from optimizing for CNOT3 gate with degree=14, D1=16
pcof = [0.014206703047363163,-0.0007404000550756441,-0.0009984714672589814,-0.0005694728281868248,-0.002501715745115248,0.0038132941334898416,0.0007116462317267991,-0.006841223438087476,-0.0023070585150656102,0.002726912304185669,0.0021534345504655482,0.0008411826515444047,-0.0018506535209815973,-0.003016058630555202,0.016049633966934613,0.017528890160984745,-0.031046857921623247,-0.010471231793747007,0.00576673415918385,-0.003037235085607017,0.0019034563937884086,-0.00269621548025083,-0.0022405789846467296,0.0027608356869845583,0.0042540873895280185,-0.0019410612339501392,0.002351716734598865,-0.0015891035993499884,0.00027566389089670037,0.00533573820808008,-0.005142978495560767,-0.02031166808124263,0.0038872595006349928,0.001749401701453946,-0.00641073199837208,-0.006889073357649663,-0.008116885234440926,0.0009660405685911396,0.008956073522034327,0.007549775920374156,0.011851558070851547,0.0051721975016686215,0.0025305278942518504,-0.004615597716686936,-0.009188552704360168,-0.0009826751804701444,0.004137740604789321,0.001976294076873065,-0.004581053667415471,0.007842651828502228,0.008731345857587123,-0.0006410027001324836,-0.010967403591470118,-0.014259223484843993,-0.00650306287617129,-0.0019148512764557185,-0.002211672527070954,0.007902705812502554,0.00807921721894839,0.0074589666546925745,0.005585593479441905,-0.006683495538458927,-0.009384665905412622,0.003874246382540184,-0.0031242840168878615,0.001928120687756253,0.0006693266711531669,0.0037148072679799022,-0.003641568161234867,-0.00365062772307532,0.0032894196045000716,-0.0023920420539574015,0.004577443983171143,-0.0055931230951403715,-5.092968148866822e-5,0.0030609312881264747,-0.002360386820624622,0.0044397005102192275,0.0009736977770266375,0.0005161127086999689,-0.0051961848866031515,0.002635932048507368,0.00024038401297035025,-0.0008219857847728555,0.0017384780236516897,0.003964850215556359,-0.0029353474747559377,-0.0013307597929656716,3.295984420467496e-5,0.0033227639167528154,-0.00042180929769978105,-0.00378001225736713,0.0016127840500289967,-0.0001641888872019785,0.00019477933106718437,0.0035008013949833147,-0.0004072308017653091,0.0003674091175372138,0.0003587496430154635,-0.0007882651771858813,0.00014953288551707066,9.65224553036758e-5,0.001632583121587648,-0.0026455651231152896,0.0008689711764723674,8.234711269603958e-5,0.002100667499337898,-0.0031970690652175266,0.00048645460193190486,0.002236433194188343,-0.002215606735306239,0.0013359251223616047,0.0006786858931989296,-0.00171056558916783,0.0026090900465467045,-0.002311818405538429,0.0013342673317499987,-0.0008398793061069168,0.0006900616402598058,0.0008538827347281442,-0.002991595583182856,0.003557834976491093,-0.0025393314050465073,0.0005187002242051925,0.002017397908124808,-0.003518869036150964,0.0032994378826123813,-0.002684348735972484,-0.003891929590288017,-0.0012408024098798096,-0.0020244985089209857,0.001948797501147367,0.0023378069280728313,0.004924588948523233,0.0020794463996769755,-0.0014518589868631176,-0.0028807598621154533,-0.0009757427275965665,-0.007492536497771161,-0.001967638334279097,-0.0069178517108224435,-0.003918863672180314,-0.0026997382411946577,-0.003926668256901013,-0.0029796892270391815,0.005347537191108047,0.0017589907868812273,-0.0014754664858328303,-0.003375183456047392,0.0032180271047125937,0.0032137265037543526,0.0009783243302176985,0.0005800037566817189,-0.0045722695461994116,0.001488089005718878,-0.0023791221121741185,0.005614233508721144,0.004921531955982609,0.006362313990562208,-0.0006638371794847855,-0.004420850857388265,0.0012792962083531915,0.002881926345854144,0.007177648839411087,0.004578372939404998,0.0026567829612900527,0.005007583283451031,-0.0037930939300424432,-0.0005138965300716757,-0.0017017194085404014,-0.010783731144485529,-0.005449130765573482,-0.003615890442526304,0.0013185552744913193,-0.0013183658842076998,-0.002154001346376969,-0.0028805908826307148,-0.0028118638088218337,-0.004574963295667305,-0.005676726099109537,-0.011402465810840888,-0.006172005547979964,-0.00243035402018065,-0.0026786376567304745,-0.00809900093687264,-0.006439216489041175,-0.005330018439044402,-0.007820064636981483,-0.006606544399090594,-0.00278589522725575,-0.0008859868160328908,-0.001625519360551106,-0.0064692279658987726,-0.005775482515237348,0.016096215609218936,0.01649559327099624,0.019180291791383346,0.009706953114914619,-0.005578470311832401,-0.011285077032252538,-0.018273317833363105,-0.022511383681479105,-0.02002655882802859,-0.009508994119235394,0.0038860973226832007,0.017264582004569068,0.011412631863814948,-0.00641615516327001,0.00733280934478686,-0.01903061930555173,-0.0097084864374851,0.0028568002178964245,0.014388776133406379,0.020246837176269936,0.015197591921245741,0.012299701185289143,0.01111772098734325,-0.0007089547830225728,-0.016105120105593288,-0.02041123528277712,-0.018073390784366965,-0.009957528977978495,0.013698829094931735,0.002013370882117593,-0.007993908433986962,0.0012440446038925045,0.009417974818047303,0.0012693278272154647,-0.009029360619055793,-0.0068655980256444905,0.005415329524354024,0.003248079323309337,0.0018391359620955011,0.0009221324721135398,-0.0020272555833737117,-0.0020206341012867558,0.0026046605057047693,0.005822756404979163,-0.0034692657131796642,0.002373016665043436,0.00272067852400584,-0.012695647240846767,-0.0003759775382934763,0.007319318614630666,-0.0013253767701881158,-0.00597078764978222,-0.00502170865852333,0.0019366438500330871,0.002118403413013919,0.0025342464538708693,-0.0006561622871415026,-0.003433363091115776,0.0011550408057884771,-0.0004474437102222443,0.006061587296459784,-0.011212339271104576,-0.002037669914416563,-0.002018667528883307,0.004055171134696443,0.0035202098739393656,-0.007138072391258802,0.0001536015205630304,0.003694298088620965,0.00831692297690751,-0.00021040861013524323,-0.0053531640693303024,-0.005814150889267475,0.0017112603264415136,0.008703334038138477,3.455283113401425e-5,-0.006150902914373094,0.001695514442260664,0.009962047875121036,-0.008693306597882311,0.007012436868809998,0.011943692643412487,-0.001262942508456683,-0.005823008403364433,-0.00502543509839941,-0.0011517450525467994,0.006027010068473857,0.001839485294707081,0.0013492114647962392,-0.0037273154288677084,-0.0028904197805052867,-0.0007386937212103546,0.007904299964579984,-0.005178057940816483]
#pcof .*= 0 # To use no controls

#=
# Dummy run to compile
rot_prob.nsteps = 1
lab_prob.nsteps = 1
history_rot = eval_forward(rot_prob, rot_controls, pcof, order=6)
history_lab = eval_forward(lab_prob, lab_controls, pcof, order=6)

# Actual run
rot_prob.nsteps = 100
lab_prob.nsteps = 100
#@time history_rot = eval_forward(rot_prob, rot_controls, pcof, order=6)
@time history_lab = eval_forward(lab_prob, lab_controls, pcof, order=6)


include("stepsize_copy.jl")
order = 6
max_walltime = 1 # In hours
filename_base = "lab_frame"
nsaves = 10
initial_nsteps = 50_000 # Nyquist limit should be about here
collect_data(lab_prob, lab_controls, pcof, order, max_walltime, filename_base,
             nsaves, initial_nsteps)
=#


# Check that this provides the correct result!
#
max_deriv_order = 5
A_deriv_funcs = ntuple(
    i -> QGD_prob_to_controlled_op(rot_prob, rot_controls, pcof, i-1),
    max_deriv_order
)



s_filon = 0
s_hermite = 2
frequencies = zeros(div(rot_prob.real_system_size, 2))
#frequencies = -1 .* Array(diag(rot_prob.system_sym))
# Try solving with filon and hermite. But use zero frequencies for filon, which should make it mathematically equivalent to hermite.
initial_cond = 4 # The fourth initial condition has phase dynamics, even with no control
f_hist = filon_solve(A_deriv_funcs, U0[:,initial_cond], frequencies, Tmax, nsteps, s_filon)
h_hist = eval_forward(
    QuantumGateDesign.VectorSchrodingerProb(rot_prob, initial_cond),
    rot_controls,
    pcof,
    order=2*(s_hermite+1)
)

@show norm(f_hist[:,end] - h_hist[:,end])
@show norm(f_hist - h_hist)
@show maximum(abs, f_hist - h_hist)
