"""
This file contains helper function for converting `SchrodingerProb`s and
controlls into controlled operators, which can be used by FilonResearch.
"""

"""
Turn a `QuantumGateDesign` control and turn the p-part into a function f(t)
"""
function QGD_control_to_func_p(
    control::QuantumGateDesign.AbstractControl,
    pcof::AbstractVector{<: Real},
    derivative_order::Integer,
)
    return t -> eval_p_derivative(control, t, pcof, derivative_order)
end

"""
Turn a `QuantumGateDesign` control and turn the q-part into a function f(t)
"""
function QGD_control_to_func_q(
    control::QuantumGateDesign.AbstractControl,
    pcof::AbstractVector{<: Real},
    derivative_order::Integer,
)
    return t -> eval_q_derivative(control, t, pcof, derivative_order)
end

"""
Given a `QuantumGateDesign` `SchrodingerProb`, a control, and a control vector,
implement (a derivative of) the Hamiltonian as a `ControlledFunctionOp`.
"""
function QGD_prob_to_controlled_op(
    prob::QuantumGateDesign.SchrodingerProb,
    controls, pcof::AbstractVector{<: Real},
    derivative_order::Integer,
)
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

"""
Given a `QuantumGateDesign` `SchrodingerProb`, a control, and a control vector,
implement the Hamiltonian and its derivatives as a tuple of
`ControlledFunctionOp`s. This is what is expected to be fed into the Filon
method.
"""
function QGD_prob_to_filon_hamiltonian(
    prob::QuantumGateDesign.SchrodingerProb,
    controls,
    pcof::AbstractVector{<: Real},
    max_deriv_order=3,
)
    return ntuple(
        i -> QGD_prob_to_controlled_op(prob, controls, pcof, i-1),
        1+max_deriv_order
    )
end

# =============================================================================
# Frame transformations for CarrierControl-based problems
#
# QuantumGateDesign's rotating-frame Hamiltonian applies the rotating wave
# approximation (RWA): with α_k(t) = p_k(t) + i q_k(t) the complex envelope of
# control k and S_k = a_k + a_k†, A_k = a_k − a_k†, the control term is
#
#     H_ctrl^rwa = Σₖ p_k S_k + q_k (i A_k)            (amplitude α_k on a_k).
#
# Both correspond to the laboratory-frame drive  f_k(t) (a_k + a_k†)  with the
# real lab pulse  f_k(t) = 2 Re[α_k(t) e^{i ω_k t}],  ω_k the rotation (angular)
# frequency of control k's subsystem.  The two functions below construct the
# controls/pcof for the same physical drive in the other frames.  Each
# `controls[k]` must be a QuantumGateDesign `CarrierControl` whose base control
# stores a coefficient slice as [p-half; q-half] (the standard QGD layout), so
# that envelope conjugation is "negate the q-half".
# =============================================================================

# The coefficient slice of carrier f within a CarrierControl's pcof.
function carrier_slice(control::QuantumGateDesign.CarrierControl, pcof_k, f)
    nbase = control.base_control.N_coeff
    return pcof_k[(f - 1) * nbase + 1 : f * nbase]
end

# Conjugate a base-control envelope E = p + iq by negating the q-half.
function conjugate_base_slice(slice)
    iseven(length(slice)) || throw(ArgumentError(
        "base-control slice length must be even ([p-half; q-half] layout)"))
    out = copy(slice)
    half = length(slice) ÷ 2
    out[half + 1 : end] .*= -1
    return out
end

"""
    drop_rwa(controls, pcof, rotation_freqs) -> (controls, pcof)

Rebuild rotating-frame `CarrierControl`s *without* the rotating wave
approximation.  In the rotating frame the full control term is
f_k(t)(a_k e^{−iω_k t} + a_k† e^{iω_k t}), i.e. complex amplitude on a_k

    γ_k(t) = α_k(t) + conj(α_k(t)) e^{−2iω_k t} ,

so each carrier Ω (envelope E) gains a partner at −(2ω_k + Ω) carrying the
conjugated envelope.  `rotation_freqs[k]` is the angular rotation frequency
(rad/time) of control k's subsystem.  The drift Hamiltonian is unchanged.
"""
function drop_rwa(controls, pcof, rotation_freqs)
    length(controls) == length(rotation_freqs) || throw(ArgumentError(
        "need one rotation frequency per control"))
    new_controls = QuantumGateDesign.CarrierControl[]
    new_pcof = Float64[]
    for (k, control) in enumerate(controls)
        control isa QuantumGateDesign.CarrierControl || throw(ArgumentError(
            "frame transformations expect QuantumGateDesign.CarrierControls; " *
            "control $k is a $(typeof(control))"))
        pcof_k = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        Ω = control.carrier_frequencies
        ω = rotation_freqs[k]
        freqs = vcat(Ω, -2ω .- Ω)
        push!(new_controls,
              QuantumGateDesign.CarrierControl(control.base_control, freqs))
        for f in eachindex(Ω)
            append!(new_pcof, carrier_slice(control, pcof_k, f))
        end
        for f in eachindex(Ω)
            append!(new_pcof, conjugate_base_slice(carrier_slice(control, pcof_k, f)))
        end
    end
    return new_controls, new_pcof
end

"""
    to_lab_frame(controls, pcof, rotation_freqs) -> (controls, pcof)

Rebuild the controls so that the *p-part alone* equals the real laboratory
pulse f_k(t) = 2 Re[α_k(t) e^{i ω_k t}]: carriers shift to ω_k + Ω and the
coefficients double.  Intended for a lab-frame problem whose antisymmetric
control operators are zero (the lab drive couples only through a_k + a_k†);
the q-part of the returned controls is nonzero but must multiply a zero
operator.
"""
function to_lab_frame(controls, pcof, rotation_freqs)
    length(controls) == length(rotation_freqs) || throw(ArgumentError(
        "need one rotation frequency per control"))
    new_controls = QuantumGateDesign.CarrierControl[]
    new_pcof = Float64[]
    for (k, control) in enumerate(controls)
        control isa QuantumGateDesign.CarrierControl || throw(ArgumentError(
            "frame transformations expect QuantumGateDesign.CarrierControls; " *
            "control $k is a $(typeof(control))"))
        pcof_k = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        freqs = rotation_freqs[k] .+ control.carrier_frequencies
        push!(new_controls,
              QuantumGateDesign.CarrierControl(control.base_control, freqs))
        append!(new_pcof, 2 .* pcof_k)
    end
    return new_controls, new_pcof
end

# =============================================================================
# Adapters to the *new* ControlledOperator-based Filon methods
# (`filon_solve_hardcoded`, `controlled_filon_solve`).
#
# Unlike the legacy `QGD_prob_to_filon_hamiltonian` above (which returns a tuple
# of derivative operators), these build a single `ControlledOperator`
# A(t) = Σₖ cₖ(t) Aₖ whose controls carry the time dependence; the solver
# differentiates the controls internally.
# =============================================================================

using LinearAlgebra: diag

"""
    qgd_ansatz_frequencies(prob) -> Vector{Float64}

The oscillatory-ansatz frequencies for the Filon methods: the negated diagonal
of the (rotating-frame) symmetric system Hamiltonian, one per state component.
"""
function qgd_ansatz_frequencies(prob::QuantumGateDesign.SchrodingerProb)
    return -1.0 .* Array(diag(prob.system_sym))
end

"""
    qgd_to_controlled_operator(prob, controls, pcof) -> ControlledOperator

Build the time-dependent generator A(t) of the rotating-frame Schrödinger
equation dψ/dt = A(t)ψ as a `ControlledOperator`, for use with
`filon_solve_hardcoded`.  With Hₛ = system_sym + i·system_asym, Sₖ the k-th
symmetric control operator and Aₖ the k-th antisymmetric one,

    A(t) = -i Hₛ + Σₖ pₖ(t) (-i Sₖ) + Σₖ qₖ(t) Aₖ ,

where pₖ, qₖ (and their derivatives) come from the QGD controls.  The matrices
are stored as a dynamic `Vector` (the system is large) and the controls as a
heterogeneous `Tuple` (type-stable drift-plus-controls layout).
"""
function qgd_to_controlled_operator(
    prob::QuantumGateDesign.SchrodingerProb, controls, pcof::AbstractVector{<: Real},
)
    Hsys = prob.system_sym .+ (im .* prob.system_asym)
    matrices = Matrix{ComplexF64}[Matrix{ComplexF64}(-im .* Hsys)]
    ctrls = Any[ConstantControl(1.0)]

    for (k, control) in enumerate(controls)
        pcof_k = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        Sk = Matrix{ComplexF64}(prob.sym_operators[k])
        Ak = Matrix{ComplexF64}(prob.asym_operators[k])

        push!(matrices, -im .* Sk)
        push!(ctrls, FunctionControl{Float64}(
            (t, n) -> eval_p_derivative(control, t, pcof_k, n)))

        push!(matrices, Ak)
        push!(ctrls, FunctionControl{Float64}(
            (t, n) -> eval_q_derivative(control, t, pcof_k, n)))
    end

    return ControlledOperator(Tuple(ctrls), matrices)
end

"""
    qgd_to_controlled_filon_operator(prob, controls, pcof) -> ControlledOperator

Like [`qgd_to_controlled_operator`](@ref), but with each control's carrier waves
factored out so the result can be fed to `controlled_filon_solve`.  Writing the
complex control αₖ(t) = pₖ(t) + i qₖ(t) = Σ_f Eₖ,_f(t) e^{i ω_{k,f} t} (with
envelope Eₖ,_f = pₖ,_f + i qₖ,_f from the f-th carrier's B-spline), the per-control
generator contribution -i pₖ Sₖ + qₖ Aₖ becomes

    Σ_f [ Eₖ,_f(t) e^{+i ω_{k,f} t} M⁺ₖ + conj(Eₖ,_f(t)) e^{-i ω_{k,f} t} M⁻ₖ ] ,

with constant matrices M⁺ₖ = -i/2 (Sₖ + Aₖ), M⁻ₖ = -i/2 (Sₖ - Aₖ).  Each (k,f)
yields two `CarrierControl` terms (carriers ±ω_{k,f}).  This reproduces exactly
the same A(t) as [`qgd_to_controlled_operator`](@ref), only regrouped.

Each `controls[k]` must be a QuantumGateDesign `CarrierControl` (it exposes
`carrier_frequencies` and `base_control`).
"""
function qgd_to_controlled_filon_operator(
    prob::QuantumGateDesign.SchrodingerProb, controls, pcof::AbstractVector{<: Real},
)
    Hsys = prob.system_sym .+ (im .* prob.system_asym)
    matrices = Matrix{ComplexF64}[Matrix{ComplexF64}(-im .* Hsys)]
    ctrls = Any[ConstantControl(1.0)]

    for (k, control) in enumerate(controls)
        control isa QuantumGateDesign.CarrierControl || throw(ArgumentError(
            "controlled-Filon adapter expects QuantumGateDesign.CarrierControls; " *
            "control $k is a $(typeof(control))"))

        pcof_k = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
        base = control.base_control
        nbase = base.N_coeff

        Sk = Matrix{ComplexF64}(prob.sym_operators[k])
        Ak = Matrix{ComplexF64}(prob.asym_operators[k])
        Mplus  = (-im / 2) .* (Sk .+ Ak)
        Mminus = (-im / 2) .* (Sk .- Ak)

        for (f, ωf) in enumerate(control.carrier_frequencies)
            slice_f = view(pcof_k, (f - 1) * nbase + 1 : f * nbase)
            env = FunctionControl{ComplexF64}(
                (t, n) -> eval_p_derivative(base, t, slice_f, n) +
                          im * eval_q_derivative(base, t, slice_f, n))
            conj_env = FunctionControl{ComplexF64}(
                (t, n) -> eval_p_derivative(base, t, slice_f, n) -
                          im * eval_q_derivative(base, t, slice_f, n))

            push!(matrices, Mplus)
            push!(ctrls, FilonResearch.CarrierControl(env, ωf))
            push!(matrices, Mminus)
            push!(ctrls, FilonResearch.CarrierControl(conj_env, -ωf))
        end
    end

    return ControlledOperator(Tuple(ctrls), matrices)
end

"""
    eval_forward_complex_history(qgd_prob, controls, pcof, initial_condition;
                                 order, nsteps, saveEveryNsteps) -> Matrix{ComplexF64}

Run QuantumGateDesign's Hermite solver `eval_forward` for the single initial
condition `initial_condition`, returning the saved state history as a complex
`N × (1 + nsteps/saveEveryNsteps)` matrix — the same layout the Filon solvers
return, so the two are directly comparable.
"""
function eval_forward_complex_history(
    qgd_prob::QuantumGateDesign.SchrodingerProb, controls, pcof::AbstractVector{<: Real},
    initial_condition::AbstractVector{<: Number};
    order::Integer, nsteps::Integer, saveEveryNsteps::Integer,
)
    prob = QuantumGateDesign.VectorSchrodingerProb(qgd_prob, 1)
    prob.nsteps = nsteps
    prob.u0 .= real(initial_condition)
    prob.v0 .= imag(initial_condition)
    history = eval_forward(prob, controls, pcof; order, saveEveryNsteps)
    return Matrix{ComplexF64}(history)
end
