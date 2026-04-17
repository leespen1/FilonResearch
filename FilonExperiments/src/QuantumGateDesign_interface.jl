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
