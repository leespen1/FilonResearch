"""
Set up lab frame Hamiltonian matrix A(t) = -iH(t) and its derivatives for the
driven two-level system.
"""
function A_lab_funcs(;ω₀::Real, ω::Real, Ω::Number)
    A(t) =    -im .* [ω₀/2       Ω*cos(ω*t);
                      Ω*cos(ω*t)      -ω₀/2]
    # Derivatives
    dA(t) =   -im .* [0             -Ω*ω*sin(ω*t);
                      -Ω*ω*sin(ω*t)             0]
    d2A(t) =  -im .* [0               -Ω*ω^2*cos(ω*t);
                      -Ω*ω^2*cos(ω*t)               0]

    d3A(t) =  -im .* [0               Ω*ω^3*sin(ω*t);
                      Ω*ω^3*sin(ω*t)               0]
    return (A, dA, d2A, d3A)
end

"""
Set up rotating  frame Hamiltonian matrix A(t) = -iH(t) and its derivatives for
the driven two-level system.
"""
function A_rot_funcs(; Ω::Number, Δ::Real)
    A(t) = -im .* [Δ/2  Ω/2;
                   Ω/2  -Δ/2]
    dA(t) = zeros(ComplexF64, 2, 2)
    d2A(t) = zeros(ComplexF64, 2, 2)
    d3A(t) = zeros(ComplexF64, 2, 2)
    return (A, dA, d2A, d3A)
end

"""
Exact solution in the rotating frame (RWA).
Ω_eff = √(Δ² + Ω²) is the generalized Rabi frequency.
"""
function exact_rotating_frame(u0, Δ, Ω, t)
    A = A_rot(Δ, Ω)
    return exp(A * t) * u0
end

"""
Transform state from rotating frame to lab frame.
U(t) = diag(e^{-iωt/2}, e^{iωt/2})
ψ_lab(t) = U(t) ψ_rot(t)
"""
function rotating_to_lab(u_rot, ω, t)
    U = [cis(-ω*t/2)  0;
         0            cis(ω*t/2)]
    return U * u_rot
end

"""
Transform state from lab frame to rotating frame.
U†(t) = diag(e^{iωt/2}, e^{-iωt/2})
ψ_rot(t) = U†(t) ψ_lab(t)
"""
function lab_to_rotating(u_lab, ω, t)
    U_dag = [cis(ω*t/2)   0;
             0            cis(-ω*t/2)]
    return U_dag * u_lab
end
