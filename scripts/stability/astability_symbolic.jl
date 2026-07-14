# Symbolic verification of the A-stability identities in the manuscript's
# stability section (s = 1): the weights w_1, w_2 from the cubic Hermite
# cardinals, their real/imaginary decompositions, the collapse of stability
# condition 1 to the monomial H1 = 12 phi^8, and the perfect-square
# factorization G2 = 108 (phi^2+4)^2 (1 + cos(phi+theta))^2 of condition 2.
# Every [OK] line is an identity checked to be exactly zero symbolically.
#
#   julia --project=. scripts/stability/astability_symbolic.jl

using DrWatson
@quickactivate "FilonExperiments"

using SymPy

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
const II = sympy.I                       # imaginary unit as a Sym
repart(e) = sympy.re(e)                  # symbolic real part (phi is real)
impart(e) = sympy.im(e)                  # symbolic imaginary part
iszero_sym(e) = simplify(expand(e)) == 0 # is the expression identically 0 ?

npass = Ref(0); nfail = Ref(0)
function check(label, e)
    ok = iszero_sym(e)
    ok ? (npass[] += 1) : (nfail[] += 1)
    println(rpad(ok ? "  [OK] " : "  [XX] ", 8), label)
    ok || println("        residual = ", simplify(expand(e)))
    ok
end
section(s) = (println(); println("="^72); println(s); println("="^72))

# real symbols
φ = symbols("phi",   real=true)
σ = symbols("sigma", real=true)
x = symbols("x",     real=true)
y = symbols("y",     real=true)

# ======================================================================
section("STEP 0.  Coefficients w_1, w_2 from the cubic Hermite cardinals")
ℓE0 = (1 - σ)^2 * (1 + 2σ)      # \hat\ell_{E,0}
ℓE1 = σ * (1 - σ)^2             # \hat\ell_{E,1}

as = symbols("a", nonzero=true)
Emom(as, k) = k == 0 ? (exp(as) - 1) / as : exp(as)/as - (k/as) * Emom(as, k - 1)
for k in 0:3
    check("E_$k matches \\int_0^1 sigma^$k e^{a sigma} dsigma",
          Emom(as, k) - integrate(σ^k * exp(as * σ), (σ, 0, 1)))
end

a  = II * φ
E0 = Emom(a, 0); E1 = Emom(a, 1); E2 = Emom(a, 2); E3 = Emom(a, 3)
w_2 = E1 - 2*E2 + E3            
c0 = E0 - 3*E2 + 2*E3           
w_1 = c0 - II * φ * w_2    

# w_1 = (R1 + i I1)/phi^4 ,  w_2 = (R2 + i I2)/phi^4  (substitute e^{a}=cos+ i sin)
R1 = simplify(repart(φ^4 * w_1));  I1 = simplify(impart(φ^4 * w_1))
R2 = simplify(repart(φ^4 * w_2));  I2 = simplify(impart(φ^4 * w_2))

R1t = 4φ^2 + 12 + (2φ^2 - 12) * cos(φ) - 12φ * sin(φ)
I1t = 2φ^3 + 12φ * cos(φ) + (2φ^2 - 12) * sin(φ)
R2t = 6 - φ^2 - 6cos(φ) - 2φ * sin(φ)
I2t = 4φ + 2φ * cos(φ) - 6sin(φ)
check("R1 matches manuscript", R1 - R1t)
check("I1 matches manuscript", I1 - I1t)
check("R2 matches manuscript", R2 - R2t)
check("I2 matches manuscript", I2 - I2t)

G1 = expand(R1 * R2 + I1 * I2)          #  Re(w_1 abar2) = G1/phi^8
G2 = expand(R1 * G1 - φ^4 * I2^2)       #  Re(w_1)Re(w_1 abar2)-Im(w_2)^2 = G2/phi^12

check("G1/phi^8   = Re(w_1 conj w_2)",
      G1 / φ^8 - simplify(repart(w_1 * conj(w_2))))
check("G2/phi^12  = Re(w_1)Re(w_1 conj w_2) - Im(w_2)^2",
      G2 / φ^12 - simplify(repart(w_1) * repart(w_1 * conj(w_2)) - impart(w_2)^2))

# ======================================================================
section("Condition 1 collapses to the monomial H1 = 12 phi^8")
# ----------------------------------------------------------------------
Pc = 4φ^4 + 24φ^2 + 144
Qc = 2φ^4 + 48φ^2 - 144
Sc = -144φ
check("G1 = P + Q cos(phi) + S sin(phi)",
      G1 - (Pc + Qc * cos(φ) + Sc * sin(φ)))
H1 = expand(Pc^2 - Qc^2 - Sc^2)
check("H1 = P^2 - Q^2 - S^2 = 12 phi^8", H1 - 12φ^8)
println("  => G1 >= P - sqrt(Q^2+S^2) and H1 = 12 phi^8 >= 0  ==>  G1 >= 0.")

# ======================================================================
section("Condition 2  108 B^2 (1+cos(phi+theta))^2")
# ----------------------------------------------------------------------
B = φ^2 + 4
# the shared diagonalising phase theta:  cos(theta)=(phi^2-4)/B, sin(theta)=4phi/B
cosθ = (φ^2 - 4) / B
sinθ = 4φ / B
check("cos^2(theta)+sin^2(theta) = 1", cosθ^2 + sinθ^2 - 1)
check("cos2theta = (phi^4-24phi^2+16)/B^2",
      (2cosθ^2 - 1) - (φ^4 - 24φ^2 + 16) / B^2)
check("sin2theta = 8 phi(phi^2-4)/B^2",
      (2sinθ * cosθ) - 8φ * (φ^2 - 4) / B^2)

# the perfect square
cos_φθ  = cos(φ) * cosθ - sin(φ) * sinθ      # cos(phi + theta)
G2_paper  = 108 * B^2 * (1 + cos_φθ)^2
check("G2 = 108 (phi^2+4)^2 (1+cos(phi+theta))^2", G2 - G2_paper)

# the identity 3 + 4 cosPsi + cos2Psi = 2(1+cosPsi)^2 used to get the square
Ψ = symbols("Psi", real=true)
check("3 + 4 cos(Psi) + cos(2 Psi) = 2 (1+cos Psi)^2",
      (3 + 4cos(Ψ) + cos(2Ψ)) - 2 * (1 + cos(Ψ))^2)

# G2/phi^12 as the manuscript writes it
check("G2/phi^12 = 108(phi^2+4)^2(1+cos(phi+theta))^2 / phi^12",
      G2 / φ^12 - G2_paper / φ^12)

println("  passed: ", npass[], "    failed: ", nfail[])
nfail[] == 0 && println("  All manuscript identities verified symbolically. ∎")
