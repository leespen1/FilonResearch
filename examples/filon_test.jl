using LinearAlgebra

s = 4

# Part 1: scalar case (done as vector case)
ws = [1]
A = zeros(ComplexF64, 1, 1)
A[1,1] = ws[1] * im 
u_n = [1]
t_n = 0
dt = 5
u_np1 = filon_timestep_static(A, u_n, ws, t_n, dt, s)[1]
u_np1_true = u_n[1] * cis(ws[1]*(t_n+dt))
println("Scalar (as vector) case (should be exact)")
@show abs.(u_np1-u_np1_true)

# Part 2: 2×2, independent example
ws = [1,2]
A = zeros(ComplexF64, 2, 2)
A[1,1] = ws[1] * im 
A[2,2] = ws[2] * im 
u_n = [1,1]
t_n = 0
dt = 5
s = 1

u_np1 = filon_timestep_static(A, u_n, ws, t_n, dt, s)
u_np1_true = [u_n[1] * cis(ws[1]*(t_n+dt)), u_n[2] * cis(ws[2]*(t_n+dt))]
println("2×2 independent case (should be exact)")
@show abs.(u_np1-u_np1_true)

# Part 3: 2×2, perturbed case
ws = [1,2]
A = zeros(ComplexF64, 2, 2)
A[1,1] = ws[1] * im 
A[2,2] = ws[2] * im 
A[1,2] = 0.01im
A[2,1] = 0.01im
u_n = [1,1]
t_n = 0
dt = 1
s = 1

u_np1 = filon_timestep_static(A, u_n, ws, t_n, dt, s)
u_np1_true = [u_n[1] * cis(ws[1]*(t_n+dt)), u_n[2] * cis(ws[2]*(t_n+dt))]
println("2×2 perturbed case (should not be exact)")
@show abs.(u_np1-u_np1_true)
