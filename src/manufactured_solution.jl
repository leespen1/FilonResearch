"""
Return 
    y = e^{i\\omega t}
    x = (1+t)e^{i\\omega t}
    z = (1+t+t^2)e^{i\\omega t}
    ⋮

which is the solution to the linear system of ODEs
    [ẏ]   [iw  1  0 ⋯ ] [y]
    [ẋ] = [0  iw  1 ⋯ ] [x]           
    [ż]   [0  0  iw ⋯ ] [z]
"""
function poly_osc_solution(frequency, degree, t)
    @assert degree >= 0 "Degree must be non-negative"
    sol = zeros(ComplexF64, 1+degree)
    sol[1] = cis(frequency*t)
    for d in 1:degree
        sol[1+d] = t*sol[d] + sol[1]
    end
    return sol
end

"""
Construct the matrix used in the linear system of ODEs
    [ẏ]   [iw  1  0 ⋯ ] [y]
    [ẋ] = [0  iw  1 ⋯ ] [x]           
    [ż]   [0  0  iw ⋯ ] [z]

Which has solution
    y = e^{i\\omega t}
    x = (1+t)e^{i\\omega t}
    z = (1+t+t^2)e^{i\\omega t}
"""
function poly_osc_ode_mat(frequency, degree, t)
    A = zeros(ComplexF64, 1+degree, 1+degree)
    for i in 1:degree
        A[i,i] = im*frequency
        A[i,i+1] = 1
    end
    A[1+degree,1+degree] = im*frequency
    return A
end

