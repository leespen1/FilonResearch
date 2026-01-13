"""
Return 
    y = e^{i\\omega t}
    x = (1+t)e^{i\\omega t}
    z = (1+t+t^2/2)e^{i\\omega t}
    ? = (1+t+t^2/2 + t^3/6)e^{i\\omega t}
    ⋮

which is the solution to the linear system of ODEs
    [ẏ]   [iw  0  0 ⋯ ] [y]
    [ẋ] = [1  iw  0 ⋯ ] [x]           
    [ż]   [0  1  iw ⋯ ] [z]
"""
function poly_osc_solution(frequency, degree, t)
    @assert degree >= 0 "Degree must be non-negative"
    sol = zeros(ComplexF64, 1+degree)
    sol[1] = cis(frequency*t)
    for d in 1:degree
        sol[1+d] = sol[d] + ((t^d)/factorial(d))*sol[1]
    end
    return sol
end

"""
Construct the matrix used in the linear system of ODEs
    [ẏ]   [iw  0  0 ⋯ ] [y]
    [ẋ] = [1  iw  0 ⋯ ] [x]           
    [ż]   [0  1  iw ⋯ ] [z]

Which has solution
    y = e^{i\\omega t}
    x = (1+t)e^{i\\omega t}
    z = (1+t+t^2/2)e^{i\\omega t}
    ? = (1+t+t^2/2 + t^3/6)e^{i\\omega t}
"""
function poly_osc_ode_mat(frequency, degree)
    A = zeros(ComplexF64, 1+degree, 1+degree)
    for i in 1:degree
        A[i,i] = im*frequency
        A[i+1,i] = 1
    end
    A[1+degree,1+degree] = im*frequency
    return A
end

