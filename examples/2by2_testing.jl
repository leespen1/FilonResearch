using FilonResearch, Plots

T = 10.0

function f(nsteps, order)
    A = ComplexF64[1im 0;0 10im]
    u0 = [1,1]
    frequencies = [1,10]

    if order == 2
        u_saves = filon_order2_size2(A, u0, frequencies, T, nsteps)
    elseif order == 4
        dA = zeros(2, 2)
        u_saves = filon_order4_size2(A, dA, u0, frequencies, T, nsteps)
    else
        throw("Invalid order $order. Please give order 2 or 4")
    end
    return u_saves
end

function g(nsteps, order)
    uf = f(nsteps, order)[end]
    uf_true = [cis(1*T), cis(10*T)]
    error = abs.(uf - uf_true)
    return error
end



order = 4
nsteps = 1
u_saves = f(nsteps, order)
@show order nsteps
println("Final time error = ", g(nsteps, order))

#pl = plot([real(u[1]) for u in u_saves])
#plot!(pl, [real(u[2]) for u in u_saves])
#pl

