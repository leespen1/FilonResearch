using FilonResearch

T = 10.0

function f(nsteps)
    A = ComplexF64[1im 0;0 10im]
    u0 = [1,1]
    frequencies = [1,10]

    u_saves = filon_order2_size2(A, u0, frequencies, T, nsteps)
    return u_saves
end

function g(nsteps)
    uf = f(nsteps)[end]
    uf_true = [cis(1*T), cis(10*T)]
    error = abs.(uf - uf_true)
    return error
end



nsteps = 1000
u_saves = f(nsteps)
pl = plot([real(u[1]) for u in u_saves])
plot!(pl, [real(u[2]) for u in u_saves])

pl

