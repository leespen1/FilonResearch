using FilonResearch, Plots

spectral_radius(M) = maximum(abs, eigvals(M))

ω = -100.0
ϵ = 0.1 # Control strength
Δt_range = 1:1:1000

condition_numbers_s0 = Float64[]
condition_numbers_s1 = Float64[]

condition_numbers_s = Dict{Int, Vector{Float64}}()
spectral_radii_s = Dict{Int, Vector{Float64}}()
for s in 0:1
    condition_numbers_s[s] = Float64[]
    spectral_radii_s[s] = Float64[]
end

for Δt in Δt_range
    frequencies = [0, ω]

    A_n = Diagonal([0.0, im*ω]) + ϵ*[0 1; 1 0]
    dA_n = zeros(2,2)
    A_np1 = A_n
    dA_np1 = zeros(2,2)

    A_n_and_derivs = [A_n, dA_n]
    A_np1_and_derivs = [A_np1, dA_np1]

    for s in 0:1
        S_explicit, S_implicit = S_explicit_implicit_filon(
            A_n_and_derivs,
            A_np1_and_derivs,
            frequencies,
            Δt,
            s
        )
        push!(condition_numbers_s[s], cond(S_implicit))
        push!(spectral_radii_s[s], spectral_radius(inv(S_implicit)*S_explicit))

    end
end

pl_cond_s0 = plot(Δt_range, condition_numbers_s[0], scale=:log10, xlabel="Δt", ylabel="cond(S₊)", title="ω=$ω, ϵ=$ϵ, s=0")
pl_cond_s1 = plot(Δt_range, condition_numbers_s[1], scale=:log10, xlabel="Δt", ylabel="cond(S₊)", title="ω=$ω, ϵ=$ϵ, s=1")
pl_spec_s0 = plot(Δt_range, spectral_radii_s[0], xlabel="Δt", ylabel="|maxᵢ λᵢ(S₊⁻¹S₋)|", title="ω=($ω), s=0")
pl_spec_s1 = plot(Δt_range, spectral_radii_s[1], xlabel="Δt", ylabel="|maxᵢ λᵢ(S₊⁻¹S₋)|", title="ω=($ω), s=1")

pl_spec = plot(pl_spec_s0, pl_spec_s1, layout=(2,1))
pl_cond = plot(pl_cond_s0, pl_cond_s1, layout=(2,1))
