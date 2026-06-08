using StaticArrays

# Units in GHz
@kwdef struct RabiOsc
    ωTrans::Float64 = 1 # Transition frequency
    ωDrive::Float64 = 1 # Drive frequency
    E::Float64 = 0.01 # Drive strength
end

I = SA_F64[1 0; 0 1]
σz = SA_F64[1 0; 0 -1]
σx = SA_F64[0 1; 1 0]
A(rabi, t) = -im*(rabi.ωTrans*σz + rabi.E*cos(rabi.ωDrive*t))


function trapezoidal_timestep1(rabi::RabiOsc, ψ_curr, t_curr, Δt)
    t_next = t_curr + Δt
    S_implicit = I - 0.5*Δt*A(rabi, t_next)
    S_explicit = I + 0.5*Δt*A(rabi, t_curr)
    return S_implicit \ (S_explicit*ψ_curr)
end

# Version 2, just a different way of writing
function trapezoidal_timestep2(rabi::RabiOsc, ψ_curr, t_curr, Δt)
    t_next = t_curr + Δt
    S_implicit = I + 0.25im*Δt*(rabi.ωTrans*σz + rabi.E*cos(rabi.ω*t_next))
    S_explicit = I - 0.25im*Δt*(rabi.ωTrans*σz - rabi.E*cos(rabi.ω*t_curr))
    return S_implicit \ (S_explicit*ψ_curr)
end

function filon_timestep(rabi::RabiOsc, ψ_curr, t_curr, Δt;
                        ω_filon = SA_F64[-rabi.ωTrans/2, rabi.ωTrans/2])


    # some stuff
end

