
using DrWatson
@quickactivate "FilonExperiments"
using FilonResearch, CairoMakie, LaTeXStrings
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

const inch = 96
const pt = 4/3

function envelope(t::Real)
    return t / (1+t^2)
end

function carrier_wave(t::Real, ω::Real)
    return cis(t*ω)
end

function full_function(t::Real, ω::Real)
    return envelope(t)*carrier_wave(t,ω)
end

function make_plotpath(config, extension)
    filename = savename("EnvelopeCarrierWave", config, extension, sort=false)
    dir = plotsdir("scalar_explicit_filon")
    mkpath(dir)
    return joinpath(dir, filename)
end


tStart = -1
tEnd = 1
ω = 10
height=3
width=3.5

config = (; ω, tStart, tEnd, height, width)

n_samples = 1001


t_samples = LinRange(tStart, tEnd, n_samples)
fig = Figure(size=(height*inch, width*inch))
ax = Axis(fig[1,1], xlabel=L"t", title=L"f(x) = \frac{x}{1+x^2}")
lines!(t_samples, envelope.(t_samples), color=:black, linestyle=:dash, linewidth=1)
lines!(t_samples, -1 .* envelope.(t_samples), color=:black, linestyle=:dash, linewidth=1)
lines!(t_samples, real.(full_function.(t_samples, ω)))

png_path = make_plotpath(config, "png")
svg_path = make_plotpath(config, "svg")
pdf_path = make_plotpath(config, "pdf")

save(png_path, fig)
save(svg_path, fig)
save(pdf_path, fig)
