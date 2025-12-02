using FilonResearch, Test
using SpecialFunctions: expint
@testset "Polynomial f(x)" begin
end

@testset "f(x) = x/(1+x^2)" begin

    f(x) = x / (1+x^2)
    df(x) = (1 - x^2) / (1+x^2)^2
    df²(x) = (8*x^3/(1+x^2)^3) - (6*x/(1+x^2)^2)
    df³(x) = -6/(x^2+1)^2 + 48/(x^2+1)^3 - 48/(x^2+1)^4

    fa_derivs = [f(a), df(a), df²(a), df³(a)]
    fb_derivs = [f(b), df(b), df²(b), df³(b)]

    Ei(x) = -expint(-x)
    indefinite_integral(x, ω) = 0.5*(exp(ω)*Ei(im*ω*(x+im)) + exp(-ω)*Ei(im*x*ω + ω))
    true_solution(a, b, ω) =  indefinite_integral(b, ω) - indefinite_integral(a, ω)

    # @test explicit_filon_integral()
end
