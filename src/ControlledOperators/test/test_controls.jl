@testset "controls: values, derivatives, inference" begin

    @testset "ConstantControl" begin
        c = ConstantControl(2.5)
        @test c(0.7) == 2.5
        @test derivative(c, 0.7, Derivative{0}()) == 2.5
        @test derivative(c, 0.7, Derivative{1}()) == 0.0
        @test derivative(c, 0.7, Derivative{3}()) == 0.0
        @test derivative(c, 0.7, DerivativeUpTo{2}()) == SVector(2.5, 0.0, 0.0)
        @test eltype(c) === Float64
        @test @inferred(derivative(c, 0.7, Derivative{0}())) == 2.5
        @test @inferred(derivative(c, 0.7, DerivativeUpTo{3}())) isa SVector{4,Float64}
    end

    @testset "FourierControl vs finite differences" begin
        a0, a, b, ω = 0.3, [0.5, 0.1], [0.2, -0.4], 1.7
        f = FourierControl(a0, a, b, ω)
        cref(t) = a0 + a[1] * cos(ω * t) + b[1] * sin(ω * t) +
                  a[2] * cos(2ω * t) + b[2] * sin(2ω * t)
        for t in (0.0, 0.3, 1.1, 2.7)
            @test f(t) ≈ cref(t)
            @test derivative(f, t, Derivative{1}()) ≈ fd_deriv(f, t, Val(1)) rtol = 1e-4
            @test derivative(f, t, Derivative{2}()) ≈ fd_deriv(f, t, Val(2)) rtol = 1e-4
            @test derivative(f, t, Derivative{3}()) ≈ fd_deriv(f, t, Val(3)) rtol = 1e-3
            # DerivativeUpTo equals the stacked individual derivatives.
            up = derivative(f, t, DerivativeUpTo{3}())
            @test up == SVector(ntuple(m -> derivative(f, t, Derivative{m - 1}()), Val(4)))
        end
        @test @inferred(derivative(f, 0.3, Derivative{2}())) isa Float64
        @test @inferred(derivative(f, 0.3, DerivativeUpTo{4}())) isa SVector{5,Float64}

        # complex-valued coefficients
        fc = FourierControl(0.0 + 0im, ComplexF64[1 + 1im], ComplexF64[2 - 1im], 1.0)
        @test eltype(fc) === ComplexF64
        @test @inferred(fc(0.4)) isa ComplexF64
        @test @inferred(derivative(fc, 0.4, DerivativeUpTo{2}())) isa SVector{3,ComplexF64}

        # pure-constant Fourier (empty coefficient vectors)
        f0 = FourierControl(1.5, Float64[], Float64[], 1.0)
        @test f0(3.0) == 1.5
        @test derivative(f0, 3.0, Derivative{1}()) == 0.0
    end

    @testset "FunctionControl" begin
        # f(t, n) = n-th derivative of sin(t) = sin(t + nπ/2)
        g = FunctionControl{Float64}((t, n) -> sin(t + n * (π / 2)))
        @test g(0.6) ≈ sin(0.6)
        @test derivative(g, 0.6, Derivative{1}()) ≈ cos(0.6)
        @test derivative(g, 0.6, Derivative{2}()) ≈ -sin(0.6)
        @test derivative(g, 0.6, DerivativeUpTo{2}()) ≈ SVector(sin(0.6), cos(0.6), -sin(0.6))
        @test @inferred(derivative(g, 0.6, Derivative{1}())) isa Float64
        @test @inferred(derivative(g, 0.6, DerivativeUpTo{2}())) isa SVector{3,Float64}
    end

    @testset "ScaledControl" begin
        f = FourierControl(0.3, [0.5], [0.2], 1.7)
        sc = ScaledControl(-2.0, f)
        @test eltype(sc) === Float64
        @test sc(0.3) ≈ -2.0 * f(0.3)
        @test derivative(sc, 0.3, Derivative{1}()) ≈ -2.0 * derivative(f, 0.3, Derivative{1}())
        @test derivative(sc, 0.3, DerivativeUpTo{2}()) ≈ -2.0 .* derivative(f, 0.3, DerivativeUpTo{2}())
        @test @inferred(derivative(sc, 0.3, DerivativeUpTo{2}())) isa SVector{3,Float64}
        # promotion: real factor × real control stays Float64; integer factor promotes
        @test eltype(ScaledControl(-1, f)) === Float64
    end

    @testset "t is restricted to Real" begin
        f = FourierControl(0.3, [0.5], [0.2], 1.7)
        gen = Generator((ConstantControl(1.0), f),
                        (SMatrix{2,2}(1.0, 0, 0, 1.0), SMatrix{2,2}(0.0, 1, 1, 0.0)))
        # Non-real time is rejected where there is no generic fallback: the call operator and
        # `evaluate`.  (With the TaylorDiff extension loaded, `derivative(c, t, order)` itself
        # has an untyped-`t` fallback, so it would instead route a non-real `t` to Taylor mode.)
        @test_throws MethodError f(1.0im)
        @test_throws MethodError f("noon")
        @test_throws MethodError evaluate(gen, 1.0im)
        # Real subtypes (Int, Rational, Float32) are accepted and agree with Float64.
        @test f(1) ≈ f(1.0)
        @test f(1 // 2) ≈ f(0.5)
        @test f(0.3f0) ≈ f(0.3) rtol = 1e-5
    end
end
