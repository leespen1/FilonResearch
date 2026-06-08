@testset "extensions: FunctionWrappers & TaylorDiff" begin

    @testset "FunctionWrappers: erase_type" begin
        using FunctionWrappers: FunctionWrapper
        raw = FunctionControl{Float64}((t, n) -> sin(t + n * (π / 2)))
        er = erase_type((t, n) -> sin(t + n * (π / 2)), Float64)
        @test er isa FunctionControl{Float64}
        # The wrapped callback type is erased to a concrete FunctionWrapper, so the closure
        # type no longer appears in the control's type.
        F = typeof(er).parameters[2]
        @test F <: FunctionWrapper
        @test er(0.5) ≈ sin(0.5)
        @test derivative(er, 0.5, Derivative{1}()) ≈ cos(0.5)
        @test derivative(er, 0.5, DerivativeUpTo{2}()) ≈ SVector(sin(0.5), cos(0.5), -sin(0.5))
        @test @inferred(derivative(er, 0.5, Derivative{1}())) isa Float64
        # erased controls of two different closures share the same concrete type
        er2 = erase_type((t, n) -> cos(t + n * (π / 2)), Float64)
        @test typeof(er) === typeof(er2)
    end

    @testset "TaylorDiff: derivative fallback for Derivative{0}-only controls" begin
        import TaylorDiff  # load to activate the extension, without importing its `derivative`
        # A control that defines ONLY the 0-th derivative; higher orders come from Taylor mode.
        struct ExpRamp <: AbstractControl{Float64}
            α::Float64
        end
        ControlledOperators.derivative(c::ExpRamp, t, ::Derivative{0}) = exp(c.α * t)

        c = ExpRamp(0.7)
        t = 0.4
        # analytic: dⁿ/dtⁿ exp(αt) = αⁿ exp(αt)
        for N in 1:4
            @test derivative(c, t, Derivative{N}()) ≈ c.α^N * exp(c.α * t)
        end
        up = derivative(c, t, DerivativeUpTo{4}())
        @test up ≈ SVector(ntuple(m -> c.α^(m - 1) * exp(c.α * t), Val(5)))
        @test @inferred(derivative(c, t, DerivativeUpTo{3}())) isa SVector{4,Float64}

        # The fallback does not shadow concrete controls' own analytic derivatives.
        f = FourierControl(0.0, [1.0], [0.0], 1.0)
        @test derivative(f, 0.3, Derivative{1}()) ≈ -sin(0.3)
    end
end
