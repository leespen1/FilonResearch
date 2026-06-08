@testset "static path: SMatrix/SVector, unrolled, zero-alloc" begin
    c0 = ConstantControl(1.0)
    f1 = FourierControl(0.0, [1.0], [0.0], 1.0)
    f2 = FourierControl(0.0, [0.0], [1.0], 2.0)

    S0 = SMatrix{2,2}(2.0, 0.0, 0.0, -1.0)
    S1 = SMatrix{2,2}(0.0, 1.0, 1.0, 0.0)
    S2 = SMatrix{2,2}(0.5, 0.0, 0.0, 0.5)

    co = ControlledOperator((c0, f1, f2), (S0, S1, S2))
    t = 0.37
    op = evaluate(co, t)
    @test op.coeffs isa SVector{3,Float64}
    @test op.matrices isa Tuple

    xs = SVector(1.0, -2.0)
    Aref = c0(t) * S0 + f1(t) * S1 + f2(t) * S2

    # `*` returns an SVector with no heap allocation.
    @test op * xs ≈ Aref * xs
    @test @inferred(op * xs) isa SVector{2,Float64}
    @test count_allocs(apply, op, xs) == 0

    # Unrolled 5-arg mul! into a mutable static vector, non-allocating.
    ys = MVector(0.0, 0.0)
    mul!(ys, op, xs)
    @test ys ≈ Aref * xs
    @test (@inferred mul!(ys, op, xs, 1.0, 0.0)) === ys
    @test count_allocs(mul5!, ys, op, xs, 2.0, 3.0) == 0
    @test count_allocs(mul3!, ys, op, xs) == 0

    # Fully-static evaluate allocates nothing (the Operator is isbits).
    @test @inferred(evaluate(co, t)) isa Operator
    @test count_allocs(do_evaluate, co, t) == 0

    # materialize returns an SMatrix on the static path.
    @test @inferred(materialize(op)) isa SMatrix{2,2,Float64}
    @test materialize(op) ≈ Aref

    if ALLOCCHECK_OK
        @testset "AllocCheck (static)" begin
            @test isempty(AllocCheck.check_allocs(
                *, (typeof(op), typeof(xs)); ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(
                mul!, (typeof(ys), typeof(op), typeof(xs), Float64, Float64); ignore_throw = true))
        end
    end
end
