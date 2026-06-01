@testset "dynamic path: Vector matrices, inference, zero-alloc" begin
    c0 = ConstantControl(1.0)
    f1 = FourierControl(0.0, [1.0], [0.0], 1.0)   # cos(t)
    f2 = FourierControl(0.0, [0.0], [1.0], 2.0)   # sin(2t)

    H0 = [2.0 0.0; 0.0 -1.0]
    H1 = [0.0 1.0; 1.0 0.0]
    H2 = [0.5 0.0; 0.0 0.5]

    # Tuple of (heterogeneous) controls + Vector of dense matrices → SVector coeffs.
    gen = Generator((c0, f1, f2), [H0, H1, H2])
    t = 0.37
    op = evaluate(gen, t)
    @test op isa Operator
    @test op.coeffs isa SVector{3,Float64}
    @test op.matrices === gen.matrices              # matrices shared, not copied

    x = [1.0, -2.0]
    y = similar(x)

    # Correctness against an explicit reference.
    Aref = c0(t) * H0 + f1(t) * H1 + f2(t) * H2
    mul!(y, op, x)
    @test y ≈ Aref * x
    @test op * x ≈ Aref * x
    @test Matrix(op) ≈ Aref
    @test materialize(op) ≈ Aref

    # 5-arg mul!: y ← β y + α A x
    fill!(y, 7.0)
    mul!(y, op, x, 2.0, 3.0)
    @test y ≈ 3.0 .* fill(7.0, 2) .+ 2.0 .* (Aref * x)

    # Inference.
    @test @inferred(evaluate(gen, t)) isa Operator
    @test (@inferred mul!(y, op, x)) === y
    @test (@inferred mul!(y, op, x, 1.0, 0.0)) === y

    # Zero allocation on the in-place paths (after warm-up).
    @test count_allocs(mul3!, y, op, x) == 0
    @test count_allocs(mul5!, y, op, x, 2.0, 3.0) == 0

    # Reusable, mutable operator (Vector coeffs) refreshed in place with zero allocation —
    # even though the controls are a heterogeneous tuple.
    rop = Operator(gen, 0.0)
    @test rop.coeffs isa Vector{Float64}
    evaluate!(rop, gen, t)
    @test Matrix(rop) ≈ Aref
    @test (@inferred evaluate!(rop, gen, t)) === rop
    @test count_allocs(do_evaluate!, rop, gen, t) == 0

    # Homogeneous Vector-of-controls path (coeffs are a Vector) → evaluate! in place.
    genv = Generator([f1, f2], [H1, H2])
    opv = evaluate(genv, t)
    @test opv.coeffs isa Vector{Float64}
    @test Matrix(opv) ≈ f1(t) * H1 + f2(t) * H2
    @test count_allocs(do_evaluate!, opv, genv, 0.9) == 0  # mutates opv to t = 0.9
    @test Matrix(opv) ≈ f1(0.9) * H1 + f2(0.9) * H2

    # Sparse matrices: mul! stays non-allocating.
    S1 = sparse(H1)
    S2 = sparse(H2)
    gens = Generator((f1, f2), [S1, S2])
    ops = evaluate(gens, t)
    ys = zeros(2)
    mul!(ys, ops, x)
    @test ys ≈ (f1(t) * H1 + f2(t) * H2) * x
    @test count_allocs(mul3!, ys, ops, x) == 0

    if ALLOCCHECK_OK
        @testset "AllocCheck (dynamic)" begin
            @test isempty(AllocCheck.check_allocs(
                mul!, (typeof(y), typeof(op), typeof(x), Float64, Float64); ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(
                evaluate!, (typeof(rop), typeof(gen), Float64); ignore_throw = true))
        end
    end
end
