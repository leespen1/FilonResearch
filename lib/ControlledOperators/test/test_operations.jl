@testset "arithmetic, materialization, AbstractMatrix, Krylov" begin
    c0 = ConstantControl(1.0)
    f1 = FourierControl(0.0, [1.0], [0.0], 1.0)
    f2 = FourierControl(0.0, [0.0], [1.0], 2.0)
    H0 = [2.0 0.0; 0.0 -1.0]
    H1 = [0.0 1.0; 1.0 0.0]
    H2 = [0.3 0.1; 0.1 0.3]
    t = 0.4
    x = [1.0, -2.0]

    @testset "ControlledOperator +/-/scale" begin
        coA = ControlledOperator((c0, f1), (SMatrix{2,2}(H0), SMatrix{2,2}(H1)))
        coB = ControlledOperator((f2,), (SMatrix{2,2}(H2),))
        gsum = coA + coB
        gdiff = coA - coB
        @test length(gsum) == 3
        ref(t) = c0(t) * H0 + f1(t) * H1 + f2(t) * H2
        @test materialize(evaluate(gsum, t)) ≈ ref(t)
        @test materialize(evaluate(gdiff, t)) ≈ c0(t) * H0 + f1(t) * H1 - f2(t) * H2
        @test materialize(evaluate(2 * coA, t)) ≈ 2 .* (c0(t) * H0 + f1(t) * H1)
        # static composition stays type-stable
        @test @inferred(evaluate(gsum, t)) isa Operator
    end

    @testset "Operator +/-/scale" begin
        opA = evaluate(ControlledOperator((c0, f1), [H0, H1]), t)
        opB = evaluate(ControlledOperator((f2,), [H2]), t)
        @test Matrix(opA + opB) ≈ Matrix(opA) + Matrix(opB)
        @test Matrix(opA - opB) ≈ Matrix(opA) - Matrix(opB)
        @test Matrix(-opA) ≈ -Matrix(opA)
        @test Matrix(3 * opA) ≈ 3 .* Matrix(opA)
        @test (opA + opB) * x ≈ Matrix(opA + opB) * x
    end

    @testset "materialization: Matrix / sparse / collect / materialize!" begin
        op = evaluate(ControlledOperator((c0, f1, f2), [H0, H1, H2]), t)
        dense = c0(t) * H0 + f1(t) * H1 + f2(t) * H2
        @test Matrix(op) ≈ dense
        @test collect(op) ≈ dense
        @test Array(sparse(op)) ≈ dense
        @test issparse(sparse(op))
        out = zeros(2, 2)
        @test materialize!(out, op) === out
        @test out ≈ dense
        # sparse-backed operator materializes too
        sop = evaluate(ControlledOperator((c0, f1), [sparse(H0), sparse(H1)]), t)
        @test Array(sparse(sop)) ≈ c0(t) * H0 + f1(t) * H1
    end

    @testset "AbstractMatrix interface" begin
        op = evaluate(ControlledOperator((c0, f1, f2), [H0, H1, H2]), t)
        @test op isa AbstractMatrix{Float64}
        @test size(op) == (2, 2)
        @test size(op, 1) == 2
        @test eltype(op) === Float64
        dense = Matrix(op)
        @test all(op[i, j] ≈ dense[i, j] for i in 1:2, j in 1:2)
        # complex realized eltype via promotion (real matrices, complex coeffs)
        coc = ControlledOperator((FourierControl(0.0 + 0im, ComplexF64[1.0], ComplexF64[0.0], 1.0),), [H1])
        @test eltype(evaluate(coc, t)) === ComplexF64
    end

    @testset "Krylov smoke test" begin
        using Krylov
        n = 12
        R = collect(reshape(range(-1.0, 1.0; length = n^2), n, n))
        M = (R + R') / (4n)                     # small symmetric perturbation
        Id = Matrix{Float64}(I, n, n)
        # A(t) = 2·I + c(t)·M, symmetric positive definite for these coefficients.
        co = ControlledOperator((ConstantControl(2.0), ConstantControl(1.0)), [Id, M])
        op = evaluate(co, 0.0)
        A = Matrix(op)
        @test issymmetric(A)
        @test isposdef(A)
        xtrue = collect(range(1.0, 2.0; length = n))
        b = A * xtrue
        sol, stats = Krylov.cg(op, b)           # op used directly as the linear operator
        @test stats.solved
        @test norm(op * sol - b) < 1e-6
        @test sol ≈ xtrue rtol = 1e-6
    end

    @testset "Operator constructor validates its containers" begin
        using OffsetArrays
        M = [1.0 0.0; 0.0 1.0]
        # mismatched number of coefficients vs matrices
        @test_throws DimensionMismatch Operator([1.0, 2.0], [M])
        @test_throws DimensionMismatch Operator(SVector(1.0, 2.0), (M,))
        # non-1-based containers (the loops assume 1-based, @inbounds access)
        @test_throws ArgumentError Operator(OffsetVector([1.0, 2.0], 0:1), [M, M])
        @test_throws ArgumentError Operator([1.0, 2.0], OffsetVector([M, M], 0:1))
        # well-formed construction (and all the normal paths through it) still works
        @test Operator([1.0, 2.0], [M, M]) isa Operator
        @test evaluate(ControlledOperator((c0, f1), [M, M]), t) isa Operator
    end
end
