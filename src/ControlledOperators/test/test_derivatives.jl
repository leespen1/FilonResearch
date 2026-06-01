@testset "time-derivative evaluate: reuse & stability" begin
    c0 = ConstantControl(1.0)
    f1 = FourierControl(0.2, [0.5], [0.3], 1.3)
    f2 = FourierControl(0.0, [0.1], [-0.2], 2.1)

    H0 = [1.0 0.0; 0.0 2.0]
    H1 = [0.0 1.0; 1.0 0.0]
    H2 = [0.0 -1.0; 1.0 0.0]
    gen = Generator((c0, f1, f2), [H0, H1, H2])
    t = 0.81

    for N in 0:3
        d = Derivative{N}()
        opN = evaluate(gen, t, d)
        # Structurally identical: the realized N-th derivative shares the SAME matrices.
        @test opN.matrices === gen.matrices
        @test all(opN.matrices[k] === gen.matrices[k] for k in 1:3)
        # Value equals Σ cₖᴺ(t) · Aₖ.
        ref = derivative(c0, t, d) * H0 + derivative(f1, t, d) * H1 + derivative(f2, t, d) * H2
        @test Matrix(opN) ≈ ref
    end

    @test @inferred(evaluate(gen, t, Derivative{2}())) isa Operator

    # Static layout: derivative evaluate is allocation-free.
    sgen = Generator((c0, f1, f2), (SMatrix{2,2}(H0), SMatrix{2,2}(H1), SMatrix{2,2}(H2)))
    @test @inferred(evaluate(sgen, t, Derivative{1}())) isa Operator
    @test count_allocs((g, tt) -> evaluate(g, tt, Derivative{1}()), sgen, t) == 0
end
