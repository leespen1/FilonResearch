# Loaded only when JET is available (see runtests.jl).  `@test_opt` asserts the call optimizes
# to fully concrete code — i.e. no runtime dispatch — restricted to our own module so we do not
# flag unrelated quirks in Base/LinearAlgebra.
@testset "JET: no dynamic dispatch on hot paths" begin
    c0 = ConstantControl(1.0)
    f1 = FourierControl(0.0, [1.0], [0.0], 1.0)
    f2 = FourierControl(0.0, [0.0], [1.0], 2.0)

    H0 = [2.0 0.0; 0.0 -1.0]
    H1 = [0.0 1.0; 1.0 0.0]
    H2 = [0.5 0.0; 0.0 0.5]

    # dynamic
    co = ControlledOperator((c0, f1, f2), [H0, H1, H2])
    op = evaluate(co, 0.3)
    rop = Operator(co, 0.0)
    x = [1.0, -2.0]
    y = similar(x)

    JET.@test_opt target_modules = (ControlledOperators,) evaluate(co, 0.3)
    JET.@test_opt target_modules = (ControlledOperators,) mul!(y, op, x)
    JET.@test_opt target_modules = (ControlledOperators,) mul!(y, op, x, 2.0, 3.0)
    JET.@test_opt target_modules = (ControlledOperators,) evaluate!(rop, co, 0.3)

    # static
    co_static = ControlledOperator((c0, f1, f2), (SMatrix{2,2}(H0), SMatrix{2,2}(H1), SMatrix{2,2}(H2)))
    sop = evaluate(co_static, 0.3)
    xs = SVector(1.0, -2.0)
    ys = MVector(0.0, 0.0)

    JET.@test_opt target_modules = (ControlledOperators,) evaluate(co_static, 0.3)
    JET.@test_opt target_modules = (ControlledOperators,) sop * xs
    JET.@test_opt target_modules = (ControlledOperators,) mul!(ys, sop, xs, 2.0, 3.0)

    # derivative-controls realize the same way
    JET.@test_opt target_modules = (ControlledOperators,) evaluate(co, 0.3, Derivative{2}())
end
