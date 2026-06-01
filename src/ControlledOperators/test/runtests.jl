using ControlledOperators
using StaticArrays
using LinearAlgebra
using SparseArrays
using Test

# --- Optional analysis tools (gated so the suite still runs without them). -------------------
const JET_OK = try
    @eval using JET
    true
catch err
    @warn "JET unavailable — skipping static-dispatch checks" err
    false
end

const ALLOCCHECK_OK = try
    @eval using AllocCheck
    true
catch err
    @warn "AllocCheck unavailable — relying on @allocated" err
    false
end

# --- Helpers shared by all test files. ------------------------------------------------------

# Function-barrier allocation count: warm up once (force compilation/specialization), then
# measure.  `f` and `args` are passed by value so the measured call is fully specialized and
# free of boxing from non-const globals.
@inline function count_allocs(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

# Named call wrappers (avoid measuring closure construction).
mul3!(y, op, x) = (mul!(y, op, x); nothing)
mul5!(y, op, x, α, β) = (mul!(y, op, x, α, β); nothing)
apply(op, x) = op * x
do_evaluate(gen, t) = evaluate(gen, t)
do_evaluate!(op, gen, t) = (evaluate!(op, gen, t); nothing)

# Central finite differences for orders 1..3 (independent check of analytic derivatives).
function fd_deriv(f, t, ::Val{1}; h = 1e-3)
    (f(t + h) - f(t - h)) / (2h)
end
function fd_deriv(f, t, ::Val{2}; h = 1e-3)
    (f(t + h) - 2f(t) + f(t - h)) / h^2
end
function fd_deriv(f, t, ::Val{3}; h = 1e-2)
    (f(t + 2h) - 2f(t + h) + 2f(t - h) - f(t - 2h)) / (2h^3)
end

@testset "ControlledOperators.jl" begin
    include("test_controls.jl")
    include("test_dynamic.jl")
    include("test_static.jl")
    include("test_derivatives.jl")
    include("test_operations.jl")
    include("test_extensions.jl")
    if JET_OK
        include("test_jet.jl")
    else
        @test_skip "JET static-dispatch checks skipped (JET unavailable)"
    end
end
