module ControlledOperatorsFunctionWrappersExt

using ControlledOperators: ControlledOperators, FunctionControl
using FunctionWrappers: FunctionWrapper

# Opt-in, type-erased FunctionControl: the user's closure is hidden behind a FunctionWrapper so
# its concrete type does not leak into the control's type.  This method is more specific than
# the `erase_type(args...)` fallback in the main package, so it wins once this extension loads.
function ControlledOperators.erase_type(f, ::Type{T}; timetype::Type = Float64) where {T}
    fw = FunctionWrapper{T,Tuple{timetype,Int}}((t, n) -> f(t, n))
    return FunctionControl{T,typeof(fw)}(fw)
end

end # module
