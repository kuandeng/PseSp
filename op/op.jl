using ApproxFun
import Base: -, \, adjoint

struct Interval{T<:AbstractFloat}
    left :: T
    right :: T
    Interval{T}(left::T, right::T) where T = left < right ? new(left, right) : error("The interval must have left <right")
end

include("../adaptiveQr/adaptiveQr.jl")
include("./fredConvOp.jl")
include("./fredOp.jl")
include("./diffOp.jl")
include("./diffOp_2D.jl")
include("./diffOp_inf2D.jl")
include("./diffOp_block.jl")
include("./gepDiffOp.jl")
include("./voltConvOp.jl")
include("./voltOp.jl")