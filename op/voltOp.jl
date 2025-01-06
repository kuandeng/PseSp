# defintion of Volterra integral operator, with preallocated matrix dimension n
mutable struct VoltOp{T<:FloatOrComplex} <: Op{T}
    volt::BandedMatrix{T} # matrix representation of Volterra integral operator
    N::Int # preallocated matrix dimension n
    rank::Int
    dom::Interval
    side::Char
    shift::T
    qrData::BandedQrData{T}
end

# default constructor
function VoltOp(coeffs_x::Tuple{Vararg{Vector{T}}}, coeffs_y::Tuple{Vararg{Vector{T}}}, rank::Int, N::Int, dom::Interval, side::Char, shift::T) where T
    volt = voltOpMatrix(N, coeffs_x, coeffs_y, dom, rank, side)
    qrData = BandedQrData(volt)
    return VoltOp(volt, N, rank, dom, side, shift, qrData)
end

# operator adjoint 
adjoint(op::VoltOp{T}) where T = VoltOp{T}(BandedMatrix(op.volt'), op.N, op.rank, op.dom, op.side == 'l' ? 'r' : 'l', op.shift', BandedQrData(BandedMatrix(op.volt')))

# solve resolvent equation using adaptive QR
function adaptiveQrSolve!(op::VoltOp{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
    nx = baqsv!(op.qrData, op.shift, y, tolSolve)
    work[1:nx] .= op.qrData.worky[1:nx]
    # println("n = ", length(y), " n_next = ", nx)
    return work[1:nx]        
    end
end

# redefine "-"
function -(op::VoltOp{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end


function inteMat(N::Int, dom::Interval, side::Char, T::Type{T1}) where T1
    if side == 'l'
        M = spdiagm(N, N, 1 => -one(T)./(2 .*(1:N-1) .+ 1), -1 => one(T) ./(2 .*(0:N-2) .+ 1))
        M[1,1] = one(T)
        return (dom.right - dom.left)/2*M
    end
end


function voltOpMatrix(N::Int, coeffs_x::Tuple{Vararg{Vector{T}}}, coeffs_y::Tuple{Vararg{Vector{T}}}, dom:: Interval, rank::Int, side::Char) where T
    
    bl = 0
    for r = 1:rank
        bl_x_r = length(coeffs_x[r]) - 1
        bl_y_r = length(coeffs_y[r]) - 1
        bl = max(bl_x_r + bl_y_r + 1, bl)
    end
    tempN = N + 5*bl
    M_volt = spzeros(T, tempN, tempN)
    if side == 'l'
        for r = 1:rank 
            M_volt = M_volt + multMat(tempN, 0, coeffs_x[r], T)*inteMat(tempN, dom, 'l', T)*multMat(tempN, 0, coeffs_y[r], T)
        end
    end
    M_volt = M_volt[1:N+bl, 1:N]
    w_right = spdiagm(N, N, sqrt.(Vector{T}((2*(0:N-1).+1)/T(2))))
    w_left =  spdiagm(N+bl, N+bl, sqrt.(Vector{T}(T(2)./(2*(0:N+bl-1).+1))))
    M_volt = w_left*M_volt*w_right

    return BandedMatrix(M_volt)
end