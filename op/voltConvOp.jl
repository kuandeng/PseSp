
# defintion of Volterra integral operator, with preallocated matrix dimension n
mutable struct VoltConvOp{T<:FloatOrComplex} <: Op{T}
    volt::BandedMatrix{T} # matrix representation of Volterra integral operator
    N::Int # preallocated matrix dimension n
    dom::Interval
    side::Char
    shift::T
    qrData::BandedQrData{T}
end

function VoltConvOp(coeffs::AbstractVector{T}, N::Int, dom::Interval, side::Char, shift::T) where T
    volt = voltConvOpMatrix(coeffs, N, dom, side)
    qrData = BandedQrData(volt)
    return VoltConvOp(volt, N, dom, side, shift, qrData)
end


# solve resolvent equation using adaptive QR
function adaptiveQrSolve!(op::VoltConvOp{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
    nx = baqsv!(op.qrData, op.shift, y, tolSolve)
    work[1:nx] .= op.qrData.worky[1:nx]
    # println("n = ", length(y), " n_next = ", nx)
    return work[1:nx]        
    end
end

# redefine "-"
function -(op::VoltConvOp{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end


# construct the matrix representation of Volterra convolution integral operator
function subtraction(v::AbstractVector{T}, start::Int) where T <: FloatOrComplex
    n = length(v)
    u = v./(2start+1:2:2n+2start-1)
    b = zeros(T,n)
    for i = 1:n-2
        b[i] = u[i]-u[i+2]
    end
    b[end] = u[end]
    b[end-1] = u[end-1]
    return b
end


function voltConvOpMatrix(coeffs::AbstractVector{T}, n::Int,  dom::Interval, side::Char) where T <: FloatOrComplex
    bu = bl = length(coeffs)
    mat = BandedMatrix{T}(undef,(n+bl,n),(bu,bl))

    #Initial the 0th column
    mat[2:bu+1,1] = side == 'l' ? subtraction(coeffs, 0) : -subtraction(coeffs, 0)
    mat[1,1] = side == 'l' ? coeffs[1]-coeffs[2]/3 : coeffs[1]+coeffs[2]/3

    #Initial the 1th column
    mat[2:bu+2,2] =  subtraction((@view mat[1:bu+1,1]), 0) - 
        (side == 'l' ? [mat[2:bu+1,1]; 0] : -[mat[2:bu+1,1]; 0])

    #Recursion below the diagonal
    for i = 2:n-1
        mat[i+1:bu+i+1,i+1] = (2*i-1)*subtraction((@view mat[i:bu+i,i]), i-1) + [mat[i+1:bu+i-1,i-1];0;0]
    end

    #Reflection
    v = ones(T, bu)
    v[1:2:end] .= -1
    for i = 1:n-bu
        mat[i,i+1:i+bu] = (2*i-1)*v.*(@view mat[i+1:i+bu,i])./(2*(i:i+bu-1).+1)
    end
    for i = n-bu+1:n-1
        mat[i,i+1:n] = (2*i-1)*v[1:n-i].*(@view mat[i+1:n,i])./(2*(i:n-1).+1)
    end

    # normalized Legendre basis
    w_right = spdiagm(n, n, sqrt.(Vector{T}((2*(0:n-1).+1)/T(2))))
    w_left =  spdiagm(n+bl, n+bl, sqrt.(Vector{T}(T(2)./(2*(0:n+bl-1).+1))))
    mat = BandedMatrix{T}(w_left)*mat*BandedMatrix{T}(w_right)

    return (dom.right - dom.left)/2*mat
end


