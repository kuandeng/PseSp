mutable struct FredOp{T<:FloatOrComplex} <: Op{T}
    fred::Matrix{T} # matrix representation of Fredholm convolution integral operator
    n::Int
    N::Int
    dom::Interval
    shift::T
    isSchur::Bool
    qrData::AbstractMatrix{T}
    qrDataShift::AbstractMatrix{T}
    qrDataFactor::Union{Factorization{T}, Nothing}
end

function FredOp(fred::AbstractMatrix{T}, N::Int, dom::Interval, shift::T, isSchur::Bool) where T
    if isSchur
        qrData = UpperTriangular(schur(fred).T)
    else 
        qrData = fred
    end
    qrDataShift = qrData - shift*I
    local qrDataFactor = nothing
    try 
        qrDataFactor = lu!(qrDataShift)
    catch
        qrDataFactor = nothing
    end
    FredOp{T}(fred, size(fred, 1), N, dom, shift, isSchur, qrData, qrDataShift, qrDataFactor)
end

# operator adjoint 
adjoint(op::FredOp{T}) where T = FredOp{T}(copy(op.fred'), op.n, op.N, op.dom, op.shift', op.isSchur, copy(op.qrData'), copy(op.qrDataShift'), nothing)


function adaptiveQrSolve!(op::FredOp{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    if length(y) > op.N
        error("please try larger preallocated dimension")
    end
    @views @inbounds begin         
        ny = length(y)
        nfred = op.n
        copyto!(work, y)
        if op.isSchur 
            temp = op.qrDataShift
        else
            temp = op.qrDataFactor
        end
        if ny < nfred
            work[ny+1:nfred] .= zero(T)
            ldiv!(temp, work[1:nfred])
        else
            ldiv!(temp, work[1:nfred])
            work[nfred+1:ny] ./= -op.shift
        end  
        return work[1:max(ny, nfred)]        
    end
end

# redefine "-"
function -(op::FredOp{T}, z::FloatOrComplex) where T    
    op.shift = z
    copyto!(op.qrDataShift, op.qrData)
    @inbounds @simd for i in 1:op.n
        op.qrDataShift[i, i] = op.qrDataShift[i, i] - z
    end
    if op.isSchur == false
        op.qrDataFactor = lu!(op.qrDataShift)
    end
    return op
end

function fredMatrix(coeffs_s::AbstractVecOrMat{T}, coeffs_t::AbstractVecOrMat{T}) where T 
    ns = size(coeffs_s, 1)
    nt = size(coeffs_t, 1)
    n = max(ns, nt)
    mat = zeros(T, n, n)
    mat[1:ns, 1:nt] = coeffs_s*transpose(coeffs_t)
    return mat
end
