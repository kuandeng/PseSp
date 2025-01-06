# Definition of a differential operator on an unbounded domain, with preallocated matrix dimension N.
mutable struct DiffOpInf2D{T<:FloatOrComplex} <: Op{T}
    diff::BandedMatrix{T}       # Matrix representation of the differential operator from the left basis to the right basis.
    N::Int                      # Preallocated matrix dimension N.
    shift::T                    # Shift value applied to the operator.
    qrData::BandedQrData{T}     # QR decomposition data for adaptive QR
end


function DiffOpInf2D(n::Int,  rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, shift::T) where {T<:FloatOrComplex, T1}
    L, bu, bl = diffOpInf2DMat(n, rank, coeffs_x, coeffs_y, T)
    L = BandedMatrix(L, (bl, bu))
    qrData = BandedQrData(L)
    return DiffOpInf2D(L, Int(n*(n+1)/2), shift, qrData)
end


# solve resolvent equation using adaptive QR
function adaptiveQrSolve!(op::DiffOpInf2D{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
    nx = baqsv!(op.qrData, op.shift, y, tolSolve)
    work[1:nx] .= op.qrData.worky[1:nx]
    # println("n = ", length(y), " n_next = ", nx)
    return work[1:nx]        
    end
end

# redefine "-"
function -(op::DiffOpInf2D{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end



function multMatInf(N::Int, coeffs::AbstractVector{T1}, T::Type{T1}) where {T1<:FloatOrComplex}
    m = size(coeffs, 1)
    if m == 1
        M = coeffs[1]*spdiagm(N, N, ones(T, N))
    else
        M = spzeros(T, 2*N, 2*N)
        mx = spdiagm(2*N, 2*N, 1=> T.(sqrt.((1:2*N-1)/2)), -1=>T.(sqrt.((1:2*N-1)/2)))
        for i = 1:m
            M = M + coeffs[i]*mx^(i-1)
        end
        M = M[1:N, 1:N]
    end
    return M 
end

function diffMatInf(N::Int, T::Type{T1}) where {T1<:FloatOrComplex}
    return spdiagm(N, N, 1=> T.(sqrt.((1:N-1)/2)), -1=>T.(-sqrt.((1:N-1)/2)))
end

function bandWidthInf2D(coeffs::Tuple{Vararg{Vector{T}}}) where T
    b = typemin(Int)
    for i = 0:size(coeffs,1)-1
        if (coeffs[i+1] != zeros(T, 1))
            b = max(i+length(coeffs[i+1])-1, b)
        end
    end
    return b
end


function bandWidthInf2D(n::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}) where T1
    b_x = [bandWidthInf2D(coeffs_x[i])[1] for i in 1:rank]
    b_y = [bandWidthInf2D(coeffs_y[i])[1] for i in 1:rank]
    b_max = max(maximum(b_x), maximum(b_y))

    b_total = b_x + b_y
    b_tmax = maximum(b_total)
    idx_max = findall(x -> x == b_tmax, b_total)
    b_temp = minimum(b_y[idx_max])
    bl_reOrdered = Int((2*n+1+b_tmax)*b_tmax/2) - b_temp
    bu_reOrdered = Int((2*n-b_tmax+1)*b_tmax/2) - b_temp
    return b_max, b_tmax, bu_reOrdered, bl_reOrdered

end

function hermiteMat(N::Int, coeffs::Tuple{Vararg{Vector{T1}}}, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    M_hermite = spzeros(T, N, N)
    for i = 0:size(coeffs,1) - 1
        M_hermite = M_hermite + multMatInf(N, coeffs[i+1], T)*diffMatInf(N, T)^i
    end
    return M_hermite
end

function hermite2DMat(n::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    M_hermite2D = spzeros(T, n^2, n^2)
    for i = 1:rank
        M_hermite2D += kron(hermiteMat(n, coeffs_x[i], T), hermiteMat(n, coeffs_y[i], T))
    end
    return M_hermite2D
end


function diffOpInf2DMat(n::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, 
    coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    b_max, b_tmax, bu_reOrdered, bl_reOrdered = bandWidthInf2D(n, rank, coeffs_x, coeffs_y)

    tempn = n+5*(b_max)
    M_hermite2D = hermite2DMat(tempn, rank, coeffs_x, coeffs_y, T)

    index_col = reOrder(n, tempn)
    index_row = reOrder(n+b_tmax, tempn)
    L = M_hermite2D[index_row, index_col]
    return L, bu_reOrdered, bl_reOrdered
end

