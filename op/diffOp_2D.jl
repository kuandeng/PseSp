# Definition of a 2D differential operator with preallocated matrix dimension N.
mutable struct DiffOp2D{T<:FloatOrComplex} <: Op{T}
    diff2D::BandedMatrix{T}  # Matrix representation of the 2D differential operator from the left basis to the right basis.
    N::Int                   # Preallocated matrix dimension N.
    domx::Interval           # Domain of the operator in the x-direction.
    domy::Interval           # Domain of the operator in the y-direction.
    diff2D_shift::BandedMatrix{T}  # Matrix representation of the identity operator from the left basis to the right basis (used for shifting).
    basisTransL::BandedMatrix{T}   # Transformation matrix: converts coefficients of the left basis to coefficients in the normalized Legendre basis.
    basisTransR::BandedMatrix{T}   # Transformation matrix: converts coefficients in the normalized Legendre basis to the coefficients of the right basis.
    shift::T                  # Shift value applied to the operator.
    qrData::GenBandedQrData{T} # QR decomposition data for adaptive QR
end


function DiffOp2D(n::Int, K::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, bcType::String, bcOrder::Int, dom_x::Interval, dom_y::Interval, shift::T) where {T<:FloatOrComplex, T1}
    L, R, L_shift, M_bRC, bu, bl = diffOp2DMat(n, K, rank, coeffs_x, coeffs_y, bcType, bcOrder, dom_x, dom_y, T)
    L = BandedMatrix(L, (bl, bu))
    R = BandedMatrix(R) 
    L_shift = BandedMatrix(L_shift)
    M_bRC = BandedMatrix(M_bRC)
    qrData = GenBandedQrData(L, L_shift)
    return DiffOp2D(L, Int(n*(n+1)/2), dom_x, dom_y, L_shift, M_bRC, R, shift, qrData)
end

function adaptiveQrSolve!(op::DiffOp2D{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
        ny = length(y)
        TR = op.basisTransR[1:ny, 1:ny]
        mul!(work[1:ny], TR, y)
        nx = gbaqsv!(op.qrData, op.shift, work[1:ny], tolSolve)
        nx_new = nx + op.basisTransL.l
        TL = op.basisTransL[1:nx_new, 1:nx]
        mul!(work[1:nx_new], TL, op.qrData.worky[1:nx])
        # println("n = ", ny, " n_next = ", nx_new)
        return work[1:nx_new]
    end
end


# redefine "-"
function -(op::DiffOp2D{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end


function ultras2DMat(n::Int, K::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, scale_x::T3, scale_y::T3, T::Type{T2}) where {T1, T2<:FloatOrComplex, T3}
    M_ultras2D = spzeros(T, n^2, n^2)
    for i = 1:rank 
        M_ultras2D += kron(ultrasMat(n, K, scale_x, coeffs_x[i], T), ultrasMat(n, K, scale_y, coeffs_y[i], T))
    end
    return M_ultras2D
end

# reOrder the tensor product basis. 
function reOrder(n::Int, ldn::Int)
    size_ind = Int((n^2+n)/2)
    index = zeros(Int, size_ind)
    count = one(Int)
    for i = 0:n-1
        for p = 0:i
            index[count] = p*ldn + (i-p) + 1
            count += 1
        end            
    end
    return index
end

function bandWidth2D(n::Int, K::Int, rank::Int, coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, bu_bRC::Int, bl_bRC::Int) where T1
    # band for split ODO matrix
    bu_x = [bandWidth_shift(K, coeffs_x[i])[1] for i in 1:rank]
    bl_x = [bandWidth_shift(K, coeffs_x[i])[2] for i in 1:rank]
    bu_y = [bandWidth_shift(K, coeffs_y[i])[1] for i in 1:rank]
    bl_y = [bandWidth_shift(K, coeffs_y[i])[2] for i in 1:rank]
    bu_max = max(maximum(bu_x), maximum(bu_y))
    bl_max = max(maximum(bl_x), maximum(bl_y))


    # maximum band for reordered PDO matrix
    bu_total = bu_x + bu_y .+ 2*bu_bRC
    bl_total = bl_x + bl_y .+ 2*bl_bRC

    bu_tmax = maximum(bu_total)
    idx_max = findall(x -> x == bu_tmax, bu_total)
    bu_temp = minimum(bu_y[idx_max]) + bu_bRC
    bu_reOrdered = Int((2*n-bu_tmax+1)*bu_tmax/2) - bu_temp


    bl_tmax = maximum(bl_total)
    idx_max = findall(x -> x == bl_tmax, bl_total)
    bl_temp = minimum(bl_y[idx_max]) + bl_bRC
    bl_reOrdered = Int((2*n+1+bl_tmax)*bl_tmax/2) - bl_temp

    return bu_max, bl_max, bu_tmax, bl_tmax, bu_reOrdered, bl_reOrdered
end



function diffOp2DMat(n::Int, K::Int, rank::Int, 
    coeffs_x::Vector{<:Tuple{Vararg{Vector{T1}}}}, 
    coeffs_y::Vector{<:Tuple{Vararg{Vector{T1}}}}, bcType::String, bcOrder::Int, dom_x::Interval, dom_y::Interval, T::Type{T2}) where {T1, T2<:FloatOrComplex}

    ~, bu_bRC, bl_bRC = basisReCombMat(bcType, bcOrder, 10, T)

    bu_max, bl_max, bu_tmax, bl_tmax, bu_reOrdered, bl_reOrdered = bandWidth2D(n, K, rank, coeffs_x, coeffs_y, bu_bRC, bl_bRC)


    tempn = n+5*(bl_max + bl_bRC)+bl_bRC


    scale_x = 2/(dom_x.right - dom_x.left)
    scale_y = 2/(dom_y.right - dom_y.left)
    M_ultras2D = ultras2DMat(tempn, K, rank, coeffs_x, coeffs_y, scale_x, scale_y, T)

    M_bRC, ~ = basisReCombMat(bcType, bcOrder, tempn-bl_bRC, T)
    M_bRC = kron(M_bRC, M_bRC)


    # construct L
    L = M_ultras2D*M_bRC
    index_col = reOrder(n, tempn-bl_bRC)
    index_row = reOrder(n+bl_tmax, tempn)
    L = L[index_row, index_col]

    # construct shift L
    S = convertMat(tempn, 0, K, T)
    L_shift = kron(S, S)*M_bRC
    L_shift = L_shift[index_row, index_col]

    # construct R
    S = convertMat(n, 0, K, T)
    w = spdiagm(n, n, sqrt.(Vector{T}((2*(0:n-1).+1)/2)))
    R = kron(S*w, S*w)
    index_R = reOrder(n, n)
    R = R[index_R, index_R]

    # construct M_bRC
    M_bRC, ~ = basisReCombMat(bcType, bcOrder, n, T)
    w = spdiagm(n+bl_bRC, n+bl_bRC, sqrt.(Vector{T}(2 ./(2*(0:n-1+bl_bRC).+1))))
    M_bRC = kron(w*M_bRC, w*M_bRC)
    M_bRC = M_bRC[reOrder(n+bl_bRC, n+bl_bRC), reOrder(n, n)]


    return L, R, L_shift, M_bRC, bu_reOrdered, bl_reOrdered

end














































































































































































































































































































































































































































































































































































































































