# Definition of a block differential operator with preallocated matrix dimension N.
mutable struct DiffOpBlock{T<:FloatOrComplex} <: Op{T}
    diff::BandedMatrix{T}       # Matrix representation of the differential operator from the left basis to the right basis.
    N::Int                      # Preallocated matrix dimension N.
    blockSize::Int              # Size of each block in the block structure of the operator.
    dom::Interval               # Domain of the operator.
    diff_shift::BandedMatrix{T} # Matrix representation of the identity operator from the left basis to the right basis (used for shifting).
    basisTransL::BandedMatrix{T} # Transformation matrix: converts coefficients of the left basis to coefficients in the normalized Legendre basis.
    basisTransR::BandedMatrix{T} # Transformation matrix: converts coefficients in the normalized Legendre basis to the coefficients of the right basis.
    shift::T                    # Shift value applied to the operator.
    qrData::GenBandedQrData{T}  # QR decomposition data for adaptive QR
end


function DiffOpBlock(N::Int, K::Int, blockSize::Int, map::Vector{Int}, coeffs::Vector{<:Tuple{Vararg{Vector{T}}}}, bcType::String, dom::Interval, shift::T) where {T<:FloatOrComplex}
    L, R, L_shift, M_bRC, bu, bl = diffOpBlockMat(N, K, blockSize, map, coeffs, bcType, dom, T)
    L = BandedMatrix(L, (bl, bu))
    R = BandedMatrix(R) 
    L_shift = BandedMatrix(L_shift)
    M_bRC = BandedMatrix(M_bRC)
    qrData = GenBandedQrData(L, L_shift)
    return DiffOpBlock(L, N, blockSize, dom, L_shift, M_bRC, R, shift, qrData)
end

function adaptiveQrSolve!(op::DiffOpBlock{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
        ny = length(y)
        ny_new = ny + op.basisTransR.l
        TR = op.basisTransR[1:ny_new, 1:ny]
        mul!(work[1:ny_new], TR, y)
        nx = gbaqsv!(op.qrData, op.shift, work[1:ny_new], tolSolve)
        nx_new = nx + op.basisTransL.l
        TL = op.basisTransL[1:nx_new, 1:nx]
        mul!(work[1:nx_new], TL, op.qrData.worky[1:nx])
        # println("n = ", ny, " n_next = ", nx_new)
        return work[1:nx_new]
    end
end

# redefine "-"
function -(op::DiffOpBlock{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end


function basisReCombMatBlock(bcType::String, N::Int, T::Type{T1}) where T1<:FloatOrComplex
    if bcType == "absorbing"
        d1 = zeros(T, N)
        d1[2:2:end] .= -1
        d2 = ones(T, N)
        d2[1:2:end] .= -1
        return spdiagm(N+2, N, -1 => d1, -2 => d2, 0=>ones(T, N)), 0, 2
    elseif bcType == "absorbing_conj"
        d1 = zeros(T, N)
        d1[2:2:end] .= 1
        d2 = ones(T, N)
        d2[1:2:end] .= -1
        return spdiagm(N+2, N, -1 => d1, -2 => d2, 0=>ones(T, N)), 0, 2
    else
        error("bcType not supported")
    end
end


function bandWidthBlock(K::Int, blockSize::Int, coeffs::Vector{<:Tuple{Vararg{Vector{T}}}}) where T
    bu = [bandWidth_shift(K, coeffs[i])[1] for i in 1:blockSize]
    bl = [bandWidth_shift(K, coeffs[i])[2] for i in 1:blockSize]
    return blockSize*maximum(bu), blockSize*maximum(bl)
end

function diffOpBlockMat(N::Int, K::Int, blockSize::Int, map::Vector{Int}, coeffs::Vector{<:Tuple{Vararg{Vector{T1}}}}, bcType::String, dom::Interval, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    bu, bl = bandWidthBlock(K, blockSize, coeffs)
    ~, bu_bRC, bl_bRC = basisReCombMatBlock(bcType, 10, T)
    tempN = N+5*(bl + bl_bRC)+bl_bRC 
    tempN_total = blockSize*tempN
    N_total = blockSize*N

    scale = 2/(dom.right - dom.left)
    M_ultras = spzeros(T, tempN_total, tempN_total)
    for i in 1:blockSize
        M_ultras[i:blockSize:end, i:blockSize:end] = ultrasMat(tempN, K, scale, coeffs[i], T)
    end

    M_bRC, ~ = basisReCombMatBlock(bcType, tempN_total-bl_bRC, T)

    # construct L
    L = M_ultras*M_bRC
    L = L[1:N_total+bl+bl_bRC, 1:N_total]

    # construct shift L
    L_shift = spzeros(T, tempN_total, tempN_total)
    for i in 1:blockSize
        L_shift[map[i]:blockSize:end, i:blockSize:end] = convertMat(tempN, 0, K, T)
    end
    L_shift = L_shift*M_bRC
    L_shift = L_shift[1:N_total+bl_bRC+blockSize-1, 1:N_total]


    # construct R
    R = spzeros(T, tempN_total, tempN_total)
    for i in 1:blockSize
        R[i:blockSize:end, map[i]:blockSize:end] .= convertMat(tempN, 0, K, T)*spdiagm(tempN, tempN, sqrt.(Vector{T}((2*(0:tempN-1).+1)/2)))
    end
    R = R[1:N_total+blockSize-1, 1:N_total]

    # construct M_bRC
    M_bRC, ~ = basisReCombMatBlock(bcType, N_total, T)
    weight = zeros(T, N_total+bl_bRC)
    for i = 1:blockSize
        idx = i:blockSize:N_total+bl_bRC
        weight[idx] .= sqrt.(2.0./((2*(0:length(idx)-1).+1)))
    end
    w = spdiagm(N_total+bl_bRC, N_total+bl_bRC, weight)
    M_bRC = w*M_bRC

    return L, R, L_shift, M_bRC, bu+bu_bRC+1, bl+bl_bRC+1
end