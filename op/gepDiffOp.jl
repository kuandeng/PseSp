
# defintion of generalized differential operator, with preallocated matrix dimension N
mutable struct GepDiffOp{T<:FloatOrComplex} <: Op{T}
    diff_L::BandedMatrix{T} # matrix representation of differential operator form left basis to right basis
    N::Int # preallocated matrix dimension N
    dom::Interval 
    isConj::Bool
    diff_shift::BandedMatrix{T} # matrix representation of right differential operator
    basisTransL::BandedMatrix{T} # 
    basisTransR::BandedMatrix{T} # 
    shift::T
    qrData::GenBandedQrData{T}
end

function GepDiffOp(N::Int, K_L::Int, K_R::Int, coeffs_L::Tuple{Vararg{Vector{T1}}}, coeffs_R::Tuple{Vararg{Vector{T1}}}, bcType_L::String, bcOrder_L::Int, dom::Interval, shift::T, isConj::Bool) where {T1, T<:FloatOrComplex}
    if isConj
        L, transR, L_shift, transL, bu, bl = gepDiffOpConjMat(N, K_L, K_R, coeffs_L, coeffs_R, bcType_L, bcOrder_L, dom, T)
    else
        L, transR, L_shift, transL, bu, bl = gepDiffOpMat(N, K_L, K_R, coeffs_L, coeffs_R, bcType_L, bcOrder_L, dom, T)
    end
    L = BandedMatrix(L, (bl, bu))
    transR = BandedMatrix(transR)
    L_shift = BandedMatrix(L_shift)
    transL = BandedMatrix(transL)
    qrData = GenBandedQrData(L, L_shift)
    return GepDiffOp(L, N, dom, isConj, L_shift, transL, transR, shift, qrData)
end


function adaptiveQrSolve!(op::GepDiffOp{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @views @inbounds begin
        ny = length(y)
        bl_R = op.basisTransR.l
        TR = op.basisTransR[1:ny+bl_R, 1:ny]
        mul!(work[1:ny+bl_R], TR, y)
        nx = gbaqsv!(op.qrData, op.shift, work[1:ny+bl_R], tolSolve)
        nx_new = nx + op.basisTransL.l
        TL = op.basisTransL[1:nx_new, 1:nx]
        mul!(work[1:nx_new], TL, op.qrData.worky[1:nx])
        # println("n = ", ny, " n_next = ", nx_new)
        return work[1:nx_new]
    end
end

# redefine "-"
function -(op::GepDiffOp{T}, z::FloatOrComplex) where T    
    op.shift = z' 
    op.qrData.qrStep = 0
    return op
end



function gepDiffOpMat(N::Int, K_L::Int, K_R::Int, coeffs_L::Tuple{Vararg{Vector{T1}}}, coeffs_R::Tuple{Vararg{Vector{T1}}}, bcType_L::String, bcOrder_L::Int, dom::Interval,T::Type{T2}) where {T1, T2<:FloatOrComplex}

    # computing bandwidth for both side
    bu_L, bl_L = bandWidth_shift(K_L, coeffs_L, coeffs_R)
    bu_R, bl_R = bandWidth_shift(K_L, coeffs_R)
    ~, bu_bRC, bl_bRC = basisReCombMat(bcType_L, bcOrder_L, 10, T)

    # using larger N to make sure the element is true
    tempN = N+5*(max(bl_L, bl_R) + bl_bRC)+ bl_bRC

    # basis recomb matrix
    M_bRC, ~ = basisReCombMat(bcType_L, bcOrder_L, tempN-bl_bRC, T)

    scale = 2/(dom.right - dom.left)
    # construct L
    L = ultrasMat(tempN, K_L, scale, coeffs_L, T)*M_bRC
    L = L[1:N+bl_L+bl_bRC, 1:N]

    # construct shift L
    L_shift = ultrasMat(tempN, K_L, scale, coeffs_R, T)*M_bRC
    L_shift = L_shift[1:N+bl_R+bl_bRC, 1:N]

    # construct transR
    transR = ultrasMat(N, K_R, K_L, scale, coeffs_R, T)

    # construct transL
    M_bRC, ~ = basisReCombMat(bcType_L, bcOrder_L, N, T)
    transL = spdiagm(N+bl_bRC, N+bl_bRC, sqrt.(Vector{T}(2 ./(2*(0:N-1+bl_bRC).+1))))*M_bRC
    return L, transR, L_shift, transL, bu_L+bu_bRC, bl_L+bl_bRC
end


function gepDiffOpConjMat(N::Int, K_L::Int, K_R::Int, coeffs_L::Tuple{Vararg{Vector{T1}}}, coeffs_R::Tuple{Vararg{Vector{T1}}}, bcType_L::String, bcOrder_L::Int, dom::Interval,T::Type{T2}) where {T1, T2<:FloatOrComplex}

    # computing bandwidth for both side
    bu_L, bl_L = bandWidth_shift(K_L, coeffs_L, coeffs_R)
    bu_R, bl_R = bandWidth_shift(K_L, coeffs_R)
    ~, bu_bRC, bl_bRC = basisReCombMat(bcType_L, bcOrder_L, 10, T)

    # using larger N to make sure the element is true
    tempN = N+5*(max(bl_L, bl_R) + bl_bRC)+ bl_bRC

    # basis recomb matrix
    M_bRC, ~ = basisReCombMat(bcType_L, bcOrder_L, tempN-bl_bRC, T)

    scale = 2/(dom.right - dom.left)
    # construct L
    L = ultrasMat(tempN, K_L, scale, coeffs_L, T)*M_bRC
    L = L[1:N+bl_L+bl_bRC, 1:N]

    # construct shift L
    L_shift = ultrasMat(tempN, K_L, scale, coeffs_R, T)*M_bRC
    L_shift = L_shift[1:N+bl_R+bl_bRC, 1:N]

    # construct transR
    transR = convertMat(N, 0, K_L, T)*spdiagm(N, N, sqrt.(Vector{T}((2*(0:N-1).+1)/2)))


    # construct transL
    bu_R_transL, bl_R_transL =  bandWidth(K_R, coeffs_R)
    tempN_transL = N + 5*(bl_R_transL + bl_bRC) + bl_bRC
    M_ultras_transL = ultrasMat(tempN_transL, 0, K_R, scale, coeffs_R, T)
    M_bRC, ~ = basisReCombMat(bcType_L, bcOrder_L, tempN_transL-bl_bRC, T)
    transL = M_ultras_transL*M_bRC
    transL = transL[1:N+bl_R_transL+bl_bRC, 1:N]

    return L, transR, L_shift, transL, bu_L+bu_bRC, bl_L+bl_bRC
end






