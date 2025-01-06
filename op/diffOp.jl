 
# defintion of differential operator, with preallocated matrix dimension N
mutable struct DiffOp{T<:FloatOrComplex} <: Op{T}
    diff::BandedMatrix{T} # matrix representation of differential operator form left basis to right basis
    N::Int # preallocated matrix dimension N
    dom::Interval 
    diff_shift::BandedMatrix{T} # matrix representation of Identity operator form left basis to right basis
    basisTransL::BandedMatrix{T} # transpose coefficients of left basis to coefficients of normalized Legendre basis
    basisTransR::BandedMatrix{T} # transpose coefficients of normalized Legendre basis to coefficients of right basis
    shift::T
    qrData::GenBandedQrData{T}
end

function DiffOp(N::Int, K::Int, coeffs::Tuple{Vararg{Vector{T}}}, bcType::String, bcOrder::Int, dom::Interval, shift::T) where {T<:FloatOrComplex}
    L, R, L_shift, M_bRC, bu, bl = diffOpMat(N, K, coeffs, bcType, bcOrder, dom, T)
    L = BandedMatrix(L, (bl, bu))
    R = BandedMatrix(R) 
    L_shift = BandedMatrix(L_shift)
    M_bRC = BandedMatrix(M_bRC)
    qrData = GenBandedQrData(L, L_shift)
    return DiffOp(L, N, dom, L_shift, M_bRC, R, shift, qrData)
end

function adaptiveQrSolve!(op::DiffOp{T}, y::AbstractVector{T}, work::AbstractVector{T}, tolSolve::AbstractFloat) where T
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
function -(op::DiffOp{T}, z::FloatOrComplex) where T    
    op.shift = z 
    op.qrData.qrStep = 0
    return op
end



function convertMat(N::Int, λ::T) where {T<:FloatOrComplex}
    d1 = zeros(T, N)
    d2 = zeros(T, N-2)
    for i = 1:N
        if (i==1) d1[i] = 1
        elseif (i==2) d1[i] = λ/(λ+1)
        else
            d1[i] = λ/(λ+i-1)
            d2[i-2] = -λ/(λ+i-1)
        end
    end
    return spdiagm(N, N, 0 => d1, 2 => d2)
end

function convertMat(N::Int, k1::Int, k2::Int, T::Type{T1}) where {T1<:FloatOrComplex}
    S = spdiagm(N, N, 0 => ones(T, N))
    for k = k1:k2-1
        S = convertMat(N, T(k)+T(1/2))*S
    end
    return S
end

# function diffMat(N::Int, k::Int, T::Type{T1}) where {T1<:FloatOrComplex}
#     if (k == 0) return spdiagm(N, N, 0 => ones(T, N));
#     else
#         temp = one(T);
#         for i = 1:k
#             temp = temp*(2i-1);
#         end 
#         return spdiagm(N, N, k => temp.*ones(T, N-k));
#     end
# end

function diffMat(N::Int, k1::Int, k2::Int, T::Type{T1}) where {T1<:FloatOrComplex}
    if (k1 == k2) return spdiagm(N, N, 0 => ones(T, N));
    else
        temp = one(T);
        for i = k1+1:k2
            temp = temp*(2i-1)
        end

        return spdiagm(N, N, k2-k1 => temp.*ones(T, N-(k2-k1)))
    end
end

diffMat(N::Int, k::Int, T::Type{T1}) where {T1<:FloatOrComplex} = diffMat(N, 0, k, T)


function multMat(N::Int, k::Int, coeffs::AbstractVector{T1}, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    if N < length(coeffs)
        error("N should be larger than the length of coeffs")
    end
    m = size(coeffs,1);
    if (m == 1)
        M = coeffs[1]*spdiagm(N, N, ones(T, N));
    else
        lam = k + T(1/2);

        d1 = zeros(T, 2*N-1)
        d2 = zeros(T, 2*N-1)
        for i = 1:2*N-1
            d1[i] = i/2/(lam+i-1)
            d2[i] = (2lam+i-1)/2/(lam+i)
        end
        Mx = spdiagm(2N, 2N, -1 =>d1, 1 => d2)
        coeffs = [coeffs;zeros(T, N-m)]
        coeffs = convertMat(N, 0, k, T)*coeffs
        M0 = spdiagm(2N, 2N, ones(T, 2N))
        M1 = 2lam*Mx
        M = coeffs[1]*M0+coeffs[2]*M1
        for i = 1:m-2
            M2 = 2*(lam+i)/(i+1)*Mx*M1-(i+2lam-1)/(i+1)*M0
            M = M + coeffs[i+2]*M2
            M0 = M1
            M1 = M2
        end    
    end
    return M[1:N,1:N]
end



function basisReCombMat(bcType::String, bcOrder::Int, N::Int, T::Type{T1}) where T1<:FloatOrComplex
    if (bcType == "Diri" && bcOrder == 2)
        # return spdiagm(N+2, N, -2 => -1*ones(T, N), 0 => ones(T, N))*spdiagm(N, N, sqrt.(1.0./(4*(1:N).+2))), 0, 2
        return spdiagm(N+2, N, -2 => -1*ones(T, N), 0 => ones(T, N)), 0, 2

        
    end

    if (bcType == "Diri" && bcOrder == 4)
        d2 = -(4.0.*(1:N) .+ 6)./(2.0.*(1:N) .+ 5)
        d4 = (2.0.*(1:N) .+ 1)./(2.0.*(1:N) .+ 5)
        scale = sqrt.(2.0 .*(2.0 .*(1:N) .+ 1).^2 .*(2.0 .*(1:N) .+ 3))
        # return spdiagm(N+4, N, -4 => d4, -2 => d2, 0 => ones(T, N))*spdiagm(N, N, 1.0 ./ scale), 0, 4
        return spdiagm(N+4, N, -4 => d4, -2 => d2, 0 => ones(T, N)), 0, 4
    end


    if (bcType == "DiriL" && bcOrder == 1)
        return spdiagm(N+1, N, -1 => ones(T, N), 0 => ones(T, N)), 0, 1
    end

    if (bcType == "DiriR" && bcOrder == 1)
        return spdiagm(N+1, N, -1 => -1.0*ones(T, N), 0 => ones(T, N)), 0, 1
    end
end

function bandWidth(K::Int, coeffs::Tuple{Vararg{Vector{T}}}) where T
    bu = typemin(Int)
    bl = typemin(Int)
    for i = 0:size(coeffs,1)-1
        if (coeffs[i+1] != zeros(T, 1))
            bu = max(length(coeffs[i+1])-i+2K-1, bu)
            bl = max(length(coeffs[i+1])-i-1, bl)
        end
    end
    return bu, bl
end

function bandWidth_shift(K::Int, coeffs::Tuple{Vararg{Vector{T}}}, coeffs_shift::Tuple{Vararg{Vector{T}}}) where T
    bu, bl = bandWidth(K, coeffs)
    bu_shift, bl_shift = bandWidth(K, coeffs_shift)
    return max(bu, bu_shift), max(bl, bl_shift)
end

bandWidth_shift(K::Int, coeffs::Tuple{Vararg{Vector{T}}}) where T = bandWidth_shift(K, coeffs, (ones(T, 1), ))


function ultrasMat(N::Int, K1::Int, K2::Int, scale::T3, coeffs::Tuple{Vararg{Vector{T1}}}, T::Type{T2}) where {T1, T2<:FloatOrComplex, T3}
    M_ultras = spzeros(T, N, N)
    for k = K1:K1+size(coeffs, 1)-1
        M_ultras = M_ultras + convertMat(N, k, K2, T)*multMat(N, k, scale^k*coeffs[k-K1+1], T)*diffMat(N, K1, k, T)
    end
    return M_ultras
end 


ultrasMat(N::Int, K::Int, scale::T3, coeffs::Tuple{Vararg{Vector{T1}}}, T::Type{T2}) where {T1, T2<:FloatOrComplex, T3} = ultrasMat(N, 0, K, scale, coeffs, T)


function diffOpMat(N::Int, K::Int, coeffs::Tuple{Vararg{Vector{T1}}}, bcType::String, bcOrder::Int, dom::Interval, T::Type{T2}) where {T1, T2<:FloatOrComplex}
    bu, bl = bandWidth_shift(K, coeffs)
    ~, bu_bRC, bl_bRC = basisReCombMat(bcType, bcOrder, 10, T)
    tempN = N+5*(bl + bl_bRC)+bl_bRC 

    scale = 2/(dom.right - dom.left)
    M_ultras = ultrasMat(tempN, K, scale, coeffs, T)

    M_bRC, ~ = basisReCombMat(bcType, bcOrder, tempN-bl_bRC, T)

    # construct L
    L = M_ultras*M_bRC
    L = L[1:N+bl+bl_bRC, 1:N]

    # construct shift L
    L_shift = convertMat(tempN, 0, K, T)*M_bRC
    L_shift = L_shift[1:N+bl_bRC, 1:N]

    # construct R
    R = convertMat(N, 0, K, T)*spdiagm(N, N, sqrt.(Vector{T}((2*(0:N-1).+1)/2)))

    # construct M_bRC
    M_bRC, ~ = basisReCombMat(bcType, bcOrder, N, T)
    M_bRC = spdiagm(N+bl_bRC, N+bl_bRC, sqrt.(Vector{T}(2 ./(2*(0:N-1+bl_bRC).+1))))*M_bRC

    return L, R, L_shift, M_bRC, bu+bu_bRC, bl+bl_bRC
end

