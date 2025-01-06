mutable struct BandedQrData{T<:FloatOrComplex}
    data::Matrix{T}
    N :: Int
    bu :: Int
    bl :: Int
    qrStep :: Int
    workA :: Matrix{T}
    worky :: Vector{T}
    workH :: Vector{T}
    tau :: Vector{T}
    refl :: Matrix{T}
end

function BandedQrData(b::BandedMatrix{T}) where T
    bu = b.u
    bl = b.l
    N = size(b, 2)
    return BandedQrData(b.data, N, bu, bl, 0, Matrix{T}(undef, 1+bu+2*bl, N), Vector{T}(undef, N), Vector{T}(undef, bu+bl+1), Vector{T}(undef, N), Matrix{T}(undef, bl+1, N))
end

mutable struct GenBandedQrData{T<:FloatOrComplex}
    data::Matrix{T}
    shiftData::Matrix{T}
    N :: Int
    bu :: Int
    bl :: Int
    bu_shift::Int
    bl_shift::Int
    qrStep :: Int
    workA :: Matrix{T}
    worky :: Vector{T}
    workH :: Vector{T}
    tau :: Vector{T}
    refl :: Matrix{T}
end

function GenBandedQrData(b::BandedMatrix{T}, b_shift::BandedMatrix{T}) where T
    bu = b.u
    bl = b.l
    bu_shift = b_shift.u
    bl_shift = b_shift.l
    N = size(b, 2)
    return GenBandedQrData(b.data, b_shift.data, N, bu, bl, bu_shift, bl_shift, 0, Matrix{T}(undef, 1+bu+2*bl, N), Vector{T}(undef, N), Vector{T}(undef, bu+bl+1), Vector{T}(undef, N), Matrix{T}(undef, bl+1, N))
end


for (fname, elty) in ((:dtbsv_,:Float64),
    (:stbsv_,:Float32),
    (:ztbsv_,:ComplexF64),
    (:ctbsv_,:ComplexF32))
    @eval begin
                #       SUBROUTINE DTBSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
                #       .. Scalar Arguments ..
                #       INTEGER INCX,LDA,N
                #       CHARACTER DIAG,TRANS,UPLO
                #       .. Array Arguments ..
                #       DOUBLE PRECISION A(LDA,*),X(*)
        function tbsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, k::Int, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            chkuplo(uplo)
            require_one_based_indexing(A, x)
            n = size(A,2)
            if n != length(x)
                throw(DimensionMismatch(lazy"size of A is $n != length(x) = $(length(x))"))
            end
            chkstride1(A)
            px, stx = vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
            GC.@preserve x ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Clong, Clong, Clong),
                uplo, trans, diag, n, k,
                A, max(1,stride(A,2)), px, stx, 1, 1, 1)
            x
        end
        function tbsv(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, k::Int, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            tbsv!(uplo, trans, diag, k, A, copy(x))
        end
    end
end

# copy banded matrix with a shift diagonal matrix
function sbcopy!(X::Matrix{T}, Y::Matrix{T}, z::T, bu::Int, bl::Int, m::Int, n::Int) where T    
    @inbounds @views begin
        copyto!(Y[bl+1:end, m:n], X[1:end, m:n])
        Y[1:bl, m:n] .= T(0)
        Y[bu+bl+1, m:n] .-= z
    end
end

# copy banded matrix with a shift banded matrix
function sbcopy!(X::Matrix{T}, Y::Matrix{T}, X_shift::Matrix{T}, z::T, bu::Int, bl::Int, bu_shift::Int, bl_shift::Int, m::Int, n::Int) where T    
    @inbounds @views begin
        copyto!(Y[bl+1:end, m:n], X[1:end, m:n])
        Y[1:bl, m:n] .= T(0)
        Y[bu+bl+1-bu_shift:bu+bl+1+bl_shift, m:n] .-= z.*X_shift[:, m:n]
    end
end

# generalized  banded matrix adaptive qr solve 
function gbaqsv!(A::Matrix{T}, N::Int, bu::Int, bl::Int, A_shift::Matrix{T}, bu_shift::Int, bl_shift::Int, y::AbstractVector{T}, z::T, qrStep::Int, tau::Vector{T}, refl::Matrix{T}, workA::Matrix{T}, worky::Vector{T}, workH::Vector{T}, tolSolve::AbstractFloat) where T
    @inbounds @views begin

        # parm initialize
        workbu = bu+bl # upper bandwith of workA
        # tol = 1*eps(real(T)) # tolerance for adaptive qr
        incStep = 1 # increment step for adaptive qr

        # initial
        ny = length(y)
        if ny+incStep+workbu > N
            error("please try larger dimension")
        end
        normy = norm(y) 
        copyto!(worky, y)
        worky[ny+1:ny+bl] .= T(0)

        # make the valid columns number of workA always equals to qrStep plus workbu
        if qrStep == 0
            sbcopy!(A, workA, A_shift, z, bu, bl, bu_shift, bl_shift, 1, qrStep+workbu)
        end

        n = 0
        # first qr at n_init
        n_init = max(ny-bl, 1)
        qrStep = gbaqrf!(A, A_shift, workA, workbu, bu, bl, bu_shift, bl_shift, z, worky, n, ny, qrStep, n_init, tau, refl, workH)
        n = n_init
        residual = norm(worky[n+1:max(ny, n+bl)])

        while (n+incStep+workbu <= N && residual > tolSolve*normy)
            n_next = n+incStep 
            qrStep = gbaqrf!(A, A_shift, workA, workbu, bu, bl, bu_shift, bl_shift, z, worky, n, ny, qrStep, n_next, tau, refl, workH)
            n = n_next
            # println(worky[n+1:n+bl], "n = ", n, "bl = ", bl)
            # println("bl = ", bl)
            # println("residual = ", norm(worky[n+1:n+bl])/normy)
            residual = norm(worky[n+1:max(ny, n+bl)])
        end
        if n+incStep+workbu > N
            error("please try larger dimension")
        else
            tbsv!('U', 'N', 'N', workbu, workA[1:workbu+1, 1:n], worky[1:n])
            return n, qrStep
        end
    end
end




# banded matrix adaptive qr factor for both side.
# at qrStep, the qrStep + bu elements in workA is preallocated and qrStep + bl elements in worky is clean.

for (larf, elty) in
    ((:dlarf_, Float64),
     (:slarf_, Float32),
     (:zlarf_, ComplexF64),
     (:clarf_, ComplexF32))
    @eval begin
        function gbaqrf!(A::Matrix{$elty}, A_shift::Matrix{$elty}, workA::Matrix{$elty}, workbu, bu::Int, bl::Int, bu_shift::Int, bl_shift::Int, z::$elty, worky::Vector{$elty}, n::Int, ny::Int, qrStep::Int, n_next::Int, tau::Vector{$elty}, refl::Matrix{$elty}, workH::Vector{$elty})
        @inbounds @views begin  
            # allocate workA and clean worky
            if qrStep < n_next
                sbcopy!(A, workA, A_shift, z, bu, bl, bu_shift, bl_shift, qrStep+1+workbu, n_next+workbu)
            end
            worky[max(ny+1, n+1+bl):n_next+bl] .= 0
            # qr factor and apply to workA and worky
            for i = qrStep+1:n_next
                refl[1:end, i] = workA[workbu+1:workbu+bl+1, i]
                v = refl[1:end, i]
                tau[i] = larfg!(v)
                ccall((@blasfunc($larf), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Clong),
                'L', bl+1, workbu+1, v, 1,
                tau[i]', workA[workbu+1:workbu+bl+1, i:i+workbu], workbu+bl, workH, 1)
            end 

            for i = n+1:n_next
                larf!('L', refl[1:end, i], tau[i]', worky[i:i+bl, 1:1], workH)
            end

        end
        return max(qrStep, n_next)
        end
    end
end

# generalized  banded matrix adaptive qr solve 
function baqsv!(A::Matrix{T}, N::Int, bu::Int, bl::Int, y::AbstractVector{T}, z::T, qrStep::Int, tau::Vector{T}, refl::Matrix{T}, workA::Matrix{T}, worky::Vector{T}, workH::Vector{T}, tolSolve::AbstractFloat) where T
    @inbounds @views begin

        # parm initialize
        workbu = bu+bl # upper bandwith of workA
        incStep = 1 # increment step for adaptive qr

        # initial
        ny = length(y)
        if ny+incStep+workbu > N
            error("please try larger dimension")
        end
        normy = norm(y) 
        copyto!(worky, y)
        worky[ny+1:ny+bl] .= T(0)

        # make the valid columns number of workA always equals to qrStep plus workbu
        if qrStep == 0
            sbcopy!(A, workA, z, bu, bl, 1, qrStep+workbu)
        end

        n = 0
        # first qr at n_init
        n_init = max(ny-2*bl, 1)
        qrStep = baqrf!(A, workA, workbu, bu, bl, z, worky, n, ny, qrStep, n_init, tau, refl, workH)
        n = n_init
        residual = norm(worky[(n+1):max(ny, n+bl)])
        # println("residual = ", residual)
        while (n+incStep+workbu <= N && residual > tolSolve*normy)
            n_next = n+incStep 
            qrStep = baqrf!(A, workA, workbu, bu, bl, z, worky, n, ny, qrStep, n_next, tau, refl, workH)
            n = n_next
            # println(worky[n+1:n+bl], "n = ", n, "bl = ", bl)
            # println("bl = ", bl)
            # println("residual = ", norm(worky[n+1:n+bl])/normy)
            residual = norm(worky[n+1:max(ny, n+bl)])
        end
        if n+incStep+workbu > N
            error("please try larger dimension")
        else
            tbsv!('U', 'N', 'N', workbu, workA[1:workbu+1, 1:n], worky[1:n])
            return n, qrStep
        end
    end
end



for (larf, elty) in
    ((:dlarf_, Float64),
     (:slarf_, Float32),
     (:zlarf_, ComplexF64),
     (:clarf_, ComplexF32))
    @eval begin
        function baqrf!(A::Matrix{$elty}, workA::Matrix{$elty}, workbu, bu::Int, bl::Int, z::$elty, worky::Vector{$elty}, n::Int, ny::Int, qrStep::Int, n_next::Int, tau::Vector{$elty}, refl::Matrix{$elty}, workH::Vector{$elty})
        @inbounds @views begin  
            # allocate workA and clean worky
            if qrStep < n_next
                sbcopy!(A, workA, z, bu, bl, qrStep+1+workbu, n_next+workbu)
            end
            worky[max(ny+1, n+1+bl):n_next+bl] .= 0
            # qr factor and apply to workA and worky
            for i = qrStep+1:n_next
                refl[1:end, i] = workA[workbu+1:workbu+bl+1, i]
                v = refl[1:end, i]
                tau[i] = larfg!(v)
                ccall((@blasfunc($larf), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Clong),
                'L', bl+1, workbu+1, v, 1,
                tau[i]', workA[workbu+1:workbu+bl+1, i:i+workbu], workbu+bl, workH, 1)
            end 

            for i = n+1:n_next
                larf!('L', refl[1:end, i], tau[i]', worky[i:i+bl, 1:1], workH)
            end

        end
        return max(qrStep, n_next)
        end
    end
end


# adaptive qr solve with workspace
function baqsv!(qrData::BandedQrData{T}, z::T, y::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @unpack data, N, bu, bl, qrStep, workA, worky, workH, tau, refl = qrData
    n, qrStep_new = baqsv!(data, N, bu, bl, y, z, qrStep, tau, refl, workA, worky, workH, tolSolve)
    qrData.qrStep = qrStep_new
    # println("n = ", n, ", qrStep = ", qrStep_new)
    return n
end

# adaptive qr solve with workspace
baqsv!(qrData::BandedQrData{T}, z::T, y::AbstractVector{T}) where T = baqsv!(qrData, z, y, eps(real(T)))


# adaptive qr solve with workspace
function gbaqsv!(qrData::GenBandedQrData{T}, z::T, y::AbstractVector{T}, tolSolve::AbstractFloat) where T
    @unpack data, shiftData, N, bu, bl, bu_shift, bl_shift, qrStep, workA, worky, workH, tau, refl = qrData
    n, qrStep_new = gbaqsv!(data, N, bu, bl, shiftData, bu_shift, bl_shift, y, z, qrStep, tau, refl, workA, worky, workH, tolSolve)
    qrData.qrStep = qrStep_new
    # println("n = ", n, ", qrStep = ", qrStep_new)
    return n
end

gbaqsv!(qrData::GenBandedQrData{T}, z::T, y::AbstractVector{T}) where T = gbaqsv!(qrData, z, y, eps(real(T)))