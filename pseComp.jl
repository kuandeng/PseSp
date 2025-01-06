mutable struct Options
    p::Int
    maxit::Int
    tol::AbstractFloat
    tolSolve::AbstractFloat
    stopCrit::String
    reOrth::Bool
end


function checkConv_pre(H, d_old, tol)
    d, Y = eigen(Symmetric(H))

    # get largest rize values
    idx = sortperm(d, rev=true)
    d = d[idx]
    isconv = (abs(d[1]/d_old-1) < tol)
    return isconv, d, Y, idx
end

function checkConv(H, β, tol)
    d, Y = eigen(Symmetric(H))
    res = abs.(β*Y[end, :])
    # get largest rize values
    idx = sortperm(d, rev=true)
    d = d[idx]
    res = res[idx]
    tolmax = 100*eps(1.0)*abs.(d[1])^(3/2)
    isconv = res[1] < max(tol*abs(d[1]), tolmax)
    return isconv, d, Y, idx
end


# copy x to workx and fill the rest with zeros
function copytoFill0!(workx::AbstractVector{T}, x::AbstractVector{T}) where T
    nx = length(x)
    @views begin
        copyto!(workx, x)
        workx[nx+1:end] .= zero(T)
        return workx[1:nx]
    end
end

function orthogonalize!(U::AbstractVecOrMat{T}, numax::Int, w::AbstractVector{T}, nw::Int, work::AbstractVector{T}, workorth::AbstractVector{T}) where T
    nw_next = max(nw, numax)
    nU = size(U, 2) 
    @views @inbounds begin
        # clean workspace
        copyto!(work, w)
        work[nw+1:nw_next] .= zero(T)
        # reorthogonalize
        mul!(workorth[1:nU], U[1:nw_next, :]', work[1:nw_next])
        gemv!('N', -one(T), U[1:nw_next, :], workorth[1:nU], one(T), work[1:nw_next])
        return work[1:nw_next]
    end 
end


# reorthogonalize w to U, note that nw and numax are not equal
function simpleReorthogonalize!(U::AbstractVecOrMat{T}, numax::Int, w::AbstractVector{T}, nw::Int, work::AbstractVector{T}, workorth::AbstractVector{T}) where T
    nw_next = max(nw, numax)
    nU = size(U, 2) 
    @views @inbounds begin
        # clean workspace
        copyto!(work, w)
        work[nw+1:nw_next] .= zero(T)
        # reorthogonalize
        mul!(workorth[1:nU], U[1:nw_next, :]', work[1:nw_next])
        gemv!('N', -one(T), U[1:nw_next, :], workorth[1:nU], one(T), work[1:nw_next])
        # reorthogonalize twice
        mul!(workorth[1:nU], U[1:nw_next, :]', work[1:nw_next])
        gemv!('N', -one(T), U[1:nw_next, :], workorth[1:nU], one(T), work[1:nw_next])
        return work[1:nw_next]
    end 
end

# axpy function with different length x and y.
function axpyDL!(α::T1, x::AbstractVector{T}, y::AbstractVector{T}, worky::AbstractVector{T}) where {T, T1}
    nx = length(x)
    ny = length(y)
    @views @inbounds begin
        if nx > ny
            worky[ny+1:nx] .= zero(T)
        end
        axpy!(α, x, worky[1:nx])
        return worky[1:max(nx, ny)]
    end
end


function innerproductDL(x::AbstractVector{T}, y::AbstractVector{T}) where T
    nx = length(x)
    ny = length(y)
    n = min(nx, ny)
    @views return x[1:n]'*y[1:n]
    
end


# inverse Lanczos method, be careful with the different 
function invLanczos(op::Op{T}, op_conj::Op{T}, u0::AbstractVector{T}, maxit::Int, p::Int, tol::AbstractFloat, tolSolve::AbstractFloat,reOrth::Bool, stopCrit::String,  U::AbstractMatrix{T}, worku::AbstractVector{T}, workv::AbstractVector{T}, workw::AbstractVector{T}, workorth::AbstractVector{T}, H::AbstractMatrix{T1}) where {T<:FloatOrComplex, T1<:AbstractFloat}
    N = op.N
    sizeU = 1
    numax = 0
    justRestarted = false
    u = u0
    α = zero(T)
    β = zero(T)
    d = zero(T)
    d_old = zero(real(T))
    pse_z = zero(real(T))
    @views @inbounds begin
    for mm = 1:maxit 
        for jj = sizeU:p
            copytoFill0!(U[:, jj], u)
            numax = max(numax, length(u))
            v = adaptiveQrSolve!(op, u, workv, tolSolve)
            w = adaptiveQrSolve!(op_conj, v, workw, tolSolve)
            nw = length(w)
            if jj > 1
                # w .-= β.*U[1:nw, jj-1]
                w = axpyDL!(-β, U[1:numax, jj-1], w, workw)
            end
            # α = real(U[1:nw, jj]'*w)
            α = real(innerproductDL(U[1:numax, jj], w))

            if justRestarted
                # # println("nw = ", nw)
                # nw = length(w)
                # w .-= U[1:nw, 1:jj]*((U[1:nw, 1:jj]'*w))
                # # w .-= U[1:nw, 1:jj]*((U[1:nw, 1:jj]'*w))
                u = orthogonalize!(U[1:numax, 1:jj], numax, w, length(w), worku, workorth)
                u = simpleReorthogonalize!(U[1:numax, 1:jj], numax, u, length(u), worku, workorth)
                justRestarted = false
            else
                # nw = length(w)
                # w .-= α.*U[1:nw, jj]
                w = axpyDL!(-α, U[1:numax, jj], w, workw)
                if reOrth
                    u = simpleReorthogonalize!(U[1:numax, 1:jj], numax, w, length(w), worku, workorth)
                else
                    u = w
                end
            end

            β = norm(u)
            u ./= β

            # construct H
            H[jj, jj] = α
            if jj < p
                H[jj, jj+1] = H[jj+1, jj] = β
            end
            # chech convergence
            if stopCrit == "pre"
                @views isconv, d, Y, idx = checkConv_pre(H[1:jj, 1:jj], d_old, tol)
            else
                @views isconv, d, Y, idx = checkConv(H[1:jj, 1:jj], β, tol)
            end
            d_old = d[1]
            try 
                pse_z = 1/sqrt(d[1])
            catch
                println("ill-condtion")
                pse_z = eps(real(T))
            end
            if isconv==1
                # println("converged ", pse_z)
                return pse_z, numax
            end
        end

        if mm == maxit
            # maximum iteration times reached
            # @warn "please try more iteration"
            return pse_z, numax
        else 
            # restart size
            k = ceil(Int, p/2)
        end

        # restart
        idx = idx[1:k]
        Y = Y[:,idx]
        U[:, 1:k] = U*Y

        # rebulid H
        H[1:k, 1:k] = diagm(d[1:k])
        H[1:k, k+1] = β*Y[end:end,:]
        H[k+1, 1:k] = β*Y[end:end,:]'
        # println("restart")
        justRestarted = true
        sizeU = k+1
    end
    end
end


function pseComp(L::Op{T}, L_conj::Op{T}, u0::AbstractVector{T}, ptx::Vector{T1}, pty::Vector{T1}, option::Options) where {T<:FloatOrComplex, T1<:AbstractFloat}

    # create workspace
    U = Matrix{T}(undef, L.N, option.p)
    worku = Vector{T}(undef, L.N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, option.p)
    H = zeros(real(T), option.p, option.p)

    reOrth = option.reOrth
    stopCrit = option.stopCrit

    # result
    nptx = length(ptx)
    npty = length(pty)
    pse = zeros(real(T), npty, nptx)
    dof = zeros(Int, npty, nptx)
    # main loop
    for i in 1:nptx
        println("$(i)/$(nptx)") 
        for j in 1:npty
            # shift operator
            z = ptx[i] + pty[j]*1.0im
            Lz = L - z
            Lz_conj = L_conj - z'
            pse[j, i], dof[j, i] = invLanczos(Lz, Lz_conj, u0, option.maxit, option.p, option.tol, option.tolSolve, reOrth, stopCrit, U, worku, workv, workw, workorth, H)
        end
    end             
    return pse, dof


end

function pseComp(L::Op{T}, L_conj::Op{T}, ptx::Vector{T1}, pty::Vector{T1}, option::Options) where {T<:FloatOrComplex, T1<:AbstractFloat}

    # default initial vector
    u0 = ones(T, 20)
    u0 = u0/norm(u0)
    return pseComp(L, L_conj, u0, ptx, pty, option)
end