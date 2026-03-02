ENV["GKSwstype"] = "100"

include("../PseSp.jl")

using ApproxFun
using LinearAlgebra
using Printf
using GenericSchur

to_big(x::BigFloat) = x
to_big(x::Real) = parse(BigFloat, string(x))

to_big_complex(z::Complex{BigFloat}) = z
to_big_complex(z::Complex) = Complex{BigFloat}(to_big(real(z)), to_big(imag(z)))

function parse_kv_args(args::Vector{String})
    d = Dict{String, String}()
    for a in args
        p = findfirst(==('='), a)
        p === nothing && continue
        k = strip(a[1:p-1])
        v = strip(a[p+1:end])
        !isempty(k) && (d[k] = v)
    end
    return d
end

function cheb_to_ultra_coeffs(
    c_cheb::AbstractVector{T},
    a::T,
    b::T;
    lambda::T = T(0.5),
) where {T<:AbstractFloat}
    S_cheb = Chebyshev(a..b)
    S_ultra = Ultraspherical(lambda, a..b)
    f_cheb = Fun(S_cheb, c_cheb)
    f_ultra = Fun(f_cheb, S_ultra)
    return coefficients(f_ultra)
end

function build_vaadv_ops(
    a0_coeffs::AbstractVector{T},
    a1_coeffs::AbstractVector{T},
    a0c_coeffs::AbstractVector{T},
    a1c_coeffs::AbstractVector{T},
    h2::T,
    N::Int,
    dom::Interval,
) where {T<:FloatOrComplex}
    coeffs = (a0_coeffs, a1_coeffs, T[h2])
    coeffs_conj = (a0c_coeffs, a1c_coeffs, T[h2])
    K = 2
    bcType = "Periodic"
    bcOrder = 2
    op = DiffOp(N, K, coeffs, bcType, bcOrder, dom, zero(T); coeff_basis = :chebT)
    op_conj = DiffOp(N, K, coeffs_conj, bcType, bcOrder, dom, zero(T); coeff_basis = :chebT)
    return op, op_conj
end

function dominant_residual(Hsub::AbstractMatrix{RT}, beta::RT) where {RT<:AbstractFloat}
    d, Y = eigen(Symmetric(Hsub))
    idx = sortperm(d, rev = true)
    d = d[idx]
    Y = Y[:, idx]
    return abs(beta * Y[end, 1]), d[1]
end

function inv_lanczos_trace(
    op::Op{T},
    op_conj::Op{T},
    u0::AbstractVector{T},
    maxit::Int,
    p::Int,
    tol::AbstractFloat,
    tolSolve::AbstractFloat,
    reOrth::Bool,
    stopCrit::String,
) where {T<:FloatOrComplex}
    N = op.N
    U = Matrix{T}(undef, N, p)
    worku = Vector{T}(undef, N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, p)
    H = zeros(real(T), p, p)

    sizeU = 1
    numax = 0
    justRestarted = false
    u = copy(u0)
    alpha = zero(T)
    beta = zero(T)
    d_old = zero(real(T))
    pse_z = zero(real(T))
    lanczos_steps = 0

    pse_hist = Vector{real(T)}()
    res_hist = Vector{real(T)}()
    conv_hist = Bool[]

    @views @inbounds for mm = 1:maxit
        for jj = sizeU:p
            lanczos_steps += 1
            copytoFill0!(U[:, jj], u)
            numax = max(numax, length(u))

            v = adaptiveQrSolve!(op, u, workv, tolSolve)
            w = adaptiveQrSolve!(op_conj, v, workw, tolSolve)

            if jj > 1
                w = axpyDL!(-beta, U[1:numax, jj - 1], w, workw)
            end

            alpha = real(innerproductDL(U[1:numax, jj], w))

            if justRestarted
                u = orthogonalize!(U[1:numax, 1:jj], numax, w, length(w), worku, workorth)
                u = simpleReorthogonalize!(U[1:numax, 1:jj], numax, u, length(u), worku, workorth)
                justRestarted = false
            else
                w = axpyDL!(-alpha, U[1:numax, jj], w, workw)
                if reOrth
                    u = simpleReorthogonalize!(U[1:numax, 1:jj], numax, w, length(w), worku, workorth)
                else
                    u = w
                end
            end

            beta = norm(u)

            H[jj, jj] = alpha
            if jj < p
                H[jj, jj + 1] = beta
                H[jj + 1, jj] = beta
            end

            isconv = false
            d = nothing
            Y = nothing
            idx = nothing

            Hsub = Matrix(H[1:jj, 1:jj])
            if stopCrit == "fixed"
                _, d, Y, idx = checkConv(Hsub, beta, tol)
                isconv = false
            elseif stopCrit == "pre"
                isconv, d, Y, idx = checkConv_pre(Hsub, d_old, tol)
            else
                isconv, d, Y, idx = checkConv(Hsub, beta, tol)
            end

            d_old = d[1]
            if d[1] > zero(real(T))
                pse_z = inv(sqrt(d[1]))
            else
                pse_z = eps(real(T))
            end

            res, _ = dominant_residual(Hsub, beta)
            push!(pse_hist, pse_z)
            push!(res_hist, res)
            push!(conv_hist, Bool(isconv))

            if isconv
                return pse_z, numax, lanczos_steps, pse_hist, res_hist, conv_hist
            end

            if beta == zero(real(T))
                return pse_z, numax, lanczos_steps, pse_hist, res_hist, conv_hist
            end
            u ./= beta
        end

        if mm == maxit
            return pse_z, numax, lanczos_steps, pse_hist, res_hist, conv_hist
        end

        k = ceil(Int, p / 2)
        idxk = idx[1:k]
        Yk = Y[:, idxk]
        U[:, 1:k] = U * Yk

        H[1:k, 1:k] = diagm(d[1:k])
        H[1:k, k + 1] = beta * Yk[end:end, :]
        H[k + 1, 1:k] = beta * Yk[end:end, :]'

        justRestarted = true
        sizeU = k + 1
    end

    return pse_z, numax, lanczos_steps, pse_hist, res_hist, conv_hist
end

function write_trace_csv(path::String, pse_hist, res_hist, conv_hist)
    open(path, "w") do io
        println(io, "step,pse,residual,isconv")
        for i in eachindex(res_hist)
            println(io, string(i, ",", pse_hist[i], ",", res_hist[i], ",", conv_hist[i]))
        end
    end
end

function run_wiener_truth_trace(; N::Int = 300, p_true::Int = 40, maxit::Int = 2, tol::Union{Nothing, String} = nothing, tolSolve::Union{Nothing, String} = nothing, prec_bits::Int = 128, z::Complex = Complex{BigFloat}(big"0.7", big"0.0"))
    return setprecision(prec_bits) do
        T = Complex{BigFloat}
        a1, b1 = BigFloat(-10.0), BigFloat(0.0)
        a2, b2 = BigFloat(0.0), BigFloat(10.0)

        f1_cheb = Fun(x -> exp(x), Chebyshev(a1..b1))
        f2_cheb = Fun(x -> exp(-x), Chebyshev(a2..b2))
        u1 = cheb_to_ultra_coeffs(coefficients(f1_cheb), a1, b1; lambda = BigFloat(0.5))
        u2 = cheb_to_ultra_coeffs(coefficients(f2_cheb), a2, b2; lambda = BigFloat(0.5))
        coeffs = T.(u1)
        coeffs_conj = T.(u2)

        dom_u = Interval{BigFloat}(zero(BigFloat), BigFloat(10.0))
        op = VoltConvOp(coeffs, N, dom_u, 'r', zero(T))
        op_conj = VoltConvOp(coeffs_conj, N, dom_u, 'l', zero(T))

        tolB = tol === nothing ? BigFloat("1e-20") : parse(BigFloat, tol)
        tolSolveB = tolSolve === nothing ? eps(BigFloat) : parse(BigFloat, tolSolve)
        zB = to_big_complex(z)

        u0 = ones(T, 20)
        u0 ./= norm(u0)

        pse, dof, steps, pse_hist, res_hist, conv_hist = inv_lanczos_trace(
            op - zB,
            op_conj - conj(zB),
            u0,
            maxit,
            p_true,
            tolB,
            tolSolveB,
            false,
            "adaptive",
        )

        return pse, dof, steps, pse_hist, res_hist, conv_hist
    end
end

function run_vaadv_truth_trace(; N::Int = 400, p_true::Int = 40, maxit::Int = 1, tol::Union{Nothing, String} = nothing, tolSolve::Union{Nothing, String} = nothing, prec_bits::Int = 128, z::Complex = Complex{BigFloat}(big"2.0", big"0.0"), h::Real = big"0.05", lo::Real = -BigFloat(pi), hi::Real = BigFloat(pi))
    return setprecision(prec_bits) do
        T = Complex{BigFloat}
        loB, hiB = to_big(lo), to_big(hi)
        domB = Interval{BigFloat}(loB, hiB)
        hB = to_big(h)

        a1 = x -> hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))
        a0c = x -> -hB * (BigFloat(2) / BigFloat(3)) * cos(x)
        a1c = x -> -hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))

        f1 = Fun(a1, Chebyshev(loB..hiB))
        f0c = Fun(a0c, Chebyshev(loB..hiB))
        f1c = Fun(a1c, Chebyshev(loB..hiB))

        c_a1 = T.(coefficients(f1))
        c_a0c = T.(coefficients(f0c))
        c_a1c = T.(coefficients(f1c))

        op, op_conj = build_vaadv_ops(T[0], c_a1, c_a0c, c_a1c, T(hB^2), N, domB)

        tolB = tol === nothing ? BigFloat("1e-20") : parse(BigFloat, tol)
        tolSolveB = tolSolve === nothing ? eps(BigFloat) : parse(BigFloat, tolSolve)
        zB = to_big_complex(z)

        u0 = ones(T, 20)
        u0 ./= norm(u0)

        pse, dof, steps, pse_hist, res_hist, conv_hist = inv_lanczos_trace(
            op - zB,
            op_conj - conj(zB),
            u0,
            maxit,
            p_true,
            tolB,
            tolSolveB,
            false,
            "adaptive",
        )

        return pse, dof, steps, pse_hist, res_hist, conv_hist
    end
end

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)

    p_true = parse(Int, get(kv, "p_true", "40"))
    prec = parse(Int, get(kv, "prec", "128"))

    N_wh = parse(Int, get(kv, "N_wh", "300"))
    maxit_wh = parse(Int, get(kv, "maxit_wh", "2"))
    zre_wh_str = get(kv, "zre_wh", "0.7")
    zim_wh_str = get(kv, "zim_wh", "0.0")
    z_wh = Complex{BigFloat}(parse(BigFloat, zre_wh_str), parse(BigFloat, zim_wh_str))
    tol_wh = haskey(kv, "tol_wh") ? get(kv, "tol_wh", "") : nothing
    tolSolve_wh = haskey(kv, "tolSolve_wh") ? get(kv, "tolSolve_wh", "") : nothing
    tol_wh = (tol_wh === nothing || isempty(tol_wh)) ? nothing : tol_wh
    tolSolve_wh = (tolSolve_wh === nothing || isempty(tolSolve_wh)) ? nothing : tolSolve_wh

    N_va = parse(Int, get(kv, "N_va", "400"))
    maxit_va = parse(Int, get(kv, "maxit_va", "1"))
    zre_va_str = get(kv, "zre_va", "0.5")
    zim_va_str = get(kv, "zim_va", "0.0")
    z_va = Complex{BigFloat}(parse(BigFloat, zre_va_str), parse(BigFloat, zim_va_str))
    h_va_str = get(kv, "h_va", "0.05")
    lo_va_str = get(kv, "lo_va", string(-BigFloat(pi)))
    hi_va_str = get(kv, "hi_va", string(BigFloat(pi)))
    h_va = parse(BigFloat, h_va_str)
    lo_va = parse(BigFloat, lo_va_str)
    hi_va = parse(BigFloat, hi_va_str)
    tol_va = haskey(kv, "tol_va") ? get(kv, "tol_va", "") : nothing
    tolSolve_va = haskey(kv, "tolSolve_va") ? get(kv, "tolSolve_va", "") : nothing
    tol_va = (tol_va === nothing || isempty(tol_va)) ? nothing : tol_va
    tolSolve_va = (tolSolve_va === nothing || isempty(tolSolve_va)) ? nothing : tolSolve_va

    println("[config] p_true fixed at ", p_true)

    pse_wh, dof_wh, steps_wh, pse_hist_wh, res_hist_wh, conv_hist_wh = run_wiener_truth_trace(
        N = N_wh,
        p_true = p_true,
        maxit = maxit_wh,
        tol = tol_wh,
        tolSolve = tolSolve_wh,
        prec_bits = prec,
        z = z_wh,
    )

    pse_va, dof_va, steps_va, pse_hist_va, res_hist_va, conv_hist_va = run_vaadv_truth_trace(
        N = N_va,
        p_true = p_true,
        maxit = maxit_va,
        tol = tol_va,
        tolSolve = tolSolve_va,
        prec_bits = prec,
        z = z_va,
        h = h_va,
        lo = lo_va,
        hi = hi_va,
    )

    out_wh = joinpath(@__DIR__, "wiener_truth_invlanczos_residual_trace.csv")
    out_va = joinpath(@__DIR__, "vaadv_truth_invlanczos_residual_trace.csv")
    write_trace_csv(out_wh, pse_hist_wh, res_hist_wh, conv_hist_wh)
    write_trace_csv(out_va, pse_hist_va, res_hist_va, conv_hist_va)

    @printf("[wiener] pse=%.16e steps=%d dof=%d final_res=%.3e csv=%s\n", Float64(pse_wh), steps_wh, dof_wh, Float64(res_hist_wh[end]), out_wh)
    @printf("[vaadv ] pse=%.16e steps=%d dof=%d final_res=%.3e csv=%s\n", Float64(pse_va), steps_va, dof_va, Float64(res_hist_va[end]), out_va)
end

main()
