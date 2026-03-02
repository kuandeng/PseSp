include("../PseSp.jl")

using LinearAlgebra
using Printf
using GenericSchur

# GenericSchur enables eigen/eigvals for BigFloat matrices.
function dominant_eigpair_sym(H::AbstractMatrix{T}) where {T<:AbstractFloat}
    F = eigen(H)
    idx = argmax(F.values)
    return F.values[idx], F.vectors[:, idx]
end

function inv_lanczos_singlepoint!(
    op::Op{T},
    op_conj::Op{T},
    u0::AbstractVector{T},
    p::Int,
    tolSolve::AbstractFloat;
    reOrth::Bool = false,
) where {T<:FloatOrComplex}
    N = op.N
    U = Matrix{T}(undef, N, p)
    worku = Vector{T}(undef, N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, p)
    H = zeros(real(T), p, p)

    u = copy(u0)
    α = zero(real(T))
    β = zero(real(T))
    numax = 0
    jlast = 0

    @inbounds @views for jj in 1:p
        copytoFill0!(U[:, jj], u)
        numax = max(numax, length(u))
        v = adaptiveQrSolve!(op, u, workv, tolSolve)
        w = adaptiveQrSolve!(op_conj, v, workw, tolSolve)

        if jj > 1
            w = axpyDL!(-β, U[1:numax, jj - 1], w, workw)
        end
        α = real(innerproductDL(U[1:numax, jj], w))
        w = axpyDL!(-α, U[1:numax, jj], w, workw)

        if reOrth
            u = simpleReorthogonalize!(U[1:numax, 1:jj], numax, w, length(w), worku, workorth)
        else
            u = w
        end

        β = norm(u)
        H[jj, jj] = α
        if jj < p
            H[jj, jj + 1] = β
            H[jj + 1, jj] = β
        end
        jlast = jj

        if β == 0
            break
        end
        u ./= β
    end

    Hsmall = H[1:jlast, 1:jlast]
    λ, y = dominant_eigpair_sym(Hsmall)
    pse = λ > 0 ? inv(sqrt(λ)) : eps(real(T))
    lanczos_res = (jlast > 0 && β != 0) ? abs(β * y[end]) : zero(real(T))
    return pse, numax, λ, lanczos_res, jlast
end

function parse_kv_args(args::Vector{String})
    parsed = Dict{String, String}()
    for arg in args
        idx = findfirst(==('='), arg)
        idx === nothing && continue
        key = strip(arg[1:idx-1])
        val = strip(arg[idx+1:end])
        !isempty(key) && (parsed[key] = val)
    end
    return parsed
end

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)
    N = parse(Int, get(kv, "N", "1000"))
    p = parse(Int, get(kv, "p", "20"))
    prec = parse(Int, get(kv, "prec", "256"))
    tolSolve_str = get(kv, "tolSolve", "")
    zre_str = get(kv, "zre", "0.7")
    zim_str = get(kv, "zim", "0.0")

    setprecision(prec)
    T = Complex{BigFloat}
    tolSolve = isempty(tolSolve_str) ? eps(BigFloat) : BigFloat(tolSolve_str)
    z = Complex{BigFloat}(BigFloat(zre_str), BigFloat(zim_str))
    dom_u = Interval{BigFloat}(BigFloat("0.0"), BigFloat("10.0"))

    kernel = x -> exp(x)
    kernel_conj = x -> exp(-x)

    t_coeff = @elapsed begin
        global kernelCoeffs = T.(Fun(kernel, Ultraspherical(BigFloat("0.5"), -10..0)).coefficients)
        global kernelCoeffs_conj = T.(Fun(kernel_conj, Ultraspherical(BigFloat("0.5"), 0..10)).coefficients)
    end

    t_op = @elapsed begin
        global op = VoltConvOp(kernelCoeffs, N, dom_u, 'r', T(0))
        global op_conj = VoltConvOp(kernelCoeffs_conj, N, dom_u, 'l', T(0))
    end

    Lz = op - z
    Lz_conj = op_conj - conj(z)

    u0 = ones(T, 20)
    u0 ./= norm(u0)

    t_lanczos = @elapsed begin
        global pse_z, dof, lambda_max, lanczos_res, jlast =
            inv_lanczos_singlepoint!(Lz, Lz_conj, u0, p, tolSolve; reOrth = false)
    end

    println("Wiener-Hopf single-point pseudospectral value (BigFloat)")
    println("z                  = ", z)
    println("precision(bits)    = ", precision(BigFloat))
    println("N (prealloc)       = ", N)
    println("Lanczos size p     = ", p)
    println("tolSolve           = ", tolSolve)
    println("kernel rank        = ", length(kernelCoeffs))
    println("Lanczos steps used = ", jlast)
    println("adaptive dof       = ", dof)
    println("lambda_max         = ", lambda_max)
    println("lanczos residual   = ", lanczos_res)
    println("pse(z)             = ", pse_z)
    @printf("timing(s): coeff=%.3f, build_op=%.3f, lanczos=%.3f\n", t_coeff, t_op, t_lanczos)
end

main()
