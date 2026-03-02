ENV["GKSwstype"] = "100"

include("../PseSp.jl")
include("remez_chebyshev.jl")

using ApproxFun
using Printf
using Plots
using LinearAlgebra
using Random
using LaTeXStrings

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

function kernel_coeffs_approxfun_fixedn(n::Int)
    f1 = Fun(x -> exp(x), Chebyshev(-10.0..0.0), n + 1)
    f2 = Fun(x -> exp(-x), Chebyshev(0.0..10.0), n + 1)

    c1 = Float64.(coefficients(f1))
    c2 = Float64.(coefficients(f2))

    u1 = cheb_to_ultra_coeffs(c1, -10.0, 0.0; lambda = 0.5)
    u2 = cheb_to_ultra_coeffs(c2, 0.0, 10.0; lambda = 0.5)

    return ComplexF64.(u1), ComplexF64.(u2)
end

function kernel_coeffs_remez64(n::Int; maxiter::Int = 40)
    c1, _, _, _ = remez_chebyshev(exp, -10.0, 0.0, n; T = Float64, maxiter = maxiter)
    c2, _, _, _ = remez_chebyshev(x -> exp(-x), 0.0, 10.0, n; T = Float64, maxiter = maxiter)

    u1 = cheb_to_ultra_coeffs(c1, -10.0, 0.0; lambda = 0.5)
    u2 = cheb_to_ultra_coeffs(c2, 0.0, 10.0; lambda = 0.5)

    return ComplexF64.(u1), ComplexF64.(u2)
end

# --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
#= function kernel_coeffs_remez_big2f64(n::Int; maxiter::Int = 40, prec_bits::Int = 128)
    c1f64, c2f64 = setprecision(prec_bits) do
        lo1, hi1 = BigFloat(-10.0), BigFloat(0.0)
        lo2, hi2 = BigFloat(0.0), BigFloat(10.0)
        c1_big, _, _, _ = remez_chebyshev(exp, lo1, hi1, n; T = BigFloat, maxiter = maxiter)
        c2_big, _, _, _ = remez_chebyshev(x -> exp(-x), lo2, hi2, n; T = BigFloat, maxiter = maxiter)
        Float64.(c1_big), Float64.(c2_big)
    end

    u1 = cheb_to_ultra_coeffs(c1f64, -10.0, 0.0; lambda = 0.5)
    u2 = cheb_to_ultra_coeffs(c2f64, 0.0, 10.0; lambda = 0.5)

    return ComplexF64.(u1), ComplexF64.(u2)
end =#

function singlepoint_pse(
    coeffs::AbstractVector{T},
    coeffs_conj::AbstractVector{T},
    N::Int,
    z::Complex,
    option::Options,
    u0::Union{Nothing,AbstractVector{T}} = nothing,
    return_steps::Bool = false,
    return_hist::Bool = false,
) where {T<:FloatOrComplex}
    dom_u = Interval{real(T)}(zero(real(T)), real(T(10)))

    op = VoltConvOp(coeffs, N, dom_u, 'r', T(0))
    op_conj = VoltConvOp(coeffs_conj, N, dom_u, 'l', T(0))

    Lz = op - T(z)
    Lz_conj = op_conj - conj(T(z))

    u = if u0 === nothing
        v = ones(T, 20)
        v ./= norm(v)
        v
    else
        v = copy(u0)
        v ./= norm(v)
        v
    end

    U = Matrix{T}(undef, N, option.p)
    worku = Vector{T}(undef, N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, option.p)
    H = zeros(real(T), option.p, option.p)

    if return_steps || return_hist
        if return_steps && return_hist
            pse_z, _, steps, hist = invLanczos(
                Lz,
                Lz_conj,
                u,
                option.maxit,
                option.p,
                option.tol,
                option.tolSolve,
                option.reOrth,
                option.stopCrit,
                U,
                worku,
                workv,
                workw,
                workorth,
                H;
                return_meta = true,
                return_hist = true,
            )
            return pse_z, steps, hist
        elseif return_steps
            pse_z, _, steps = invLanczos(
                Lz,
                Lz_conj,
                u,
                option.maxit,
                option.p,
                option.tol,
                option.tolSolve,
                option.reOrth,
                option.stopCrit,
                U,
                worku,
                workv,
                workw,
                workorth,
                H;
                return_meta = true,
            )
            return pse_z, steps
        else
            pse_z, _, hist = invLanczos(
                Lz,
                Lz_conj,
                u,
                option.maxit,
                option.p,
                option.tol,
                option.tolSolve,
                option.reOrth,
                option.stopCrit,
                U,
                worku,
                workv,
                workw,
                workorth,
                H;
                return_hist = true,
            )
            return pse_z, hist
        end
    else
        pse_z, _ = invLanczos(
            Lz,
            Lz_conj,
            u,
            option.maxit,
            option.p,
            option.tol,
            option.tolSolve,
            option.reOrth,
            option.stopCrit,
            U,
            worku,
            workv,
            workw,
            workorth,
            H,
        )
        return pse_z
    end
end

function singlepoint_pse(
    coeffs::AbstractVector{T},
    coeffs_conj::AbstractVector{T},
    N::Int,
    z::Complex,
    option::Options;
    u0::Union{Nothing,AbstractVector{T}} = nothing,
    return_steps::Bool = false,
    return_hist::Bool = false,
) where {T<:FloatOrComplex}
    return singlepoint_pse(coeffs, coeffs_conj, N, z, option, u0, return_steps, return_hist)
end

function parse_iter_window(s::String)
    if isempty(s)
        return Int[]
    end
    m = match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", s)
    m === nothing && error("avg_iters must be like 4:6")
    a = parse(Int, m.captures[1])
    b = parse(Int, m.captures[2])
    a <= b || error("avg_iters start must be <= end")
    return collect(a:b)
end

function random_u0(::Type{T}, m::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    v = randn(rng, T, m)
    v ./= norm(v)
    return v
end

function random_u0(::Type{Complex{T}}, m::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    v = randn(rng, T, m) .+ im .* randn(rng, T, m)
    v ./= norm(v)
    return v
end

function compute_truth_bigfloat(
    z::ComplexF64;
    N::Int = 300,
    p::Int = 30,
    maxit::Int = 2,
    tol::Union{Nothing,String} = nothing,
    tolSolve::Union{Nothing,String} = nothing,
    prec_bits::Int = 128,
)
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

        tolB = tol === nothing ? 100 * eps(BigFloat) : parse(BigFloat, tol)
        tolSolveB = tolSolve === nothing ? eps(BigFloat) : parse(BigFloat, tolSolve)
        option = Options(p, maxit, tolB, tolSolveB, "adaptive", false)
        zB = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))

        pse_true, steps = singlepoint_pse(coeffs, coeffs_conj, N, zB, option, nothing, true)

        println("[truth] precision(bits) = ", precision(BigFloat))
        println("[truth] ultra lengths   = ", length(u1), " / ", length(u2))
        println("[truth] pse_true         = ", pse_true)
        println("[truth] steps            = ", steps)

        pse_true
    end
end

# function write_csv(path::String, degrees, e1, e2, e3)
#     open(path, "w") do io
#         println(io, "degree,relerr_approxfun_f64,relerr_remez_f64,relerr_remez_big128_to_f64")
#         for i in eachindex(degrees)
#             println(io, string(degrees[i], ",", e1[i], ",", e2[i], ",", e3[i]))
#         end
#     end
# end

function write_csv(path::String, degrees, e1, e2)
    open(path, "w") do io
        println(io, "degree,relerr_chebinterp_f64,relerr_remez_f64")
        for i in eachindex(degrees)
            println(io, string(degrees[i], ",", e1[i], ",", e2[i]))
        end
    end
end

# function write_pse_csv(path::String, degrees, pse_true, pA, pR, pB, sA, sR, sB)
#     open(path, "w") do io
#         println(io, "degree,pse_true,pse_approxfun_f64,pse_remez_f64,pse_remez_big128_to_f64,avg_steps_A,avg_steps_R,avg_steps_B")
#         for i in eachindex(degrees)
#             println(
#                 io,
#                 string(
#                     degrees[i], ",",
#                     pse_true, ",",
#                     pA[i], ",",
#                     pR[i], ",",
#                     pB[i], ",",
#                     sA[i], ",",
#                     sR[i], ",",
#                     sB[i],
#                 ),
#             )
#         end
#     end
# end

function write_pse_csv(path::String, degrees, pse_true, pA, pR, sA, sR)
    open(path, "w") do io
        println(io, "degree,pse_true,pse_chebinterp_f64,pse_remez_f64,avg_steps_chebinterp,avg_steps_remez")
        for i in eachindex(degrees)
            println(
                io,
                string(
                    degrees[i], ",",
                    pse_true, ",",
                    pA[i], ",",
                    pR[i], ",",
                    sA[i], ",",
                    sR[i],
                ),
            )
        end
    end
end

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)

    nmin = parse(Int, get(kv, "nmin", "5"))
    nmax = parse(Int, get(kv, "nmax", "32"))
    N_eval = parse(Int, get(kv, "N_eval", "300"))
    N_truth = parse(Int, get(kv, "N_truth", "300"))

    p_eval = parse(Int, get(kv, "p_eval", "20"))
    maxit_eval = parse(Int, get(kv, "maxit_eval", "1"))
    tol_eval = parse(Float64, get(kv, "tol_eval", "1e-14"))
    tolSolve_eval = parse(Float64, get(kv, "tolSolve_eval", "2.2e-16"))

    p_truth = parse(Int, get(kv, "p_truth", "20"))
    maxit_truth = parse(Int, get(kv, "maxit_truth", "1"))
    tol_truth = haskey(kv, "tol_truth") ? get(kv, "tol_truth", "") : nothing
    tolSolve_truth = haskey(kv, "tolSolve_truth") ? get(kv, "tolSolve_truth", "") : nothing

    maxiter_remez = parse(Int, get(kv, "maxiter_remez", "40"))
    prec = parse(Int, get(kv, "prec", "128"))
    dpi = parse(Int, get(kv, "dpi", "300"))
    titlefs = parse(Int, get(kv, "titlefs", "12"))
    guidefs = parse(Int, get(kv, "guidefs", "12"))
    tickfs = parse(Int, get(kv, "tickfs", "12"))
    legendfs = parse(Int, get(kv, "legendfs", "9"))
    nrepeat_eval = parse(Int, get(kv, "nrepeat_eval", "5"))
    seed = parse(Int, get(kv, "seed", "1234"))
    fixed_iter = parse(Int, get(kv, "fixed_iter", "0"))
    avg_iters = parse_iter_window(get(kv, "avg_iters", ""))

    zre = parse(Float64, get(kv, "zre", "0.7"))
    zim = parse(Float64, get(kv, "zim", "0.0"))
    z = ComplexF64(zre, zim)

    out_png = get(kv, "out_png", joinpath(@__DIR__, "wienerhopf_z0p7_relerr_three_methods.png"))
    out_csv = get(kv, "out_csv", joinpath(@__DIR__, "wienerhopf_z0p7_relerr_three_methods.csv"))
    out_pse_csv = get(kv, "out_pse_csv", joinpath(@__DIR__, "wienerhopf_z0p7_pse_values.csv"))

    degrees = collect(nmin:nmax)
    option_eval = fixed_iter > 0 ?
        Options(fixed_iter, 1, tol_eval, tolSolve_eval, "fixed", false) :
        Options(p_eval, maxit_eval, tol_eval, tolSolve_eval, "adaptive", false)
    rng = MersenneTwister(seed)

    pse_true = compute_truth_bigfloat(
        z;
        N = N_truth,
        p = p_truth,
        maxit = maxit_truth,
        tol = tol_truth,
        tolSolve = tolSolve_truth,
        prec_bits = prec,
    )

    err_a = BigFloat[]
    err_r64 = BigFloat[]
    # --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
    # err_rbig2f64 = BigFloat[]
    pse_a = BigFloat[]
    pse_r64 = BigFloat[]
    # pse_rbig2f64 = BigFloat[]
    step_a = Float64[]
    step_r64 = Float64[]
    # step_rbig2f64 = Float64[]

    println("\ndegree  relerr_chebinterp   relerr_remez64      steps(C/R)")
    for n in degrees
        cA, cA_conj = kernel_coeffs_approxfun_fixedn(n)
        cR, cR_conj = kernel_coeffs_remez64(n; maxiter = maxiter_remez)
        # --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
        # cB, cB_conj = kernel_coeffs_remez_big2f64(n; maxiter = maxiter_remez, prec_bits = prec)

        if !(length(cA) == n + 1 == length(cR) == length(cA_conj) == length(cR_conj))
            error("coefficient length mismatch at n=$n")
        end

        pA_sum = 0.0
        pR_sum = 0.0
        # pB_sum = 0.0
        sA_sum = 0
        sR_sum = 0
        # sB_sum = 0
        for _ in 1:nrepeat_eval
            u0 = random_u0(ComplexF64, 40, rng)
            if isempty(avg_iters)
                pA_i, sA_i = singlepoint_pse(cA, cA_conj, N_eval, z, option_eval, u0, true)
                pR_i, sR_i = singlepoint_pse(cR, cR_conj, N_eval, z, option_eval, u0, true)
                # pB_i, sB_i = singlepoint_pse(cB, cB_conj, N_eval, z, option_eval, u0, true)
            else
                _, sA_i, hA = singlepoint_pse(cA, cA_conj, N_eval, z, option_eval, u0, true, true)
                _, sR_i, hR = singlepoint_pse(cR, cR_conj, N_eval, z, option_eval, u0, true, true)
                # _, sB_i, hB = singlepoint_pse(cB, cB_conj, N_eval, z, option_eval, u0, true, true)
                idxA = [k for k in avg_iters if k <= length(hA)]
                idxR = [k for k in avg_iters if k <= length(hR)]
                # idxB = [k for k in avg_iters if k <= length(hB)]
                isempty(idxA) && error("avg_iters out of range for method A at n=$n")
                isempty(idxR) && error("avg_iters out of range for method R at n=$n")
                # isempty(idxB) && error("avg_iters out of range for method B at n=$n")
                pA_i = sum(hA[idxA]) / length(idxA)
                pR_i = sum(hR[idxR]) / length(idxR)
                # pB_i = sum(hB[idxB]) / length(idxB)
            end
            pA_sum += pA_i
            pR_sum += pR_i
            # pB_sum += pB_i
            sA_sum += sA_i
            sR_sum += sR_i
            # sB_sum += sB_i
        end
        pA = pA_sum / nrepeat_eval
        pR = pR_sum / nrepeat_eval
        # pB = pB_sum / nrepeat_eval
        sA_avg = sA_sum / nrepeat_eval
        sR_avg = sR_sum / nrepeat_eval
        # sB_avg = sB_sum / nrepeat_eval
        push!(pse_a, BigFloat(pA))
        push!(pse_r64, BigFloat(pR))
        # push!(pse_rbig2f64, BigFloat(pB))
        push!(step_a, sA_avg)
        push!(step_r64, sR_avg)
        # push!(step_rbig2f64, sB_avg)

        eA = abs((BigFloat(pA) - pse_true) / pse_true)
        eR = abs((BigFloat(pR) - pse_true) / pse_true)
        # eB = abs((BigFloat(pB) - pse_true) / pse_true)

        push!(err_a, eA)
        push!(err_r64, eR)
        # push!(err_rbig2f64, eB)

        @printf(
            "%-6d  %-16.6e  %-16.6e  %.2f/%.2f\n",
            n,
            Float64(eA),
            Float64(eR),
            sA_avg,
            sR_avg,
        )
        flush(stdout)
    end

    err_a_f64 = Float64.(err_a)
    err_r64_f64 = Float64.(err_r64)
    # err_rbig2f64_f64 = Float64.(err_rbig2f64)

    plt = plot(
        degrees,
        err_a_f64;
        label = L"Chebshev\ interpolation(double\ precision)",
        marker = :circle,
        linewidth = 2,
        dpi = dpi,
        titlefontsize = titlefs,
        guidefontsize = guidefs,
        tickfontsize = tickfs,
        legendfontsize = legendfs,
        yscale = :log10,
        xlabel = L"\mathrm{Polynomial\ degree}\;n",
        ylabel = L"\mathrm{Relative\ error}",
        title = L"\mathrm{Wiener-Hopf\ operator},\ z=0.7+0.0i",
    )
    plot!(plt, degrees, err_r64_f64; label = L"Remez(double\ precision)", marker = :diamond, linewidth = 2)
    # plot!(plt, degrees, err_rbig2f64_f64; label = L"Remez\ BigFloat\rightarrow Float64", marker = :star5, linewidth = 2)
    savefig(plt, out_png)

    write_csv(out_csv, degrees, err_a, err_r64)
    write_pse_csv(out_pse_csv, degrees, pse_true, pse_a, pse_r64, step_a, step_r64)

    println("\nSaved:")
    println(out_png)
    println(out_csv)
    println(out_pse_csv)
end

main()
# julia -q approx/compare_wienerhopf_singlepoint_methods.jl fixed_iter=5 nmin=5 nmax=32 N_eval=300 N_truth=300 prec=128 p_eval=20 maxit_eval=1
