ENV["GKSwstype"] = "100"

include("../PseSp.jl")
include("remez_chebyshev.jl")

using ApproxFun
using LinearAlgebra
using Printf
using Plots
using LaTeXStrings
using GenericSchur

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

function kernel_coeffs_chebinterp_f64_to_big(n::Int)
    f1 = Fun(x -> exp(x), Chebyshev(-10.0..0.0), n + 1)
    f2 = Fun(x -> exp(-x), Chebyshev(0.0..10.0), n + 1)
    c1_f64 = Float64.(coefficients(f1))
    c2_f64 = Float64.(coefficients(f2))

    u1_f64 = cheb_to_ultra_coeffs(c1_f64, -10.0, 0.0; lambda = 0.5)
    u2_f64 = cheb_to_ultra_coeffs(c2_f64, 0.0, 10.0; lambda = 0.5)

    Tbig = Complex{BigFloat}
    return Tbig.(u1_f64), Tbig.(u2_f64)
end

function kernel_coeffs_remez_f64_to_big(n::Int; maxiter::Int = 40)
    c1, _, _, _ = remez_chebyshev(exp, -10.0, 0.0, n; T = Float64, maxiter = maxiter)
    c2, _, _, _ = remez_chebyshev(x -> exp(-x), 0.0, 10.0, n; T = Float64, maxiter = maxiter)

    u1_f64 = cheb_to_ultra_coeffs(c1, -10.0, 0.0; lambda = 0.5)
    u2_f64 = cheb_to_ultra_coeffs(c2, 0.0, 10.0; lambda = 0.5)

    Tbig = Complex{BigFloat}
    return Tbig.(u1_f64), Tbig.(u2_f64)
end

function kernel_coeffs_truth_big()
    a1, b1 = BigFloat(-10.0), BigFloat(0.0)
    a2, b2 = BigFloat(0.0), BigFloat(10.0)
    f1 = Fun(x -> exp(x), Chebyshev(a1..b1))
    f2 = Fun(x -> exp(-x), Chebyshev(a2..b2))

    c1 = BigFloat.(coefficients(f1))
    c2 = BigFloat.(coefficients(f2))
    u1 = cheb_to_ultra_coeffs(c1, a1, b1; lambda = BigFloat(0.5))
    u2 = cheb_to_ultra_coeffs(c2, a2, b2; lambda = BigFloat(0.5))

    Tbig = Complex{BigFloat}
    return Tbig.(u1), Tbig.(u2)
end

function singlepoint_pse_big(
    coeffs::AbstractVector{Complex{BigFloat}},
    coeffs_conj::AbstractVector{Complex{BigFloat}},
    N::Int,
    z::Complex{BigFloat},
    option::Options,
)
    T = Complex{BigFloat}
    dom_u = Interval{BigFloat}(zero(BigFloat), BigFloat(10.0))

    op = VoltConvOp(coeffs, N, dom_u, 'r', zero(T))
    op_conj = VoltConvOp(coeffs_conj, N, dom_u, 'l', zero(T))

    u0 = ones(T, 20)
    u0 ./= norm(u0)

    U = Matrix{T}(undef, N, option.p)
    worku = Vector{T}(undef, N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, option.p)
    H = zeros(BigFloat, option.p, option.p)

    pse_z, _, steps = invLanczos(
        op - z,
        op_conj - conj(z),
        u0,
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
end

function compute_truth_bigfloat(
    z::Complex{BigFloat};
    N::Int = 1000,
    p::Int = 40,
    maxit::Int = 1,
    tol::BigFloat = 100 * eps(BigFloat),
    tolSolve::BigFloat = eps(BigFloat),
)
    coeffs, coeffs_conj = kernel_coeffs_truth_big()
    option = Options(p, maxit, tol, tolSolve, "adaptive", false)
    pse_true, steps = singlepoint_pse_big(coeffs, coeffs_conj, N, z, option)
    println("[truth] pse_true = ", pse_true, ", steps = ", steps)
    return pse_true
end

function write_csv(path::String, degrees, e_cheb, e_remez)
    open(path, "w") do io
        println(io, "degree,relerr_chebinterp_f64_to_big128,relerr_remez_f64_to_big128")
        for i in eachindex(degrees)
            println(io, string(degrees[i], ",", e_cheb[i], ",", e_remez[i]))
        end
    end
end

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)

    nmin = parse(Int, get(kv, "nmin", "2"))
    nmax = parse(Int, get(kv, "nmax", "32"))
    nstep = parse(Int, get(kv, "nstep", "2"))

    N_eval = parse(Int, get(kv, "N_eval", "1000"))
    N_truth = parse(Int, get(kv, "N_truth", "1000"))
    p_eval = parse(Int, get(kv, "p_eval", "40"))
    maxit_eval = parse(Int, get(kv, "maxit_eval", "1"))
    p_truth = parse(Int, get(kv, "p_truth", "40"))
    maxit_truth = parse(Int, get(kv, "maxit_truth", "1"))
    maxiter_remez = parse(Int, get(kv, "maxiter_remez", "40"))
    prec = parse(Int, get(kv, "prec", "128"))
    dpi = parse(Int, get(kv, "dpi", "300"))
    titlefs = parse(Int, get(kv, "titlefs", "12"))
    guidefs = parse(Int, get(kv, "guidefs", "12"))
    tickfs = parse(Int, get(kv, "tickfs", "12"))
    legendfs = parse(Int, get(kv, "legendfs", "9"))

    zre = parse(BigFloat, get(kv, "zre", "0.7"))
    zim = parse(BigFloat, get(kv, "zim", "0.0"))
    z = Complex{BigFloat}(zre, zim)

    out_png = get(kv, "out_png", joinpath(@__DIR__, "wienerhopf_f64coeff_to_big_relerr.png"))
    out_csv = get(kv, "out_csv", joinpath(@__DIR__, "wienerhopf_f64coeff_to_big_relerr.csv"))

    # User-requested tolerances for f64->big evaluations.
    tol_eval = big"1e-24"
    tolSolve_eval = eps(BigFloat)

    setprecision(prec) do
        degrees = collect(nmin:nstep:nmax)
        option_eval = Options(p_eval, maxit_eval, tol_eval, tolSolve_eval, "adaptive", false)
        pse_true = compute_truth_bigfloat(
            z;
            N = N_truth,
            p = p_truth,
            maxit = maxit_truth,
            tol = 100 * eps(BigFloat),
            tolSolve = eps(BigFloat),
        )

        err_cheb = BigFloat[]
        err_remez = BigFloat[]

        println("\ndegree  relerr_chebinterp_f64->big   relerr_remez_f64->big")
        for n in degrees
            c_cheb, c_cheb_conj = kernel_coeffs_chebinterp_f64_to_big(n)
            c_rem, c_rem_conj = kernel_coeffs_remez_f64_to_big(n; maxiter = maxiter_remez)

            p_cheb, _ = singlepoint_pse_big(c_cheb, c_cheb_conj, N_eval, z, option_eval)
            p_rem, _ = singlepoint_pse_big(c_rem, c_rem_conj, N_eval, z, option_eval)

            e_cheb = abs((p_cheb - pse_true) / pse_true)
            e_rem = abs((p_rem - pse_true) / pse_true)
            push!(err_cheb, e_cheb)
            push!(err_remez, e_rem)

            @printf("%-6d  %-24.6e  %-24.6e\n", n, Float64(e_cheb), Float64(e_rem))
            flush(stdout)
        end

        plt = plot(
            degrees,
            Float64.(err_cheb);
            marker = :circle,
            linewidth = 2,
            dpi = dpi,
            titlefontsize = titlefs,
            guidefontsize = guidefs,
            tickfontsize = tickfs,
            legendfontsize = legendfs,
            yscale = :log10,
            xlabel = L"\mathrm{polynomial\ degree}\;n",
            ylabel = L"\mathrm{relative\ error}",
            label = L"Chebshev\ interpolation\ (double\ precision)",
            title = L"\mathrm{Wiener-Hopf\ operator},\ z=0.7+0.0i",
        )
        plot!(
            plt,
            degrees,
            Float64.(err_remez);
            marker = :diamond,
            linewidth = 2,
            label = L"Remez\ (double\ precision)",
        )
        savefig(plt, out_png)
        write_csv(out_csv, degrees, err_cheb, err_remez)

        println("\nSaved:")
        println(out_png)
        println(out_csv)
    end
end

main()
