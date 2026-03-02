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

function build_ops_from_coeffs(
    a0_coeffs::AbstractVector{T},
    a1_coeffs::AbstractVector{T},
    a0c_coeffs::AbstractVector{T},
    a1c_coeffs::AbstractVector{T},
    h2::T,
    N::Int,
    dom::Interval,
    coeff_basis::Symbol = :chebT,
) where {T<:FloatOrComplex}
    coeffs = (a0_coeffs, a1_coeffs, T[h2])
    coeffs_conj = (a0c_coeffs, a1c_coeffs, T[h2])
    K = 2
    bcType = "Periodic"
    bcOrder = 2
    op = DiffOp(N, K, coeffs, bcType, bcOrder, dom, zero(T); coeff_basis = coeff_basis)
    op_conj = DiffOp(N, K, coeffs_conj, bcType, bcOrder, dom, zero(T); coeff_basis = coeff_basis)
    return op, op_conj
end

function singlepoint_pse(
    op::Op{T},
    op_conj::Op{T},
    z::Complex,
    option::Options,
) where {T<:FloatOrComplex}
    Lz = op - T(z)
    Lz_conj = op_conj - conj(T(z))

    u0 = ones(T, 20)
    u0 ./= norm(u0)

    N = op.N
    U = Matrix{T}(undef, N, option.p)
    worku = Vector{T}(undef, N)
    workv = similar(worku)
    workw = similar(worku)
    workorth = Vector{T}(undef, option.p)
    H = zeros(real(T), option.p, option.p)

    pse_z, _, steps = invLanczos(
        Lz,
        Lz_conj,
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

function coeffs_chebinterp_f64_to_big(
    n::Int,
    h::Float64,
    lo::Float64,
    hi::Float64,
)
    a1 = x -> h * (1 + (2 / 3) * sin(x))
    a0c = x -> -h * (2 / 3) * cos(x)
    a1c = x -> -h * (1 + (2 / 3) * sin(x))

    f_a1 = Fun(a1, Chebyshev(lo..hi), n + 1)
    f_a0c = Fun(a0c, Chebyshev(lo..hi), n + 1)
    f_a1c = Fun(a1c, Chebyshev(lo..hi), n + 1)

    Tbig = Complex{BigFloat}
    return Tbig[0], Tbig.(coefficients(f_a1)), Tbig.(coefficients(f_a0c)), Tbig.(coefficients(f_a1c))
end

function coeffs_remez_f64_to_big(
    n::Int,
    h::Float64,
    lo::Float64,
    hi::Float64;
    maxiter::Int = 40,
)
    a1 = x -> h * (1 + (2 / 3) * sin(x))
    a0c = x -> -h * (2 / 3) * cos(x)
    a1c = x -> -h * (1 + (2 / 3) * sin(x))

    c1, _, _, _ = remez_chebyshev(a1, lo, hi, n; T = Float64, maxiter = maxiter)
    c0c, _, _, _ = remez_chebyshev(a0c, lo, hi, n; T = Float64, maxiter = maxiter)
    c1c, _, _, _ = remez_chebyshev(a1c, lo, hi, n; T = Float64, maxiter = maxiter)

    Tbig = Complex{BigFloat}
    return Tbig[0], Tbig.(c1), Tbig.(c0c), Tbig.(c1c)
end

function truth_bigfloat_pse(
    z::Complex{BigFloat},
    h::BigFloat,
    lo::BigFloat,
    hi::BigFloat;
    N::Int = 400,
    p::Int = 60,
    maxit::Int = 1,
    tol::BigFloat = big"1e-20",
    tolSolve::BigFloat = eps(BigFloat),
)
    T = Complex{BigFloat}
    domB = Interval{BigFloat}(lo, hi)

    a1 = x -> h * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))
    a0c = x -> -h * (BigFloat(2) / BigFloat(3)) * cos(x)
    a1c = x -> -h * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))

    f1 = Fun(a1, Chebyshev(lo..hi))
    f0c = Fun(a0c, Chebyshev(lo..hi))
    f1c = Fun(a1c, Chebyshev(lo..hi))
    c_a1 = T.(coefficients(f1))
    c_a0c = T.(coefficients(f0c))
    c_a1c = T.(coefficients(f1c))

    op, op_conj = build_ops_from_coeffs(T[0], c_a1, c_a0c, c_a1c, T(h^2), N, domB, :chebT)
    option = Options(p, maxit, tol, tolSolve, "adaptive", false)
    return singlepoint_pse(op, op_conj, z, option)
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

    N_eval = parse(Int, get(kv, "N_eval", "600"))
    N_truth = parse(Int, get(kv, "N_truth", "600"))
    p_eval = parse(Int, get(kv, "p_eval", "60"))
    maxit_eval = parse(Int, get(kv, "maxit_eval", "1"))
    p_truth = parse(Int, get(kv, "p_truth", get(kv, "p_true", "60")))
    maxit_truth = parse(Int, get(kv, "maxit_truth", "1"))
    maxiter_remez = parse(Int, get(kv, "maxiter_remez", "40"))
    prec = parse(Int, get(kv, "prec", "128"))

    h_str = get(kv, "h", "0.05")
    lo_str = get(kv, "lo", string(-BigFloat(pi)))
    hi_str = get(kv, "hi", string(BigFloat(pi)))
    zre_str = get(kv, "zre", "-2.0")
    zim_str = get(kv, "zim", "-2.0")

    h_f64 = parse(Float64, h_str)
    lo_f64 = parse(Float64, lo_str)
    hi_f64 = parse(Float64, hi_str)

    h_big = parse(BigFloat, h_str)
    lo_big = parse(BigFloat, lo_str)
    hi_big = parse(BigFloat, hi_str)
    z_big = Complex{BigFloat}(parse(BigFloat, zre_str), parse(BigFloat, zim_str))

    dpi = parse(Int, get(kv, "dpi", "300"))
    titlefs = parse(Int, get(kv, "titlefs", "12"))
    guidefs = parse(Int, get(kv, "guidefs", "12"))
    tickfs = parse(Int, get(kv, "tickfs", "12"))
    legendfs = parse(Int, get(kv, "legendfs", "9"))

    out_png = get(kv, "out_png", joinpath(@__DIR__, "vaadv_f64coeff_to_big_relerr.png"))
    out_csv = get(kv, "out_csv", joinpath(@__DIR__, "vaadv_f64coeff_to_big_relerr.csv"))

    # User-requested tolerances.
    tol_eval = big"1e-20"
    tolSolve_eval = eps(BigFloat)

    setprecision(prec) do
        degrees = collect(nmin:nstep:nmax)
        dom_big = Interval{BigFloat}(lo_big, hi_big)
        option_eval = Options(p_eval, maxit_eval, tol_eval, tolSolve_eval, "adaptive", false)

        pse_true, truth_steps = truth_bigfloat_pse(
            z_big,
            h_big,
            lo_big,
            hi_big;
            N = N_truth,
            p = p_truth,
            maxit = maxit_truth,
            tol = big"1e-20",
            tolSolve = eps(BigFloat),
        )
        println("[truth] pse_true = ", pse_true, ", steps = ", truth_steps)

        err_cheb = BigFloat[]
        err_remez = BigFloat[]

        println("\ndegree  relerr_chebinterp_f64->big   relerr_remez_f64->big")
        for n in degrees
            a0A, a1A, a0cA, a1cA = coeffs_chebinterp_f64_to_big(n, h_f64, lo_f64, hi_f64)
            opA, opAc = build_ops_from_coeffs(a0A, a1A, a0cA, a1cA, Complex{BigFloat}(h_big^2), N_eval, dom_big)
            pA, _ = singlepoint_pse(opA, opAc, z_big, option_eval)

            a0R, a1R, a0cR, a1cR = coeffs_remez_f64_to_big(n, h_f64, lo_f64, hi_f64; maxiter = maxiter_remez)
            opR, opRc = build_ops_from_coeffs(a0R, a1R, a0cR, a1cR, Complex{BigFloat}(h_big^2), N_eval, dom_big)
            pR, _ = singlepoint_pse(opR, opRc, z_big, option_eval)

            eA = abs((pA - pse_true) / pse_true)
            eR = abs((pR - pse_true) / pse_true)
            push!(err_cheb, eA)
            push!(err_remez, eR)

            @printf("%-6d  %-24.6e  %-24.6e\n", n, Float64(eA), Float64(eR))
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
            title = L"\mathrm{Advection-Diffusion\ operator},\ z=-2.0-2.0i",
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
