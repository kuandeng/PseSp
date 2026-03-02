ENV["GKSwstype"] = "100"

include("remez_chebyshev.jl")

using ApproxFun
using LinearAlgebra
using Plots
using Printf
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

to_big_fun(coeffs::AbstractVector, Sbig) = Fun(Sbig, BigFloat.(coeffs))

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)

    lo = parse(Float64, get(kv, "lo", "-10.0"))
    hi = parse(Float64, get(kv, "hi", "0.0"))
    nmin = parse(Int, get(kv, "nmin", "2"))
    nmax = parse(Int, get(kv, "nmax", "32"))
    nstep = parse(Int, get(kv, "nstep", "2"))
    precision_bits = parse(Int, get(kv, "precision", "128"))
    maxiter = parse(Int, get(kv, "maxiter", "40"))
    dpi = parse(Int, get(kv, "dpi", "300"))
    titlefs = parse(Int, get(kv, "titlefs", "12"))
    guidefs = parse(Int, get(kv, "guidefs", "12"))
    tickfs = parse(Int, get(kv, "tickfs", "12"))
    legendfs = parse(Int, get(kv, "legendfs", "9"))
    # legendpos = Symbol(get(kv, "legendpos", "outertopright"))

    degrees = collect(nmin:nstep:nmax)

    remez64_l2 = Float64[]
    remez64_linf = Float64[]
    remezbig_l2 = Float64[]
    remezbig_linf = Float64[]
    remezbig2f64_l2 = Float64[]
    remezbig2f64_linf = Float64[]
    cheb64_l2 = Float64[]
    cheb64_linf = Float64[]

    setprecision(precision_bits) do
        loB = BigFloat(lo)
        hiB = BigFloat(hi)
        Sbig = Chebyshev(loB..hiB)
        S64 = Chebyshev(lo..hi)

        # BigFloat ApproxFun reference ("ground truth" for this comparison).
        f_ref_big = Fun(x -> exp(x), Sbig)

        @printf(
            "%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
            "deg",
            "remez64_L2",
            "remez64_Linf",
            "remezBig_L2",
            "remezBig_Linf",
            "big2f64_L2",
            "big2f64_Linf",
            "cheb64_L2",
            "cheb64_Linf",
        )

        for n in degrees
            c64, _, _, _ = remez_chebyshev(exp, lo, hi, n; T = Float64, maxiter = maxiter)
            p_remez64_big = to_big_fun(c64, Sbig)
            e_remez64 = f_ref_big - p_remez64_big
            l2_r64 = norm(e_remez64, 2)
            li_r64 = norm(e_remez64, Inf)
            push!(remez64_l2, Float64(l2_r64))
            push!(remez64_linf, Float64(li_r64))

            cbig, _, _, _ = remez_chebyshev(exp, loB, hiB, n; T = BigFloat, maxiter = maxiter)
            p_remezbig = Fun(Sbig, cbig)
            e_remezbig = f_ref_big - p_remezbig
            l2_rbig = norm(e_remezbig, 2)
            li_rbig = norm(e_remezbig, Inf)
            push!(remezbig_l2, Float64(l2_rbig))
            push!(remezbig_linf, Float64(li_rbig))

            cbig64 = Float64.(cbig)
            p_big2f64_big = Fun(Sbig, BigFloat.(cbig64))
            e_big2f64 = f_ref_big - p_big2f64_big
            l2_big2f64 = norm(e_big2f64, 2)
            li_big2f64 = norm(e_big2f64, Inf)
            push!(remezbig2f64_l2, Float64(l2_big2f64))
            push!(remezbig2f64_linf, Float64(li_big2f64))

            f_cheb64 = Fun(x -> exp(x), S64, n + 1)
            p_cheb64_big = to_big_fun(coefficients(f_cheb64), Sbig)
            e_cheb64 = f_ref_big - p_cheb64_big
            l2_c64 = norm(e_cheb64, 2)
            li_c64 = norm(e_cheb64, Inf)
            push!(cheb64_l2, Float64(l2_c64))
            push!(cheb64_linf, Float64(li_c64))

            @printf(
                "%-6d %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n",
                n,
                Float64(l2_r64),
                Float64(li_r64),
                Float64(l2_rbig),
                Float64(li_rbig),
                Float64(l2_big2f64),
                Float64(li_big2f64),
                Float64(l2_c64),
                Float64(li_c64),
            )
            flush(stdout)
        end
    end

    p_l2 = plot(
        degrees,
        remez64_l2;
        label = L"Remez\ (double\ precision)",
        marker = :circle,
        linewidth = 2,
        dpi = dpi,
        titlefontsize = titlefs,
        guidefontsize = guidefs,
        tickfontsize = tickfs,
        legendfontsize = legendfs,
        # legend = legendpos,
        yscale = :log10,
        xlabel = L"\mathrm{polynomial\ degree}\;n",
        ylabel = L"L^2\ \mathrm{error}",
        title = L"f(x)=e^x,\ x\in[-10,0]",
    )
    plot!(p_l2, degrees, remezbig_l2; label = L"Remez\ (quadruple\ precision)", marker = :diamond, linewidth = 2)
    # plot!(p_l2, degrees, remezbig2f64_l2; label = L"Remez\ BigFloat\rightarrow Float64", marker = :star5, linewidth = 2)
    plot!(p_l2, degrees, cheb64_l2; label = L"Chebshev\ interpolation\ (double\ precision)", marker = :utriangle, linewidth = 2)
    out_l2 = joinpath(@__DIR__, "exp_error_L2_remez64_remezBig_big2f64_vs_cheb64_bigref.png")
    out_l2_pdf = joinpath(@__DIR__, "exp_error_L2_remez64_remezBig_big2f64_vs_cheb64_bigref.pdf")
    out_l2_svg = joinpath(@__DIR__, "exp_error_L2_remez64_remezBig_big2f64_vs_cheb64_bigref.svg")
    savefig(p_l2, out_l2)
    savefig(p_l2, out_l2_pdf)
    savefig(p_l2, out_l2_svg)

    p_linf = plot(
        degrees,
        remez64_linf;
        label = L"Remez\ (double\ precision)",
        marker = :circle,
        linewidth = 2,
        dpi = dpi,
        titlefontsize = titlefs,
        guidefontsize = guidefs,
        tickfontsize = tickfs,
        legendfontsize = legendfs,
        # legend = legendpos,
        yscale = :log10,
        xlabel = L"\mathrm{polynomial\ degree}\;n",
        ylabel = L"L^\infty\ \mathrm{error}",
        title = L"f(x)=e^x,\ x\in[-10,0]",
    )
    plot!(p_linf, degrees, remezbig_linf; label = L"Remez\ (quadruple\ precision)", marker = :diamond, linewidth = 2)
    # plot!(p_linf, degrees, remezbig2f64_linf; label = L"Remez\ BigFloat\rightarrow Float64", marker = :star5, linewidth = 2)
    plot!(p_linf, degrees, cheb64_linf; label = L"Chebshev\ interpolation\ (double\ precision)", marker = :utriangle, linewidth = 2)
    out_linf = joinpath(@__DIR__, "exp_error_Linf_remez64_remezBig_big2f64_vs_cheb64_bigref.png")
    out_linf_pdf = joinpath(@__DIR__, "exp_error_Linf_remez64_remezBig_big2f64_vs_cheb64_bigref.pdf")
    out_linf_svg = joinpath(@__DIR__, "exp_error_Linf_remez64_remezBig_big2f64_vs_cheb64_bigref.svg")
    savefig(p_linf, out_linf)
    savefig(p_linf, out_linf_pdf)
    savefig(p_linf, out_linf_svg)

    println("\nSaved:")
    println(out_l2)
    println(out_l2_pdf)
    println(out_l2_svg)
    println(out_linf)
    println(out_linf_pdf)
    println(out_linf_svg)
end

main()
