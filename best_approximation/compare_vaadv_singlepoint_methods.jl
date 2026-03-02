ENV["GKSwstype"] = "100"

include("../PseSp.jl")
include("remez_chebyshev.jl")

using ApproxFun
using Printf
using Plots
using LinearAlgebra
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

function singlepoint_pse(op::Op{T}, op_conj::Op{T}, z::Complex, option::Options) where {T<:FloatOrComplex}
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

function coeffs_method_approxfun_float64(n::Int, h::Float64, lo::Float64, hi::Float64)
    a1 = x -> h * (1 + (2 / 3) * sin(x))
    a0c = x -> -h * (2 / 3) * cos(x)
    a1c = x -> -h * (1 + (2 / 3) * sin(x))

    f_a1 = Fun(a1, Chebyshev(lo..hi), n + 1)
    f_a0c = Fun(a0c, Chebyshev(lo..hi), n + 1)
    f_a1c = Fun(a1c, Chebyshev(lo..hi), n + 1)

    return ComplexF64[0.0], ComplexF64.(coefficients(f_a1)), ComplexF64.(coefficients(f_a0c)), ComplexF64.(coefficients(f_a1c))
end

function coeffs_method_remez_float64(n::Int, h::Float64, lo::Float64, hi::Float64; maxiter::Int = 40)
    a1 = x -> h * (1 + (2 / 3) * sin(x))
    a0c = x -> -h * (2 / 3) * cos(x)
    a1c = x -> -h * (1 + (2 / 3) * sin(x))

    c1, _, _, _ = remez_chebyshev(a1, lo, hi, n; T = Float64, maxiter = maxiter)
    c0c, _, _, _ = remez_chebyshev(a0c, lo, hi, n; T = Float64, maxiter = maxiter)
    c1c, _, _, _ = remez_chebyshev(a1c, lo, hi, n; T = Float64, maxiter = maxiter)

    return ComplexF64[0.0], ComplexF64.(c1), ComplexF64.(c0c), ComplexF64.(c1c)
end

# --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
#= function coeffs_method_remez_big2f64(n::Int, h::Float64, lo::Float64, hi::Float64; maxiter::Int = 40, prec_bits::Int = 128)
    c1f, c0cf, c1cf = setprecision(prec_bits) do
        loB, hiB = BigFloat(lo), BigFloat(hi)
        hB = BigFloat(h)
        a1B = x -> hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))
        a0cB = x -> -hB * (BigFloat(2) / BigFloat(3)) * cos(x)
        a1cB = x -> -hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))

        c1, _, _, _ = remez_chebyshev(a1B, loB, hiB, n; T = BigFloat, maxiter = maxiter)
        c0c, _, _, _ = remez_chebyshev(a0cB, loB, hiB, n; T = BigFloat, maxiter = maxiter)
        c1c, _, _, _ = remez_chebyshev(a1cB, loB, hiB, n; T = BigFloat, maxiter = maxiter)

        Float64.(c1), Float64.(c0c), Float64.(c1c)
    end

    return ComplexF64[0.0], ComplexF64.(c1f), ComplexF64.(c0cf), ComplexF64.(c1cf)
end =#

function truth_bigfloat_pse(
    z::Complex,
    h::Real,
    lo::Real,
    hi::Real;
    N::Int = 400,
    p::Int = 20,
    maxit::Int = 1,
    tol::Union{Nothing,String} = nothing,
    tolSolve::Union{Nothing,String} = nothing,
    prec_bits::Int = 128,
)
    return setprecision(prec_bits) do
        T = Complex{BigFloat}
        loB = lo isa BigFloat ? lo : parse(BigFloat, string(lo))
        hiB = hi isa BigFloat ? hi : parse(BigFloat, string(hi))
        domB = Interval{BigFloat}(loB, hiB)
        hB = h isa BigFloat ? h : parse(BigFloat, string(h))

        a1 = x -> hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))
        a0c = x -> -hB * (BigFloat(2) / BigFloat(3)) * cos(x)
        a1c = x -> -hB * (one(BigFloat) + (BigFloat(2) / BigFloat(3)) * sin(x))

        f1 = Fun(a1, Chebyshev(loB..hiB))
        f0c = Fun(a0c, Chebyshev(loB..hiB))
        f1c = Fun(a1c, Chebyshev(loB..hiB))
        c_a1 = T.(coefficients(f1))
        c_a0c = T.(coefficients(f0c))
        c_a1c = T.(coefficients(f1c))

        op, op_conj = build_ops_from_coeffs(T[0], c_a1, c_a0c, c_a1c, T(hB^2), N, domB, :chebT)

        tolB = tol === nothing ? big"1e-20" : parse(BigFloat, tol)
        tolSolveB = tolSolve === nothing ? eps(BigFloat) : parse(BigFloat, tolSolve)
        option = Options(p, maxit, tolB, tolSolveB, "adaptive", false)
        zB = Complex{BigFloat}(
            real(z) isa BigFloat ? real(z) : parse(BigFloat, string(real(z))),
            imag(z) isa BigFloat ? imag(z) : parse(BigFloat, string(imag(z))),
        )
        return singlepoint_pse(op, op_conj, zB, option)
    end
end

function main(args::Vector{String} = ARGS)
    kv = parse_kv_args(args)

    lo_str = get(kv, "lo", string(-BigFloat(pi)))
    hi_str = get(kv, "hi", string(BigFloat(pi)))
    h_str = get(kv, "h", "0.05")
    lo = parse(Float64, lo_str)
    hi = parse(Float64, hi_str)
    h = parse(Float64, h_str)
    lo_truth = parse(BigFloat, lo_str)
    hi_truth = parse(BigFloat, hi_str)
    h_truth = parse(BigFloat, h_str)

    nmin = parse(Int, get(kv, "nmin", "2"))
    nmax = parse(Int, get(kv, "nmax", "32"))
    nstep = parse(Int, get(kv, "nstep", "2"))

    N_eval = parse(Int, get(kv, "N_eval", "400"))
    N_truth = parse(Int, get(kv, "N_truth", "400"))

    p_eval = parse(Int, get(kv, "p_eval", "20"))
    maxit_eval = parse(Int, get(kv, "maxit_eval", "1"))
    tol_eval = parse(Float64, get(kv, "tol_eval", "1e-14"))
    tolSolve_eval = parse(Float64, get(kv, "tolSolve_eval", "2.2e-16"))
    fixed_iter = parse(Int, get(kv, "fixed_iter", "6"))

    p_truth = parse(Int, get(kv, "p_truth", "20"))
    maxit_truth = parse(Int, get(kv, "maxit_truth", "1"))
    tol_truth = haskey(kv, "tol_truth") ? get(kv, "tol_truth", "") : nothing
    tolSolve_truth = haskey(kv, "tolSolve_truth") ? get(kv, "tolSolve_truth", "") : nothing

    prec = parse(Int, get(kv, "prec", "128"))
    maxiter_remez = parse(Int, get(kv, "maxiter_remez", "40"))
    dpi = parse(Int, get(kv, "dpi", "300"))
    titlefs = parse(Int, get(kv, "titlefs", "12"))
    guidefs = parse(Int, get(kv, "guidefs", "12"))
    tickfs = parse(Int, get(kv, "tickfs", "12"))
    legendfs = parse(Int, get(kv, "legendfs", "9"))

    zre_str = get(kv, "zre", "2.0")
    zim_str = get(kv, "zim", "0.0")
    zre = parse(Float64, zre_str)
    zim = parse(Float64, zim_str)
    z = ComplexF64(zre, zim)
    z_truth = Complex{BigFloat}(parse(BigFloat, zre_str), parse(BigFloat, zim_str))

    degrees = collect(nmin:nstep:nmax)

    dom64 = Interval{Float64}(lo, hi)
    option_eval = Options(fixed_iter, 1, tol_eval, tolSolve_eval, "fixed", false)

    pse_true, truth_steps = truth_bigfloat_pse(
        z_truth,
        h_truth,
        lo_truth,
        hi_truth;
        N = N_truth,
        p = p_truth,
        maxit = maxit_truth,
        tol = tol_truth,
        tolSolve = tolSolve_truth,
        prec_bits = prec,
    )

    println("[truth] pse_true = ", pse_true, ", steps = ", truth_steps)

    eA = Float64[]
    eR = Float64[]
    # --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
    # eB = Float64[]

    @printf(
        "\n%-6s %-16s %-16s %-14s\n",
        "deg",
        "relerr_chebinterp",
        "relerr_Remez64",
        "steps(C/R)",
    )

    for n in degrees
        a0A, a1A, a0cA, a1cA = coeffs_method_approxfun_float64(n, h, lo, hi)
        opA, opAc = build_ops_from_coeffs(a0A, a1A, a0cA, a1cA, ComplexF64(h^2), N_eval, dom64)
        pA, sA = singlepoint_pse(opA, opAc, z, option_eval)

        a0R, a1R, a0cR, a1cR = coeffs_method_remez_float64(n, h, lo, hi; maxiter = maxiter_remez)
        opR, opRc = build_ops_from_coeffs(a0R, a1R, a0cR, a1cR, ComplexF64(h^2), N_eval, dom64)
        pR, sR = singlepoint_pse(opR, opRc, z, option_eval)

        # --- disabled: BigFloat->Float64 comparison path (kept for future use) ---
        # a0B, a1B, a0cB, a1cB = coeffs_method_remez_big2f64(n, h, lo, hi; maxiter = maxiter_remez, prec_bits = prec)
        # opB, opBc = build_ops_from_coeffs(a0B, a1B, a0cB, a1cB, ComplexF64(h^2), N_eval, dom64)
        # pB, sB = singlepoint_pse(opB, opBc, z, option_eval)

        erA = Float64(abs((BigFloat(pA) - pse_true) / pse_true))
        erR = Float64(abs((BigFloat(pR) - pse_true) / pse_true))
        # erB = Float64(abs((BigFloat(pB) - pse_true) / pse_true))

        push!(eA, erA)
        push!(eR, erR)
        # push!(eB, erB)

        @printf("%-6d %-16.6e %-16.6e %d/%d\n", n, erA, erR, sA, sR)
        flush(stdout)
    end

    plt = plot(
        degrees,
        eA;
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
        title = L"\mathrm{Advection-diffusion\ operator},\ z=2.0+0.0i",
    )
    plot!(plt, degrees, eR; label = L"Remez(double\ precision)", marker = :diamond, linewidth = 2)
    # plot!(plt, degrees, eB; label = L"Remez\ BigFloat\rightarrow Float64", marker = :star5, linewidth = 2)

    out = joinpath(@__DIR__, "vaadv_singlepoint_z2_relerr_three_methods.png")
    savefig(plt, out)

    println("\nSaved:")
    println(out)
end

main()
# julia -q approx/compare_vaadv_singlepoint_methods.jl nmin=5 nmax=30 nstep=2 fixed_iter=8 N_eval=400 N_truth=400 prec=128 zre=1.0 zim=0.0
