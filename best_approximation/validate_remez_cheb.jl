include("remez_chebyshev.jl")

using Printf

function run_case_float64()
    a, b = -10.0, 0.0
    n = 10
    coeffs, err_inf, xk, st = remez_chebyshev(exp, a, b, n; T = Float64, maxiter = 40)

    xs = collect(range(a, b; length = 4001))
    ytrue = exp.(xs)
    yhat = [eval_cheb_series(coeffs, a, b, x) for x in xs]
    err = yhat .- ytrue
    dx = (b - a) / (length(xs) - 1)

    @printf("[Float64] converged=%s iter=%d rel_gap=%.3e err_inf=%.3e sample_linf=%.3e sample_l2=%.3e\n",
        string(st.converged), st.iter, st.rel_gap, err_inf, maximum(abs.(err)), l2_from_samples(err, dx))

    @assert length(coeffs) == n + 1
    @assert length(xk) == n + 2
    @assert isfinite(err_inf)
    return st, maximum(abs.(err))
end

function run_case_bigfloat()
    n = 10
    a = BigFloat("-10.0")
    b = BigFloat("0.0")
    coeffs, err_inf, xk, st = remez_chebyshev(exp, a, b, n; T = BigFloat, precision = 256, maxiter = 40)

    xs = collect(range(a, b; length = 2001))
    ytrue = exp.(xs)
    yhat = [eval_cheb_series(coeffs, a, b, x) for x in xs]
    err = yhat .- ytrue
    dx = (b - a) / BigFloat(length(xs) - 1)

    @printf("[BigFloat] converged=%s iter=%d rel_gap=%.3e err_inf=%.3e sample_linf=%.3e sample_l2=%.3e\n",
        string(st.converged), st.iter, Float64(st.rel_gap), Float64(err_inf), Float64(maximum(abs.(err))), Float64(l2_from_samples(err, dx)))

    @assert length(coeffs) == n + 1
    @assert length(xk) == n + 2
    @assert isfinite(err_inf)
    return st, maximum(abs.(err))
end

function main()
    st64, linf64 = run_case_float64()
    stbig, linfbig = run_case_bigfloat()

    @assert st64.converged
    @assert stbig.converged
    @assert isfinite(linf64)
    @assert isfinite(linfbig)

    println("ALL CHECKS PASSED")
end

main()
