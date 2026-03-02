using GenericSchur
using LinearAlgebra
using Printf

"""
    remez_chebyshev(f, a, b, n; T=Float64, precision=nothing, tol=nothing,
                    maxiter=30, init_xk=nothing, exchange_method=:full,
                    grid_refine=3)

Polynomial-only Remez algorithm (no rational mode), aligned with the
`minimax.m` polynomial branch.

Returns `(coeffs_cheb, err_inf, xk, status)` where `coeffs_cheb` are
Chebyshev-basis coefficients on `[a,b]`:
`p(x) = sum_{k=0}^n coeffs_cheb[k+1] * T_k((2x-a-b)/(b-a))`.
"""
function remez_chebyshev(
    f::Function,
    a::Real,
    b::Real,
    n::Integer;
    T::Type{<:AbstractFloat} = Float64,
    precision::Union{Nothing,Int} = nothing,
    tol::Union{Nothing,Real} = nothing,
    maxiter::Int = 30,
    init_xk = nothing,
    exchange_method::Symbol = :full,
    grid_refine::Int = 3,
)
    if T == BigFloat && precision !== nothing
        return setprecision(precision) do
            _remez_chebyshev_impl(f, T(a), T(b), Int(n), tol, maxiter, init_xk, exchange_method, grid_refine)
        end
    end
    return _remez_chebyshev_impl(f, T(a), T(b), Int(n), tol, maxiter, init_xk, exchange_method, grid_refine)
end

function default_tol(n::Int)
    return 1e-14 * (n^2 + 10)
end

function _remez_chebyshev_impl(
    f::Function,
    a::T,
    b::T,
    n::Int,
    tol_input::Union{Nothing,Real},
    maxiter::Int,
    init_xk,
    exchange_method::Symbol,
    grid_refine::Int,
) where {T<:AbstractFloat}
    n < 0 && throw(ArgumentError("degree n must be nonnegative"))
    maxiter <= 0 && throw(ArgumentError("maxiter must be positive"))
    grid_refine < 1 && throw(ArgumentError("grid_refine must be >= 1"))
    exchange_method == :full || throw(ArgumentError("only exchange_method=:full is implemented"))

    if a == b
        throw(ArgumentError("interval endpoints must be distinct"))
    elseif a > b
        a, b = b, a
    end

    N = n
    Npts = N + 2
    tolT = tol_input === nothing ? T(default_tol(N)) : T(tol_input)

    xk = init_xk === nothing ? chebpts_interval(a, b, N + 1, T) : T.(collect(init_xk))
    length(xk) == Npts || throw(ArgumentError("init_xk must have length n+2 = $Npts"))
    xk = sort(xk)

    normf = estimate_inf_norm(f, a, b, T)

    iter = 0
    delta = max(normf, eps(T))
    deltamin = T(Inf)
    diffx = one(T)
    xo = copy(xk)
    err = normf
    h = 2 * err + one(T)
    interp_success = true

    best_pk = similar(xk)
    best_xref = copy(xk)
    best_w = barycentric_weights(xk)
    best_err = T(Inf)
    best_h = zero(T)
    best_rel_gap = T(Inf)

    machine_guard = T(1e-14)

    while (abs(abs(h) - abs(err)) / max(abs(err), eps(T)) > tolT) &&
          (iter < maxiter) && (diffx > zero(T)) && interp_success
        if abs(abs(h) - abs(err)) / max(normf, eps(T)) < machine_guard
            break
        end

        fk = T.(f.(xk))
        xk_trial = copy(xk)
        w = barycentric_weights(xk)
        pk, h = compute_trial_function_polynomial(fk, xk, w, N)

        if h == 0
            h = T(1e-19)
        end

        errfun = x -> T(f(x)) - barycentric_eval(x, pk, xk, w)

        xk_new, err_new, interp_success = exchange_step(xk, h, 2, errfun, Npts, a, b, grid_refine)

        # Overshoot fallback exactly as minimax.m polynomial branch.
        if interp_success && err_new / max(normf, eps(T)) > T(1e5)
            xk_new, err_new, interp_success = exchange_step(xo, h, 1, errfun, Npts, a, b, grid_refine)
        end

        if !interp_success
            break
        end

        diffx = maximum(abs.(xo .- xk_new))
        xk = xk_new
        err = err_new
        delta = err - abs(h)

        if delta < deltamin
            deltamin = delta
            best_pk = copy(pk)
            # pk/w belong to the trial reference before exchange.
            best_xref = xk_trial
            best_w = copy(w)
            best_err = err
            best_h = h
            best_rel_gap = abs(abs(h) - abs(err)) / max(abs(err), eps(T))
        end

        xo = copy(xk)
        iter += 1
    end

    if !isfinite(best_err)
        fk = T.(f.(xk))
        w = barycentric_weights(xk)
        best_pk, best_h = compute_trial_function_polynomial(fk, xk, w, N)
        best_xref = copy(xk)
        best_w = copy(w)
        errfun = x -> T(f(x)) - barycentric_eval(x, best_pk, best_xref, best_w)
        best_err = maximum(abs, errfun.(best_xref))
        best_rel_gap = abs(abs(best_h) - abs(best_err)) / max(abs(best_err), eps(T))
        deltamin = best_err - abs(best_h)
    end

    # Convert final barycentric polynomial to Chebyshev coefficients.
    tnodes = chebpts_standard(n, T)
    xnodes = map_t_to_x.(tnodes, Ref(a), Ref(b))
    vals = [barycentric_eval(x, best_pk, best_xref, best_w) for x in xnodes]
    coeffs_cheb = cheb_coeffs_from_values(vals)

    converged = best_rel_gap <= tolT || abs(abs(best_h) - abs(best_err)) / max(normf, eps(T)) < machine_guard
    reason = converged ? :ok : (interp_success ? :maxiter : :insufficient_extrema)

    status = (
        converged = converged,
        iter = iter,
        delta = deltamin,
        h_level = best_h,
        rel_gap = best_rel_gap,
        diffx = diffx,
        reason = reason,
    )

    return coeffs_cheb, best_err, best_xref, status
end

# ------------------------------
# Core polynomial trial function
# ------------------------------

function compute_trial_function_polynomial(
    fk::AbstractVector{T},
    xk::AbstractVector{T},
    w::AbstractVector{T},
    N::Int,
) where {T<:AbstractFloat}
    sigma = alternating_signs(N + 2, T)
    h = dot(w, fk) / dot(w, sigma)
    pk = fk .- h .* sigma
    return pk, h
end

# ------------------------------
# Exchange / extrema
# ------------------------------

function exchange_step(
    xk::Vector{T},
    h::T,
    method::Int,
    errfun::Function,
    Npts::Int,
    a::T,
    b::T,
    grid_refine::Int,
) where {T<:AbstractFloat}
    rr = find_extrema_error(errfun, a, b, xk, grid_refine)
    isempty(rr) && return xk, T(Inf), false

    err_rr = errfun.(rr)

    pos = if method == 1
        [argmax(abs.(err_rr))]
    else
        findall(abs.(err_rr) .>= abs(h))
    end

    r = vcat(rr[pos], xk)
    v = alternating_signs(Npts, T)
    er = vcat(err_rr[pos], v .* h)

    perm = sortperm(r)
    r = r[perm]
    er = er[perm]

    # Delete repeated points exactly as in minimax.m (diff(r) == 0).
    if length(r) > 1
        keep = trues(length(r))
        for i in 1:(length(r) - 1)
            if r[i + 1] - r[i] == 0
                keep[i + 1] = false
            end
        end
        r = r[keep]
        er = er[keep]
    end

    isempty(r) && return xk, T(Inf), false

    s = T[r[1]]
    es = T[er[1]]
    for i in 2:length(r)
        if sign(er[i]) == sign(es[end])
            if abs(er[i]) > abs(es[end])
                s[end] = r[i]
                es[end] = er[i]
            end
        elseif sign(er[i]) != sign(es[end])
            push!(s, r[i])
            push!(es, er[i])
        end
    end

    isempty(es) && return xk, T(Inf), false

    norme, index = findmax(abs.(es))
    d = max(index - Npts + 1, 1)

    if Npts <= length(s)
        return s[d:d + Npts - 1], norme, true
    end

    return s, norme, false
end

function find_extrema_error(errfun::Function, a::T, b::T, xk::Vector{T}, grid_refine::Int) where {T<:AbstractFloat}
    doms = sort(unique(vcat(T[a, b], xk)))
    nn = Int(2^grid_refine)

    rts = T[]
    for k in 1:(length(doms) - 1)
        left = doms[k]
        right = doms[k + 1]
        right <= left && continue

        xx = chebpts_interval(left, right, nn, T)
        vals = errfun.(xx)
        rnow = rootsdiff(vals, (left, right), errfun)
        append!(rts, rnow)
    end

    rts = sort(unique(vcat(T[a, b], rts)))
    return rts
end

# ------------------------------
# rootsdiff aligned with minimax.m
# ------------------------------

function rootsdiff(vals::AbstractVector{T}, dom::Tuple{T,T}, errfun::Function) where {T<:AbstractFloat}
    left, right = dom

    n = length(vals) - 1
    if n < 0
        return T[]
    end

    tol = T(1e-3)
    c = T[one(T)]

    while abs(c[end] / c[1]) > tol && n <= 64
        c = cheb_coeffs_fftlike(vals)

        cU = [c[k + 1] * T(k) for k in 1:length(c)-1]
        if isempty(cU)
            return T[]
        end
        nrm = norm(cU)
        if nrm == 0
            return T[]
        end

        idx = findlast(v -> abs(v) / nrm > T(1e-14), cU)
        if idx === nothing || idx <= 1
            return T[]
        end
        cU = reverse(cU[1:idx])

        if abs(c[end] / c[1]) > tol
            n *= 2
            if n > 64
                break
            end
            mid = (left + right) / T(2)
            rad = (right - left) / T(2)
            ts = [cospi(T(k) / T(n)) for k in n:-1:0]
            vals = [errfun(mid + t * rad) for t in ts]
        end
    end

    # Recompute cU with final vals.
    c = cheb_coeffs_fftlike(vals)
    cU = [c[k + 1] * T(k) for k in 1:length(c)-1]
    isempty(cU) && return T[]

    nrm = norm(cU)
    nrm == 0 && return T[]

    idx = findlast(v -> abs(v) / nrm > T(1e-14), cU)
    if idx === nothing || idx <= 1
        return T[]
    end
    cU = reverse(cU[1:idx])

    eis = T[]
    mon_desc = chebyshevU_to_monomial_desc(cU)
    λ = polyroots_companion(mon_desc)
    imag_tol = T(1e-5)
    for z in λ
        if abs(imag(z)) < imag_tol
            zr = T(real(z))
            if abs(zr) <= one(T) + T(1e-7)
                push!(eis, zr)
            end
        end
    end

    mid = (left + right) / T(2)
    rad = (right - left) / T(2)
    return [mid + ei * rad for ei in eis]
end

# ------------------------------
# Chebyshev utilities
# ------------------------------

function cheb_coeffs_fftlike(vals::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(vals) - 1
    n < 0 && return T[]
    if n == 0
        return T[vals[1]]
    end

    y = vcat(reverse(vals), vals[2:end-1]) # length = 2n
    two_n = 2n
    c = zeros(T, n + 1)
    for j in 0:n
        s = zero(Complex{T})
        for k in 0:(two_n - 1)
            θ = -T(pi) * T(j * k) / T(n)
            s += Complex{T}(y[k + 1]) * cis(θ)
        end
        c[j + 1] = real(s / T(n))
    end
    c[1] /= T(2)
    return c
end

# Convert coefficients in Chebyshev-U basis (highest degree first) to
# monomial basis coefficients (highest degree first).
function chebyshevU_to_monomial_desc(cU_desc::Vector{T}) where {T<:AbstractFloat}
    m = length(cU_desc)
    m == 0 && return T[]
    m == 1 && return T[cU_desc[1]]

    U = Vector{Vector{T}}(undef, m)
    U[1] = T[one(T)]            # U0(x) = 1
    U[2] = T[zero(T), T(2)]     # U1(x) = 2x

    for k in 3:m
        # Uk = 2x*U{k-1} - U{k-2}
        p1 = vcat(T[zero(T)], T(2) .* U[k - 1])
        p2 = U[k - 2]
        len = max(length(p1), length(p2))
        q = zeros(T, len)
        q[1:length(p1)] .+= p1
        q[1:length(p2)] .-= p2
        U[k] = q
    end

    p = zeros(T, m) # ascending powers
    for j in 1:m
        Uk = U[m - j + 1]
        p[1:length(Uk)] .+= cU_desc[j] .* Uk
    end

    lastnz = findlast(!iszero, p)
    lastnz === nothing && return T[zero(T)]
    return reverse(p[1:lastnz])
end

# MATLAB-like roots(c) via companion matrix, where c is descending-order
# monomial coefficients.
function polyroots_companion(c::AbstractVector{T}) where {T<:AbstractFloat}
    isempty(c) && return Complex{T}[]
    all(isfinite, c) || throw(ArgumentError("Non-finite polynomial coefficients"))

    cc = collect(c)
    inz = findall(!iszero, cc)
    isempty(inz) && return Complex{T}[]

    n = length(cc)
    cc = cc[inz[1]:inz[end]]
    r = Complex{T}[Complex{T}(zero(T)) for _ in 1:(n - inz[end])]

    if length(cc) > 1
        d = cc[2:end] ./ cc[1]
        while any(isinf, d)
            cc = cc[2:end]
            length(cc) <= 1 && return r
            d = cc[2:end] ./ cc[1]
        end
    end

    m = length(cc)
    if m > 1
        A = zeros(T, m - 1, m - 1)
        A[1, :] .= -d
        for i in 1:(m - 2)
            A[i + 1, i] = one(T)
        end
        append!(r, Complex{T}.(eigen(A).values))
    end
    return r
end

function cheb_coeffs_from_values(vals::Vector{T}) where {T<:AbstractFloat}
    n = length(vals) - 1
    n < 0 && return T[]
    if n == 0
        return T[vals[1]]
    end

    t = chebpts_standard(n, T)
    V = cheb_vandermonde(t, n)
    return V \ vals
end

function cheb_vandermonde(t::Vector{T}, n::Int) where {T<:AbstractFloat}
    V = Matrix{T}(undef, length(t), n + 1)
    @inbounds for i in eachindex(t)
        V[i, 1] = one(T)
        if n >= 1
            V[i, 2] = t[i]
        end
        for k in 2:n
            V[i, k + 1] = T(2) * t[i] * V[i, k] - V[i, k - 1]
        end
    end
    return V
end

function chebpts_standard(n::Int, ::Type{T}) where {T<:AbstractFloat}
    n == 0 && return T[one(T)]
    return [cospi(T(k) / T(n)) for k in n:-1:0]
end

function chebpts_interval(a::T, b::T, n::Int, ::Type{T}) where {T<:AbstractFloat}
    t = chebpts_standard(n, T)
    return map_t_to_x.(t, Ref(a), Ref(b))
end

map_t_to_x(t::T, a::T, b::T) where {T<:AbstractFloat} = (a + b) / T(2) + t * (b - a) / T(2)

function alternating_signs(n::Int, ::Type{T}) where {T<:AbstractFloat}
    v = ones(T, n)
    for i in 2:2:n
        v[i] = -one(T)
    end
    return v
end

function barycentric_weights(xk::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(xk)
    w = ones(T, n)
    for j in 1:n
        denom = one(T)
        xj = xk[j]
        for k in 1:n
            if k != j
                denom *= (xj - xk[k])
            end
        end
        w[j] = inv(denom)
    end
    s = maximum(abs, w)
    s != 0 && (w ./= s)
    return w
end

function barycentric_eval(x::T, yk::AbstractVector{T}, xk::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    num = zero(T)
    den = zero(T)
    for j in eachindex(xk)
        t = w[j] / (x - xk[j])
        num += t * yk[j]
        den += t
    end
    fx = num / den

    # Chebfun-style NaN cleanup at support points (inf/inf -> exact node value).
    if isnan(fx)
        for j in eachindex(xk)
            if x == xk[j]
                return yk[j]
            end
        end
    end
    return fx
end

function eval_cheb_series(c::AbstractVector{T}, a::T, b::T, x::T) where {T<:AbstractFloat}
    t = (T(2) * x - (a + b)) / (b - a)
    n = length(c) - 1
    n < 0 && return zero(T)

    b1 = zero(T)
    b2 = zero(T)
    for k in n:-1:1
        b0 = c[k + 1] + T(2) * t * b1 - b2
        b2 = b1
        b1 = b0
    end
    return c[1] + t * b1 - b2
end

function estimate_inf_norm(f::Function, a::T, b::T, ::Type{T}) where {T<:AbstractFloat}
    xs = range(a, b; length = 5000)
    return maximum(abs.(T.(f.(xs))))
end

function l2_from_samples(err::AbstractVector{T}, dx::T) where {T<:AbstractFloat}
    length(err) < 2 && return zero(T)
    s = T(0.5) * err[1]^2 + T(0.5) * err[end]^2 + sum(abs2, @view(err[2:end-1]))
    return sqrt(dx * s)
end

function demo()
    a, b = -10.0, 0.0
    xs = collect(range(a, b; length = 4001))
    dx = (b - a) / (length(xs) - 1)
    ytrue = exp.(xs)

    @printf("%-8s %-14s %-14s %-10s %-10s\n", "deg", "L2", "Linf", "conv", "rel_gap")
    for n in 2:2:30
        c, _, _, st = remez_chebyshev(exp, a, b, n; T = Float64, maxiter = 40)
        y = [eval_cheb_series(c, a, b, x) for x in xs]
        e = y .- ytrue
        @printf("%-8d %-14.6e %-14.6e %-10s %-10.3e\n", n, l2_from_samples(e, dx), maximum(abs.(e)), string(st.converged), st.rel_gap)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
