include("../PseSp.jl")

using LinearAlgebra
using Random

rand_scalar(::Type{Float64}, rng::AbstractRNG) = 2 * rand(rng) - 1
rand_scalar(::Type{ComplexF64}, rng::AbstractRNG) =
    ComplexF64(rand_scalar(Float64, rng), rand_scalar(Float64, rng))
rand_scalar(::Type{BigFloat}, rng::AbstractRNG) = BigFloat(2 * rand(rng) - 1)
rand_scalar(::Type{Complex{BigFloat}}, rng::AbstractRNG) =
    Complex{BigFloat}(rand_scalar(BigFloat, rng), rand_scalar(BigFloat, rng))

function random_banded_matrix(::Type{T}, N::Int, bu::Int, bl::Int, rng::AbstractRNG) where {T}
    b = BandedMatrix{T}(undef, (N + bl, N), (bl, bu))
    fill!(b, zero(T))
    for j in 1:N
        i1 = max(1, j - bu)
        i2 = min(N + bl, j + bl)
        for i in i1:i2
            b[i, j] = T(0.05) * rand_scalar(T, rng)
        end
        b[j, j] += T(2)
    end
    return b
end

function identity_shift_matrix(::Type{T}, N::Int, bl::Int) where {T}
    b = BandedMatrix{T}(undef, (N + bl, N), (0, 0))
    fill!(b, zero(T))
    for i in 1:N
        b[i, i] = one(T)
    end
    return b
end

function run_baqsv(::Type{T}; N::Int = 120, bu::Int = 4, bl::Int = 3, ny::Int = 20, seed::Int = 7, tolSolve = eps(real(T))) where {T}
    rng = MersenneTwister(seed)
    b = random_banded_matrix(T, N, bu, bl, rng)
    y = [rand_scalar(T, rng) for _ in 1:ny]
    z = T(real(one(real(T))) / 5)

    qr = BandedQrData(b)
    n = baqsv!(qr, z, y, tolSolve)
    x = copy(qr.worky[1:n])
    stop_res = norm(qr.worky[n + 1:max(ny, n + bl)]) / max(norm(y), eps(real(T)))
    return (; n, x, stop_res)
end

function run_gbaqsv(::Type{T}; N::Int = 120, bu::Int = 4, bl::Int = 3, ny::Int = 20, seed::Int = 19, tolSolve = eps(real(T))) where {T}
    rng = MersenneTwister(seed)
    b = random_banded_matrix(T, N, bu, bl, rng)
    b_shift = identity_shift_matrix(T, N, bl)
    y = [rand_scalar(T, rng) for _ in 1:ny]
    z = T(real(one(real(T))) / 5)

    qr = GenBandedQrData(b, b_shift)
    n = gbaqsv!(qr, z, y, tolSolve)
    x = copy(qr.worky[1:n])
    stop_res = norm(qr.worky[n + 1:max(ny, n + bl)]) / max(norm(y), eps(real(T)))
    return (; n, x, stop_res)
end

function validate_tbsv(::Type{T}; n::Int = 24, k::Int = 4, seed::Int = 101) where {T}
    rng = MersenneTwister(seed)
    A_band = zeros(T, k + 1, n)
    A_dense = zeros(T, n, n)

    for j in 1:n
        i1 = max(1, j - k)
        for i in i1:j
            a = rand_scalar(T, rng)
            if i == j
                a += T(2)
            end
            A_band[k + 1 + i - j, j] = a
            A_dense[i, j] = a
        end
    end

    b = [rand_scalar(T, rng) for _ in 1:n]
    x_ref = UpperTriangular(A_dense) \ b
    x = copy(b)
    tbsv!('U', 'N', 'N', k, A_band, x)
    rel = norm(x - x_ref) / max(norm(x_ref), eps(real(T)))
    return rel
end

function check_case(header::String, result64, resultBig; tol64::Float64, tolBig::BigFloat)
    println(header)
    println("  n(Float64)      = ", result64.n)
    println("  n(BigFloat)     = ", resultBig.n)
    println("  stop_res(Float) = ", result64.stop_res, "  target <= ", tol64)
    println("  stop_res(Big)   = ", resultBig.stop_res, "  target <= ", tolBig)

    ok = true
    ok &= result64.n > 0
    ok &= resultBig.n > 0
    ok &= result64.stop_res <= 5 * tol64
    ok &= resultBig.stop_res <= 5 * tolBig
    println("  status          = ", ok ? "PASS" : "FAIL")
    return ok
end

function monotonicity_bigfloat()
    setprecision(256)
    rng_seed = 31
    tols = BigFloat[big"1e-20", big"1e-30", big"1e-40"]
    ns = Int[]
    for tol in tols
        r = run_baqsv(BigFloat; seed = rng_seed, tolSolve = tol)
        push!(ns, r.n)
    end
    is_mono = ns[1] <= ns[2] <= ns[3]
    println("monotonicity (BigFloat baqsv!)")
    println("  tols   = ", tols)
    println("  n(tol) = ", ns)
    println("  status = ", is_mono ? "PASS" : "FAIL")
    return is_mono
end

function main()
    setprecision(256)

    ok = true

    tb_real = validate_tbsv(BigFloat)
    tb_cplx = validate_tbsv(Complex{BigFloat})
    println("tbsv! BigFloat rel err         = ", tb_real)
    println("tbsv! Complex{BigFloat} rel err = ", tb_cplx)
    ok &= tb_real <= big"1e-40"
    ok &= tb_cplx <= big"1e-40"

    tol64 = 1e-12
    tolBig = BigFloat("1e-12")

    ba64 = run_baqsv(Float64; tolSolve = tol64, seed = 7)
    babig = run_baqsv(BigFloat; tolSolve = tolBig, seed = 7)
    ok &= check_case("baqsv real", ba64, babig; tol64 = tol64, tolBig = tolBig)

    bz64 = run_baqsv(ComplexF64; tolSolve = tol64, seed = 11)
    bzbig = run_baqsv(Complex{BigFloat}; tolSolve = tolBig, seed = 11)
    ok &= check_case("baqsv complex", bz64, bzbig; tol64 = tol64, tolBig = tolBig)

    ga64 = run_gbaqsv(Float64; tolSolve = tol64, seed = 19)
    gabig = run_gbaqsv(BigFloat; tolSolve = tolBig, seed = 19)
    ok &= check_case("gbaqsv real", ga64, gabig; tol64 = tol64, tolBig = tolBig)

    gz64 = run_gbaqsv(ComplexF64; tolSolve = tol64, seed = 23)
    gzbig = run_gbaqsv(Complex{BigFloat}; tolSolve = tolBig, seed = 23)
    ok &= check_case("gbaqsv complex", gz64, gzbig; tol64 = tol64, tolBig = tolBig)

    ok &= monotonicity_bigfloat()

    if ok
        println("ALL CHECKS PASSED")
    else
        error("Validation failed")
    end
end

main()
