include("../PseSp.jl")

using Printf
using LinearAlgebra

function run_case(name::String, op::Op{T}, op_conj::Op{T}, ptx::Vector{Float64}, pty::Vector{Float64}, option::Options) where {T<:FloatOrComplex}
    pse, dof = pseComp(op, op_conj, ptx, pty, option)
    ok = all(isfinite, pse) && all(x -> x > zero(real(T)), pse)
    @printf("%-16s ok=%-5s size=%s dof=[%d,%d] pse=[%.6e, %.6e]\n",
        name,
        string(ok),
        string(size(pse)),
        minimum(dof),
        maximum(dof),
        minimum(real.(pse)),
        maximum(real.(pse)),
    )
    return ok
end

function main()
    T = ComplexF64
    ptx2 = [-1.0, 1.0]
    pty2 = [-1.0, 1.0]
    ok_all = true

    # firstOrder.jl
    let
        N = 50000
        dom = Interval{Float64}(0.0, 2.0)
        op = DiffOp(N, 1, (zeros(T, 1), T[1.0 + 0im]), "DiriR", 1, dom, 0.0 + 0im)
        op_conj = DiffOp(N, 1, (zeros(T, 1), T[-1.0 + 0im]), "DiriL", 1, dom, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)
        ok_all &= run_case("firstOrder", op, op_conj, [-12.0, 0.0], [-4e4, 4e4], option)
    end

    # advDiffusion.jl
    let
        N = 200
        dom = Interval{Float64}(0.0, 1.0)
        op = DiffOp(N, 2, (zeros(T, 1), T[1.0 + 0im], T[0.015 + 0im]), "Diri", 2, dom, 0.0 + 0im)
        op_conj = DiffOp(N, 2, (zeros(T, 1), T[-1.0 + 0im], T[0.015 + 0im]), "Diri", 2, dom, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)
        ok_all &= run_case("advDiffusion", op, op_conj, [-60.0, 20.0], [-40.0, 40.0], option)
    end

    # vaadvDiffusion.jl (Chebyshev coeffs)
    let
        N = 400
        dom = Interval{Float64}(-Float64(pi), Float64(pi))
        h = 1 / 20
        S = Chebyshev(dom.left..dom.right)
        a1 = x -> h * (1 + (2 / 3) * sin(x))
        a0_conj = x -> -h * (2 / 3) * cos(x)
        a1_conj = x -> -h * (1 + (2 / 3) * sin(x))
        coeffs = (zeros(T, 1), T.(coefficients(Fun(a1, S))), T[h^2])
        coeffs_conj = (T.(coefficients(Fun(a0_conj, S))), T.(coefficients(Fun(a1_conj, S))), T[h^2])
        op = DiffOp(N, 2, coeffs, "Periodic", 2, dom, 0.0 + 0im)
        op_conj = DiffOp(N, 2, coeffs_conj, "Periodic", 2, dom, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)
        ok_all &= run_case("vaadvDiffusion", op, op_conj, [-10.0, 2.0], [-6.0, 6.0], option)
    end

    # wave.jl
    let
        N = 500
        dom = Interval{Float64}(0.0, Float64(pi))
        K = 1
        coeffs = [(zeros(T, 1), T[1.0 + 0im]), (zeros(T, 1), T[1.0 + 0im])]
        coeffs_conj = [(zeros(T, 1), T[-1.0 + 0im]), (zeros(T, 1), T[-1.0 + 0im])]
        map = [2, 1]
        op = DiffOpBlock(N, K, 2, map, coeffs, "absorbing", dom, 0.0 + 0im)
        op_conj = DiffOpBlock(N, K, 2, map, coeffs_conj, "absorbing_conj", dom, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)
        ok_all &= run_case("wave", op, op_conj, [-5.0, 3.0], [-4.0, 4.0], option)
    end

    # advDiffusion2D.jl
    let
        n = 100
        domx = Interval{Float64}(-1.0, 1.0)
        domy = Interval{Float64}(-1.0, 1.0)
        coeffs_x = [([1.0],), ([0.0], [0.0], [0.05])]
        coeffs_y = [([0.0], [-1.0], [0.05]), ([1.0],)]
        coeffs_x_conj = [([1.0],), ([0.0], [0.0], [0.05])]
        coeffs_y_conj = [([0.0], [1.0], [0.05]), ([1.0],)]
        op = DiffOp2D(n, 2, 2, coeffs_x, coeffs_y, "Diri", 2, domx, domy, 0.0 + 0im)
        op_conj = DiffOp2D(n, 2, 2, coeffs_x_conj, coeffs_y_conj, "Diri", 2, domx, domy, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 1e-8, "adaptive", false)
        ok_all &= run_case("advDiffusion2D", op, op_conj, [-20.0, 0.0], [-15.0, 15.0], option)
    end

    # davis2D.jl
    let
        n = 200
        coeffs_x = [([0.0, 0.0, 1.0im], [0.0im], [-0.8 + 0.0im]), ([1.0 + 0.0im],)]
        coeffs_y = [([1.0 + 0.0im],), ([0.0, 0.0, 1.0im], [0.0im], [-0.8 + 0.0im])]
        coeffs_x_conj = [([0.0, 0.0, -1.0im], [0.0im], [-0.8 + 0.0im]), ([1.0 - 0.0im],)]
        coeffs_y_conj = [([1.0 + 0.0im],), ([0.0, 0.0, -1.0im], [0.0im], [-0.8 + 0.0im])]
        op = DiffOpInf2D(n, 2, coeffs_x, coeffs_y, 0.0 + 0im)
        op_conj = DiffOpInf2D(n, 2, coeffs_x_conj, coeffs_y_conj, 0.0 + 0im)
        option = Options(20, 1, 1e-3, 1e-8, "adaptive", false)
        ok_all &= run_case("davis2D", op, op_conj, [0.0, 30.0], [0.0, 30.0], option)
    end

    # orrSommerfeld.jl (Chebyshev coeffs)
    let
        N = 1000
        dom = Interval{Float64}(-1.0, 1.0)
        R = 10000
        a = 1.02
        S = Chebyshev(dom.left..dom.right)

        a0 = Fun(x -> 1.0im * a^3 * (1 - x^2) + a^4 / R - 2.0im * a, S)
        a2 = Fun(x -> 1.0im * a * (x^2 - 1) - 2 * a^2 / R, S)
        a4 = Fun(1 / R + 0.0im, S)
        coeffs_L = (a0.coefficients, zeros(T, 1), a2.coefficients, zeros(T, 1), a4.coefficients)
        coeffs_R = (T[-a^2 + 0im], zeros(T, 1), T[1.0 + 0im])
        op_conj = GepDiffOp(N, 4, 2, coeffs_L, coeffs_R, "Diri", 4, dom, 0.0 + 0im, false)

        a0c = Fun(x -> -1.0im * a^3 * (1 - x^2) + a^4 / R, S)
        a1c = Fun(x -> -4.0im * a * x, S)
        a2c = Fun(x -> 1.0im * a * (1 - x^2) - 2 * a^2 / R, S)
        a4c = Fun(1 / R + 0.0im, S)
        coeffs_L_conj = (a0c.coefficients, a1c.coefficients, a2c.coefficients, zeros(T, 1), a4c.coefficients)
        op = GepDiffOp(N, 4, 2, coeffs_L_conj, coeffs_R, "Diri", 4, dom, 0.0 + 0im, true)

        option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)
        ok_all &= run_case("orrSommerfeld", op, op_conj, [-1.0, 0.2], [-1.2, 0.2], option)
    end

    println(ok_all ? "\nAll 2x2 checks passed." : "\n2x2 checks failed.")
end

main()
