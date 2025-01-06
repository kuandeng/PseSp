include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 1000

T = ComplexF64

# L^2([-1, 1])
dom = Interval{Float64}(-1.0, 1.0)


R = 10000;
a = 1.02;
a0 = Fun(x->1.0im*a^3*(1-x^2)+a^4/R-2.0im*a, Ultraspherical(0.5))
a2 = Fun(x->1.0im*a*(x^2-1)-2*a^2/R, Ultraspherical(0.5))
a4 = Fun(1/R+0.0im, Ultraspherical(0.5))
coeffs_L = (a0.coefficients, zeros(T, 1), a2.coefficients, zeros(T, 1), a4.coefficients)
K_L = 4
bcType_L = "Diri"
bcOrder_L = 4


coeffs_R = ([-a^2+0.0im], zeros(T, 1), [1.0+0.0im])
K_R = 2
# orr-sommerfeld operator
op_conj = GepDiffOp(N, K_L, K_R, coeffs_L, coeffs_R, bcType_L, bcOrder_L, dom, 0.0+0.0im, false)


# operator adjoint
a0_conj = Fun(x->-1.0im*a^3*(1-x^2)+a^4/R, Ultraspherical(0.5))
a1_conj = Fun(x->-4.0im*a*x, Ultraspherical(0.5))
a2_conj = Fun(x->1.0im*a*(1-x^2)-2a^2/R, Ultraspherical(0.5))
a4_conj = Fun(1/R+0.0im, Ultraspherical(0.5))

coeffs_L_conj = (a0_conj.coefficients, a1_conj.coefficients, a2_conj.coefficients, zeros(T, 1), a4_conj.coefficients)

coeffs_R_conj = coeffs_R

op = GepDiffOp(N, K_L, K_R, coeffs_L_conj, coeffs_R_conj, bcType_L, bcOrder_L, dom, 0.0+0.0im, true)

# grid points
nptx = 400
npty = 400
ax = [-1.0, 0.2]
ay = [-1.2, 0.2]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))


# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -8:1:0
contour(ptx, pty, log10.(pse), levels = level)



