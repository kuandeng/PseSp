include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 2000

T = ComplexF64

# L^2[0, 1]
dom = Interval{Float64}(0.0, 1.0)

# low rank represent of kernel function
fx = x->exp(-10*(x-1/3)^2)
coeffs_fx = T.(Fun(fx, Ultraspherical(0.5, 0..1)).coefficients)
coeffs_fy = coeffs_fx
rank = 1
coeffs_x = (coeffs_fx, )
coeffs_y = (coeffs_fy, )

# gutleb-olver operator
op = VoltOp(coeffs_x, coeffs_y, rank, 2000, dom, 'l', 0.0+0.0im)

# operator adjoint
op_conj = op'

# grid points
nptx = 600
npty = 600
ax = [-0.02, 0.08]
ay = [-0.05, 0.05]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -12:1:-2
contour(ptx, pty, log10.(pse), levels = level)