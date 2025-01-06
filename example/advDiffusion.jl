include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 200

T = ComplexF64

# L^2[0, 1]
dom = Interval{Float64}(0.0, 1.0)

# advection-diffusion operator
coeffs = (zeros(T, 1), [(1.0+0.0im)], [(0.015+0.0im)])
bcType = "Diri"
bcOrder = 2
K = 2
op = DiffOp(N, K, coeffs, bcType, bcOrder, dom, 0.0+0.0im)

# operator adjoint
coeffs_conj = (zeros(T, 1), [(-1.0+0.0im)], [(0.015+0.0im)])
op_conj = DiffOp(N, K, coeffs_conj, bcType, bcOrder, dom, 0.0+0.0im)

# grid points
nptx = 200
npty = 200
ax = [-60, 20]
ay = [-40, 40]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -11:1:0
contour(ptx, pty, log10.(pse), levels = level)

