include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 50000

T = ComplexF64

# L^2[0, 2]
dom = Interval{Float64}(0.0, 2.0)

# first order differential operator
bcType = "DiriR"
bcOrder = 1
coeffs = (zeros(T, 1), [(1.0+0.0im)])
K = 1
op = DiffOp(N, K, coeffs, bcType, bcOrder, dom, 0.0+0.0im)

# operator adjoint
bcType = "DiriL"
bcOrder = 1
coeffs_conj = (zeros(T, 1), [(-1.0+0.0im)])
K = 1
op_conj = DiffOp(N, K, coeffs_conj, bcType, bcOrder, dom, 0.0+0.0im)

# grid points
nptx = 50
npty = 50
ax = [-12, 0]
ay = [-40000, 40000]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -8:1:-1
contour(ptx, pty, log10.(pse), levels = level)