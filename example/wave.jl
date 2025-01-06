include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 500

T = ComplexF64

# L^2[0, pi]
dom = Interval{Float64}(0.0, Float64.(pi))

# wave operator
K = 1
coeffs = [(zeros(T, 1), [(1.0+0.0im)]), (zeros(T, 1), [(1.0+0.0im)])]
blockSize = 2
map = [2, 1]
bcType = "absorbing"
op = DiffOpBlock(N, K, blockSize, map, coeffs, bcType, dom, 0.0+0.0im);

# operator adjoint
coeffs_conj = [(zeros(T, 1), [(-1.0+0.0im)]), (zeros(T, 1), [(-1.0+0.0im)])]
bcType_conj = "absorbing_conj"
op_conj = DiffOpBlock(N, K, blockSize, map, coeffs_conj, bcType_conj, dom, 0.0+0.0im);

# grid points
nptx = 200
npty = 200
ax = [-5, 3]
ay = [-4, 4]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -1.0:0.2:1.0
contour(ptx, pty, log10.(pse), levels = level)

