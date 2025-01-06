include("../PseSp.jl")

T = ComplexF64

# L^2([-1, 1])
dom_u = Interval{Float64}(-1.0, 1.0)

# kernel coefficients under normalizedLengendre basis, results from chebfun2 and cheb2leg
matfile = matopen("./example/kernel_laser.mat", "r")
coeffs_s = read(matfile, "coeffs_s")
coeffs_t = read(matfile, "coeffs_t")

n = max(size(coeffs_s, 1), size(coeffs_t, 1))

# preallocated dimension N
N = 2*n

# for speedup, since unitary transformation do not change the resolvent norm
isSchur = true

# Huygens Fresnel operator
op = FredOp(fredMatrix(coeffs_s, coeffs_t), N, dom_u, 0.0+0.0im, isSchur)

# operator adjoint
op_conj = op'

# grid points
nptx = 200
npty = 200
ax = [-1.2, 1.2]
ay = [-1.2, 1.2]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# the dof during the Lanczos iteration always equals to N.
u0 = ones(T, N)
u0 = u0/norm(u0)

# results and dof
pse, dof = pseComp(op, op_conj, u0, ptx, pty, option)

# polt contour
level = -3:0.5:-1
contour(ptx, pty, log10.(pse), levels = level)
