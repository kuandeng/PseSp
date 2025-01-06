include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 30000

T = ComplexF64

# L^2([0, 10])
dom_u = Interval{Float64}(0.0, 10.0)

# kernel function
kernel = x->exp(x)
kernelCoeffs = T.(Fun(kernel, Ultraspherical(0.5, -10..0)).coefficients)

# wiener-hopf operator
side = 'r'
op = VoltConvOp(kernelCoeffs, N, dom_u, side, T(0.0))

# convolution kernel adjoint
kernel_conj = x->exp(-x) 
kernelCoeffs_conj = T.(Fun(kernel_conj, Ultraspherical(0.5, 0..10)).coefficients)

# operator adjoint
side = 'l'
op_conj = VoltConvOp(kernelCoeffs_conj, N, dom_u, side, T(0.0))

# grid points
nptx = 600
npty = 600
ax = [-0.1, 0.8]
ay = [-0.5, 0.5]

ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -13:1:-1
contour(ptx, pty, log10.(pse), levels = level)
