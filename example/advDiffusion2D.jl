include("../PseSp.jl")

# preallocated dimension N = n*(n+1)/2 for adaptiveQR solve.
n = 100

# L^2[-1,1] \times L^2[-1,1]
dom_x = Interval{Float64}(-1.0, 1.0)
dom_y = Interval{Float64}(-1.0, 1.0)

# 2D advection-diffusion operator
rank = 2
coeffs_x = [([1.0], ), ([0.0], [0.0], [0.05])]
coeffs_y = [([0.0], [-1.0], [0.05]), ([1.0], )]
bcType = "Diri"
bcOrder = 2
op = DiffOp2D(n, 2, 2, coeffs_x, coeffs_y, bcType, bcOrder, dom_x, dom_y, 0.0+0.0im)

# operator conj
coeffs_x_conj = [([1.0], ), ([0.0], [0.0], [0.05])]
coeffs_y_conj = [([0.0], [1.0], [0.05]), ([1.0], )]
op_conj = DiffOp2D(n, 2, 2, coeffs_x_conj, coeffs_y_conj, bcType, bcOrder, dom_x, dom_y, 0.0+0.0im)

# grid ponits
nptx = 50
npty = 50
ax = [-20, 0]
ay = [-15, 15]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 1e-8, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -6:1:0
contour(ptx, pty, log10.(pse), levels = level)
