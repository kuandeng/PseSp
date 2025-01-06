include("../PseSp.jl")

# preallocated dimension N = n*(n+1)/2 for adaptiveQR solve.
n = 200

T = ComplexF64

# 2D davies operator
rank = 2
coeffs_x = [([0.0, 0.0, 1.0im], [0.0im], [-0.8+0.0im]),  ([1.0+0.0im], )]
coeffs_y = [([1.0+0.0im], ), ([0.0, 0.0, 1.0im], [0.0im], [-0.8+0.0im])]
op = DiffOpInf2D(n, rank, coeffs_x, coeffs_y, 0.0+0.0im)

# operator adjoint
coeffs_x_conj = [([0.0, 0.0, -1.0im], [0.0im], [-0.8+0.0im]),  ([1.0-0.0im], )]
coeffs_y_conj = [([1.0+0.0im], ), ([0.0, 0.0, -1.0im], [0.0im], [-0.8+0.0im])]
op_conj = DiffOpInf2D(n, rank, coeffs_x_conj, coeffs_y_conj, 0.0+0.0im)

# grid points
nptx = 50
npty = 50
ax = [0, 30]
ay = [0, 30]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 1e-8, "adaptive", false)
                                                        
# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

# polt contour
level = -7:1:0
contour(ptx, pty, log10.(pse), levels = level)