include("../PseSp.jl")

T = ComplexF64

# L^2([-1, 1])
dom_u = Interval{Float64}(-1.0, 1.0)

# kernel coefficients under Lengendre basis
kernel =  x->sqrt(16.0im)*exp(-16.0im*pi*x^2)
kernelCoeffs = T.(Fun(kernel, Ultraspherical(0.5, -2..2)).coefficients) 

# preallocated dimension N, resulting in an arrow-shaped representation of the operator
N = 2*length(kernelCoeffs)

# for speedup: unitary transformations do not affect the resolvent norm
isSchur = true

# Huygens Fresnel operator
op = FredConvOp(fredConvMatrix(kernelCoeffs, dom_u), N, dom_u, 0.0+0.0im, isSchur)

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

# since the operator has an arrow-shaped representation, degrees of freedom (DOF) during the Lanczos iteration always equal N
u0 = ones(T, N)
u0 = u0/norm(u0)

# results and dof
pse, dof = pseComp(op, op_conj, u0, ptx, pty, option)

# polt contour
level = -3:0.5:-1
contour(ptx, pty, log10.(pse), levels = level)
