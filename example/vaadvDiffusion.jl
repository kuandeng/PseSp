ENV["GKSwstype"] = "100"
include("../PseSp.jl")

# preallocated dimension N for adaptiveQR solve.
N = 400

T = ComplexF64

# L^2[-pi, pi]
dom = Interval{Float64}(-Float64(pi), Float64(pi))

# Au = h^2*u'' + h*(1+2/3sin(x))u', 周期边界, h = 1/20
h = 1 / 20
S = Chebyshev(dom.left..dom.right)

# operator
a1 = x -> h * (1 + (2 / 3) * sin(x))
coeffs = (
    zeros(T, 1),
    T.(coefficients(Fun(a1, S))),
    T[h^2]
)
bcType = "Periodic"
bcOrder = 2
K = 2
op = DiffOp(N, K, coeffs, bcType, bcOrder, dom, zero(T))

# operator adjoint (periodic): A* = h^2*d^2 - h*(a1*d + a1')
a0_conj = x -> -h * (2 / 3) * cos(x)
a1_conj = x -> -h * (1 + (2 / 3) * sin(x))
coeffs_conj = (
    T.(coefficients(Fun(a0_conj, S))),
    T.(coefficients(Fun(a1_conj, S))),
    T[h^2]
)
op_conj = DiffOp(N, K, coeffs_conj, bcType, bcOrder, dom, zero(T))


# grid points
nptx = 200
npty = 200
ax = [-10, 2]
ay = [-6, 6]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# parameters for Lanczos iteration
option = Options(20, 1, 1e-3, 2.2e-16, "adaptive", false)

# results and dof
pse, dof = pseComp(op, op_conj, ptx, pty, option)

println("pse size = ", size(pse))
println("dof range = [", minimum(dof), ", ", maximum(dof), "]")
println("log10(pse) range = [", minimum(log10.(pse)), ", ", maximum(log10.(pse)), "]")

levels = -6:0.5:1
plt = contour(ptx, pty, log10.(pse), levels = levels, xlabel = "Re(z)", ylabel = "Im(z)", title = "vaadvDiffusion pseudospectrum (200x200)")
fig_path = joinpath(@__DIR__, "vaadvDiffusion_200x200.png")
savefig(plt, fig_path)
println("saved figure: ", fig_path)
