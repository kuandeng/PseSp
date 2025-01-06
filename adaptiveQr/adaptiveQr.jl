using LinearAlgebra.LAPACK: larf!, larfg!, chkside
using LinearAlgebra.BLAS: blascopy!, @blasfunc, chkuplo, vec_pointer_stride
using LinearAlgebra: checksquare, BlasFloat, BlasInt, chkstride1
using LinearAlgebra
using libblastrampoline_jll
using Base: require_one_based_indexing

include("./bandedQr.jl")