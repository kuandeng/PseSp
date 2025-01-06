using LinearAlgebra
using SparseArrays
using BandedMatrices
using Plots
using Parameters
using LinearAlgebra.BLAS
using MAT

const AbstractComplex = Complex{T} where T <: AbstractFloat
const FloatOrComplex = Union{AbstractFloat, AbstractComplex}

abstract type Op{T<:FloatOrComplex} end

include("op/op.jl")
include("pseComp.jl")

