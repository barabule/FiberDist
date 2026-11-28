

using Test
using BenchmarkTools

import FiberDist as FD
const GeometryBasics = FD.GeometryBasics
const StaticArrays   = FD.StaticArrays
using .GeometryBasics
using .StaticArrays

@testset "Mesh" verbose = true begin
include("testset_import.jl")
include("benchmark_test.jl")
end

