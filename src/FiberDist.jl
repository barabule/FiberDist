module FiberDist

using LinearAlgebra
using SparseArrays
using Gmsh
using Raycore
using FileIO
using MeshIO
using GeometryBasics
using StaticArrays
using KernelAbstractions
using GLMakie
using DistanceTransforms
using Interpolations
using DataStructures
using Printf
using Random
using Adapt

include("typedefs.jl")
include("import_mesh.jl") #stage I - load CAD, mesh and return meshdata struct

include("voxelize.jl") #stage II voxelization - in/out, local thickness
# #stage III - solve Eikonal , voxel gradient, defects
include("solve_eikonal_parallel.jl") #parallel voxel eikonal
# include("tangent_projection.jl") # stage IV project to surface  mesh and compute 2.5d local FOT per half thickness layer

# include("exporter.jl") #stage V - export to different formats

# include("viz.jl") #extra
# include("gui.jl")



end # module FiberDist
