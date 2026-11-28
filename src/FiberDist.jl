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
using NativeFileDialog
using JLD

include("typedefs.jl") #types

include("import_mesh.jl") #stage I - load CAD, mesh and return MeshData struct

include("voxelize.jl") #stage II voxelization - in/out, local thickness -> VoxelData

include("solve_eikonal_parallel.jl") # stage III - parallel voxel eikonal

include("tangent_projection.jl") # stage IV project to surface mesh per triangle and compute 2.5d local FOT per half thickness layer

# include("exporter.jl") #stage V - export to different formats

# include("sparse_bricks.jl")
# using .SparseBrick


include("savefile.jl") #save / load functionality

include("viz.jl") #extra viz
include("gui.jl")



# export SparseBrickGrid,
#        build_sparse_bricks,
#        compute_brick_edt!,
#        solve_eikonal_sparse!,
#        voxelize_and_solve_sparse,
#        project_to_surface_sparse,
#        query_sparse_point

end # module FiberDist
