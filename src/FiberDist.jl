module FiberDist

using LinearAlgebra
using SparseArrays
using Gmsh
using Raycore
using GeometryBasics
using StaticArrays
using KernelAbstractions
using GLMakie
using DistanceTransforms
using Interpolations
using DataStructures
using Printf


include("typedefs.jl")
include("meshgen.jl")
include("voxelize.jl")
include("exporter.jl")



function run_example(;res = 100, markersize = 10)

    fn_step = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs.step")

    meshdata = load_step_mesh_as_surface(fn_step)
    voxeldata = voxelize_mesh(meshdata; res)

    thickness_field = compute_squared_thickness_u16(voxeldata)

    gates = [
        ValveGate(Point3f(0, 25, 0), 0.0),
        ValveGate(Point3f(100, 25,0), 0.0)
    ]

    fill_time = solve_eikonal_voxel(voxeldata, thickness_field, gates)

    stats = analyze_eikonal_field(fill_time, voxeldata, thickness_field)

    fig = Figure()
    ax = Axis3(fig[1,1])

    mesh!(ax, meshdata.mesh, color = (:lightgray, 0.25), transparency = true)

    # scatter!(ax, mask_points(fill_time, voxeldata.voxel_spacing, voxeldata.corner), markersize = 6)
    scatter!(ax, mask_points(voxeldata.grid, voxeldata.voxel_spacing, voxeldata.corner), 
                color = fill_time[isfinite.(fill_time)], 
                markersize= markersize)
    # scatter!(ax, mask_points(stats.masks.weld_lines, voxeldata.voxel_spacing, voxeldata.corner), color = :red, markersize = 6)
    # scatter!(ax, mask_points(stats.masks.air_traps,  voxeldata.voxel_spacing, voxeldata.corner), color = :gold, markersize = 10)

    fig
end


end # module FiberDist
