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
using Random

include("typedefs.jl")
include("meshgen.jl")
include("voxelize.jl")
include("exporter.jl")
include("viz.jl")


function run_example(;res = 100,
                    draw_arrows = false,
                    draw_weldlines = false,
                    draw_airtraps = false,
                    draw_streamlines = false, 
                    draw_voxel_pos = false,
                    draw_filling_time = false,
                    markersize = 10, 
                    lengthscale = 0.4,
                    t2 = 0.0,
                    p1 = Point3f(0, 25, 0),
                    p2 = Point3f(100, 25,0),
                    sl_stepsize = 1.0,
                    sl_seed_radius = 2.0,
                    sl_num_steps = 1000,
                    sl_num_streamlines = 10,
                    sl_smooth_passes = 3,
                    )

    fn_step = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")

    meshdata = load_step_mesh_as_surface(fn_step)
    voxeldata = voxelize_mesh(meshdata; res)

    thickness_field = compute_squared_thickness_u16(voxeldata)

    gates = [
        ValveGate(p1, 0.0),
        ValveGate(p2, t2)
    ]

    fill_time = solve_eikonal_voxel(voxeldata, thickness_field, gates)

    stats = analyze_eikonal_field(fill_time, voxeldata, thickness_field)

    fig = Figure()
    ax = Axis3(fig[1,1], aspect = :data)

    mesh!(ax, meshdata.mesh, color = (:lightgray, 0.25), transparency = true)

    interior_points = mask_points(voxeldata.grid, voxeldata.voxel_spacing, voxeldata.corner)
    
    if draw_weldlines
        scatter!(ax, mask_points(stats.masks.weld_lines, voxeldata.voxel_spacing, voxeldata.corner), 
                    color = :red, markersize = 6)
    end

    if draw_airtraps
        scatter!(ax, mask_points(stats.masks.air_traps,  voxeldata.voxel_spacing, voxeldata.corner), color = :gold, markersize = 10)
    end

    if draw_voxel_pos
        scatter!(ax, interior_points, 
            color = fill_time[isfinite.(fill_time)], 
            markersize= markersize)
    end
    if draw_filling_time
        scatter!(ax, mask_points(fill_time, voxeldata.voxel_spacing, voxeldata.corner), markersize = 6)
    end

    if draw_arrows
        interior_flow_dirs = stats.flow_dirs[isfinite.(fill_time)]
        arrows3d!(ax, interior_points, interior_flow_dirs, color = (:black, 0.5), lengthscale = lengthscale)
    end
    # 
    # 
    if draw_streamlines
        # streamlines  = generate_streamlines([p1, p2], stats.flow_dirs, voxeldata; 
        #                                         r_sq = thickness_field.r_sq,
        #                                         stepsize = sl_stepsize, 
        #                                         # weld_mask = stats.masks.weld_lines, 
        #                                         seed_radius = sl_seed_radius,
        #                                         num_streamlines = sl_num_streamlines,
        #                                         num_steps = sl_num_steps,
        #                                         )
        # @info streamlines
        gate_pos = [p1, p2]
        streamlines = generate_streamlines(gate_pos, 
                                            stats.flow_dirs,
                                            voxeldata;
                                            fill_time,
                                            r_sq = thickness_field.r_sq,
                                            num_streamlines = sl_num_streamlines,
                                            num_steps = sl_num_steps,
                                            weld_mask = stats.masks.weld_lines,
                                            seed_radius = sl_seed_radius,
                                            stepsize = sl_stepsize,
                                            smooth_passes = sl_smooth_passes,
                                            )

        
        for sl in streamlines
            lines!(ax, sl)
        end
    end

    fig
end


end # module FiberDist
