
"""
    export_fot_vtk(filepath, voxel_grid, voxel_spacing, corner, T_field, analysis)

Exports the 3D Eikonal arrival field, diagnostic masks, flow vectors, and 6-component
symmetric Fiber Orientation Tensor (`TENSORS FOT float`) to a Legacy VTK `.vtk` file.
"""
function export_fot_vtk(
    filepath::AbstractString,
    voxel_grid::AbstractArray{Bool, 3},
    voxel_spacing,
    corner,
    T_field::AbstractArray{<:Real, 3},
    analysis
)
    nx, ny, nz = size(voxel_grid)
    n_pts = nx * ny * nz
    dx, dy, dz = Float64(voxel_spacing[1]), Float64(voxel_spacing[2]), Float64(voxel_spacing[3])
    # VTK structured point coordinates sit at voxel centers
    ox = Float64(corner[1]) + 0.5 * dx
    oy = Float64(corner[2]) + 0.5 * dy
    oz = Float64(corner[3]) + 0.5 * dz

    fot   = analysis.fot
    masks = analysis.masks
    dirs  = analysis.flow_dirs

    open(filepath, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "FiberDist Eikonal FOT and Defect Analysis")
        println(io, "ASCII")
        println(io, "DATASET STRUCTURED_POINTS")
        @printf(io, "DIMENSIONS %d %d %d\n", nx, ny, nz)
        @printf(io, "ORIGIN %.6f %.6f %.6f\n", ox, oy, oz)
        @printf(io, "SPACING %.6f %.6f %.6f\n", dx, dy, dz)
        @printf(io, "POINT_DATA %d\n", n_pts)

        # 1. Solid Mask
        println(io, "SCALARS SolidMask int 1")
        println(io, "LOOKUP_TABLE default")
        for k in 1:nz, j in 1:ny, i in 1:nx
            println(io, voxel_grid[i, j, k] ? 1 : 0)
        end

        # 2. Arrival Time
        println(io, "SCALARS ArrivalTime float 1")
        println(io, "LOOKUP_TABLE default")
        for k in 1:nz, j in 1:ny, i in 1:nx
            val = (voxel_grid[i, j, k] && isfinite(T_field[i, j, k])) ? Float32(T_field[i, j, k]) : -1.0f0
            @printf(io, "%.6g\n", val)
        end

        # 3. Diagnostic BitArray Masks
        for (name, mask) in (
            ("WeldLine",   masks.weld_lines),
            ("MeldLine",   masks.meld_lines),
            ("AirTrap",    masks.air_traps),
            ("Hesitation", masks.hesitation)
        )
            println(io, "SCALARS $name int 1")
            println(io, "LOOKUP_TABLE default")
            for k in 1:nz, j in 1:ny, i in 1:nx
                println(io, mask[i, j, k] ? 1 : 0)
            end
        end

        # 4. Flow Direction Vectors
        println(io, "VECTORS FlowDirection float")
        for k in 1:nz, j in 1:ny, i in 1:nx
            u = dirs[i, j, k]
            @printf(io, "%.5f %.5f %.5f\n", Float32(u[1]), Float32(u[2]), Float32(u[3]))
        end

        # 5. Full Symmetric 3x3 Fiber Orientation Tensor
        println(io, "TENSORS FOT float")
        for k in 1:nz, j in 1:ny, i in 1:nx
            v11 = Float32(fot.a11[i, j, k])
            v22 = Float32(fot.a22[i, j, k])
            v33 = Float32(fot.a33[i, j, k])
            v12 = Float32(fot.a12[i, j, k])
            v13 = Float32(fot.a13[i, j, k])
            v23 = Float32(fot.a23[i, j, k])
            @printf(io, "%.5f %.5f %.5f\n%.5f %.5f %.5f\n%.5f %.5f %.5f\n\n",
                v11, v12, v13,
                v12, v22, v23,
                v13, v23, v33
            )
        end
    end
    return filepath
end


"""
    export_fot_moldflow_xml(filepath, voxel_grid, voxel_spacing, corner, analysis)

Exports interior solid voxel centroids and their 6 symmetric FOT components
`(a11, a22, a33, a12, a13, a23)` in standard Moldflow / Digimat XML format.
"""
function export_fot_moldflow_xml(
    filepath::AbstractString,
    voxel_grid::AbstractArray{Bool, 3},
    voxel_spacing,
    corner,
    analysis
)
    nx, ny, nz = size(voxel_grid)
    dx, dy, dz = Float64(voxel_spacing[1]), Float64(voxel_spacing[2]), Float64(voxel_spacing[3])
    ox, oy, oz = Float64(corner[1]), Float64(corner[2]), Float64(corner[3])
    fot = analysis.fot

    n_solid = count(voxel_grid)

    open(filepath, "w") do io
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        println(io, "<MoldflowSimulationData version=\"1.0\">")
        println(io, "  <Header>")
        println(io, "    <Generator>FiberDist.jl</Generator>")
        println(io, "    <DataType>FiberOrientationTensor</DataType>")
        println(io, "    <CoordinateSystem>GlobalCartesian</CoordinateSystem>")
        println(io, "    <TensorOrder>a11 a22 a33 a12 a13 a23</TensorOrder>")
        @printf(io, "    <PointCount>%d</PointCount>\n", n_solid)
        println(io, "  </Header>")
        println(io, "  <OrientationDataset>")

        idx = 0
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            voxel_grid[i, j, k] || continue
            idx += 1
            px = ox + (i - 0.5) * dx
            py = oy + (j - 0.5) * dy
            pz = oz + (k - 0.5) * dz

            @printf(io,
                "    <Point id=\"%d\" x=\"%.5f\" y=\"%.5f\" z=\"%.5f\" a11=\"%.5f\" a22=\"%.5f\" a33=\"%.5f\" a12=\"%.5f\" a13=\"%.5f\" a23=\"%.5f\"/>\n",
                idx, px, py, pz,
                fot.a11[i, j, k], fot.a22[i, j, k], fot.a33[i, j, k],
                fot.a12[i, j, k], fot.a13[i, j, k], fot.a23[i, j, k]
            )
        end

        println(io, "  </OrientationDataset>")
        println(io, "</MoldflowSimulationData>")
    end
    return filepath
end