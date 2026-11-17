

# 1. Load STEP and generate surface mesh via Gmsh
function load_step_mesh_as_surface(filepath::String;
            target_element_size = 1.0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(filepath)
    gmsh.model.occ.synchronize()
    
    # --- CONTROL MESH SIZE HERE ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 15)

    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    
    node_tags, coord_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    
    gmsh.finalize()
    
    nodes = reshape(coord_coords, 3, :)
    # Extract triangles (element type 2 in Gmsh)
    tri_node_tags = elem_node_tags[findfirst(elem_types .== 2)]
    elements = reshape(tri_node_tags, 3, :)
    
    return (;nodes, elements) |> build_scene_bvh
end

# 2. Build Raycore BVH Acceleration Structure from Mesh
function build_scene_bvh(data)
    nodes, elements = data
    # Convert nodes & elements into GeometryBasics triangles format required by Raycore
    points = [Point3f(nodes[1, i], nodes[2, i], nodes[3, i]) for i in axes(nodes, 2)]
    
    triangles = map(1:size(elements, 2)) do i
        i1, i2, i3 = elements[1, i], elements[2, i], elements[3, i]
        GeometryBasics.TriangleFace(i1, i2, i3)
    end
    # @info eltype(triangles)
    # @info eltype(triangles) <:GeometryBasics.AbstractFace
    # Build Raycore Top-Level Acceleration Structure (TLAS / BVH)
    mesh = normal_mesh(points, triangles)
    # tlas = Raycore.TLAS(KernelAbstractions.CPU()) 
    # push!(tlas, mesh)
    # sync!(tlas)
    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    # return (;mesh, tlas)
    return MeshData{Float32}(mesh, tlas)
end



"""
    project_voxel_to_mesh_vertex(
        meshdata::MeshData,
        voxeldata::VoxelData,
        field::AbstractArray{FT, 3};
        max_search_radius::Int = 4,
        default_val::FT = zero(FT)
    ) -> Vector{FT}

Samples a 3D voxel field (`field`) at every vertex of `meshdata.mesh` using mask-aware
trilinear interpolation over valid interior voxels (`voxeldata.grid == true`), falling back
to the nearest interior voxel if a boundary vertex lies outside the 2×2×2 cell centers.

Returns a `Vector{FT}` matching `length(meshdata.mesh.position)`.
"""
function project_voxel_to_mesh_vertex(
    meshdata::MeshData,
    voxeldata::VoxelData,
    field::AbstractArray{FT, 3};
    max_search_radius::Int = 4,
    default_val::FT = zero(FT)
) where {FT}
    grid = voxeldata.grid
    @assert size(grid) == size(field) "voxeldata.grid and field must have identical dimensions"

    positions = meshdata.mesh.position
    n_verts = length(positions)
    out = Vector{FT}(undef, n_verts)

    spacing_raw = hasproperty(voxeldata, :voxel_spacing) ? voxeldata.voxel_spacing : voxeldata.stepsize
    corner_raw  = voxeldata.corner

    nx, ny, nz = size(grid)
    dx, dy, dz = Float64(spacing_raw[1]), Float64(spacing_raw[2]), Float64(spacing_raw[3])
    ox, oy, oz = Float64(corner_raw[1]),  Float64(corner_raw[2]),  Float64(corner_raw[3])

    cast_output(val::Float64) = FT <: Integer ? round(FT, val) : FT(val)

    @inbounds for idx in 1:n_verts
        p = positions[idx]
        px, py, pz = Float64(p[1]), Float64(p[2]), Float64(p[3])

        # Continuous 1-based voxel-center coordinates
        fx = (px - ox) / dx + 0.5
        fy = (py - oy) / dy + 0.5
        fz = (pz - oz) / dz + 0.5

        i0 = clamp(floor(Int, fx), 1, nx - 1)
        j0 = clamp(floor(Int, fy), 1, ny - 1)
        k0 = clamp(floor(Int, fz), 1, nz - 1)

        tx = clamp(fx - i0, 0.0, 1.0)
        ty = clamp(fy - j0, 0.0, 1.0)
        tz = clamp(fz - k0, 0.0, 1.0)

        wx0, wx1 = 1.0 - tx, tx
        wy0, wy1 = 1.0 - ty, ty
        wz0, wz1 = 1.0 - tz, tz

        val_acc = 0.0
        w_acc   = 0.0

        # 1. Mask-aware trilinear interpolation across the 2×2×2 surrounding voxel centers
        for dk in 0:1, dj in 0:1, di in 0:1
            ii, jj, kk = i0 + di, j0 + dj, k0 + dk
            grid[ii, jj, kk] || continue

            raw_v = field[ii, jj, kk]
            (raw_v isa AbstractFloat && !isfinite(raw_v)) && continue

            w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
            val_acc += w * Float64(raw_v)
            w_acc   += w
        end

        if w_acc > 1e-9
            out[idx] = cast_output(val_acc / w_acc)
            continue
        end

        # 2. Nearest-interior-voxel fallback for vertices sitting slightly outside voxel centers
        ic = clamp(round(Int, fx), 1, nx)
        jc = clamp(round(Int, fy), 1, ny)
        kc = clamp(round(Int, fz), 1, nz)

        best_d2  = Inf
        best_val = default_val
        found    = false

        for r in 1:max_search_radius
            for kk in max(1, kc - r):min(nz, kc + r),
                jj in max(1, jc - r):min(ny, jc + r),
                ii in max(1, ic - r):min(nx, ic + r)
                grid[ii, jj, kk] || continue
                raw_v = field[ii, jj, kk]
                (raw_v isa AbstractFloat && !isfinite(raw_v)) && continue

                vx = ox + (ii - 0.5) * dx
                vy = oy + (jj - 0.5) * dy
                vz = oz + (kk - 0.5) * dz
                d2 = (vx - px)^2 + (vy - py)^2 + (vz - pz)^2
                if d2 < best_d2
                    best_d2  = d2
                    best_val = raw_v
                    found    = true
                end
            end
            found && break
        end

        out[idx] = best_val
    end
    @info extrema(out)
    return out
end
