

"""
    voxelize_mesh(bvh, points, resolution=(100, 100, 50))

Voxelizes a Raycore acceleration structure into a 3D boolean array 
using a ray-casting parity count (even/odd rule) across the Z-axis.
"""
function voxelize_mesh(meshdata; res = 100)
    bvh, points = meshdata.tlas, meshdata.mesh.position
    # 1. Compute bounding box of the mesh
    bbox = GeometryBasics.Rect3(meshdata.mesh)
    min_bound, max_bound = extrema(bbox)
    span = max_bound - min_bound
    @info "span", span
    # Add a small padding around the part
    padding = span * 0.05
    min_pt = min_bound - padding
    max_pt = max_bound + padding
    span = max_pt - min_pt
    
    max_span = max(span[1], max(span[2], span[3]))
    dx = dy = dz = max_span / res #uniform voxelsize
    res_x = ceil(Int, span[1] / dx)
    res_y = ceil(Int, span[2] / dx)
    res_z = ceil(Int, span[3] / dx)
    
    voxel_grid = falses(res_x, res_y, res_z)
    
    # 2. Scanline ray-casting along the Z-axis for each (x, y) column
    # An odd number of hits means the voxel is inside the solid part.
    @info "Voxelizing mesh into $(res_x)x$(res_y)x$(res_z) grid..."
    for i in 1:res_x
        for j in 1:res_y
            x = min_pt[1] + (i - 0.5f0) * dx
            y = min_pt[2] + (j - 0.5f0) * dy
            
            # Cast a ray starting from well outside the bounding box upwards along +Z
            ray_origin = Point3f(x, y, min_pt[3] - 1.0f0)
            ray_dir = SVector(0.0f0, 0.0f0, 1.0f0)
            
            # Collect all intersection distances along this column ray
            hits = Float32[]
            current_origin = ray_origin
            
            while true
                ray = Raycore.Ray(o=current_origin, d=ray_dir)
                hit_found, triangle, distance, bary, instance_idx = Raycore.closest_hit(bvh, ray)
                
                if !hit_found break end
                
                push!(hits, distance)
                # Step origin slightly past the hit point to find subsequent intersections (back-faces)
                current_origin = current_origin + ray_dir * (distance + 1f-3)
            end
            
            # If we have hits, determine inside/outside intervals using parity rules
            if length(hits) >= 2
                # Sort hits along Z
                sort!(hits)
                # Fill voxels residing between entry and exit pairs (even/odd boundary crossings)
                for k in 1:res_z
                    z = min_pt[3] + (k - 0.5f0) * dz
                    z_rel = z - (min_pt[3] - 1.0f0)
                    
                    # Check if z falls inside any solid pair interval
                    is_inside = false
                    for m in 1:2:length(hits)-1
                        if hits[m] <= z_rel <= hits[m+1]
                            is_inside = true
                            break
                        end
                    end
                    voxel_grid[i, j, k] = is_inside
                end
            end
        end
    end
    
    return (;grid = voxel_grid, stepsize = (dx, dy, dz), corner = min_pt)
end

"""
    compute_thickness_field_edt(voxel_grid, voxel_spacing)

Computes the Euclidean Distance Transform and scales it to get wall-to-wall thickness.
"""
function compute_thickness_field_edt(voxeldata)
    voxel_grid, voxel_spacing = voxeldata.grid, voxeldata.stepsize
    # 1. Convert boolean grid to binary indicator array format expected by DistanceTransforms.jl
    binary_arr = BitArray(voxel_grid)
    
    # 2. Compute Euclidean Distance Transform (EDT)
    # This gives the distance from each interior voxel to the nearest background (0) voxel
    DT = DistanceTransforms.Felzenszwalb()
    edt_result = transform(binary_arr, DT)
    
    # Extract the distance data array
    distance_field = edt_result 
    
    # 3. Local thickness is twice the distance to the nearest wall, scaled by voxel dimensions
    # Assuming roughly isotropic spacing or scaling appropriately:
    # avg_spacing = sum(voxel_spacing) / length(voxel_spacing)
    # @info "avg spacing", avg_spacing
    # thickness_field = 2.0f0 .* distance_field .* avg_spacing
    
    return compute_local_thickness(distance_field, voxel_spacing)
end

"""
    compute_local_thickness(edt_field, voxel_spacing)

Computes true local thickness (Hildebrand-Rüegsegger) by propagating 
medial-axis sphere radii outward to the boundaries.
"""
function compute_local_thickness(edt_field, voxel_spacing)
    h_avg = sum(voxel_spacing) / length(voxel_spacing)
    radii = edt_field .* h_avg
    thickness = radii .* 2.0f0
    
    nx, ny, nz = size(radii)
    
    # Iterate over interior voxels that sit on or near the medial ridge
    for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        r = radii[i, j, k]
        r <= h_avg && continue
        
        # Check if (i,j,k) is a local maximum in at least one axis (medial ridge)
        is_ridge = (r >= radii[i-1, j, k] && r >= radii[i+1, j, k]) ||
                   (r >= radii[i, j-1, k] && r >= radii[i, j+1, k]) ||
                   (r >= radii[i, j, k-1] && r >= radii[i, j, k+1])
        is_ridge || continue
        
        # Stamp the sphere of diameter 2r onto all interior voxels within radius r
        rad_vox = ceil(Int, r / h_avg)
        r_sq = r^2
        d = 2.0f0 * r
        
        for dk in -rad_vox:rad_vox
            kk = k + dk
            (1 <= kk <= nz) || continue
            for dj in -rad_vox:rad_vox
                jj = j + dj
                (1 <= jj <= ny) || continue
                for di in -rad_vox:rad_vox
                    ii = i + di
                    (1 <= ii <= nx) || continue
                    
                    # Only propagate to solid interior voxels
                    edt_field[ii, jj, kk] > 0 || continue
                    
                    dist_sq = (di * voxel_spacing[1])^2 + 
                              (dj * voxel_spacing[2])^2 + 
                              (dk * voxel_spacing[3])^2
                              
                    if dist_sq <= r_sq && d > thickness[ii, jj, kk]
                        thickness[ii, jj, kk] = d
                    end
                end
            end
        end
    end
    
    return thickness
end

"""
    map_thickness_to_nodes(thickness_field, voxel_spacing, min_pt, nodes)

Interpolates the 3D voxel thickness grid onto the Gmsh surface mesh nodes.
"""
function map_thickness_to_nodes(thickness_field, meshdata, voxeldata)
    dx, dy, dz = voxeldata.stepsize
    nodes = meshdata.mesh.position

    # 1. Define the coordinate axes of the 3D voxel grid in physical space
    nx, ny, nz = size(thickness_field)
    min_pt = voxeldata.corner
    x_range = range(min_pt[1], step=dx, length=nx)
    y_range = range(min_pt[2], step=dy, length=ny)
    z_range = range(min_pt[3], step=dz, length=nz)
    
    # 2. Build a b-spline linear interpolator over the grid
    # Interpolations.jl expects (x, y, z) axes matching dimensions 1, 2, 3
    itp = interpolate(thickness_field, BSpline(Linear()))
    sitp = scale(itp, x_range, y_range, z_range)
    
    # Optional: Handle out-of-bounds nodes gracefully (e.g., clamp or set to NaN/zero)
    extrapolated_sitp = extrapolate(sitp, 0.0f0) 
    
    # 3. Query thickness for every node in the surface mesh
    num_nodes = length(nodes)
    node_thicknesses = zeros(Float32, num_nodes)
    
    for i in 1:num_nodes
        point = nodes[i]
        
        # Evaluate interpolated thickness at node coordinate
        node_thicknesses[i] = extrapolated_sitp(point...)
    end
    
    return node_thicknesses
end







function vizualize_tfield(meshdata, node_thicknesses)


    fig = Figure()
    ax = Axis3(fig[1,1])
    mesh!(ax, meshdata.mesh, color = node_thicknesses)
    wireframe!(ax, meshdata.mesh, color = :black, alpha = 0.5)
    climits = extrema(node_thicknesses)
    Colorbar(fig[1,2], limits = climits, tellheight= false)
    fig
end


function run_example(;
                    target_element_size = 1.0,
                    res = 100)

    fn_step = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs.step")
    data = load_step_mesh_as_surface(fn_step; target_element_size)
    meshdata = build_scene_bvh(data)
    voxeldata = voxelize_mesh(meshdata; res)
    tfield = compute_thickness_field_edt(voxeldata)
    nthick = map_thickness_to_nodes(tfield, meshdata, voxeldata)
    fig = vizualize_tfield(meshdata, nthick)

    return (;fn_step, 
            data, 
            meshdata, 
            voxeldata, 
            nthick, 
            fig)
end