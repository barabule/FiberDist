



"""
    voxelize_mesh(meshdata; res=100, padding=0.05)

Voxelizes a Raycore acceleration structure (`meshdata.tlas`, `meshdata.mesh`) into a 
3D `BitArray` using multi-threaded Z-axis ray-casting parity (even/odd rule).
Returns `(; grid, stepsize, corner)`.
"""
function voxelize_mesh(
    meshdata;
    res::Integer = 100,       # resolution across longest axis
    padding::Real = 0.05      # relative padding to bounding box span
)
    bvh, mesh = meshdata.tlas, meshdata.mesh

    # 1. Compute bounding box and infer floating-point type from mesh coordinates
    bbox = GeometryBasics.Rect3(mesh)
    min_bound, max_bound = extrema(bbox)
    T = float(eltype(min_bound))

    orig_span = max_bound - min_bound
    pad_vec   = orig_span .* T(padding)
    min_pt    = min_bound - pad_vec
    max_pt    = max_bound + pad_vec
    span      = max_pt - min_pt

    max_span = maximum(span)
    dx = max_span / T(res) # uniform cubic voxel size
    dy = dx
    dz = dx

    res_x = max(1, ceil(Int, span[1] / dx))
    res_y = max(1, ceil(Int, span[2] / dy))
    res_z = max(1, ceil(Int, span[3] / dz))

    voxel_grid = BitArray{3}(undef, res_x, res_y, res_z)
    fill!(voxel_grid, false)
    z_start = min_pt[3] - dx
    ray_dir = SVector{3, T}(zero(T), zero(T), one(T))
    eps_step = dx * T(1e-3)

    @info "Voxelizing mesh into $(res_x)×$(res_y)×$(res_z) BitArray grid..."

    # 2. Parallelize in chunks of 64 along the leading dimension `i` so each thread
    #    owns disjoint 64-bit words in `voxel_grid.chunks` (100% race-condition free).
    Threads.@threads for i_start in 1:64:res_x
        i_end = min(i_start + 63, res_x)
        hits = T[] # reused per thread to avoid allocations

        for j in 1:res_y
            y = min_pt[2] + (T(j) - T(0.5)) * dy

            for i in i_start:i_end
                x = min_pt[1] + (T(i) - T(0.5)) * dx
                empty!(hits)

                curr_z = z_start
                while true
                    ray = Raycore.Ray(o = Point3{T}(x, y, curr_z), d = ray_dir)
                    hit_found, _, distance, _, _ = Raycore.closest_hit(bvh, ray)
                    hit_found || break

                    z_hit = curr_z + T(distance)
                    push!(hits, z_hit)
                    curr_z = z_hit + eps_step
                end

                # Even/odd parity fill along Z-column
                n_hits = length(hits)
                if n_hits >= 2
                    sort!(hits)
                    # Iterate over [entry, exit] pairs directly in index space
                    for m in 1:2:(n_hits - 1)
                        z_enter = hits[m]
                        z_exit  = hits[m + 1]

                        # Voxel center is at min_pt[3] + (k - 0.5) * dz
                        k_min = clamp(ceil(Int,  (z_enter - min_pt[3]) / dz + T(0.5)), 1, res_z)
                        k_max = clamp(floor(Int, (z_exit  - min_pt[3]) / dz + T(0.5)), 1, res_z)

                        for k in k_min:k_max
                            @inbounds voxel_grid[i, j, k] = true
                        end
                    end
                end
            end
        end
    end

    return (; grid = voxel_grid, stepsize = (dx, dy, dz), corner = min_pt)
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
    node_thicknesses = zeros(T, num_nodes)
    
    for i in 1:num_nodes
        point = nodes[i]
        
        # Evaluate interpolated thickness at node coordinate
        node_thicknesses[i] = extrapolated_sitp(point...)
    end
    
    return node_thicknesses
end






"""
    cartesian_upwind_3d(T, i, j, k, F, dx, dy, dz)

Solves the exact 3D Cartesian upwind quadratic equation:
  max((T - Tx)/dx, 0)^2 + max((T - Ty)/dy, 0)^2 + max((T - Tz)/dz, 0)^2 = 1 / F^2
using minimum accepted neighbor values along each axis.
"""
function cartesian_upwind_3d(
    thickness_field::Array{T, 3},
    i::Int, j::Int, k::Int,
    F::T,
    dx::T, dy::T, dz::T
) where T<:Real
    nx, ny, nz = size(thickness_field)

    tx = min(i > 1 ? thickness_field[i-1, j, k] : Inf32, i < nx ? thickness_field[i+1, j, k] : Inf32)
    ty = min(j > 1 ? thickness_field[i, j-1, k] : Inf32, j < ny ? thickness_field[i, j+1, k] : Inf32)
    tz = min(k > 1 ? thickness_field[i, j, k-1] : Inf32, k < nz ? thickness_field[i, j, k+1] : Inf32)

    # Sort active upwind dimensions by arrival time: t1 <= t2 <= t3
    dims = (
        (tx, dx),
        (ty, dy),
        (tz, dz)
    )
    # Manual 3-element sort to stay allocation-free
    d1, d2, d3 = dims[1], dims[2], dims[3]
    if d1[1] > d2[1]; d1, d2 = d2, d1; end
    if d2[1] > d3[1]; d2, d3 = d3, d2; end
    if d1[1] > d2[1]; d1, d2 = d2, d1; end

    invF = 1.0f0 / F
    t1, h1 = d1
    !isfinite(t1) && return Inf32

    # 1D update
    t_cand = t1 + h1 * invF

    # 2D quadratic update if second neighbor is behind current candidate
    t2, h2 = d2
    if isfinite(t2) && t2 < t_cand
        a = (1.0f0 / h1^2) + (1.0f0 / h2^2)
        b = -2.0f0 * (t1 / h1^2 + t2 / h2^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) - invF^2
        disc = b^2 - 4.0f0 * a * c
        if disc >= 0.0f0
            t_cand = (-b + sqrt(disc)) / (2.0f0 * a)
        end
    end

    # 3D quadratic update if third neighbor is behind current candidate
    t3, h3 = d3
    if isfinite(t3) && t3 < t_cand
        a = (1.0f0 / h1^2) + (1.0f0 / h2^2) + (1.0f0 / h3^2)
        b = -2.0f0 * (t1 / h1^2 + t2 / h2^2 + t3 / h3^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) + (t3^2 / h3^2) - invF^2
        disc = b^2 - 4.0f0 * a * c
        if disc >= 0.0f0
            t_cand = (-b + sqrt(disc)) / (2.0f0 * a)
        end
    end

    return t_cand
end

"""
    solve_eikonal_voxel(voxel_grid, thickness_field, voxel_spacing, min_pt, gates; min_thickness=0.1f0)

Solves the 3D Eikonal equation on interior solid voxels using Fast Marching.
"""
function solve_eikonal_voxel(
    voxel_grid::AbstractArray{Bool, 3},
    thickness_field::AbstractArray{T1, 3},
    voxel_spacing::Union{NTuple{3, T2}, StaticVector{3, T2}},
    min_pt::StaticVector{3, T3},
    gates::AbstractVector{<:ValveGate};
    min_thickness::Real = 0.1
) where {T1<:Real, T2<:Real, T3<:Real}

    nx, ny, nz = size(voxel_grid)
    dx, dy, dz = T.(voxel_spacing)

    T = fill(Inf32, nx, ny, nz)
    accepted = falses(nx, ny, nz)
    pq = PriorityQueue{CartesianIndex{3}, T}()

    # Helper: convert world coordinate to nearest interior voxel
    function find_nearest_interior_voxel(pos::Point3f)
        best_idx = CartesianIndex(1, 1, 1)
        best_dist_sq = Inf32
        for k in 1:nz, j in 1:ny, i in 1:nx
            voxel_grid[i, j, k] || continue
            vx = min_pt[1] + (i - 0.5f0) * dx
            vy = min_pt[2] + (j - 0.5f0) * dy
            vz = min_pt[3] + (k - 0.5f0) * dz
            d2 = (vx - pos[1])^2 + (vy - pos[2])^2 + (vz - pos[3])^2
            if d2 < best_dist_sq
                best_dist_sq = d2
                best_idx = CartesianIndex(i, j, k)
            end
        end
        return best_idx
    end

    # Seed gates (supporting cascaded opening times)
    for gate in gates
        idx = find_nearest_interior_voxel(gate.position)
        if gate.open_time < thickness_field[idx]
            thickness_field[idx] = gate.open_time
            pq[idx] = gate.open_time
        end
    end

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    while !isempty(pq)
        curr, curr_t = dequeue_pair!(pq)
        accepted[curr] && continue
        accepted[curr] = true

        for off in offsets
            nbr = curr + off
            i, j, k = Tuple(nbr)
            (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
            (voxel_grid[i, j, k] && !accepted[i, j, k]) || continue

            F_local = max(thickness_field[i, j, k], min_thickness)^2
            t_cand = cartesian_upwind_3d(T, i, j, k, F_local, dx, dy, dz)

            if t_cand < thickness_field[i, j, k]
                thickness_field[i, j, k] = t_cand
                pq[nbr] = t_cand
            end
        end
    end

    return T
end

"""
    project_voxel_eikonal_to_mesh(T, voxel_grid, voxel_spacing, min_pt, centroids, normals)

Computes clean one-sided gradients on the interior voxel grid and projects
arrival times and tangent flow direction vectors onto surface triangle centroids.
"""
function project_voxel_eikonal_to_mesh(
    thickness_field::AbstractArray{T1, 3},
    voxel_grid::AbstractArray{Bool, 3},
    voxel_spacing::Union{NTuple{3, T2}, StaticVector{3, T2}},
    min_pt::StaticVector{3, T3},
    centroids::AbstractVector{<:StaticVector{3, T4}},
    normals::AbstractVector{<:StaticVector{3, T5}}
) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    T = float(promote_type(T1, T2, T3, T4, T5))
    nx, ny, nz = size(T_field)
    dx, dy, dz = T.(voxel_spacing)

    # 1. Compute interior-only gradient field (gx, gy, gz) using valid neighbors
    gx = zeros(T, nx, ny, nz)
    gy = zeros(T, nx, ny, nz)
    gz = zeros(T, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        (voxel_grid[i, j, k] && isfinite(thickness_field[i, j, k])) || continue
        tc = thickness_field[i, j, k]

        # X derivative (central if both interior, one-sided at boundary)
        has_xm = i > 1  && voxel_grid[i-1, j, k] && isfinite(thickness_field[i-1, j, k])
        has_xp = i < nx && voxel_grid[i+1, j, k] && isfinite(thickness_field[i+1, j, k])
        gx[i, j, k] = if has_xm && has_xp
            (thickness_field[i+1, j, k] - thickness_field[i-1, j, k]) / (2.0f0 * dx)
        elseif has_xp
            (thickness_field[i+1, j, k] - tc) / dx
        elseif has_xm
            (tc - thickness_field[i-1, j, k]) / dx
        else
            0.0f0
        end

        # Y derivative
        has_ym = j > 1  && voxel_grid[i, j-1, k] && isfinite(thickness_field[i, j-1, k])
        has_yp = j < ny && voxel_grid[i, j+1, k] && isfinite(thickness_field[i, j+1, k])
        gy[i, j, k] = if has_ym && has_yp
            (thickness_field[i, j+1, k] - thickness_field[i, j-1, k]) / (2.0f0 * dy)
        elseif has_yp
            (thickness_field[i, j+1, k] - tc) / dy
        elseif has_ym
            (tc - thickness_field[i, j-1, k]) / dy
        else
            0.0f0
        end

        # Z derivative
        has_zm = k > 1  && voxel_grid[i, j, k-1] && isfinite(thickness_field[i, j, k-1])
        has_zp = k < nz && voxel_grid[i, j, k+1] && isfinite(thickness_field[i, j, k+1])
        gz[i, j, k] = if has_zm && has_zp
            (thickness_field[i, j, k+1] - thickness_field[i, j, k-1]) / (2.0f0 * dz)
        elseif has_zp
            (thickness_field[i, j, k+1] - tc) / dz
        elseif has_zm
            (tc - thickness_field[i, j, k-1]) / dz
        else
            0.0f0
        end
    end

    # 2. Sample at centroids nudged slightly inward along -normal so we hit interior voxels
    #    (avoids Inf32 exterior cells during nearest/trilinear lookup)
    n_tris = length(centroids)
    tri_T = zeros(T, n_tris)
    tri_flow_dirs = Vector{SVector{3, T}}(undef, n_tris)
    inward_nudge = 0.6f0 * min(dx, dy, dz)

    for t in 1:n_tris
        p = SVector{3, T}(centroids[t]) - inward_nudge * normals[t]

        # Continuous voxel index (voxel centers are at min_pt + (i - 0.5) * dx)
        fi = clamp((p[1] - min_pt[1]) / dx + 0.5f0, 1.0f0, T(nx))
        fj = clamp((p[2] - min_pt[2]) / dy + 0.5f0, 1.0f0, T(ny))
        fk = clamp((p[3] - min_pt[3]) / dz + 0.5f0, 1.0f0, T(nz))

        # Trilinear interpolation over valid interior corners of the 2x2x2 cell
        i0 = clamp(floor(Int, fi), 1, nx - 1); wi = fi - i0
        j0 = clamp(floor(Int, fj), 1, ny - 1); wj = fj - j0
        k0 = clamp(floor(Int, fk), 1, nz - 1); wk = fk - k0

        w_sum = 0.0f0
        t_acc = 0.0f0
        g_acc = SVector(0.0f0, 0.0f0, 0.0f0)

        for dk in 0:1, dj in 0:1, di in 0:1
            ii, jj, kk = i0 + di, j0 + dj, k0 + dk
            (voxel_grid[ii, jj, kk] && isfinite(thickness_field[ii, jj, kk])) || continue

            w = (di == 0 ? 1.0f0 - wi : wi) *
                (dj == 0 ? 1.0f0 - wj : wj) *
                (dk == 0 ? 1.0f0 - wk : wk)

            w_sum += w
            t_acc += w * thickness_field[ii, jj, kk]
            g_acc += w * SVector(gx[ii, jj, kk], gy[ii, jj, kk], gz[ii, jj, kk])
        end

        if w_sum > 1f-6
            tri_thickness_field[t] = t_acc / w_sum
            g_vec = g_acc / w_sum
            # Project onto triangle tangent plane so surface arrows stay strictly on the surface
            g_tan = g_vec - dot(g_vec, normals[t]) * normals[t]
            gn = norm(g_tan)
            tri_flow_dirs[t] = gn > 1f-9 ? (g_tan / gn) : SVector(0.0f0, 0.0f0, 0.0f0)
        else
            # Fallback to nearest valid interior voxel if nudged point fell outside
            ii, jj, kk = round(Int, fi), round(Int, fj), round(Int, fk)
            tri_thickness_field[t] = thickness_field[ii, jj, kk]
            g_vec = SVector(gx[ii, jj, kk], gy[ii, jj, kk], gz[ii, jj, kk])
            g_tan = g_vec - dot(g_vec, normals[t]) * normals[t]
            gn = norm(g_tan)
            tri_flow_dirs[t] = gn > 1f-9 ? (g_tan / gn) : SVector(0.0f0, 0.0f0, 0.0f0)
        end
    end

    return tri_T, tri_flow_dirs
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