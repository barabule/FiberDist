



"""
    voxelize_mesh(meshdata; res=100, padding=0.05)

Voxelizes a Raycore acceleration structure (`meshdata.tlas`, `meshdata.mesh`) into a 
3D `BitArray` using multi-threaded Z-axis ray-casting parity (even/odd rule).
Returns `(; grid, voxel_spacing, corner)`.
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
    corner    = min_bound - pad_vec
    max_pt    = max_bound + pad_vec
    span      = max_pt - corner

    max_span = maximum(span)
    dx = max_span / T(res) # uniform cubic voxel size
    dy = dx
    dz = dx

    res_x = max(1, ceil(Int, span[1] / dx))
    res_y = max(1, ceil(Int, span[2] / dy))
    res_z = max(1, ceil(Int, span[3] / dz))

    voxel_grid = BitArray{3}(undef, res_x, res_y, res_z)
    fill!(voxel_grid, false)
    z_start = corner[3] - dx
    ray_dir = SVector{3, T}(zero(T), zero(T), one(T))
    eps_step = dx * T(1e-3)

    @info "Voxelizing mesh into $(res_x)×$(res_y)×$(res_z) BitArray grid..."

    # 2. Parallelize in chunks of 64 along the leading dimension `i` so each thread
    #    owns disjoint 64-bit words in `voxel_grid.chunks` (100% race-condition free).
    Threads.@threads for i_start in 1:64:res_x
        i_end = min(i_start + 63, res_x)
        hits = T[] # reused per thread to avoid allocations

        for j in 1:res_y
            y = corner[2] + (T(j) - T(0.5)) * dy

            for i in i_start:i_end
                x = corner[1] + (T(i) - T(0.5)) * dx
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

                        # Voxel center is at corner[3] + (k - 0.5) * dz
                        k_min = clamp(ceil(Int,  (z_enter - corner[3]) / dz + T(0.5)), 1, res_z)
                        k_max = clamp(floor(Int, (z_exit  - corner[3]) / dz + T(0.5)), 1, res_z)

                        for k in k_min:k_max
                            @inbounds voxel_grid[i, j, k] = true
                        end
                    end
                end
            end
        end
    end

    # return (; grid = voxel_grid, voxel_spacing = (dx, dy, dz), corner = corner)
    grid = voxel_grid
    voxel_spacing = (dx, dy, dz)
    corner = corner
    return VoxelData{T}(grid, voxel_spacing, corner)
end

# """
#     compute_thickness_field_edt(voxel_grid, voxel_spacing)

# Computes the Euclidean Distance Transform and scales it to get wall-to-wall thickness.
# """
# function compute_thickness_field_edt(voxeldata::VoxelData)
#     voxel_grid, voxel_spacing = voxeldata.grid, voxeldata.voxel_spacing
#     # 1. Convert boolean grid to binary indicator array format expected by DistanceTransforms.jl
#     binary_arr = BitArray(voxel_grid)
    
#     # 2. Compute Euclidean Distance Transform (EDT)
#     # This gives the distance from each interior voxel to the nearest background (0) voxel
#     DT = DistanceTransforms.Felzenszwalb()
#     edt_result = transform(binary_arr, DT)
    
#     # Extract the distance data array
#     distance_field = edt_result 
    
#     # 3. Local thickness is twice the distance to the nearest wall, scaled by voxel dimensions
#     # Assuming roughly isotropic spacing or scaling appropriately:
#     # avg_spacing = sum(voxel_spacing) / length(voxel_spacing)
#     # @info "avg spacing", avg_spacing
#     # thickness_field = 2.0f0 .* distance_field .* avg_spacing
    
#     return compute_local_thickness(distance_field, voxel_spacing)
# end

# """
#     compute_local_thickness(edt_field, voxel_spacing)

# Computes true local thickness (Hildebrand-Rüegsegger) by propagating 
# medial-axis sphere radii outward to the boundaries.
# """
# function compute_local_thickness(edt_field, voxel_spacing)
#     h_avg = sum(voxel_spacing) / length(voxel_spacing)
#     radii = edt_field .* h_avg
#     thickness = radii .* 2.0f0
    
#     nx, ny, nz = size(radii)
    
#     # Iterate over interior voxels that sit on or near the medial ridge
#     for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
#         r = radii[i, j, k]
#         r <= h_avg && continue
        
#         # Check if (i,j,k) is a local maximum in at least one axis (medial ridge)
#         is_ridge = (r >= radii[i-1, j, k] && r >= radii[i+1, j, k]) ||
#                    (r >= radii[i, j-1, k] && r >= radii[i, j+1, k]) ||
#                    (r >= radii[i, j, k-1] && r >= radii[i, j, k+1])
#         is_ridge || continue
        
#         # Stamp the sphere of diameter 2r onto all interior voxels within radius r
#         rad_vox = ceil(Int, r / h_avg)
#         r_sq = r^2
#         d = 2.0f0 * r
        
#         for dk in -rad_vox:rad_vox
#             kk = k + dk
#             (1 <= kk <= nz) || continue
#             for dj in -rad_vox:rad_vox
#                 jj = j + dj
#                 (1 <= jj <= ny) || continue
#                 for di in -rad_vox:rad_vox
#                     ii = i + di
#                     (1 <= ii <= nx) || continue
                    
#                     # Only propagate to solid interior voxels
#                     edt_field[ii, jj, kk] > 0 || continue
                    
#                     dist_sq = (di * voxel_spacing[1])^2 + 
#                               (dj * voxel_spacing[2])^2 + 
#                               (dk * voxel_spacing[3])^2
                              
#                     if dist_sq <= r_sq && d > thickness[ii, jj, kk]
#                         thickness[ii, jj, kk] = d
#                     end
#                 end
#             end
#         end
#     end
    
#     return thickness
# end



"""
    edt_1d_parabolic!(f, d, v, z, n, inf_sq)

Felzenszwalb-Huttenlocher 1D exact squared Euclidean distance transform in-place.
Uses `Int64` 1D scratch buffers so `d[q] = diff^2 + f[v[k]]` can never overflow.
"""
function edt_1d_parabolic!(
    f::Vector{Int64},
    d::Vector{Int64},
    v::Vector{Int32},
    z::Vector{Float64},
    n::Int,
    inf_sq::Int64
)
    # Fast-path: if the entire 1D line has no boundary points, output stays inf_sq
    has_boundary = false
    @inbounds for q in 1:n
        if f[q] < inf_sq
            has_boundary = true
            break
        end
    end
    if !has_boundary
        @inbounds for q in 1:n
            d[q] = inf_sq
        end
        return nothing
    end

    k = 1
    @inbounds begin
        v[1] = Int32(1)
        z[1] = -Inf64
        z[2] = Inf64

        for q in 2:n
            fq = Float64(f[q])
            q2 = Float64(q * q)
            vk = Int(v[k])
            s = ((fq + q2) - Float64(f[vk] + vk * vk)) / Float64(2 * (q - vk))
            while s <= z[k]
                k -= 1
                vk = Int(v[k])
                s = ((fq + q2) - Float64(f[vk] + vk * vk)) / Float64(2 * (q - vk))
            end
            k += 1
            v[k] = Int32(q)
            z[k] = s
            z[k + 1] = Inf64
        end

        k = 1
        for q in 1:n
            while z[k + 1] < Float64(q)
                k += 1
            end
            vk = Int(v[k])
            diff = q - vk
            d[q] = Int64(diff * diff) + f[vk]
        end
    end
    return nothing
end

"""
    compute_squared_thickness_u16(voxel_grid::BitArray{3}) -> Array{UInt16, 3}

Computes the exact 3D Euclidean Distance Transform and inscribed-sphere local thickness
directly in squared voxel units (`UInt16`, 2 bytes/voxel) from a `BitArray{3}` mask.
"""
function compute_squared_thickness_u16(voxeldata::VoxelData)
    voxel_grid = voxeldata.grid
    nx, ny, nz = size(voxel_grid)
    INF_SQ = Int64(nx)^2 + Int64(ny)^2 + Int64(nz)^2 + Int64(1)
    MAX_U16 = Int64(typemax(UInt16))

    r_sq = Array{UInt16, 3}(undef, nx, ny, nz)
    @inbounds for idx in eachindex(voxel_grid, r_sq)
        r_sq[idx] = voxel_grid[idx] ? typemax(UInt16) : UInt16(0)
    end

    # Pass 1: Transform along X (per Y-Z row, multi-threaded)
    Threads.@threads for k in 1:nz
        f = Vector{Int64}(undef, nx)
        d = Vector{Int64}(undef, nx)
        v = Vector{Int32}(undef, nx)
        z = Vector{Float64}(undef, nx + 1)

        @inbounds for j in 1:ny
            for i in 1:nx
                f[i] = voxel_grid[i, j, k] ? INF_SQ : Int64(0)
            end
            edt_1d_parabolic!(f, d, v, z, nx, INF_SQ)
            for i in 1:nx
                r_sq[i, j, k] = UInt16(clamp(d[i], Int64(0), MAX_U16))
            end
        end
    end

    # Pass 2: Transform along Y (per X-Z column, multi-threaded)
    Threads.@threads for k in 1:nz
        f = Vector{Int64}(undef, ny)
        d = Vector{Int64}(undef, ny)
        v = Vector{Int32}(undef, ny)
        z = Vector{Float64}(undef, ny + 1)

        @inbounds for i in 1:nx
            for j in 1:ny
                val = Int64(r_sq[i, j, k])
                f[j] = val == MAX_U16 ? INF_SQ : val
            end
            edt_1d_parabolic!(f, d, v, z, ny, INF_SQ)
            for j in 1:ny
                r_sq[i, j, k] = UInt16(clamp(d[j], Int64(0), MAX_U16))
            end
        end
    end

    # Pass 3: Transform along Z (per X-Y pillar, multi-threaded)
    Threads.@threads for j in 1:ny
        f = Vector{Int64}(undef, nz)
        d = Vector{Int64}(undef, nz)
        v = Vector{Int32}(undef, nz)
        z = Vector{Float64}(undef, nz + 1)

        @inbounds for i in 1:nx
            for k in 1:nz
                val = Int64(r_sq[i, j, k])
                f[k] = val == MAX_U16 ? INF_SQ : val
            end
            edt_1d_parabolic!(f, d, v, z, nz, INF_SQ)
            for k in 1:nz
                r_sq[i, j, k] = UInt16(clamp(d[k], Int64(0), MAX_U16))
            end
        end
    end

    # Inscribed-Sphere Local Thickness (Hildebrand-Rüegsegger) in pure integer squared units
    thick_r_sq = copy(r_sq)

    @inbounds for k in 2:(nz - 1), j in 2:(ny - 1), i in 2:(nx - 1)
        rsq = Int64(r_sq[i, j, k])
        rsq <= 1 && continue

        is_ridge = (rsq >= Int64(r_sq[i-1, j, k]) && rsq >= Int64(r_sq[i+1, j, k])) ||
                   (rsq >= Int64(r_sq[i, j-1, k]) && rsq >= Int64(r_sq[i, j+1, k])) ||
                   (rsq >= Int64(r_sq[i, j, k-1]) && rsq >= Int64(r_sq[i, j, k+1]))
        is_ridge || continue

        rad_vox = ceil(Int, sqrt(Float64(rsq)))
        u_val = UInt16(clamp(rsq, Int64(0), MAX_U16))

        for dk in -rad_vox:rad_vox
            kk = k + dk
            (1 <= kk <= nz) || continue
            dk2 = dk * dk
            for dj in -rad_vox:rad_vox
                jj = j + dj
                (1 <= jj <= ny) || continue
                djk2 = dk2 + dj * dj
                djk2 > rsq && continue
                for di in -rad_vox:rad_vox
                    ii = i + di
                    (1 <= ii <= nx) || continue
                    if (djk2 + di * di) <= rsq && voxel_grid[ii, jj, kk]
                        if u_val > thick_r_sq[ii, jj, kk]
                            thick_r_sq[ii, jj, kk] = u_val
                        end
                    end
                end
            end
        end
    end

    # return (;thick_r_sq, r_sq)
    return TField(thick_r_sq, r_sq)
end


"""
    cartesian_upwind_3d(T_arr, i, j, k, F, dx, dy, dz)

Solves the exact 3D Cartesian upwind quadratic equation:
  max((T - Tx)/dx, 0)^2 + max((T - Ty)/dy, 0)^2 + max((T - Tz)/dz, 0)^2 = 1 / F^2
"""
function cartesian_upwind_3d(
    T_arr::Array{T, 3},
    i::Int, j::Int, k::Int,
    F::T,
    dx::T, dy::T, dz::T
) where {T<:AbstractFloat}
    nx, ny, nz = size(T_arr)
    inf_val = T(Inf)

    tx = min(i > 1 ? T_arr[i-1, j, k] : inf_val, i < nx ? T_arr[i+1, j, k] : inf_val)
    ty = min(j > 1 ? T_arr[i, j-1, k] : inf_val, j < ny ? T_arr[i, j+1, k] : inf_val)
    tz = min(k > 1 ? T_arr[i, j, k-1] : inf_val, k < nz ? T_arr[i, j, k+1] : inf_val)

    # Sort active upwind dimensions by arrival time: t1 <= t2 <= t3
    d1 = (tx, dx)
    d2 = (ty, dy)
    d3 = (tz, dz)
    if d1[1] > d2[1]; d1, d2 = d2, d1; end
    if d2[1] > d3[1]; d2, d3 = d3, d2; end
    if d1[1] > d2[1]; d1, d2 = d2, d1; end

    invF = inv(F)
    t1, h1 = d1
    !isfinite(t1) && return inf_val

    # 1D update
    t_cand = t1 + h1 * invF

    # 2D quadratic update
    t2, h2 = d2
    if isfinite(t2) && t2 < t_cand
        a = inv(h1^2) + inv(h2^2)
        b = -T(2) * (t1 / h1^2 + t2 / h2^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) - invF^2
        disc = b^2 - T(4) * a * c
        if disc >= zero(T)
            t_cand = (-b + sqrt(disc)) / (T(2) * a)
        end
    end

    # 3D quadratic update
    t3, h3 = d3
    if isfinite(t3) && t3 < t_cand
        a = inv(h1^2) + inv(h2^2) + inv(h3^2)
        b = -T(2) * (t1 / h1^2 + t2 / h2^2 + t3 / h3^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) + (t3^2 / h3^2) - invF^2
        disc = b^2 - T(4) * a * c
        if disc >= zero(T)
            t_cand = (-b + sqrt(disc)) / (T(2) * a)
        end
    end

    return t_cand
end

"""
    solve_eikonal_voxel(
    voxeldata::VoxelData{T},
    tfield::TField,
    gates::AbstractVector{<:ValveGate}; 
    min_thickness::Real = 0.1, #
    shift::Integer = 0
) where {T<:Real}

Solves the 3D Eikonal equation |∇T| = 1/F directly on interior solid voxels using
the `UInt16` squared-radius thickness volume `thick_r_sq`.

Computes Hele-Shaw speed on the fly without `sqrt` or allocating a Float thickness array:
    h² = 4 * (thick_r_sq << shift) * dx_mean²
    F  = max(h², min_thickness²)
"""
function solve_eikonal_voxel(
    voxeldata::VoxelData{T},
    tfield::TField,
    gates::AbstractVector{<:ValveGate}; 
    min_thickness::Real = 0.1, #
    shift::Integer = 0
) where {T<:Real}


    voxel_grid, voxel_spacing, corner = voxeldata.grid, voxeldata.voxel_spacing, voxeldata.corner
    thick_r_sq = tfield.thick_r_sq

    @assert size(voxel_grid) == size(thick_r_sq) "voxel_grid and thick_r_sq must have identical dimensions"

    # Infer working floating-point precision from inputs
    T_gate = isempty(gates) ? Float32 : eltype(gates[1].open_time)
    

    nx, ny, nz = size(voxel_grid)
    dx, dy, dz = T(voxel_spacing[1]), T(voxel_spacing[2]), T(voxel_spacing[3])
    ox, oy, oz = T(corner[1]), T(corner[2]), T(corner[3])

    # Precompute constant physical scaling factor: 4 * 2^shift * dx_mean^2
    dx_mean = (dx + dy + dz) / T(3)
    speed_scale = T(4) * T(1 << shift) * (dx_mean * dx_mean)
    min_F = T(min_thickness)^2

    fill_time = fill(T(Inf), nx, ny, nz)
    accepted = falses(nx, ny, nz)
    pq = PriorityQueue{CartesianIndex{3}, T}()

    # Locate nearest interior solid voxel to each gate coordinate via O(1) direct index
    # lookup followed by an expanding spherical/cubic shell search
    function find_nearest_interior_voxel(pos)
        px, py, pz = T(pos[1]), T(pos[2]), T(pos[3])

        # Direct index mapping (voxel centers sit at origin + (idx - 0.5) * spacing)
        i0 = clamp(round(Int, (px - ox) / dx + T(0.5)), 1, nx)
        j0 = clamp(round(Int, (py - oy) / dy + T(0.5)), 1, ny)
        k0 = clamp(round(Int, (pz - oz) / dz + T(0.5)), 1, nz)

        # Fast-path: target voxel itself is inside the solid
        @inbounds if voxel_grid[i0, j0, k0]
            return CartesianIndex(i0, j0, k0)
        end

        # Expand outward in concentric shells r = 1, 2, ... up to a local search radius
        max_shell = min(max(nx, ny, nz), 32)
        best_idx = CartesianIndex(i0, j0, k0)
        best_dist_sq = T(Inf)

        @inbounds for r in 1:max_shell
            i_min, i_max = max(1, i0 - r), min(nx, i0 + r)
            j_min, j_max = max(1, j0 - r), min(ny, j0 + r)
            k_min, k_max = max(1, k0 - r), min(nz, k0 + r)

            for k in k_min:k_max, j in j_min:j_max, i in i_min:i_max
                # Only inspect the outer shell surface (skip interior already checked in r-1)
                (abs(i - i0) == r || abs(j - j0) == r || abs(k - k0) == r) || continue
                voxel_grid[i, j, k] || continue

                vx = ox + (T(i) - T(0.5)) * dx
                vy = oy + (T(j) - T(0.5)) * dy
                vz = oz + (T(k) - T(0.5)) * dz
                d2 = (vx - px)^2 + (vy - py)^2 + (vz - pz)^2

                if d2 < best_dist_sq
                    best_dist_sq = d2
                    best_idx = CartesianIndex(i, j, k)
                end
            end

            # Once we find at least one hit in shell r, check one more shell (r + 1) to guarantee
            # true Euclidean closest match across corners/anisotropic spacings, then return
            if isfinite(best_dist_sq)
                r_next = r + 1
                if r_next <= max_shell
                    for k in max(1, k0 - r_next):min(nz, k0 + r_next),
                        j in max(1, j0 - r_next):min(ny, j0 + r_next),
                        i in max(1, i0 - r_next):min(nx, i0 + r_next)
                        (abs(i - i0) == r_next || abs(j - j0) == r_next || abs(k - k0) == r_next) || continue
                        voxel_grid[i, j, k] || continue

                        vx = ox + (T(i) - T(0.5)) * dx
                        vy = oy + (T(j) - T(0.5)) * dy
                        vz = oz + (T(k) - T(0.5)) * dz
                        d2 = (vx - px)^2 + (vy - py)^2 + (vz - pz)^2
                        if d2 < best_dist_sq
                            best_dist_sq = d2
                            best_idx = CartesianIndex(i, j, k)
                        end
                    end
                end
                return best_idx
            end
        end

        # Fallback full-grid scan only if gate was placed > 32 voxels away from any solid
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            voxel_grid[i, j, k] || continue
            vx = ox + (T(i) - T(0.5)) * dx
            vy = oy + (T(j) - T(0.5)) * dy
            vz = oz + (T(k) - T(0.5)) * dz
            d2 = (vx - px)^2 + (vy - py)^2 + (vz - pz)^2
            if d2 < best_dist_sq
                best_dist_sq = d2
                best_idx = CartesianIndex(i, j, k)
            end
        end
        return best_idx
    end

    # Seed gates (supports cascaded opening times t_open >= 0)
    for gate in gates
        t_open = T(gate.open_time)
        @assert t_open >= zero(T) "Gate opening time must be >= 0"
        idx = find_nearest_interior_voxel(gate.position)
        if t_open < fill_time[idx]
            fill_time[idx] = t_open
            pq[idx] = t_open
        end
    end

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    @inbounds while !isempty(pq)
        curr, _ = dequeue_pair!(pq)
        accepted[curr] && continue
        accepted[curr] = true

        for off in offsets
            nbr = curr + off
            i, j, k = Tuple(nbr)
            (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
            (voxel_grid[i, j, k] && !accepted[i, j, k]) || continue

            # On-the-fly Hele-Shaw speed directly from UInt16 squared voxel radius
            F_local = max(T(thick_r_sq[i, j, k]) * speed_scale, min_F)
            t_cand = cartesian_upwind_3d(fill_time, i, j, k, F_local, dx, dy, dz)

            if t_cand < fill_time[i, j, k]
                fill_time[i, j, k] = t_cand
                pq[nbr] = t_cand
            end
        end
    end

    return fill_time
end




# Physical wall thickness h at voxel (i, j, k):
@inline get_thickness(thick_r_sq::Array{UInt16, 3}, i, j, k, dx::T) where {T<:AbstractFloat} =
    T(2) * sqrt(T(thick_r_sq[i, j, k])) * dx

# Bonus: Eikonal Hele-Shaw speed F = h² needs NO sqrt() at all!
# Since h = 2 * sqrt(r_sq) * dx,  h² = 4 * r_sq * dx²
@inline get_hele_shaw_speed(thick_r_sq::Array{UInt16, 3}, i, j, k, dx::T) where {T<:AbstractFloat} =
    T(4) * T(thick_r_sq[i, j, k]) * (dx * dx)




"""
    find_closest_interior_voxel(ray, meshdata, voxeldata; max_march_dist=nothing)

1. Casts `ray` against `meshdata.tlas` to find the closest surface hit point `p_hit`.
2. Traverses `voxeldata.grid` along `ray.d` from `p_hit` inward using exact 3D DDA
   voxel stepping until it enters the first interior solid voxel (`true`).

Returns `(; voxel_idx::CartesianIndex{3}, hit_point::SVector{3, T}, hit_found::Bool)`
or `nothing` if the ray misses the mesh or passes through without hitting a solid voxel.
"""
function find_closest_interior_voxel(
    ray::Raycore.AbstractRay,
    meshdata::MeshData,
    voxeldata::VoxelData;
    max_march_dist::Union{Nothing, Real} = nothing
)
    bvh  = meshdata.tlas
    grid = voxeldata.grid

    
    spacing = voxeldata.voxel_spacing
    corner  = voxeldata.corner

    T = float(promote_type(eltype(ray.o), eltype(ray.d), eltype(spacing), eltype(corner)))
    nx, ny, nz = size(grid)

    dx, dy, dz = T(spacing[1]), T(spacing[2]), T(spacing[3])
    ox, oy, oz = T(corner[1]),  T(corner[2]),  T(corner[3])

    # 1. Raycast against the surface BVH to find the surface entry point
    hit_found, _, dist, _, _ = Raycore.closest_hit(bvh, ray)
    hit_found || return nothing

    ro = SVector{3, T}(ray.o)
    rd = normalize(SVector{3, T}(ray.d))
    p_hit = ro + rd * T(dist)

    # Maximum distance to march inside the voxel grid (defaults to grid diagonal)
    max_dist = isnothing(max_march_dist) ?
        sqrt((nx * dx)^2 + (ny * dy)^2 + (nz * dz)^2) : T(max_march_dist)

    # 2. Initialize 3D DDA traversal at `p_hit`
    # Voxel (i, j, k) occupies [ox + (i-1)*dx, ox + i*dx] × ...
    gx = (p_hit[1] - ox) / dx
    gy = (p_hit[2] - oy) / dy
    gz = (p_hit[3] - oz) / dz

    i = clamp(floor(Int, gx) + 1, 1, nx)
    j = clamp(floor(Int, gy) + 1, 1, ny)
    k = clamp(floor(Int, gz) + 1, 1, nz)

    # Fast-path: `p_hit` already falls directly inside an interior voxel
    @inbounds if grid[i, j, k]
        return (; voxel_idx = CartesianIndex(i, j, k), hit_point = p_hit, hit_found = true)
    end

    # Step directions (+1 or -1) and parametric t increments per voxel cell
    step_x = rd[1] > zero(T) ? 1 : (rd[1] < zero(T) ? -1 : 0)
    step_y = rd[2] > zero(T) ? 1 : (rd[2] < zero(T) ? -1 : 0)
    step_z = rd[3] > zero(T) ? 1 : (rd[3] < zero(T) ? -1 : 0)

    t_delta_x = step_x != 0 ? abs(dx / rd[1]) : T(Inf)
    t_delta_y = step_y != 0 ? abs(dy / rd[2]) : T(Inf)
    t_delta_z = step_z != 0 ? abs(dz / rd[3]) : T(Inf)

    # Distance from `p_hit` to the first voxel boundary along each axis
    t_max_x = if step_x > 0
        ((ox + T(i) * dx) - p_hit[1]) / rd[1]
    elseif step_x < 0
        ((ox + T(i - 1) * dx) - p_hit[1]) / rd[1]
    else
        T(Inf)
    end

    t_max_y = if step_y > 0
        ((oy + T(j) * dy) - p_hit[2]) / rd[2]
    elseif step_y < 0
        ((oy + T(j - 1) * dy) - p_hit[2]) / rd[2]
    else
        T(Inf)
    end

    t_max_z = if step_z > 0
        ((oz + T(k) * dz) - p_hit[3]) / rd[3]
    elseif step_z < 0
        ((oz + T(k - 1) * dz) - p_hit[3]) / rd[3]
    else
        T(Inf)
    end

    # 3. Walk voxel-by-voxel along `rd` until entering the first `true` voxel
    t_curr = zero(T)
    @inbounds while t_curr <= max_dist
        if t_max_x < t_max_y
            if t_max_x < t_max_z
                t_curr = t_max_x
                i += step_x
                t_max_x += t_delta_x
            else
                t_curr = t_max_z
                k += step_z
                t_max_z += t_delta_z
            end
        else
            if t_max_y < t_max_z
                t_curr = t_max_y
                j += step_y
                t_max_y += t_delta_y
            else
                t_curr = t_max_z
                k += step_z
                t_max_z += t_delta_z
            end
        end

        (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || break

        if grid[i, j, k]
            return (; voxel_idx = CartesianIndex(i, j, k), hit_point = p_hit, hit_found = true)
        end
    end

    # 4. Grazing-ray fallback: if the ray skimmed a thin edge/corner without entering a voxel center,
    #    check the 3×3×3 neighborhood around the initial surface hit voxel
    i0 = clamp(floor(Int, gx) + 1, 1, nx)
    j0 = clamp(floor(Int, gy) + 1, 1, ny)
    k0 = clamp(floor(Int, gz) + 1, 1, nz)
    best_idx = CartesianIndex(i0, j0, k0)
    best_d2 = T(Inf)

    @inbounds for kk in max(1, k0-1):min(nz, k0+1),
                  jj in max(1, j0-1):min(ny, j0+1),
                  ii in max(1, i0-1):min(nx, i0+1)
        grid[ii, jj, kk] || continue
        vx = ox + (T(ii) - T(0.5)) * dx
        vy = oy + (T(jj) - T(0.5)) * dy
        vz = oz + (T(kk) - T(0.5)) * dz
        d2 = (vx - p_hit[1])^2 + (vy - p_hit[2])^2 + (vz - p_hit[3])^2
        if d2 < best_d2
            best_d2 = d2
            best_idx = CartesianIndex(ii, jj, kk)
        end
    end

    return isfinite(best_d2) ?
        (; voxel_idx = best_idx, hit_point = p_hit, hit_found = true) :
        nothing
end



"""
    analyze_eikonal_field(
    fill_time::AbstractArray{T, 3}, #from eikonal
    voxeldata::VoxelData, #from voxelize
    tfield::TField #from compute_squared_thickness_u16
    ;
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05,
    hesitation_quantile::Real = 0.15,
    core_exponent::Real = 2.0,
    div_scale::Real = 5.0,
    max_shear_tilt::Real = 0.08
) where {T<:Real}

Extracts diagnostic `BitArray{3}` masks, unit flow direction vectors `û = ∇T / |∇T|`,
and the full 6-component symmetric Fiber Orientation Tensor `(a11, a22, a33, a12, a13, a23)`
directly on the 3D voxel grid.
"""
function analyze_eikonal_field(
    fill_time::AbstractArray{T, 3}, #from eikonal
    voxeldata::VoxelData, #from voxelize
    tfield::TField #from compute_squared_thickness_u16
    ;
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05,
    hesitation_quantile::Real = 0.15,
    core_exponent::Real = 2.0,
    div_scale::Real = 5.0,
    max_shear_tilt::Real = 0.08
) where {T<:Real}

    voxel_grid = voxeldata.grid
    voxel_spacing = voxeldata.voxel_spacing    

    thick_r_sq, r_sq = tfield

    nx, ny, nz = size(voxel_grid)
    dx, dy, dz = T(voxel_spacing[1]), T(voxel_spacing[2]), T(voxel_spacing[3])

    # 1. Compute interior-only 3D gradient ∇T and unit flow direction û
    flow_dirs = fill(zero(SVector{3, T}), nx, ny, nz)
    front_speed = zeros(T, nx, ny, nz)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        (voxel_grid[i, j, k] && isfinite(fill_time[i, j, k])) || continue
        tc = T(fill_time[i, j, k])

        xm = i > 1  && voxel_grid[i-1, j, k] && isfinite(fill_time[i-1, j, k])
        xp = i < nx && voxel_grid[i+1, j, k] && isfinite(fill_time[i+1, j, k])
        gx = (xm && xp) ? (T(fill_time[i+1, j, k]) - T(fill_time[i-1, j, k])) / (T(2) * dx) :
             xp         ? (T(fill_time[i+1, j, k]) - tc) / dx :
             xm         ? (tc - T(fill_time[i-1, j, k])) / dx : zero(T)

        ym = j > 1  && voxel_grid[i, j-1, k] && isfinite(fill_time[i, j-1, k])
        yp = j < ny && voxel_grid[i, j+1, k] && isfinite(fill_time[i, j+1, k])
        gy = (ym && yp) ? (T(fill_time[i, j+1, k]) - T(fill_time[i, j-1, k])) / (T(2) * dy) :
             yp         ? (T(fill_time[i, j+1, k]) - tc) / dy :
             ym         ? (tc - T(fill_time[i, j-1, k])) / dy : zero(T)

        zm = k > 1  && voxel_grid[i, j, k-1] && isfinite(fill_time[i, j, k-1])
        zp = k < nz && voxel_grid[i, j, k+1] && isfinite(fill_time[i, j, k+1])
        gz = (zm && zp) ? (T(fill_time[i, j, k+1]) - T(fill_time[i, j, k-1])) / (T(2) * dz) :
             zp         ? (T(fill_time[i, j, k+1]) - tc) / dz :
             zm         ? (tc - T(fill_time[i, j, k-1])) / dz : zero(T)

        g = SVector{3, T}(gx, gy, gz)
        gn = norm(g)
        if gn > T(1e-9)
            flow_dirs[i, j, k] = g / gn
            front_speed[i, j, k] = inv(gn)
        end
    end

    valid_speeds = [front_speed[idx] for idx in eachindex(voxel_grid) if voxel_grid[idx] && front_speed[idx] > zero(T)]
    speed_cutoff = isempty(valid_speeds) ? zero(T) :
        sort(valid_speeds)[max(1, round(Int, T(hesitation_quantile) * length(valid_speeds)))]

    # 2. Allocate BitArray{3} diagnostic masks & all 6 FOT components
    weld_lines      = BitArray{3}(undef, nx, ny, nz); fill!(weld_lines, false)
    meld_lines      = BitArray{3}(undef, nx, ny, nz); fill!(meld_lines, false)
    diverging_flow  = BitArray{3}(undef, nx, ny, nz); fill!(diverging_flow, false)
    converging_flow = BitArray{3}(undef, nx, ny, nz); fill!(converging_flow, false)
    air_traps       = BitArray{3}(undef, nx, ny, nz); fill!(air_traps, false)
    hesitation      = BitArray{3}(undef, nx, ny, nz); fill!(hesitation, false)

    a11 = zeros(T, nx, ny, nz)
    a22 = zeros(T, nx, ny, nz)
    a33 = zeros(T, nx, ny, nz)
    a12 = zeros(T, nx, ny, nz)
    a13 = zeros(T, nx, ny, nz)
    a23 = zeros(T, nx, ny, nz)

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    w_thresh = T(weld_dot_thresh)
    m_thresh = T(meld_dot_thresh)
    d_thresh = T(div_thresh)
    p_exp    = T(core_exponent)
    d_scale  = T(div_scale)
    s_tilt   = T(max_shear_tilt)

    # 3. Single-pass extraction of masks and full 6-component 3D FOT
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        (voxel_grid[i, j, k] && isfinite(fill_time[i, j, k])) || continue

        tc = T(fill_time[i, j, k])
        u_c = flow_dirs[i, j, k]
        has_flow = norm(u_c) > T(1e-6)

        min_dot = one(T)
        is_local_max = true
        has_valid_nbr = false

        for off in offsets
            ii, jj, kk = i + off[1], j + off[2], k + off[3]
            (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
            (voxel_grid[ii, jj, kk] && isfinite(fill_time[ii, jj, kk])) || continue

            has_valid_nbr = true
            if T(fill_time[ii, jj, kk]) > tc
                is_local_max = false
            end

            u_n = flow_dirs[ii, jj, kk]
            if has_flow && norm(u_n) > T(1e-6)
                d_un = dot(u_c, u_n)
                if d_un < min_dot
                    min_dot = d_un
                end
            end
        end

        if has_valid_nbr && is_local_max && tc > zero(T)
            air_traps[i, j, k] = true
        end

        if has_flow
            if min_dot < w_thresh
                weld_lines[i, j, k] = true
            elseif min_dot < m_thresh
                meld_lines[i, j, k] = true
            end
        end

        if front_speed[i, j, k] > zero(T) && front_speed[i, j, k] <= speed_cutoff
            hesitation[i, j, k] = true
        end

        # Flow Divergence ∇ ⋅ û
        dux_dx = (i > 1 && i < nx && voxel_grid[i-1,j,k] && voxel_grid[i+1,j,k]) ?
                 (flow_dirs[i+1,j,k][1] - flow_dirs[i-1,j,k][1]) / (T(2) * dx) : zero(T)
        duy_dy = (j > 1 && j < ny && voxel_grid[i,j-1,k] && voxel_grid[i,j+1,k]) ?
                 (flow_dirs[i,j+1,k][2] - flow_dirs[i,j-1,k][2]) / (T(2) * dy) : zero(T)
        duz_dz = (k > 1 && k < nz && voxel_grid[i,j,k-1] && voxel_grid[i,j,k+1]) ?
                 (flow_dirs[i,j,k+1][3] - flow_dirs[i,j,k-1][3]) / (T(2) * dz) : zero(T)

        div_u = dux_dx + duy_dy + duz_dz
        if div_u > d_thresh
            diverging_flow[i, j, k] = true
        elseif div_u < -d_thresh
            converging_flow[i, j, k] = true
        end

        if !has_flow
            a11[i, j, k] = T(1) / T(3)
            a22[i, j, k] = T(1) / T(3)
            a33[i, j, k] = T(1) / T(3)
            continue
        end

        # Local orthonormal frame (e1 = flow, e2 = in-plane transverse, e3 = wall normal)
        e1 = u_c
        if !isnothing(r_sq) && i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
            gn_x = T(Int32(r_sq[i+1,j,k]) - Int32(r_sq[i-1,j,k]))
            gn_y = T(Int32(r_sq[i,j+1,k]) - Int32(r_sq[i,j-1,k]))
            gn_z = T(Int32(r_sq[i,j,k+1]) - Int32(r_sq[i,j,k-1]))
            n_raw = SVector{3, T}(gn_x, gn_y, gn_z)
            n_ortho = n_raw - dot(n_raw, e1) * e1
            if norm(n_ortho) > T(1e-5)
                e3 = normalize(n_ortho)
            else
                ref = abs(e1[1]) < T(0.9) ? SVector{3, T}(1, 0, 0) : SVector{3, T}(0, 1, 0)
                e3 = normalize(cross(e1, ref))
            end
        else
            ref = abs(e1[1]) < T(0.9) ? SVector{3, T}(1, 0, 0) : SVector{3, T}(0, 1, 0)
            e3 = normalize(cross(e1, ref))
        end
        e2 = cross(e3, e1)

        # Normalized through-thickness depth ζ ∈ [0 (core), 1 (mold wall skin)]
        ζ = if !isnothing(r_sq) && thick_r_sq[i, j, k] > UInt16(0)
            ratio = clamp(T(r_sq[i, j, k]) / T(thick_r_sq[i, j, k]), zero(T), one(T))
            one(T) - sqrt(ratio)
        else
            T(0.5)
        end

        # Local tensor components (λ_flow, λ_trans, λ_norm, λ_13)
        if weld_lines[i, j, k]
            λ_flow, λ_trans, λ_norm, λ_13 = T(0.15), T(0.78), T(0.07), zero(T)
        else
            α_div = T(0.5) * (one(T) + tanh(div_u * d_scale))
            c_flow  = (one(T) - α_div) * T(0.72) + α_div * T(0.18)
            c_trans = (one(T) - α_div) * T(0.20) + α_div * T(0.74)
            c_norm  = T(0.08)

            s_flow, s_trans, s_norm = T(0.84), T(0.12), T(0.04)

            w_shear = ζ^p_exp
            λ_flow  = (one(T) - w_shear) * c_flow  + w_shear * s_flow
            λ_trans = (one(T) - w_shear) * c_trans + w_shear * s_trans
            λ_norm  = (one(T) - w_shear) * c_norm  + w_shear * s_norm
            # Out-of-plane shear tilt peaks in the high-shear transition layer (ζ ≈ 0.7)
            λ_13    = s_tilt * T(4) * ζ * (one(T) - ζ) * w_shear
        end

        # Full symmetric tensor rotation: A_global = R * A_local * Rᵀ
        # Diagonal terms:
        a11[i, j, k] = λ_flow * e1[1]^2 + λ_trans * e2[1]^2 + λ_norm * e3[1]^2 +
                       T(2) * λ_13 * e1[1] * e3[1]
        a22[i, j, k] = λ_flow * e1[2]^2 + λ_trans * e2[2]^2 + λ_norm * e3[2]^2 +
                       T(2) * λ_13 * e1[2] * e3[2]
        a33[i, j, k] = λ_flow * e1[3]^2 + λ_trans * e2[3]^2 + λ_norm * e3[3]^2 +
                       T(2) * λ_13 * e1[3] * e3[3]

        # Off-diagonal shear terms:
        a12[i, j, k] = λ_flow * e1[1]*e1[2] + λ_trans * e2[1]*e2[2] + λ_norm * e3[1]*e3[2] +
                       λ_13 * (e1[1]*e3[2] + e3[1]*e1[2])
        a13[i, j, k] = λ_flow * e1[1]*e1[3] + λ_trans * e2[1]*e2[3] + λ_norm * e3[1]*e3[3] +
                       λ_13 * (e1[1]*e3[3] + e3[1]*e1[3])
        a23[i, j, k] = λ_flow * e1[2]*e1[3] + λ_trans * e2[2]*e2[3] + λ_norm * e3[2]*e3[3] +
                       λ_13 * (e1[2]*e3[3] + e3[2]*e1[3])
    end

    masks = (;
        weld_lines,
        meld_lines,
        diverging_flow,
        converging_flow,
        air_traps,
        hesitation
    )

    fot = (; a11, a22, a33, a12, a13, a23)

    return (; masks, flow_dirs, front_speed, fot)
end


# Helper to extract world coordinates of all `true` voxels in any BitArray{3} mask
function mask_points(mask::BitArray{3}, spacing, corner)
    pts = Point3f[]
    for idx in findall(mask)
        i, j, k = Tuple(idx)
        push!(pts, Point3f(
            corner[1] + (i - 0.5f0) * spacing[1],
            corner[2] + (j - 0.5f0) * spacing[2],
            corner[3] + (k - 0.5f0) * spacing[3]
        ))
    end
    return pts
end

# # Plot weld lines (red) and air traps (yellow) over a semi-transparent part mesh
# mesh!(ax, meshdata.mesh, color = (:lightgray, 0.25), transparency = true)
# scatter!(ax, mask_points(res.masks.weld_lines, voxeldata.stepsize, voxeldata.corner), color = :red, markersize = 6)
# scatter!(ax, mask_points(res.masks.air_traps,  voxeldata.stepsize, voxeldata.corner), color = :gold, markersize = 10)