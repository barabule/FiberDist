



"""
    VoxelData(meshdata; res=100, padding=0.05)

Voxelizes a Raycore acceleration structure (`meshdata.tlas`, `meshdata.mesh`) into a 
3D `BitArray` using multi-threaded Z-axis ray-casting parity (even/odd rule).
Returns `(; grid, voxel_spacing, corner)`.
"""
function VoxelData(
    meshdata::MeshData;
    fullres = false, #if true the thickness is stored at full grid resolution, else at half res -> 1/8 the size!
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
    dx = max_span / T(res) # uniform cubic voxel size for now
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

    
    grid = voxel_grid
    voxel_spacing = (dx, dy, dz)
    corner = corner
    if fullres
        thickness = compute_squared_thickness_u16(grid)
    else
        thickness = compute_halfres_thickness(grid)
    end
    return VoxelData(grid, thickness, voxel_spacing, corner)
end



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
    compute_squared_thickness_u16(voxel_grid)

Computes the exact 3D Euclidean Distance Transform and Hildebrand–Rüegsegger
inscribed-sphere local thickness directly in squared voxel units (`UInt16`),
skipping empty 1D rows/columns/pillars in all three EDT passes and discarding
the temporary boundary distance buffer upon completion.
"""
function compute_squared_thickness_u16(voxel_grid)
    # voxel_grid = voxeldata.grid
    nx, ny, nz = size(voxel_grid)
    INF_SQ = Int64(nx)^2 + Int64(ny)^2 + Int64(nz)^2 + Int64(1)
    MAX_U16 = Int64(typemax(UInt16))

    # Temporary boundary distance buffer (released when function returns)
    r_sq = zeros(UInt16, nx, ny, nz)

    # Pass 1: Transform along X (per Y-Z row, multi-threaded, skipping empty rows)
    Threads.@threads for k in 1:nz
        f = Vector{Int64}(undef, nx)
        d = Vector{Int64}(undef, nx)
        v = Vector{Int32}(undef, nx)
        z = Vector{Float64}(undef, nx + 1)

        @inbounds for j in 1:ny
            any_solid = false
            for i in 1:nx
                if voxel_grid[i, j, k]
                    f[i] = INF_SQ
                    any_solid = true
                else
                    f[i] = Int64(0)
                end
            end
            any_solid || continue

            edt_1d_parabolic!(f, d, v, z, nx, INF_SQ)
            for i in 1:nx
                r_sq[i, j, k] = UInt16(clamp(d[i], Int64(0), MAX_U16))
            end
        end
    end

    # Pass 2: Transform along Y (per X-Z column, multi-threaded, skipping empty columns)
    Threads.@threads for k in 1:nz
        f = Vector{Int64}(undef, ny)
        d = Vector{Int64}(undef, ny)
        v = Vector{Int32}(undef, ny)
        z = Vector{Float64}(undef, ny + 1)

        @inbounds for i in 1:nx
            any_solid = false
            for j in 1:ny
                val = Int64(r_sq[i, j, k])
                if val > 0
                    f[j] = val == MAX_U16 ? INF_SQ : val
                    any_solid = true
                else
                    f[j] = Int64(0)
                end
            end
            any_solid || continue

            edt_1d_parabolic!(f, d, v, z, ny, INF_SQ)
            for j in 1:ny
                r_sq[i, j, k] = UInt16(clamp(d[j], Int64(0), MAX_U16))
            end
        end
    end

    # Pass 3: Transform along Z (per X-Y pillar, multi-threaded, skipping empty pillars)
    Threads.@threads for j in 1:ny
        f = Vector{Int64}(undef, nz)
        d = Vector{Int64}(undef, nz)
        v = Vector{Int32}(undef, nz)
        z = Vector{Float64}(undef, nz + 1)

        @inbounds for i in 1:nx
            any_solid = false
            for k in 1:nz
                val = Int64(r_sq[i, j, k])
                if val > 0
                    f[k] = val == MAX_U16 ? INF_SQ : val
                    any_solid = true
                else
                    f[k] = Int64(0)
                end
            end
            any_solid || continue

            edt_1d_parabolic!(f, d, v, z, nz, INF_SQ)
            for k in 1:nz
                r_sq[i, j, k] = UInt16(clamp(d[k], Int64(0), MAX_U16))
            end
        end
    end

    # Inscribed-Sphere Local Thickness (Hildebrand–Rüegsegger) in pure integer squared units
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

    return thick_r_sq
end



"""
    edt_1d_parabolic_strided!(f, d, v, z, n, stride::Int, inf_val::Int64)

1D Felzenszwalb–Huttenlocher parabolic EDT where adjacent array elements are separated
by `stride` fine-voxel units (so squared distance between index `p` and `q` is `(stride * (p - q))²`).
"""
function edt_1d_parabolic_strided!(
    f::Vector{Int64},
    d::Vector{Int64},
    v::Vector{Int32},
    z::Vector{Float64},
    n::Int,
    stride::Int,
    inf_val::Int64
)
    s2 = Int64(stride) * Int64(stride)
    inv_2s2 = 1.0 / (2.0 * Float64(s2))

    k = 1
    @inbounds begin
        v[1] = Int32(1)
        z[1] = -Inf
        z[2] = Inf

        for q in 2:n
            fq = f[q]
            fq >= inf_val && continue

            vk = Int(v[k])
            s = ((fq + s2 * Int64(q) * Int64(q)) - (f[vk] + s2 * Int64(vk) * Int64(vk))) * inv_2s2 / Float64(q - vk)
            while s <= z[k]
                k -= 1
                vk = Int(v[k])
                s = ((fq + s2 * Int64(q) * Int64(q)) - (f[vk] + s2 * Int64(vk) * Int64(vk))) * inv_2s2 / Float64(q - vk)
            end
            k += 1
            v[k] = Int32(q)
            z[k] = s
            z[k + 1] = Inf
        end

        k = 1
        for q in 1:n
            while z[k + 1] < Float64(q)
                k += 1
            end
            vk = Int(v[k])
            diff = Int64(q - vk)
            d[q] = s2 * diff * diff + f[vk]
        end
    end
    return nothing
end

"""
    compute_halfres_thickness(grid::BitArray{3}) -> HalfResThickness

Computes 3D inscribed-sphere local thickness on a 1/2-resolution grid (`cld.(size(grid), 2)`)
**without** dilating the solid mask:
- Evaluates exact fine-grid boundary distances along X at full resolution, taking the minimum
  over each 2×2 Y-Z fine fiber so thin walls are never overestimated.
- Propagates Y and Z parabolic envelopes with `stride = 2` directly in fine squared-voxel units.
- Runs Hildebrand–Rüegsegger ridge propagation on the 1/8-size coarse grid.
"""
function compute_halfres_thickness(grid::BitArray{3})
    full_dims = size(grid)
    fnx, fny, fnz = full_dims
    cnx, cny, cnz = cld(fnx, 2), cld(fny, 2), cld(fnz, 2)

    INF_SQ  = Int64(fnx)^2 + Int64(fny)^2 + Int64(fnz)^2 + Int64(1)
    MAX_U16 = Int64(typemax(UInt16))

    r_sq        = fill(typemax(UInt16), cnx, cny, cnz)
    coarse_mask = falses(cnx, cny, cnz)

    # Pass 1: Full-resolution X-EDT, strided into coarse (cnx, cny, cnz) without dilation
    Threads.@threads for ck in 1:cnz
        f = Vector{Int64}(undef, fnx)
        d = Vector{Int64}(undef, fnx)
        v = Vector{Int32}(undef, fnx)
        z = Vector{Float64}(undef, fnx + 1)

        k0, k1 = 2ck - 1, min(2ck, fnz)
        @inbounds for cj in 1:cny
            j0, j1 = 2cj - 1, min(2cj, fny)

            # Initialize coarse cells to INF_SQ where any fine voxel in the 2x2x2 block is solid
            for ci in 1:cnx
                i0, i1 = 2ci - 1, min(2ci, fnx)
                has_solid = false
                for fk in k0:k1, fj in j0:j1, fi in i0:i1
                    if grid[fi, fj, fk]
                        has_solid = true
                        break
                    end
                end
                coarse_mask[ci, cj, ck] = has_solid
                r_sq[ci, cj, ck] = has_solid ? typemax(UInt16) : UInt16(0)
            end

            # Scan all fine X-lines in this 2x2 Y-Z bundle and keep the tightest wall distance
            for fk in k0:k1, fj in j0:j1
                any_solid = false
                for fi in 1:fnx
                    if grid[fi, fj, fk]
                        f[fi] = INF_SQ
                        any_solid = true
                    else
                        f[fi] = Int64(0)
                    end
                end
                if !any_solid
                    for ci in 1:cnx
                        r_sq[ci, cj, ck] = UInt16(0)
                    end
                    continue
                end

                edt_1d_parabolic_strided!(f, d, v, z, fnx, 1, INF_SQ)

                for ci in 1:cnx
                    coarse_mask[ci, cj, ck] || continue
                    i0, i1 = 2ci - 1, min(2ci, fnx)
                    # Take the max distance among the 2 fine X-voxels (captures centerline of thin X-walls)
                    # while taking min across the 2x2 Y-Z fibers (never overestimates Y/Z walls)
                    d_x = i1 > i0 ? max(d[i0], d[i1]) : d[i0]
                    u_x = UInt16(clamp(d_x, Int64(0), MAX_U16))
                    if u_x < r_sq[ci, cj, ck]
                        r_sq[ci, cj, ck] = u_x
                    end
                end
            end
        end
    end

    # Pass 2: Y-axis on coarse grid with stride = 2 (in fine squared-voxel units)
    Threads.@threads for ck in 1:cnz
        f = Vector{Int64}(undef, cny)
        d = Vector{Int64}(undef, cny)
        v = Vector{Int32}(undef, cny)
        z = Vector{Float64}(undef, cny + 1)

        @inbounds for ci in 1:cnx
            any_solid = false
            for cj in 1:cny
                val = Int64(r_sq[ci, cj, ck])
                if val > 0
                    f[cj] = val == MAX_U16 ? INF_SQ : val
                    any_solid = true
                else
                    f[cj] = Int64(0)
                end
            end
            any_solid || continue

            edt_1d_parabolic_strided!(f, d, v, z, cny, 2, INF_SQ)
            for cj in 1:cny
                r_sq[ci, cj, ck] = UInt16(clamp(d[cj], Int64(0), MAX_U16))
            end
        end
    end

    # Pass 3: Z-axis on coarse grid with stride = 2 (in fine squared-voxel units)
    Threads.@threads for cj in 1:cny
        f = Vector{Int64}(undef, cnz)
        d = Vector{Int64}(undef, cnz)
        v = Vector{Int32}(undef, cnz)
        z = Vector{Float64}(undef, cnz + 1)

        @inbounds for ci in 1:cnx
            any_solid = false
            for ck in 1:cnz
                val = Int64(r_sq[ci, cj, ck])
                if val > 0
                    f[ck] = val == MAX_U16 ? INF_SQ : val
                    any_solid = true
                else
                    f[ck] = Int64(0)
                end
            end
            any_solid || continue

            edt_1d_parabolic_strided!(f, d, v, z, cnz, 2, INF_SQ)
            for ck in 1:cnz
                r_sq[ci, cj, ck] = UInt16(clamp(d[ck], Int64(0), MAX_U16))
            end
        end
    end

    # 4. Hildebrand–Rüegsegger ridge propagation (radius in coarse cells = ceil(sqrt(rsq) / 2))
    thick_coarse = copy(r_sq)

    @inbounds for ck in 2:(cnz - 1), cj in 2:(cny - 1), ci in 2:(cnx - 1)
        rsq = Int64(r_sq[ci, cj, ck])
        rsq <= 1 && continue

        is_ridge = (rsq >= Int64(r_sq[ci-1, cj, ck]) && rsq >= Int64(r_sq[ci+1, cj, ck])) ||
                   (rsq >= Int64(r_sq[ci, cj-1, ck]) && rsq >= Int64(r_sq[ci, cj+1, ck])) ||
                   (rsq >= Int64(r_sq[ci, cj, ck-1]) && rsq >= Int64(r_sq[ci, cj, ck+1]))
        is_ridge || continue

        # Sphere radius in coarse grid units (each coarse step = 2 fine voxels)
        rad_coarse = ceil(Int, sqrt(Float64(rsq)) * 0.5)
        u_val = UInt16(clamp(rsq, Int64(0), MAX_U16))

        for dk in -rad_coarse:rad_coarse
            kk = ck + dk
            (1 <= kk <= cnz) || continue
            dk2 = 4 * dk * dk
            for dj in -rad_coarse:rad_coarse
                jj = cj + dj
                (1 <= jj <= cny) || continue
                djk2 = dk2 + 4 * dj * dj
                djk2 > rsq && continue
                for di in -rad_coarse:rad_coarse
                    ii = ci + di
                    (1 <= ii <= cnx) || continue
                    if (djk2 + 4 * di * di) <= rsq && coarse_mask[ii, jj, kk]
                        if u_val > thick_coarse[ii, jj, kk]
                            thick_coarse[ii, jj, kk] = u_val
                        end
                    end
                end
            end
        end
    end

    return HalfResThickness(thick_coarse, full_dims)
end




# Locate nearest interior solid voxel to each gate coordinate via O(1) direct index
# lookup followed by an expanding spherical/cubic shell search
function find_nearest_interior_voxel(pos, voxeldata::VoxelData{T}) where T
    ox, oy, oz = voxeldata.corner
    voxel_grid = voxeldata.grid
    dx, dy, dz = voxeldata.voxel_spacing
    nx, ny, nz = size(voxel_grid)
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


# # Physical wall thickness h at voxel (i, j, k):
# @inline get_thickness(thick_r_sq::Array{UInt16, 3}, i, j, k, dx::T) where {T<:AbstractFloat} =
#     T(2) * sqrt(T(thick_r_sq[i, j, k])) * dx

# # Bonus: Eikonal Hele-Shaw speed F = h² needs NO sqrt() at all!
# # Since h = 2 * sqrt(r_sq) * dx,  h² = 4 * r_sq * dx²
# @inline get_hele_shaw_speed(thick_r_sq::Array{UInt16, 3}, i, j, k, dx::T) where {T<:AbstractFloat} =
#     T(4) * T(thick_r_sq[i, j, k]) * (dx * dx)




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


"""
    gates_to_voxels(
        gates::AbstractVector{ValveGate},
        voxeldata;
        fallback_search_radius_vox::Int = 3
    ) -> Vector{CartesianIndex{3}}

Maps a vector of `ValveGate`s onto active interior solid voxels in `voxeldata.grid`.
- Collects all solid voxels within `gate.radius` of `gate.position`.
- If `gate.radius == 0` or no solid voxel lies strictly within `gate.radius` (e.g., when
  picking directly on a surface triangle), snaps to the nearest solid voxel within
  `fallback_search_radius_vox`.
"""
function gates_to_voxels(
    gates::AbstractVector{ValveGate},
    voxeldata::VoxelData;
    fallback_search_radius_vox::Int = 3
)
    grid = voxeldata.grid
    nx, ny, nz = size(grid)

    spacing_raw = voxeldata.voxel_spacing
    corner_raw  = voxeldata.corner

    dx, dy, dz = Float32(spacing_raw[1]), Float32(spacing_raw[2]), Float32(spacing_raw[3])
    ox, oy, oz = Float32(corner_raw[1]),  Float32(corner_raw[2]),  Float32(corner_raw[3])

    gate_voxels = CartesianIndex{3}[]
    seen = Set{CartesianIndex{3}}()

    for gate in gates
        gx, gy, gz = gate.position[1], gate.position[2], gate.position[3]

        # Continuous voxel index corresponding to voxel centers at ox + (i - 0.5) * dx
        fi = (gx - ox) / dx + 0.5f0
        fj = (gy - oy) / dy + 0.5f0
        fk = (gz - oz) / dz + 0.5f0

        found_any = false
        r = max(gate.radius, 0.0f0)
        r2 = r * r

        if r > 0.0f0
            rx_vox = ceil(Int, r / dx) + 1
            ry_vox = ceil(Int, r / dy) + 1
            rz_vox = ceil(Int, r / dz) + 1

            ic, jc, kc = round(Int, fi), round(Int, fj), round(Int, fk)
            i_lo, i_hi = max(1, ic - rx_vox), min(nx, ic + rx_vox)
            j_lo, j_hi = max(1, jc - ry_vox), min(ny, jc + ry_vox)
            k_lo, k_hi = max(1, kc - rz_vox), min(nz, kc + rz_vox)

            @inbounds for k in k_lo:k_hi, j in j_lo:j_hi, i in i_lo:i_hi
                grid[i, j, k] || continue
                vx = ox + (Float32(i) - 0.5f0) * dx
                vy = oy + (Float32(j) - 0.5f0) * dy
                vz = oz + (Float32(k) - 0.5f0) * dz
                d2 = (vx - gx)^2 + (vy - gy)^2 + (vz - gz)^2

                if d2 <= r2
                    c = CartesianIndex(i, j, k)
                    if !(c in seen)
                        push!(seen, c)
                        push!(gate_voxels, c)
                    end
                    found_any = true
                end
            end
        end

        # Fallback: snap to nearest solid voxel (handles radius == 0 or surface-picked points)
        if !found_any
            ic = clamp(round(Int, fi), 1, nx)
            jc = clamp(round(Int, fj), 1, ny)
            kc = clamp(round(Int, fk), 1, nz)

            best_d2 = Inf32
            best_c = CartesianIndex(0, 0, 0)

            for rad in 0:fallback_search_radius_vox
                @inbounds for k in max(1, kc - rad):min(nz, kc + rad),
                              j in max(1, jc - rad):min(ny, jc + rad),
                              i in max(1, ic - rad):min(nx, ic + rad)
                    grid[i, j, k] || continue
                    vx = ox + (Float32(i) - 0.5f0) * dx
                    vy = oy + (Float32(j) - 0.5f0) * dy
                    vz = oz + (Float32(k) - 0.5f0) * dz
                    d2 = (vx - gx)^2 + (vy - gy)^2 + (vz - gz)^2
                    if d2 < best_d2
                        best_d2 = d2
                        best_c = CartesianIndex(i, j, k)
                    end
                end
                isfinite(best_d2) && break
            end

            if isfinite(best_d2) && !(best_c in seen)
                push!(seen, best_c)
                push!(gate_voxels, best_c)
            end
        end
    end

    return gate_voxels
end

function KernelAbstractions.get_backend(A::FiberDist.HalfResThickness)
    return KernelAbstractions.get_backend(A.data)
end